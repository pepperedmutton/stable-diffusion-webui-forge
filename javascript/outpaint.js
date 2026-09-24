(function() {
    "use strict";

    // ForgeCanvas keeps the original input. This editor changes geometry only.
    const fieldNames = [
        "region_mode",
        "left",
        "right",
        "top",
        "bottom",
        "canvas_width",
        "canvas_height",
        "source_x",
        "source_y",
        "max_megapixels",
    ];
    const geometryFieldNames = [
        "left",
        "right",
        "top",
        "bottom",
        "canvas_width",
        "canvas_height",
        "source_x",
        "source_y",
    ];
    const prefixes = ["img2img_outpaint_", "script_forge_outpaint_"];
    const controlSelector = prefixes
        .flatMap(prefix => fieldNames.map(name => `#${prefix}${name}`))
        .join(",");
    const edgeHandles = [
        "left",
        "right",
        "top",
        "bottom",
        "top-left",
        "top-right",
        "bottom-left",
        "bottom-right",
    ];

    let eventRoot = null;
    let observedImageRoot = null;
    let imageObserver = null;
    let observedPreview = null;
    let previewResizeObserver = null;
    let lastPreviewWidth = 0;
    let lastState = null;
    let frameRequest = 0;
    let syncDepth = 0;
    let dragState = null;

    function appRoot() {
        return typeof gradioApp === "function" ? gradioApp() : document;
    }

    function isVisible(element) {
        if (!element) {
            return false;
        }

        const bounds = element.getBoundingClientRect();
        const style = getComputedStyle(element);
        return bounds.width > 0 && bounds.height > 0 && style.display !== "none" && style.visibility !== "hidden";
    }

    function activeContext() {
        const root = appRoot();
        const nativePreview = root.querySelector("#img2img_outpaint_preview");
        const legacyPreview = root.querySelector("#forge_outpaint_preview");

        if (isVisible(nativePreview) || !legacyPreview) {
            return {
                controlPrefix: "img2img_outpaint_",
                imageRoot: root.querySelector("#img2img_outpaint"),
                preview: nativePreview,
                interactive: true,
            };
        }

        return {
            controlPrefix: "script_forge_outpaint_",
            imageRoot: root.querySelector("#img2img_image"),
            preview: legacyPreview,
            interactive: false,
        };
    }

    function componentRoot(name, context) {
        return appRoot().querySelector(`#${context.controlPrefix}${name}`);
    }

    function componentInput(name, context) {
        const root = componentRoot(name, context);
        if (!root) {
            return null;
        }

        return root.querySelector('input[type="number"], input[type="range"], input:not([type])');
    }

    function rawValue(name, context) {
        const root = componentRoot(name, context);
        if (!root) {
            return "";
        }

        const checked = root.querySelector('input[type="radio"]:checked');
        if (checked) {
            return checked.value;
        }

        const select = root.querySelector("select");
        if (select) {
            return select.value;
        }

        const input = componentInput(name, context);
        return input ? input.value : "";
    }

    function numericValue(name, context) {
        const raw = rawValue(name, context).trim();
        if (!raw) {
            return NaN;
        }

        const value = Number(raw);
        return Number.isFinite(value) ? value : NaN;
    }

    function emitInput(input) {
        if (typeof updateInput === "function") {
            updateInput(input);
        } else {
            input.dispatchEvent(new Event("input", {bubbles: true}));
        }
    }

    function writeNumber(name, value, context, sendChange) {
        const input = componentInput(name, context);
        if (!input) {
            return;
        }

        const next = String(value);
        if (input.value !== next) {
            input.value = next;
            emitInput(input);
        }
        if (sendChange) {
            input.dispatchEvent(new Event("change", {bubbles: true}));
        }
    }

    function inputImage(context) {
        const root = context.imageRoot;
        if (!root) {
            return null;
        }

        const images = Array.from(root.querySelectorAll(".forge-image"));
        return images.find(image => image.tagName === "IMG" && image.complete && image.getAttribute("src") && image.naturalWidth > 0 && image.naturalHeight > 0) || null;
    }

    function makeArea(imageWidth, imageHeight, left, right, top, bottom) {
        return {
            imageWidth,
            imageHeight,
            left,
            right,
            top,
            bottom,
            canvasWidth: imageWidth + left + right,
            canvasHeight: imageHeight + top + bottom,
            sourceX: left,
            sourceY: top,
        };
    }

    function isSafeNonnegativeInteger(value) {
        return Number.isSafeInteger(value) && value >= 0;
    }

    function geometry(image, context) {
        const imageWidth = image.naturalWidth;
        const imageHeight = image.naturalHeight;
        const mode = rawValue("region_mode", context).trim().toLowerCase();
        const canvasMode = mode.includes("canvas") || mode.includes("position");
        let left;
        let right;
        let top;
        let bottom;

        if (canvasMode) {
            const canvasWidth = numericValue("canvas_width", context);
            const canvasHeight = numericValue("canvas_height", context);
            const sourceX = numericValue("source_x", context);
            const sourceY = numericValue("source_y", context);
            if (![canvasWidth, canvasHeight, sourceX, sourceY].every(Number.isSafeInteger)) {
                return null;
            }

            left = sourceX;
            top = sourceY;
            right = canvasWidth - sourceX - imageWidth;
            bottom = canvasHeight - sourceY - imageHeight;
        } else {
            left = numericValue("left", context);
            right = numericValue("right", context);
            top = numericValue("top", context);
            bottom = numericValue("bottom", context);
        }

        if (![left, right, top, bottom].every(isSafeNonnegativeInteger)) {
            return null;
        }
        if (!left && !right && !top && !bottom) {
            return null;
        }

        const area = makeArea(imageWidth, imageHeight, left, right, top, bottom);
        if (area.canvasWidth > 32768 || area.canvasHeight > 32768) {
            return null;
        }
        return area;
    }

    function synchronizeArea(area, context, sendChange) {
        const values = {
            left: area.left,
            right: area.right,
            top: area.top,
            bottom: area.bottom,
            canvas_width: area.canvasWidth,
            canvas_height: area.canvasHeight,
            source_x: area.sourceX,
            source_y: area.sourceY,
        };

        syncDepth += 1;
        try {
            geometryFieldNames.forEach(name => writeNumber(name, values[name], context, sendChange));
        } finally {
            syncDepth -= 1;
        }
    }

    function previewParts(preview) {
        let canvas = preview.querySelector(".forge-outpaint-preview-canvas");
        let message = preview.querySelector(".forge-outpaint-preview-message");
        let overlay = preview.querySelector(".forge-outpaint-editor-overlay");

        if (!canvas) {
            canvas = document.createElement("canvas");
            canvas.className = "forge-outpaint-preview-canvas";
            canvas.setAttribute("aria-hidden", "true");
            preview.appendChild(canvas);
        }

        if (!message) {
            message = document.createElement("div");
            message.className = "forge-outpaint-preview-message";
            message.setAttribute("role", "status");
            preview.appendChild(message);
        }

        if (!overlay) {
            overlay = document.createElement("div");
            overlay.className = "forge-outpaint-editor-overlay";
            overlay.setAttribute("role", "group");
            overlay.setAttribute("aria-label", "Outpaint output area editor");
            overlay.setAttribute("aria-describedby", "img2img_outpaint_editor_help");

            edgeHandles.forEach(handle => {
                const control = document.createElement("div");
                control.className = `forge-outpaint-handle forge-outpaint-handle--${handle}`;
                control.dataset.outpaintHandle = handle;
                control.tabIndex = 0;
                control.setAttribute("role", handle.includes("-") ? "button" : "slider");
                if (!handle.includes("-")) {
                    control.setAttribute("aria-valuemin", "0");
                }
                control.setAttribute("aria-label", `${handle.replace("-", " ")} output edge`);
                overlay.appendChild(control);
            });

            const sourceMover = document.createElement("div");
            sourceMover.className = "forge-outpaint-source-mover";
            sourceMover.dataset.outpaintHandle = "move";
            sourceMover.tabIndex = 0;
            sourceMover.setAttribute("role", "button");
            sourceMover.setAttribute("aria-label", "Move input image inside output area");
            overlay.appendChild(sourceMover);

            const status = document.createElement("div");
            status.className = "forge-outpaint-editor-status";
            status.setAttribute("role", "status");
            status.setAttribute("aria-live", "polite");
            overlay.appendChild(status);

            overlay.addEventListener("pointerdown", pointerDown);
            overlay.addEventListener("pointermove", pointerMove);
            overlay.addEventListener("pointerup", pointerUp);
            overlay.addEventListener("pointercancel", pointerCancel);
            overlay.addEventListener("lostpointercapture", pointerCaptureLost);
            overlay.addEventListener("keydown", editorKeyDown);
            preview.appendChild(overlay);
        }

        return {canvas, message, overlay};
    }

    function showMessage(preview, text) {
        const {canvas, message, overlay} = previewParts(preview);
        canvas.hidden = true;
        overlay.hidden = true;
        message.hidden = false;
        message.textContent = text;
        preview.__forgeOutpaintEditorState = null;
    }

    function previewColor(preview, property, fallback) {
        return getComputedStyle(preview).getPropertyValue(property).trim() || fallback;
    }

    function fitView(area, logicalWidth, logicalHeight) {
        const horizontalPadding = 46;
        const topPadding = 50;
        const bottomPadding = 64;
        const availableWidth = Math.max(1, logicalWidth - horizontalPadding * 2);
        const availableHeight = Math.max(1, logicalHeight - topPadding - bottomPadding);
        const scale = Math.max(0.0001, Math.min(availableWidth / area.canvasWidth, availableHeight / area.canvasHeight));
        const frameWidth = area.canvasWidth * scale;
        const frameHeight = area.canvasHeight * scale;
        const frameX = (logicalWidth - frameWidth) / 2;
        const frameY = topPadding + (availableHeight - frameHeight) / 2;

        return {
            logicalWidth,
            logicalHeight,
            scale,
            originX: frameX + area.left * scale,
            originY: frameY + area.top * scale,
        };
    }

    function displayMetrics(area, view) {
        return {
            frameX: view.originX - area.left * view.scale,
            frameY: view.originY - area.top * view.scale,
            frameWidth: area.canvasWidth * view.scale,
            frameHeight: area.canvasHeight * view.scale,
            sourceX: view.originX,
            sourceY: view.originY,
            sourceWidth: area.imageWidth * view.scale,
            sourceHeight: area.imageHeight * view.scale,
        };
    }

    function setBox(element, left, top, width, height) {
        element.style.left = `${left}px`;
        element.style.top = `${top}px`;
        element.style.width = `${Math.max(0, width)}px`;
        element.style.height = `${Math.max(0, height)}px`;
    }

    function positionEditor(preview, area, metrics, interactive) {
        const {overlay} = previewParts(preview);
        overlay.hidden = !interactive;
        if (!interactive) {
            return;
        }

        const hit = 28;
        const corner = 38;
        const halfHit = hit / 2;
        const halfCorner = corner / 2;
        const horizontalLength = Math.max(0, metrics.frameWidth - corner);
        const verticalLength = Math.max(0, metrics.frameHeight - corner);
        const byHandle = handle => overlay.querySelector(`[data-outpaint-handle="${handle}"]`);

        setBox(byHandle("left"), metrics.frameX - halfHit, metrics.frameY + halfCorner, hit, verticalLength);
        setBox(byHandle("right"), metrics.frameX + metrics.frameWidth - halfHit, metrics.frameY + halfCorner, hit, verticalLength);
        setBox(byHandle("top"), metrics.frameX + halfCorner, metrics.frameY - halfHit, horizontalLength, hit);
        setBox(byHandle("bottom"), metrics.frameX + halfCorner, metrics.frameY + metrics.frameHeight - halfHit, horizontalLength, hit);
        setBox(byHandle("top-left"), metrics.frameX - halfCorner, metrics.frameY - halfCorner, corner, corner);
        setBox(byHandle("top-right"), metrics.frameX + metrics.frameWidth - halfCorner, metrics.frameY - halfCorner, corner, corner);
        setBox(byHandle("bottom-left"), metrics.frameX - halfCorner, metrics.frameY + metrics.frameHeight - halfCorner, corner, corner);
        setBox(byHandle("bottom-right"), metrics.frameX + metrics.frameWidth - halfCorner, metrics.frameY + metrics.frameHeight - halfCorner, corner, corner);

        const mover = byHandle("move");
        const moverInset = Math.min(8, metrics.sourceWidth / 4, metrics.sourceHeight / 4);
        setBox(
            mover,
            metrics.sourceX + moverInset,
            metrics.sourceY + moverInset,
            metrics.sourceWidth - moverInset * 2,
            metrics.sourceHeight - moverInset * 2
        );
        mover.hidden = metrics.sourceWidth < 24 || metrics.sourceHeight < 24;
        mover.setAttribute("aria-label", `Move input image. Current position ${area.sourceX}, ${area.sourceY}`);

        const edgeValues = {
            left: area.left,
            right: area.right,
            top: area.top,
            bottom: area.bottom,
            "top-left": `${area.left}, ${area.top}`,
            "top-right": `${area.right}, ${area.top}`,
            "bottom-left": `${area.left}, ${area.bottom}`,
            "bottom-right": `${area.right}, ${area.bottom}`,
        };
        edgeHandles.forEach(handle => {
            const control = byHandle(handle);
            const value = edgeValues[handle];
            if (typeof value === "number") {
                control.setAttribute("aria-valuenow", String(value));
                control.removeAttribute("aria-valuetext");
            } else {
                control.removeAttribute("aria-valuenow");
                control.setAttribute("aria-valuetext", value);
            }
        });

        const status = overlay.querySelector(".forge-outpaint-editor-status");
        status.textContent = `Output ${area.canvasWidth} by ${area.canvasHeight}. Input position ${area.sourceX}, ${area.sourceY}.`;
    }

    function alignedWorkPixels(area) {
        const width = Math.ceil(area.canvasWidth / 64) * 64;
        const height = Math.ceil(area.canvasHeight / 64) * 64;
        return width * height;
    }

    function maximumPixels(context) {
        const megapixels = numericValue("max_megapixels", context);
        return Number.isFinite(megapixels) && megapixels > 0 ? Math.floor(megapixels * 1000000) : Infinity;
    }

    function drawPreview(preview, image, area, contextInfo, frozenView) {
        const {canvas, message} = previewParts(preview);
        const logicalWidth = Math.max(280, Math.round(preview.clientWidth || 720));
        const logicalHeight = Math.max(320, Math.min(640, Math.round(logicalWidth * 0.62)));
        const pixelRatio = Math.max(1, Math.min(4, window.devicePixelRatio || 1));
        const view = frozenView || fitView(area, logicalWidth, logicalHeight);
        const metrics = displayMetrics(area, view);

        canvas.hidden = false;
        message.hidden = true;
        canvas.style.height = `${view.logicalHeight}px`;
        canvas.width = Math.round(view.logicalWidth * pixelRatio);
        canvas.height = Math.round(view.logicalHeight * pixelRatio);

        const drawing = canvas.getContext("2d");
        drawing.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
        drawing.clearRect(0, 0, view.logicalWidth, view.logicalHeight);

        const background = previewColor(preview, "--forge-outpaint-preview-background", "#ffffff");
        const outerFill = previewColor(preview, "--forge-outpaint-preview-new-area", "#e5e7eb");
        const imageFill = previewColor(preview, "--forge-outpaint-preview-image-area", "#ffffff");
        const border = previewColor(preview, "--forge-outpaint-preview-border", "#9ca3af");
        const accent = previewColor(preview, "--forge-outpaint-preview-accent", "#3b82f6");
        const text = previewColor(preview, "--forge-outpaint-preview-text", "#111827");
        const subdued = previewColor(preview, "--forge-outpaint-preview-subdued", "#4b5563");

        drawing.fillStyle = background;
        drawing.fillRect(0, 0, view.logicalWidth, view.logicalHeight);

        drawing.fillStyle = outerFill;
        drawing.fillRect(metrics.frameX, metrics.frameY, metrics.frameWidth, metrics.frameHeight);

        drawing.fillStyle = imageFill;
        drawing.fillRect(metrics.sourceX, metrics.sourceY, metrics.sourceWidth, metrics.sourceHeight);
        drawing.save();
        drawing.beginPath();
        drawing.rect(metrics.sourceX, metrics.sourceY, metrics.sourceWidth, metrics.sourceHeight);
        drawing.clip();
        drawing.drawImage(image, metrics.sourceX, metrics.sourceY, metrics.sourceWidth, metrics.sourceHeight);
        drawing.restore();

        drawing.save();
        drawing.strokeStyle = border;
        drawing.lineWidth = 1.5;
        drawing.setLineDash([6, 5]);
        drawing.strokeRect(metrics.sourceX, metrics.sourceY, metrics.sourceWidth, metrics.sourceHeight);
        drawing.restore();

        drawing.strokeStyle = accent;
        drawing.lineWidth = 3;
        drawing.strokeRect(metrics.frameX, metrics.frameY, metrics.frameWidth, metrics.frameHeight);

        const limitExceeded = alignedWorkPixels(area) > maximumPixels(contextInfo);
        drawing.fillStyle = limitExceeded ? "#dc2626" : text;
        drawing.font = "600 14px system-ui, sans-serif";
        drawing.textAlign = "left";
        drawing.textBaseline = "middle";
        drawing.fillText(`Output: ${area.canvasWidth} × ${area.canvasHeight}`, 18, 24);

        if (limitExceeded) {
            drawing.textAlign = "right";
            drawing.fillText("Maximum megapixels exceeded", view.logicalWidth - 18, 24);
        }

        const imageLabel = `Input: ${area.imageWidth} × ${area.imageHeight}`;
        const positionLabel = `Position: (${area.sourceX}, ${area.sourceY})`;
        drawing.fillStyle = subdued;
        drawing.font = "12px system-ui, sans-serif";
        drawing.textAlign = "left";

        if (drawing.measureText(imageLabel).width + drawing.measureText(positionLabel).width + 54 > view.logicalWidth) {
            drawing.fillText(imageLabel, 18, view.logicalHeight - 34);
            drawing.fillText(positionLabel, 18, view.logicalHeight - 16);
        } else {
            drawing.fillText(imageLabel, 18, view.logicalHeight - 22);
            drawing.textAlign = "right";
            drawing.fillText(positionLabel, view.logicalWidth - 18, view.logicalHeight - 22);
        }

        preview.__forgeOutpaintEditorState = {
            area,
            context: contextInfo,
            image,
            metrics,
            view,
        };
        positionEditor(preview, area, metrics, contextInfo.interactive);
    }

    function copyMargins(area) {
        return {
            left: area.left,
            right: area.right,
            top: area.top,
            bottom: area.bottom,
        };
    }

    function candidateForDelta(start, handle, dx, dy) {
        const margins = copyMargins(start);
        if (handle === "move") {
            const moveX = Math.max(-start.left, Math.min(start.right, dx));
            const moveY = Math.max(-start.top, Math.min(start.bottom, dy));
            margins.left = start.left + moveX;
            margins.right = start.right - moveX;
            margins.top = start.top + moveY;
            margins.bottom = start.bottom - moveY;
        } else {
            if (handle.includes("left")) {
                margins.left = start.left - dx;
            }
            if (handle.includes("right")) {
                margins.right = start.right + dx;
            }
            if (handle.includes("top")) {
                margins.top = start.top - dy;
            }
            if (handle.includes("bottom")) {
                margins.bottom = start.bottom + dy;
            }
        }

        Object.keys(margins).forEach(name => {
            margins[name] = Math.max(0, Math.min(16384, Math.round(margins[name])));
        });
        return makeArea(start.imageWidth, start.imageHeight, margins.left, margins.right, margins.top, margins.bottom);
    }

    function keepCanvasLimits(area, handle) {
        const margins = copyMargins(area);
        const horizontalLimit = Math.max(0, 32768 - area.imageWidth);
        const verticalLimit = Math.max(0, 32768 - area.imageHeight);

        if (margins.left + margins.right > horizontalLimit) {
            if (handle.includes("left")) {
                margins.left = Math.max(0, horizontalLimit - margins.right);
            } else {
                margins.right = Math.max(0, horizontalLimit - margins.left);
            }
        }
        if (margins.top + margins.bottom > verticalLimit) {
            if (handle.includes("top")) {
                margins.top = Math.max(0, verticalLimit - margins.bottom);
            } else {
                margins.bottom = Math.max(0, verticalLimit - margins.top);
            }
        }

        if (!margins.left && !margins.right && !margins.top && !margins.bottom) {
            if (handle.includes("left")) {
                margins.left = 1;
            } else if (handle.includes("top")) {
                margins.top = 1;
            } else if (handle.includes("bottom")) {
                margins.bottom = 1;
            } else {
                margins.right = 1;
            }
        }
        return makeArea(area.imageWidth, area.imageHeight, margins.left, margins.right, margins.top, margins.bottom);
    }

    function interpolateArea(start, target, amount) {
        const interpolate = name => {
            const delta = target[name] - start[name];
            const value = start[name] + delta * amount;
            return delta >= 0 ? Math.floor(value) : Math.ceil(value);
        };
        return makeArea(
            start.imageWidth,
            start.imageHeight,
            interpolate("left"),
            interpolate("right"),
            interpolate("top"),
            interpolate("bottom")
        );
    }

    function keepMegapixelLimit(start, target, context) {
        const limit = maximumPixels(context);
        if (alignedWorkPixels(target) <= limit || !Number.isFinite(limit)) {
            return target;
        }
        if (alignedWorkPixels(start) > limit) {
            return start;
        }

        let low = 0;
        let high = 1;
        let best = start;
        for (let index = 0; index < 24; index += 1) {
            const middle = (low + high) / 2;
            const candidate = interpolateArea(start, target, middle);
            if (alignedWorkPixels(candidate) <= limit) {
                best = candidate;
                low = middle;
            } else {
                high = middle;
            }
        }
        return best;
    }

    function normalizedCandidate(start, target, handle, context) {
        const withinCanvas = keepCanvasLimits(target, handle);
        return keepMegapixelLimit(start, withinCanvas, context);
    }

    function editorState(event) {
        const preview = event.currentTarget.closest(".forge-outpaint-preview");
        return preview ? preview.__forgeOutpaintEditorState : null;
    }

    function pointerDown(event) {
        const target = event.target.closest("[data-outpaint-handle]");
        const state = editorState(event);
        if (!target || !state || !state.context.interactive || event.button !== 0 || dragState) {
            return;
        }

        dragState = {
            pointerId: event.pointerId,
            target,
            preview: event.currentTarget.closest(".forge-outpaint-preview"),
            context: state.context,
            image: state.image,
            handle: target.dataset.outpaintHandle,
            startX: event.clientX,
            startY: event.clientY,
            startArea: state.area,
            currentArea: state.area,
            view: {...state.view},
        };
        target.setPointerCapture(event.pointerId);
        event.currentTarget.classList.add("forge-outpaint-editor-overlay--dragging");
        event.preventDefault();
    }

    function updateDrag(event) {
        if (!dragState || event.pointerId !== dragState.pointerId) {
            return;
        }

        const dx = Math.round((event.clientX - dragState.startX) / dragState.view.scale);
        const dy = Math.round((event.clientY - dragState.startY) / dragState.view.scale);
        const rawCandidate = candidateForDelta(dragState.startArea, dragState.handle, dx, dy);
        const candidate = normalizedCandidate(dragState.startArea, rawCandidate, dragState.handle, dragState.context);
        dragState.currentArea = candidate;
        synchronizeArea(candidate, dragState.context, false);
        drawPreview(dragState.preview, dragState.image, candidate, dragState.context, dragState.view);
    }

    function pointerMove(event) {
        if (!dragState || event.pointerId !== dragState.pointerId) {
            return;
        }
        updateDrag(event);
        event.preventDefault();
    }

    function finishDrag(commit) {
        if (!dragState) {
            return;
        }

        const active = dragState;
        const finalArea = commit ? active.currentArea : active.startArea;
        dragState = null;
        if (active.target.hasPointerCapture(active.pointerId)) {
            active.target.releasePointerCapture(active.pointerId);
        }
        const overlay = active.preview.querySelector(".forge-outpaint-editor-overlay");
        overlay?.classList.remove("forge-outpaint-editor-overlay--dragging");
        synchronizeArea(finalArea, active.context, true);
        drawPreview(active.preview, active.image, finalArea, active.context, null);
        scheduleRender();
    }

    function pointerUp(event) {
        if (dragState && event.pointerId === dragState.pointerId) {
            updateDrag(event);
            finishDrag(true);
            event.preventDefault();
        }
    }

    function pointerCancel(event) {
        if (dragState && event.pointerId === dragState.pointerId) {
            finishDrag(false);
            event.preventDefault();
        }
    }

    function pointerCaptureLost(event) {
        if (dragState && event.pointerId === dragState.pointerId) {
            finishDrag(true);
        }
    }

    function editorKeyDown(event) {
        if (event.key === "Escape" && dragState) {
            finishDrag(false);
            event.preventDefault();
            return;
        }

        const target = event.target.closest("[data-outpaint-handle]");
        const state = editorState(event);
        if (!target || !state || !state.context.interactive) {
            return;
        }

        const step = event.shiftKey ? 16 : 1;
        let dx = 0;
        let dy = 0;
        if (event.key === "ArrowLeft") {
            dx = -step;
        } else if (event.key === "ArrowRight") {
            dx = step;
        } else if (event.key === "ArrowUp") {
            dy = -step;
        } else if (event.key === "ArrowDown") {
            dy = step;
        } else {
            return;
        }

        const handle = target.dataset.outpaintHandle;
        const rawCandidate = candidateForDelta(state.area, handle, dx, dy);
        const candidate = normalizedCandidate(state.area, rawCandidate, handle, state.context);
        synchronizeArea(candidate, state.context, true);
        drawPreview(event.currentTarget.closest(".forge-outpaint-preview"), state.image, candidate, state.context, null);
        scheduleRender();
        event.preventDefault();
    }

    function captureState() {
        const context = activeContext();
        const image = inputImage(context);

        return {
            context,
            imageRoot: context.imageRoot,
            preview: context.preview,
            image,
            imageSrc: image ? image.getAttribute("src") : "",
            imageWidth: image ? image.naturalWidth : 0,
            imageHeight: image ? image.naturalHeight : 0,
            controls: fieldNames.map(name => rawValue(name, context)).join("\u001f"),
            theme: `${document.documentElement.className}\u001f${document.body.className}`,
        };
    }

    function statesMatch(first, second) {
        return Boolean(first && second &&
            first.imageRoot === second.imageRoot &&
            first.preview === second.preview &&
            first.image === second.image &&
            first.imageSrc === second.imageSrc &&
            first.imageWidth === second.imageWidth &&
            first.imageHeight === second.imageHeight &&
            first.controls === second.controls &&
            first.theme === second.theme);
    }

    function render() {
        frameRequest = 0;
        const state = captureState();
        lastState = state;

        if (!state.preview) {
            return;
        }
        if (!state.image) {
            showMessage(state.preview, "Select an input image.");
            return;
        }

        const area = geometry(state.image, state.context);
        if (!area) {
            showMessage(state.preview, "Use output dimensions that contain the complete input image.");
            return;
        }

        synchronizeArea(area, state.context, false);
        const frozenView = dragState && dragState.preview === state.preview ? dragState.view : null;
        drawPreview(state.preview, state.image, area, state.context, frozenView);
    }

    function scheduleRender() {
        if (!frameRequest) {
            frameRequest = window.requestAnimationFrame(render);
        }
    }

    function fieldChanged(event) {
        if (syncDepth) {
            return;
        }
        const target = event.target;
        if (target instanceof Element && target.closest(controlSelector)) {
            scheduleRender();
        }
    }

    function imageLoaded(event) {
        const target = event.target;
        if (target instanceof Element && target.matches("#img2img_outpaint .forge-image, #img2img_image .forge-image")) {
            scheduleRender();
        }
    }

    function connectEventRoot(root) {
        if (eventRoot === root) {
            return;
        }

        if (eventRoot) {
            eventRoot.removeEventListener("input", fieldChanged, true);
            eventRoot.removeEventListener("change", fieldChanged, true);
            eventRoot.removeEventListener("load", imageLoaded, true);
        }

        eventRoot = root;
        eventRoot.addEventListener("input", fieldChanged, true);
        eventRoot.addEventListener("change", fieldChanged, true);
        eventRoot.addEventListener("load", imageLoaded, true);
    }

    function connectImageObserver(root) {
        if (observedImageRoot === root) {
            return;
        }

        imageObserver?.disconnect();
        observedImageRoot = root;
        if (!root) {
            return;
        }

        imageObserver = new MutationObserver(scheduleRender);
        imageObserver.observe(root, {
            attributes: true,
            attributeFilter: ["src"],
            childList: true,
            subtree: true,
        });
    }

    function connectPreviewObserver(preview) {
        if (observedPreview === preview) {
            return;
        }

        previewResizeObserver?.disconnect();
        observedPreview = preview;
        lastPreviewWidth = 0;
        if (!preview || typeof ResizeObserver !== "function") {
            return;
        }

        previewResizeObserver = new ResizeObserver(entries => {
            const width = Math.round(entries[0].contentRect.width);
            if (width && width !== lastPreviewWidth) {
                lastPreviewWidth = width;
                scheduleRender();
            }
        });
        previewResizeObserver.observe(preview);
    }

    function connect() {
        const root = appRoot();
        connectEventRoot(root);

        const nextState = captureState();
        connectImageObserver(nextState.imageRoot);
        connectPreviewObserver(nextState.preview);

        if (!statesMatch(lastState, nextState)) {
            scheduleRender();
        }
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(connect);
    } else {
        document.addEventListener("DOMContentLoaded", connect, {once: true});
    }

    if (typeof onAfterUiUpdate === "function") {
        onAfterUiUpdate(connect);
    }

    window.addEventListener("resize", scheduleRender, {passive: true});
})();
