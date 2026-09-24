#!/usr/bin/env python3
"""Add an explicit, transactional touch editor to the audited Forge outpaint UI.

Only the copy passed to --root is changed. --check is read-only. Desktop pointer
behavior is retained. Mobile drafts never write generation parameters before an
explicit Apply. Unknown or partly modified source is rejected before any write.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile


AUDITED_SHA256 = "76ce05a7f4e26c2035e7c8e613b25062a14570f4ac44c07ed9a755ed7f75f24d"
CSS_START = "/* BEGIN FORGE OUTPAINT TOUCH SAFETY */"
CSS_END = "/* END FORGE OUTPAINT TOUCH SAFETY */"

HELPERS = r'''
    // BEGIN FORGE OUTPAINT TOUCH SAFETY
    let touchSession = null;
    const touchPreviews = new Set();

    function touchMode() {
        return window.matchMedia("(pointer: coarse)").matches;
    }

    function touchControlsKey(context) {
        return fieldNames.map(name => rawValue(name, context)).join("\u001f");
    }

    function touchText(element, text) {
        if (element.textContent !== text) {
            element.textContent = text;
        }
    }

    function touchHidden(element, hidden) {
        if (element.hidden !== hidden) {
            element.hidden = hidden;
        }
    }

    function touchTools(preview) {
        let tools = preview.__forgeOutpaintTouchTools;
        if (!tools || !tools.isConnected) {
            tools = document.createElement("div");
            tools.className = "forge-outpaint-mobile-safety";
            tools.setAttribute("role", "group");
            tools.setAttribute("aria-label", "Mobile outpaint touch protection");
            const status = document.createElement("span");
            status.className = "forge-outpaint-touch-status";
            status.setAttribute("role", "status");
            status.setAttribute("aria-live", "polite");
            tools.appendChild(status);
            [["edit", "Edit outpaint"], ["apply", "Apply and lock"], ["cancel", "Discard and lock"]].forEach(([action, label]) => {
                const button = document.createElement("button");
                button.type = "button";
                button.dataset.forgeOutpaintAction = action;
                button.textContent = label;
                button.addEventListener("click", () => {
                    if (!touchMode()) {
                        return;
                    }
                    if (action === "edit") {
                        const state = preview.__forgeOutpaintEditorState;
                        if (!state || !isVisible(preview)) {
                            return;
                        }
                        lockTouchEditor();
                        touchSession = {
                            preview,
                            context: state.context,
                            image: state.image,
                            imageSrc: state.image.getAttribute("src"),
                            controls: touchControlsKey(state.context),
                            area: state.area,
                        };
                        preview.__forgeOutpaintTouchNotice = "";
                        updateTouchTools(preview);
                    } else if (action === "apply" && touchSession?.preview === preview) {
                        if (dragState) {
                            lockTouchEditor("Gesture incomplete. Draft discarded and locked.");
                            return;
                        }
                        const session = touchSession;
                        touchSession = null;
                        synchronizeArea(session.area, session.context, true);
                        preview.__forgeOutpaintTouchNotice = "Outpaint changes applied and locked. You can scroll the page.";
                        updateTouchTools(preview);
                        scheduleRender();
                    } else if (action === "cancel") {
                        lockTouchEditor("Draft discarded and locked. You can scroll the page.");
                    }
                });
                tools.appendChild(button);
            });
            preview.parentNode.insertBefore(tools, preview);
            preview.__forgeOutpaintTouchTools = tools;
            touchPreviews.add(preview);
        }
        return tools;
    }

    function updateTouchTools(preview) {
        if (!touchMode() && !preview.__forgeOutpaintTouchTools) {
            return;
        }
        const tools = touchTools(preview);
        const editing = touchMode() && touchSession?.preview === preview;
        touchHidden(tools, !touchMode() || !isVisible(preview) || !preview.__forgeOutpaintEditorState);
        const mode = touchMode() ? (editing ? "editing" : "locked") : "desktop";
        if (preview.dataset.forgeOutpaintTouch !== mode) {
            preview.dataset.forgeOutpaintTouch = mode;
        }
        touchHidden(tools.querySelector('[data-forge-outpaint-action="edit"]'), editing);
        touchHidden(tools.querySelector('[data-forge-outpaint-action="apply"]'), !editing);
        touchHidden(tools.querySelector('[data-forge-outpaint-action="cancel"]'), !editing);
        touchText(tools.querySelector(".forge-outpaint-touch-status"), editing
            ? `Unapplied draft: ${touchSession.area.canvasWidth} × ${touchSession.area.canvasHeight}. Tap Apply and lock to save.`
            : (preview.__forgeOutpaintTouchNotice || "Outpaint locked. Scroll the page, or edit exact values in the parameter section."));
        preview.querySelectorAll("[data-outpaint-handle]").forEach(handle => {
            const disabled = touchMode() && !editing;
            if (handle.tabIndex !== (disabled ? -1 : 0)) {
                handle.tabIndex = disabled ? -1 : 0;
            }
            if (handle.getAttribute("aria-disabled") !== String(disabled)) {
                handle.setAttribute("aria-disabled", String(disabled));
            }
        });
    }

    function lockTouchEditor(notice = "Unapplied outpaint changes discarded and locked.") {
        if (!touchSession) {
            return;
        }
        const session = touchSession;
        if (dragState && dragState.preview === session.preview) {
            finishDrag(false);
        }
        touchSession = null;
        session.preview.__forgeOutpaintTouchNotice = notice;
        updateTouchTools(session.preview);
        scheduleRender();
    }

    function touchOutsidePointer(event) {
        if (!touchSession) {
            return;
        }
        if (event.pointerType === "touch" && event.isPrimary === false) {
            lockTouchEditor("Multi-touch canceled. Unapplied changes discarded and locked.");
            return;
        }
        const target = event.target;
        if (!(target instanceof Element) ||
            (!touchSession.preview.contains(target) && !touchTools(touchSession.preview).contains(target))) {
            lockTouchEditor();
        }
    }

    function validateTouchSession(state) {
        if (touchSession && (!touchMode() || document.hidden ||
            !touchSession.preview.isConnected || !isVisible(touchSession.preview) ||
            state.preview !== touchSession.preview || state.image !== touchSession.image ||
            state.imageSrc !== touchSession.imageSrc || state.controls !== touchSession.controls)) {
            lockTouchEditor();
        }
        touchPreviews.forEach(preview => {
            if (!preview.isConnected) {
                preview.__forgeOutpaintTouchTools?.remove();
                touchPreviews.delete(preview);
            } else {
                updateTouchTools(preview);
            }
        });
    }

    window.addEventListener("blur", () => lockTouchEditor());
    window.addEventListener("pagehide", () => lockTouchEditor());
    window.addEventListener("resize", () => lockTouchEditor(), {passive: true});
    document.addEventListener("visibilitychange", () => {
        if (document.hidden) {
            lockTouchEditor();
        }
    });
    document.addEventListener("scroll", () => lockTouchEditor(), {capture: true, passive: true});
    document.addEventListener("keydown", event => {
        if (event.key === "Escape" && touchSession) {
            lockTouchEditor("Draft discarded and locked.");
            event.preventDefault();
        }
    }, true);
    // END FORGE OUTPAINT TOUCH SAFETY
'''

EDITS = [
    ("    function appRoot() {", HELPERS + "\n    function appRoot() {"),
    (
        "        preview.__forgeOutpaintEditorState = null;\n    }",
        "        preview.__forgeOutpaintEditorState = null;\n        updateTouchTools(preview);\n    }",
    ),
    (
        "        positionEditor(preview, area, metrics, contextInfo.interactive);",
        "        positionEditor(preview, area, metrics, contextInfo.interactive);\n        if (contextInfo.interactive) {\n            updateTouchTools(preview);\n        }",
    ),
    (
        '    function pointerDown(event) {\n        const target',
        '    function pointerDown(event) {\n        if (touchMode() && (!touchSession || event.currentTarget.closest(".forge-outpaint-preview") !== touchSession.preview)) {\n            return;\n        }\n        const target',
    ),
    (
        "            currentArea: state.area,\n            view:",
        "            currentArea: state.area,\n            touchDraft: Boolean(touchMode() && touchSession),\n            moved: false,\n            view:",
    ),
    (
        "        const dx = Math.round((event.clientX - dragState.startX) / dragState.view.scale);",
        "        if (dragState.touchDraft && !dragState.moved) {\n            if (Math.hypot(event.clientX - dragState.startX, event.clientY - dragState.startY) < 8) {\n                return;\n            }\n            dragState.moved = true;\n        }\n        const dx = Math.round((event.clientX - dragState.startX) / dragState.view.scale);",
    ),
    (
        "        dragState.currentArea = candidate;\n        synchronizeArea(candidate, dragState.context, false);",
        "        dragState.currentArea = candidate;\n        if (dragState.touchDraft && touchSession) {\n            touchSession.area = candidate;\n        } else {\n            synchronizeArea(candidate, dragState.context, false);\n        }",
    ),
    (
        "        synchronizeArea(finalArea, active.context, true);",
        "        if (active.touchDraft) {\n            if (touchSession) {\n                touchSession.area = finalArea;\n            }\n        } else {\n            synchronizeArea(finalArea, active.context, true);\n        }",
    ),
    (
        "    function pointerCancel(event) {\n        if (dragState && event.pointerId === dragState.pointerId) {\n            finishDrag(false);",
        "    function pointerCancel(event) {\n        if (dragState && event.pointerId === dragState.pointerId) {\n            if (dragState.touchDraft) {\n                lockTouchEditor(\"Gesture canceled. Unapplied changes discarded and locked.\");\n            } else {\n                finishDrag(false);\n            }",
    ),
    (
        "    function pointerCaptureLost(event) {\n        if (dragState && event.pointerId === dragState.pointerId) {\n            finishDrag(true);",
        "    function pointerCaptureLost(event) {\n        if (dragState && event.pointerId === dragState.pointerId) {\n            if (dragState.touchDraft) {\n                lockTouchEditor(\"Gesture interrupted. Unapplied changes discarded and locked.\");\n            } else {\n                finishDrag(true);\n            }",
    ),
    (
        '    function editorKeyDown(event) {\n        if (event.key === "Escape"',
        '    function editorKeyDown(event) {\n        if (touchMode() && !touchSession) {\n            return;\n        }\n        if (event.key === "Escape"',
    ),
    (
        "        synchronizeArea(candidate, state.context, true);\n        drawPreview(event.currentTarget.closest",
        "        if (touchMode() && touchSession) {\n            touchSession.area = candidate;\n        } else {\n            synchronizeArea(candidate, state.context, true);\n        }\n        drawPreview(event.currentTarget.closest",
    ),
    (
        "        const state = captureState();\n        lastState = state;",
        "        const state = captureState();\n        validateTouchSession(state);\n        lastState = state;\n        if (touchSession && touchSession.preview === state.preview) {\n            const frozenView = dragState ? dragState.view : null;\n            drawPreview(state.preview, state.image, touchSession.area, state.context, frozenView);\n            return;\n        }",
    ),
    (
        '            eventRoot.removeEventListener("load", imageLoaded, true);',
        '            eventRoot.removeEventListener("load", imageLoaded, true);\n            eventRoot.removeEventListener("pointerdown", touchOutsidePointer, true);',
    ),
    (
        '        eventRoot.addEventListener("load", imageLoaded, true);',
        '        eventRoot.addEventListener("load", imageLoaded, true);\n        eventRoot.addEventListener("pointerdown", touchOutsidePointer, true);',
    ),
    (
        "        const nextState = captureState();\n        connectImageObserver",
        "        const nextState = captureState();\n        validateTouchSession(nextState);\n        connectImageObserver",
    ),
]

CSS = r'''
.forge-outpaint-mobile-safety { display: none; }
.forge-outpaint-mobile-safety [hidden], .forge-outpaint-mobile-safety[hidden] { display: none !important; }
@media (pointer: coarse) {
    .forge-outpaint-mobile-safety {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px;
        padding: 12px;
        margin: 6px 0 10px;
        border: 1px solid var(--block-border-color, #9ca3af);
        border-radius: 8px;
        background: var(--block-background-fill, #f3f4f6);
        color: var(--body-text-color, #111827);
        touch-action: pan-y pinch-zoom;
    }
    .forge-outpaint-touch-status { flex: 1 1 100%; font-size: 14px; line-height: 1.5; }
    .forge-outpaint-mobile-safety button {
        min-width: 120px;
        min-height: 48px;
        padding: 10px 14px;
        border: 1px solid var(--block-border-color, #9ca3af);
        border-radius: 8px;
        background: var(--button-secondary-background-fill, #fff);
        color: var(--button-secondary-text-color, #111827);
        font-size: 16px;
        touch-action: manipulation;
    }
    #img2img_outpaint_preview .forge-outpaint-editor-overlay {
        touch-action: pan-y pinch-zoom !important;
    }
    #img2img_outpaint_preview:not([data-forge-outpaint-touch="editing"]) [data-outpaint-handle] {
        pointer-events: none !important;
        touch-action: pan-y pinch-zoom !important;
        cursor: auto;
    }
    #img2img_outpaint_preview:not([data-forge-outpaint-touch="editing"]) .forge-outpaint-handle::after {
        opacity: 0.28;
    }
    #img2img_outpaint_preview[data-forge-outpaint-touch="editing"] [data-outpaint-handle] {
        touch-action: none !important;
    }
    #img2img_outpaint_preview[data-forge-outpaint-touch="editing"] {
        outline: 2px solid var(--primary-500, #3b82f6);
        outline-offset: 2px;
    }
}
'''.strip()


def normalized(value: str) -> str:
    return value.replace("\r\n", "\n")


def audited_source(current: str) -> str:
    source = normalized(current)
    if "// BEGIN FORGE OUTPAINT TOUCH SAFETY" in source:
        for old, new in reversed(EDITS):
            if source.count(new) != 1:
                raise ValueError("Existing touch safety patch has been modified; no files changed.")
            source = source.replace(new, old, 1)
    # The existing, independently idempotent hit-target patch may run before or
    # after this patch. Ignore only its exact audited block when checking source.
    hit_patch = "\n".join([
        "        // BEGIN FORGE MOBILE HIT TARGETS",
        '        const coarsePointer = window.matchMedia("(pointer: coarse)").matches;',
        "        const hit = coarsePointer ? 44 : 28;",
        "        const corner = coarsePointer ? 48 : 38;",
        "        // END FORGE MOBILE HIT TARGETS",
    ])
    canonical = source.replace(hit_patch, "        const hit = 28;\n        const corner = 38;")
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != AUDITED_SHA256:
        raise ValueError("Outpaint source differs from the audited checkout; no files changed.")
    return source


def patched_javascript(current: str) -> str:
    source = audited_source(current)
    for old, new in EDITS:
        if source.count(old) != 1:
            raise ValueError(f"Expected exactly one patch anchor: {old[:65]!r}")
        source = source.replace(old, new, 1)
    return source.replace("\n", "\r\n") if "\r\n" in current else source


def patched_css(current: str) -> str:
    nl = "\r\n" if "\r\n" in current else "\n"
    block = nl.join([CSS_START, CSS.replace("\n", nl), CSS_END])
    if CSS_START in current or CSS_END in current:
        if current.count(CSS_START) != 1 or current.count(CSS_END) != 1:
            raise ValueError("Incomplete or repeated CSS patch markers; no files changed.")
        start = current.index(CSS_START)
        end = current.index(CSS_END) + len(CSS_END)
        if end <= start:
            raise ValueError("Reversed CSS patch markers; no files changed.")
        return current[:start] + block + current[end:]
    separator = "" if not current else (nl if current.endswith(("\n", "\r")) else nl * 2)
    return current + separator + block + nl


def atomic_write(path: Path, content: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content.encode("utf-8"))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    javascript = root / "javascript" / "outpaint.js"
    stylesheet = root / "user.css"
    if not (root / "launch.py").is_file() or not javascript.is_file():
        parser.error("--root must be a Forge copy with launch.py and javascript/outpaint.js")
    try:
        old_js = javascript.read_bytes().decode("utf-8")
        old_css = stylesheet.read_bytes().decode("utf-8") if stylesheet.exists() else ""
        changes = [
            (javascript, old_js, patched_javascript(old_js)),
            (stylesheet, old_css, patched_css(old_css)),
        ]
    except (ValueError, UnicodeError) as error:
        parser.error(str(error))
    for path, old, new in changes:
        if old == new:
            print(f"Already current: {path}")
        elif args.check:
            print(f"Would update: {path}")
        else:
            atomic_write(path, new)
            print(f"Updated: {path}")


if __name__ == "__main__":
    main()
