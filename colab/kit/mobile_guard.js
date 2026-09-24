/* Forge Colab mobile safety. Loaded before the other javascript/*.js files.
 * This is a user-interaction guard, not a security boundary. Programmatic model
 * updates and Gradio's hidden controls retain their original behavior.
 */
(() => {
    "use strict";
    // BEGIN ENGLISH DISPLAY UPGRADE
    // Upgrade only this kit's display nodes. Never install a second interaction
    // guard over an older server guard, and never translate prompts/model data.
    function installEnglishDisplay() {
        if (window.forgeMobileEnglishDisplay) return;
        const pairs = [["\u8c03\u6574\u4e2d\uff0c\u5c1a\u672a\u5e94\u7528\uff1a(.+?)\\ \u00d7\\ (.+?)\uff1b\u70b9\u51fb\u201c\u5e94\u7528\u5e76\u9501\u5b9a\u201d\u4fdd\u5b58\u3002","Unapplied draft: $1 \u00d7 $2. Tap Apply and lock to save."],["(.+?)\u5df2\u5e94\u7528\u4e3a\\ ([^\\s\u8def]+)","$1 set to $2"],["\u53c2\u6570\u4fee\u6539\u4e2d\\ \u00b7\\ \u6570\u503c\u9700\u5e94\u7528\u786e\u8ba4\\ \u00b7\\ \u95f2\u7f6e\\ 60\\ \u79d2\u81ea\u52a8\u9501\u5b9a","Editing parameters \u00b7 Apply numbers to confirm \u00b7 Locks after 60 seconds idle"],["\u6269\u56fe\u5df2\u9501\u5b9a\uff1b\u53ef\u6ed1\u52a8\u9875\u9762\u3002\u7cbe\u786e\u6570\u503c\u53ef\u5728\u53c2\u6570\u533a\u4fee\u6539\u3002","Outpaint locked. Scroll the page, or edit exact values in the parameter section."],["\u539f\u53c2\u6570\u5df2\u88ab\u6a21\u578b\u6216\u754c\u9762\u66f4\u65b0\uff0c\u8bf7\u53d6\u6d88\u540e\u91cd\u65b0\u6253\u5f00\u3002","The model or interface changed this parameter. Cancel and reopen the editor."],["\u53c2\u6570\u5df2\u9501\u5b9a\\ \u00b7\\ \u53ef\u6eda\u52a8\u3001\u5199\u63d0\u793a\u8bcd\u548c\u4e0a\u4f20\u56fe\u7247","Parameters locked \u00b7 Scroll, write prompts and upload images"],["\u591a\u6307\u624b\u52bf\u5df2\u53d6\u6d88\uff0c\u672a\u5e94\u7528\u7684\u8c03\u6574\u5df2\u64a4\u9500\u5e76\u9501\u5b9a\u3002","Multi-touch canceled. Unapplied changes discarded and locked."],["\u3002\u786e\u8ba4\u540e\u53c2\u6570\u81ea\u52a8\u9501\u5b9a\uff0c\u672c\u6b21\u4ec5\u63d0\u4ea4\u4e00\u6b21\u3002",". Parameters lock after confirmation. This submits once."],["\u624b\u52bf\u5df2\u53d6\u6d88\uff0c\u672a\u5e94\u7528\u7684\u8c03\u6574\u5df2\u64a4\u9500\u5e76\u9501\u5b9a\u3002","Gesture canceled. Unapplied changes discarded and locked."],["\u624b\u52bf\u5df2\u4e2d\u65ad\uff0c\u672a\u5e94\u7528\u7684\u8c03\u6574\u5df2\u64a4\u9500\u5e76\u9501\u5b9a\u3002","Gesture interrupted. Unapplied changes discarded and locked."],["\u53c2\u6570\u5df2\u9501\u5b9a\u6216\u539f\u63a7\u4ef6\u5df2\u4e0d\u53ef\u7528\uff0c\u8bf7\u53d6\u6d88\u3002","This parameter is locked or unavailable. Cancel this edit."],["\u6269\u56fe\u8c03\u6574\u5df2\u5e94\u7528\uff0c\u5df2\u9501\u5b9a\uff1b\u53ef\u6ed1\u52a8\u9875\u9762\u3002","Outpaint changes applied and locked. You can scroll the page."],["\u539f\u64cd\u4f5c\u5df2\u4e0d\u53ef\u7528\uff0c\u8bf7\u53d6\u6d88\u540e\u91cd\u65b0\u68c0\u67e5\u3002","This action is no longer available. Cancel and check the page."],["\u8bf7\u6309\\ (.+?)\\ \u7684\u95f4\u9694\u586b\u5199\u3002","Use increments of $1."],["\u5df2\u64a4\u9500\u672c\u6b21\u8c03\u6574\u5e76\u9501\u5b9a\uff1b\u53ef\u6ed1\u52a8\u9875\u9762\u3002","Draft discarded and locked. You can scroll the page."],["\u672a\u5e94\u7528\u7684\u6269\u56fe\u8c03\u6574\u5df2\u64a4\u9500\uff0c\u5df2\u9501\u5b9a\u3002","Unapplied outpaint changes discarded and locked."],["\u53c2\u6570\u5df2\u9501\u5b9a\uff0c\u8bf7\u5148\u70b9\u201c\u4fee\u6539\u53c2\u6570\u201d","Parameters are locked. Tap Edit parameters first"],["\u5f53\u524d\u503c\\ ([^\\s\u8def]+)","Current value $1"],["\u624b\u673a\u5b89\u5168\u6a21\u5f0f\u4e0d\u652f\u6301\u8fde\u7eed\u81ea\u52a8\u751f\u6210","Continuous automatic generation is disabled in mobile safety mode"],["\u753b\u5e03\u5df2\u9501\u5b9a\uff0c\u8bf7\u5148\u70b9\u201c\u4fee\u6539\u53c2\u6570\u201d","Canvas locked. Tap Edit parameters first"],["\u65b0\u6570\u503c\uff08\u5e94\u7528\u524d\u4e0d\u4f1a\u6539\u53d8\u53c2\u6570\uff09","New value (the parameter changes only after Apply)"],["\\ \u6b64\u64cd\u4f5c\u53ea\u6709\u70b9\u786e\u8ba4\u540e\u624d\u6267\u884c\u3002"," This action runs only after you confirm."],["\u5df2\u786e\u8ba4\u8986\u76d6\u64cd\u4f5c\uff0c\u53c2\u6570\u91cd\u65b0\u9501\u5b9a","Replacement confirmed. Parameters locked again"],["\u6570\u503c\u4e0d\u80fd\u5c0f\u4e8e\\ (.+?)\u3002","The value must be at least $1."],["\u6570\u503c\u4e0d\u80fd\u5927\u4e8e\\ (.+?)\u3002","The value must not exceed $1."],["\u95f2\u7f6e\\ 60\\ \u79d2\uff0c\u53c2\u6570\u5df2\u9501\u5b9a","Idle for 60 seconds. Parameters locked"],["\u53c2\u6570\u5df2\u9501\u5b9a\uff0c\u5df2\u786e\u8ba4\u672c\u6b21\u751f\u6210","Parameters locked. This generation was confirmed"],["\u624b\u52bf\u672a\u5b8c\u6210\uff0c\u5df2\u64a4\u9500\u5e76\u9501\u5b9a\u3002","Gesture incomplete. Draft discarded and locked."],["\u5df2\u5207\u6362\u9875\u9762\uff0c\u53c2\u6570\u81ea\u52a8\u9501\u5b9a","Page changed. Parameters locked"],["\u9875\u9762\u5931\u53bb\u7126\u70b9\uff0c\u53c2\u6570\u5df2\u9501\u5b9a","Page lost focus. Parameters locked"],["\u9875\u9762\u5207\u5230\u540e\u53f0\uff0c\u53c2\u6570\u5df2\u9501\u5b9a","Page moved to background. Parameters locked"],["\u590d\u5236\u56fe\u7247\u5e76\u8986\u76d6\u76ee\u6807\u753b\u5e03","Copy image and replace the destination canvas"],["\u53d1\u9001\u56fe\u7247\u5e76\u8986\u76d6\u76ee\u6807\u53c2\u6570","Send image and replace destination settings"],["\u5df2\u53d6\u6d88\u64cd\u4f5c\uff0c\u53c2\u6570\u5df2\u9501\u5b9a","Action canceled. Parameters locked"],["\u5df2\u64a4\u9500\u672c\u6b21\u8c03\u6574\u5e76\u9501\u5b9a\u3002","Draft discarded and locked."],["\u53d6\u6d88\u4f1a\u4fdd\u7559\u5f53\u524d\u72b6\u6001\u3002","Cancel keeps the current state."],["\u786e\u8ba4(.+?)\uff1f","Confirm: $1?"],["\u95f4\u9694\\ ([^\\s\u8def]+)","Step $1"],["\u4fee\u6539\u5b8c\u6210\uff0c\u53c2\u6570\u5df2\u9501\u5b9a","Editing finished. Parameters locked"],["\u8fd4\u56de\u9875\u9762\uff0c\u53c2\u6570\u5df2\u9501\u5b9a","Returned to page. Parameters locked"],["\u6700\u5c0f\\ ([^\\s\u8def]+)","Minimum $1"],["\u6700\u5927\\ ([^\\s\u8def]+)","Maximum $1"],["\u5220\u9664\u5f53\u524d\u753b\u5e03\u56fe\u7247","Delete the canvas image"],["\u8bf7\u8f93\u5165\u6709\u6548\u6570\u5b57\u3002","Enter a valid number."],["\u5207\u6362\u6b63\u8d1f\u53f7\\ \u00b1","Switch sign +/-"],["\u624b\u673a\u9632\u8bef\u89e6\u4fdd\u62a4","Mobile touch protection"],["\uff08\u9700\u5e94\u7528\u786e\u8ba4\uff09"," (Apply to confirm)"],["\u624b\u673a\u6269\u56fe\u9632\u8bef\u89e6","Mobile outpaint touch protection"],["\u8df3\u8fc7\u5f53\u524d\u6279\u6b21","Skip this batch"],["\u91cd\u7f6e\u5f53\u524d\u753b\u5e03","Reset the canvas"],["\u64a4\u9500\u753b\u5e03\u4fee\u6539","Undo the canvas edit"],["\u91cd\u505a\u753b\u5e03\u4fee\u6539","Redo the canvas edit"],["\u70b9\u6b64\u7f16\u8f91\u6570\u503c","Edit value"],["\u6e05\u7a7a\u63d0\u793a\u8bcd","Clear prompts"],["\u53c2\u6570\u5df2\u9501\u5b9a","Parameters locked"],["\u5e94\u7528\u5e76\u9501\u5b9a","Apply and lock"],["\u64a4\u9500\u5e76\u9501\u5b9a","Discard and lock"],["\u4fee\u6539\u53c2\u6570","Edit parameters"],["\u5b8c\u6210\u4fee\u6539","Finish editing"],["\u5f00\u59cb\u751f\u6210","Start generation"],["\u4e2d\u65ad\u751f\u6210","Interrupt generation"],["\u6267\u884c\u64cd\u4f5c","Run action"],["\u6bcf\u6279\u56fe\u7247","Images per batch"],["\u6570\u503c\u53c2\u6570","Numeric parameter"],["\u5e94\u7528\u6570\u503c","Apply value"],["\u8c03\u6574\u6269\u56fe","Edit outpaint"],["\u6279\u6b21\u6570","Batches"],["\u4fee\u6539\u201c","Edit: "],["\u6b65\u6570","Steps"],["\u53d6\u6d88","Cancel"],["\u786e\u8ba4","Confirm "],["\u7f16\u8f91","Edit "],["\u5bbd","Width"],["\u9ad8","Height"],["\u201d",""]].map(([pattern, text]) => [new RegExp(pattern, "g"), text]);
        const scopes = "#forge-mobile-guard-bar,#forge-mobile-guard-dialog,.forge-mobile-number-edit,.forge-outpaint-mobile-safety";
        const convert = raw => {
            if (!/[\u3400-\u9fff]/.test(raw)) return raw;
            let value = raw;
            for (const [pattern, text] of pairs) value = value.replace(pattern, text);
            return /[\u3400-\u9fff]/.test(value) ? "Review this item before applying." : value;
        };
        let scheduled = false;
        const run = () => {
            scheduled = false;
            document.querySelectorAll(scopes).forEach(scope => {
                const walker = document.createTreeWalker(scope, NodeFilter.SHOW_TEXT);
                let node;
                while ((node = walker.nextNode())) {
                    const value = convert(node.nodeValue);
                    if (value !== node.nodeValue) node.nodeValue = value;
                }
                for (const element of [scope, ...scope.querySelectorAll("[aria-label],[title]")]) {
                    for (const attribute of ["aria-label", "title"]) {
                        const raw = element.getAttribute(attribute);
                        if (raw !== null) { const value = convert(raw); if (value !== raw) element.setAttribute(attribute, value); }
                    }
                }
            });
        };
        const init = () => {
            run();
            new MutationObserver(() => { if (!scheduled) { scheduled = true; queueMicrotask(run); } })
                .observe(document.body, {childList: true, subtree: true, characterData: true,
                    attributes: true, attributeFilter: ["aria-label", "title"]});
        };
        window.forgeMobileEnglishDisplay = Object.freeze({version: 1});
        if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, {once: true});
        else init();
    }
    installEnglishDisplay();
    // END ENGLISH DISPLAY UPGRADE
    if (window.forgeMobileGuard) return;

    const coarse = window.matchMedia("(pointer: coarse)");
    const own = "#forge-mobile-guard-bar, #forge-mobile-guard-dialog";
    const outpaint = ".forge-outpaint-mobile-safety, .forge-outpaint-editor-overlay";
    const numericSelector = 'input[type="number"], input[type="range"]';
    const snapshots = new Map();
    const covers = new Map();
    const approved = new WeakSet();
    const committed = new WeakMap();
    let locked = true;
    let timer;
    let bar;
    let status;
    let toggle;
    let dialogState;
    let scanQueued = false;
    let gesture;
    let suppressClickUntil = 0;
    let noticeTimer;
    let pendingHistoryClose = 0;

    function root() { return typeof gradioApp === "function" ? gradioApp() : document; }
    function element(event) { return event.composedPath().find(node => node instanceof Element); }
    function excluded(target) { return !target || !!target.closest(own + ", " + outpaint); }
    function stop(event, prevent = true) {
        if (prevent && event.cancelable) event.preventDefault();
        event.stopImmediatePropagation();
    }
    function visible(target) {
        return !!target && target.isConnected && target.getClientRects().length > 0 &&
            getComputedStyle(target).visibility !== "hidden";
    }
    function announce(message) {
        if (!status) return;
        status.textContent = message;
        clearTimeout(noticeTimer);
        noticeTimer = setTimeout(render, 3200);
    }
    function render() {
        document.documentElement.classList.toggle("forge-mobile-guard-active", coarse.matches);
        document.documentElement.classList.toggle("forge-mobile-parameters-locked", coarse.matches && locked);
        if (!bar) return;
        bar.hidden = !coarse.matches;
        bar.dataset.locked = String(locked);
        toggle.textContent = locked ? "Edit parameters" : "Finish editing";
        toggle.setAttribute("aria-pressed", String(!locked));
        status.textContent = locked ? "Parameters locked · Scroll, write prompts and upload images" : "Editing parameters · Apply numbers to confirm · Locks after 60 seconds idle";
    }
    function armTimer() {
        clearTimeout(timer);
        if (!locked && coarse.matches) timer = setTimeout(() => lock("Idle for 60 seconds. Parameters locked"), 60000);
    }
    function lock(message, close = true) {
        locked = true;
        clearTimeout(timer);
        if (close && dialogState) dismiss(false);
        const active = document.activeElement;
        if (active && !active.closest(own) && isParameter(active)) active.blur();
        render();
        if (message) announce(message);
    }
    function canUse(target) {
        return target && !target.disabled && target.getAttribute("aria-disabled") !== "true" && visible(target);
    }
    function isNavigation(target) {
        return !!target.closest('.tab-nav > button, [role="tab"], .extra-network-subdirs, .tree-list-content-dir');
    }
    function isPromptOrSearch(target) {
        return !!target.closest('#txt2img_prompt, #txt2img_neg_prompt, #img2img_prompt, #img2img_neg_prompt, #settings_search, .extra-network-search') ||
            target.matches('input[type="search"], input[type="file"]') ||
            (target.matches("input, textarea") && /search|filter/i.test(target.getAttribute("placeholder") || ""));
    }
    function numeric(target) {
        if (!target || excluded(target)) return null;
        if (target.matches(numericSelector)) return target;
        if (target.classList.contains("forge-mobile-number-edit")) return covers.get(target)?.input || null;
        const field = target.closest(".gradio-slider, .gradio-number, .forge-range-row");
        return field?.querySelector('input[type="number"]') || field?.querySelector('input[type="range"]') || null;
    }
    function isParameter(target) {
        if (!target || excluded(target) || isPromptOrSearch(target) || isNavigation(target)) return false;
        if (numeric(target)) return true;
        if (target.closest('.input-accordion > .label-wrap, .extra-network-cards .card, [id$="_cards"] .card, .tree-list-content-file[onclick]')) return true;
        if (target.closest('.gradio-dropdown, .gradio-radio, .gradio-checkbox, .gradio-checkboxgroup, [role="listbox"], [role="option"]')) return true;
        if (target.matches("input, textarea, select, [role='checkbox'], [role='radio'], [role='switch']")) return true;
        const button = target.closest("button, .gradio-button");
        if (!button) return false;
        return /(?:random|reuse)_(?:sub)?seed|res_switch|detect_image_size|style_apply|style_copy|^paste$|_send_to_|^img2img_copy_to_|_interrogate|^interrogate$|^deepbooru$/.test(button.id) ||
            !!button.closest('[id^="img2img_copy_to_"]') ||
            !!button.closest("#quicksettings") ||
            /read generation parameters|send.*parameters|apply.*styles/i.test(button.title || "");
    }
    function dangerous(target) {
        const button = target?.closest('button, .gradio-button, [role="button"], a');
        if (!button || excluded(button)) return null;
        const id = button.id || "";
        if (/(?:^|_)(generate|interrupt|skip|interrupting|clear_prompt|upscale)$/.test(id)) return button;
        if (/^img2img_copy_to_|_send_to_/.test(id) || button.closest('[id^="img2img_copy_to_"]')) return button;
        if (/^(remove|reset|undo|redo)Button_/.test(id)) return button;
        if (/^(settings_submit|settings_restart_gradio|settings_reload_script_bodies|sett_unload_sd_model|ui_defaults_apply|calculate_all_checkpoint_hash)$/.test(id)) return button;
        if (/_edit_style_(save|delete)$/.test(id)) return button;
        const labels = [button.textContent, button.title, button.getAttribute("aria-label")].filter(Boolean).map(text => text.trim());
        if (labels.some(label => /^(generate|interrupt|skip|delete|remove|reset|clear|restart|reload|restore|apply settings|save settings|save defaults|unload|apply and restart|install)(?:\s|$)/i.test(label))) return button;
        if (labels.some(label => /^(生成|中断|跳过|删除|移除|重置|清空|清除|重启|恢复默认|保存设置|应用设置|安装)/.test(label))) return button;
        return null;
    }
    function isCanvas(target) {
        return !!target.closest(".forge-drawing-canvas, .forge-image, .forge-resize-line") && !target.closest(outpaint);
    }
    function actionName(button) {
        const id = button.id || "";
        if (/_generate$/.test(id)) return "Start generation";
        if (/_interrupt$/.test(id)) return "Interrupt generation";
        if (/_skip$/.test(id)) return "Skip this batch";
        if (/_clear_prompt$/.test(id)) return "Clear prompts";
        if (/^removeButton_/.test(id)) return "Delete the canvas image";
        if (/^resetButton_/.test(id)) return "Reset the canvas";
        if (/^undoButton_/.test(id)) return "Undo the canvas edit";
        if (/^redoButton_/.test(id)) return "Redo the canvas edit";
        if (/^img2img_copy_to_/.test(id) || button.closest('[id^="img2img_copy_to_"]')) return "Copy image and replace the destination canvas";
        if (/_send_to_/.test(id)) return "Send image and replace destination settings";
        const label = (button.getAttribute("aria-label") || button.title || button.textContent || "Run action").trim().slice(0, 180);
        return /[\u3400-\u9fff]/.test(label) ? "Run action" : label;
    }
    function generationSummary(button) {
        const tab = button.id.startsWith("img2img") ? "img2img" : "txt2img";
        const value = suffix => root().querySelector(`#${tab}_${suffix} input[type="number"], #${tab}_${suffix} input[type="range"]`)?.value;
        const parts = [["width", "Width"], ["height", "Height"], ["steps", "Steps"], ["batch_count", "Batches"], ["batch_size", "Images per batch"]]
            .map(([suffix, label]) => value(suffix) === undefined ? "" : `${label} ${value(suffix)}`).filter(Boolean);
        return parts.join(" · ");
    }
    function fieldLabel(input) {
        const field = input.closest(".gradio-slider, .gradio-number, .forge-range-row, .block");
        const label = (field?.querySelector("label span, label, .forge-toolbar-label")?.textContent || input.getAttribute("aria-label") || field?.id || input.id || "Numeric parameter").trim();
        return /[\u3400-\u9fff]/.test(label) ? (field?.id || input.id || "Numeric parameter") : label;
    }
    function dismiss(fromHistory) {
        const state = dialogState;
        if (!state) return;
        dialogState = null;
        if (state.dialog.open) state.dialog.close();
        state.dialog.remove();
        if (!fromHistory && history.state?.forgeMobileDialog === state.id) {
            pendingHistoryClose += 1;
            history.back();
        }
        if (state.returnFocus && visible(state.returnFocus)) state.returnFocus.focus({preventScroll: true});
    }
    function modal(title, description, applyLabel, callback, inputSpec) {
        if (dialogState || pendingHistoryClose) return;
        const dialog = document.createElement("dialog");
        dialog.id = "forge-mobile-guard-dialog";
        dialog.setAttribute("aria-labelledby", "forge-mobile-guard-title");
        dialog.setAttribute("aria-describedby", "forge-mobile-guard-description");
        const heading = document.createElement("h2");
        heading.id = "forge-mobile-guard-title";
        heading.textContent = title;
        const desc = document.createElement("p");
        desc.id = "forge-mobile-guard-description";
        desc.textContent = description;
        const error = document.createElement("p");
        error.className = "forge-mobile-error";
        error.setAttribute("role", "alert");
        const buttons = document.createElement("div");
        buttons.className = "forge-mobile-dialog-buttons";
        const cancel = document.createElement("button");
        cancel.type = "button";
        cancel.id = "forge-mobile-guard-cancel";
        cancel.textContent = "Cancel";
        const apply = document.createElement("button");
        apply.type = "button";
        apply.id = "forge-mobile-guard-apply";
        apply.textContent = applyLabel;
        dialog.append(heading, desc);
        let draft;
        if (inputSpec) {
            const label = document.createElement("label");
            label.htmlFor = "forge-mobile-guard-draft";
            label.textContent = "New value (the parameter changes only after Apply)";
            draft = document.createElement("input");
            draft.type = "text";
            draft.inputMode = inputSpec.decimal ? "decimal" : "numeric";
            draft.id = "forge-mobile-guard-draft";
            draft.autocomplete = "off";
            draft.value = inputSpec.value;
            draft.setAttribute("aria-describedby", "forge-mobile-guard-description");
            draft.addEventListener("input", armTimer);
            dialog.append(label, draft);
            if (inputSpec.negative) {
                const sign = document.createElement("button");
                sign.type = "button";
                sign.id = "forge-mobile-guard-sign";
                sign.textContent = "Switch sign +/-";
                sign.addEventListener("click", () => {
                    draft.value = draft.value.startsWith("-") ? draft.value.slice(1) : "-" + draft.value;
                    armTimer();
                    draft.focus();
                });
                dialog.append(sign);
            }
        }
        dialog.append(error, buttons);
        buttons.append(cancel, apply);
        document.body.append(dialog);
        const id = `forge-${Date.now()}-${Math.random().toString(36).slice(2)}`;
        dialogState = {dialog, id, returnFocus: document.activeElement, opening: performance.now()};
        try { history.pushState({...history.state, forgeMobileDialog: id}, "", location.href); } catch (_) { /* Sandboxed embedding can deny history. */ }
        cancel.addEventListener("click", () => dismiss(false));
        apply.addEventListener("click", () => {
            if (!dialogState || apply.disabled || performance.now() - dialogState.opening < 250) return;
            const result = callback(draft?.value, true);
            if (typeof result === "string") { error.textContent = result; draft?.focus(); return; }
            apply.disabled = true;
            dismiss(false);
            callback(draft?.value, false);
        });
        dialog.addEventListener("cancel", event => { event.preventDefault(); dismiss(false); lock("Action canceled. Parameters locked", false); });
        dialog.addEventListener("click", event => { if (event.target === dialog) {
            const bounds = dialog.getBoundingClientRect();
            if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dismiss(false);
        } });
        dialog.showModal();
        cancel.focus({preventScroll: true});
    }
    function confirmAction(button) {
        if (!canUse(button) || dialogState || performance.now() - (committed.get(button) || -Infinity) < 1500) return;
        const name = actionName(button);
        const generate = /_(generate|upscale)$/.test(button.id);
        const detail = generate ? generationSummary(button) : "Cancel keeps the current state.";
        modal(`Confirm: ${name}?`, detail + (generate ? ". Parameters lock after confirmation. This submits once." : " This action runs only after you confirm."), "Confirm " + name, (_draft, validate) => {
            if (validate) return canUse(button) ? null : "This action is no longer available. Cancel and check the page.";
            if (generate) lock("Parameters locked. This generation was confirmed", false);
            else if (/^img2img_copy_to_|_send_to_/.test(button.id) || button.closest('[id^="img2img_copy_to_"]')) lock("Replacement confirmed. Parameters locked again", false);
            committed.set(button, performance.now());
            approved.add(button);
            button.click();
            approved.delete(button);
        });
    }
    function editNumber(input) {
        if (locked) { announce("Parameters are locked. Tap Edit parameters first"); return; }
        if (!canUse(input) || snapshots.get(input)?.readOnly || dialogState) return;
        armTimer();
        const wrapper = input.closest(".gradio-slider, .gradio-number, .forge-range-row");
        const range = input.type === "range" ? input : wrapper?.querySelector('input[type="range"]');
        const source = range || input;
        const min = source.min === "" ? null : Number(source.min);
        const max = source.max === "" ? null : Number(source.max);
        const stepText = source.step || (source.type === "range" ? "1" : "any");
        const step = stepText === "any" ? null : Number(stepText);
        const original = input.value;
        const detail = [`Current value ${original}`, min !== null ? `Minimum ${min}` : "", max !== null ? `Maximum ${max}` : "", step ? `Step ${step}` : ""].filter(Boolean).join(" · ");
        modal("Edit: " + fieldLabel(input), detail, "Apply value", (raw, validate) => {
            const value = Number(raw);
            if (validate) {
                if (locked || !canUse(input)) return "This parameter is locked or unavailable. Cancel this edit.";
                if (input.value !== original) return "The model or interface changed this parameter. Cancel and reopen the editor.";
                if (!raw?.trim() || !Number.isFinite(value)) return "Enter a valid number.";
                if (min !== null && value < min) return `The value must be at least ${min}.`;
                if (max !== null && value > max) return `The value must not exceed ${max}.`;
                if (step && Math.abs((value - (min ?? 0)) / step - Math.round((value - (min ?? 0)) / step)) > 0.000001) return `Use increments of ${step}.`;
                return null;
            }
            input.value = String(value);
            input.dispatchEvent(new Event("input", {bubbles: true}));
            input.dispatchEvent(new Event("change", {bubbles: true}));
            armTimer();
            announce(`${fieldLabel(input)} set to ${value}`);
        }, {value: original, decimal: stepText === "any" || String(stepText).includes("."), negative: min === null || min < 0});
    }
    function guardClick(event) {
        if (!coarse.matches) return;
        const target = element(event);
        if (!target) return;
        if (event.isTrusted && performance.now() < suppressClickUntil) { stop(event); return; }
        if (excluded(target)) return;
        if (target.closest("#context-menu")) { stop(event); announce("Continuous automatic generation is disabled in mobile safety mode"); return; }
        const action = dangerous(target);
        if (action && approved.has(action)) { approved.delete(action); return; }
        if (action && event.isTrusted && locked && isParameter(action)) {
            stop(event); announce("Parameters are locked. Tap Edit parameters first"); return;
        }
        if (action) { stop(event); confirmAction(action); return; }
        if (!event.isTrusted) return;
        if (isNavigation(target)) {
            if (target.closest("#tabs > .tab-nav, #mode_img2img > .tab-nav")) lock("Page changed. Parameters locked");
            return;
        }
        const number = numeric(target);
        if (number) { stop(event); editNumber(number); return; }
        if (locked && isParameter(target)) { stop(event); announce("Parameters are locked. Tap Edit parameters first"); return; }
        if (locked && isCanvas(target)) { stop(event); announce("Canvas locked. Tap Edit parameters first"); return; }
        if (!locked) armTimer();
    }
    function guardDown(event) {
        if (!coarse.matches || !event.isTrusted) return;
        const target = element(event);
        if (excluded(target)) return;
        if (target.closest("#context-menu")) { stop(event); return; }
        if (dangerous(target)) { stop(event, false); return; }
        if (numeric(target) || (locked && (isParameter(target) || isCanvas(target)))) {
            // Do not cancel touch scrolling. Readonly fields and range covers
            // prevent native edits; click capture handles the intentional tap.
            stop(event, event.type === "mousedown");
        }
    }
    function guardKey(event) {
        if (!coarse.matches || !event.isTrusted) return;
        const target = element(event);
        if (dialogState) {
            if (event.key === "Escape") { stop(event); dismiss(false); lock("Action canceled. Parameters locked", false); return; }
            if (event.key === "Enter" && !target?.matches("#forge-mobile-guard-cancel, #forge-mobile-guard-apply")) { stop(event); return; }
            // Keep Forge's document/global shortcuts away from the draft input.
            if (event.ctrlKey || event.metaKey || event.altKey) event.stopImmediatePropagation();
            return;
        }
        if (excluded(target)) return;
        const enter = event.key === "Enter";
        const shortcut = (enter && (event.ctrlKey || event.metaKey || event.altKey)) || event.key === "Escape";
        if (shortcut && !visible(root().querySelector("#lightboxModal"))) {
            const tab = visible(root().querySelector("#tab_img2img")) ? "img2img" : "txt2img";
            const suffix = event.key === "Escape" ? "interrupt" : event.altKey ? "skip" : "generate";
            const action = root().querySelector(`#${tab}_${suffix}`);
            if (canUse(action)) { stop(event); confirmAction(action); return; }
        }
        if (locked && isParameter(target) && !["Tab", "Shift", "Escape"].includes(event.key)) { stop(event); announce("Parameters are locked. Tap Edit parameters first"); return; }
        const number = numeric(target);
        if (number && ["Enter", " ", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "PageUp", "PageDown", "Home", "End"].includes(event.key)) {
            stop(event); editNumber(number);
        }
    }
    function scan() {
        scanQueued = false;
        if (!bar || !bar.isConnected) {
            bar = document.createElement("div");
            bar.id = "forge-mobile-guard-bar";
            bar.setAttribute("role", "region");
            bar.setAttribute("aria-label", "Mobile touch protection");
            status = document.createElement("span");
            status.id = "forge-mobile-guard-status";
            status.setAttribute("role", "status");
            status.setAttribute("aria-live", "polite");
            toggle = document.createElement("button");
            toggle.id = "forge-mobile-guard-toggle";
            toggle.type = "button";
            toggle.addEventListener("click", () => {
                if (!coarse.matches) return;
                if (locked) { locked = false; render(); armTimer(); }
                else lock("Editing finished. Parameters locked");
            });
            bar.append(status, toggle);
            document.body.append(bar);
            render();
        }
        for (const [input, saved] of snapshots) {
            if (!input.isConnected) { snapshots.delete(input); continue; }
            if (!coarse.matches) {
                input.readOnly = saved.readOnly;
                if (saved.tabindex === null) input.removeAttribute("tabindex"); else input.setAttribute("tabindex", saved.tabindex);
                input.removeAttribute("data-forge-mobile-number");
                snapshots.delete(input);
            }
        }
        for (const [cover, item] of covers) {
            if (!coarse.matches || !item.input.isConnected) { cover.remove(); covers.delete(cover); }
        }
        if (!coarse.matches) return;
        root().querySelectorAll(numericSelector).forEach(input => {
            if (excluded(input)) return;
            if (!snapshots.has(input)) snapshots.set(input, {readOnly: input.readOnly, tabindex: input.getAttribute("tabindex")});
            input.setAttribute("data-forge-mobile-number", "");
            if (input.type === "number") input.readOnly = true;
            if (input.type === "range") {
                input.tabIndex = -1;
                // A separate sibling leaves Svelte's input and hidden state in place.
                let cover = input.nextElementSibling;
                if (!cover?.classList.contains("forge-mobile-number-edit")) {
                    cover = document.createElement("button");
                    cover.className = "forge-mobile-number-edit";
                    cover.type = "button";
                    cover.textContent = "Edit value";
                    cover.setAttribute("aria-label", "Edit " + fieldLabel(input) + " (Apply to confirm)");
                    input.insertAdjacentElement("afterend", cover);
                    covers.set(cover, {input});
                }
                // Honor inline display:none used by the Qwen frontend, including
                // when only a brush input (rather than its wrapper) is hidden.
                const hideCover = getComputedStyle(input).display === "none" || input.hidden;
                if (cover.hidden !== hideCover) cover.hidden = hideCover;
            }
        });
    }
    function queueScan() {
        if (scanQueued) return;
        scanQueued = true;
        requestAnimationFrame(scan);
    }

    // Window capture runs before existing document/Gradio bubble listeners.
    window.addEventListener("click", guardClick, true);
    window.addEventListener("pointerdown", event => {
        if (coarse.matches && event.isTrusted) {
            if (gesture) gesture.moved = true;
            else {
                // A fresh, deliberate finger-down starts a new gesture. Only
                // the click synthesized from the previous drag is suppressed.
                suppressClickUntil = 0;
                gesture = {id: event.pointerId, x: event.clientX, y: event.clientY, moved: false};
            }
        }
        guardDown(event);
    }, true);
    window.addEventListener("pointermove", event => {
        if (gesture && (Math.abs(event.clientX - gesture.x) > 10 || Math.abs(event.clientY - gesture.y) > 10)) gesture.moved = true;
    }, {capture: true, passive: true});
    window.addEventListener("pointerup", () => {
        if (gesture?.moved) suppressClickUntil = performance.now() + 700;
        gesture = null;
    }, true);
    window.addEventListener("pointercancel", () => { gesture = null; suppressClickUntil = performance.now() + 700; }, true);
    window.addEventListener("mousedown", guardDown, true);
    window.addEventListener("touchstart", event => {
        if (!coarse.matches) return;
        const target = element(event);
        if (dangerous(target) || target?.closest("#context-menu")) stop(event, false);
        else guardDown(event);
        if (event.touches.length > 1 && !target?.closest(outpaint)) suppressClickUntil = performance.now() + 900;
    }, {capture: true, passive: false});
    window.addEventListener("contextmenu", event => {
        if (coarse.matches && (dangerous(element(event)) || element(event)?.closest("#context-menu"))) { stop(event); announce("Continuous automatic generation is disabled in mobile safety mode"); }
    }, true);
    window.addEventListener("wheel", event => {
        // Readonly number fields cannot native-step. Keep the default page
        // scroll while stopping application wheel handlers from changing them.
        if (coarse.matches && numeric(element(event))) stop(event, false);
    }, {capture: true, passive: false});
    window.addEventListener("keydown", guardKey, true);
    window.addEventListener("beforeinput", event => {
        const target = element(event);
        if (coarse.matches && event.isTrusted && !excluded(target) && (numeric(target) || (locked && isParameter(target)))) stop(event);
    }, true);
    window.addEventListener("popstate", () => {
        if (pendingHistoryClose) { pendingHistoryClose -= 1; return; }
        if (dialogState) dismiss(true);
        lock("Returned to page. Parameters locked", false);
    });
    window.addEventListener("blur", () => { if (coarse.matches) lock("Page lost focus. Parameters locked"); });
    document.addEventListener("visibilitychange", () => { if (coarse.matches && document.hidden) lock("Page moved to background. Parameters locked"); });
    coarse.addEventListener("change", () => { lock(); scan(); });
    window.forgeMobileGuard = Object.freeze({
        get active() { return coarse.matches; },
        get locked() { return locked; },
        lock: () => lock("Parameters locked"),
        version: "2026-09-24.1"
    });
    function init() {
        scan();
        const observer = new MutationObserver(queueScan);
        observer.observe(document.body, {childList: true, subtree: true, attributes: true, attributeFilter: ["style", "hidden", "class"]});
        if (typeof onAfterUiUpdate === "function") onAfterUiUpdate(queueScan);
        if (typeof onUiLoaded === "function") onUiLoaded(queueScan);
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, {once: true});
    else init();
})();
