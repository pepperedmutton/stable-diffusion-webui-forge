function getForgePresetValue() {
    const presetRoot = gradioApp().querySelector("#forge_ui_preset");
    if (!presetRoot) return null;

    const checked = presetRoot.querySelector('input[type="radio"]:checked');
    if (checked?.value) {
        return checked.value.toLowerCase();
    }

    const selectedLabel = Array.from(presetRoot.querySelectorAll("label")).find((label) => {
        return label.classList.contains("selected") || label.getAttribute("aria-checked") === "true";
    });

    return selectedLabel?.textContent?.trim().toLowerCase() || null;
}

// Forge loads UI scripts in isolated scopes. Export the preset reader explicitly
// so Extra Networks can apply the same semantic preset after cards are rendered.
window.getForgePresetValue = getForgePresetValue;

let lastForgePresetValue = null;
let forgePresetApplyTimer = null;
let forgePresetRefreshTimer = null;

function refreshForgePresetTokenCounters() {
    // Called by Gradio only after the backend has selected the new model family.
    // Page load can update the radio without firing a user input event.
    applyForgePresetClass();
    ["txt2img_token_button", "img2img_token_button"].forEach((buttonId) => {
        gradioApp().getElementById(buttonId)?.click();
    });
    return [];
}
window.refreshForgePresetTokenCounters = refreshForgePresetTokenCounters;

function applyForgePresetClass() {
    const value = getForgePresetValue();
    document.body.classList.toggle("forge-preset-krea", value === "krea");
    document.body.classList.toggle("forge-preset-qwen", value === "qwen");

    const presetInitialized = lastForgePresetValue === null && value !== null;
    const presetChanged = lastForgePresetValue !== null &&
        value !== null &&
        value !== lastForgePresetValue;
    lastForgePresetValue = value;

    if (presetInitialized || presetChanged) {
        if (value === "qwen") {
            ["txt2img", "img2img"].forEach((tabName) => {
                const nav = gradioApp().querySelector(`#${tabName}_extra_tabs > .tab-nav`);
                const unsupportedActive = nav?.querySelector(
                    'button.selected[id$="_lora-button"], button.selected[id$="_textual_inversion-button"], button.selected[id$="_hypernetworks-button"]'
                );
                if (unsupportedActive) nav.querySelector('button')?.click();
            });
        }
        clearTimeout(forgePresetRefreshTimer);
        forgePresetRefreshTimer = setTimeout(() => {
            if (typeof window.clickLoraRefresh === "function") {
                window.clickLoraRefresh();
            }

        }, 50);
    }
}

function scheduleForgePresetClassUpdate() {
    clearTimeout(forgePresetApplyTimer);
    forgePresetApplyTimer = setTimeout(applyForgePresetClass, 0);
}

onUiLoaded(function() {
    applyForgePresetClass();

    const presetRoot = gradioApp().querySelector("#forge_ui_preset");
    if (!presetRoot) return;

    presetRoot.addEventListener("input", scheduleForgePresetClassUpdate, true);
    presetRoot.addEventListener("change", scheduleForgePresetClassUpdate, true);

    const observer = new MutationObserver(scheduleForgePresetClassUpdate);
    observer.observe(presetRoot, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ["aria-checked", "checked", "class", "value"],
    });
});
