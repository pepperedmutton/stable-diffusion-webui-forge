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

function applyForgePresetClass() {
    const value = getForgePresetValue();
    document.body.classList.toggle("forge-preset-lumina", value === "lumina");
}

onUiLoaded(function() {
    applyForgePresetClass();

    const observer = new MutationObserver(() => applyForgePresetClass());
    observer.observe(gradioApp(), {childList: true, subtree: true, attributes: true});
});
