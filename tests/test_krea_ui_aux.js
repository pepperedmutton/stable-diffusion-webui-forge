const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const repoRoot = path.resolve(__dirname, '..');

function createScriptContext(additions = {}) {
    const context = {
        Array,
        console,
        clearTimeout,
        document: {
            body: {classList: {toggle() {}}},
            createElement() {
                return {};
            },
        },
        executeCallbacks() {},
        gradioApp() {
            return {querySelectorAll() { return []; }};
        },
        MutationObserver: class {
            observe() {}
        },
        onUiLoaded() {},
        opts: {lora_filter_disabled: false},
        setTimeout,
        window: {addEventListener() {}},
        ...additions,
    };
    vm.createContext(context);
    return context;
}

function runScript(context, relativePath) {
    const source = fs.readFileSync(path.join(repoRoot, relativePath), 'utf8');
    vm.runInContext(source, context, {filename: relativePath});
}

function loadScript(relativePath, additions = {}) {
    const context = createScriptContext(additions);
    runScript(context, relativePath);
    return context;
}

function makeCard(sdversion, name = 'ordinary_lora') {
    const classes = new Set();
    return {
        classList: {
            add(className) {
                classes.add(className);
            },
            contains(className) {
                return classes.has(className);
            },
            remove(className) {
                classes.delete(className);
            },
        },
        getAttribute(attribute) {
            if (attribute === 'data-sort-sdversion') return sdversion;
            if (attribute === 'data-name') return name;
            return null;
        },
        querySelectorAll() {
            return [
                {textContent: name},
                {textContent: ''},
                {textContent: name},
            ];
        },
    };
}

function testExtraNetworkFamilyFilter() {
    const context = loadScript('javascript/extraNetworks.js');
    const matches = context.extraNetworkCardMatchesPreset;

    assert.equal(matches(makeCard('SdVersion.SDXL'), 'krea', false), false);
    assert.equal(matches(makeCard('SdVersion.Flux'), 'krea', false), false);
    assert.equal(matches(makeCard('SdVersion.Krea'), 'krea', false), true);
    assert.equal(matches(makeCard('SdVersion.Unknown'), 'krea', false), false);
    assert.equal(matches(makeCard('SdVersion.Unknown', 'portrait_krea2_v1'), 'krea', false), true);
    assert.equal(matches(makeCard(null, 'CocoaMixZero-detail'), 'krea', false), true);
    assert.equal(matches(makeCard('SdVersion.SDXL', 'portrait_krea2_v1'), 'krea', false), false);
    assert.equal(matches(makeCard('SdVersion.SDXL'), 'xl', false), true);
    assert.equal(matches(makeCard('SdVersion.Flux'), 'xl', false), false);
    assert.equal(matches(makeCard('SdVersion.Flux'), 'krea', true), true);
}

function testPresetClassAndAutomaticRefresh() {
    const listeners = {};
    const selected = {value: 'xl'};
    const classState = new Map();
    let loadedCallback = null;
    let observerCallback = null;
    let refreshCount = 0;
    const tokenButtonClicks = {txt2img_token_button: 0, img2img_token_button: 0};

    const presetRoot = {
        addEventListener(name, callback) {
            listeners[name] = callback;
        },
        querySelector() {
            return selected;
        },
        querySelectorAll() {
            return [];
        },
    };
    const app = {
        getElementById(id) {
            if (!(id in tokenButtonClicks)) return null;
            return {
                click() {
                    tokenButtonClicks[id] += 1;
                },
            };
        },
        querySelector(selector) {
            return selector === '#forge_ui_preset' ? presetRoot : null;
        },
    };
    class TestMutationObserver {
        constructor(callback) {
            observerCallback = callback;
        }
        observe() {}
    }

    const context = loadScript('javascript/forgePreset.js', {
        clearTimeout() {},
        document: {
            body: {
                classList: {
                    toggle(name, enabled) {
                        classState.set(name, enabled);
                    },
                },
            },
        },
        gradioApp() {
            return app;
        },
        MutationObserver: TestMutationObserver,
        onUiLoaded(callback) {
            loadedCallback = callback;
        },
        setTimeout(callback) {
            callback();
            return 1;
        },
        window: {
            clickLoraRefresh() {
                refreshCount += 1;
            },
        },
    });

    loadedCallback();
    assert.equal(classState.get('forge-preset-krea'), false);
    assert.equal(refreshCount, 1);
    assert.deepEqual(tokenButtonClicks, {txt2img_token_button: 0, img2img_token_button: 0});
    // The UI observer must not race the backend's model-family update.
    context.window.refreshForgePresetTokenCounters();
    assert.deepEqual(tokenButtonClicks, {txt2img_token_button: 1, img2img_token_button: 1});

    selected.value = 'krea';
    observerCallback();
    assert.equal(classState.get('forge-preset-krea'), true);
    assert.equal(refreshCount, 2);
    assert.deepEqual(tokenButtonClicks, {txt2img_token_button: 1, img2img_token_button: 1});
    context.window.refreshForgePresetTokenCounters();
    assert.deepEqual(tokenButtonClicks, {txt2img_token_button: 2, img2img_token_button: 2});

    observerCallback();
    assert.equal(refreshCount, 2);
    assert.deepEqual(tokenButtonClicks, {txt2img_token_button: 2, img2img_token_button: 2});

    selected.value = 'xl';
    listeners.change();
    assert.equal(classState.get('forge-preset-krea'), false);
    assert.equal(refreshCount, 3);
    assert.deepEqual(tokenButtonClicks, {txt2img_token_button: 2, img2img_token_button: 2});
    context.window.refreshForgePresetTokenCounters();
    assert.deepEqual(tokenButtonClicks, {txt2img_token_button: 3, img2img_token_button: 3});
    assert.equal(typeof context.applyForgePresetClass, 'function');
    assert.equal(context.window.getForgePresetValue, context.getForgePresetValue);

    // Gradio can replace the startup radio value after reload without user input.
    selected.value = 'qwen';
    context.window.refreshForgePresetTokenCounters();
    assert.equal(classState.get('forge-preset-qwen'), true);
    assert.equal(classState.get('forge-preset-krea'), false);
    assert.deepEqual(tokenButtonClicks, {txt2img_token_button: 4, img2img_token_button: 4});
}

function testWindowExportsAndFirstLazyLoraOpen() {
    const selected = {value: 'krea'};
    const presetRoot = {
        querySelector() {
            return selected;
        },
        querySelectorAll() {
            return [];
        },
    };
    const activeLoraTab = {
        getAttribute(attribute) {
            return attribute === 'aria-selected' ? 'true' : null;
        },
    };
    const app = {
        getElementById(id) {
            return id === 'txt2img_lora-button' ? activeLoraTab : null;
        },
        querySelector(selector) {
            return selector === '#forge_ui_preset' ? presetRoot : null;
        },
        querySelectorAll() {
            return [];
        },
    };
    const context = createScriptContext({
        gradioApp() {
            return app;
        },
        setTimeout(callback) {
            callback();
            return 1;
        },
    });

    runScript(context, 'javascript/forgePreset.js');
    runScript(context, 'javascript/extraNetworks.js');

    assert.equal(context.window.getForgePresetValue, context.getForgePresetValue);
    assert.equal(context.window.clickLoraRefresh, context.clickLoraRefresh);
    assert.equal(context.window.getForgePresetValue(), 'krea');

    // The first refresh can occur before Gradio inserts the lazy Lora cards.
    context.window.clickLoraRefresh();

    const cards = [
        makeCard('SdVersion.SDXL', 'csky_sora_nova_il_r32_a16'),
        makeCard('SdVersion.SDXL', 'galkan'),
        makeCard('SdVersion.SDXL', 'shadow_beast'),
        makeCard('SdVersion.Krea', 'krea2_detail'),
    ];
    context.extraNetworksApplyFilter.txt2img_lora = function() {
        const preset = context.window.getForgePresetValue();
        cards.forEach((card) => {
            const visible = context.extraNetworkCardMatchesPreset(card, preset, false);
            card.classList[visible ? 'remove' : 'add']('hidden');
        });
    };

    // Opening the lazy page applies the current preset without another preset change.
    context.applyExtraNetworkFilter('txt2img_lora');

    assert.deepEqual(cards.map((card) => card.classList.contains('hidden')), [true, true, true, false]);
}

testExtraNetworkFamilyFilter();
testPresetClassAndAutomaticRefresh();
testWindowExportsAndFirstLazyLoraOpen();
console.log('Krea UI auxiliary tests passed.');
