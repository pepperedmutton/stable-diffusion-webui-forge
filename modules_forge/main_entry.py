import os
import torch
import gradio as gr

from gradio.context import Context
from modules import shared, paths
from backend import memory_management, stream
from backend.args import dynamic_args
from modules.shared import cmd_opts
from modules import shared_options as shared_options_module


if shared.options_templates is None:
    shared.options_templates = shared_options_module.options_templates


total_vram = int(memory_management.total_vram)

ui_forge_preset: gr.Radio = None

ui_checkpoint: gr.Dropdown = None
ui_vae: gr.Dropdown = None
ui_clip_skip: gr.Slider = None

ui_forge_unet_storage_dtype_options: gr.Radio = None
ui_forge_async_loading: gr.Radio = None
ui_forge_pin_shared_memory: gr.Radio = None
ui_forge_inference_memory: gr.Slider = None



forge_unet_storage_dtype_options = {
    'Automatic': (None, False),
    'Automatic (fp16 LoRA)': (None, True),
    'bnb-nf4': ('nf4', False),
    'bnb-nf4 (fp16 LoRA)': ('nf4', True),
    'float8-e4m3fn': (torch.float8_e4m3fn, False),
    'float8-e4m3fn (fp16 LoRA)': (torch.float8_e4m3fn, True),
    'bnb-fp4': ('fp4', False),
    'bnb-fp4 (fp16 LoRA)': ('fp4', True),
    'float8-e5m2': (torch.float8_e5m2, False),
    'float8-e5m2 (fp16 LoRA)': (torch.float8_e5m2, True),
}

module_list = {}
QWEN_DEFAULT_CHECKPOINT = "qwen_image_fp8_e4m3fn.safetensors"
QWEN_DEFAULT_MODULES = [
    os.path.abspath(os.path.join(paths.models_path, "text_encoder", "qwen_2.5_vl_7b_fp8_scaled.safetensors")),
    os.path.abspath(os.path.join(paths.models_path, "VAE", "qwen_image_vae.safetensors")),
]
XL_DEFAULT_CHECKPOINT = "novaAnimeXL_ilV170.safetensors"
XL_DEFAULT_MODULES = [
    os.path.abspath(os.path.join(paths.models_path, "VAE", "pppanimixVAE_il.safetensors")),
]
XL_DEFAULT_SAMPLER = "Euler a"
ANIMA_DEFAULT_CHECKPOINT = "anima-base-v1.0.safetensors"
ANIMA_DEFAULT_MODULES = [
    os.path.abspath(os.path.join(paths.models_path, "text_encoder", "anima_text_encoder.safetensors")),
    os.path.abspath(os.path.join(paths.models_path, "VAE", "anima_vae.safetensors")),
]
ANIMA_STABLE_LIVE_PREVIEW_OPTIONS = {
    "live_previews_enable": True,
    "show_progress_grid": False,
    "show_progress_type": "Approx NN",
    "show_progress_every_n_steps": 1,
}
_anima_live_preview_backup = None


def get_default_qwen_modules():
    return [m for m in QWEN_DEFAULT_MODULES if os.path.exists(m)]


def get_default_xl_modules():
    return [m for m in XL_DEFAULT_MODULES if os.path.exists(m)]


def get_default_anima_modules():
    return [m for m in ANIMA_DEFAULT_MODULES if os.path.exists(m)]


def sync_anima_live_preview_options(previous_preset, new_preset):
    global _anima_live_preview_backup

    previous_preset = normalize_forge_preset(previous_preset)
    new_preset = normalize_forge_preset(new_preset)

    if new_preset == "anima":
        if _anima_live_preview_backup is None:
            _anima_live_preview_backup = {
                key: shared.opts.data.get(key)
                for key in ANIMA_STABLE_LIVE_PREVIEW_OPTIONS.keys()
            }

        for key, value in ANIMA_STABLE_LIVE_PREVIEW_OPTIONS.items():
            if shared.opts.data.get(key) != value:
                shared.opts.set(key, value)
        return

    if previous_preset == "anima" and _anima_live_preview_backup is not None:
        for key, value in _anima_live_preview_backup.items():
            if shared.opts.data.get(key) != value:
                shared.opts.set(key, value)
        _anima_live_preview_backup = None


def resolve_existing_modules(modules):
    if not modules:
        return []
    result = []
    for module_path in modules:
        if module_path and os.path.exists(module_path):
            result.append(os.path.abspath(module_path))
    return sorted(list(dict.fromkeys(result)))


def get_preset_additional_modules(preset, checkpoint_name=""):
    preset = normalize_forge_preset(preset)
    checkpoint_profile = infer_module_profile_from_checkpoint(checkpoint_name)

    # Checkpoint-driven behavior:
    # - if checkpoint is Anima/Qwen, force matching required modules
    # - if checkpoint is another family, unload extra modules
    # - if checkpoint is empty (startup edge), fall back to preset defaults
    if checkpoint_profile is not None:
        effective_profile = checkpoint_profile
    else:
        checkpoint_text = str(checkpoint_name).strip().lower() if checkpoint_name is not None else ""
        effective_profile = preset if checkpoint_text in {"", "none"} and preset in {"qwen", "anima"} else None

    if effective_profile == 'qwen':
        saved = shared.opts.data.get("forge_additional_modules_qwen", []) or get_default_qwen_modules()
        modules = resolve_existing_modules(saved)
        return modules if modules else get_default_qwen_modules()

    if effective_profile == 'anima':
        saved = shared.opts.data.get("forge_additional_modules_anima", []) or get_default_anima_modules()
        modules = resolve_existing_modules(saved)
        return modules if modules else get_default_anima_modules()

    if preset == 'xl':
        return get_default_xl_modules()

    if preset == 'lumina' or is_lumina_checkpoint_name(checkpoint_name):
        return []

    return []


def normalize_forge_preset(preset):
    if preset in ['sd', 'all']:
        return 'qwen'
    if preset in ['qwen', 'lumina', 'xl', 'anima']:
        return preset
    return 'qwen'


def is_lumina_checkpoint_name(value):
    if value is None:
        return False

    text = str(value).lower()
    return "lumina" in text or "neta" in text


def normalize_checkpoint_text(value):
    return ''.join(ch for ch in str(value).lower() if ch.isalnum())


def is_qwen_checkpoint_name(value):
    text = normalize_checkpoint_text(value)
    return "qwen" in text


def is_anima_checkpoint_name(value):
    text = normalize_checkpoint_text(value)
    return "anima" in text


def infer_module_profile_from_checkpoint(value):
    if value is None:
        return None

    if is_qwen_checkpoint_name(value):
        return "qwen"

    if is_anima_checkpoint_name(value):
        return "anima"

    return None


def is_netayume_checkpoint_name(value):
    text = normalize_checkpoint_text(value)
    return "netayume" in text or "netalumina" in text


def is_novaanime_checkpoint_name(value):
    text = normalize_checkpoint_text(value)
    return "novaanime" in text


def is_xl_checkpoint_name(value):
    text = normalize_checkpoint_text(value)
    if not text or is_qwen_checkpoint_name(value) or is_anima_checkpoint_name(value) or is_lumina_checkpoint_name(value):
        return False

    return any(keyword in text for keyword in [
        "xl",
        "sdxl",
        "illustrious",
        "noobai",
        "janku",
        "novaanime",
        "wai",
    ])


def find_checkpoint_name_by_keywords(*keywords):
    from modules import sd_models

    if len(sd_models.checkpoints_list) == 0:
        sd_models.list_models()

    normalized_keywords = [normalize_checkpoint_text(x) for x in keywords if x]
    candidates = []

    for checkpoint_info in sd_models.checkpoints_list.values():
        haystack = " ".join([
            getattr(checkpoint_info, "name", ""),
            getattr(checkpoint_info, "title", ""),
            getattr(checkpoint_info, "filename", ""),
        ])
        haystack = normalize_checkpoint_text(haystack)

        if all(keyword in haystack for keyword in normalized_keywords):
            candidates.append(checkpoint_info)

    if not candidates:
        return None

    best = sorted(candidates, key=lambda x: len(getattr(x, "name", "")))[0]
    return getattr(best, "name", None)


def get_auto_checkpoint_for_preset_transition(previous_preset, new_preset, current_checkpoint_name):
    if previous_preset == 'lumina' and new_preset == 'xl' and is_netayume_checkpoint_name(current_checkpoint_name):
        return (
            find_checkpoint_name_by_keywords("novaanime", "xl")
            or find_checkpoint_name_by_keywords("novaanime")
        )

    if previous_preset == 'xl' and new_preset == 'lumina' and is_novaanime_checkpoint_name(current_checkpoint_name):
        return (
            find_checkpoint_name_by_keywords("netayume", "lumina")
            or find_checkpoint_name_by_keywords("netayume")
        )

    return None


def bind_to_opts(comp, k, save=False, callback=None):
    def on_change(v):
        shared.opts.set(k, v)
        if save:
            shared.opts.save(shared.config_filename)
        if callback is not None:
            callback()
        return

    comp.change(on_change, inputs=[comp], queue=False, show_progress=False)
    return


def make_checkpoint_manager_ui():
    from modules import shared_items, sd_models, ui_common

    global ui_checkpoint, ui_vae, ui_clip_skip, ui_forge_unet_storage_dtype_options, ui_forge_async_loading, ui_forge_pin_shared_memory, ui_forge_inference_memory, ui_forge_preset

    if shared.opts.sd_model_checkpoint in [None, 'None', 'none', '']:
        if len(sd_models.checkpoints_list) == 0:
            sd_models.list_models()
        if len(sd_models.checkpoints_list) > 0:
            shared.opts.set('sd_model_checkpoint', next(iter(sd_models.checkpoints_list.values())).name)

    migrated_preset = normalize_forge_preset(shared.opts.forge_preset)
    if migrated_preset != shared.opts.forge_preset:
        shared.opts.set('forge_preset', migrated_preset)
        shared.opts.save(shared.config_filename)

    ui_forge_preset = gr.Radio(label="UI", value=lambda: normalize_forge_preset(shared.opts.forge_preset), choices=['qwen', 'lumina', 'xl', 'anima'], elem_id="forge_ui_preset")

    ckpt_list, vae_list = refresh_models()

    ui_checkpoint = gr.Dropdown(
        value=lambda: shared.opts.sd_model_checkpoint,
        label="Checkpoint",
        elem_classes=['model_selection'],
        choices=ckpt_list
    )

    ui_vae = gr.Dropdown(
        value=lambda: [os.path.basename(x) for x in shared.opts.forge_additional_modules],
        multiselect=True,
        label="VAE / Text Encoder",
        render=False,
        choices=vae_list
    )

    def gr_refresh_models():
        a, b = refresh_models()
        return gr.update(choices=a), gr.update(choices=b)

    refresh_button = ui_common.ToolButton(value=ui_common.refresh_symbol, elem_id=f"forge_refresh_checkpoint", tooltip="Refresh")
    refresh_button.click(
        fn=gr_refresh_models,
        inputs=[],
        outputs=[ui_checkpoint, ui_vae],
        show_progress=False,
        queue=False
    )
    Context.root_block.load(
        fn=gr_refresh_models,
        inputs=[],
        outputs=[ui_checkpoint, ui_vae],
        show_progress=False,
        queue=False
    )

    ui_vae.render()

    ui_forge_unet_storage_dtype_options = gr.Dropdown(label="Diffusion in Low Bits", value=lambda: shared.opts.forge_unet_storage_dtype, choices=list(forge_unet_storage_dtype_options.keys()))
    bind_to_opts(ui_forge_unet_storage_dtype_options, 'forge_unet_storage_dtype', save=True, callback=refresh_model_loading_parameters)

    ui_forge_async_loading = gr.Radio(label="Swap Method", value=lambda: shared.opts.forge_async_loading, choices=['Queue', 'Async'])
    ui_forge_pin_shared_memory = gr.Radio(label="Swap Location", value=lambda: shared.opts.forge_pin_shared_memory, choices=['CPU', 'Shared'])
    ui_forge_inference_memory = gr.Slider(label="GPU Weights (MB)", value=lambda: total_vram - shared.opts.forge_inference_memory, minimum=0, maximum=int(memory_management.total_vram), step=1)

    mem_comps = [ui_forge_inference_memory, ui_forge_async_loading, ui_forge_pin_shared_memory]

    ui_forge_inference_memory.change(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)
    ui_forge_async_loading.change(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)
    ui_forge_pin_shared_memory.change(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)

    Context.root_block.load(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)

    ui_clip_skip = gr.Slider(label="Clip skip", value=lambda: shared.opts.CLIP_stop_at_last_layers, **{"minimum": 1, "maximum": 12, "step": 1})
    bind_to_opts(ui_clip_skip, 'CLIP_stop_at_last_layers', save=True)

    ui_checkpoint.change(checkpoint_change, inputs=[ui_checkpoint, ui_forge_preset], show_progress=False)
    ui_vae.change(modules_change, inputs=[ui_vae, ui_forge_preset], queue=False, show_progress=False)

    return


def find_files_with_extensions(base_path, extensions):
    found_files = {}
    for root, _, files in os.walk(base_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                full_path = os.path.join(root, file)
                found_files[file] = full_path
    return found_files


def refresh_models():
    from modules import shared_items

    global module_list

    shared_items.refresh_checkpoints()
    ckpt_list = shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)

    file_extensions = ['ckpt', 'pt', 'bin', 'safetensors', 'gguf']

    module_list.clear()
    
    module_paths = [
        os.path.abspath(os.path.join(paths.models_path, "VAE")),
        os.path.abspath(os.path.join(paths.models_path, "text_encoder")),
    ]

    if isinstance(shared.cmd_opts.vae_dir, str):
        module_paths.append(os.path.abspath(shared.cmd_opts.vae_dir))
    if isinstance(shared.cmd_opts.text_encoder_dir, str):
        module_paths.append(os.path.abspath(shared.cmd_opts.text_encoder_dir))

    for vae_path in module_paths:
        vae_files = find_files_with_extensions(vae_path, file_extensions)
        module_list.update(vae_files)

    return ckpt_list, module_list.keys()


def ui_refresh_memory_management_settings(model_memory, async_loading, pin_shared_memory):
    """ Passes precalculated 'model_memory' from "GPU Weights" UI slider (skip redundant calculation) """
    refresh_memory_management_settings(
        async_loading=async_loading,
        pin_shared_memory=pin_shared_memory,
        model_memory=model_memory  # Use model_memory directly from UI slider value
    )

def refresh_memory_management_settings(async_loading=None, inference_memory=None, pin_shared_memory=None, model_memory=None):
    from modules import processing

    # Fallback to defaults if values are not passed
    async_loading = async_loading if async_loading is not None else shared.opts.forge_async_loading
    inference_memory = inference_memory if inference_memory is not None else shared.opts.forge_inference_memory
    pin_shared_memory = pin_shared_memory if pin_shared_memory is not None else shared.opts.forge_pin_shared_memory

    # If model_memory is provided, calculate inference memory accordingly, otherwise use inference_memory directly
    if model_memory is None:
        model_memory = total_vram - inference_memory
    else:
        inference_memory = total_vram - model_memory

    shared.opts.set('forge_async_loading', async_loading)
    shared.opts.set('forge_inference_memory', inference_memory)
    shared.opts.set('forge_pin_shared_memory', pin_shared_memory)

    stream.stream_activated = async_loading == 'Async'
    memory_management.current_inference_memory = inference_memory * 1024 * 1024  # Convert MB to bytes
    memory_management.PIN_SHARED_MEMORY = pin_shared_memory == 'Shared'

    log_dict = dict(
        stream=stream.should_use_stream(),
        inference_memory=memory_management.minimum_inference_memory() / (1024 * 1024),
        pin_shared_memory=memory_management.PIN_SHARED_MEMORY
    )

    print(f'Environment vars changed: {log_dict}')

    if inference_memory < min(512, total_vram * 0.05):
        print('------------------')
        print(f'[Low VRAM Warning] You just set Forge to use 100% GPU memory ({model_memory:.2f} MB) to load model weights.')
        print('[Low VRAM Warning] This means you will have 0% GPU memory (0.00 MB) to do matrix computation. Computations may fallback to CPU or go Out of Memory.')
        print('[Low VRAM Warning] In many cases, image generation will be 10x slower.')
        print("[Low VRAM Warning] To solve the problem, you can set the 'GPU Weights' (on the top of page) to a lower value.")
        print("[Low VRAM Warning] If you cannot find 'GPU Weights', use the UI preset selector in the left-top corner of the webpage.")
        print('[Low VRAM Warning] Make sure that you know what you are testing.')
        print('------------------')
    else:
        compute_percentage = (inference_memory / total_vram) * 100.0
        print(f'[GPU Setting] You will use {(100 - compute_percentage):.2f}% GPU memory ({model_memory:.2f} MB) to load weights, and use {compute_percentage:.2f}% GPU memory ({inference_memory:.2f} MB) to do matrix computation.')

    processing.need_global_unload = True
    return


def refresh_model_loading_parameters():
    from modules import processing
    from modules.sd_models import select_checkpoint, model_data

    checkpoint_info = select_checkpoint()
    checkpoint_name = getattr(checkpoint_info, "name", None) or getattr(checkpoint_info, "filename", "")
    preset = normalize_forge_preset(shared.opts.forge_preset)

    unet_storage_dtype, lora_fp16 = forge_unet_storage_dtype_options.get(shared.opts.forge_unet_storage_dtype, (None, False))

    dynamic_args['online_lora'] = lora_fp16

    additional_modules = get_preset_additional_modules(preset, checkpoint_name)
    if additional_modules != sorted(shared.opts.data.get('forge_additional_modules', [])):
        shared.opts.set('forge_additional_modules', additional_modules)

    model_data.forge_loading_parameters = dict(
        checkpoint_info=checkpoint_info,
        additional_modules=additional_modules,
        unet_storage_dtype=unet_storage_dtype
    )

    print(f'Model selected: {model_data.forge_loading_parameters}')
    print(f'Using online LoRAs in FP16: {lora_fp16}')
    processing.need_global_unload = True

    return


def checkpoint_change(ckpt_name: str, preset=None, save=True, refresh=True):
    from modules import sd_models

    """ checkpoint name can be a number of valid aliases. Returns True if checkpoint changed. """
    sampler_changed = False
    if normalize_forge_preset(preset) == 'xl':
        sampler_changed = sync_xl_sampler_defaults(save=False)

    new_ckpt_info = sd_models.get_closet_checkpoint_match(ckpt_name)
    current_ckpt_info = sd_models.get_closet_checkpoint_match(shared.opts.data.get('sd_model_checkpoint', ''))
    if new_ckpt_info == current_ckpt_info:
        if save and sampler_changed:
            shared.opts.save(shared.config_filename)
        return False

    shared.opts.set('sd_model_checkpoint', ckpt_name)
    if preset == 'qwen':
        shared.opts.set('forge_checkpoint_qwen', ckpt_name)
    if preset == 'anima':
        shared.opts.set('forge_checkpoint_anima', ckpt_name)

    if save:
        shared.opts.save(shared.config_filename)
    if refresh:
        refresh_model_loading_parameters()
    return True


def sync_xl_sampler_defaults(save=True):
    changed = False
    for key in ("xl_t2i_sampler", "xl_i2i_sampler"):
        if shared.opts.data.get(key) != XL_DEFAULT_SAMPLER:
            shared.opts.set(key, XL_DEFAULT_SAMPLER)
            changed = True

    if save and changed:
        shared.opts.save(shared.config_filename)

    return changed


def should_apply_xl_sampler_for_checkpoint(ckpt_name, preset=None):
    preset = normalize_forge_preset(preset)
    return preset == 'xl' or is_xl_checkpoint_name(ckpt_name)


def on_checkpoint_sampler_ui_sync(ckpt_name, preset=None):
    if not should_apply_xl_sampler_for_checkpoint(ckpt_name, preset):
        return [gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()]

    shared.opts.set('forge_preset', 'xl')
    shared.opts.set('xl_t2i_steps', 30)
    shared.opts.set('xl_i2i_steps', 30)
    sync_xl_sampler_defaults(save=False)
    modules_change(get_default_xl_modules(), preset='xl', save=False, refresh=False)
    shared.opts.save(shared.config_filename)
    refresh_model_loading_parameters()

    return [
        gr.update(value='xl'),
        gr.update(visible=True, value=[os.path.basename(x) for x in shared.opts.forge_additional_modules]),
        gr.update(value=30),
        gr.update(value=30),
        gr.update(value=XL_DEFAULT_SAMPLER),
        gr.update(value=XL_DEFAULT_SAMPLER),
    ]


def modules_change(module_values: list, preset=None, save=True, refresh=True) -> bool:
    """ module values may be provided as file paths, or just the module names. Returns True if modules changed. """
    modules = []
    for v in module_values:
        module_name = os.path.basename(v) # If the input is a filepath, extract the file name
        if module_name in module_list:
            modules.append(module_list[module_name])
    modules = sorted(modules)
    
    # skip further processing if value unchanged
    if modules == sorted(shared.opts.data.get('forge_additional_modules', [])):
        return False

    shared.opts.set('forge_additional_modules', modules)
    if preset == 'qwen':
        shared.opts.set('forge_additional_modules_qwen', modules)
    if preset == 'anima':
        shared.opts.set('forge_additional_modules_anima', modules)

    if save:
        shared.opts.save(shared.config_filename)
    if refresh:
        refresh_model_loading_parameters()
    return True


def get_a1111_ui_component(tab, label):
    from modules import infotext_utils

    fields = infotext_utils.paste_fields[tab]['fields']
    for f in fields:
        if f.label == label or f.api == label:
            return f.component


def forge_main_entry():
    ui_txt2img_steps = get_a1111_ui_component('txt2img', 'Steps')
    ui_img2img_steps = get_a1111_ui_component('img2img', 'Steps')
    ui_txt2img_width = get_a1111_ui_component('txt2img', 'Size-1')
    ui_txt2img_height = get_a1111_ui_component('txt2img', 'Size-2')
    ui_txt2img_cfg = get_a1111_ui_component('txt2img', 'CFG scale')
    ui_txt2img_distilled_cfg = get_a1111_ui_component('txt2img', 'Distilled CFG Scale')
    ui_txt2img_sampler = get_a1111_ui_component('txt2img', 'sampler_name')
    ui_txt2img_scheduler = get_a1111_ui_component('txt2img', 'scheduler')

    ui_img2img_width = get_a1111_ui_component('img2img', 'Size-1')
    ui_img2img_height = get_a1111_ui_component('img2img', 'Size-2')
    ui_img2img_cfg = get_a1111_ui_component('img2img', 'CFG scale')
    ui_img2img_distilled_cfg = get_a1111_ui_component('img2img', 'Distilled CFG Scale')
    ui_img2img_sampler = get_a1111_ui_component('img2img', 'sampler_name')
    ui_img2img_scheduler = get_a1111_ui_component('img2img', 'scheduler')

    ui_txt2img_hr_cfg = get_a1111_ui_component('txt2img', 'Hires CFG Scale')
    ui_txt2img_hr_distilled_cfg = get_a1111_ui_component('txt2img', 'Hires Distilled CFG Scale')

    output_targets = [
        ui_checkpoint,
        ui_vae,
        ui_clip_skip,
        ui_forge_unet_storage_dtype_options,
        ui_forge_async_loading,
        ui_forge_pin_shared_memory,
        ui_forge_inference_memory,
        ui_txt2img_steps,
        ui_img2img_steps,
        ui_txt2img_width,
        ui_img2img_width,
        ui_txt2img_height,
        ui_img2img_height,
        ui_txt2img_cfg,
        ui_img2img_cfg,
        ui_txt2img_distilled_cfg,
        ui_img2img_distilled_cfg,
        ui_txt2img_sampler,
        ui_img2img_sampler,
        ui_txt2img_scheduler,
        ui_img2img_scheduler,
        ui_txt2img_hr_cfg,
        ui_txt2img_hr_distilled_cfg,
    ]

    preset_inputs = [ui_forge_preset, ui_txt2img_width, ui_img2img_width, ui_txt2img_height, ui_img2img_height]
    ui_forge_preset.change(on_preset_change, inputs=preset_inputs, outputs=output_targets, queue=False, show_progress=False)
    ui_forge_preset.change(js="clickLoraRefresh", fn=None, queue=False, show_progress=False)
    ui_checkpoint.change(
        on_checkpoint_sampler_ui_sync,
        inputs=[ui_checkpoint, ui_forge_preset],
        outputs=[ui_forge_preset, ui_vae, ui_txt2img_steps, ui_img2img_steps, ui_txt2img_sampler, ui_img2img_sampler],
        queue=False,
        show_progress=False,
    )
    Context.root_block.load(on_preset_change, inputs=preset_inputs, outputs=output_targets, queue=False, show_progress=False)

    refresh_model_loading_parameters()
    return


def on_preset_change(
    preset=None,
    current_t2i_width=None,
    current_i2i_width=None,
    current_t2i_height=None,
    current_i2i_height=None,
):
    from modules import ui_loadsave, sd_models

    def opt(key, default):
        value = shared.opts.data.get(key, default)
        return default if value is None else value

    previous_preset = normalize_forge_preset(shared.opts.forge_preset)
    current_checkpoint_name = shared.opts.data.get('sd_model_checkpoint', '')
    preset = normalize_forge_preset(preset if preset is not None else shared.opts.forge_preset)
    checkpoint_update = gr.update()
    loadsave = ui_loadsave.UiLoadsave(cmd_opts.ui_config_file)
    ui_settings_from_file = loadsave.ui_settings.copy()
    ui_settings_from_file_get = ui_settings_from_file.get

    if current_t2i_width is None:
        current_t2i_width = ui_settings_from_file_get('txt2img/Width/value', 512)
    if current_i2i_width is None:
        current_i2i_width = ui_settings_from_file_get('img2img/Width/value', 512)
    if current_t2i_height is None:
        current_t2i_height = ui_settings_from_file_get('txt2img/Height/value', 512)
    if current_i2i_height is None:
        current_i2i_height = ui_settings_from_file_get('img2img/Height/value', 512)

    auto_checkpoint_name = get_auto_checkpoint_for_preset_transition(previous_preset, preset, current_checkpoint_name)
    if auto_checkpoint_name:
        checkpoint_change(auto_checkpoint_name, save=True, refresh=False)
        checkpoint_update = gr.update(value=auto_checkpoint_name)

    if shared.opts.forge_preset != preset:
        shared.opts.set('forge_preset', preset)
        shared.opts.save(shared.config_filename)

    if preset is not None:
        shared.opts.set('forge_preset', preset)

    sync_anima_live_preview_options(previous_preset, preset)

    if preset == 'qwen':
        qwen_checkpoint = opt("forge_checkpoint_qwen", QWEN_DEFAULT_CHECKPOINT)
        qwen_modules = get_preset_additional_modules('qwen')

        if sd_models.get_closet_checkpoint_match(qwen_checkpoint) is not None:
            checkpoint_change(qwen_checkpoint, preset='qwen', save=True, refresh=False)
            checkpoint_update = gr.update(value=qwen_checkpoint)

        modules_change(qwen_modules, preset='qwen', save=True, refresh=False)

        model_mem = opt("qwen_GPU_MB", total_vram - 1024)
        if model_mem < 0 or model_mem > total_vram:
            model_mem = total_vram - 1024

        refresh_model_loading_parameters()
        return [
            checkpoint_update,                                                         # ui_checkpoint
            gr.update(visible=True, value=[os.path.basename(x) for x in shared.opts.forge_additional_modules]),  # ui_vae
            gr.update(visible=False, value=1),                                         # ui_clip_skip
            gr.update(visible=True, value='Automatic'),                                # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                    # ui_forge_async_loading
            gr.update(visible=True, value='CPU'),                                      # ui_forge_pin_shared_memory
            gr.update(visible=True, value=model_mem),                                  # ui_forge_inference_memory
            gr.update(value=opt("qwen_t2i_steps", 8)),                                 # ui_txt2img_steps
            gr.update(value=opt("qwen_i2i_steps", 8)),                                 # ui_img2img_steps
            gr.update(value=current_t2i_width),                                          # ui_txt2img_width
            gr.update(value=current_i2i_width),                                          # ui_img2img_width
            gr.update(value=current_t2i_height),                                         # ui_txt2img_height
            gr.update(value=current_i2i_height),                                         # ui_img2img_height
            gr.update(value=opt("qwen_t2i_cfg", 1.0)),                                 # ui_txt2img_cfg
            gr.update(value=opt("qwen_i2i_cfg", 1.0)),                                 # ui_img2img_cfg
            gr.update(visible=False, value=3.5),                                       # ui_txt2img_distilled_cfg
            gr.update(visible=False, value=3.5),                                       # ui_img2img_distilled_cfg
            gr.update(value=opt("qwen_t2i_sampler", 'LCM')),                           # ui_txt2img_sampler
            gr.update(value=opt("qwen_i2i_sampler", 'LCM')),                           # ui_img2img_sampler
            gr.update(value=opt("qwen_t2i_scheduler", 'Normal')),                      # ui_txt2img_scheduler
            gr.update(value=opt("qwen_i2i_scheduler", 'Normal')),                      # ui_img2img_scheduler
            gr.update(visible=False, value=opt("qwen_t2i_hr_cfg", 1.0)),               # ui_txt2img_hr_cfg
            gr.update(visible=False, value=3.5),                                       # ui_txt2img_hr_distilled_cfg
        ]

    if preset == 'anima':
        anima_checkpoint = opt("forge_checkpoint_anima", ANIMA_DEFAULT_CHECKPOINT)
        anima_modules = get_preset_additional_modules('anima')

        if sd_models.get_closet_checkpoint_match(anima_checkpoint) is not None:
            checkpoint_change(anima_checkpoint, preset='anima', save=True, refresh=False)
            checkpoint_update = gr.update(value=anima_checkpoint)

        modules_change(anima_modules, preset='anima', save=True, refresh=False)

        model_mem = opt("anima_GPU_MB", total_vram - 1024)
        if model_mem < 0 or model_mem > total_vram:
            model_mem = total_vram - 1024

        refresh_model_loading_parameters()
        return [
            checkpoint_update,                                                         # ui_checkpoint
            gr.update(visible=False, value=[os.path.basename(x) for x in shared.opts.forge_additional_modules]),  # ui_vae
            gr.update(visible=False, value=1),                                         # ui_clip_skip
            gr.update(visible=True, value='Automatic'),                                # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                    # ui_forge_async_loading
            gr.update(visible=True, value='CPU'),                                      # ui_forge_pin_shared_memory
            gr.update(visible=True, value=model_mem),                                  # ui_forge_inference_memory
            gr.update(value=30),                                                        # ui_txt2img_steps
            gr.update(value=30),                                                        # ui_img2img_steps
            gr.update(value=current_t2i_width),                                          # ui_txt2img_width
            gr.update(value=current_i2i_width),                                          # ui_img2img_width
            gr.update(value=current_t2i_height),                                         # ui_txt2img_height
            gr.update(value=current_i2i_height),                                         # ui_img2img_height
            gr.update(value=4.0),                                                       # ui_txt2img_cfg
            gr.update(value=4.0),                                                       # ui_img2img_cfg
            gr.update(visible=True, value=3.5),                                         # ui_txt2img_distilled_cfg
            gr.update(visible=True, value=3.5),                                         # ui_img2img_distilled_cfg
            gr.update(value='ER-SDE'),                                                  # ui_txt2img_sampler
            gr.update(value='ER-SDE'),                                                  # ui_img2img_sampler
            gr.update(value='Simple'),                                                  # ui_txt2img_scheduler
            gr.update(value='Simple'),                                                  # ui_img2img_scheduler
            gr.update(visible=True, value=4.0),                                         # ui_txt2img_hr_cfg
            gr.update(visible=True, value=3.5),                                         # ui_txt2img_hr_distilled_cfg
        ]

    if preset == 'lumina':
        model_mem = opt("lumina_GPU_MB", total_vram - 1024)
        if model_mem < 0 or model_mem > total_vram:
            model_mem = total_vram - 1024
        modules_change([], preset=preset, save=False, refresh=False)
        refresh_model_loading_parameters()
        return [
            checkpoint_update,                                                         # ui_checkpoint
            gr.update(visible=False),                                                   # ui_vae
            gr.update(visible=False, value=1),                                          # ui_clip_skip
            gr.update(visible=True, value='Automatic'),                                 # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                     # ui_forge_async_loading
            gr.update(visible=True, value='CPU'),                                       # ui_forge_pin_shared_memory
            gr.update(visible=True, value=model_mem),                                   # ui_forge_inference_memory
            gr.update(value=opt("lumina_t2i_steps", 25)),                               # ui_txt2img_steps
            gr.update(value=opt("lumina_i2i_steps", 25)),                               # ui_img2img_steps
            gr.update(value=current_t2i_width),                                          # ui_txt2img_width
            gr.update(value=current_i2i_width),                                          # ui_img2img_width
            gr.update(value=current_t2i_height),                                         # ui_txt2img_height
            gr.update(value=current_i2i_height),                                         # ui_img2img_height
            gr.update(value=opt("lumina_t2i_cfg", 3.5)),                                # ui_txt2img_cfg
            gr.update(value=opt("lumina_i2i_cfg", 3.5)),                                # ui_img2img_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_distilled_cfg
            gr.update(visible=False, value=3.5),                                        # ui_img2img_distilled_cfg
            gr.update(value=opt("lumina_t2i_sampler", 'RES Multistep')),                # ui_txt2img_sampler
            gr.update(value=opt("lumina_i2i_sampler", 'RES Multistep')),                # ui_img2img_sampler
            gr.update(value=opt("lumina_t2i_scheduler", 'Simple')),                     # ui_txt2img_scheduler
            gr.update(value=opt("lumina_i2i_scheduler", 'Simple')),                     # ui_img2img_scheduler
            gr.update(visible=False, value=opt("lumina_t2i_hr_cfg", 3.5)),              # ui_txt2img_hr_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_hr_distilled_cfg
        ]

    if preset == 'xl':
        xl_checkpoint = XL_DEFAULT_CHECKPOINT
        xl_modules = get_default_xl_modules()
        sync_xl_sampler_defaults(save=True)

        if sd_models.get_closet_checkpoint_match(xl_checkpoint) is not None:
            checkpoint_change(xl_checkpoint, save=True, refresh=False)
            checkpoint_update = gr.update(value=xl_checkpoint)

        model_mem = getattr(shared.opts, "xl_GPU_MB", total_vram - 1024)
        if model_mem < 0 or model_mem > total_vram:
            model_mem = total_vram - 1024
        modules_change(xl_modules, save=True, refresh=False)
        refresh_model_loading_parameters()
        return [
            checkpoint_update,                                                         # ui_checkpoint
            gr.update(visible=True, value=[os.path.basename(x) for x in shared.opts.forge_additional_modules]),  # ui_vae
            gr.update(visible=False, value=1),                                          # ui_clip_skip
            gr.update(visible=True, value='Automatic'),                                 # ui_forge_unet_storage_dtype_options
            gr.update(visible=False, value='Queue'),                                    # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),                                      # ui_forge_pin_shared_memory
            gr.update(visible=True, value=model_mem),                                   # ui_forge_inference_memory
            gr.update(value=opt("xl_t2i_steps", 30)),                                # ui_txt2img_steps
            gr.update(value=opt("xl_i2i_steps", 30)),                                # ui_img2img_steps
            gr.update(value=current_t2i_width),                                          # ui_txt2img_width
            gr.update(value=current_i2i_width),                                          # ui_img2img_width
            gr.update(value=current_t2i_height),                                         # ui_txt2img_height
            gr.update(value=current_i2i_height),                                         # ui_img2img_height
            gr.update(value=getattr(shared.opts, "xl_t2i_cfg", 5)),                     # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "xl_i2i_cfg", 5)),                     # ui_img2img_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_distilled_cfg
            gr.update(visible=False, value=3.5),                                        # ui_img2img_distilled_cfg
            gr.update(value=XL_DEFAULT_SAMPLER),                                        # ui_txt2img_sampler
            gr.update(value=XL_DEFAULT_SAMPLER),                                        # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "xl_t2i_scheduler", 'Automatic')),     # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "xl_i2i_scheduler", 'Automatic')),     # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "xl_t2i_hr_cfg", 5.0)),  # ui_txt2img_hr_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_hr_distilled_cfg
        ]

    if preset == 'flux':
        model_mem = getattr(shared.opts, "flux_GPU_MB", total_vram - 1024)
        if model_mem < 0 or model_mem > total_vram:
            model_mem = total_vram - 1024
        modules_change([], preset=preset, save=False, refresh=False)
        refresh_model_loading_parameters()
        return [
            checkpoint_update,                                                         # ui_checkpoint
            gr.update(visible=False),                                                   # ui_vae
            gr.update(visible=False, value=1),                                          # ui_clip_skip
            gr.update(visible=True, value='Automatic'),                                 # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                     # ui_forge_async_loading
            gr.update(visible=True, value='CPU'),                                       # ui_forge_pin_shared_memory
            gr.update(visible=True, value=model_mem),                                   # ui_forge_inference_memory
            gr.update(value=ui_settings_from_file_get("customscript/sampler.py/txt2img/Sampling steps/value", 20)), # ui_txt2img_steps
            gr.update(value=ui_settings_from_file_get("customscript/sampler.py/img2img/Sampling steps/value", 20)), # ui_img2img_steps
            gr.update(value=current_t2i_width),                                          # ui_txt2img_width
            gr.update(value=current_i2i_width),                                          # ui_img2img_width
            gr.update(value=current_t2i_height),                                         # ui_txt2img_height
            gr.update(value=current_i2i_height),                                         # ui_img2img_height
            gr.update(value=getattr(shared.opts, "flux_t2i_cfg", 1)),                   # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "flux_i2i_cfg", 1)),                   # ui_img2img_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_d_cfg", 3.5)), # ui_txt2img_distilled_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_i2i_d_cfg", 3.5)), # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "flux_t2i_sampler", 'Euler')),         # ui_txt2img_sampler
            gr.update(value=getattr(shared.opts, "flux_i2i_sampler", 'Euler')),         # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "flux_t2i_scheduler", 'Simple')),      # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "flux_i2i_scheduler", 'Simple')),      # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_hr_cfg", 1.0)),    # ui_txt2img_hr_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_hr_d_cfg", 3.5)),  # ui_txt2img_hr_distilled_cfg
        ]

    modules_change([], preset=preset, save=False, refresh=False)
    refresh_model_loading_parameters()
    return [
        checkpoint_update,  # ui_checkpoint
        gr.update(visible=False),  # ui_vae
        gr.update(visible=True, value=1),  # ui_clip_skip
        gr.update(visible=True, value='Automatic'),  # ui_forge_unet_storage_dtype_options
        gr.update(visible=True, value='Queue'),  # ui_forge_async_loading
        gr.update(visible=True, value='CPU'),  # ui_forge_pin_shared_memory
        gr.update(visible=True, value=total_vram - 1024),  # ui_forge_inference_memory
        gr.update(value=ui_settings_from_file_get('customscript/sampler.py/txt2img/Sampling steps/value', 20)),  # ui_txt2img_steps
        gr.update(value=ui_settings_from_file_get('customscript/sampler.py/img2img/Sampling steps/value', 20)),  # ui_img2img_steps
        gr.update(value=current_t2i_width),  # ui_txt2img_width
        gr.update(value=current_i2i_width),  # ui_img2img_width
        gr.update(value=current_t2i_height),  # ui_txt2img_height
        gr.update(value=current_i2i_height),  # ui_img2img_height
        gr.update(value=ui_settings_from_file_get('txt2img/CFG Scale/value', 7.0)),  # ui_txt2img_cfg
        gr.update(value=ui_settings_from_file_get('img2img/CFG Scale/value', 7.0)),  # ui_img2img_cfg
        gr.update(visible=True, value=ui_settings_from_file_get('txt2img/Distilled CFG Scale/value', 3.5)),  # ui_txt2img_distilled_cfg
        gr.update(visible=True, value=ui_settings_from_file_get('img2img/Distilled CFG Scale/value', 3.5)),  # ui_img2img_distilled_cfg
        gr.update(value=ui_settings_from_file_get('customscript/sampler.py/txt2img/Sampling method/value', 'Euler')),  # ui_txt2img_sampler
        gr.update(value=ui_settings_from_file_get('customscript/sampler.py/img2img/Sampling method/value', 'Euler')),  # ui_img2img_sampler
        gr.update(value=ui_settings_from_file_get('customscript/sampler.py/txt2img/Schedule type/value', 'Automatic')),  # ui_txt2img_scheduler
        gr.update(value=ui_settings_from_file_get('customscript/sampler.py/img2img/Schedule type/value', 'Automatic')),  # ui_img2img_scheduler
        gr.update(visible=True, value=ui_settings_from_file_get('txt2img/Hires CFG Scale/value', 7.0)), # ui_txt2img_hr_cfg
        gr.update(visible=True, value=ui_settings_from_file_get('txt2img/Hires Distilled CFG Scale/value', 3.5)), # ui_txt2img_hr_distilled_cfg
    ]

shared.options_templates.update(shared.options_section(('ui_lumina', "UI defaults 'lumina'", "ui"), {
    "lumina_t2i_width":  shared.OptionInfo(1200, "txt2img width",      gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "lumina_t2i_height": shared.OptionInfo(1600, "txt2img height",     gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "lumina_t2i_cfg":    shared.OptionInfo(3.5,  "txt2img CFG",        gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "lumina_t2i_hr_cfg": shared.OptionInfo(3.5,  "txt2img HiRes CFG",  gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "lumina_i2i_width":  shared.OptionInfo(1200, "img2img width",      gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "lumina_i2i_height": shared.OptionInfo(1600, "img2img height",     gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "lumina_i2i_cfg":    shared.OptionInfo(3.5,  "img2img CFG",        gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "lumina_GPU_MB":     shared.OptionInfo(total_vram - 1024, "GPU Weights (MB)", gr.Slider, {"minimum": 0,  "maximum": total_vram,   "step": 1}),
}))
shared.options_templates.update(shared.options_section(('ui_xl', "UI defaults 'xl'", "ui"), {
    "xl_t2i_width":  shared.OptionInfo(896,  "txt2img width",      gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "xl_t2i_height": shared.OptionInfo(1152, "txt2img height",     gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "xl_t2i_cfg":    shared.OptionInfo(5,    "txt2img CFG",        gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "xl_t2i_hr_cfg": shared.OptionInfo(5,    "txt2img HiRes CFG",  gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "xl_i2i_width":  shared.OptionInfo(1024, "img2img width",      gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "xl_i2i_height": shared.OptionInfo(1024, "img2img height",     gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "xl_i2i_cfg":    shared.OptionInfo(5,    "img2img CFG",        gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "xl_GPU_MB":     shared.OptionInfo(total_vram - 1024, "GPU Weights (MB)", gr.Slider, {"minimum": 0,  "maximum": total_vram,   "step": 1}),
}))
shared.options_templates.update(shared.options_section(('ui_flux', "UI defaults 'flux'", "ui"), {
    "flux_t2i_width":    shared.OptionInfo(896,  "txt2img width",                gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "flux_t2i_height":   shared.OptionInfo(1152, "txt2img height",               gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "flux_t2i_cfg":      shared.OptionInfo(1,    "txt2img CFG",                  gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "flux_t2i_hr_cfg":   shared.OptionInfo(1,    "txt2img HiRes CFG",            gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "flux_t2i_d_cfg":    shared.OptionInfo(3.5,  "txt2img Distilled CFG",        gr.Slider, {"minimum": 0,  "maximum": 30,   "step": 0.1}),
    "flux_t2i_hr_d_cfg": shared.OptionInfo(3.5,  "txt2img Distilled HiRes CFG",  gr.Slider, {"minimum": 0,  "maximum": 30,   "step": 0.1}),
    "flux_i2i_width":    shared.OptionInfo(1024, "img2img width",                gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "flux_i2i_height":   shared.OptionInfo(1024, "img2img height",               gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "flux_i2i_cfg":      shared.OptionInfo(1,    "img2img CFG",                  gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "flux_i2i_d_cfg":    shared.OptionInfo(3.5,  "img2img Distilled CFG",        gr.Slider, {"minimum": 0,  "maximum": 30,   "step": 0.1}),
    "flux_GPU_MB":       shared.OptionInfo(total_vram - 1024, "GPU Weights (MB)",gr.Slider, {"minimum": 0,  "maximum": total_vram,   "step": 1}),
}))
shared.options_templates.update(shared.options_section(('ui_anima', "UI defaults 'anima'", "ui"), {
    "anima_t2i_width":    shared.OptionInfo(896,  "txt2img width",                gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "anima_t2i_height":   shared.OptionInfo(1152, "txt2img height",               gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "anima_t2i_cfg":      shared.OptionInfo(4.0,  "txt2img CFG",                  gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "anima_t2i_hr_cfg":   shared.OptionInfo(4.0,  "txt2img HiRes CFG",            gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "anima_t2i_d_cfg":    shared.OptionInfo(3.5,  "txt2img Distilled CFG",        gr.Slider, {"minimum": 0,  "maximum": 30,   "step": 0.1}),
    "anima_t2i_hr_d_cfg": shared.OptionInfo(3.5,  "txt2img Distilled HiRes CFG",  gr.Slider, {"minimum": 0,  "maximum": 30,   "step": 0.1}),
    "anima_i2i_width":    shared.OptionInfo(1024, "img2img width",                gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "anima_i2i_height":   shared.OptionInfo(1024, "img2img height",               gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "anima_i2i_cfg":      shared.OptionInfo(4.0,  "img2img CFG",                  gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "anima_i2i_d_cfg":    shared.OptionInfo(3.5,  "img2img Distilled CFG",        gr.Slider, {"minimum": 0,  "maximum": 30,   "step": 0.1}),
    "anima_GPU_MB":       shared.OptionInfo(total_vram - 1024, "GPU Weights (MB)",gr.Slider, {"minimum": 0,  "maximum": total_vram,   "step": 1}),
    "anima_t2i_steps":    shared.OptionInfo(30, "txt2img steps",                  gr.Slider, {"minimum": 1,  "maximum": 150,  "step": 1}),
    "anima_i2i_steps":    shared.OptionInfo(30, "img2img steps",                  gr.Slider, {"minimum": 1,  "maximum": 150,  "step": 1}),
}))
shared.options_templates.update(shared.options_section(('ui_qwen', "UI defaults 'qwen'", "ui"), {
    "qwen_t2i_width":  shared.OptionInfo(1328, "txt2img width",      gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "qwen_t2i_height": shared.OptionInfo(1328, "txt2img height",     gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "qwen_t2i_cfg":    shared.OptionInfo(1.0,  "txt2img CFG",        gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "qwen_t2i_hr_cfg": shared.OptionInfo(1.0,  "txt2img HiRes CFG",  gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "qwen_i2i_width":  shared.OptionInfo(1328, "img2img width",      gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "qwen_i2i_height": shared.OptionInfo(1328, "img2img height",     gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "qwen_i2i_cfg":    shared.OptionInfo(1.0,  "img2img CFG",        gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "qwen_GPU_MB":     shared.OptionInfo(total_vram - 1024, "GPU Weights (MB)", gr.Slider, {"minimum": 0,  "maximum": total_vram, "step": 1}),
    "qwen_t2i_steps":  shared.OptionInfo(8, "txt2img steps", gr.Slider, {"minimum": 1, "maximum": 150, "step": 1}),
    "qwen_i2i_steps":  shared.OptionInfo(8, "img2img steps", gr.Slider, {"minimum": 1, "maximum": 150, "step": 1}),
}))
