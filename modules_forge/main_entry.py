import os
import re
import torch
import gradio as gr

from gradio.context import Context
from modules import shared, paths
from backend import memory_management, stream
from backend.args import dynamic_args
from modules.shared import cmd_opts
from modules import shared_options as shared_options_module
from modules_forge.checkpoint_inspection import inspect_checkpoint_file


if shared.options_templates is None:
    shared.options_templates = shared_options_module.options_templates


total_vram = int(memory_management.total_vram)

ui_forge_preset: gr.Radio = None
ui_qwen21_precision: gr.Radio = None

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

PRESET_BASE_OUTPUT_COUNT = 23
PRESET_KREA_STATE_OUTPUT_COUNT = 5
PRESET_OUTPUT_COUNT = PRESET_BASE_OUTPUT_COUNT + PRESET_KREA_STATE_OUTPUT_COUNT + 3
PRESET_STORAGE_DTYPE_KEYS = {
    'qwen': 'forge_unet_storage_dtype_qwen',
    'krea': 'forge_unet_storage_dtype_krea',
}

module_list = {}
QWEN_DEFAULT_CHECKPOINT = "Qwen-Image-2.1-NF4"
QWEN21_PRECISION_CHECKPOINTS = {
    "nf4": "Qwen-Image-2.1-NF4",
    "int8": "Qwen-Image-2.1-INT8",
    "bf16": "Qwen-Image-2.1-BF16",
}
QWEN21_PRECISION_CHOICES = [
    ("BF16 · Original precision", "bf16"),
    ("INT8 · 8-bit quantization", "int8"),
    ("NF4 · 4-bit quantization", "nf4"),
]
QWEN_DEFAULT_MODULES = []
QWEN21_DEFAULT_STEPS = 40
QWEN21_DEFAULT_CFG = 1.0
QWEN21_DEFAULT_SAMPLER = "Euler"
QWEN21_DEFAULT_SCHEDULER = "Simple"
KREA_DEFAULT_CHECKPOINT = "krea2Cocoamixzero_v10.safetensors"
KREA_DEFAULT_MODULES = [
    os.path.abspath(os.path.join(paths.models_path, "text_encoder", "qwen3vl_4b_fp8_scaled.safetensors")),
    os.path.abspath(os.path.join(paths.models_path, "VAE", "qwen_image_vae.safetensors")),
]
KREA_DEFAULT_SAMPLER = "Euler"
KREA_DEFAULT_SCHEDULER = "Simple"
KREA_DEFAULT_STEPS = 8
KREA_DEFAULT_CFG = 1.0
KREA_DEFAULT_CLIP_SKIP = 1
KREA_DEFAULT_WIDTH = 832
KREA_DEFAULT_HEIGHT = 1248
XL_DEFAULT_CHECKPOINT = "miaomiaoHarem_v20.safetensors"
XL_DEFAULT_MODULES = []
XL_FALLBACK_MODULES = [
    os.path.abspath(os.path.join(paths.models_path, "VAE", "pppanimixVAE_il.safetensors")),
]
XL_DEFAULT_SAMPLER = "Euler a"
XL_DEFAULT_SCHEDULER = "Automatic"
XL_DEFAULT_STEPS = 30
XL_DEFAULT_CFG = 6.0
XL_DEFAULT_CLIP_SKIP = 1
XL_DEFAULT_WIDTH = 896
XL_DEFAULT_HEIGHT = 1152
XL_INFERENCE_MEMORY_MB = 4096
ANIMA_DEFAULT_CHECKPOINT = "miaomiaoHarem_anima15.safetensors"
ANIMA_DEFAULT_MODULES = [
    os.path.abspath(os.path.join(paths.models_path, "text_encoder", "anima_text_encoder.safetensors")),
    os.path.abspath(os.path.join(paths.models_path, "VAE", "anima_vae.safetensors")),
]
ANIMA_DEFAULT_SAMPLER = "Euler a"
ANIMA_DEFAULT_SCHEDULER = "Normal"
ANIMA_DEFAULT_STEPS = 30
ANIMA_DEFAULT_CFG = 4.0
ANIMA_DEFAULT_WIDTH = 896
ANIMA_DEFAULT_HEIGHT = 1152
FORGE_PRESET_UI_CHOICES = [
    ("Qwen Image 2.1", "qwen"),
    ("Krea 2 - CocoaMixZero v1.0", "krea"),
    ("MiaoMiao Harem - Illustrious v2.0", "xl"),
    ("MiaoMiao Harem - Anima_1.5", "anima"),
]
CHECKPOINT_UI_DISPLAY_NAMES = {
    KREA_DEFAULT_CHECKPOINT.lower(): "Krea 2 - CocoaMixZero v1.0",
    XL_DEFAULT_CHECKPOINT.lower(): "MiaoMiao Harem - Illustrious v2.0",
    ANIMA_DEFAULT_CHECKPOINT.lower(): "MiaoMiao Harem - Anima_1.5",
}
ANIMA_STABLE_LIVE_PREVIEW_OPTIONS = {
    "live_previews_enable": True,
    "show_progress_grid": False,
    "show_progress_type": "Approx NN",
    "show_progress_every_n_steps": 1,
}
_anima_live_preview_backup = None


def get_default_qwen_modules():
    # The isolated Qwen 2.1 pipeline owns its tokenizer, encoder and VAE.
    return []


def qwen21_precision_from_checkpoint(value):
    name = str(value or "").replace("\\", "/").rsplit("/", 1)[-1].split(" [", 1)[0].lower()
    for precision, checkpoint in QWEN21_PRECISION_CHECKPOINTS.items():
        if name == checkpoint.lower():
            return precision
    if name in ("qwen-image-2.1", "qwen image 2.1"):
        return "nf4"
    return None


def get_qwen21_precision():
    active = qwen21_precision_from_checkpoint(shared.opts.data.get("sd_model_checkpoint"))
    if active is not None:
        return active
    saved = shared.opts.data.get("forge_qwen21_precision")
    if saved in QWEN21_PRECISION_CHECKPOINTS:
        return saved
    return qwen21_precision_from_checkpoint(shared.opts.data.get("forge_checkpoint_qwen")) or "nf4"


def remember_qwen21_checkpoint(checkpoint_name):
    precision = qwen21_precision_from_checkpoint(checkpoint_name)
    if precision is None:
        return False
    changed = False
    for key, value in (("forge_qwen21_precision", precision),
                       ("forge_checkpoint_qwen", QWEN21_PRECISION_CHECKPOINTS[precision])):
        if shared.opts.data.get(key) != value:
            shared.opts.set(key, value)
            changed = True
    return changed


def migrate_qwen21_options(save=True):
    """Upgrade old Qwen aliases once while retaining a selected precision."""
    precision = get_qwen21_precision()
    active_name = str(shared.opts.data.get("sd_model_checkpoint", "") or "").lower()
    if "qwen" in active_name and qwen21_precision_from_checkpoint(active_name) is None:
        precision = "nf4"
    desired = {
        "forge_checkpoint_qwen": QWEN21_PRECISION_CHECKPOINTS[precision],
        "forge_qwen21_precision": precision,
        "forge_additional_modules_qwen": [],
        "forge_unet_storage_dtype_qwen": "Automatic",
        "forge_qwen21_profile_version": 2,
    }
    if "qwen" in active_name:
        desired["sd_model_checkpoint"] = QWEN21_PRECISION_CHECKPOINTS[precision]
        desired["forge_additional_modules"] = []
    changed = False
    for key, value in desired.items():
        if shared.opts.data.get(key) != value:
            shared.opts.set(key, value)
            changed = True
    if changed and save:
        shared.opts.save(shared.config_filename)
    return changed


def sync_qwen21_precision_ui():
    return gr.update(value=get_qwen21_precision(),
                     visible=normalize_forge_preset(shared.opts.forge_preset) == "qwen")


def on_qwen21_precision_change(precision, *current_ui_state):
    from modules import sd_models

    if precision not in QWEN21_PRECISION_CHECKPOINTS:
        raise ValueError("Select BF16, INT8, or NF4 for Qwen Image 2.1.")
    checkpoint_name = QWEN21_PRECISION_CHECKPOINTS[precision]
    if sd_models.get_closet_checkpoint_match(checkpoint_name) is None:
        raise gr.Error(f"{checkpoint_name} is not installed or its weights are incomplete. Refresh the model list after installation.")
    checkpoint_change(checkpoint_name, preset="qwen", save=False, refresh=False)
    remember_qwen21_checkpoint(checkpoint_name)
    return on_preset_change("qwen", *current_ui_state)


def qwen21_dimension(value):
    """Keep the requested canvas, aligning only incompatible dimensions."""
    try:
        return max(256, min(2048, int(float(value)) // 32 * 32))
    except (TypeError, ValueError, OverflowError):
        return 1024


def get_default_krea_modules():
    return [m for m in KREA_DEFAULT_MODULES if os.path.exists(m)]


def get_default_xl_modules(checkpoint_name=""):
    default_modules = [m for m in XL_DEFAULT_MODULES if os.path.exists(m)]
    if default_modules:
        return default_modules

    checkpoint_file = resolve_checkpoint_file(checkpoint_name)
    family, has_builtin_vae = inspect_checkpoint_file(checkpoint_file) if checkpoint_file else (None, None)
    if family == "xl" and has_builtin_vae is False:
        return [m for m in XL_FALLBACK_MODULES if os.path.exists(m)]
    return []


def get_xl_model_memory():
    default_model_memory = total_vram - XL_INFERENCE_MEMORY_MB
    model_memory = getattr(shared.opts, "xl_GPU_MB", default_model_memory)
    if model_memory < 0 or model_memory > default_model_memory:
        return default_model_memory
    return model_memory


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


def get_checkpoint_preferred_modules(checkpoint_name):
    checkpoint_file = resolve_checkpoint_file(checkpoint_name)
    if checkpoint_file is None:
        return None

    try:
        from modules import extra_networks

        metadata = extra_networks.get_user_metadata(checkpoint_file)
    except Exception:
        return None

    selected = metadata.get("vae_te")
    if selected is None:
        selected = metadata.get("vae")
    if selected is None:
        return None
    if isinstance(selected, str):
        selected = [selected]

    modules = []
    for value in selected:
        if value in {None, "", "Built in", "Automatic", "None"}:
            continue
        module_name = os.path.basename(str(value))
        module_path = module_list.get(module_name)
        if module_path and os.path.exists(module_path):
            modules.append(os.path.abspath(module_path))
        elif os.path.isfile(str(value)):
            modules.append(os.path.abspath(str(value)))
    return sorted(list(dict.fromkeys(modules)))


def get_preset_additional_modules(preset, checkpoint_name=""):
    preset = normalize_forge_preset(preset)
    checkpoint_profile = infer_module_profile_from_checkpoint(checkpoint_name)

    # Checkpoint-driven behavior:
    # - if checkpoint is Krea/Anima/Qwen, force matching required modules
    # - if checkpoint is another family, unload extra modules
    # - if checkpoint is empty (startup edge), fall back to preset defaults
    if checkpoint_profile is not None:
        effective_profile = checkpoint_profile
    else:
        checkpoint_text = str(checkpoint_name).strip().lower() if checkpoint_name is not None else ""
        effective_profile = preset if checkpoint_text in {"", "none"} and preset in {"qwen", "krea", "anima"} else None

    if effective_profile == 'qwen':
        return []

    if effective_profile == 'krea':
        saved = shared.opts.data.get("forge_additional_modules_krea", []) or get_default_krea_modules()
        modules = resolve_existing_modules(saved)
        return modules if modules else get_default_krea_modules()

    if effective_profile == 'anima':
        saved = shared.opts.data.get("forge_additional_modules_anima", []) or get_default_anima_modules()
        modules = resolve_existing_modules(saved)
        return modules if modules else get_default_anima_modules()

    checkpoint_text = str(checkpoint_name).strip().lower() if checkpoint_name is not None else ""
    checkpoint_preset = infer_forge_preset_from_checkpoint(checkpoint_name) if checkpoint_text not in {"", "none"} else None
    if checkpoint_preset == 'xl' or (preset == 'xl' and checkpoint_text in {"", "none"}):
        preferred_modules = get_checkpoint_preferred_modules(checkpoint_name)
        if preferred_modules is not None:
            return preferred_modules
        if shared.opts.data.get("forge_additional_modules_xl_configured", False):
            saved = shared.opts.data.get("forge_additional_modules_xl", [])
            modules = resolve_existing_modules(saved)
            return modules if modules or not saved else get_default_xl_modules(checkpoint_name)
        return get_default_xl_modules(checkpoint_name)

    if is_lumina_checkpoint_name(checkpoint_name):
        return []

    return []


def migrate_xl_module_state(checkpoint_name):
    if shared.opts.data.get("forge_additional_modules_xl_configured", False):
        return False
    if infer_forge_preset_from_checkpoint(checkpoint_name) != "xl":
        return False

    legacy_modules = resolve_existing_modules(shared.opts.data.get("forge_additional_modules", []))
    if not legacy_modules:
        return False

    shared.opts.set("forge_additional_modules_xl", legacy_modules)
    shared.opts.set("forge_additional_modules_xl_configured", True)
    return True


def normalize_forge_preset(preset):
    if preset in ['sd', 'all']:
        return 'qwen'
    if preset == 'lumina':
        return 'krea'
    if preset in ['qwen', 'krea', 'xl', 'anima']:
        return preset
    return 'qwen'


def normalize_storage_dtype(value):
    return value if value in forge_unet_storage_dtype_options else 'Automatic'


def preset_storage_dtype_key(preset):
    return PRESET_STORAGE_DTYPE_KEYS.get(normalize_forge_preset(preset))


def activate_preset_storage_dtype(previous_preset, new_preset):
    previous_preset = normalize_forge_preset(previous_preset)
    new_preset = normalize_forge_preset(new_preset)
    current_value = normalize_storage_dtype(shared.opts.data.get('forge_unet_storage_dtype', 'Automatic'))
    changed = False

    previous_key = preset_storage_dtype_key(previous_preset)
    if previous_preset != new_preset and previous_key is not None and shared.opts.data.get(previous_key) != current_value:
        shared.opts.set(previous_key, current_value)
        changed = True

    new_key = preset_storage_dtype_key(new_preset)
    new_value = normalize_storage_dtype(shared.opts.data.get(new_key, 'Automatic')) if new_key is not None else 'Automatic'
    if new_key is not None and shared.opts.data.get(new_key) != new_value:
        shared.opts.set(new_key, new_value)
        changed = True
    if shared.opts.data.get('forge_unet_storage_dtype') != new_value:
        shared.opts.set('forge_unet_storage_dtype', new_value)
        changed = True

    return new_value, changed


def on_storage_dtype_change(value, preset):
    value = normalize_storage_dtype(value)
    preset = normalize_forge_preset(preset)
    shared.opts.set('forge_unet_storage_dtype', value)

    preset_key = preset_storage_dtype_key(preset)
    if preset_key is not None:
        shared.opts.set(preset_key, value)

    shared.opts.save(shared.config_filename)
    refresh_model_loading_parameters()


def is_lumina_checkpoint_name(value):
    if value is None:
        return False

    text = str(value).lower()
    return "lumina" in text or "neta" in text


def normalize_checkpoint_text(value):
    return ''.join(ch for ch in str(value).lower() if ch.isalnum())


def resolve_checkpoint_file(value):
    if value is None:
        return None

    filename = getattr(value, "filename", None)
    if filename and os.path.isfile(filename):
        return os.path.abspath(filename)

    text = str(value)
    if os.path.isfile(text):
        return os.path.abspath(text)

    try:
        from modules import sd_models

        checkpoint_info = sd_models.get_closet_checkpoint_match(text)
    except Exception:
        checkpoint_info = None

    filename = getattr(checkpoint_info, "filename", None)
    return os.path.abspath(filename) if filename and os.path.isfile(filename) else None


def checkpoint_structure_family(value):
    filename = resolve_checkpoint_file(value)
    family, _ = inspect_checkpoint_file(filename) if filename else (None, None)
    return family


def checkpoint_ui_choices(use_short=False):
    from modules import sd_models

    choices = []
    for checkpoint_info in sd_models.checkpoints_list.values():
        value = checkpoint_info.name
        visible_name = checkpoint_info.short_title if use_short else checkpoint_info.name
        display_name = CHECKPOINT_UI_DISPLAY_NAMES.get(os.path.basename(checkpoint_info.name).lower())
        label = f"{display_name} - {visible_name}" if display_name else visible_name
        choices.append((label, value))
    return choices


def is_qwen_checkpoint_name(value):
    text = normalize_checkpoint_text(value)
    return "qwen" in text


def is_krea_checkpoint_name(value):
    text = normalize_checkpoint_text(value)
    return "krea2" in text or "cocoamixzero" in text


def is_anima_checkpoint_name(value):
    text = normalize_checkpoint_text(value)
    return "anima" in text


def infer_module_profile_from_checkpoint(value):
    if value is None:
        return None

    if checkpoint_structure_family(value) == "xl":
        return None

    if is_krea_checkpoint_name(value):
        return "krea"

    if is_qwen_checkpoint_name(value):
        return "qwen"

    if is_anima_checkpoint_name(value):
        return "anima"

    return None


def infer_forge_preset_from_checkpoint(value):
    structure_family = checkpoint_structure_family(value)
    if structure_family == "xl":
        return "xl"
    if structure_family == "non_xl":
        return None

    module_profile = infer_module_profile_from_checkpoint(value)
    if module_profile in {"qwen", "krea", "anima"}:
        return module_profile
    if is_xl_checkpoint_name(value):
        return "xl"
    return None


def is_xl_checkpoint_name(value):
    text = normalize_checkpoint_text(value)
    filename_text = normalize_checkpoint_text(os.path.basename(str(value)))
    if not text:
        return False

    if filename_text == normalize_checkpoint_text(XL_DEFAULT_CHECKPOINT):
        return True

    structure_family = checkpoint_structure_family(value)
    if structure_family is not None:
        return structure_family == "xl"

    if is_qwen_checkpoint_name(value) or is_krea_checkpoint_name(value) or is_anima_checkpoint_name(value) or is_lumina_checkpoint_name(value):
        return False

    return any(keyword in text for keyword in [
        "xl",
        "sdxl",
        "illustrious",
        "specustrious",
        "animij",
        "noobai",
        "janku",
        "novaanime",
        "wai",
    ])


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

    global ui_checkpoint, ui_vae, ui_clip_skip, ui_forge_unet_storage_dtype_options, ui_forge_async_loading, ui_forge_pin_shared_memory, ui_forge_inference_memory, ui_forge_preset, ui_qwen21_precision

    migrate_qwen21_options()

    if shared.opts.sd_model_checkpoint in [None, 'None', 'none', '']:
        if len(sd_models.checkpoints_list) == 0:
            sd_models.list_models()
        if len(sd_models.checkpoints_list) > 0:
            shared.opts.set('sd_model_checkpoint', next(iter(sd_models.checkpoints_list.values())).name)

    migrated_preset = normalize_forge_preset(shared.opts.forge_preset)
    if migrated_preset != shared.opts.forge_preset:
        shared.opts.set('forge_preset', migrated_preset)
        shared.opts.save(shared.config_filename)

    ui_forge_preset = gr.Radio(
        label="Model preset",
        value=lambda: normalize_forge_preset(shared.opts.forge_preset),
        choices=FORGE_PRESET_UI_CHOICES,
        elem_id="forge_ui_preset",
    )
    ui_qwen21_precision = gr.Radio(
        label="Qwen Image 2.1 precision",
        value=get_qwen21_precision,
        choices=QWEN21_PRECISION_CHOICES,
        visible=normalize_forge_preset(shared.opts.forge_preset) == "qwen",
        elem_id="forge_qwen21_precision",
    )
    gr.Markdown(
        "**Qwen Image 2.1** · For image editing, upload an image and describe the change in the prompt. "
        "Precision applies to the image model and text encoder; the RGBA VAE stays BF16. "
        "Negative prompts, LoRAs, Hires fix and refiners are unavailable for this model.",
        elem_id="forge_qwen21_help",
    )

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
        checkpoint_info = sd_models.get_closet_checkpoint_match(shared.opts.sd_model_checkpoint)
        checkpoint_name = checkpoint_info.name if checkpoint_info is not None else shared.opts.sd_model_checkpoint
        if checkpoint_name != shared.opts.sd_model_checkpoint:
            shared.opts.set('sd_model_checkpoint', checkpoint_name)
            shared.opts.save(shared.config_filename)
        return gr.update(choices=a, value=checkpoint_name), gr.update(choices=b)

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
    ui_forge_unet_storage_dtype_options.change(
        on_storage_dtype_change,
        inputs=[ui_forge_unet_storage_dtype_options, ui_forge_preset],
        queue=False,
        show_progress=False,
    )

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

    ui_vae.input(modules_change, inputs=[ui_vae, ui_forge_preset], queue=False, show_progress=False)

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
    ckpt_list = checkpoint_ui_choices(shared.opts.sd_checkpoint_dropdown_use_short)

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

    migrate_xl_module_state(checkpoint_name)
    additional_modules = get_preset_additional_modules(preset, checkpoint_name)
    if additional_modules != sorted(shared.opts.data.get('forge_additional_modules', [])):
        shared.opts.set('forge_additional_modules', additional_modules)

    model_data.forge_loading_parameters = dict(
        checkpoint_info=checkpoint_info,
        additional_modules=additional_modules,
        unet_storage_dtype=unet_storage_dtype
    )

    from modules_forge.qwen21 import release_if_inactive
    release_if_inactive(checkpoint_info)

    print(f'Model selected: {model_data.forge_loading_parameters}')
    print(f'Using online LoRAs in FP16: {lora_fp16}')
    processing.need_global_unload = True

    return


def checkpoint_change(ckpt_name: str, preset=None, save=True, refresh=True):
    from modules import sd_models

    """ checkpoint name can be a number of valid aliases. Returns True if checkpoint changed. """
    sampler_changed = False
    if is_xl_checkpoint_name(ckpt_name):
        sampler_changed = sync_xl_sampler_defaults(save=False)

    new_ckpt_info = sd_models.get_closet_checkpoint_match(ckpt_name)
    current_ckpt_info = sd_models.get_closet_checkpoint_match(shared.opts.data.get('sd_model_checkpoint', ''))
    canonical_ckpt_name = new_ckpt_info.name if new_ckpt_info is not None else ckpt_name
    checkpoint_profile = infer_module_profile_from_checkpoint(canonical_ckpt_name)
    checkpoint_preset = infer_forge_preset_from_checkpoint(canonical_ckpt_name)
    precision_changed = remember_qwen21_checkpoint(canonical_ckpt_name) if checkpoint_profile == 'qwen' else False
    if new_ckpt_info == current_ckpt_info:
        checkpoint_name_changed = shared.opts.data.get('sd_model_checkpoint') != canonical_ckpt_name
        if checkpoint_name_changed:
            shared.opts.set('sd_model_checkpoint', canonical_ckpt_name)
        if preset == 'qwen' and checkpoint_profile == 'qwen' and shared.opts.data.get('forge_checkpoint_qwen') != canonical_ckpt_name:
            shared.opts.set('forge_checkpoint_qwen', canonical_ckpt_name)
            checkpoint_name_changed = True
        if preset == 'krea' and checkpoint_profile == 'krea' and shared.opts.data.get('forge_checkpoint_krea') != canonical_ckpt_name:
            shared.opts.set('forge_checkpoint_krea', canonical_ckpt_name)
            checkpoint_name_changed = True
        if preset == 'anima' and checkpoint_profile == 'anima' and shared.opts.data.get('forge_checkpoint_anima') != canonical_ckpt_name:
            shared.opts.set('forge_checkpoint_anima', canonical_ckpt_name)
            checkpoint_name_changed = True
        if preset == 'xl' and checkpoint_preset == 'xl' and shared.opts.data.get('forge_checkpoint_xl') != canonical_ckpt_name:
            shared.opts.set('forge_checkpoint_xl', canonical_ckpt_name)
            checkpoint_name_changed = True
        if save and (sampler_changed or checkpoint_name_changed or precision_changed):
            shared.opts.save(shared.config_filename)
        return False

    shared.opts.set('sd_model_checkpoint', canonical_ckpt_name)
    if preset == 'qwen' and checkpoint_profile == 'qwen':
        shared.opts.set('forge_checkpoint_qwen', canonical_ckpt_name)
    if preset == 'krea' and checkpoint_profile == 'krea':
        shared.opts.set('forge_checkpoint_krea', canonical_ckpt_name)
    if preset == 'anima' and checkpoint_profile == 'anima':
        shared.opts.set('forge_checkpoint_anima', canonical_ckpt_name)
    if preset == 'xl' and checkpoint_preset == 'xl':
        shared.opts.set('forge_checkpoint_xl', canonical_ckpt_name)

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
    if ckpt_name not in (None, '', 'None', 'none'):
        return is_xl_checkpoint_name(ckpt_name)
    return preset == 'xl'


def on_checkpoint_sampler_ui_sync(
    ckpt_name,
    preset=None,
    current_t2i_width=None,
    current_i2i_width=None,
    current_t2i_height=None,
    current_i2i_height=None,
    current_t2i_negative_prompt=None,
    current_i2i_negative_prompt=None,
    current_enable_hr=None,
    current_t2i_refiner=None,
    current_i2i_refiner=None,
    qwen_ui_backup=None,
    current_t2i_prompt=None,
    current_i2i_prompt=None,
):
    target_preset = infer_forge_preset_from_checkpoint(ckpt_name)
    if target_preset is None:
        checkpoint_change(ckpt_name, preset=preset, save=False, refresh=False)
        refresh_model_loading_parameters()
        shared.opts.save(shared.config_filename)
        return [gr.update() for _ in range(PRESET_OUTPUT_COUNT)]

    checkpoint_change(ckpt_name, preset=target_preset, save=False, refresh=False)
    preset_updates = on_preset_change(
        target_preset,
        current_t2i_width,
        current_i2i_width,
        current_t2i_height,
        current_i2i_height,
        current_t2i_negative_prompt=current_t2i_negative_prompt,
        current_i2i_negative_prompt=current_i2i_negative_prompt,
        current_enable_hr=current_enable_hr,
        current_t2i_refiner=current_t2i_refiner,
        current_i2i_refiner=current_i2i_refiner,
        qwen_ui_backup=qwen_ui_backup,
        current_t2i_prompt=current_t2i_prompt,
        current_i2i_prompt=current_i2i_prompt,
    )
    shared.opts.save(shared.config_filename)
    return [gr.update(value=target_preset), *preset_updates[1:]]


def modules_change(module_values: list, preset=None, save=True, refresh=True, persist_profile=True) -> bool:
    """ module values may be provided as file paths, or just the module names. Returns True if modules changed. """
    modules = []
    for v in module_values:
        module_name = os.path.basename(v) # If the input is a filepath, extract the file name
        if module_name in module_list:
            modules.append(module_list[module_name])
    modules = sorted(modules)
    
    profile_key = {
        'qwen': 'forge_additional_modules_qwen',
        'krea': 'forge_additional_modules_krea',
        'anima': 'forge_additional_modules_anima',
        'xl': 'forge_additional_modules_xl',
    }.get(preset) if persist_profile else None
    modules_changed = modules != sorted(shared.opts.data.get('forge_additional_modules', []))
    profile_changed = profile_key is not None and modules != sorted(shared.opts.data.get(profile_key, []))
    xl_configuration_changed = preset == 'xl' and persist_profile and not shared.opts.data.get('forge_additional_modules_xl_configured', False)

    if not modules_changed and not profile_changed and not xl_configuration_changed:
        return False

    if modules_changed:
        shared.opts.set('forge_additional_modules', modules)
    if profile_changed:
        shared.opts.set(profile_key, modules)
    if xl_configuration_changed:
        shared.opts.set('forge_additional_modules_xl_configured', True)

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


def qwen21_clean_prompt(prompt):
    return re.sub(r'<(?:lora|lyco|hypernet):[^<>]*>', '', prompt, flags=re.IGNORECASE)


def qwen21_ui_state_updates(preset, values, backup, positive_prompts=(None, None)):
    """Store disabled UI values per browser session and restore them on exit."""
    if preset == 'qwen':
        if backup is None:
            backup = dict(zip(('t2i_negative', 'i2i_negative', 'hr', 't2i_refiner', 'i2i_refiner'), values))
            for key, prompt in zip(('t2i_prompt', 'i2i_prompt'), positive_prompts):
                if prompt is not None:
                    backup[key] = prompt
                    backup[key + '_clean'] = qwen21_clean_prompt(prompt)
        prompt_updates = [gr.update(value=qwen21_clean_prompt(prompt)) if prompt is not None else gr.update()
                          for prompt in positive_prompts]
        return ([
            gr.update(value='', visible=False, interactive=False),
            gr.update(value='', visible=False, interactive=False),
            gr.update(value=False, visible=False, interactive=False),
            gr.update(value=False, visible=False, interactive=False),
            gr.update(value=False, visible=False, interactive=False),
        ], backup, prompt_updates)
    if backup is None:
        return None
    keys = ('t2i_negative', 'i2i_negative', 'hr', 't2i_refiner', 'i2i_refiner')
    # Hires/refiner use hidden state checkboxes behind visible InputAccordions.
    # Revealing the hidden components creates duplicate controls in the page.
    restored = [gr.update(value=backup.get(key), visible=index < 2, interactive=True)
                for index, key in enumerate(keys)]
    if preset == 'krea':
        # Krea already clears these unsupported settings on every selection.
        for index, update in enumerate(restored):
            update['value'] = '' if index < 2 else False
    prompt_updates = []
    for key, prompt in zip(('t2i_prompt', 'i2i_prompt'), positive_prompts):
        # Restore removed tags when the prompt was unchanged; keep edits made in Qwen.
        if key in backup and prompt == backup.get(key + '_clean'):
            prompt_updates.append(gr.update(value=backup[key]))
        else:
            prompt_updates.append(gr.update())
    return restored, None, prompt_updates


def build_preset_ui_updates(preset, updates, qwen_ui_state=None):
    if len(updates) != PRESET_BASE_OUTPUT_COUNT:
        raise RuntimeError(f'Preset update count is {len(updates)}; expected {PRESET_BASE_OUTPUT_COUNT}.')

    if normalize_forge_preset(preset) == 'krea':
        krea_state_updates = [
            gr.update(value=''),
            gr.update(value=''),
            gr.update(value=False),
            gr.update(value=False),
            gr.update(value=False),
        ]
    else:
        krea_state_updates = [gr.update() for _ in range(PRESET_KREA_STATE_OUTPUT_COUNT)]

    is_qwen = normalize_forge_preset(preset) == 'qwen'
    for index in (13, 14, 17, 18, 19, 20):
        updates[index]['interactive'] = not is_qwen
    for index in (9, 10, 11, 12):
        updates[index]['step'] = 32 if is_qwen else 8
        updates[index]['minimum'] = 256 if is_qwen else 64
    backup_update = gr.update()
    prompt_updates = [gr.update(), gr.update()]
    if qwen_ui_state is not None:
        krea_state_updates, backup_update, prompt_updates = qwen_ui_state
    # Set interactivity on every selection, including when the Qwen backup was
    # already consumed. Otherwise Krea's CFG=1 can leave Anima's prompts disabled.
    for index, update in enumerate(krea_state_updates):
        update['visible'] = not is_qwen and index < 2
        update['interactive'] = not is_qwen and (
            index >= 2 or updates[13 + index].get('value', 1.0) != 1.0
        )
    return [*updates, *krea_state_updates, backup_update, *prompt_updates]


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
    ui_txt2img_negative_prompt = get_a1111_ui_component('txt2img', 'Negative prompt')
    ui_img2img_negative_prompt = get_a1111_ui_component('img2img', 'Negative prompt')
    ui_txt2img_prompt = get_a1111_ui_component('txt2img', 'Prompt')
    ui_img2img_prompt = get_a1111_ui_component('img2img', 'Prompt')
    ui_txt2img_enable_hr = get_a1111_ui_component('txt2img', 'enable_hr')
    ui_txt2img_enable_refiner = get_a1111_ui_component('txt2img', 'refiner_enable')
    ui_img2img_enable_refiner = get_a1111_ui_component('img2img', 'refiner_enable')
    ui_qwen_backup = gr.State(value=None)

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
        ui_txt2img_negative_prompt,
        ui_img2img_negative_prompt,
        ui_txt2img_enable_hr,
        ui_txt2img_enable_refiner,
        ui_img2img_enable_refiner,
        ui_qwen_backup,
        ui_txt2img_prompt,
        ui_img2img_prompt,
    ]

    saved_state_inputs = [ui_txt2img_negative_prompt, ui_img2img_negative_prompt, ui_txt2img_enable_hr,
                          ui_txt2img_enable_refiner, ui_img2img_enable_refiner, ui_qwen_backup,
                          ui_txt2img_prompt, ui_img2img_prompt]
    preset_inputs = [ui_forge_preset, ui_txt2img_width, ui_img2img_width, ui_txt2img_height, ui_img2img_height, *saved_state_inputs]
    ui_forge_preset.input(
        on_preset_change, inputs=preset_inputs, outputs=output_targets, queue=False, show_progress=False,
    ).then(fn=sync_qwen21_precision_ui, inputs=[], outputs=[ui_qwen21_precision], queue=False, show_progress=False
    ).then(fn=None, js="refreshForgePresetTokenCounters", inputs=[], outputs=[], queue=False, show_progress=False)
    ui_forge_preset.input(js="clickLoraRefresh", fn=None, queue=False, show_progress=False)
    ui_qwen21_precision.input(
        on_qwen21_precision_change, inputs=[ui_qwen21_precision, *preset_inputs[1:]],
        outputs=output_targets, queue=False, show_progress=False,
    ).then(fn=sync_qwen21_precision_ui, inputs=[], outputs=[ui_qwen21_precision], queue=False, show_progress=False
    ).then(fn=None, js="refreshForgePresetTokenCounters", inputs=[], outputs=[], queue=False, show_progress=False)
    ui_checkpoint.change(
        on_checkpoint_sampler_ui_sync,
        inputs=[
            ui_checkpoint,
            ui_forge_preset,
            ui_txt2img_width,
            ui_img2img_width,
            ui_txt2img_height,
            ui_img2img_height,
            *saved_state_inputs,
        ],
        outputs=[ui_forge_preset, *output_targets[1:]],
        queue=False,
        show_progress=False,
    ).then(fn=sync_qwen21_precision_ui, inputs=[], outputs=[ui_qwen21_precision], queue=False, show_progress=False
    ).then(fn=None, js="refreshForgePresetTokenCounters", inputs=[], outputs=[], queue=False, show_progress=False)
    Context.root_block.load(
        on_preset_page_load, inputs=preset_inputs[1:], outputs=[ui_forge_preset, *output_targets],
        queue=False, show_progress=False,
    ).then(fn=sync_qwen21_precision_ui, inputs=[], outputs=[ui_qwen21_precision], queue=False, show_progress=False
    ).then(fn=None, js="refreshForgePresetTokenCounters", inputs=[], outputs=[], queue=False, show_progress=False)

    refresh_model_loading_parameters()
    return


def on_preset_page_load(*current_ui_state):
    """A browser reload must use the live selection, not startup UI defaults."""
    preset = normalize_forge_preset(shared.opts.forge_preset)
    updates = on_preset_change(preset, *current_ui_state)
    return [gr.update(value=preset), *updates]


def on_preset_change(
    preset=None,
    current_t2i_width=None,
    current_i2i_width=None,
    current_t2i_height=None,
    current_i2i_height=None,
    current_t2i_negative_prompt=None,
    current_i2i_negative_prompt=None,
    current_enable_hr=None,
    current_t2i_refiner=None,
    current_i2i_refiner=None,
    qwen_ui_backup=None,
    current_t2i_prompt=None,
    current_i2i_prompt=None,
):
    from modules import ui_loadsave, sd_models

    def opt(key, default):
        value = shared.opts.data.get(key, default)
        return default if value is None else value

    previous_preset = normalize_forge_preset(shared.opts.forge_preset)
    preset = normalize_forge_preset(preset if preset is not None else shared.opts.forge_preset)
    qwen_ui_state = qwen21_ui_state_updates(preset, (
        current_t2i_negative_prompt, current_i2i_negative_prompt, current_enable_hr,
        current_t2i_refiner, current_i2i_refiner,
    ), qwen_ui_backup, (current_t2i_prompt, current_i2i_prompt))
    storage_dtype, storage_dtype_changed = activate_preset_storage_dtype(previous_preset, preset)
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

    preset_changed = shared.opts.forge_preset != preset
    if preset_changed:
        shared.opts.set('forge_preset', preset)
    if preset_changed or storage_dtype_changed:
        shared.opts.save(shared.config_filename)

    sync_anima_live_preview_options(previous_preset, preset)

    if preset == 'qwen':
        migrate_qwen21_options()
        qwen_checkpoint = QWEN21_PRECISION_CHECKPOINTS[get_qwen21_precision()]
        qwen_modules = []

        if sd_models.get_closet_checkpoint_match(qwen_checkpoint) is not None:
            checkpoint_change(qwen_checkpoint, preset='qwen', save=True, refresh=False)
            checkpoint_update = gr.update(value=qwen_checkpoint)

        modules_change(qwen_modules, preset='qwen', save=True, refresh=False)

        current_t2i_width = qwen21_dimension(current_t2i_width)
        current_i2i_width = qwen21_dimension(current_i2i_width)
        current_t2i_height = qwen21_dimension(current_t2i_height)
        current_i2i_height = qwen21_dimension(current_i2i_height)

        refresh_model_loading_parameters()
        return build_preset_ui_updates(preset, [
            checkpoint_update,                                                         # ui_checkpoint
            gr.update(visible=False, value=[]),                                       # ui_vae
            gr.update(visible=False, value=1),                                         # ui_clip_skip
            gr.update(visible=False, value='Automatic'),                              # ui_forge_unet_storage_dtype_options
            gr.update(visible=False, value='Queue'),                                  # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),                                    # ui_forge_pin_shared_memory
            gr.update(visible=False),                                                # ui_forge_inference_memory
            gr.update(value=opt("qwen21_t2i_steps", QWEN21_DEFAULT_STEPS)),            # ui_txt2img_steps
            gr.update(value=opt("qwen21_i2i_steps", QWEN21_DEFAULT_STEPS)),            # ui_img2img_steps
            gr.update(value=current_t2i_width),                                          # ui_txt2img_width
            gr.update(value=current_i2i_width),                                          # ui_img2img_width
            gr.update(value=current_t2i_height),                                         # ui_txt2img_height
            gr.update(value=current_i2i_height),                                         # ui_img2img_height
            gr.update(value=QWEN21_DEFAULT_CFG),                                      # ui_txt2img_cfg
            gr.update(value=QWEN21_DEFAULT_CFG),                                      # ui_img2img_cfg
            gr.update(visible=False, value=3.5),                                       # ui_txt2img_distilled_cfg
            gr.update(visible=False, value=3.5),                                       # ui_img2img_distilled_cfg
            gr.update(value=QWEN21_DEFAULT_SAMPLER),                                  # ui_txt2img_sampler
            gr.update(value=QWEN21_DEFAULT_SAMPLER),                                  # ui_img2img_sampler
            gr.update(value=QWEN21_DEFAULT_SCHEDULER),                                 # ui_txt2img_scheduler
            gr.update(value=QWEN21_DEFAULT_SCHEDULER),                                 # ui_img2img_scheduler
            gr.update(visible=False, value=QWEN21_DEFAULT_CFG),                        # ui_txt2img_hr_cfg
            gr.update(visible=False, value=3.5),                                       # ui_txt2img_hr_distilled_cfg
        ], qwen_ui_state)

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
        return build_preset_ui_updates(preset, [
            checkpoint_update,                                                         # ui_checkpoint
            gr.update(visible=False, value=[os.path.basename(x) for x in shared.opts.forge_additional_modules]),  # ui_vae
            gr.update(visible=False, value=1),                                         # ui_clip_skip
            gr.update(visible=True, value=storage_dtype),                              # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                    # ui_forge_async_loading
            gr.update(visible=True, value='CPU'),                                      # ui_forge_pin_shared_memory
            gr.update(visible=True, value=model_mem),                                  # ui_forge_inference_memory
            gr.update(value=ANIMA_DEFAULT_STEPS),                                       # ui_txt2img_steps
            gr.update(value=ANIMA_DEFAULT_STEPS),                                       # ui_img2img_steps
            gr.update(value=current_t2i_width),                                          # ui_txt2img_width
            gr.update(value=current_i2i_width),                                          # ui_img2img_width
            gr.update(value=current_t2i_height),                                         # ui_txt2img_height
            gr.update(value=current_i2i_height),                                         # ui_img2img_height
            gr.update(value=ANIMA_DEFAULT_CFG),                                         # ui_txt2img_cfg
            gr.update(value=ANIMA_DEFAULT_CFG),                                         # ui_img2img_cfg
            gr.update(visible=True, value=3.5),                                         # ui_txt2img_distilled_cfg
            gr.update(visible=True, value=3.5),                                         # ui_img2img_distilled_cfg
            gr.update(value=ANIMA_DEFAULT_SAMPLER),                                     # ui_txt2img_sampler
            gr.update(value=ANIMA_DEFAULT_SAMPLER),                                     # ui_img2img_sampler
            gr.update(value=ANIMA_DEFAULT_SCHEDULER),                                   # ui_txt2img_scheduler
            gr.update(value=ANIMA_DEFAULT_SCHEDULER),                                   # ui_img2img_scheduler
            gr.update(visible=True, value=ANIMA_DEFAULT_CFG),                            # ui_txt2img_hr_cfg
            gr.update(visible=True, value=3.5),                                         # ui_txt2img_hr_distilled_cfg
        ], qwen_ui_state)

    if preset == 'krea':
        selected_krea_checkpoint = sd_models.get_closet_checkpoint_match(
            shared.opts.data.get('sd_model_checkpoint', '')
        )
        if selected_krea_checkpoint is not None and is_krea_checkpoint_name(selected_krea_checkpoint.name):
            krea_checkpoint = selected_krea_checkpoint.name
        else:
            krea_checkpoint = opt("forge_checkpoint_krea", KREA_DEFAULT_CHECKPOINT)
            if not is_krea_checkpoint_name(krea_checkpoint):
                krea_checkpoint = KREA_DEFAULT_CHECKPOINT

        if sd_models.get_closet_checkpoint_match(krea_checkpoint) is not None:
            checkpoint_change(krea_checkpoint, preset='krea', save=True, refresh=False)
            checkpoint_update = gr.update(value=krea_checkpoint)

        krea_modules = get_preset_additional_modules('krea', krea_checkpoint)
        modules_change(krea_modules, preset='krea', save=True, refresh=False)

        model_mem = opt("krea_GPU_MB", total_vram - 4096)
        if model_mem < 0 or model_mem > total_vram:
            model_mem = total_vram - 4096
        refresh_model_loading_parameters()
        return build_preset_ui_updates(preset, [
            checkpoint_update,                                                         # ui_checkpoint
            gr.update(visible=True, value=[os.path.basename(x) for x in shared.opts.forge_additional_modules]),  # ui_vae
            gr.update(visible=False, value=KREA_DEFAULT_CLIP_SKIP),                     # ui_clip_skip
            gr.update(visible=True, value=storage_dtype),                               # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                     # ui_forge_async_loading
            gr.update(visible=True, value='CPU'),                                       # ui_forge_pin_shared_memory
            gr.update(visible=True, value=model_mem),                                   # ui_forge_inference_memory
            gr.update(value=opt("krea_t2i_steps", KREA_DEFAULT_STEPS)),                 # ui_txt2img_steps
            gr.update(value=opt("krea_i2i_steps", KREA_DEFAULT_STEPS)),                 # ui_img2img_steps
            gr.update(value=current_t2i_width),                                         # ui_txt2img_width
            gr.update(value=current_i2i_width),                                         # ui_img2img_width
            gr.update(value=current_t2i_height),                                        # ui_txt2img_height
            gr.update(value=current_i2i_height),                                        # ui_img2img_height
            gr.update(value=opt("krea_t2i_cfg", KREA_DEFAULT_CFG)),                     # ui_txt2img_cfg
            gr.update(value=opt("krea_i2i_cfg", KREA_DEFAULT_CFG)),                     # ui_img2img_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_distilled_cfg
            gr.update(visible=False, value=3.5),                                        # ui_img2img_distilled_cfg
            gr.update(value=opt("krea_t2i_sampler", KREA_DEFAULT_SAMPLER)),             # ui_txt2img_sampler
            gr.update(value=opt("krea_i2i_sampler", KREA_DEFAULT_SAMPLER)),             # ui_img2img_sampler
            gr.update(value=opt("krea_t2i_scheduler", KREA_DEFAULT_SCHEDULER)),         # ui_txt2img_scheduler
            gr.update(value=opt("krea_i2i_scheduler", KREA_DEFAULT_SCHEDULER)),         # ui_img2img_scheduler
            gr.update(visible=False, value=opt("krea_t2i_hr_cfg", KREA_DEFAULT_CFG)),   # ui_txt2img_hr_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_hr_distilled_cfg
        ], qwen_ui_state)

    if preset == 'xl':
        selected_xl_checkpoint = sd_models.get_closet_checkpoint_match(shared.opts.data.get('sd_model_checkpoint', ''))
        if selected_xl_checkpoint is not None and is_xl_checkpoint_name(selected_xl_checkpoint.name):
            xl_checkpoint = selected_xl_checkpoint.name
        else:
            saved_xl_checkpoint = opt("forge_checkpoint_xl", XL_DEFAULT_CHECKPOINT)
            selected_xl_checkpoint = sd_models.get_closet_checkpoint_match(saved_xl_checkpoint)
            if selected_xl_checkpoint is not None and is_xl_checkpoint_name(selected_xl_checkpoint.name):
                xl_checkpoint = selected_xl_checkpoint.name
            else:
                xl_checkpoint = XL_DEFAULT_CHECKPOINT
        xl_modules = get_preset_additional_modules('xl', xl_checkpoint)
        sync_xl_sampler_defaults(save=True)

        if sd_models.get_closet_checkpoint_match(xl_checkpoint) is not None:
            checkpoint_change(xl_checkpoint, preset='xl', save=True, refresh=False)
            checkpoint_update = gr.update(value=xl_checkpoint)

        model_mem = get_xl_model_memory()
        modules_change(xl_modules, preset='xl', save=True, refresh=False, persist_profile=False)
        refresh_model_loading_parameters()
        return build_preset_ui_updates(preset, [
            checkpoint_update,                                                         # ui_checkpoint
            gr.update(visible=True, value=[os.path.basename(x) for x in shared.opts.forge_additional_modules]),  # ui_vae
            gr.update(visible=False, value=XL_DEFAULT_CLIP_SKIP),                       # ui_clip_skip
            gr.update(visible=True, value=storage_dtype),                               # ui_forge_unet_storage_dtype_options
            gr.update(visible=False, value='Queue'),                                    # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),                                      # ui_forge_pin_shared_memory
            gr.update(visible=True, value=model_mem),                                   # ui_forge_inference_memory
            gr.update(value=opt("xl_t2i_steps", XL_DEFAULT_STEPS)),                    # ui_txt2img_steps
            gr.update(value=opt("xl_i2i_steps", XL_DEFAULT_STEPS)),                    # ui_img2img_steps
            gr.update(value=current_t2i_width),                                          # ui_txt2img_width
            gr.update(value=current_i2i_width),                                          # ui_img2img_width
            gr.update(value=current_t2i_height),                                         # ui_txt2img_height
            gr.update(value=current_i2i_height),                                         # ui_img2img_height
            gr.update(value=getattr(shared.opts, "xl_t2i_cfg", XL_DEFAULT_CFG)),        # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "xl_i2i_cfg", XL_DEFAULT_CFG)),        # ui_img2img_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_distilled_cfg
            gr.update(visible=False, value=3.5),                                        # ui_img2img_distilled_cfg
            gr.update(value=XL_DEFAULT_SAMPLER),                                        # ui_txt2img_sampler
            gr.update(value=XL_DEFAULT_SAMPLER),                                        # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "xl_t2i_scheduler", XL_DEFAULT_SCHEDULER)),  # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "xl_i2i_scheduler", XL_DEFAULT_SCHEDULER)),  # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "xl_t2i_hr_cfg", XL_DEFAULT_CFG)),  # ui_txt2img_hr_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_hr_distilled_cfg
        ], qwen_ui_state)

    if preset == 'flux':
        model_mem = getattr(shared.opts, "flux_GPU_MB", total_vram - 1024)
        if model_mem < 0 or model_mem > total_vram:
            model_mem = total_vram - 1024
        modules_change([], preset=preset, save=False, refresh=False)
        refresh_model_loading_parameters()
        return build_preset_ui_updates(preset, [
            checkpoint_update,                                                         # ui_checkpoint
            gr.update(visible=False),                                                   # ui_vae
            gr.update(visible=False, value=1),                                          # ui_clip_skip
            gr.update(visible=True, value=storage_dtype),                               # ui_forge_unet_storage_dtype_options
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
        ], qwen_ui_state)

    modules_change([], preset=preset, save=False, refresh=False)
    refresh_model_loading_parameters()
    return build_preset_ui_updates(preset, [
        checkpoint_update,  # ui_checkpoint
        gr.update(visible=False),  # ui_vae
        gr.update(visible=True, value=1),  # ui_clip_skip
        gr.update(visible=True, value=storage_dtype),  # ui_forge_unet_storage_dtype_options
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
    ], qwen_ui_state)

shared.options_templates.update(shared.options_section(('ui_krea', "UI defaults: Krea 2 - CocoaMixZero v1.0", "ui"), {
    "krea_t2i_width":  shared.OptionInfo(KREA_DEFAULT_WIDTH,  "txt2img width",      gr.Slider, {"minimum": 64, "maximum": 2048, "step": 16}),
    "krea_t2i_height": shared.OptionInfo(KREA_DEFAULT_HEIGHT, "txt2img height",     gr.Slider, {"minimum": 64, "maximum": 2048, "step": 16}),
    "krea_t2i_cfg":    shared.OptionInfo(KREA_DEFAULT_CFG,    "txt2img CFG",        gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "krea_t2i_hr_cfg": shared.OptionInfo(KREA_DEFAULT_CFG,    "txt2img HiRes CFG",  gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "krea_i2i_width":  shared.OptionInfo(KREA_DEFAULT_WIDTH,  "img2img width",      gr.Slider, {"minimum": 64, "maximum": 2048, "step": 16}),
    "krea_i2i_height": shared.OptionInfo(KREA_DEFAULT_HEIGHT, "img2img height",     gr.Slider, {"minimum": 64, "maximum": 2048, "step": 16}),
    "krea_i2i_cfg":    shared.OptionInfo(KREA_DEFAULT_CFG,    "img2img CFG",        gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "krea_GPU_MB":     shared.OptionInfo(total_vram - 4096, "GPU Weights (MB)", gr.Slider, {"minimum": 0,  "maximum": total_vram,   "step": 1}),
}))
shared.options_templates.update(shared.options_section(('ui_xl', "UI defaults: MiaoMiao Harem - Illustrious v2.0", "ui"), {
    "xl_t2i_width":  shared.OptionInfo(XL_DEFAULT_WIDTH,  "txt2img width",  gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "xl_t2i_height": shared.OptionInfo(XL_DEFAULT_HEIGHT, "txt2img height", gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "xl_t2i_cfg":    shared.OptionInfo(XL_DEFAULT_CFG, "txt2img CFG",       gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "xl_t2i_hr_cfg": shared.OptionInfo(XL_DEFAULT_CFG, "txt2img HiRes CFG", gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "xl_i2i_width":  shared.OptionInfo(XL_DEFAULT_WIDTH,  "img2img width",  gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "xl_i2i_height": shared.OptionInfo(XL_DEFAULT_HEIGHT, "img2img height", gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "xl_i2i_cfg":    shared.OptionInfo(XL_DEFAULT_CFG, "img2img CFG",       gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "xl_GPU_MB":     shared.OptionInfo(total_vram - XL_INFERENCE_MEMORY_MB, "GPU Weights (MB)", gr.Slider, {"minimum": 0,  "maximum": total_vram,   "step": 1}),
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
shared.options_templates.update(shared.options_section(('ui_anima', "UI defaults: MiaoMiao Harem - Anima_1.5", "ui"), {
    "anima_t2i_width":    shared.OptionInfo(ANIMA_DEFAULT_WIDTH,  "txt2img width",  gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "anima_t2i_height":   shared.OptionInfo(ANIMA_DEFAULT_HEIGHT, "txt2img height", gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "anima_t2i_cfg":      shared.OptionInfo(ANIMA_DEFAULT_CFG, "txt2img CFG",     gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "anima_t2i_hr_cfg":   shared.OptionInfo(ANIMA_DEFAULT_CFG, "txt2img HiRes CFG", gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "anima_t2i_d_cfg":    shared.OptionInfo(3.5,  "txt2img Distilled CFG",        gr.Slider, {"minimum": 0,  "maximum": 30,   "step": 0.1}),
    "anima_t2i_hr_d_cfg": shared.OptionInfo(3.5,  "txt2img Distilled HiRes CFG",  gr.Slider, {"minimum": 0,  "maximum": 30,   "step": 0.1}),
    "anima_i2i_width":    shared.OptionInfo(ANIMA_DEFAULT_WIDTH,  "img2img width",  gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "anima_i2i_height":   shared.OptionInfo(ANIMA_DEFAULT_HEIGHT, "img2img height", gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "anima_i2i_cfg":      shared.OptionInfo(ANIMA_DEFAULT_CFG, "img2img CFG",     gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "anima_i2i_d_cfg":    shared.OptionInfo(3.5,  "img2img Distilled CFG",        gr.Slider, {"minimum": 0,  "maximum": 30,   "step": 0.1}),
    "anima_GPU_MB":       shared.OptionInfo(total_vram - 1024, "GPU Weights (MB)",gr.Slider, {"minimum": 0,  "maximum": total_vram,   "step": 1}),
    "anima_t2i_steps":    shared.OptionInfo(ANIMA_DEFAULT_STEPS, "txt2img steps", gr.Slider, {"minimum": 1,  "maximum": 150,  "step": 1}),
    "anima_i2i_steps":    shared.OptionInfo(ANIMA_DEFAULT_STEPS, "img2img steps", gr.Slider, {"minimum": 1,  "maximum": 150,  "step": 1}),
}))
shared.options_templates.update(shared.options_section(('ui_qwen21', "UI defaults: Qwen Image 2.1", "ui"), {
    "forge_qwen21_precision": shared.OptionInfo("nf4", "Qwen precision (managed by model selector)", gr.State),
    "qwen21_t2i_steps": shared.OptionInfo(QWEN21_DEFAULT_STEPS, "txt2img steps", gr.Slider, {"minimum": 1, "maximum": 150, "step": 1}),
    "qwen21_i2i_steps": shared.OptionInfo(QWEN21_DEFAULT_STEPS, "Image editing steps", gr.Slider, {"minimum": 1, "maximum": 150, "step": 1}),
}))
