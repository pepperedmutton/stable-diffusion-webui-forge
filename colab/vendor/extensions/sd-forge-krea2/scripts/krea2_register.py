"""Register Krea 2 on the older Forge runtime used by this installation.

The extension's upstream loader targets Forge Neo.  This local adapter uses
the APIs that exist in this Forge revision and keeps scaled-FP8 weights in
their compact storage format.
"""

import json
import os
import struct
import sys
import traceback


EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXT_ROOT not in sys.path:
    sys.path.insert(0, EXT_ROOT)

CFG_DIR = os.path.join(EXT_ROOT, "hf_config", "Krea2")
_MAX_SAFETENSORS_HEADER = 64 * 1024 * 1024


def _safetensors_header(path):
    try:
        if not str(path).lower().endswith((".safetensors", ".sft")):
            return {}, {}
        file_size = os.path.getsize(path)
        with open(path, "rb") as handle:
            size_bytes = handle.read(8)
            if len(size_bytes) != 8:
                return {}, {}
            header_size = struct.unpack("<Q", size_bytes)[0]
            if not 2 <= header_size <= min(_MAX_SAFETENSORS_HEADER, file_size - 8):
                return {}, {}
            header = json.loads(handle.read(header_size))
        if not isinstance(header, dict):
            return {}, {}
        metadata = header.pop("__metadata__", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        tensors = {key: value for key, value in header.items() if isinstance(value, dict)}
        return metadata, tensors
    except Exception:
        return {}, {}


def _safetensors_keys(path):
    return set(_safetensors_header(path)[1])


def _canonical_text_key(key):
    while key.startswith(("transformer.", "text_encoder.")):
        key = key.split(".", 1)[1]
    if key.startswith("model.language_model."):
        key = "model." + key[len("model.language_model."):]
    return key


def _tensor_shape(tensors, key):
    value = tensors.get(key, {})
    shape = value.get("shape") if isinstance(value, dict) else None
    return tuple(shape) if isinstance(shape, list) else ()


def _is_krea2_text_encoder(path):
    metadata, raw_tensors = _safetensors_header(path)
    tensors = {_canonical_text_key(key): value for key, value in raw_tensors.items()}
    architecture = str(metadata.get("modelspec.architecture", "")).lower()
    if architecture and not any(name in architecture for name in ("qwen", "krea")):
        return False
    return (
        _tensor_shape(tensors, "model.embed_tokens.weight") == (151936, 2560)
        and "model.layers.0.self_attn.q_norm.weight" in tensors
        and "model.layers.35.self_attn.q_norm.weight" in tensors
        and "model.layers.35.mlp.down_proj.weight" in tensors
    )


def _is_krea2_vae(path):
    metadata, tensors = _safetensors_header(path)
    architecture = str(metadata.get("modelspec.architecture", "")).lower()
    if architecture and not any(name in architecture for name in ("qwen", "krea", "wan")):
        return False
    return (
        _tensor_shape(tensors, "encoder.conv1.weight") == (96, 3, 3, 3, 3)
        and _tensor_shape(tensors, "decoder.conv1.weight") == (384, 16, 3, 3, 3)
        and _tensor_shape(tensors, "decoder.head.2.weight") == (3, 96, 3, 3, 3)
        and _tensor_shape(tensors, "conv1.weight") == (32, 32, 1, 1, 1)
    )


def _is_bare_krea2(path):
    keys = _safetensors_keys(path)
    if not keys:
        return False
    has_dit = any(
        "txtfusion.projector" in key or "blocks.0.mod.lin" in key
        for key in keys
    )
    has_text_encoder = any(
        key.startswith("text_encoders.") or ".language_model." in key
        for key in keys
    )
    return has_dit and not has_text_encoder


def _find_krea2_modules():
    from modules import paths, shared

    cmd_opts = getattr(shared, "cmd_opts", None)
    text_dirs = [os.path.join(paths.models_path, "text_encoder")]
    text_dirs.extend(list(getattr(cmd_opts, "text_encoder_dirs", []) or []))
    vae_dirs = [os.path.join(paths.models_path, "VAE")]
    vae_dirs.extend(list(getattr(cmd_opts, "vae_dirs", []) or []))

    def find_valid(directories, validator, preferred_names):
        candidates = []
        seen = set()
        for directory in map(os.path.abspath, directories):
            normalized = os.path.normcase(directory)
            if normalized in seen:
                continue
            seen.add(normalized)
            try:
                for name in os.listdir(directory):
                    lower = name.lower()
                    path = os.path.join(directory, name)
                    if lower.endswith((".safetensors", ".sft")) and validator(path):
                        candidates.append(path)
            except OSError:
                continue

        def rank(path):
            lower = os.path.basename(path).lower()
            return tuple(0 if name in lower else 1 for name in preferred_names) + (
                os.path.normcase(path),
            )

        return min(candidates, key=rank) if candidates else None

    text_encoder = find_valid(
        text_dirs,
        _is_krea2_text_encoder,
        ("qwen3vl", "qwen3-vl", "4b", "bf16"),
    )
    vae = find_valid(
        vae_dirs,
        _is_krea2_vae,
        ("qwen_image_vae", "qwen", "vae"),
    )
    return [path for path in (text_encoder, vae) if path]


def _register_architecture():
    import torch
    from transformers import modeling_utils

    import backend.loader as loader
    from backend import memory_management
    from backend.nn.qwen3 import IntegratedQwen3Model
    from backend.operations import using_forge_operations
    from backend.state_dict import load_state_dict
    from backend.utils import read_arbitrary_config
    from huggingface_guess import detection, latent, model_list

    from krea2.ops import ScaledFP8Operations, prepare_scaled_fp8_state_dict

    krea_base = getattr(model_list, "AnimaCosmos", model_list.BASE)

    class Krea2Base(krea_base):
        huggingface_repo = CFG_DIR
        unet_config = {"image_model": "krea2"}
        unet_extra_config = {}
        sampling_settings = {"mu": 1.15}
        latent_format = latent.Wan21
        memory_usage_factor = 1.0
        supported_inference_dtypes = [torch.bfloat16, torch.float32]
        vae_key_prefix = ["vae."]
        text_encoder_key_prefix = ["text_encoders."]
        unet_target = "transformer"

        def inpaint_model(self):
            return False

        def model_type(self, state_dict, prefix=""):
            return model_list.ModelType.FLUX

        def clip_target(self, state_dict=None):
            state_dict = state_dict or {}
            prefix = self.text_encoder_key_prefix[0]
            if any(
                key in state_dict
                for key in (
                    f"{prefix}gemma2_2b.model.embed_tokens.weight",
                    f"{prefix}gemma2_2b.embed_tokens.weight",
                )
            ):
                return {"gemma2_2b": "text_encoder"}
            return {}

    if not any(getattr(item, "__name__", "") == "Krea2Base" for item in model_list.models):
        model_list.models.insert(0, Krea2Base)

    original_detect = detection.detect_unet_config

    def detect_krea2(state_dict, key_prefix):
        if (
            f"{key_prefix}txtfusion.projector.weight" in state_dict
            or f"{key_prefix}blocks.0.mod.lin" in state_dict
        ):
            return {"image_model": "krea2"}
        return original_detect(state_dict, key_prefix)

    if not getattr(detection.detect_unet_config, "_krea2_backport", False):
        detect_krea2._krea2_backport = True
        detection.detect_unet_config = detect_krea2

    def build_krea2_dit(guess, state_dict):
        from krea2.dit import SingleStreamDiT

        state_dict = prepare_scaled_fp8_state_dict(state_dict, "Krea 2 transformer")
        parameters = memory_management.state_dict_parameters(state_dict)
        state_dtype = memory_management.state_dict_dtype(state_dict)
        float8_types = (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
        )
        storage_dtype = state_dtype if state_dtype in float8_types else torch.bfloat16
        load_device = memory_management.get_torch_device()
        computation_dtype = memory_management.get_computation_dtype(
            load_device,
            parameters=parameters,
            supported_dtypes=guess.supported_inference_dtypes,
        )
        initial_device = memory_management.unet_inital_load_device(
            parameters=parameters,
            dtype=storage_dtype,
        )
        unet_config = {
            key: value
            for key, value in guess.unet_config.items()
            if key not in ("image_model", "audio_model")
        }

        with modeling_utils.no_init_weights():
            with using_forge_operations(
                operations=ScaledFP8Operations,
                device=initial_device,
                dtype=computation_dtype,
                manual_cast_enabled=True,
            ):
                model = SingleStreamDiT(
                    **unet_config,
                    device=initial_device,
                    dtype=computation_dtype,
                    operations=ScaledFP8Operations,
                )

        load_state_dict(model, state_dict, log_name="Krea2Transformer")
        model.config = unet_config
        model.storage_dtype = storage_dtype
        model.computation_dtype = computation_dtype
        model.load_device = load_device
        model.initial_device = initial_device
        model.offload_device = memory_management.unet_offload_device()
        return model

    def load_krea2_text_encoder(repo_path, state_dict):
        state_dict = prepare_scaled_fp8_state_dict(state_dict, "Krea 2 text encoder")
        normalized = {}
        for key, value in state_dict.items():
            if key.startswith("transformer."):
                key = key[len("transformer."):]
            if key.startswith("model."):
                key = key[len("model."):]
            normalized[key] = value

        config_path = os.path.join(repo_path, "text_encoder")
        config = read_arbitrary_config(config_path)
        device = memory_management.cpu
        dtype = torch.bfloat16

        with modeling_utils.no_init_weights():
            with using_forge_operations(
                operations=ScaledFP8Operations,
                device=device,
                dtype=dtype,
                manual_cast_enabled=True,
            ):
                model = IntegratedQwen3Model(
                    config=config,
                    device=device,
                    dtype=dtype,
                )

        missing, unexpected = model.load_state_dict(normalized, strict=False)
        if missing:
            print(f"Krea2 Qwen3-VL Missing: {missing}")
        if unexpected:
            print(f"Krea2 Qwen3-VL Unexpected: {unexpected}")
        return model

    original_component_loader = loader.load_huggingface_component

    def load_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if isinstance(guess, Krea2Base):
            if component_name == "transformer" and cls_name == "SingleStreamDiT":
                return build_krea2_dit(guess, state_dict)
            if component_name == "text_encoder":
                return load_krea2_text_encoder(repo_path, state_dict)
        return original_component_loader(
            guess,
            component_name,
            lib_name,
            cls_name,
            repo_path,
            state_dict,
        )

    if not getattr(loader.load_huggingface_component, "_krea2_backport", False):
        load_component._krea2_backport = True
        loader.load_huggingface_component = load_component

    original_replace = loader.replace_state_dict

    def replace_krea2_state_dict(sd, additional_sd, guess):
        if isinstance(guess, Krea2Base) and any(
            "self_attn.q_norm" in key for key in additional_sd
        ):
            flattened = {}
            for key, value in additional_sd.items():
                if key.startswith(("model.visual.", "visual.")):
                    continue
                if key.startswith("model.language_model."):
                    key = "model." + key[len("model.language_model."):]
                flattened[key] = value
            additional_sd = prepare_scaled_fp8_state_dict(
                flattened,
                "Krea 2 text encoder module",
            )
        return original_replace(sd, additional_sd, guess)

    if not getattr(loader.replace_state_dict, "_krea2_backport", False):
        replace_krea2_state_dict._krea2_backport = True
        loader.replace_state_dict = replace_krea2_state_dict

    from krea2.engine import Krea2

    Krea2.matched_guesses = [Krea2Base]
    # Krea2Base inherits AnimaCosmos on this Forge revision so it can reuse the
    # compatible latent/config contract.  The generic Anima engine also matches
    # AnimaCosmos instances, therefore Krea must be tested first.
    if Krea2 in loader.possible_models:
        loader.possible_models.remove(Krea2)
    loader.possible_models.insert(0, Krea2)

    print(
        "[krea2] registered Krea 2 backport: SingleStreamDiT, "
        "Qwen3-VL 4B, Qwen Image VAE, and scaled-FP8 loading"
    )


def _register_auto_modules():
    import backend.loader as loader
    import modules.sd_models as sd_models

    original_loader = sd_models.forge_loader
    if getattr(original_loader, "_krea2_auto_modules", False):
        return

    def forge_loader_with_krea2_modules(sd, additional_state_dicts=None, source_path=None):
        if source_path and _is_bare_krea2(source_path):
            selected = list(additional_state_dicts or [])
            has_text = any(_is_krea2_text_encoder(path) for path in selected)
            has_vae = any(_is_krea2_vae(path) for path in selected)
            if not (has_text and has_vae):
                for module_path in _find_krea2_modules():
                    if _is_krea2_text_encoder(module_path) and not has_text:
                        selected.insert(0, module_path)
                        has_text = True
                    elif _is_krea2_vae(module_path) and not has_vae:
                        selected.append(module_path)
                        has_vae = True
                additional_state_dicts = selected
                if has_text and has_vae:
                    print("[krea2] attached the required text encoder and VAE")

        return original_loader(
            sd,
            additional_state_dicts=additional_state_dicts,
            source_path=source_path,
        )

    forge_loader_with_krea2_modules._krea2_auto_modules = True
    sd_models.forge_loader = forge_loader_with_krea2_modules
    loader.forge_loader = forge_loader_with_krea2_modules


def _register_scaled_fp8_lora():
    import torch

    from backend import utils
    from backend.patcher.base import ModelPatcher

    original_add_patches = ModelPatcher.add_patches
    if getattr(original_add_patches, "_krea2_scaled_fp8_online", False):
        return

    float8_dtypes = tuple(
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
        )
        if dtype is not None
    )

    def targets_scaled_fp8(model, patch_key):
        if not isinstance(patch_key, str):
            try:
                patch_key = patch_key[0]
            except (IndexError, TypeError):
                return False
        if not isinstance(patch_key, str) or not patch_key.endswith(".weight"):
            return False

        parent_path = patch_key.rsplit(".", 1)[0]
        try:
            layer = utils.get_attr(model, parent_path) if parent_path else model
        except (AttributeError, IndexError, KeyError):
            return False

        weight = getattr(layer, "weight", None)
        scale = getattr(layer, "scale_weight", None)
        return (
            isinstance(weight, torch.Tensor)
            and weight.dtype in float8_dtypes
            and isinstance(scale, torch.Tensor)
            and scale.numel() == 1
        )

    def add_scaled_fp8_patches(
        self,
        *,
        filename,
        patches,
        strength_patch=1.0,
        strength_model=1.0,
        online_mode=False,
    ):
        if not online_mode and any(
            targets_scaled_fp8(self.model, patch_key) for patch_key in patches
        ):
            online_mode = True
            print(
                "[krea2] scaled-FP8 weights require on-the-fly LoRA: "
                f"{os.path.basename(str(filename))}"
            )

        return original_add_patches(
            self,
            filename=filename,
            patches=patches,
            strength_patch=strength_patch,
            strength_model=strength_model,
            online_mode=online_mode,
        )

    add_scaled_fp8_patches._krea2_scaled_fp8_online = True
    ModelPatcher.add_patches = add_scaled_fp8_patches


try:
    _register_architecture()
except Exception:
    print("[krea2] architecture registration failed; Forge can still start:\n" + traceback.format_exc())

try:
    _register_auto_modules()
except Exception:
    print("[krea2] automatic module attachment was skipped:\n" + traceback.format_exc())

try:
    _register_scaled_fp8_lora()
except Exception:
    print("[krea2] scaled-FP8 LoRA guard was skipped:\n" + traceback.format_exc())
