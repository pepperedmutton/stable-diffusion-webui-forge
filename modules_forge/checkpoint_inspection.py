import functools
import json
import os
import struct


LEGACY_MODEL_TYPE_KEY = "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"
BUILTIN_VAE_KEY = "first_stage_model.decoder.conv_in.weight"
MAX_SAFETENSORS_HEADER_SIZE = 128 * 1024 * 1024


def _metadata_family(metadata):
    if not isinstance(metadata, dict):
        return None

    values = [
        metadata.get("modelspec.architecture"),
        metadata.get("modelspec.implementation"),
        metadata.get("ss_base_model_version"),
    ]
    text = " ".join(str(value).lower() for value in values if value is not None)
    if any(marker in text for marker in ("sdxl", "stable-diffusion-xl", "illustrious")):
        return "xl"
    if any(marker in text for marker in ("stable-diffusion-v1", "stable-diffusion-v2", "flux", "qwen", "anima")):
        return "non_xl"
    return None


@functools.lru_cache(maxsize=128)
def _inspect_safetensors_file(filename, file_size, modified_time_ns):
    del modified_time_ns

    try:
        with open(filename, "rb") as file:
            length_bytes = file.read(8)
            if len(length_bytes) != 8:
                return None, None

            header_size = struct.unpack("<Q", length_bytes)[0]
            if header_size < 2 or header_size > MAX_SAFETENSORS_HEADER_SIZE or header_size > file_size - 8:
                return None, None

            header = json.loads(file.read(header_size))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None, None

    if not isinstance(header, dict):
        return None, None

    model_entry = header.get(LEGACY_MODEL_TYPE_KEY)
    model_shape = model_entry.get("shape") if isinstance(model_entry, dict) else None
    family = None
    if isinstance(model_shape, list) and len(model_shape) >= 2:
        cross_attention_dim = model_shape[1]
        if cross_attention_dim == 2048:
            family = "xl"
        elif cross_attention_dim in {768, 1024, 1280}:
            family = "non_xl"

    if family is None:
        family = _metadata_family(header.get("__metadata__"))

    vae_entry = header.get(BUILTIN_VAE_KEY)
    vae_shape = vae_entry.get("shape") if isinstance(vae_entry, dict) else None
    has_builtin_vae = isinstance(vae_shape, list) and len(vae_shape) >= 2 and vae_shape[1] == 4
    return family, has_builtin_vae


def inspect_checkpoint_file(filename):
    try:
        filename = os.path.abspath(os.fspath(filename))
        if not filename.lower().endswith((".safetensors", ".sft")):
            return None, None
        file_stat = os.stat(filename)
        if not os.path.isfile(filename):
            return None, None
    except (OSError, TypeError, ValueError):
        return None, None

    return _inspect_safetensors_file(filename, file_stat.st_size, file_stat.st_mtime_ns)
