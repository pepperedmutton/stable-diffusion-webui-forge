"""Scaled-FP8 loading helpers for the Krea 2 Forge backport.

ComfyUI scaled-FP8 files store a float8 weight, a scalar ``weight_scale``,
and a small ``comfy_quant`` descriptor for each quantized linear layer.  This
Forge revision already knows how to apply ``scale_weight`` during a manual
cast, but it predates the ComfyUI metadata names.  The helpers below bridge
that small format difference without expanding the weights in memory.
"""

import json

import torch

from backend import operations


_FLOAT8_E4M3 = getattr(torch, "float8_e4m3fn", None)


def _decode_comfy_quant(value, key):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
        raise ValueError(f"{key} must be a uint8 JSON tensor")
    try:
        payload = bytes(value.detach().cpu().reshape(-1).tolist()).decode("utf-8")
        decoded = json.loads(payload)
    except Exception as exc:
        raise ValueError(f"{key} contains invalid quantization metadata") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{key} must contain a JSON object")
    return decoded


def prepare_scaled_fp8_state_dict(state_dict, label="Krea 2"):
    """Validate and normalize a ComfyUI scaled-FP8 state dict.

    The operation is idempotent.  It returns a shallow copy, renames every
    ``*.weight_scale`` tensor to Forge's ``*.scale_weight`` spelling, and
    removes the now-validated ``*.comfy_quant`` descriptors.
    """
    if not isinstance(state_dict, dict):
        raise TypeError(f"{label} state dict must be a dictionary")

    normalized = dict(state_dict)
    scale_keys = [
        key for key in normalized
        if key.endswith(".weight_scale") or key.endswith(".scale_weight")
    ]
    quant_keys = [key for key in normalized if key.endswith(".comfy_quant")]

    if not scale_keys and not quant_keys:
        return normalized

    full_precision_count = 0
    quantized_stems = set()
    for quant_key in quant_keys:
        stem = quant_key[:-len(".comfy_quant")]
        weight_key = stem + ".weight"
        scale_key = stem + ".weight_scale"
        forge_scale_key = stem + ".scale_weight"
        if weight_key not in normalized or (
            scale_key not in normalized and forge_scale_key not in normalized
        ):
            raise ValueError(f"{label}: orphan quantization descriptor {quant_key}")

        descriptor = _decode_comfy_quant(normalized[quant_key], quant_key)
        quant_format = descriptor.get("format")
        if quant_format != "float8_e4m3fn":
            raise ValueError(
                f"{label}: unsupported quantization format {quant_format!r} in {quant_key}"
            )
        unknown_fields = set(descriptor) - {
            "format",
            "full_precision_matrix_mult",
        }
        if unknown_fields:
            raise ValueError(
                f"{label}: unsupported quantization fields in {quant_key}: "
                f"{sorted(unknown_fields)}"
            )
        full_precision = descriptor.get("full_precision_matrix_mult", False)
        if not isinstance(full_precision, bool):
            raise ValueError(
                f"{label}: full_precision_matrix_mult in {quant_key} must be boolean"
            )
        full_precision_count += int(full_precision)
        quantized_stems.add(stem)

    for scale_key in scale_keys:
        suffix = ".weight_scale" if scale_key.endswith(".weight_scale") else ".scale_weight"
        stem = scale_key[:-len(suffix)]
        weight_key = stem + ".weight"
        weight = normalized.get(weight_key)
        scale = normalized.get(scale_key)

        if not isinstance(weight, torch.Tensor) or weight.dtype != _FLOAT8_E4M3:
            raise ValueError(f"{label}: {weight_key} is not a float8_e4m3fn tensor")
        if not isinstance(scale, torch.Tensor) or scale.numel() != 1:
            raise ValueError(f"{label}: {scale_key} must be a scalar tensor")
        if not scale.is_floating_point():
            raise ValueError(f"{label}: {scale_key} must use a floating-point dtype")
        if suffix == ".weight_scale" and stem not in quantized_stems:
            raise ValueError(
                f"{label}: {scale_key} has no matching comfy_quant descriptor"
            )

        forge_scale_key = stem + ".scale_weight"
        existing = normalized.get(forge_scale_key)
        if existing is not None and forge_scale_key != scale_key:
            if existing.shape != scale.shape or not torch.equal(existing.cpu(), scale.cpu()):
                raise ValueError(f"{label}: conflicting scale tensors for {stem}")
        normalized[forge_scale_key] = scale
        if scale_key != forge_scale_key:
            del normalized[scale_key]

    for quant_key in quant_keys:
        del normalized[quant_key]

    print(
        f"[krea2] {label}: using {len(scale_keys)} scaled-FP8 linear weights "
        f"without expanding them in memory "
        f"({full_precision_count} request non-FP8 matrix multiplication)"
    )
    return normalized


class ScaledFP8Operations(operations.ForgeOperations):
    """Forge operations that keep scaled-FP8 linear weights in float8 storage."""

    class Linear(torch.nn.Module):
        def __init__(self, in_features, out_features, *args, **kwargs):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.dummy = torch.nn.Parameter(
                torch.empty(
                    1,
                    device=operations.current_device,
                    dtype=operations.current_dtype,
                ),
                requires_grad=False,
            )
            self.weight = None
            self.scale_weight = None
            self.bias = None
            self.parameters_manual_cast = True

        def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            if not hasattr(self, "dummy"):
                return super()._load_from_state_dict(
                    state_dict,
                    prefix,
                    local_metadata,
                    strict,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                )

            weight = state_dict.get(prefix + "weight")
            if weight is not None:
                self.weight = torch.nn.Parameter(
                    weight.to(device=self.dummy.device),
                    requires_grad=False,
                )

            scale = state_dict.get(prefix + "scale_weight")
            if scale is not None:
                self.scale_weight = torch.nn.Parameter(
                    scale.to(device=self.dummy.device, dtype=torch.float32),
                    requires_grad=False,
                )

            bias = state_dict.get(prefix + "bias")
            if bias is not None:
                self.bias = torch.nn.Parameter(
                    bias.to(device=self.dummy.device, dtype=self.dummy.dtype),
                    requires_grad=False,
                )

            del self.dummy

        def forward(self, x):
            weight, bias, signal = operations.weights_manual_cast(self, x)
            with operations.main_stream_worker(weight, bias, signal):
                return torch.nn.functional.linear(x, weight, bias)
