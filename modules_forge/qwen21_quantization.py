"""Adapt bitsandbytes INT8 input layout and component CPU offload."""

from __future__ import annotations

from types import MethodType


def patch_int8_cpu_offload(model) -> int:
    """Patch this component's device moves and return its INT8 layer count.

    PyTorch Module._apply can preserve an Int8Params object while replacing its
    data, leaving CB/SCB and the first-forward MatmulLtState caches on the old
    device. The normal bitsandbytes Linear8bitLt.to override is bypassed when a
    parent module moves. Keep packed CB aliased to the moved weight and move its
    scales too. This is scoped to the supplied component, not any library class.

    Use _apply rather than an instance .to wrapper: Accelerate removes instance
    .to attributes whenever Diffusers rebuilds its CPU-offload hooks.
    """
    import bitsandbytes as bnb
    import torch

    layers = tuple(layer for layer in model.modules() if isinstance(layer, bnb.nn.Linear8bitLt))
    for layer in layers:
        if layer.weight.dtype != torch.int8 or layer.weight.has_fp16_weights:
            raise ValueError("Qwen INT8 offload requires prequantized INT8 inference weights.")
    if not layers:
        return 0

    def contiguous_input(module, args, kwargs):
        # bitsandbytes 0.50.2's CUDA int8_vectorwise_quant passes A.data_ptr()
        # into a row-major kernel without respecting strides. Qwen attention
        # can produce transposed/noncontiguous activations. Unlike weight
        # quantization error, this changed a real 4096x4096 layer's relative
        # error from 1.4% to 142%. Normalize layout before bnb casts to FP16.
        if args and isinstance(args[0], torch.Tensor):
            if not args[0].is_contiguous():
                args = (args[0].contiguous(), *args[1:])
        elif isinstance(kwargs.get("x"), torch.Tensor) and not kwargs["x"].is_contiguous():
            kwargs = {**kwargs, "x": kwargs["x"].contiguous()}
        return args, kwargs

    for layer in layers:
        if not hasattr(layer, "_qwen21_int8_contiguous_hook_handle"):
            layer._qwen21_int8_contiguous_hook_handle = layer.register_forward_pre_hook(
                contiguous_input, with_kwargs=True,
            )
    existing = getattr(model, "_qwen21_int8_offload_layer_count", None)
    if existing is not None:
        return existing
    original_apply = model._apply

    def apply_with_int8_state(self, *args, **kwargs):
        result = original_apply(*args, **kwargs)
        for layer in layers:
            device = layer.weight.device
            for container in (layer.weight, layer.state):
                if container.CB is not None:
                    # CB is the packed weight itself; copying it separately
                    # would double INT8 weight memory and retain a stale alias.
                    container.CB = layer.weight.data
                if container.SCB is not None:
                    container.SCB = container.SCB.to(device)
            # Outlier indices and any other initialized inference caches must
            # follow the weight without changing their individual dtypes.
            for field, value in vars(layer.state).items():
                if field not in ("CB", "SCB") and isinstance(value, torch.Tensor):
                    setattr(layer.state, field, value.to(device))
        return result

    model._apply = MethodType(apply_with_int8_state, model)
    model._qwen21_int8_offload_layer_count = len(layers)
    return len(layers)
