"""
Krea 2 (K2) single-stream MMDiT — ported to Forge Neo's backend.

Ported verbatim from ComfyUI `comfy/ldm/krea2/model.py` (the ground-truth, working
implementation), with only the plumbing repointed to Forge's ComfyUI-lineage backend:
  comfy.model_management.cast_to        -> backend.memory_management.cast_to_device
  comfy.ldm.flux.layers.EmbedND/timestep_embedding -> backend.nn.flux.*
  comfy.ldm.flux.math.apply_rope        -> backend.nn.flux.apply_rope
  comfy.ldm.common_dit.pad_to_patch_size-> backend.utils.pad_to_patch_size
  optimized_attention_masked(skip_reshape=True)  -> backend.attention.attention_function
        (with backend-specific additive padding-mask shapes)
  comfy.patcher_extension wrapper        -> dropped (Forge calls forward() directly)

The ML (GQA + per-head QK-norm + sigmoid-gated attention, DoubleSharedModulation,
SwiGLU, 3-axis RoPE [40,44,44] θ=1000, txtfusion adapter) is unchanged.
"""
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from backend import memory_management
from backend.attention import attention_function, attention_pytorch
from backend.nn.flux import EmbedND, timestep_embedding, apply_rope

from . import enhance


_XFORMERS_MASK_LIMIT_BYTES = 128 * 1024 * 1024


class PreparedAttentionMask(NamedTuple):
    bias: torch.Tensor
    valid_queries: torch.Tensor
    backend_name: str


def _prepare_attention_mask(mask, heads, sequence_length, dtype, device, backend_name=None):
    """Convert a padding mask to the additive shape required by the active backend."""
    if mask is None or isinstance(mask, PreparedAttentionMask):
        return mask
    if mask.ndim != 2:
        raise ValueError(f"Krea2 expects a [batch, sequence] padding mask, got {tuple(mask.shape)}")
    if mask.shape[1] != sequence_length:
        raise ValueError(
            f"Krea2 mask length {mask.shape[1]} does not match sequence length {sequence_length}"
        )
    if not dtype.is_floating_point:
        raise TypeError(f"Krea2 attention requires a floating-point dtype, got {dtype}")

    valid = mask.to(device=device, dtype=torch.bool)
    if bool(valid.all()):
        return None
    selected_backend = backend_name or getattr(attention_function, "__name__", "")
    batch = valid.shape[0]
    element_size = torch.empty((), dtype=dtype).element_size()
    xformers_bytes = batch * heads * sequence_length * sequence_length * element_size

    # Forge xformers flattens batch and heads for skip_reshape=True.  A dense
    # mask can be too large for image-token sequences, so use SDPA in that case.
    if selected_backend == "attention_xformers" and xformers_bytes > _XFORMERS_MASK_LIMIT_BYTES:
        selected_backend = "attention_pytorch"

    pair_is_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    bias = torch.zeros(pair_is_valid.shape, device=device, dtype=dtype)
    bias.masked_fill_(~pair_is_valid, -torch.finfo(dtype).max)
    if selected_backend == "attention_xformers":
        bias = bias.repeat_interleave(heads, dim=0)
    else:
        bias = bias.unsqueeze(1)
    return PreparedAttentionMask(bias, valid, selected_backend)


def pad_to_patch_size(image, patch_size=(2, 2), padding_mode="circular"):
    if padding_mode == "circular" and (torch.jit.is_tracing() or torch.jit.is_scripting()):
        padding_mode = "reflect"

    pad = ()
    for index in range(image.ndim - 2):
        amount = (patch_size[index] - image.shape[index + 2] % patch_size[index]) % patch_size[index]
        pad = (0, amount) + pad
    return torch.nn.functional.pad(image, pad, mode=padding_mode)


class RMSNorm(nn.Module):
    """RMSNorm with the reference (1 + scale) weight convention (scale stored zero-centered)."""

    def __init__(self, features: int, eps: float = 1e-5, device=None, dtype=None, operations=None):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.empty(features, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        weight = memory_management.cast_to_device(self.scale, x.device, torch.float32) + 1.0
        return F.rms_norm(x.float(), (x.shape[-1],), weight=weight, eps=self.eps).to(dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int, device=None, dtype=None, operations=None):
        super().__init__()
        self.qnorm = RMSNorm(dim, device=device, dtype=dtype, operations=operations)
        self.knorm = RMSNorm(dim, device=device, dtype=dtype, operations=operations)

    def forward(self, q, k):
        return self.qnorm(q), self.knorm(k)


class SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128,
                 device=None, dtype=None, operations=None):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = operations.Linear(features, mlpdim, bias=bias, device=device, dtype=dtype)
        self.up = operations.Linear(features, mlpdim, bias=bias, device=device, dtype=dtype)
        self.down = operations.Linear(mlpdim, features, bias=bias, device=device, dtype=dtype)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)).mul_(self.up(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: Optional[int] = None, bias: bool = False,
                 device=None, dtype=None, operations=None):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads
        self.wq = operations.Linear(dim, self.headdim * self.heads, bias=bias, device=device, dtype=dtype)
        self.wk = operations.Linear(dim, self.headdim * self.kvheads, bias=bias, device=device, dtype=dtype)
        self.wv = operations.Linear(dim, self.headdim * self.kvheads, bias=bias, device=device, dtype=dtype)
        self.gate = operations.Linear(dim, dim, bias=bias, device=device, dtype=dtype)
        self.qknorm = QKNorm(self.headdim, device=device, dtype=dtype, operations=operations)
        self.wo = operations.Linear(dim, dim, bias=bias, device=device, dtype=dtype)

    def forward(self, x, freqs=None, mask=None, transformer_options={}):
        q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
        q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
        q, k = self.qknorm(q, k)
        if freqs is not None:
            q, k = apply_rope(q, k, freqs)
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        prepared_mask = _prepare_attention_mask(
            mask,
            heads=self.heads,
            sequence_length=x.shape[1],
            dtype=q.dtype,
            device=q.device,
        )
        backend = attention_function
        backend_mask = None
        if prepared_mask is not None:
            backend_mask = prepared_mask.bias
            if prepared_mask.backend_name == "attention_pytorch" and getattr(
                attention_function, "__name__", ""
            ) != "attention_pytorch":
                backend = attention_pytorch
        out = backend(q, k, v, self.heads, backend_mask, skip_reshape=True)
        if prepared_mask is not None:
            out = out * prepared_mask.valid_queries.to(out.dtype).unsqueeze(-1)
        return self.wo(out * F.sigmoid(gate))


class SimpleModulation(nn.Module):
    def __init__(self, dim: int, device=None, dtype=None, operations=None):
        super().__init__()
        self.lin = nn.Parameter(torch.empty(2, dim, device=device, dtype=dtype))

    def forward(self, vec):
        out = vec + memory_management.cast_to_device(self.lin, vec.device, vec.dtype).unsqueeze(0)
        scale, shift = out.chunk(2, dim=1)
        return scale, shift


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int, device=None, dtype=None, operations=None):
        super().__init__()
        self.lin = nn.Parameter(torch.empty(6 * dim, device=device, dtype=dtype))

    def forward(self, vec):
        out = vec + memory_management.cast_to_device(self.lin, vec.device, vec.dtype)
        return out.chunk(6, dim=-1)


class TextFusionBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None, device=None, dtype=None, operations=None):
        super().__init__()
        self.prenorm = RMSNorm(features, device=device, dtype=dtype, operations=operations)
        self.postnorm = RMSNorm(features, device=device, dtype=dtype, operations=operations)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias, device=device, dtype=dtype, operations=operations)
        self.mlp = SwiGLU(features, multiplier, bias, device=device, dtype=dtype, operations=operations)

    def forward(self, x, mask=None, transformer_options={}):
        x = x + self.attn(self.prenorm(x), mask=mask, transformer_options=transformer_options)
        x = x + self.mlp(self.postnorm(x))
        return x


class TextFusionTransformer(nn.Module):
    def __init__(self, num_txt_layers, txt_dim, heads, multiplier, bias=False, kvheads=None, device=None, dtype=None, operations=None):
        super().__init__()
        self.heads = heads
        self.layerwise_blocks = nn.ModuleList([
            TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads, device=device, dtype=dtype, operations=operations)
            for _ in range(2)
        ])
        self.projector = operations.Linear(num_txt_layers, 1, bias=False, device=device, dtype=dtype)
        self.refiner_blocks = nn.ModuleList([
            TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads, device=device, dtype=dtype, operations=operations)
            for _ in range(2)
        ])

    def forward(self, x, mask=None, transformer_options={}):
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None, transformer_options=transformer_options)
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x).squeeze(-1)
        mask = _prepare_attention_mask(mask, self.heads, l, x.dtype, x.device)
        for block in self.refiner_blocks:
            x = block(x, mask=mask, transformer_options=transformer_options)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None, device=None, dtype=None, operations=None):
        super().__init__()
        self.mod = DoubleSharedModulation(features, device=device, dtype=dtype, operations=operations)
        self.prenorm = RMSNorm(features, device=device, dtype=dtype, operations=operations)
        self.postnorm = RMSNorm(features, device=device, dtype=dtype, operations=operations)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias, device=device, dtype=dtype, operations=operations)
        self.mlp = SwiGLU(features, multiplier, bias, device=device, dtype=dtype, operations=operations)

    def forward(self, x, vec, freqs, mask=None, transformer_options={}):
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs, mask, transformer_options=transformer_options)
        x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)
        return x


class LastLayer(nn.Module):
    def __init__(self, features, patch, channels, device=None, dtype=None, operations=None):
        super().__init__()
        self.norm = RMSNorm(features, device=device, dtype=dtype, operations=operations)
        self.linear = operations.Linear(features, patch * patch * channels, bias=True, device=device, dtype=dtype)
        self.modulation = SimpleModulation(features, device=device, dtype=dtype, operations=operations)

    def forward(self, x, tvec):
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        return self.linear(x)


class SingleStreamDiT(nn.Module):
    def __init__(self, features=6144, tdim=256, txtdim=2560, heads=48, kvheads=12, multiplier=4,
                 layers=28, patch=2, channels=16, bias=False, theta=1e3, txtlayers=12,
                 txtheads=20, txtkvheads=20, image_model=None,
                 device=None, dtype=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.patch = patch
        self.channels = channels
        self.tdim = tdim
        self.heads = heads
        self.txtdim = txtdim
        self.txtlayers = txtlayers

        headdim = features // heads
        axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
        assert sum(axes) == headdim, f"axes {axes} sum != headdim {headdim}"
        self.pe_embedder = EmbedND(dim=headdim, theta=int(theta), axes_dim=axes)

        self.first = operations.Linear(channels * patch ** 2, features, bias=True, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            SingleStreamBlock(features, heads, multiplier, bias, kvheads, device=device, dtype=dtype, operations=operations)
            for _ in range(layers)
        ])
        self.tmlp = nn.Sequential(
            operations.Linear(tdim, features, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            operations.Linear(features, features, device=device, dtype=dtype),
        )
        self.txtfusion = TextFusionTransformer(txtlayers, txtdim, txtheads, multiplier, bias, txtkvheads,
                                               device=device, dtype=dtype, operations=operations)
        self.txtmlp = nn.Sequential(
            RMSNorm(txtdim, device=device, dtype=dtype, operations=operations),
            operations.Linear(txtdim, features, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            operations.Linear(features, features, device=device, dtype=dtype),
        )
        self.last = LastLayer(features, patch, channels, device=device, dtype=dtype, operations=operations)
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"),
            operations.Linear(features, features * 6, device=device, dtype=dtype),
        )

    def forward(self, x, timesteps, context, attention_mask=None, transformer_options={}, control=None, **kwargs):
        temporal = x.ndim == 5
        if temporal:
            b5, c5, t5, h5, w5 = x.shape
            x = x.reshape(b5 * t5, c5, h5, w5)
        bs, c, H_orig, W_orig = x.shape
        patch = self.patch
        x = pad_to_patch_size(x, (patch, patch))
        H, W = x.shape[-2], x.shape[-1]
        h_, w_ = H // patch, W // patch

        context = self._unpack_context(context)
        context = enhance.maybe_detail_boost(context)   # no-op unless Detail Boost is enabled

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
        img = self.first(img)

        t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
        tvec = self.tproj(t)

        text_mask = None
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape != (bs, context.shape[1]):
                raise ValueError(
                    "Krea2 attention mask shape "
                    f"{tuple(attention_mask.shape)} does not match text shape {(bs, context.shape[1])}"
                )
            text_mask = attention_mask.to(device=context.device, dtype=torch.bool)

        context = self.txtfusion(context, mask=text_mask, transformer_options=transformer_options)
        context = self.txtmlp(context)

        txtlen, imglen = context.shape[1], img.shape[1]
        combined = torch.cat((context, img), dim=1)
        combined_mask = None
        if text_mask is not None:
            image_mask = torch.ones(bs, imglen, device=combined.device, dtype=torch.bool)
            combined_mask = torch.cat((text_mask.to(combined.device), image_mask), dim=1)
            combined_mask = _prepare_attention_mask(
                combined_mask,
                self.heads,
                txtlen + imglen,
                combined.dtype,
                combined.device,
            )

        device = combined.device
        txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
        imgids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
        imgids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
        imgids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
        imgpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
        pos = torch.cat((txtpos, imgpos), dim=1)

        freqs = self.pe_embedder(pos)

        for block in self.blocks:
            combined = block(
                combined,
                tvec,
                freqs,
                combined_mask,
                transformer_options=transformer_options,
            )

        final = self.last(combined, t)
        out = final[:, txtlen:txtlen + imglen, :]
        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                        h=h_, w=w_, ph=patch, pw=patch, c=self.channels)
        out = out[:, :, :H_orig, :W_orig]
        if temporal:
            out = out.reshape(b5, t5, self.channels, H_orig, W_orig).movedim(1, 2)
        return out

    def _unpack_context(self, context):
        b, seq, fused = context.shape
        if fused != self.txtlayers * self.txtdim:
            raise ValueError(
                f"Krea2 expects conditioning with {self.txtlayers}x{self.txtdim}={self.txtlayers * self.txtdim} "
                f"features (a {self.txtlayers}-layer Qwen3-VL stack) but got {fused}. "
                f"Load the Krea2 text encoder (Qwen3-VL-4B with the 12-layer tap)."
            )
        return context.reshape(b, seq, self.txtlayers, self.txtdim)
