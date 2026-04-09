import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from backend import attention


@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    head_dim: int = 128
    qkv_bias: bool = False
    q_norm: Optional[str] = "gemma3"
    k_norm: Optional[str] = "gemma3"
    mlp_activation: str = "silu"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        norm = torch.rsqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps).to(x.dtype)
        return x * norm * self.weight.to(x)


def rotate_half(x: torch.Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = (xq * cos) + (rotate_half(xq) * sin)
    k = (xk * cos) + (rotate_half(xk) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, seq_len: int, batch_size: int, device: torch.device, dtype: torch.dtype):
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        freqs = torch.einsum("bi,j->bij", position_ids.float(), self.inv_freq.float().to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, device=None, dtype=None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.inner_size = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, self.inner_size, bias=config.qkv_bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.o_proj = nn.Linear(self.inner_size, config.hidden_size, bias=False, device=device, dtype=dtype)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype) if config.q_norm == "gemma3" else None
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype) if config.k_norm == "gemma3" else None

    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor], cos: torch.Tensor, sin: torch.Tensor):
        b, t, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q, k = apply_rope(q, k, cos, sin)

        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        out = attention.attention_function(q, k, v, self.num_heads, mask=mask, skip_reshape=True)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config, device=None, dtype=None):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype)
        self.activation = torch.nn.functional.silu if config.mlp_activation == "silu" else torch.nn.functional.gelu

    def forward(self, x: torch.Tensor):
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Block(nn.Module):
    def __init__(self, config: Qwen3Config, device=None, dtype=None):
        super().__init__()
        self.self_attn = Qwen3Attention(config, device=device, dtype=dtype)
        self.mlp = Qwen3MLP(config, device=device, dtype=dtype)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor], cos: torch.Tensor, sin: torch.Tensor):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask=mask, cos=cos, sin=sin)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class IntegratedQwen3Model(nn.Module):
    def __init__(self, config: dict, device=None, dtype=None):
        super().__init__()

        cfg = dict(config or {})
        cfg.setdefault("q_norm", "gemma3")
        cfg.setdefault("k_norm", "gemma3")
        cfg.setdefault("qkv_bias", cfg.get("attention_bias", False))
        cfg.setdefault("mlp_activation", cfg.get("hidden_activation", cfg.get("hidden_act", "silu")))
        self.config = Qwen3Config(**{k: v for k, v in cfg.items() if k in Qwen3Config.__annotations__})

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList([Qwen3Block(self.config, device=device, dtype=dtype) for _ in range(self.config.num_hidden_layers)])
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, device=device, dtype=dtype)
        self.rotary_emb = RotaryEmbedding(self.config.head_dim, self.config.rope_theta)

    @property
    def dtype(self):
        return self.embed_tokens.weight.dtype

    def _build_mask(self, attention_mask: Optional[torch.Tensor], seq_len: int, dtype: torch.dtype, device: torch.device):
        neg_inf = torch.finfo(dtype).min / 4
        mask = None

        if attention_mask is not None:
            b = attention_mask.shape[0]
            pad_mask = 1.0 - attention_mask.to(dtype).reshape(b, 1, 1, attention_mask.shape[-1]).expand(b, 1, seq_len, attention_mask.shape[-1])
            mask = pad_mask.masked_fill(pad_mask.to(torch.bool), neg_inf)

        if seq_len > 1:
            causal = torch.empty(seq_len, seq_len, dtype=dtype, device=device).fill_(neg_inf).triu_(1)
            mask = causal if mask is None else mask + causal

        return mask

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, output_hidden_states: bool = False):
        x = self.embed_tokens(input_ids)
        b, t, _ = x.shape
        mask = self._build_mask(attention_mask, t, x.dtype, x.device)
        cos, sin = self.rotary_emb(seq_len=t, batch_size=b, device=x.device, dtype=x.dtype)

        hidden_states = [x] if output_hidden_states else None
        for layer in self.layers:
            x = layer(x, mask=mask, cos=cos, sin=sin)
            if output_hidden_states:
                hidden_states.append(x)

        x = self.norm(x)
        if output_hidden_states:
            hidden_states.append(x)

        return BaseModelOutput(
            last_hidden_state=x,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
            attentions=None,
        )
