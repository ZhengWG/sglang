"""Primitive rotary embedding ops: _rotate_neox, _rotate_gptj, _apply_rotary_emb,
apply_rotary_pos_emb variants."""

from __future__ import annotations

from typing import Tuple

import torch

from sglang.srt.utils import (
    cpu_has_amx_support,
    get_compiler_backend,
    is_cpu,
    is_cuda,
    is_npu,
)

_is_npu = is_npu()
_is_cpu = is_cpu()
_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()

if _is_npu:
    import torch_npu

    NPU_ROTARY_MUL_MAX_NUM_HEADS = 1000
    NPU_ROTARY_MUL_MAX_HEAD_SIZE = 896


def rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


# Copied from transformers
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.compile(dynamic=True, backend=get_compiler_backend())
def apply_rotary_pos_emb_native(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim=1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()

    # embedding is performed in float
    cos = cos.unsqueeze(unsqueeze_dim).float()
    sin = sin.unsqueeze(unsqueeze_dim).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)

    return q_embed, k_embed


def apply_rotary_pos_emb_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim=1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ascend implementation equivalent to apply_rotary_pos_emb_native.

    Args:
        q: [num_tokens, num_heads, head_size]
        k: [num_tokens, num_kv_heads, head_size]
        cos: [num_tokens, head_size]
        sin: [num_tokens, head_size]
    """
    if (
        cos.dim() != 2
        or q.dim() != 3
        or q.shape[1] >= NPU_ROTARY_MUL_MAX_NUM_HEADS
        or q.shape[2] >= NPU_ROTARY_MUL_MAX_HEAD_SIZE
    ):
        # Note: num_heads and head_size of q must be less than 1000 and 896, respectively
        return apply_rotary_pos_emb_native(q, k, cos, sin, unsqueeze_dim)
    cos = cos.unsqueeze(unsqueeze_dim).unsqueeze(0)
    sin = sin.unsqueeze(unsqueeze_dim).unsqueeze(0)
    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    q_embed = q_embed.squeeze(0)
    k_embed = k_embed.squeeze(0)
    return q_embed, k_embed


def apply_rotary_pos_emb_cuda_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CUDA-graph-safe ViT/LM RoPE via a Triton kernel (no torch.compile).

    Semantics match :func:`apply_rotary_pos_emb_native` for the common ViT
    layout where ``q`` is ``[num_tokens, num_heads, head_size]`` and ``cos``,
    ``sin`` are ``[num_tokens, head_size]``. Falls back to the native
    implementation for less common layouts (e.g. ``unsqueeze_dim != 1``,
    4D ``q``/``k`` with a leading batch axis, mismatched dtypes that the
    kernel does not own).

    The kernel writes ``q`` and ``k`` in-place when their dtype is already
    compatible; this both avoids extra allocations during CUDA-graph capture
    and prevents the dynamic shape / RNG state that ``torch.compile`` would
    otherwise inject.
    """
    # Lazy import to avoid pulling Triton on CPU/NPU paths at module import.
    from sglang.srt.layers.rotary_embedding.triton_kernels import (
        triton_vision_rope_qk_inplace,
    )

    # Only handle the ViT / "flat token" layout in-kernel. Anything else
    # (e.g., LM-side 4D [B, S, H, D] with unsqueeze_dim=1, broadcasted cos/sin,
    # or non-2D cos/sin) is safely delegated to the native (torch.compile'd)
    # implementation which is *not* used inside cuda-graph capture.
    if (
        unsqueeze_dim == 1
        and q.dim() == 3
        and k.dim() == 3
        and cos.dim() == 2
        and sin.dim() == 2
        and cos.shape[0] == q.shape[0]
        and cos.shape[-1] == q.shape[-1]
        and q.dtype == k.dtype
        and q.is_cuda
        and q.stride(-1) == 1
        and q.stride(-2) == q.shape[-1]
        and k.stride(-1) == 1
        and k.stride(-2) == k.shape[-1]
    ):
        triton_vision_rope_qk_inplace(q, k, cos, sin)
        return q, k

    return apply_rotary_pos_emb_native(q, k, cos, sin, unsqueeze_dim)


if _is_npu:
    apply_rotary_pos_emb = apply_rotary_pos_emb_npu
elif _is_cpu and _is_cpu_amx_available:
    apply_rotary_pos_emb = torch.ops.sgl_kernel.apply_rotary_pos_emb_cpu
elif _is_cuda:
    # CUDA path: prefer the Triton kernel so that the operator is safe to
    # capture inside torch.cuda.graph (used by ViT CUDA Graph Runner /
    # encode_server for Qwen3-VL, Qwen3.5, Qwen2.5-VL, etc.). The previous
    # default `apply_rotary_pos_emb_native` is `torch.compile(dynamic=True)`-
    # wrapped and conflicts with cuda-graph capture (manifests as random_rng
    # / dynamic-shape guard failures).
    apply_rotary_pos_emb = apply_rotary_pos_emb_cuda_triton
else:
    apply_rotary_pos_emb = apply_rotary_pos_emb_native
