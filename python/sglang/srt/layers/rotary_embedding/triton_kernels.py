"""Triton JIT kernels for multimodal rotary positional embeddings."""

from __future__ import annotations

from typing import List

import torch
import triton
import triton.language as tl


@triton.jit
def _triton_mrope_forward_fused(
    q_ptr,
    k_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_stride,
    k_stride,
    positions_stride,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
    is_interleaved_glm: tl.constexpr,
    is_neox_style: tl.constexpr,
    axis_map_ptr,
):
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * q_stride
    k_ptr = k_ptr + pid * k_stride
    half_rd = rd // 2
    t = tl.load(positions_ptr + 0 * positions_stride + pid)
    h = tl.load(positions_ptr + 1 * positions_stride + pid)
    w = tl.load(positions_ptr + 2 * positions_stride + pid)
    t_cos = cos_sin_cache_ptr + t * rd
    h_cos = cos_sin_cache_ptr + h * rd
    w_cos = cos_sin_cache_ptr + w * rd
    t_sin = t_cos + half_rd
    h_sin = h_cos + half_rd
    w_sin = w_cos + half_rd
    cos_offsets = tl.arange(0, pad_hd // 2)
    if is_interleaved:
        if is_interleaved_glm:
            axes = tl.load(axis_map_ptr + cos_offsets, mask=cos_offsets < (pad_hd // 2))
            t_mask = axes == 0
            h_mask = axes == 1
            w_mask = axes == 2
        else:
            h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
            w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
            t_mask = ~(h_mask | w_mask)
    else:
        t_end = mrope_section_t
        h_end = t_end + mrope_section_h
        t_mask = cos_offsets < mrope_section_t
        h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
        w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)
    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)
    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row
    if is_neox_style:
        fhq = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
        fhk = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
        fqm = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
            tl.arange(0, pad_hd // 2)[None, :] < rd // 2
        )
        fkm = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
            tl.arange(0, pad_hd // 2)[None, :] < rd // 2
        )
        q1 = tl.load(q_ptr + fhq, mask=fqm, other=0).to(sin_row.dtype)
        k1 = tl.load(k_ptr + fhk, mask=fkm, other=0).to(sin_row.dtype)
        shq = fhq + (rd // 2)
        shk = fhk + (rd // 2)
        q2 = tl.load(q_ptr + shq, mask=fqm, other=0).to(sin_row.dtype)
        k2 = tl.load(k_ptr + shk, mask=fkm, other=0).to(sin_row.dtype)
        tl.store(q_ptr + fhq, q1 * cos_row - q2 * sin_row, mask=fqm)
        tl.store(q_ptr + shq, q2 * cos_row + q1 * sin_row, mask=fqm)
        tl.store(k_ptr + fhk, k1 * cos_row - k2 * sin_row, mask=fkm)
        tl.store(k_ptr + shk, k2 * cos_row + k1 * sin_row, mask=fkm)
    else:
        bq = tl.arange(0, pad_n_qh)[:, None] * hd
        bk = tl.arange(0, pad_n_kh)[:, None] * hd
        ei = 2 * tl.arange(0, pad_hd // 2)[None, :]
        oi = ei + 1
        im = tl.arange(0, pad_hd // 2)[None, :] < (rd // 2)
        qm = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & im
        km = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & im
        qe = tl.load(q_ptr + bq + ei, mask=qm, other=0).to(sin_row.dtype)
        qo = tl.load(q_ptr + bq + oi, mask=qm, other=0).to(sin_row.dtype)
        ke = tl.load(k_ptr + bk + ei, mask=km, other=0).to(sin_row.dtype)
        ko = tl.load(k_ptr + bk + oi, mask=km, other=0).to(sin_row.dtype)
        tl.store(q_ptr + bq + ei, qe * cos_row - qo * sin_row, mask=qm)
        tl.store(q_ptr + bq + oi, qo * cos_row + qe * sin_row, mask=qm)
        tl.store(k_ptr + bk + ei, ke * cos_row - ko * sin_row, mask=km)
        tl.store(k_ptr + bk + oi, ko * cos_row + ke * sin_row, mask=km)


def triton_mrope_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    mrope_section: List[int],
    head_size: int,
    rotary_dim: int,
    mrope_interleaved: bool,
    mrope_interleaved_glm: bool,
    is_neox_style: bool,
    axis_map: torch.Tensor,
) -> None:
    num_tokens, n_q_dim = q.shape
    n_k_dim = k.shape[1]
    n_qh = n_q_dim // head_size
    n_kh = n_k_dim // head_size
    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)
    pad_hd = triton.next_power_of_2(head_size)
    _triton_mrope_forward_fused[(num_tokens,)](
        q,
        k,
        cos_sin_cache,
        positions,
        q.stride(0),
        k.stride(0),
        positions.stride(0),
        n_qh,
        n_kh,
        head_size,
        rotary_dim,
        pad_n_qh,
        pad_n_kh,
        pad_hd,
        mrope_section[0],
        mrope_section[1],
        mrope_section[2],
        mrope_interleaved,
        mrope_interleaved_glm,
        is_neox_style,
        axis_map,
    )


@triton.jit
def _triton_ernie45_rope_qk_fused(
    q_ptr,
    k_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_stride0: tl.constexpr,
    k_stride0: tl.constexpr,
    pos_stride0: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    section_hw: tl.constexpr,
    is_neox_style: tl.constexpr,
):
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * q_stride0
    k_ptr = k_ptr + pid * k_stride0
    half_rd = rd // 2
    tpos = tl.load(positions_ptr + 0 * pos_stride0 + pid).to(tl.int32)
    hpos = tl.load(positions_ptr + 1 * pos_stride0 + pid).to(tl.int32)
    wpos = tl.load(positions_ptr + 2 * pos_stride0 + pid).to(tl.int32)
    ridx = tl.arange(0, pad_hd // 2)
    rmask = ridx < half_rd
    use_hw = ridx < section_hw
    use_h = (ridx & 1) == 0
    pos = tl.where(use_hw, tl.where(use_h, hpos, wpos), tpos)
    cos = tl.load(cos_sin_cache_ptr + pos * rd + ridx, mask=rmask, other=0.0)
    sin = tl.load(
        cos_sin_cache_ptr + pos * rd + (ridx + half_rd), mask=rmask, other=0.0
    )
    if is_neox_style:
        qh = tl.arange(0, pad_n_qh)[:, None]
        kh = tl.arange(0, pad_n_kh)[:, None]
        d = tl.arange(0, pad_hd // 2)[None, :]
        qm = (qh < n_qh) & (d < half_rd)
        km = (kh < n_kh) & (d < half_rd)
        qo0 = qh * hd + d
        ko0 = kh * hd + d
        qo1 = qo0 + half_rd
        ko1 = ko0 + half_rd
        q0 = tl.load(q_ptr + qo0, mask=qm, other=0.0).to(cos.dtype)
        q1 = tl.load(q_ptr + qo1, mask=qm, other=0.0).to(cos.dtype)
        k0 = tl.load(k_ptr + ko0, mask=km, other=0.0).to(cos.dtype)
        k1 = tl.load(k_ptr + ko1, mask=km, other=0.0).to(cos.dtype)
        cb = cos[None, :]
        sb = sin[None, :]
        tl.store(q_ptr + qo0, q0 * cb - q1 * sb, mask=qm)
        tl.store(q_ptr + qo1, q1 * cb + q0 * sb, mask=qm)
        tl.store(k_ptr + ko0, k0 * cb - k1 * sb, mask=km)
        tl.store(k_ptr + ko1, k1 * cb + k0 * sb, mask=km)
    else:
        qh = tl.arange(0, pad_n_qh)[:, None]
        kh = tl.arange(0, pad_n_kh)[:, None]
        p = tl.arange(0, pad_hd // 2)[None, :]
        qm = (qh < n_qh) & (p < half_rd)
        km = (kh < n_kh) & (p < half_rd)
        even = 2 * p
        odd = even + 1
        qe = tl.load(q_ptr + qh * hd + even, mask=qm, other=0.0).to(cos.dtype)
        qo = tl.load(q_ptr + qh * hd + odd, mask=qm, other=0.0).to(cos.dtype)
        ke = tl.load(k_ptr + kh * hd + even, mask=km, other=0.0).to(cos.dtype)
        ko = tl.load(k_ptr + kh * hd + odd, mask=km, other=0.0).to(cos.dtype)
        cb = cos[None, :]
        sb = sin[None, :]
        tl.store(q_ptr + qh * hd + even, qe * cb - qo * sb, mask=qm)
        tl.store(q_ptr + qh * hd + odd, qo * cb + qe * sb, mask=qm)
        tl.store(k_ptr + kh * hd + even, ke * cb - ko * sb, mask=km)
        tl.store(k_ptr + kh * hd + odd, ko * cb + ke * sb, mask=km)


def triton_ernie45_rope_fused_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    mrope_section: list,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    num_tokens = q.shape[0]
    n_qh = q.shape[1] // head_size
    n_kh = k.shape[1] // head_size
    rd = rotary_dim
    section_h, section_w, section_t = mrope_section
    assert section_h == section_w, "Ernie4.5 layout assumes section_h == section_w"
    assert section_h + section_w + section_t == rd // 2
    if cos_sin_cache.dtype != q.dtype or cos_sin_cache.device != q.device:
        cos_sin_cache = cos_sin_cache.to(device=q.device, dtype=q.dtype)
    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)
    pad_hd = triton.next_power_of_2(head_size)
    num_warps = 4 if (pad_n_qh * pad_hd) <= 8192 else 8
    _triton_ernie45_rope_qk_fused[(num_tokens,)](
        q,
        k,
        cos_sin_cache,
        positions,
        q.stride(0),
        k.stride(0),
        positions.stride(0),
        n_qh=n_qh,
        n_kh=n_kh,
        hd=head_size,
        rd=rd,
        pad_n_qh=pad_n_qh,
        pad_n_kh=pad_n_kh,
        pad_hd=pad_hd,
        section_hw=section_h + section_w,
        is_neox_style=is_neox_style,
        num_warps=num_warps,
    )


@triton.jit
def _triton_vision_rope_qk_inplace_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    q_stride_t,
    k_stride_t,
    cos_stride_t,
    sin_stride_t,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd_half: tl.constexpr,
):
    """Apply NEOX-style (rotate_half) rotary embedding to q,k in-place.

    Layout:
        q: (num_tokens, n_qh, hd) contiguous in last two dims with stride
           q_stride_t between tokens.
        k: (num_tokens, n_kh, hd) likewise.
        cos/sin: (num_tokens, hd). The two halves of cos/sin may be equal
        (when constructed from `cat([c, c], dim=-1)` upstream) or different
        (e.g., HF position_embeddings); both are supported.

    Performs the standard formula:
        q1' = q1 * cos[:hd//2] - q2 * sin[:hd//2]
        q2' = q2 * cos[hd//2:] + q1 * sin[hd//2:]
    in float for numerical parity with `apply_rotary_pos_emb_native`.
    """
    pid = tl.program_id(0)
    q_row = q_ptr + pid * q_stride_t
    k_row = k_ptr + pid * k_stride_t
    cos_row = cos_ptr + pid * cos_stride_t
    sin_row = sin_ptr + pid * sin_stride_t

    half_hd = hd // 2

    d = tl.arange(0, pad_hd_half)
    dmask = d < half_hd

    cos1 = tl.load(cos_row + d, mask=dmask, other=0.0).to(tl.float32)
    cos2 = tl.load(cos_row + half_hd + d, mask=dmask, other=0.0).to(tl.float32)
    sin1 = tl.load(sin_row + d, mask=dmask, other=0.0).to(tl.float32)
    sin2 = tl.load(sin_row + half_hd + d, mask=dmask, other=0.0).to(tl.float32)

    qh = tl.arange(0, pad_n_qh)[:, None]
    dd = d[None, :]
    qmask = (qh < n_qh) & (dd < half_hd)
    q_lo_off = qh * hd + dd
    q_hi_off = q_lo_off + half_hd

    q1 = tl.load(q_row + q_lo_off, mask=qmask, other=0.0).to(tl.float32)
    q2 = tl.load(q_row + q_hi_off, mask=qmask, other=0.0).to(tl.float32)
    q1_new = q1 * cos1[None, :] - q2 * sin1[None, :]
    q2_new = q2 * cos2[None, :] + q1 * sin2[None, :]
    tl.store(q_row + q_lo_off, q1_new, mask=qmask)
    tl.store(q_row + q_hi_off, q2_new, mask=qmask)

    kh = tl.arange(0, pad_n_kh)[:, None]
    kmask = (kh < n_kh) & (dd < half_hd)
    k_lo_off = kh * hd + dd
    k_hi_off = k_lo_off + half_hd

    k1 = tl.load(k_row + k_lo_off, mask=kmask, other=0.0).to(tl.float32)
    k2 = tl.load(k_row + k_hi_off, mask=kmask, other=0.0).to(tl.float32)
    k1_new = k1 * cos1[None, :] - k2 * sin1[None, :]
    k2_new = k2 * cos2[None, :] + k1 * sin2[None, :]
    tl.store(k_row + k_lo_off, k1_new, mask=kmask)
    tl.store(k_row + k_hi_off, k2_new, mask=kmask)


def triton_vision_rope_qk_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> None:
    """CUDA-graph-safe vision NEOX RoPE applied in-place to q and k.

    Replaces ``apply_rotary_pos_emb_native`` for the ViT path. The native
    implementation is wrapped in ``torch.compile(dynamic=True)``, which on
    capture-time injects dynamic-shape guards / RNG state that conflicts with
    ``torch.cuda.graph`` capture (manifesting as ``random_rng`` errors).

    Accepted shapes:
        q: (num_tokens, n_qh, head_size) — float16/bfloat16/float32
        k: (num_tokens, n_kh, head_size)
        cos: (num_tokens, head_size)
        sin: (num_tokens, head_size)

    The kernel reads cos/sin in fp32 and accumulates in fp32, matching the
    semantics of ``apply_rotary_pos_emb_native``.
    """
    assert q.dim() == 3 and k.dim() == 3, (q.shape, k.shape)
    num_tokens, n_qh, head_size = q.shape
    assert k.shape[0] == num_tokens and k.shape[2] == head_size, (q.shape, k.shape)
    n_kh = k.shape[1]
    assert cos.shape[-1] == head_size and sin.shape[-1] == head_size, (
        cos.shape,
        sin.shape,
        head_size,
    )
    assert head_size % 2 == 0, head_size
    # q / k are written in-place; require contiguous storage in the last two dims
    # so the (head, dim) plane has stride (head_size, 1).
    assert q.stride(-1) == 1 and q.stride(-2) == head_size, q.stride()
    assert k.stride(-1) == 1 and k.stride(-2) == head_size, k.stride()
    assert cos.stride(-1) == 1 and sin.stride(-1) == 1, (cos.stride(), sin.stride())

    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)
    pad_hd_half = triton.next_power_of_2(head_size // 2)
    num_warps = 4 if (pad_n_qh * pad_hd_half) <= 4096 else 8

    _triton_vision_rope_qk_inplace_kernel[(num_tokens,)](
        q,
        k,
        cos,
        sin,
        q.stride(0),
        k.stride(0),
        cos.stride(0),
        sin.stride(0),
        n_qh=n_qh,
        n_kh=n_kh,
        hd=head_size,
        pad_n_qh=pad_n_qh,
        pad_n_kh=pad_n_kh,
        pad_hd_half=pad_hd_half,
        num_warps=num_warps,
    )
