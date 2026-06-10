# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/fused_norm_gate.py
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import torch
import torch.nn as nn
import triton
import triton.language as tl

from sglang.srt.utils import (
    cdiv,
    cpu_has_amx_support,
    is_cpu,
    is_npu,
    next_power_of_2,
)

_is_npu = is_npu()
_use_cpu = is_cpu() and cpu_has_amx_support()

# Maximum rows per Triton block for layernorm gated kernel
MAX_ROWS_PER_BLOCK = 4


@triton.jit
def layer_norm_gated_fwd_kernel(
    x,  # pointer to the input
    g,  # pointer to the gate
    y,  # pointer to the output
    w,  # pointer to the weights
    b,  # pointer to the biases
    residual,  # pointer to the residual
    residual_out,  # pointer to the residual
    mean,  # pointer to the mean
    rstd,  # pointer to the 1/std
    eps,  # epsilon to avoid division by zero
    T,  # number of rows in x
    G_OUTER,  # number of outer rows in g
    D: tl.constexpr,  # number of columns in x
    BT: tl.constexpr,
    BD: tl.constexpr,
    G_NUM_HEADS: tl.constexpr,
    G_BLOCK_OUTER: tl.constexpr,
    G_STRIDE_0: tl.constexpr,
    G_STRIDE_1: tl.constexpr,
    G_STRIDE_2: tl.constexpr,
    USE_3D_G_BLOCK: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0)

    o_d = tl.arange(0, BD)
    m_d = o_d < D

    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    if HAS_RESIDUAL:
        p_res = tl.make_block_ptr(
            residual, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
        )
        b_x += tl.load(p_res, boundary_check=(0, 1)).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        p_res_out = tl.make_block_ptr(
            residual_out, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
        )
        tl.store(p_res_out, b_x.to(p_res_out.dtype.element_ty), boundary_check=(0, 1))
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=1) / D
        p_mean = tl.make_block_ptr(mean, (T,), (1,), (i_t * BT,), (BT,), (0,))
        tl.store(p_mean, b_mean.to(p_mean.dtype.element_ty), boundary_check=(0,))
        b_xbar = tl.where(m_d[None, :], b_x - b_mean[:, None], 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    else:
        b_xbar = tl.where(m_d[None, :], b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)

    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (
        (b_x - b_mean[:, None]) * b_rstd[:, None]
        if not IS_RMS_NORM
        else b_x * b_rstd[:, None]
    )
    b_y = b_x_hat * b_w[None, :] if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b[None, :]

    # swish/sigmoid output gate. Compact 2D gates keep a 2D block pointer load;
    # strided 3D gates are loaded as [outer, head, D] blocks and flattened.
    if G_NUM_HEADS == 1:
        if G_STRIDE_0 == D and G_STRIDE_2 == 1:
            p_g = tl.make_block_ptr(g, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
        else:
            p_g = tl.make_block_ptr(
                g,
                (T, D),
                (G_STRIDE_0, G_STRIDE_2),
                (i_t * BT, 0),
                (BT, BD),
                (1, 0),
            )
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    elif USE_3D_G_BLOCK:
        p_g = tl.make_block_ptr(
            g,
            (G_OUTER, G_NUM_HEADS, D),
            (G_STRIDE_0, G_STRIDE_1, G_STRIDE_2),
            (i_t * G_BLOCK_OUTER, 0, 0),
            (G_BLOCK_OUTER, G_NUM_HEADS, BD),
            (2, 1, 0),
        )
        b_g = tl.load(p_g, boundary_check=(0, 1, 2), padding_option="zero").to(
            tl.float32
        )
        b_g = b_g.reshape((BT, BD))
    else:
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        g_outer = o_t // G_NUM_HEADS
        g_inner = o_t - g_outer * G_NUM_HEADS
        p_g = (
            g
            + g_outer[:, None] * G_STRIDE_0
            + g_inner[:, None] * G_STRIDE_1
            + o_d[None, :] * G_STRIDE_2
        )
        b_g = tl.load(p_g, mask=m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == "sigmoid":
        b_y = b_y * tl.sigmoid(b_g)

    # Write output
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def layer_norm_gated_fwd_kernel1(
    x,  # pointer to the input
    g,  # pointer to the gate
    y,  # pointer to the output
    w,  # pointer to the weights
    b,  # pointer to the biases
    residual,  # pointer to the residual
    residual_out,  # pointer to the residual
    mean,  # pointer to the mean
    rstd,  # pointer to the 1/std
    eps,  # epsilon to avoid division by zero
    D: tl.constexpr,  # number of columns in x
    BD: tl.constexpr,
    G_NUM_HEADS: tl.constexpr,
    G_STRIDE_0: tl.constexpr,
    G_STRIDE_1: tl.constexpr,
    G_STRIDE_2: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    if HAS_RESIDUAL:
        residual += i_t * D
    if STORE_RESIDUAL_OUT:
        residual_out += i_t * D

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    b_x = tl.load(x + o_d, mask=m_d, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        b_x += tl.load(residual + o_d, mask=m_d, other=0.0).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        tl.store(residual_out + o_d, b_x, mask=m_d)
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=0) / D
        tl.store(mean + i_t, b_mean)
        b_xbar = tl.where(m_d, b_x - b_mean, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    else:
        b_xbar = tl.where(m_d, b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    tl.store(rstd + i_t, b_rstd)

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b

    # swish/sigmoid output gate. Compact 2D gates keep the original row load;
    # strided 3D gates use logical [outer, head, D] indexing.
    if G_NUM_HEADS == 1:
        if G_STRIDE_0 == D and G_STRIDE_2 == 1:
            p_g = g + i_t * D + o_d
        else:
            p_g = g + i_t * G_STRIDE_0 + o_d * G_STRIDE_2
    else:
        g_outer = i_t // G_NUM_HEADS
        g_inner = i_t - g_outer * G_NUM_HEADS
        p_g = g + g_outer * G_STRIDE_0 + g_inner * G_STRIDE_1 + o_d * G_STRIDE_2
    b_g = tl.load(p_g, mask=m_d, other=0.0).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == "sigmoid":
        b_y = b_y * tl.sigmoid(b_g)

    # Write output
    tl.store(y + o_d, b_y, mask=m_d)


def layer_norm_gated_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    eps: float = 1e-5,
    residual: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    residual_dtype: torch.dtype = None,
    is_rms_norm: bool = False,
    g_num_heads: int = 1,
    g_stride_0: int | None = None,
    g_stride_1: int | None = None,
    g_stride_2: int | None = None,
):
    if residual is not None:
        residual_dtype = residual.dtype
    T, D = x.shape
    if g_stride_0 is None:
        g_stride_0 = g.stride(0)
    if g_stride_1 is None:
        g_stride_1 = 0
    if g_stride_2 is None:
        g_stride_2 = g.stride(-1)
    assert T % g_num_heads == 0
    if residual is not None:
        assert residual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (D,)
    if bias is not None:
        assert bias.shape == (D,)
    # allocate output
    y = x if out_dtype is None else torch.empty_like(x, dtype=out_dtype)
    if residual is not None or (
        residual_dtype is not None and residual_dtype != x.dtype
    ):
        residual_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = (
        torch.empty((T,), dtype=torch.float, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps

    if D <= 512:
        use_3d_g_block = g_num_heads > 1 and g_num_heads == next_power_of_2(g_num_heads)
        if use_3d_g_block:
            g_block_outer = max(1, 32 // g_num_heads)
            BT = g_block_outer * g_num_heads
        else:
            BT = 32
            g_block_outer = 1
        g_outer = T // g_num_heads
        layer_norm_gated_fwd_kernel[(cdiv(T, BT),)](
            x=x,
            g=g,
            y=y,
            w=weight,
            b=bias,
            residual=residual,
            residual_out=residual_out,
            mean=mean,
            rstd=rstd,
            eps=eps,
            T=T,
            G_OUTER=g_outer,
            D=D,
            BD=BD,
            BT=BT,
            G_NUM_HEADS=g_num_heads,
            G_BLOCK_OUTER=g_block_outer,
            G_STRIDE_0=g_stride_0,
            G_STRIDE_1=g_stride_1,
            G_STRIDE_2=g_stride_2,
            USE_3D_G_BLOCK=use_3d_g_block,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
            STORE_RESIDUAL_OUT=residual_out is not None,
            HAS_RESIDUAL=residual is not None,
            HAS_WEIGHT=weight is not None,
            HAS_BIAS=bias is not None,
            num_warps=4,
        )
    else:
        layer_norm_gated_fwd_kernel1[(T,)](
            x=x,
            g=g,
            y=y,
            w=weight,
            b=bias,
            residual=residual,
            residual_out=residual_out,
            mean=mean,
            rstd=rstd,
            eps=eps,
            D=D,
            BD=BD,
            G_NUM_HEADS=g_num_heads,
            G_STRIDE_0=g_stride_0,
            G_STRIDE_1=g_stride_1,
            G_STRIDE_2=g_stride_2,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
            STORE_RESIDUAL_OUT=residual_out is not None,
            HAS_RESIDUAL=residual is not None,
            HAS_WEIGHT=weight is not None,
            HAS_BIAS=bias is not None,
            num_warps=4,
        )
    # residual_out is None if residual is None and residual_dtype == input_dtype
    return y, mean, rstd, residual_out if residual_out is not None else x


class LayerNormGatedFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        activation: str,
        residual: torch.Tensor | None = None,
        eps: float = 1e-6,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
        is_rms_norm: bool = False,
    ):
        x_shape_og = x.shape
        g_shape_og = g.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if g.ndim == 2:
            assert g.shape == x.shape
            g_num_heads = 1
            g_stride_0 = g.stride(0)
            g_stride_1 = 0
            g_stride_2 = g.stride(1)
        else:
            assert g.shape[-1] == x.shape[-1]
            assert g.numel() == x.numel()
            while g.ndim > 3 and g.shape[0] == 1:
                g = g.squeeze(0)
            if g.is_contiguous():
                g = g.reshape(-1, g.shape[-1])
                g_num_heads = 1
                g_stride_0 = g.stride(0)
                g_stride_1 = 0
                g_stride_2 = g.stride(1)
            elif g.ndim == 3:
                g_num_heads = g.shape[-2]
                g_stride_0 = g.stride(-3)
                g_stride_1 = g.stride(-2)
                g_stride_2 = g.stride(-1)
            else:
                g = g.reshape(-1, g.shape[-1])
                g_num_heads = 1
                g_stride_0 = g.stride(0)
                g_stride_1 = 0
                g_stride_2 = g.stride(1)
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float if residual_in_fp32 else None)
        )
        y, mean, rstd, residual_out = layer_norm_gated_fwd(
            x=x,
            g=g,
            weight=weight,
            bias=bias,
            activation=activation,
            eps=eps,
            residual=residual,
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
            g_num_heads=g_num_heads,
            g_stride_0=g_stride_0,
            g_stride_1=g_stride_1,
            g_stride_2=g_stride_2,
        )
        ctx.save_for_backward(residual_out, g, weight, bias, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.g_shape_og = g_shape_og
        ctx.activation = activation
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        y = y.reshape(x_shape_og)
        return y if not prenorm else (y, residual_out.reshape(x_shape_og))


def rms_norm_gated(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    residual: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    eps: float = 1e-6,
):
    return LayerNormGatedFunction.apply(
        x,
        g,
        weight,
        bias,
        activation,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        True,
    )


class FusedRMSNormGated(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        activation: str = "swish",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.activation = activation

        if self.activation not in ["swish", "silu", "sigmoid"]:
            raise ValueError(f"Unsupported activation: {self.activation}")

        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor:
        if _use_cpu:
            assert (
                self.activation == "silu"
            ), "CPU rmsnorm_gated currently only supports activation silu"
            return torch.ops.sgl_kernel.fused_rmsnorm_gated_cpu(
                x, self.weight, g, self.eps
            )
        else:
            return rms_norm_gated(
                x,
                g,
                self.weight,
                self.bias,
                self.activation,
                residual=residual,
                eps=self.eps,
                prenorm=prenorm,
                residual_in_fp32=residual_in_fp32,
            )
