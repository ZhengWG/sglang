"""Unit tests for cuLA SM90 KDA prefill kernel.

Compares cuLA kernel output against Triton chunk_kda reference.
"""

import math

import pytest
import torch
import torch.nn.functional as F

# cuLA kernel uses exp2() internally, gates must be in log-base-2 space.
RCP_LN2 = 1.0 / math.log(2.0)

# Skip all tests if not on SM90 GPU
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="cuLA KDA requires SM90 (Hopper) GPU",
)


def _try_import_cula():
    try:
        from sgl_kernel import kda_fwd_prefill

        return kda_fwd_prefill
    except ImportError:
        pytest.skip("sgl_kernel.kda_fwd_prefill not available (cula_kda_ops not built)")


def _try_import_triton_ref():
    try:
        from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
        from sglang.srt.layers.attention.fla.kda import chunk_kda
        from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd

        return chunk_kda, l2norm_fwd, chunk_local_cumsum
    except ImportError:
        pytest.skip("Triton reference (sglang) not available")


def _make_cu_seqlens(batch_size, seq_len, device):
    """Create uniform cu_seqlens for batch_size sequences of seq_len."""
    seqlens = [seq_len] * batch_size
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i, l in enumerate(seqlens):
        cu_seqlens[i + 1] = cu_seqlens[i] + l
    return cu_seqlens


def _run_cula_vs_triton(B, T, H, D, device="cuda"):
    """Run cuLA and Triton KDA, compare outputs."""
    kda_fwd_prefill = _try_import_cula()
    chunk_kda, l2norm_fwd, chunk_local_cumsum = _try_import_triton_ref()

    torch.manual_seed(42)
    packed_seq = B * T

    # Generate inputs in [1, packed_seq, H, D] format (SGLang convention)
    q = torch.randn(1, packed_seq, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, packed_seq, H, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, packed_seq, H, D, dtype=torch.bfloat16, device=device)
    # Gate values: logsigmoid produces values in (-inf, 0), clamp to safe range [-5, 0]
    g = F.logsigmoid(
        torch.randn(1, packed_seq, H, D, dtype=torch.float32, device=device)
    )
    g = g.clamp(-5, 0).to(torch.bfloat16)
    # Beta: sigmoid constrains to (0, 1) range (matches model behavior)
    beta = torch.randn(1, packed_seq, H, dtype=torch.float32, device=device).sigmoid()

    # Initial state in VK layout [N, H, V, K] (SGLang convention, used by both paths)
    initial_state_vk = (
        torch.randn(B, H, D, D, dtype=torch.float32, device=device) * 0.01
    )

    cu_seqlens = _make_cu_seqlens(B, T, device)

    # --- Triton reference ---
    # chunk_kda does its own cumsum internally (base-e, no scale)
    triton_out = chunk_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone().contiguous(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=initial_state_vk.clone(),
        initial_state_indices=torch.arange(B, device=device),
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens.long(),
    )
    # chunk_kda returns (output, chunk-boundary states); keep only the output.
    if isinstance(triton_out, tuple):
        triton_out = triton_out[0]

    # --- cuLA kernel ---
    # Preprocess: l2norm Q, K
    q_norm = l2norm_fwd(q.clone().contiguous())
    k_norm = l2norm_fwd(k.clone().contiguous())

    # Gate cumsum with RCP_LN2 scale (cuLA uses exp2 internally)
    g_cum = chunk_local_cumsum(
        g.clone(), chunk_size=64, scale=RCP_LN2, cu_seqlens=cu_seqlens.long()
    )

    # Reshape for C++ kernel: [packed_seq, H, D]
    q_packed = q_norm.reshape(packed_seq, H, D).contiguous()
    k_packed = k_norm.reshape(packed_seq, H, D).contiguous()
    v_packed = v.reshape(packed_seq, H, D).contiguous()
    g_packed = g_cum.reshape(packed_seq, H, D).contiguous()
    beta_packed = beta.reshape(packed_seq, H).contiguous()

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    workspace = torch.zeros(sm_count * 128, dtype=torch.uint8, device=device)

    scale = D**-0.5

    cula_output, cula_state = kda_fwd_prefill(
        q=q_packed,
        k=k_packed,
        v=v_packed,
        cu_seqlens=cu_seqlens,
        workspace_buffer=workspace,
        scale=scale,
        safe_gate=True,
        input_state=initial_state_vk.clone(),
        alpha=g_packed,
        beta=beta_packed,
    )

    # Reshape cuLA output back to [1, packed_seq, H, D]
    cula_output = cula_output.reshape(1, packed_seq, H, D)

    # Compare outputs
    # Use relaxed tolerance for bf16 + fused kernel differences
    atol = 5e-2
    rtol = 5e-2
    torch.testing.assert_close(cula_output, triton_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "B,T,H,D",
    [
        (1, 63, 1, 128),
        (2, 500, 3, 128),
        (4, 1024, 4, 128),
        (4, 2048, 8, 128),
    ],
)
def test_cula_vs_triton(B, T, H, D):
    _run_cula_vs_triton(B, T, H, D)


def test_cula_varlen():
    """Test with variable-length sequences."""
    kda_fwd_prefill = _try_import_cula()

    torch.manual_seed(42)
    device = "cuda"
    H, D = 4, 128
    seqlens = [63, 128, 256]
    B = len(seqlens)
    packed_seq = sum(seqlens)

    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
    for i, l in enumerate(seqlens):
        cu_seqlens[i + 1] = cu_seqlens[i] + l

    # L2-normalize Q, K (required for numerical stability, matches model usage)
    q = F.normalize(
        torch.randn(packed_seq, H, D, dtype=torch.float32, device=device), p=2, dim=-1
    ).bfloat16()
    k = F.normalize(
        torch.randn(packed_seq, H, D, dtype=torch.float32, device=device), p=2, dim=-1
    ).bfloat16()
    v = torch.randn(packed_seq, H, D, dtype=torch.bfloat16, device=device)
    # Gate values in safe range [-5, 0) (pre-cumsum'd for direct kernel call)
    g = F.logsigmoid(
        torch.randn(packed_seq, H, D, dtype=torch.float32, device=device)
    ).clamp(-5, 0)
    # Beta must be in (0, 1) via sigmoid (matches model behavior)
    beta = torch.randn(packed_seq, H, dtype=torch.float32, device=device).sigmoid()
    initial_state = torch.randn(B, H, D, D, dtype=torch.float32, device=device) * 0.01

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    workspace = torch.zeros(sm_count * 128, dtype=torch.uint8, device=device)

    scale = D**-0.5

    output, output_state = kda_fwd_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        workspace_buffer=workspace,
        scale=scale,
        safe_gate=True,
        input_state=initial_state,
        alpha=g,
        beta=beta,
    )

    # Basic shape checks
    assert output.shape == (packed_seq, H, D)
    assert output_state.shape == (B, H, D, D)
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"


def test_cula_no_initial_state():
    """Test without initial state (should allocate zeros internally)."""
    kda_fwd_prefill = _try_import_cula()

    torch.manual_seed(42)
    device = "cuda"
    B, T, H, D = 2, 256, 4, 128
    packed_seq = B * T

    cu_seqlens = _make_cu_seqlens(B, T, device)
    # L2-normalize Q, K (required for numerical stability, matches model usage)
    q = F.normalize(
        torch.randn(packed_seq, H, D, dtype=torch.float32, device=device), p=2, dim=-1
    ).bfloat16()
    k = F.normalize(
        torch.randn(packed_seq, H, D, dtype=torch.float32, device=device), p=2, dim=-1
    ).bfloat16()
    v = torch.randn(packed_seq, H, D, dtype=torch.bfloat16, device=device)
    # Gate values in safe range [-5, 0)
    g = F.logsigmoid(
        torch.randn(packed_seq, H, D, dtype=torch.float32, device=device)
    ).clamp(-5, 0)
    # Beta must be in (0, 1) via sigmoid
    beta = torch.randn(packed_seq, H, dtype=torch.float32, device=device).sigmoid()

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    workspace = torch.zeros(sm_count * 128, dtype=torch.uint8, device=device)

    scale = D**-0.5

    output, output_state = kda_fwd_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        workspace_buffer=workspace,
        scale=scale,
        safe_gate=True,
        alpha=g,
        beta=beta,
    )

    assert output.shape == (packed_seq, H, D)
    assert output_state.shape == (B, H, D, D)
    assert not torch.isnan(output).any(), "Output contains NaN"


def _try_import_cula_kernel_wrapper():
    try:
        from sglang.srt.layers.attention.linear.kernels.kda_cula import CulaKDAKernel

        return CulaKDAKernel
    except ImportError:
        pytest.skip("CulaKDAKernel (sglang) not available")


def _run_cula_extend_vs_triton_raw_gate(B, T, H, D, use_lower_bound, device="cuda"):
    """End-to-end gate-contract test for the cuLA prefill path.

    Unlike _run_cula_vs_triton (which feeds an already-activated gate, i.e. the
    pre-#730 contract), this drives the *raw* gate through the real
    CulaKDAKernel.extend path -- which must apply the KDA gate activation
    (-exp(A_log)*softplus(g+dt_bias), or the safe-gate form when lower_bound is
    set) via kda_gate_chunk_cumsum -- and compares against Triton chunk_kda fed
    the same raw gate + A_log/dt_bias/lower_bound. Covers the activation that
    PR #730 (#23038) moved into the kernel, which the kernel-only test skips.
    """
    _try_import_cula()  # ensure the cuLA kernel is built/loadable
    chunk_kda, _, _ = _try_import_triton_ref()
    CulaKDAKernel = _try_import_cula_kernel_wrapper()

    torch.manual_seed(0)
    packed_seq = B * T

    q = torch.randn(1, packed_seq, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, packed_seq, H, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, packed_seq, H, D, dtype=torch.bfloat16, device=device)
    # RAW gate: model projection output BEFORE activation (post-#730 contract).
    g_raw = torch.randn(1, packed_seq, H, D, dtype=torch.bfloat16, device=device)
    beta = torch.randn(1, packed_seq, H, dtype=torch.float32, device=device).sigmoid()

    # Per-head gate params (model shapes: A_log [H], dt_bias [H*K]).
    A_log = torch.randn(H, dtype=torch.float32, device=device) * 0.5
    dt_bias = torch.randn(H * D, dtype=torch.float32, device=device) * 0.1
    # Keep |lower_bound| modest. The cuLA chunk kernel evaluates exp2(+/- chunk-
    # local cumsum) over a 64-token chunk; a per-token log-decay floor as deep as
    # -8 makes the cumsum ~-512 and overflows fp32 (exp2(738)). Real KDA
    # safe-gate floors are mild, and the Triton reference (which also does not
    # clamp) shares this constraint, so an extreme floor is not a meaningful case.
    lower_bound = -1.0 if use_lower_bound else None

    init_state = torch.randn(B, H, D, D, dtype=torch.float32, device=device) * 0.01
    cu_seqlens = _make_cu_seqlens(B, T, device)
    cache_indices = torch.arange(B, dtype=torch.int32, device=device)

    # --- cuLA path under test: raw gate -> CulaKDAKernel.extend ---
    kernel = CulaKDAKernel()
    cula_out, _ = kernel.extend(
        q.clone(),
        k.clone(),
        v.clone(),
        g_raw.clone(),
        beta.clone(),
        ssm_states=init_state.clone(),
        cache_indices=cache_indices,
        query_start_loc=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )
    cula_out = cula_out.reshape(1, packed_seq, H, D)

    # --- Triton reference: same raw gate + gate params (ground truth) ---
    triton_out = chunk_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone().contiguous(),
        g=g_raw.clone(),
        beta=beta.clone(),
        initial_state=init_state.clone(),
        initial_state_indices=cache_indices.long(),
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens.long(),
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )
    if isinstance(triton_out, tuple):
        triton_out = triton_out[0]

    torch.testing.assert_close(cula_out, triton_out, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    "use_lower_bound", [False, True], ids=["standard_gate", "safe_gate"]
)
@pytest.mark.parametrize("B,T,H,D", [(1, 128, 4, 128), (2, 512, 4, 128)])
def test_cula_extend_raw_gate_vs_triton(B, T, H, D, use_lower_bound):
    _run_cula_extend_vs_triton_raw_gate(B, T, H, D, use_lower_bound)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
