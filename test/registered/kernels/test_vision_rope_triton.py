"""Tests for the CUDA-graph-safe vision RoPE Triton kernel.

These cover:
1. Numerical parity between ``triton_apply_rotary_pos_emb`` and the existing
   ``apply_rotary_pos_emb_native`` (the torch.compile'd reference) across
   fp16 / bf16 / fp32 and several head/dim configurations.
2. Numerical parity between ``apply_rotary_pos_emb_eager`` and
   ``apply_rotary_pos_emb_native`` (these must be bit-equivalent in fp32 since
   they implement identical math).
3. ``torch.cuda.CUDAGraph`` capture + replay correctness for the Triton
   wrapper. This is the headline scenario: the existing native rope cannot
   be captured (the well-known ``random_rng`` failure caused by the
   ``@torch.compile`` decorator); the Triton kernel must capture cleanly
   and produce numerically identical replays.
4. Negative test: 4D q/k (LM-side callers like Gemma3) must raise so a
   global swap of the rope op cannot silently break those paths.

All GPU-only tests are guarded by ``torch.cuda.is_available()``; the parity
test between native and eager runs on CPU as well.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.layers.rotary_embedding import (
    apply_rotary_pos_emb_eager,
)
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_pos_emb_native
from sglang.test.ci.ci_register import register_cuda_ci

# Quick kernel-level test: parity + tiny CUDA-graph capture. Estimated runtime
# is well under 10s on a single GPU.
register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-large")

CUDA = torch.cuda.is_available()
SHAPES = [
    # (N, H_q, H_kv, D)
    (1, 1, 1, 64),
    (16, 12, 4, 128),
    (1024, 16, 8, 80),
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
TOL = {torch.float32: 1e-5, torch.float16: 2e-2, torch.bfloat16: 2e-2}


def _make_inputs(N, H_q, H_kv, D, dtype, device):
    torch.manual_seed(N * 1000 + H_q * 31 + H_kv * 7 + D)
    q = torch.randn(N, H_q, D, dtype=dtype, device=device)
    k = torch.randn(N, H_kv, D, dtype=dtype, device=device)
    # cos/sin generated in fp32 then cast — matches the production path
    # where rot_pos_emb returns fp32 cached values that get cast on use.
    cos = torch.randn(N, D, dtype=torch.float32, device=device).to(dtype)
    sin = torch.randn(N, D, dtype=torch.float32, device=device).to(dtype)
    return q, k, cos, sin


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_eager_vs_native_parity_cpu(shape, dtype):
    """Eager fallback must match the native (compiled) implementation.

    Runs on CPU so the test is meaningful even on a machine without GPU.
    """
    N, H_q, H_kv, D = shape
    q, k, cos, sin = _make_inputs(N, H_q, H_kv, D, dtype, device="cpu")
    q_ref, k_ref = apply_rotary_pos_emb_native(q.clone(), k.clone(), cos, sin)
    q_eag, k_eag = apply_rotary_pos_emb_eager(q.clone(), k.clone(), cos, sin)
    atol = TOL[dtype]
    assert torch.allclose(q_ref, q_eag, atol=atol, rtol=atol), (
        f"q parity failed for {shape}/{dtype}: "
        f"max diff {(q_ref.float()-q_eag.float()).abs().max().item():.3e}"
    )
    assert torch.allclose(k_ref, k_eag, atol=atol, rtol=atol)


@pytest.mark.skipif(not CUDA, reason="requires CUDA + Triton")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_vs_native_parity_cuda(shape, dtype):
    """Triton vision-rope kernel must match the native implementation."""
    from sglang.srt.layers.rotary_embedding.triton_kernels import (
        triton_apply_rotary_pos_emb,
    )

    N, H_q, H_kv, D = shape
    q, k, cos, sin = _make_inputs(N, H_q, H_kv, D, dtype, device="cuda")

    q_ref, k_ref = apply_rotary_pos_emb_native(q.clone(), k.clone(), cos, sin)
    q_tri, k_tri = triton_apply_rotary_pos_emb(q.clone(), k.clone(), cos, sin)

    atol = TOL[dtype]
    dq = (q_ref.float() - q_tri.float()).abs().max().item()
    dk = (k_ref.float() - k_tri.float()).abs().max().item()
    assert dq <= atol, f"q parity failed for {shape}/{dtype}: max diff {dq:.3e}"
    assert dk <= atol, f"k parity failed for {shape}/{dtype}: max diff {dk:.3e}"


@pytest.mark.skipif(not CUDA, reason="requires CUDA + Triton")
def test_triton_inplace_semantics_cuda():
    """Wrapper must operate in-place (data_ptr preserved when contiguous)."""
    from sglang.srt.layers.rotary_embedding.triton_kernels import (
        triton_apply_rotary_pos_emb,
    )

    q, k, cos, sin = _make_inputs(16, 12, 4, 128, torch.float16, device="cuda")
    q_ptr, k_ptr = q.data_ptr(), k.data_ptr()
    q_out, k_out = triton_apply_rotary_pos_emb(q, k, cos, sin)
    assert q_out.data_ptr() == q_ptr, "q should be modified in-place"
    assert k_out.data_ptr() == k_ptr, "k should be modified in-place"


@pytest.mark.skipif(not CUDA, reason="requires CUDA + Triton")
def test_triton_cuda_graph_capture_replay():
    """The whole point: the Triton kernel must capture inside torch.cuda.graph.

    The native rope (``apply_rotary_pos_emb_native``) cannot be captured
    because ``@torch.compile`` injects host RNG state ("random_rng" failure).
    The Triton kernel must succeed and replay must match eager output.
    """
    from sglang.srt.layers.rotary_embedding.triton_kernels import (
        triton_apply_rotary_pos_emb,
    )

    N, H_q, H_kv, D = 64, 12, 4, 128
    dtype = torch.float16
    q_buf, k_buf, cos_buf, sin_buf = _make_inputs(N, H_q, H_kv, D, dtype, "cuda")

    # Reference (eager) for the input we will replay with
    q_in = q_buf.clone()
    k_in = k_buf.clone()
    q_ref, k_ref = apply_rotary_pos_emb_eager(q_in.clone(), k_in.clone(), cos_buf, sin_buf)

    # Warmup outside graph (to compile triton kernel and get cudaMalloc done)
    triton_apply_rotary_pos_emb(
        q_buf.clone(), k_buf.clone(), cos_buf.clone(), sin_buf.clone()
    )
    torch.cuda.synchronize()

    # Stable input/output buffers required for capture
    q_static = q_in.clone()
    k_static = k_in.clone()
    cos_static = cos_buf.clone()
    sin_static = sin_buf.clone()

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            triton_apply_rotary_pos_emb(q_static, k_static, cos_static, sin_static)
    torch.cuda.current_stream().wait_stream(s)

    # Reset buffers to the original input then replay; result must match eager.
    q_static.copy_(q_in)
    k_static.copy_(k_in)
    g.replay()
    torch.cuda.synchronize()

    atol = TOL[dtype]
    assert torch.allclose(q_static, q_ref, atol=atol, rtol=atol), (
        f"q replay diff: {(q_static.float()-q_ref.float()).abs().max().item():.3e}"
    )
    assert torch.allclose(k_static, k_ref, atol=atol, rtol=atol), (
        f"k replay diff: {(k_static.float()-k_ref.float()).abs().max().item():.3e}"
    )


@pytest.mark.skipif(not CUDA, reason="requires CUDA + Triton")
def test_triton_handles_zero_tokens_cuda():
    """Empty-input (N=0) requests must not crash. Vision encoder may be
    called with zero patches when a request has no images of the relevant
    modality; the kernel must early-return and leave q / k unchanged.
    """
    from sglang.srt.layers.rotary_embedding.triton_kernels import (
        triton_apply_rotary_pos_emb,
    )

    q = torch.empty((0, 12, 128), dtype=torch.float16, device="cuda")
    k = torch.empty((0, 4, 128), dtype=torch.float16, device="cuda")
    cos = torch.empty((0, 64), dtype=torch.float16, device="cuda")
    sin = torch.empty((0, 64), dtype=torch.float16, device="cuda")
    q_out, k_out = triton_apply_rotary_pos_emb(q, k, cos, sin)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_triton_rejects_4d_inputs_cpu():
    """4D q/k are LM-side callers (e.g. Gemma3); the wrapper must refuse so a
    caller doing a global swap of apply_rotary_pos_emb cannot silently break
    them. We only need pure-Python validation, so this runs on CPU even when
    CUDA/Triton is unavailable.
    """
    if not CUDA:
        # Import path that doesn't actually launch the kernel — just the
        # Python guard. We do this lazily so CPU-only env doesn't import
        # triton at module-import time.
        try:
            from sglang.srt.layers.rotary_embedding.triton_kernels import (
                triton_apply_rotary_pos_emb,
            )
        except Exception:
            pytest.skip("triton not importable in this environment")
    else:
        from sglang.srt.layers.rotary_embedding.triton_kernels import (
            triton_apply_rotary_pos_emb,
        )

    q = torch.randn(2, 16, 12, 128)
    k = torch.randn(2, 16, 4, 128)
    cos = torch.randn(16, 64)
    sin = torch.randn(16, 64)
    with pytest.raises(AssertionError):
        triton_apply_rotary_pos_emb(q, k, cos, sin)


@pytest.mark.skipif(not CUDA, reason="sglang.srt.layers.attention.vision import requires CUDA torch")
class TestVisionRopeSelector:
    """End-to-end selector wiring: VisionAttention picks the right rope fn
    based on ``(SGLANG_VIT_ENABLE_CUDA_GRAPH, _is_cuda)``. Gated on CUDA
    because importing ``sglang.srt.layers.attention.vision`` pulls in CUDA
    torch symbols that are absent on a CPU build (unrelated to the rope
    fix itself); on a CUDA host, the test patches ``vision._is_cuda`` to
    exercise both branches without depending on the actual hardware.
    """

    def setup_method(self):
        # Reset the module-level cache before each case so monkey-patching
        # the env var / _is_cuda actually affects resolution.
        from sglang.srt.layers.attention import vision

        vision._vision_rope_fn = None

    def teardown_method(self):
        from sglang.srt.layers.attention import vision

        vision._vision_rope_fn = None

    def test_selector_returns_native_when_env_off(self, monkeypatch):
        from sglang.srt.layers.attention import vision
        from sglang.srt.layers.rotary_embedding.utils import (
            apply_rotary_pos_emb,
        )

        monkeypatch.delenv("SGLANG_VIT_ENABLE_CUDA_GRAPH", raising=False)
        # Even if hardware is CUDA, env-off must keep behavior identical to
        # the unpatched main branch.
        monkeypatch.setattr(vision, "_is_cuda", True)
        fn = vision._get_vision_rope_fn()
        assert fn is apply_rotary_pos_emb

    def test_selector_returns_native_on_non_cuda_even_with_env_on(
        self, monkeypatch
    ):
        from sglang.srt.layers.attention import vision
        from sglang.srt.layers.rotary_embedding.utils import (
            apply_rotary_pos_emb,
        )

        monkeypatch.setenv("SGLANG_VIT_ENABLE_CUDA_GRAPH", "1")
        monkeypatch.setattr(vision, "_is_cuda", False)
        fn = vision._get_vision_rope_fn()
        assert fn is apply_rotary_pos_emb

    def test_selector_returns_triton_on_cuda_when_env_on(self, monkeypatch):
        from sglang.srt.layers.attention import vision
        from sglang.srt.layers.rotary_embedding.triton_kernels import (
            triton_apply_rotary_pos_emb,
        )

        monkeypatch.setenv("SGLANG_VIT_ENABLE_CUDA_GRAPH", "1")
        monkeypatch.setattr(vision, "_is_cuda", True)
        fn = vision._get_vision_rope_fn()
        assert fn is triton_apply_rotary_pos_emb

    def test_selector_caches_resolution(self, monkeypatch):
        from sglang.srt.layers.attention import vision

        monkeypatch.setenv("SGLANG_VIT_ENABLE_CUDA_GRAPH", "1")
        monkeypatch.setattr(vision, "_is_cuda", True)
        first = vision._get_vision_rope_fn()
        # Even if env / hardware change after first resolve, the cached
        # callable is returned (resolution is process-level).
        monkeypatch.setattr(vision, "_is_cuda", False)
        monkeypatch.delenv("SGLANG_VIT_ENABLE_CUDA_GRAPH", raising=False)
        second = vision._get_vision_rope_fn()
        assert first is second


if __name__ == "__main__":
    # Required by sglang's CI runner (`python3 file.py -f`) — without this
    # block, pytest-style tests would silently skip under that invocation.
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
