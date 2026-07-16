"""Unit tests for the CUDA-graph-safe Triton ViT RoPE kernel.

These tests cover two things:

1. Numerical parity between the Triton kernel
   (``triton_vision_rope_qk_inplace`` / ``apply_rotary_pos_emb_cuda_triton``)
   and the reference ``apply_rotary_pos_emb_native`` implementation.

2. CUDA-graph safety. The native implementation is wrapped in
   ``torch.compile(dynamic=True)`` which conflicts with
   ``torch.cuda.graph`` capture (the cause of the historical
   "random_rng" / dynamic-shape-guard error). The Triton kernel must be
   safely capturable + replayable, which is asserted directly here.
"""

import unittest

import torch

from sglang.srt.layers.rotary_embedding.utils import (
    apply_rotary_pos_emb_cuda_triton,
    apply_rotary_pos_emb_native,
)
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestVisionRopeTritonKernel(CustomTestCase):
    """Numerical and CUDA-graph correctness tests for the ViT Triton RoPE."""

    def _make_inputs(self, num_tokens, num_q_heads, num_kv_heads, head_size, dtype):
        device = "cuda"
        torch.manual_seed(0)
        q = torch.randn(num_tokens, num_q_heads, head_size, device=device, dtype=dtype)
        k = torch.randn(num_tokens, num_kv_heads, head_size, device=device, dtype=dtype)
        cos_half = torch.rand(num_tokens, head_size // 2, device=device).to(dtype)
        sin_half = torch.rand(num_tokens, head_size // 2, device=device).to(dtype)
        cos = torch.cat([cos_half, cos_half], dim=-1).contiguous()
        sin = torch.cat([sin_half, sin_half], dim=-1).contiguous()
        return q, k, cos, sin

    def test_parity_with_native(self):
        configs = [
            # (num_tokens, num_q_heads, num_kv_heads, head_size, dtype)
            (137, 16, 16, 80, torch.bfloat16),
            (256, 12, 12, 64, torch.float16),
            (1024, 32, 4, 128, torch.bfloat16),
            (64, 8, 8, 128, torch.float32),
        ]
        for ntok, nqh, nkh, hd, dtype in configs:
            with self.subTest(ntok=ntok, nqh=nqh, nkh=nkh, hd=hd, dtype=dtype):
                q, k, cos, sin = self._make_inputs(ntok, nqh, nkh, hd, dtype)

                q_ref = q.clone()
                k_ref = k.clone()
                q_out_ref, k_out_ref = apply_rotary_pos_emb_native(
                    q_ref, k_ref, cos, sin
                )

                q_triton, k_triton = apply_rotary_pos_emb_cuda_triton(
                    q.clone(), k.clone(), cos, sin
                )

                tol = {"atol": 1e-2, "rtol": 1e-2}
                if dtype == torch.float32:
                    tol = {"atol": 1e-5, "rtol": 1e-5}
                torch.testing.assert_close(q_triton, q_out_ref, **tol)
                torch.testing.assert_close(k_triton, k_out_ref, **tol)

    def test_inplace_semantics(self):
        """Triton path is expected to write q/k in-place."""
        ntok, nqh, nkh, hd, dtype = 64, 8, 8, 80, torch.bfloat16
        q, k, cos, sin = self._make_inputs(ntok, nqh, nkh, hd, dtype)
        q_in = q.clone()
        k_in = k.clone()
        q_out, k_out = apply_rotary_pos_emb_cuda_triton(q_in, k_in, cos, sin)
        # In-place: returned tensors share storage with the inputs.
        self.assertTrue(q_out.data_ptr() == q_in.data_ptr())
        self.assertTrue(k_out.data_ptr() == k_in.data_ptr())

    def test_falls_back_for_4d(self):
        """4D q/k (LM-style) should fall back to the native path."""
        device = "cuda"
        bs, s, h, d = 2, 17, 8, 64
        q4 = torch.randn(1, h, s, d, device=device, dtype=torch.bfloat16)
        k4 = torch.randn(1, h, s, d, device=device, dtype=torch.bfloat16)
        cos = torch.rand(s, d, device=device, dtype=torch.bfloat16)
        sin = torch.rand(s, d, device=device, dtype=torch.bfloat16)
        q_out, k_out = apply_rotary_pos_emb_cuda_triton(q4, k4, cos, sin)
        # Native returns *new* tensors (not in-place), so storage will differ.
        self.assertEqual(q_out.shape, q4.shape)
        self.assertEqual(k_out.shape, k4.shape)

    def test_cuda_graph_capture_and_replay(self):
        """The kernel must be capturable inside ``torch.cuda.graph`` and
        replay to the same numerical result as eager execution.

        This is the exact contract that the prior ``torch.compile`` based
        implementation violated, producing the ``random_rng`` capture error
        for Qwen3.5 / Qwen3-VL ViT runs under
        ``SGLANG_VIT_ENABLE_CUDA_GRAPH=1``.
        """
        ntok, nqh, nkh, hd = 256, 16, 16, 80
        dtype = torch.bfloat16
        q, k, cos, sin = self._make_inputs(ntok, nqh, nkh, hd, dtype)

        # Static buffers for capture.
        q_static = q.clone()
        k_static = k.clone()
        cos_static = cos.clone()
        sin_static = sin.clone()

        # Warm-up to ensure Triton has already JIT-compiled before capture.
        _ = apply_rotary_pos_emb_cuda_triton(
            q_static.clone(), k_static.clone(), cos_static, sin_static
        )
        torch.cuda.synchronize()

        # Capture: write into stable buffers in-place.
        graph = torch.cuda.CUDAGraph()
        # Reset the in-place buffers to a known state before capture.
        cap_q = q.clone()
        cap_k = k.clone()
        with torch.cuda.graph(graph):
            apply_rotary_pos_emb_cuda_triton(cap_q, cap_k, cos_static, sin_static)

        # Reseed buffers and replay -> should produce the same as eager.
        cap_q.copy_(q)
        cap_k.copy_(k)
        graph.replay()
        torch.cuda.synchronize()

        q_ref, k_ref = apply_rotary_pos_emb_cuda_triton(
            q.clone(), k.clone(), cos, sin
        )
        torch.testing.assert_close(cap_q, q_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(cap_k, k_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
