import math
from typing import Dict, Optional

import torch

# cuLA kernel uses exp2() internally, so gate values must be in log-base-2 space.
# RCP_LN2 converts from natural log space (model output) to log-base-2 space.
RCP_LN2 = 1.0 / math.log(2.0)

# cuLA kernel chunk size. Sequences shorter than this fall back to Triton.
_CULA_CHUNK_SIZE = 64

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)


def _triton_fallback(
    q,
    k,
    v,
    g,
    beta,
    ssm_states,
    cache_indices,
    query_start_loc,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
):
    """Fall back to the Triton chunk_kda kernel (handles all preprocessing).

    `g` is the RAW gate (PR #730 / #23038 contract); chunk_kda applies the gate
    activation internally when A_log is provided, so they must be threaded
    through here too -- otherwise the fallback silently skips activation.
    """
    from sglang.kernels.ops.attention.fla.kda import chunk_kda

    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=ssm_states,
        initial_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=query_start_loc,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )


class CulaKDAKernel(LinearAttnKernelBase):
    """cuLA SM90 fully-fused kernel for KDA (Kimi Delta Attention) prefill.

    cuLA only supports safe_gate=True mode, where gating values are clamped
    to > -5.  Sequences shorter than the chunk size (64) fall back to Triton.
    """

    def __init__(self):
        super().__init__()
        # Cache workspace buffers per CUDA device
        self._workspace_cache: Dict[int, torch.Tensor] = {}

    def _get_workspace_buffer(self, device: torch.device) -> torch.Tensor:
        """Get or create a workspace buffer for the given device."""
        device_idx = device.index if device.index is not None else 0
        if device_idx not in self._workspace_cache:
            sm_count = torch.cuda.get_device_properties(device).multi_processor_count
            self._workspace_cache[device_idx] = torch.zeros(
                sm_count * 128, dtype=torch.uint8, device=device
            )
        return self._workspace_cache[device_idx]

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("CulaKDAKernel only supports prefill (extend)")

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Guard: sequences shorter than cuLA chunk size fall back to Triton.
        seq_lens = query_start_loc[1:] - query_start_loc[:-1]
        min_seq_len = seq_lens.min().item()
        if min_seq_len < _CULA_CHUNK_SIZE:
            return _triton_fallback(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
            )

        return self._cula_extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
        )

    def _cula_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from sgl_kernel import kda_fwd_prefill

        from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
        from sglang.kernels.ops.attention.fla.kda import kda_gate_chunk_cumsum
        from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd

        # Input shapes: q, k, v = [1, packed_seq, H, D], g = [1, packed_seq, H, D], beta = [1, packed_seq, H]
        batch_size = q.shape[0]  # should be 1
        packed_seq = q.shape[1]
        num_heads = q.shape[2]
        head_dim = q.shape[3]

        # 1. L2 normalize Q, K (consistent with Triton path use_qk_l2norm_in_kernel=True)
        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())

        # 2. Gate activation + chunk-local cumsum.
        # PR #730 (#23038) moved KDA gate activation into the kernel, so `g`
        # arriving here is now the RAW gate. Apply the same activation as the
        # Triton chunk_kda path -- standard gate -exp(A_log)*softplus(g+dt_bias),
        # or safe gate lower_bound*sigmoid(exp(A_log)*(g+dt_bias)) when
        # lower_bound is set -- fused with the chunk-local cumsum. scale=RCP_LN2
        # puts the cumulative gate in log-base-2 space for cuLA's exp2 kernel.
        if A_log is not None:
            g = kda_gate_chunk_cumsum(
                g,
                A_log=A_log,
                chunk_size=64,
                scale=RCP_LN2,
                dt_bias=dt_bias,
                cu_seqlens=query_start_loc,
                lower_bound=lower_bound,
            )
        else:
            # Legacy contract: `g` is already gate-activated; cumsum only.
            g = chunk_local_cumsum(
                g, chunk_size=64, scale=RCP_LN2, cu_seqlens=query_start_loc
            )

        # 4. Reshape [1, packed_seq, H, D] -> [packed_seq, H, D], ensure contiguous
        q = q.reshape(packed_seq, num_heads, head_dim).contiguous()
        k = k.reshape(packed_seq, num_heads, head_dim).contiguous()
        v = v.reshape(packed_seq, num_heads, head_dim).contiguous()
        g = g.reshape(packed_seq, num_heads, head_dim).contiguous()
        beta = beta.reshape(packed_seq, num_heads).contiguous()

        # 5. State gather: get per-batch states from the pool (VK layout [N, H, V, K])
        # The kernel natively uses VK layout via CuTe LayoutLeft (K, V, H, N).
        input_state = ssm_states[cache_indices].contiguous()

        # 6. cu_seqlens
        cu_seqlens = query_start_loc.to(torch.int32)

        # 7. Workspace buffer
        workspace_buffer = self._get_workspace_buffer(q.device)

        # 8. Scale
        scale = head_dim**-0.5

        # 9. Call C++ kernel (safe_gate=True)
        output, output_state = kda_fwd_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens=cu_seqlens,
            workspace_buffer=workspace_buffer,
            scale=scale,
            safe_gate=True,
            input_state=input_state,
            alpha=g,
            beta=beta,
        )

        # 10. Write output state back (already in VK layout from C++ API)
        ssm_states[cache_indices] = output_state

        # 11. Reshape output: [packed_seq, H, D] -> [1, packed_seq, H, D]
        output = output.reshape(batch_size, packed_seq, num_heads, head_dim)

        return output, None

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("CulaKDAKernel does not support target_verify")
