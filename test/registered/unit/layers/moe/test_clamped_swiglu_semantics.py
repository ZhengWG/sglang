from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
    apply_marlin_swiglu,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.humming import HummingRunnerCore
from sglang.srt.models.bailing_moe_v3 import BailingMoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _humming_activation(
    inputs: torch.Tensor,
    *,
    gemm1_clamp_limit: float | None,
) -> torch.Tensor:
    runner = HummingRunnerCore(
        MoeRunnerConfig(
            num_experts=2,
            num_local_experts=2,
            activation="silu",
            gemm1_clamp_limit=gemm1_clamp_limit,
        )
    )
    outputs = torch.empty((*inputs.shape[:-1], inputs.shape[-1] // 2))
    runner.apply_activation(inputs, outputs)
    return outputs


def test_marlin_keeps_the_three_clamp_semantics_distinct():
    inputs = torch.tensor([[3.0, -2.0, 2.0, -3.0]])
    gate, up = inputs.chunk(2, dim=-1)
    limit = 1.5

    raw_gate_clamp = torch.empty_like(gate)
    apply_marlin_swiglu(raw_gate_clamp, inputs, swiglu_limit=limit)
    torch.testing.assert_close(
        raw_gate_clamp,
        F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit),
    )

    post_silu_clamp = torch.empty_like(gate)
    apply_marlin_swiglu(
        post_silu_clamp,
        inputs,
        gemm1_clamp_limit=limit,
    )
    torch.testing.assert_close(
        post_silu_clamp,
        F.silu(gate).clamp(max=limit) * up.clamp(min=-limit, max=limit),
    )

    alpha = 1.702
    alpha_swiglu = torch.empty_like(gate)
    apply_marlin_swiglu(
        alpha_swiglu,
        inputs,
        gemm1_alpha=alpha,
        gemm1_clamp_limit=limit,
    )
    torch.testing.assert_close(
        alpha_swiglu,
        gate.clamp(max=limit)
        * torch.sigmoid(gate.clamp(max=limit) * alpha)
        * (up.clamp(min=-limit, max=limit) + 1),
    )

    assert not torch.allclose(raw_gate_clamp, post_silu_clamp)
    assert not torch.allclose(alpha_swiglu, post_silu_clamp)


def test_humming_matches_bailing_limit_only_swiglu():
    inputs = torch.tensor([[2.0, -1.0, 3.0, -4.0]])
    gate, up = inputs.chunk(2, dim=-1)
    limit = 1.5

    actual = _humming_activation(inputs, gemm1_clamp_limit=limit)
    torch.testing.assert_close(
        actual,
        F.silu(gate).clamp(max=limit) * up.clamp(min=-limit, max=limit),
    )


def test_bailing_v3_resolves_per_layer_limits_from_config_lists():
    expert_limits = [0] * 35 + [4] * 7
    shared_limits = [0] * 34 + [5] * 6 + [7] * 2

    assert BailingMoE._get_swiglu_limit(expert_limits, 34) is None
    assert BailingMoE._get_swiglu_limit(expert_limits, 35) == 4
    assert BailingMoE._get_swiglu_limit(shared_limits, 33) is None
    assert BailingMoE._get_swiglu_limit(shared_limits, 34) == 5
    assert BailingMoE._get_swiglu_limit(shared_limits, 40) == 7


def test_bailing_v3_disables_fusion_for_different_shared_clamp_limits():
    config = SimpleNamespace(
        num_hidden_layers=42,
        first_k_dense_replace=2,
        expert_swiglu_limit_list=[0] * 35 + [4] * 7,
        share_expert_swiglu_limit_list=[0] * 34 + [5] * 6 + [7] * 2,
    )

    assert not BailingMoE._swiglu_limits_match_for_fusion(config)

    config.share_expert_swiglu_limit_list = [0] * 35 + [4] * 7
    assert BailingMoE._swiglu_limits_match_for_fusion(config)
