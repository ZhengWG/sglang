from types import SimpleNamespace

import pytest
import torch

from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FailingParam:
    @staticmethod
    def weight_loader(*args, **kwargs):
        raise RuntimeError("weight copy failed")


def _fake_model(*, num_experts=None):
    config = SimpleNamespace(tie_word_embeddings=False)
    if num_experts is not None:
        config.num_experts = num_experts
    return SimpleNamespace(
        config=config,
        model=SimpleNamespace(),
        named_parameters=lambda: [("weight", _FailingParam())],
        pp_group=SimpleNamespace(is_last_rank=False),
    )


def test_qwen3_propagates_async_weight_loader_errors():
    model = _fake_model()

    with pytest.raises(RuntimeError, match="weight copy failed"):
        Qwen3ForCausalLM.load_weights(
            model,
            [("weight", torch.empty(1))],
        )


def test_qwen3_moe_propagates_async_weight_loader_errors():
    model = _fake_model(num_experts=1)

    with pytest.raises(RuntimeError, match="weight copy failed"):
        Qwen3MoeForCausalLM.load_weights(
            model,
            [("weight", torch.empty(1))],
        )
