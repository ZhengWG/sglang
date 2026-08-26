"""CPU-only coverage for MiMo audio attention parallel-group wiring."""

from unittest.mock import patch

import pytest
from torch import nn

from sglang.srt.models.mimo_audio import AudioEncoderAttention
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("dp_attention_enabled", [False, True])
def test_audio_attention_selects_matching_reduce_group(dp_attention_enabled):
    captured = {}

    class CapturingVisionAttention(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    with (
        patch(
            "sglang.srt.models.mimo_audio.is_dp_attention_enabled",
            return_value=dp_attention_enabled,
        ),
        patch(
            "sglang.srt.models.mimo_audio.VisionAttention",
            CapturingVisionAttention,
        ),
    ):
        AudioEncoderAttention(embed_dim=8, num_heads=2)

    assert captured["use_dp_attention_reduce"] is dp_attention_enabled


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
