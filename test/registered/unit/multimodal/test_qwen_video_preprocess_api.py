import asyncio

from sglang.srt.multimodal.processors.qwen_vl import (
    _merge_video_preprocess_config,
    preprocess_video,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_request_video_config_overrides_server_config():
    server_config = {"fps": 2.0, "min_frames": 4, "max_pixels": 1024}
    request_config = {"fps": 5.0, "nframes": 8}

    merged = _merge_video_preprocess_config(server_config, request_config)

    assert merged == {
        "fps": 5.0,
        "min_frames": 4,
        "max_pixels": 1024,
        "nframes": 8,
    }
    assert server_config["fps"] == 2.0
    assert request_config["fps"] == 5.0


def test_public_video_config_api_preserves_preprocessed_inputs():
    video = object()

    processed, metadata = asyncio.run(
        preprocess_video(video, video_config={"fps": 5.0})
    )

    assert processed is video
    assert metadata is None
