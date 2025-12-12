import asyncio
import math
import os
import re
import time
from typing import Dict, List, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import InterpolationMode

from sglang.srt.environ import envs
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.utils import logger
from sglang.srt.managers.io_struct import MMProcessMetrics

MAX_RATIO = 200
RESIZE_RESAMPLE = getattr(Image, envs.SGLANG_RESIZE_RESAMPLE.get(), None)
if envs.SGLANG_RESIZE_RESAMPLE.is_set() and RESIZE_RESAMPLE is None:
    logger.warning(
        f"Invalid RESIZE_RESAMPLE value: '{envs.SGLANG_RESIZE_RESAMPLE.get()}'. "
        f"Ignoring and using default."
    )


def smart_resize_for_video(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = 128 * 128,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if num_frames < temporal_factor:
        raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )

    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)
    t_bar = round_by_factor(num_frames, temporal_factor)

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def smart_resize_for_image(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def resize_image(
    image,
    min_pixels: int,
    max_pixels: int,
    size_factor: int,
    mm_sampling_kwargs: dict = {},
) -> Image.Image:
    width, height = image.size

    # 自定义长宽
    if (mm_sampling_kwargs and "resized_height" in mm_sampling_kwargs
        and "resized_width" in mm_sampling_kwargs):
        resized_height = mm_sampling_kwargs["resized_height"]
        resized_width = mm_sampling_kwargs["resized_width"]
        if resized_height > 0 and resized_width > 0:
            logger.info(f"Resize image to {height}x{width} -> {resized_height}x{resized_width}")
            height = resized_height
            width = resized_width

    resized_height, resized_width = smart_resize_for_image(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height), resample=RESIZE_RESAMPLE)
    return image


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


async def resize_image_async(
    image,
    min_pixels: int,
    max_pixels: int,
    size_factor: int,
    mm_sampling_kwargs: dict = {},
):
    return resize_image(image, min_pixels, max_pixels, size_factor, mm_sampling_kwargs=mm_sampling_kwargs)


def smart_nframes(
    ele: dict,
    total_frames: int,
    temporal_factor: int,
    video_fps: int | float,
    default_fps: int | float,
    default_fps_min_frames: int,
    default_fps_max_frames: int,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        temporal_factor (int): the original temporal factor.
        video_fps (int | float): the original fps of the video.
        default_fps (int | float): the target fps of the video.
        default_fps_min_frames (int): the min frames of the video.
        default_fps_max_frames (int): the max frames of the video.

    Raises:
        ValueError: nframes should in interval [temporal_factor, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not (
        "fps" in ele and "nframes" in ele
    ), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], temporal_factor)
    else:
        fps = ele.get("fps", default_fps)
        min_frames = ceil_by_factor(
            ele.get("min_frames", default_fps_min_frames), temporal_factor
        )
        max_frames = floor_by_factor(
            ele.get("max_frames", min(default_fps_max_frames, total_frames)),
            temporal_factor,
        )
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(
                f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]"
            )
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, temporal_factor)
    if not (temporal_factor <= nframes and nframes <= total_frames):
        raise ValueError(
            f"nframes should in interval [{temporal_factor}, {total_frames}], but got {nframes}."
        )
    return nframes


# process video, qwen-specific
async def preprocess_video(
    vr,
    image_factor: int,
    video_min_pixels: int,
    video_max_pixels: int,
    temporal_factor: int,
    default_fps: int | float,
    default_fps_min_frames: int,
    default_fps_max_frames: int,
    mm_sampling_kwargs: dict = {},
    # vr: VideoReader, image_factor: int = IMAGE_FACTOR
) -> torch.Tensor:
    try:
        entry_time = time.perf_counter()
        ele = {}
        if mm_sampling_kwargs:
            ele.update(mm_sampling_kwargs)

        video = vr
        total_frames = video_fps = idx = None
        if not isinstance(vr, np.ndarray):
            total_frames, video_fps = len(vr), vr.get_avg_fps()
            nframes = smart_nframes(
                ele,
                total_frames=total_frames,
                temporal_factor=temporal_factor,
                video_fps=video_fps,
                default_fps=default_fps,
                default_fps_min_frames=default_fps_min_frames,
                default_fps_max_frames=default_fps_max_frames,
            )
            idx = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)
            idx = np.unique(idx)
            video_np = vr.get_batch(idx).asnumpy()
            video = torch.from_numpy(video_np).pin_memory()
        video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", video_min_pixels)
        max_pixels = ele.get("max_pixels", video_max_pixels)
        max_pixels = max(max_pixels, int(min_pixels * 1.05))

        get_batch_time = time.perf_counter()

        if max_pixels > video_max_pixels:
            logger.warning(
                f"The given max_pixels[{max_pixels}] exceeds limit[{video_max_pixels}]."
            )
        max_pixels = min(max_pixels, video_max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize_for_video(
                nframes,
                ele["resized_height"],
                ele["resized_width"],
                temporal_factor=temporal_factor,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        else:
            resized_height, resized_width = smart_resize_for_video(
                nframes,
                height,
                width,
                temporal_factor=temporal_factor,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        smart_resize_time = time.perf_counter()
        video = torchvision.transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BILINEAR,
        )
        video = video.pin_memory()

        if total_frames is None and video_fps is None and idx is None:
            total_frames = nframes
            video_fps = ele.get("fps", default_fps)
            idx = list(range(nframes))
        video_metadata = {
            "fps": video_fps,
            "duration": total_frames / video_fps,
            "total_num_frames": total_frames,
            "frames_indices": idx,
            "video_backend": "torchvision",
            "width": resized_width,
            "height": resized_height,
        }
        torchvision_resize_time = time.perf_counter()
        logger.info(
            f"[preprocess_video Perf], "
            f"get_batch_time: {(get_batch_time - entry_time) * 1000:.2f} ms, "
            f"smart_resize_time: {(smart_resize_time - get_batch_time) * 1000:.2f} ms, "
            f"torchvision_resize_time: {(torchvision_resize_time - smart_resize_time) * 1000:.2f} ms, "
            f"total_time: {(torchvision_resize_time - entry_time) * 1000:.2f} ms"
        )
        return video, video_metadata
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise ValueError(f"Error processing video: {str(e)}")


# Compatible with Qwen-VL & Qwen-Omni Series
class QwenVLImageProcessor(SGLangBaseProcessor):
    models = [
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
        Qwen3OmniMoeForConditionalGeneration,
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.model_type = hf_config.model_type
        if hf_config.model_type == "qwen3_omni_moe":
            hf_config = hf_config.thinker_config

        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.vision_start_token_id = hf_config.vision_start_token_id
        self.vision_end_token_id = getattr(hf_config, "vision_end_token_id", None)

        self.audio_start_token_id = getattr(hf_config, "audio_start_token_id", None)
        self.audio_token_id = getattr(hf_config, "audio_token_id", None)

        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 4 * 28 * 28
        self.MAX_PIXELS = envs.SGLANG_IMAGE_MAX_PIXELS.get()
        # FIXME(yudian.zy): 临时把qwen2.5-vl的单图大小限制为1k*1k，防止rank0 OOM
        self.MAX_PIXELS = int(self.MAX_PIXELS // 12.25)

        self.VIDEO_MIN_PIXELS = 4 * 28 * 28 # 3136
        self.VIDEO_MAX_PIXELS = 16384 * 28 * 28 # 12845056
        self.FPS = 2.0
        self.FPS_MIN_FRAMES = 4
        self.FPS_MAX_FRAMES = 768
        self.TEMPORAL_PATCH_SIZE = 2

        if self.model_type in ("qwen3_vl", "qwen3_vl_moe"):
            image_processor = getattr(_processor, "image_processor", None)
            self.IMAGE_FACTOR = image_processor.patch_size * image_processor.merge_size

            # FIXME(yudian.zy): 临时把qwen3-vl的单图大小限制为1k*1k，防止rank0 OOM
            image_longest_edge = image_processor.size["longest_edge"]
            if image_longest_edge >= (32 * self.IMAGE_FACTOR) ** 2:
                image_processor.size["longest_edge"] = image_longest_edge // 16
            self.MIN_PIXELS = image_processor.size["shortest_edge"]
            self.MAX_PIXELS = image_processor.size["longest_edge"]

            video_processor = getattr(_processor, "video_processor", None)
            self.VIDEO_MIN_PIXELS = video_processor.size["shortest_edge"]
            self.VIDEO_MAX_PIXELS = video_processor.size["longest_edge"]
            self.FPS = video_processor.fps
            self.FPS_MIN_FRAMES = video_processor.min_frames
            self.FPS_MAX_FRAMES = video_processor.max_frames
            self.TEMPORAL_PATCH_SIZE = video_processor.temporal_patch_size

        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_id=hf_config.image_token_id,
            # The regex that matches expanded image tokens.
            image_token_regex=re.compile(
                r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
            ),
            video_token_id=hf_config.video_token_id,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        mm_metric = MMProcessMetrics()
        mm_metric.mm_entry_time = time.perf_counter()
        mm_metric.mm_entry_time_ts = time.time()
        base_output = await self.load_mm_data_async(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            audio_data=request_obj.audio_data,
            multimodal_tokens=self.mm_tokens,
        )

        mm_metric.mm_load_time = time.perf_counter()

        rid = getattr(request_obj, "rid", "anonymous_rid")
        mm_sampling_kwargs = getattr(request_obj, "mm_sampling_kwargs", {})

        # Qwen-specific: resize images if they are raw Image objects
        if (self.model_type != "paddleocr_vl" and base_output.images and
            isinstance(base_output.images[0], Image.Image)):
            resize_tasks = [
                resize_image_async(
                    image, self.MIN_PIXELS, self.MAX_PIXELS, self.IMAGE_FACTOR, mm_sampling_kwargs=mm_sampling_kwargs,
                )
                for image in base_output.images
            ]
            base_output.images = await asyncio.gather(*resize_tasks)

        video_metadata = None
        if base_output.videos:
            video_results = await asyncio.gather(
                *[
                    preprocess_video(
                        video,
                        image_factor=self.IMAGE_FACTOR,
                        video_min_pixels=self.VIDEO_MIN_PIXELS,
                        video_max_pixels=self.VIDEO_MAX_PIXELS,
                        temporal_factor=self.TEMPORAL_PATCH_SIZE,
                        default_fps=self.FPS,
                        default_fps_min_frames=self.FPS_MIN_FRAMES,
                        default_fps_max_frames=self.FPS_MAX_FRAMES,
                        mm_sampling_kwargs=mm_sampling_kwargs,
                    )
                    for video in base_output.videos
                ]
            )
            base_output.videos, video_metadata = map(list, zip(*video_results))

        mm_metric.mm_preprocess_time = time.perf_counter()

        # NOTE: for qwen3-vl, video_meta need to be passed in, since do_sample_frames is already done in preprocess_video
        if self.model_type in ("qwen3_vl", "qwen3_vl_moe"):
            mm_items, input_ids, ret = self.process_and_combine_mm_data(
                base_output,
                self.mm_tokens,
                video_metadata=video_metadata,
                do_sample_frames=False,
            )
        else:
            mm_items, input_ids, ret = self.process_and_combine_mm_data(
                base_output, self.mm_tokens
            )

        audio_feature_lengths = None

        if self.model_type == "qwen3_omni_moe":
            audio_item = next((mm for mm in mm_items if mm.is_audio()), None)
            if audio_item:
                audio_feature_lengths = torch.sum(
                    audio_item.feature_attention_mask, dim=1
                )

        second_per_grid_ts = getattr(ret, "second_per_grid_ts", None) or getattr(
            ret, "video_second_per_grid", None
        )

        mm_metric.mm_process_time = time.perf_counter()

        input_ids = input_ids.flatten()

        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.model_type,
            tokens_per_second=getattr(
                self.hf_config.vision_config, "tokens_per_second", None
            ),
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            video_grid_thw=getattr(ret, "video_grid_thw", None),
            second_per_grid_ts=second_per_grid_ts,
            use_audio_in_video=False,
            audio_seqlens=audio_feature_lengths,
            audio_token_id=getattr(self.hf_config, "audio_token_id", None),
            audio_start_token_id=self.audio_start_token_id,
            position_id_per_seconds=getattr(
                self.hf_config, "position_id_per_seconds", None
            ),
        )
        mrope_positions = mrope_positions.squeeze(1)
        mm_metric.mm_get_rope_index_time = time.perf_counter()

        if hasattr(request_obj,"metrics") and isinstance(request_obj.metrics, Dict):
            request_obj.metrics.update(mm_metric.to_dict())

        logger.info(
            f"[QwenVLProcessor Perf] {rid=}, "
            f"load_time: {(mm_metric.mm_load_time - mm_metric.mm_entry_time) * 1000:.2f} ms, "
            f"preprocess_time: {(mm_metric.mm_preprocess_time - mm_metric.mm_load_time) * 1000:.2f} ms, "
            f"process_time: {(mm_metric.mm_process_time - mm_metric.mm_preprocess_time) * 1000:.2f} ms, "
            f"get_rope_index_time: {(mm_metric.mm_get_rope_index_time - mm_metric.mm_process_time) * 1000:.2f} ms, "
            f"total_time: {(mm_metric.mm_get_rope_index_time - mm_metric.mm_entry_time) * 1000:.2f} ms"
        )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_start_id": self.vision_start_token_id,
            "im_end_id": self.vision_end_token_id,
            "im_token_id": self.mm_tokens.image_token_id,
            "video_token_id": self.mm_tokens.video_token_id,
            "audio_token_id": self.mm_tokens.audio_token_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }

