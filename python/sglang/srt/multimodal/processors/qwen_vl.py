import asyncio
import math
import os
import re
from typing import List, Union, Dict, Any

import torch
import torchvision
from PIL import Image
from torchvision.transforms import InterpolationMode

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.utils import logger

# Base configuration templates based on transformers reference
QWEN2_SERIES_CONFIG = {
    # Based on Qwen2VLImageProcessor and Qwen2VLVideoProcessor
    "IMAGE_FACTOR": 28,  # factor = patch_size * merge_size = 14 * 2
    "MIN_PIXELS": 56 * 56,  # from Qwen2VLImageProcessor default
    "MAX_PIXELS": 28 * 28 * 1280,  # from Qwen2VLImageProcessor default
    "VIDEO_MIN_PIXELS": 128 * 28 * 28,  # from Qwen2VLVideoProcessor
    "VIDEO_MAX_PIXELS": 28 * 28 * 768,  # from Qwen2VLVideoProcessor
    "VIDEO_TOTAL_PIXELS": int(float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9))),
    "PATCH_SIZE": 14,  # from Qwen2VLImageProcessor
    "TEMPORAL_PATCH_SIZE": 2,  # from Qwen2VLVideoProcessor
    "MERGE_SIZE": 2,  # from Qwen2VLImageProcessor
    # Frame and video processing parameters
    "FRAME_FACTOR": 2,  # from Qwen2VLVideoProcessor temporal_patch_size
    "FPS": 2.0,  # from Qwen2VLVideoProcessor default fps
    "FPS_MIN_FRAMES": 4,  # from Qwen2VLVideoProcessor min_frames
    "FPS_MAX_FRAMES": 768,  # from Qwen2VLVideoProcessor max_frames
    "MAX_RATIO": 200,  # from Qwen2VLImageProcessor smart_resize
}

QWEN3_SERIES_CONFIG = {
    # Based on Qwen3VLVideoProcessor
    "IMAGE_FACTOR": 32,  # factor = patch_size * merge_size = 16 * 2
    "MIN_PIXELS": 128 * 128,  # from Qwen3VLVideoProcessor smart_resize
    "MAX_PIXELS": 16 * 16 * 2 * 2 * 2 * 6144,  # from Qwen3VLVideoProcessor smart_resize
    "VIDEO_MIN_PIXELS": 128 * 32 * 32,  # from Qwen3VLVideoProcessor size
    "VIDEO_MAX_PIXELS": 32 * 32 * 768,  # from Qwen3VLVideoProcessor size
    "VIDEO_TOTAL_PIXELS": int(float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9))),
    "PATCH_SIZE": 16,  # from Qwen3VLVideoProcessor
    "TEMPORAL_PATCH_SIZE": 2,  # from Qwen3VLVideoProcessor
    "MERGE_SIZE": 2,  # from Qwen3VLVideoProcessor
    # Frame and video processing parameters
    "FRAME_FACTOR": 2,  # from Qwen3VLVideoProcessor temporal_patch_size
    "FPS": 2,  # from Qwen3VLVideoProcessor fps
    "FPS_MIN_FRAMES": 4,  # from Qwen3VLVideoProcessor min_frames
    "FPS_MAX_FRAMES": 768,  # from Qwen3VLVideoProcessor max_frames
    "MAX_RATIO": 200,  # from Qwen3VLVideoProcessor smart_resize
}

# Version-specific parameter configurations - grouped by similar configs
QWEN_VERSION_CONFIGS = {
    # Qwen2 and Qwen2.5 series - identical configuration and logic
    "qwen2_vl": {**QWEN2_SERIES_CONFIG, "smart_nframes_version": "qwen2"},
    "qwen2_5_vl": {**QWEN2_SERIES_CONFIG, "smart_nframes_version": "qwen2"},
    
    # Qwen3 series - identical configuration and logic for both qwen3_vl and qwen3_vl_moe
    "qwen3_vl": {**QWEN3_SERIES_CONFIG, "smart_nframes_version": "qwen3"},
    "qwen3_vl_moe": {**QWEN3_SERIES_CONFIG, "smart_nframes_version": "qwen3"},
}


def detect_qwen_version(hf_config) -> str:
    """
    Detect Qwen version from HuggingFace config.
    
    Args:
        hf_config: HuggingFace model configuration
        
    Returns:
        str: Detected Qwen version ('qwen2_vl', 'qwen2_5_vl', 'qwen3_vl', 'qwen3_vl_moe')
    """
    model_type = getattr(hf_config, 'model_type', '')
    
    if model_type in QWEN_VERSION_CONFIGS:
        return model_type
    
    # Fallback: try to detect from model name or other attributes
    if hasattr(hf_config, 'architectures') and hf_config.architectures:
        arch = hf_config.architectures[0]
        if 'Qwen2VL' in arch:
            return 'qwen2_vl'
        elif 'Qwen2_5_VL' in arch:
            return 'qwen2_5_vl'
        elif 'Qwen3VLMoe' in arch:
            return 'qwen3_vl_moe'
        elif 'Qwen3VL' in arch:
            return 'qwen3_vl'
    
    # Default fallback
    logger.warning(f"Could not detect Qwen version from model_type '{model_type}', using qwen2_vl as default")
    return 'qwen2_vl'


def get_version_config(version: str) -> Dict[str, Any]:
    """
    Get version-specific configuration parameters.
    
    Args:
        version: Qwen version string
        
    Returns:
        Dict containing version-specific parameters
    """
    return QWEN_VERSION_CONFIGS.get(version, QWEN_VERSION_CONFIGS['qwen2_vl'])


def smart_resize(
    height: int,
    width: int,
    factor: int = None,
    min_pixels: int = None,
    max_pixels: int = None,
    version: str = "qwen2_vl",
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    
    Args:
        height: Image height
        width: Image width
        factor: Resize factor (if None, uses version-specific default)
        min_pixels: Minimum pixels (if None, uses version-specific default)
        max_pixels: Maximum pixels (if None, uses version-specific default)
        version: Qwen version for default parameters
    """
    # Get version-specific defaults if not provided
    version_config = get_version_config(version)
    if factor is None:
        factor = version_config["IMAGE_FACTOR"]
    if min_pixels is None:
        min_pixels = version_config["MIN_PIXELS"]
    if max_pixels is None:
        max_pixels = version_config["MAX_PIXELS"]
    
    if max(height, width) / min(height, width) > version_config["MAX_RATIO"]:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {version_config['MAX_RATIO']}, got {max(height, width) / min(height, width)}"
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
    min_pixels: int = None,
    max_pixels: int = None,
    size_factor: int = None,
    version: str = "qwen2_vl",
) -> Image.Image:
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        version=version,
    )
    image = image.resize((resized_width, resized_height))
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
    min_pixels: int = None,
    max_pixels: int = None,
    size_factor: int = None,
    version: str = "qwen2_vl",
):
    return resize_image(image, min_pixels, max_pixels, size_factor, version)


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
    version: str = "qwen2_vl",
) -> int:
    """Calculate the number of frames for video used for model inputs.
    
    Based on transformers implementations:
    - Qwen2/Qwen2.5: Uses temporal_patch_size for frame adjustment, torch.arange for indices
    - Qwen3: Uses np.linspace for indices, no temporal_patch_size adjustment

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.
        version (str): Qwen version for different logic implementations and parameter selection.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not (
        "fps" in ele and "nframes" in ele
    ), "Only accept either `fps` or `nframes`"
    
    # Get version-specific configuration
    version_config = get_version_config(version)
    temporal_patch_size = version_config["TEMPORAL_PATCH_SIZE"]
    default_fps = version_config["FPS"]
    fps_min_frames = version_config["FPS_MIN_FRAMES"]
    fps_max_frames = version_config["FPS_MAX_FRAMES"]
    
    # Determine if this is Qwen2 series or Qwen3 series
    is_qwen2_series = version in ["qwen2_vl", "qwen2_5_vl"]
    is_qwen3_series = version in ["qwen3_vl", "qwen3_vl_moe"]
    
    if "nframes" in ele:
        num_frames = ele["nframes"]
        if is_qwen2_series:
            # Qwen2/Qwen2.5: round to temporal_patch_size
            num_frames = round(num_frames / temporal_patch_size) * temporal_patch_size
    else:
        # Use ele["fps"] or version-specific default FPS
        fps = ele.get("fps", default_fps)
        min_frames = ele.get("min_frames", fps_min_frames)
        max_frames = ele.get("max_frames", fps_max_frames)
        
        if is_qwen2_series:
            # Qwen2/Qwen2.5 logic (based on Qwen2VLVideoProcessor.sample_frames)
            max_frames = math.floor(min(max_frames, total_frames) / temporal_patch_size) * temporal_patch_size
            num_frames = total_frames / video_fps * fps
            num_frames = min(min(max(num_frames, min_frames), max_frames), total_frames)
            num_frames = math.floor(num_frames / temporal_patch_size) * temporal_patch_size
            
        elif is_qwen3_series:
            # Qwen3 logic (based on Qwen3VLVideoProcessor.sample_frames)
            num_frames = int(total_frames / video_fps * fps)
            num_frames = min(min(max(num_frames, min_frames), max_frames), total_frames)
        else:
            # This should never happen as detect_qwen_version ensures valid versions
            raise ValueError(f"Unsupported Qwen version: {version}")
    
    # Validation
    if num_frames > total_frames:
        raise ValueError(
            f"Video can't be sampled. The inferred `num_frames={num_frames}` exceeds `total_num_frames={total_frames}`. "
            "Decrease `num_frames` or `fps` for sampling."
        )
    
    return int(num_frames)


# process video, qwen-specific
async def preprocess_video(
    vr,
    image_factor: int = None,
    version: str = "qwen2_vl",
    # vr: VideoReader, image_factor: int = None
) -> torch.Tensor:
    """
    Process video for Qwen models with version-specific parameters.
    
    Args:
        vr: VideoReader object
        image_factor: Image factor for resizing (if None, uses version-specific default)
        version: Qwen version for parameter selection
    """
    ele = {}
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    
    # Get version-specific configuration
    version_config = get_version_config(version)
    smart_nframes_version = version_config["smart_nframes_version"]
    
    # Use version-specific image_factor if not provided
    if image_factor is None:
        image_factor = version_config["IMAGE_FACTOR"]
    
    # Use version-specific smart_nframes
    nframes = smart_nframes(
        {}, 
        total_frames=total_frames, 
        video_fps=video_fps,
        version=smart_nframes_version
    )
    
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    nframes, _, height, width = video.shape
    
    # Use version-specific pixel parameters
    min_pixels = ele.get("min_pixels", version_config["VIDEO_MIN_PIXELS"])
    total_pixels = ele.get("total_pixels", version_config["VIDEO_TOTAL_PIXELS"])
    frame_factor = version_config["FRAME_FACTOR"]
    max_pixels = max(
        min(version_config["VIDEO_MAX_PIXELS"], total_pixels / nframes * frame_factor),
        int(min_pixels * 1.05),
    )
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(
            f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}]."
        )
    max_pixels = min(max_pixels_supposed, max_pixels)
    
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    video = torchvision.transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return video


# Compatible with Qwen2VL, Qwen2_5VL, Qwen3VL, and Qwen3VL_MoE
class QwenVLImageProcessor(SGLangBaseProcessor):
    models = [
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        
        # Detect Qwen version and get version-specific configuration
        self.qwen_version = detect_qwen_version(hf_config)
        self.version_config = get_version_config(self.qwen_version)
        
        # The regex that matches expanded image tokens.
        self.IM_START_TOKEN_ID = hf_config.vision_start_token_id
        self.IM_END_TOKEN_ID = hf_config.vision_end_token_id
        self.vision_start_token_id = hf_config.vision_start_token_id
        self.vision_end_token_id = hf_config.vision_end_token_id
        self.NUM_TOKEN_PER_FRAME = 770
        
        # Version-specific parameters from transformers reference
        self.IMAGE_FACTOR = self.version_config["IMAGE_FACTOR"]
        self.MIN_PIXELS = self.version_config["MIN_PIXELS"]
        self.MAX_PIXELS = self.version_config["MAX_PIXELS"]
        self.VIDEO_MIN_PIXELS = self.version_config["VIDEO_MIN_PIXELS"]
        self.VIDEO_MAX_PIXELS = self.version_config["VIDEO_MAX_PIXELS"]
        self.VIDEO_TOTAL_PIXELS = self.version_config["VIDEO_TOTAL_PIXELS"]
        self.PATCH_SIZE = self.version_config["PATCH_SIZE"]
        self.TEMPORAL_PATCH_SIZE = self.version_config["TEMPORAL_PATCH_SIZE"]
        self.MERGE_SIZE = self.version_config["MERGE_SIZE"]
        # Frame and video processing parameters
        self.FRAME_FACTOR = self.version_config["FRAME_FACTOR"]
        self.FPS = self.version_config["FPS"]
        self.FPS_MIN_FRAMES = self.version_config["FPS_MIN_FRAMES"]
        self.FPS_MAX_FRAMES = self.version_config["FPS_MAX_FRAMES"]
        self.MAX_RATIO = self.version_config["MAX_RATIO"]
        
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_id=hf_config.image_token_id,
            image_token_regex=re.compile(
                r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
            ),
            video_token_id=hf_config.video_token_id,
        ).build(_processor)
        
        logger.info(f"Initialized QwenVLImageProcessor for version: {self.qwen_version}")

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):

        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            multimodal_tokens=self.mm_tokens,
        )

        # Qwen-specific: resize images if they are raw Image objects
        if base_output.images and isinstance(base_output.images[0], Image.Image):
            resize_tasks = [
                resize_image_async(image, version=self.qwen_version) 
                for image in base_output.images
            ]
            base_output.images = await asyncio.gather(*resize_tasks)

        if base_output.videos:
            # Use version-specific video preprocessing
            video_tasks = [
                preprocess_video(
                    video, 
                    version=self.qwen_version
                ) 
                for video in base_output.videos
            ]
            base_output.videos = await asyncio.gather(*video_tasks)

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        input_ids = input_ids.flatten()
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.hf_config.model_type,
            tokens_per_second=getattr(
                self.hf_config.vision_config, "tokens_per_second", None
            ),
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            video_grid_thw=getattr(ret, "video_grid_thw", None),
            second_per_grid_ts=getattr(ret, "second_per_grid_ts", None),
        )
        mrope_positions = mrope_positions.squeeze(1)

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
            "im_token_id": self.mm_tokens.image_token_id,
            "video_token_id": self.mm_tokens.video_token_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }