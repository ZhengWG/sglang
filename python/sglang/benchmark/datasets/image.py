import io
import json
import os
import warnings
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybase64
import requests
from PIL import Image
from transformers import AutoProcessor

from sglang.benchmark.datasets.common import (
    BaseDataset,
    DatasetRow,
    compute_random_lens,
    gen_mm_prompt,
    log_special_token_filter_stats,
)
from sglang.benchmark.utils import get_processor


@dataclass
class ImageDataset(BaseDataset):
    num_requests: int
    image_count: int
    input_len: int
    output_len: int
    range_ratio: float
    image_content: str
    image_format: str
    image_resolution: str
    backend: str
    random_image_count: bool
    image_url_list: Optional[str] = None
    image_url_probe_count: int = 1

    @classmethod
    def from_args(cls, args: Namespace) -> "ImageDataset":
        return cls(
            num_requests=args.num_prompts,
            image_count=args.image_count,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            range_ratio=args.random_range_ratio,
            image_content=args.image_content,
            image_format=args.image_format,
            image_resolution=args.image_resolution,
            backend=args.backend,
            random_image_count=args.random_image_count,
            image_url_list=getattr(args, "image_url_list", None),
            image_url_probe_count=getattr(args, "image_url_probe_count", 1),
        )

    def load(self, tokenizer=None, model_id=None) -> List[DatasetRow]:
        processor = get_processor(model_id)
        return sample_image_requests(
            num_requests=self.num_requests,
            image_count=self.image_count,
            input_len=self.input_len,
            output_len=self.output_len,
            range_ratio=self.range_ratio,
            processor=processor,
            image_content=self.image_content,
            image_format=self.image_format,
            image_resolution=self.image_resolution,
            backend=self.backend,
            random_image_count=self.random_image_count,
            image_url_list=self.image_url_list,
            image_url_probe_count=self.image_url_probe_count,
        )


def load_image_url_list(path_or_inline: str) -> List[str]:
    """Load a list of image URLs / paths.

    Accepts either:
    - A path to a JSON file containing a list of strings.
    - A path to a text file with one URL per line (``#`` starts a comment).
    - An inline comma-separated list of URLs.
    """
    if not path_or_inline:
        return []

    # Inline comma list (only when it's not an existing file path).
    if "," in path_or_inline and not os.path.exists(path_or_inline):
        return [s.strip() for s in path_or_inline.split(",") if s.strip()]

    if not os.path.exists(path_or_inline):
        raise FileNotFoundError(f"image_url_list file not found: {path_or_inline}")

    with open(path_or_inline, "r") as f:
        text = f.read()

    text_stripped = text.strip()
    if text_stripped.startswith("["):
        try:
            data = json.loads(text_stripped)
            if isinstance(data, list):
                urls = [str(x).strip() for x in data if str(x).strip()]
                if urls:
                    return urls
        except json.JSONDecodeError:
            pass

    urls: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        urls.append(s)
    return urls


def fetch_image_from_source(url_or_path: str, timeout: float = 30.0) -> Image.Image:
    """Fetch a single image into a PIL Image, supporting http(s), data URIs,
    and local files."""
    if url_or_path.startswith(("http://", "https://")):
        resp = requests.get(url_or_path, timeout=timeout)
        resp.raise_for_status()
        buf = io.BytesIO(resp.content)
    elif url_or_path.startswith("data:"):
        _, _, b64 = url_or_path.partition(",")
        buf = io.BytesIO(pybase64.b64decode(b64))
    else:
        with open(url_or_path, "rb") as f:
            buf = io.BytesIO(f.read())
    return Image.open(buf).convert("RGB")


def parse_image_resolution(image_resolution: str) -> Tuple[int, int]:
    """Parse image resolution into (width, height).

    Supports presets '1080p', '720p', '360p' and custom 'heightxwidth' format
    (e.g., '1080x1920' means height=1080, width=1920).
    """
    resolution_to_size = {
        "4k": (3840, 2160),
        "1080p": (1920, 1080),
        "720p": (1280, 720),
        "360p": (640, 360),
    }
    if image_resolution in resolution_to_size:
        return resolution_to_size[image_resolution]

    res = image_resolution.strip().lower()
    if "x" in res:
        parts = res.split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            height = int(parts[0])
            width = int(parts[1])
            if height > 0 and width > 0:
                return (width, height)

    raise ValueError(
        f"Unsupported image resolution: {image_resolution}. "
        "Choose from 4k, 1080p, 720p, 360p, or provide custom 'heightxwidth' (e.g., 1080x1920)."
    )


def create_mm_data_row(
    text_prompt, images: list, images_base64, output_len, processor, backend
):
    try:
        if type(processor).__name__ == "Phi4MMProcessor":
            # <|endoftext10|> is the image token used in the phi-4-multimodal model.
            content_items = text_prompt.replace("image 1", "|endoftext10|")
        else:
            content_items = [
                {"type": "image", "image": {"url": image_base64}}
                for image_base64 in images_base64
            ]
            content_items.append({"type": "text", "text": text_prompt})
        prompt_str = processor.apply_chat_template(
            [{"role": "user", "content": content_items}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception as e:
        # Note (Xinyuan): This is a workaround for an issue where some tokenizers do not support content as a list. (e.g. InternVL)
        print(f"Error applying chat template: {e}, fallback to <image> tag")
        # Some tokenizers do not support list content; fall back to a placeholder in the text
        prompt_str = f"<image>{text_prompt}"

    # Calculate total tokens (text + vision)
    if type(processor).__name__ == "KimiK25Processor":
        medias = [{"type": "image", "image": img} for img in images]
        prompt_len = processor(
            text=prompt_str,
            medias=medias,
            return_tensors="pt",
        )["input_ids"].numel()
    else:
        prompt_len = processor(
            text=[prompt_str],
            images=images,
            padding=False,
            return_tensors="pt",
        )["input_ids"].numel()

    # Calculate text-only tokens
    try:
        # Create text-only version of the prompt
        text_only_prompt = processor.apply_chat_template(
            [{"role": "user", "content": text_prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        text_prompt_len = processor(
            text=[text_only_prompt],
            padding=False,
            return_tensors="pt",
        )["input_ids"].numel()
    except Exception:
        # Fallback: just tokenize the text prompt directly
        tokenizer_to_use = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        text_prompt_len = len(tokenizer_to_use.encode(text_prompt))

    # Vision tokens = total tokens - text tokens
    vision_prompt_len = prompt_len - text_prompt_len

    supported_backends = ["sglang", "sglang-native", "sglang-oai-chat"]
    if backend not in supported_backends:
        raise ValueError(
            f"Image dataset only supports backends: {supported_backends}, "
            f"got '{backend}'."
        )

    # sglang-oai-chat: server's chat handler applies chat template, so send raw text.
    # sglang/sglang-native: /generate does not apply chat template, so send prompt_str
    #         which contains image placeholder tokens needed by the multimodal processor.
    use_raw_prompt = backend == "sglang-oai-chat"

    return DatasetRow(
        prompt=text_prompt if use_raw_prompt else prompt_str,
        prompt_len=prompt_len,
        output_len=output_len,
        text_prompt_len=text_prompt_len,
        vision_prompt_len=vision_prompt_len,
        image_data=images_base64,
    )


def sample_image_requests(
    num_requests: int,
    image_count: int,
    input_len: int,
    output_len: int,
    range_ratio: float,
    processor: AutoProcessor,
    image_content: str,
    image_format: str,
    image_resolution: str,
    backend: str,
    random_image_count: bool = False,
    image_url_list: Optional[str] = None,
    image_url_probe_count: int = 1,
) -> List[DatasetRow]:
    """Generate requests with images.

    - If ``random_image_count`` is True, each request includes a random number of images between 1 and ``image_count``.
    - If ``random_image_count`` is False, each request includes exactly ``image_count`` images.
    - When ``image_url_list`` is provided, images are randomly sampled (with replacement)
      from that list of URLs / file paths. The raw URL strings are passed through to
      the server in ``image_data`` so the server fetches them; we additionally download
      each unique URL once locally so the multimodal processor can compute accurate
      ``prompt_len`` / ``vision_prompt_len``. ``image_resolution`` / ``image_format`` /
      ``image_content`` are ignored in this mode.
    - Supported resolutions: 4k (3840x2160), 1080p (1920x1080), 720p (1280x720), 360p (640x360),
      or custom 'heightxwidth' (e.g., 1080x1920).
    - Text lengths follow the 'random' dataset sampling rule. ``prompt_len``
      only counts text tokens and excludes image data.
    """

    use_url_list = bool(image_url_list)

    if use_url_list:
        url_pool = load_image_url_list(image_url_list)
        if not url_pool:
            raise ValueError(
                f"image_url_list '{image_url_list}' did not yield any URLs/paths"
            )
        print(
            f"[image-dataset] loaded {len(url_pool)} entries from "
            f"image_url_list='{image_url_list}'; ignoring "
            f"image_resolution/image_format/image_content in URL mode."
        )
        width = height = 0
    else:
        url_pool = []
        # Parse resolution (supports presets and 'heightxwidth')
        width, height = parse_image_resolution(image_resolution)

    # Determine image counts for each request
    if random_image_count:
        # Random number of images per request
        image_counts = np.random.randint(1, image_count + 1, size=num_requests)
        total_images = int(np.sum(image_counts))
    else:
        # Fixed number of images per request
        image_counts = np.full(num_requests, image_count)
        total_images = image_count * num_requests

    # Check for potentially problematic combinations and warn user
    if not use_url_list and width * height >= 1920 * 1080 and total_images >= 100:
        warnings.warn(
            f"High resolution ({width}x{height}) with {total_images} total images "
            f"may take a long time. Consider reducing resolution or image count.",
            UserWarning,
            stacklevel=2,
        )

    # Sample text lengths
    input_lens = compute_random_lens(
        full_len=input_len,
        range_ratio=range_ratio,
        num=num_requests,
    )
    output_lens = compute_random_lens(
        full_len=output_len,
        range_ratio=range_ratio,
        num=num_requests,
    )

    def _gen_random_image_data_uri(
        width: int = width, height: int = height
    ) -> Tuple[Image.Image, str, int]:
        if image_content == "blank":
            # Generate blank white image
            arr = np.full((height, width, 3), 255, dtype=np.uint8)
        else:
            # Generate random colored image
            arr = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format=image_format, quality=85)
        encoded = pybase64.b64encode(buf.getvalue()).decode("utf-8")
        image_data = f"data:image/{image_format};base64,{encoded}"
        image_bytes = len(image_data.encode("utf-8"))
        return img, image_data, image_bytes

    # Lazy per-URL PIL-image cache so the processor can tokenize without
    # re-downloading on every request.
    url_image_cache: Dict[str, Image.Image] = {}

    def _get_image_for_url(url: str) -> Image.Image:
        cached = url_image_cache.get(url)
        if cached is not None:
            return cached
        img = fetch_image_from_source(url)
        url_image_cache[url] = img
        return img

    # In URL-list mode, only download a small "probe" pool of images locally
    # (used purely to let the processor compute prompt_len / vision_prompt_len).
    # The per-request URLs sent to the server are still randomly sampled from
    # the full pool, so this does NOT change the server-side workload.
    probe_images: List[Image.Image] = []
    if use_url_list:
        probe_count = max(1, min(image_url_probe_count, len(url_pool)))
        # Sample probe URLs deterministically from the front of the pool so
        # repeated runs are reproducible.
        probe_urls = url_pool[:probe_count]
        print(
            f"[image-dataset] downloading {probe_count} probe image(s) for "
            f"local prompt_len estimation; per-request URLs will be sent to "
            f"the server unchanged. "
            f"(override with --image-url-probe-count)"
        )
        for u in probe_urls:
            probe_images.append(_get_image_for_url(u))

    # Vision-token cost only depends on (image_count, image_resolution).
    # In URL-list mode every request uses the same probe image(s), so once
    # we've measured ``vision_prompt_len`` for a given ``image_count`` we can
    # reuse it for every subsequent request and skip the heavy image
    # processor call (the expensive part is the per-image patch / grid_thw
    # computation, which is image-dimension-dependent and identical here).
    vision_len_cache: Dict[int, int] = {}

    def _text_only_prompt_len(text_prompt: str) -> int:
        """Cheap: tokenize the text-only chat-template (no images)."""
        try:
            text_only_prompt = processor.apply_chat_template(
                [{"role": "user", "content": text_prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            return processor(
                text=[text_only_prompt],
                padding=False,
                return_tensors="pt",
            )["input_ids"].numel()
        except Exception:
            tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            return len(tok.encode(text_prompt))

    dataset: List[DatasetRow] = []
    total_image_bytes = 0
    for i in range(num_requests):
        # Get the number of images for this request
        request_image_count = int(image_counts[i])

        # Generate text prompt
        text_prompt = gen_mm_prompt(
            processor.tokenizer,
            processor.image_token_id if hasattr(processor, "image_token_id") else None,
            int(input_lens[i]),
        )

        # Generate image list
        if use_url_list:
            sampled_urls = [
                url_pool[np.random.randint(0, len(url_pool))]
                for _ in range(request_image_count)
            ]
            total_image_bytes += sum(len(u.encode("utf-8")) for u in sampled_urls)

            cached_vision_len = vision_len_cache.get(request_image_count)
            if cached_vision_len is None:
                # First time we see this image_count: pay the full processor
                # cost once on the probe image(s), then cache the vision
                # token contribution for reuse by all subsequent requests
                # with the same image_count.
                images = [
                    probe_images[j % len(probe_images)]
                    for j in range(request_image_count)
                ]
                data_row = create_mm_data_row(
                    text_prompt,
                    list(images),
                    list(sampled_urls),
                    int(output_lens[i]),
                    processor,
                    backend,
                )
                measured_vision_len = int(data_row.vision_prompt_len or 0)
                vision_len_cache[request_image_count] = measured_vision_len
                print(
                    f"[image-dataset] cached vision_prompt_len="
                    f"{measured_vision_len} for image_count="
                    f"{request_image_count} (probe-derived, reused for "
                    f"subsequent requests)",
                    flush=True,
                )
            else:
                # Fast path: skip image processor entirely. Only re-tokenize
                # the (cheap) text portion and add the cached vision cost.
                text_prompt_len = _text_only_prompt_len(text_prompt)
                if backend == "sglang-oai-chat":
                    # Server applies chat template; send raw text + URLs.
                    prompt_field = text_prompt
                else:
                    content_items = [
                        {"type": "image", "image": {"url": u}} for u in sampled_urls
                    ]
                    content_items.append({"type": "text", "text": text_prompt})
                    try:
                        prompt_field = processor.apply_chat_template(
                            [{"role": "user", "content": content_items}],
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                    except Exception:
                        prompt_field = f"<image>{text_prompt}"
                data_row = DatasetRow(
                    prompt=prompt_field,
                    prompt_len=text_prompt_len + cached_vision_len,
                    output_len=int(output_lens[i]),
                    text_prompt_len=text_prompt_len,
                    vision_prompt_len=cached_vision_len,
                    image_data=list(sampled_urls),
                )
        else:
            images, images_base64, images_bytes = zip(
                *[_gen_random_image_data_uri() for _ in range(request_image_count)]
            )
            images = list(images)
            images_base64 = list(images_base64)
            total_image_bytes += sum(images_bytes)
            data_row = create_mm_data_row(
                text_prompt,
                list(images),
                list(images_base64),
                int(output_lens[i]),
                processor,
                backend,
            )

        dataset.append(data_row)

        # Light progress heartbeat so users can tell the dataset prep loop is
        # making progress (the per-request processor() call on large images
        # can otherwise look like a hang).
        if (i + 1) == 1 or (i + 1) % 50 == 0 or (i + 1) == num_requests:
            print(
                f"[image-dataset] prepared {i + 1}/{num_requests} requests",
                flush=True,
            )

    # Print statistics
    print(f"#Input tokens: {np.sum([x.prompt_len for x in dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in dataset])}")
    print(f"#Total images: {total_images}")

    if random_image_count:
        print(
            f"#Images per request: min={np.min(image_counts)}, max={np.max(image_counts)}, mean={np.mean(image_counts):.2f}"
        )
    else:
        print(f"#Images per request: {image_count} (fixed)")

    if use_url_list:
        print(
            f"\nCreated {len(dataset)} requests sampled from {len(url_pool)} image URL(s) "
            f"({len(url_image_cache)} unique fetched), average "
            f"{total_image_bytes // max(num_requests, 1)} URL bytes per request"
        )
    else:
        print(
            f"\nCreated {len(dataset)} {image_content} {image_format} images with average {total_image_bytes // num_requests} bytes per request"
        )
    # Confirm the bench-side multimodal special-token guards were exercised.
    log_special_token_filter_stats()
    return dataset
