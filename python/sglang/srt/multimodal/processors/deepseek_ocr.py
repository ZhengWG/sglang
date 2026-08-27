import hashlib
from typing import Any, Dict, List, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.mm_utils import hash_feature
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.deepseek_ocr import DeepseekOCRForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils.hf_transformers.common import _is_deepseek_ocr2_model


class DeepseekOCRProcessor(BaseMultimodalProcessor):
    models = [DeepseekOCRForCausalLM]
    allowed_mm_sampling_kwargs = {
        "base_size",
        "crop_mode",
        "cropping",
        "image_size",
    }

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        ocr2_mode = _is_deepseek_ocr2_model(hf_config) or (
            str(
                getattr(getattr(hf_config, "vision_config", None), "model_name", "")
            ).lower()
            == "deepencoderv2"
            or getattr(getattr(hf_config, "projector_config", None), "input_dim", None)
            == 896
        )
        _processor.configure_ocr_mode(ocr2_mode)
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<image>", image_token_id=self._processor.image_token_id
        ).build(_processor)

    @staticmethod
    def _digest_bytes(payload: bytes) -> int:
        digest = hashlib.sha256(payload).digest()[:8]
        return int.from_bytes(digest, byteorder="big", signed=False)

    @classmethod
    def _digest_value(cls, value: Any) -> int:
        if value is None:
            return cls._digest_bytes(b"none")
        if isinstance(value, torch.Tensor):
            digest = hashlib.sha256()
            digest.update(b"tensor")
            digest.update(str(tuple(value.shape)).encode())
            digest.update(str(value.dtype).encode())
            digest.update(hash_feature(value).to_bytes(8, "big", signed=False))
            return int.from_bytes(
                digest.digest()[:8], byteorder="big", signed=False
            )
        if isinstance(value, bool):
            return cls._digest_bytes(b"bool:" + (b"1" if value else b"0"))
        return cls._digest_bytes(repr(value).encode())

    def _set_deepseek_ocr_cache_hashes(self, mm_items) -> None:
        if envs.SGLANG_MM_SKIP_COMPUTE_HASH.get():
            import uuid

            for item in mm_items:
                item.hash = uuid.uuid4().int
                item.pad_value = None
            return

        model_generation = (
            "deepseek_ocr2" if self._processor.ocr2_mode else "deepseek_ocr1"
        )
        for item in mm_items:
            digest = hashlib.sha256()
            for value in (
                model_generation,
                item.feature,
                getattr(item, "images_crop", None),
                getattr(item, "images_spatial_crop", None),
                getattr(item, "has_local_crops", None),
            ):
                digest.update(
                    self._digest_value(value).to_bytes(8, "big", signed=False)
                )
            item.hash = int.from_bytes(
                digest.digest()[:8], byteorder="big", signed=False
            )
            item.pad_value = None

    def _postprocess_mm_items_before_transport(self, mm_items, *, images):
        mm_items = super()._postprocess_mm_items_before_transport(
            mm_items, images=images
        )
        self._set_deepseek_ocr_cache_hashes(mm_items)
        return mm_items

    async def process_mm_data_async(
        self, image_data: List[Union[str, bytes]], input_text, *args, **kwargs
    ):
        processor_kwargs: Dict[str, Any] = {}
        request_obj = kwargs.get("request_obj")
        if request_obj is not None:
            processor_kwargs = getattr(request_obj, "mm_sampling_kwargs", None) or {}
        if not isinstance(processor_kwargs, dict):
            raise ValueError("DeepSeek OCR mm_sampling_kwargs must be a dict.")
        processor_kwargs = dict(processor_kwargs)
        unsupported_keys = set(processor_kwargs) - self.allowed_mm_sampling_kwargs
        if unsupported_keys:
            raise ValueError(
                "Unsupported DeepSeek OCR mm_sampling_kwargs: "
                f"{sorted(unsupported_keys)}"
            )

        base_output = await self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
        )

        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            base_output, self.mm_tokens, **processor_kwargs
        )

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            im_token_id=self.mm_tokens.image_token_id,
        )
