import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import torch
from PIL import Image

from sglang.srt.configs import deepseek_ocr as deepseek_ocr_config
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.processors.deepseek_ocr import (
    DeepseekOCRProcessor as RuntimeDeepseekOCRProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _TokenizerStub:
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return [7] if text else []


class _ImageTransformStub:
    mean = (0.5, 0.5, 0.5)

    @staticmethod
    def __call__(image):
        del image
        return torch.zeros((3, 4, 4), dtype=torch.float32)


def _make_config_processor(ocr2_mode: bool):
    processor = deepseek_ocr_config.DeepseekOCRProcessor.__new__(
        deepseek_ocr_config.DeepseekOCRProcessor
    )
    processor.tokenizer = _TokenizerStub()
    processor.patch_size = 16
    processor.downsample_ratio = 4
    processor.base_size = 1024
    processor.image_token = "<image>"
    processor.image_token_id = 99
    processor.ignore_id = -100
    processor.image_transform = _ImageTransformStub()
    processor.configure_ocr_mode(ocr2_mode)
    return processor


class TestDeepseekOCRPreprocessing(unittest.TestCase):
    def test_generation_specific_defaults(self):
        ocr1 = _make_config_processor(ocr2_mode=False)
        self.assertEqual(ocr1.image_size, 640)
        self.assertEqual(ocr1.max_crops, 9)
        self.assertEqual(ocr1.direct_resize_limit, 640)

        ocr2 = _make_config_processor(ocr2_mode=True)
        self.assertEqual(ocr2.image_size, 768)
        self.assertEqual(ocr2.max_crops, 6)
        self.assertEqual(ocr2.direct_resize_limit, 768)

    def test_generation_specific_crop_sizes_and_token_layouts(self):
        image = Image.new("RGB", (1200, 800))

        for ocr2_mode, expected_size, expected_max_crops, expected_tokens in (
            (False, 640, 9, 483),
            (True, 768, 6, 545),
        ):
            with self.subTest(ocr2_mode=ocr2_mode):
                processor = _make_config_processor(ocr2_mode)
                dynamic_preprocess = MagicMock(
                    return_value=([Image.new("RGB", (4, 4))] * 2, (2, 1))
                )
                with (
                    patch.object(
                        deepseek_ocr_config,
                        "dynamic_preprocess",
                        dynamic_preprocess,
                    ),
                    patch.object(
                        deepseek_ocr_config,
                        "pad_image",
                        return_value=Image.new("RGB", (4, 4)),
                    ),
                ):
                    output = processor.tokenize_with_images("<image>", [image])

                self.assertEqual(output[5], [expected_tokens])
                self.assertEqual(output[4].tolist(), [[2, 1]])
                self.assertEqual(output[2].shape, (1, 2, 3, 4, 4))
                dynamic_preprocess.assert_called_once()
                self.assertEqual(
                    dynamic_preprocess.call_args.kwargs,
                    {"max_num": expected_max_crops, "image_size": expected_size},
                )

    def test_noncrop_resize_threshold_and_token_layout_follow_generation(self):
        image = Image.new("RGB", (1200, 800))

        for ocr2_mode, expected_size, expected_tokens in (
            (False, 640, 111),
            (True, 768, 145),
        ):
            with self.subTest(ocr2_mode=ocr2_mode):
                processor = _make_config_processor(ocr2_mode)
                resize_image = MagicMock(return_value=Image.new("RGB", (4, 4)))
                with (
                    patch.object(
                        deepseek_ocr_config, "resize_image", resize_image
                    ),
                    patch.object(
                        deepseek_ocr_config,
                        "pad_image",
                        return_value=Image.new("RGB", (4, 4)),
                    ) as pad_image,
                ):
                    output = processor.tokenize_with_images(
                        "<image>", [image], cropping=False
                    )

                resize_image.assert_called_once_with(
                    image, (expected_size, expected_size)
                )
                self.assertEqual(
                    pad_image.call_args.args[1], (expected_size, expected_size)
                )
                self.assertEqual(output[5], [expected_tokens])

    def test_request_options_are_validated(self):
        processor = _make_config_processor(ocr2_mode=True)
        self.assertEqual(
            processor._resolve_preprocess_options(
                cropping=None,
                crop_mode=None,
                base_size=None,
                image_size=None,
            ),
            (True, 1024, 768),
        )

        with self.assertRaisesRegex(ValueError, "must match"):
            processor._resolve_preprocess_options(
                cropping=True,
                crop_mode=False,
                base_size=None,
                image_size=None,
            )
        with self.assertRaisesRegex(ValueError, "visual stride"):
            processor._resolve_preprocess_options(
                cropping=None,
                crop_mode=None,
                base_size=None,
                image_size=700,
            )


class TestDeepseekOCRRuntimeProcessor(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _make_runtime_processor(ocr2_mode=False):
        processor = RuntimeDeepseekOCRProcessor.__new__(
            RuntimeDeepseekOCRProcessor
        )
        processor._processor = SimpleNamespace(ocr2_mode=ocr2_mode)
        return processor

    def test_constructor_configures_ocr2_profile_from_official_auto_map(self):
        hf_config = SimpleNamespace(
            auto_map={
                "AutoModel": (
                    "modeling_deepseekocr2.DeepseekOCR2ForCausalLM"
                )
            },
            vision_config=SimpleNamespace(model_name="deeplip_b_l"),
            projector_config=SimpleNamespace(input_dim=2048),
        )
        hf_processor = SimpleNamespace(
            configure_ocr_mode=MagicMock(), image_token_id=99
        )

        def initialize_base(instance, _config, _args, processor, *args, **kwargs):
            del args, kwargs
            instance._processor = processor

        with (
            patch(
                "sglang.srt.multimodal.processors.deepseek_ocr."
                "BaseMultimodalProcessor.__init__",
                new=initialize_base,
            ),
            patch(
                "sglang.srt.multimodal.processors.deepseek_ocr."
                "MultimodalSpecialTokens.build",
                return_value=SimpleNamespace(image_token_id=99),
            ),
        ):
            RuntimeDeepseekOCRProcessor(hf_config, None, hf_processor)

        hf_processor.configure_ocr_mode.assert_called_once_with(True)

    def test_cache_hash_covers_local_crops_and_model_generation(self):
        processor = self._make_runtime_processor(ocr2_mode=False)

        def make_item(local_value):
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=torch.zeros((3, 4, 4), dtype=torch.float32),
            )
            item.images_crop = torch.full(
                (1, 2, 3, 4, 4), local_value, dtype=torch.float32
            )
            item.images_spatial_crop = torch.tensor([[2, 1]], dtype=torch.long)
            item.has_local_crops = True
            return item

        first = make_item(0)
        same = make_item(0)
        changed_crop = make_item(1)
        processor._set_deepseek_ocr_cache_hashes([first, same, changed_crop])

        self.assertEqual(first.hash, same.hash)
        self.assertNotEqual(first.hash, changed_crop.hash)
        self.assertIsNone(first.pad_value)

        processor._processor.ocr2_mode = True
        processor._set_deepseek_ocr_cache_hashes([same])
        self.assertNotEqual(first.hash, same.hash)

    def test_cache_hash_respects_skip_compute_hash(self):
        processor = self._make_runtime_processor()
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.zeros((3, 4, 4), dtype=torch.float32),
        )

        with (
            patch.object(
                RuntimeDeepseekOCRProcessor,
                "_digest_value",
                side_effect=AssertionError("feature hashing must be skipped"),
            ),
            patch(
                "sglang.srt.multimodal.processors.deepseek_ocr."
                "envs.SGLANG_MM_SKIP_COMPUTE_HASH.get",
                return_value=True,
            ),
            patch("uuid.uuid4", return_value=SimpleNamespace(int=123)),
        ):
            processor._set_deepseek_ocr_cache_hashes([item])

        self.assertEqual(item.hash, 123)
        self.assertIsNone(item.pad_value)

    async def test_mm_sampling_kwargs_are_forwarded(self):
        processor = self._make_runtime_processor()
        processor.mm_tokens = SimpleNamespace(image_token_id=99)
        processor.load_mm_data = AsyncMock(return_value="loaded")
        processor.process_and_combine_mm_data = MagicMock(
            return_value=([], torch.tensor([1, 2]), {})
        )
        request_obj = SimpleNamespace(
            mm_sampling_kwargs={
                "base_size": 1024,
                "image_size": 768,
                "crop_mode": True,
            }
        )

        output = await processor.process_mm_data_async(
            [b"image"], "<image>", request_obj=request_obj
        )

        processor.process_and_combine_mm_data.assert_called_once_with(
            "loaded",
            processor.mm_tokens,
            base_size=1024,
            image_size=768,
            crop_mode=True,
        )
        self.assertEqual(output.input_ids, [1, 2])

    async def test_unsupported_mm_sampling_kwarg_is_rejected(self):
        processor = self._make_runtime_processor()
        processor.mm_tokens = SimpleNamespace(image_token_id=99)

        with self.assertRaisesRegex(ValueError, "Unsupported"):
            await processor.process_mm_data_async(
                [b"image"],
                "<image>",
                request_obj=SimpleNamespace(mm_sampling_kwargs={"max_crops": 3}),
            )


if __name__ == "__main__":
    unittest.main()
