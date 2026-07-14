"""Tests for caller-supplied mm_hashes plumbing.

Verifies the contract that:
  1. GenerateReqInput.mm_hashes is an optional list of hex strings.
  2. MultimodalDataItem.set_pad_value() honors a pre-set hash and does NOT
     overwrite it via hash_feature().
  3. The derived pad_value is deterministic across requests with identical
     mm_hashes — the property external KV routers depend on.

The wiring step that copies GenerateReqInput.mm_hashes into per-item
MultimodalDataItem.hash lives in tokenizer_manager.py and is exercised by
the e2e serve tests; this file pins the unit-level invariants the wiring
relies on.
"""

import unittest
from unittest.mock import patch

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=2, suite="stage-b-test-1-gpu-small-amd")


class TestMmHashesContract(CustomTestCase):
    def test_generate_req_input_accepts_mm_hashes(self):
        """GenerateReqInput exposes mm_hashes as an optional field."""
        req = GenerateReqInput(
            text="hi",
            image_data=["http://example.com/img.png"],
            mm_hashes=["deadbeefcafe1234"],
        )
        self.assertEqual(req.mm_hashes, ["deadbeefcafe1234"])

    def test_generate_req_input_defaults_mm_hashes_to_none(self):
        """Absent mm_hashes preserves existing (None) behavior."""
        req = GenerateReqInput(text="hi")
        self.assertIsNone(req.mm_hashes)

    def test_set_pad_value_honors_preset_hash(self):
        """set_pad_value() must use a pre-set hash without recomputing."""
        vocab_size = 1_000_000
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=0xDEADBEEF,
            model_vocab_size=vocab_size,
        )
        # If hash_feature is invoked, the test fails — we patch it to
        # raise so any accidental recompute is loud.
        with patch(
            "sglang.srt.managers.mm_utils.hash_feature",
            side_effect=AssertionError(
                "hash_feature must NOT be called when hash is preset"
            ),
        ):
            item.set_pad_value()
        self.assertEqual(item.hash, 0xDEADBEEF)
        raw_pad_value = 0xDEADBEEF % (1 << 30)
        expected_pad_value = (
            raw_pad_value + vocab_size
            if raw_pad_value <= vocab_size
            else raw_pad_value
        )
        self.assertEqual(item.pad_value, expected_pad_value)

    def test_existing_pad_value_gets_vocab_offset_when_vocab_arrives(self):
        """A pad computed before vocab propagation must be corrected later."""
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=123,
            pad_value=123,
            model_vocab_size=1_000,
        )
        item.set_pad_value()
        self.assertEqual(item.pad_value, 1_123)

    def test_public_fixed_shift_pad_is_normalized_from_hash(self):
        """A public/main fixed-shift pad must not survive on the tracker branch."""
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=123,
            pad_value=1_000_123,
            model_vocab_size=1_000,
        )
        item.set_pad_value()
        self.assertEqual(item.pad_value, 1_123)

    def test_vocab_offset_is_idempotent_when_raw_pad_is_zero(self):
        """Repeated calls must not add vocab_size more than once."""
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=0,
            model_vocab_size=1_000,
        )
        item.set_pad_value()
        item.set_pad_value()
        self.assertEqual(item.pad_value, 1_000)

    @patch(
        "sglang.srt.managers.schedule_batch.envs.SGLANG_MM_SKIP_COMPUTE_HASH.get",
        return_value=True,
    )
    @patch("uuid.uuid4")
    def test_skip_compute_hash_still_applies_vocab_offset(
        self, mock_uuid4, _mock_skip_hash
    ):
        """Skipping feature hashing must not allow a normal token-id collision."""
        mock_uuid4.return_value.int = 123
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            model_vocab_size=1_000,
        )
        item.set_pad_value()
        self.assertEqual(item.hash, 123)
        self.assertEqual(item.pad_value, 1_123)

    def test_processor_output_refreshes_precomputed_padded_ids(self):
        """Scheduler-side vocab propagation must also refresh padded_input_ids."""
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=123,
            pad_value=123,
            offsets=[(1, 2)],
        )
        processor_output = MultimodalProcessorOutput(
            mm_items=[item],
            input_ids=[10, 20, 20, 30],
            padded_input_ids=[10, 123, 123, 30],
            model_vocab_size=1_000,
        )
        mm_inputs = MultimodalInputs.from_processor_output(processor_output)
        self.assertEqual(mm_inputs.mm_items[0].pad_value, 1_123)
        self.assertEqual(mm_inputs.padded_input_ids, [10, 1_123, 1_123, 30])

    def test_processor_output_from_dict_preserves_model_vocab_size(self):
        """Dictionary reconstruction must not drop the dynamic vocab offset."""
        output = MultimodalProcessorOutput.from_dict(
            {"mm_items": [], "model_vocab_size": 1_000}
        )
        self.assertEqual(output.model_vocab_size, 1_000)

    def test_set_pad_value_is_deterministic_across_items(self):
        """Two items with the same preset hash must derive the same pad_value."""
        a = MultimodalDataItem(modality=Modality.IMAGE, hash=0x123456789ABCDEF0)
        b = MultimodalDataItem(modality=Modality.IMAGE, hash=0x123456789ABCDEF0)
        # No feature payload — set_pad_value uses the preset hash.
        a.set_pad_value()
        b.set_pad_value()
        self.assertEqual(a.pad_value, b.pad_value)
        self.assertEqual(a.hash, b.hash)

    def test_set_pad_value_distinguishes_different_preset_hashes(self):
        """Distinct preset hashes must produce distinct pad_values."""
        a = MultimodalDataItem(modality=Modality.IMAGE, hash=0xAAAA)
        b = MultimodalDataItem(modality=Modality.IMAGE, hash=0xBBBB)
        a.set_pad_value()
        b.set_pad_value()
        self.assertNotEqual(a.pad_value, b.pad_value)


if __name__ == "__main__":
    unittest.main()
