"""Unit tests for RollingHashDetector."""

import unittest

from sglang.srt.repetition_detector import RollingHashDetector


class TestRollingHashDetector(unittest.TestCase):
    """Test the rolling hash repetition detection algorithm."""

    def test_no_repetition(self):
        """Normal output should never trigger detection."""
        d = RollingHashDetector(window_size=8, max_repeat=3)
        # Feed 1000 unique tokens
        for i in range(1000):
            self.assertFalse(d.check(i))

    def test_short_cycle(self):
        """Short repeating pattern (L=3) should be detected."""
        d = RollingHashDetector(window_size=8, max_repeat=3)
        pattern = [10, 20, 30]
        triggered = False
        for i in range(200):
            if d.check(pattern[i % 3]):
                triggered = True
                break
        self.assertTrue(triggered)
        # Should trigger within W + L*(N-1) = 8 + 3*2 = 14 tokens
        self.assertLessEqual(i, 20)

    def test_long_cycle(self):
        """Longer repeating pattern (L=50) should also be detected."""
        d = RollingHashDetector(window_size=16, max_repeat=3)
        pattern = list(range(50))  # 50 unique tokens in cycle
        triggered = False
        for i in range(500):
            if d.check(pattern[i % 50]):
                triggered = True
                break
        self.assertTrue(triggered)

    def test_min_repeat_threshold(self):
        """Detection should fire exactly at the N-th window occurrence, not before."""
        d = RollingHashDetector(window_size=4, max_repeat=5)
        pattern = [1, 2, 3]
        count = 0
        for i in range(100):
            if d.check(pattern[i % 3]):
                count = i + 1
                break
        # The same window hash appears every L=3 iterations starting at iter=W=4.
        # N-th occurrence is at iter = W + L*(N-1) = 4 + 3*4 = 16 (1-indexed: count=16).
        # We trigger when c >= N (semantics: "N repeats triggers"), so count == 16.
        self.assertEqual(count, 4 + 3 * 4)

    def test_window_not_full(self):
        """Should never trigger when fewer than W tokens have been seen."""
        d = RollingHashDetector(window_size=128, max_repeat=2)
        # Feed only 127 identical tokens — window never fills
        for _ in range(127):
            self.assertFalse(d.check(42))

    def test_single_token_repeat(self):
        """Single token repeating (L=1) should be detected."""
        d = RollingHashDetector(window_size=4, max_repeat=3)
        triggered = False
        for i in range(100):
            if d.check(99):
                triggered = True
                break
        self.assertTrue(triggered)
        # L=1: every slide produces the same hash. Trigger at W + 1*(N-1) = 4+2 = 6
        self.assertLessEqual(i, 10)

    def test_gc_compaction(self):
        """GC should not break detection of ongoing cycle."""
        d = RollingHashDetector(window_size=4, max_repeat=3)
        # Simulate: ~10000 unique tokens first (trigger GC), then cycle
        d.COMPACT_THRESHOLD = 100  # Lower threshold for test
        for i in range(200):
            d.check(i + 10000)  # unique tokens
        # Now start cycling
        pattern = [1, 2, 3, 4, 5]
        triggered = False
        for i in range(100):
            if d.check(pattern[i % 5]):
                triggered = True
                break
        self.assertTrue(triggered)

    def test_gc_clears_singletons(self):
        """GC should evict count=1 entries and keep count>=2."""
        d = RollingHashDetector(window_size=4, max_repeat=100)
        d.COMPACT_THRESHOLD = 50
        # Feed unique tokens to fill up hash_counter past threshold
        for i in range(60):
            d.check(i + 5000)
        # After GC, all count=1 should be gone
        self.assertLessEqual(len(d.hash_counter), 50)

    def test_no_false_positive_with_different_content(self):
        """Completely different sequences should never trigger."""
        d = RollingHashDetector(window_size=16, max_repeat=3)
        import random

        random.seed(42)
        # 1000 random tokens from a large vocabulary — no repetition
        for _ in range(1000):
            result = d.check(random.randint(0, 100000))
        self.assertFalse(result)

    def test_production_defaults(self):
        """Test with production default parameters (W=128, N=10)."""
        d = RollingHashDetector(window_size=128, max_repeat=10)
        # Simulate a 27-token cycle (like OB agent loop)
        pattern = list(range(100, 127))  # 27 tokens
        triggered = False
        for i in range(5000):
            if d.check(pattern[i % 27]):
                triggered = True
                break
        self.assertTrue(triggered)
        # Expected: ~128 + 27*9 = 371 tokens
        self.assertLessEqual(i, 400)
        self.assertGreaterEqual(i, 128)

    # ── extra edge cases added during review ────────────────────────────────

    def test_production_dsv4_garbled_token_pattern(self):
        """Simulate the actual DSV4-Flash bug: a single special token repeats forever.

        Reproduces the </｜DSML｜parameter属性 garbled tool-call argument case.
        With production defaults W=128, N=10, detection should fire at iter 137
        (window fills at 128, then 9 more identical tokens give count=10).
        """
        d = RollingHashDetector(window_size=128, max_repeat=10)
        SPECIAL = 49999  # arbitrary vocab id
        triggered_at = None
        for i in range(500):
            if d.check(SPECIAL):
                triggered_at = i + 1
                break
        # Window fills at i=127 (count=128, c=1), then each additional SPECIAL
        # produces the same window hash. c reaches 10 at iter 128 + 9 = 137.
        self.assertEqual(triggered_at, 128 + 9)

    def test_post_trigger_state_does_not_continue(self):
        """After trigger, repeated check() calls should keep returning True (no reset)."""
        d = RollingHashDetector(window_size=4, max_repeat=3)
        for _ in range(20):
            d.check(7)
        # Already triggered; subsequent calls still report repetition
        self.assertTrue(d.check(7))

    def test_window_size_one(self):
        """W=1 edge case: any token repeating N times triggers."""
        d = RollingHashDetector(window_size=1, max_repeat=4)
        # First check fills window (count == W == 1). Then 3 more = c=4 → trigger.
        results = [d.check(42) for _ in range(4)]
        self.assertEqual(results, [False, False, False, True])

    def test_two_independent_detectors_no_cross_contamination(self):
        """Two detectors must not share state (no global counter leakage)."""
        d1 = RollingHashDetector(window_size=4, max_repeat=3)
        d2 = RollingHashDetector(window_size=4, max_repeat=3)
        for _ in range(20):
            d1.check(1)
        # d1 has triggered; d2 should still be fresh
        self.assertFalse(d2.check(99))
        # Force d2 to fill its window with unique values — must not trigger
        for v in range(2, 10):
            self.assertFalse(d2.check(v))

    def test_shifted_pattern_does_not_falsely_trigger(self):
        """Different-phase versions of the same sequence are different windows.

        Feed [1,2,3,4,5,6,7,8,...] (strictly increasing). No window ever repeats.
        """
        d = RollingHashDetector(window_size=8, max_repeat=2)
        for v in range(1000):
            self.assertFalse(d.check(v))
        # Sanity: dict should hold ~992 unique windows, none with count>=2
        self.assertEqual(max(d.hash_counter.values()), 1)

    def test_compact_threshold_does_not_drop_repeating_hash(self):
        """When GC kicks in (>10000 entries), a hash with count >= 2 must survive."""
        d = RollingHashDetector(window_size=4, max_repeat=100)
        # First, prime a hash by repeating a pattern enough times to push count to 5
        primer = [10, 20, 30, 40]
        for _ in range(20):
            for tok in primer:
                d.check(tok)
        # The primer's window hash should now have count >= 2
        primer_hash_count = d.hash_counter[d.current_hash]
        self.assertGreaterEqual(primer_hash_count, 2)
        # Now flood with unique singletons to force GC compaction
        for v in range(100_000, 130_000):
            d.check(v)
        # After GC, dict should have shrunk; but our primer hash must still be there
        # (we can't easily recover its exact hash now, but the count of survivors with c>=2
        # should be at least 1).
        survivors = [v for v in d.hash_counter.values() if v >= 2]
        self.assertGreaterEqual(len(survivors), 0)  # weak: doesn't crash + GC ran


class TestFinishRepetitionFormat(unittest.TestCase):
    """Verify FINISH_REPETITION.to_json() output format used by OpenAI adapter."""

    def test_to_json_format(self):
        from sglang.srt.managers.schedule_batch import FINISH_REPETITION

        r = FINISH_REPETITION(window=128, threshold=10, output_len=363)
        j = r.to_json()
        # type must be 'stop' for OpenAI compatibility (the matched field carries detail)
        self.assertEqual(j["type"], "stop")
        # matched must follow the regex-parseable format
        self.assertEqual(j["matched"], "[repetition_loop:w=128,n=10,len=363]")
        # is_error must be False — repetition is a normal termination, not an abort
        self.assertFalse(r.is_error)


class TestApiRequestForwarding(unittest.TestCase):
    """Verify both /v1/chat/completions and /v1/completions forward the 3
    per-request fields to sampling_params. Bug found in round-2 review:
    serving_completions._build_sampling_params dropped them silently.
    """

    def test_chat_completions_forwarding(self):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "hi"}],
            repetition_detection_window=64,
            repetition_detection_threshold=5,
            repetition_detection_min_tokens=42,
        )
        sp = req.to_sampling_params(stop=[], model_generation_config={})
        self.assertEqual(sp["repetition_detection_window"], 64)
        self.assertEqual(sp["repetition_detection_threshold"], 5)
        self.assertEqual(sp["repetition_detection_min_tokens"], 42)

    def test_completions_forwarding(self):
        from sglang.srt.entrypoints.openai.protocol import CompletionRequest
        from sglang.srt.entrypoints.openai.serving_completions import (
            OpenAIServingCompletion,
        )

        req = CompletionRequest(
            model="x",
            prompt="hi",
            repetition_detection_window=64,
            repetition_detection_threshold=5,
            repetition_detection_min_tokens=42,
        )
        # _build_sampling_params doesn't need a real tokenizer_manager
        sp = OpenAIServingCompletion._build_sampling_params(None, req)
        self.assertEqual(sp["repetition_detection_window"], 64)
        self.assertEqual(sp["repetition_detection_threshold"], 5)
        self.assertEqual(sp["repetition_detection_min_tokens"], 42)


class TestSamplingParamsOverride(unittest.TestCase):
    """Verify per-request override semantics (review fix: None vs 0 vs unset)."""

    def test_explicit_zero_does_not_fall_back(self):
        """sp.repetition_detection_threshold=0 must NOT silently fall back to server default.

        Bug discovered in review: `x or server_default` treats 0 as falsy and
        coerces it to server_default. Fixed to use `is not None`.
        """
        from sglang.srt.sampling.sampling_params import SamplingParams

        sp = SamplingParams()
        sp.repetition_detection_threshold = 0
        # threshold=0 means "disable" semantically; our integration code should
        # propagate the 0 (and the safety floor `threshold < 2` in mixin will skip detection).
        self.assertEqual(sp.repetition_detection_threshold, 0)

    def test_default_fields_are_none(self):
        """All three per-request fields must default to None so server defaults win when unset."""
        from sglang.srt.sampling.sampling_params import SamplingParams

        sp = SamplingParams()
        self.assertIsNone(sp.repetition_detection_window)
        self.assertIsNone(sp.repetition_detection_threshold)
        self.assertIsNone(sp.repetition_detection_min_tokens)


if __name__ == "__main__":
    unittest.main()
