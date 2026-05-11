import random
from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np

ASSISTANT_SUFFIX = "Assistant:"
SHAREGPT_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
MOONCAKE_DATASET_URL = {
    "mooncake": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl",
    "conversation": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl",
    "synthetic": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/synthetic_trace.jsonl",
    "toolagent": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl",
}


@dataclass
class DatasetRow:
    prompt: Any
    prompt_len: int
    output_len: int
    text_prompt_len: Optional[int] = None
    vision_prompt_len: Optional[int] = None
    image_data: Optional[List[str]] = None
    timestamp: Optional[float] = None
    routing_key: Optional[str] = None
    extra_request_body: Optional[Dict[str, Any]] = None  # Per-request API parameters

    def __post_init__(self):
        if self.text_prompt_len is None:
            self.text_prompt_len = self.prompt_len
        if self.vision_prompt_len is None:
            self.vision_prompt_len = 0
        if self.extra_request_body is None:
            self.extra_request_body = {}


@dataclass
class BaseDataset(ABC):
    @classmethod
    @abstractmethod
    def from_args(cls, args: Namespace) -> "BaseDataset": ...

    @abstractmethod
    def load(
        self,
        tokenizer: Any,
        model_id: Optional[str] = None,
    ) -> List[DatasetRow]: ...


def compute_random_lens(full_len: int, range_ratio: float, num: int) -> List[int]:
    # full_len=0 is valid for embedding benchmarks where no output tokens are generated
    if full_len <= 0:
        return [0] * num
    return np.random.randint(
        max(int(full_len * range_ratio), 1),
        full_len + 1,
        size=num,
    ).tolist()


# Counters for verifying that the special-token guards are wired into the
# prompt-generation hot path. Reported by ``log_special_token_filter_stats``.
_SCRUB_HIT_COUNT: Dict[str, int] = {}
_SCRUB_CALL_COUNT: int = 0


@lru_cache(maxsize=1)
def get_available_tokens(tokenizer):
    """Get token ids safe for random prompt generation.

    Excludes all special tokens (BOS/EOS/PAD/etc.) and any extra special
    tokens registered by the tokenizer (e.g. ``<|image_pad|>``,
    ``<|video_pad|>``, ``<|vision_start|>``, ``<|vision_end|>``,
    ``<|audio_pad|>``, chat template tokens). Random sampling that includes
    these would produce prompts that the multimodal processor / chat
    template misinterprets and can crash the scheduler in EPD mode.
    """
    vocab_ids = set(tokenizer.get_vocab().values())
    special_ids = set()
    for attr in ("all_special_ids", "additional_special_tokens_ids"):
        special_ids.update(getattr(tokenizer, attr, None) or [])
    # added_tokens_decoder maps id -> AddedToken; treat anything in it as
    # special even if the tokenizer didn't surface it via all_special_ids.
    added = getattr(tokenizer, "added_tokens_decoder", None) or {}
    special_ids.update(added.keys())
    available = list(vocab_ids - special_ids)
    # One-time visibility log (lru_cache makes this fire exactly once per
    # tokenizer object) so we can verify the bench-side filter is active.
    sample_excluded = sorted(special_ids)[:8]
    print(
        f"[bench-serving][prompt-gen] random-token pool: vocab={len(vocab_ids)}, "
        f"excluded_special_ids={len(special_ids)} (sample={sample_excluded}), "
        f"available={len(available)}",
        flush=True,
    )
    return available


@lru_cache(maxsize=1)
def get_special_token_strings(tokenizer):
    """Collect literal string forms of special tokens.

    Filtering by token *id* alone is insufficient: a sequence of ordinary BPE
    sub-tokens can decode into a string that exactly matches a special token
    surface form (e.g. ``"<|video_pad|>"``). When the server later runs the
    chat template + tokenizer over our prompt, those literal strings would be
    re-encoded back into the special token id and confuse the multimodal
    processor. We post-process generated text to strip them.
    """
    strings = set()
    for tok in getattr(tokenizer, "all_special_tokens", None) or []:
        if tok:
            strings.add(str(tok))
    added = getattr(tokenizer, "added_tokens_decoder", None) or {}
    for added_tok in added.values():
        s = getattr(added_tok, "content", None) or str(added_tok)
        if s:
            strings.add(s)
    # Drop empties just in case
    strings.discard("")
    result = tuple(strings)
    print(
        f"[bench-serving][prompt-gen] special-token surface forms to scrub "
        f"({len(result)}): {sorted(result)}",
        flush=True,
    )
    return result


def _scrub_special_token_strings(text: str, tokenizer) -> str:
    """Replace any literal special-token surface forms with a single space.

    Returns a tuple-like behaviour via the module-level counters: the call
    count is bumped unconditionally so callers can prove the patched code is
    running, and the per-token hit count is bumped only when a scrub fires.
    """
    global _SCRUB_CALL_COUNT
    _SCRUB_CALL_COUNT += 1
    hit_summary: Dict[str, int] = {}
    for s in get_special_token_strings(tokenizer):
        if s in text:
            n = text.count(s)
            text = text.replace(s, " ")
            _SCRUB_HIT_COUNT[s] = _SCRUB_HIT_COUNT.get(s, 0) + n
            hit_summary[s] = n
    if hit_summary:
        print(
            f"[bench-serving][prompt-gen] scrub call #{_SCRUB_CALL_COUNT}: "
            f"hit {hit_summary} (cumulative={_SCRUB_HIT_COUNT})",
            flush=True,
        )
    return text


def _emit_gen_log(fn_name: str, token_num: int, scrubbed_text: str) -> None:
    """Unconditional per-call log so users can verify the patched generator
    is actually being invoked by the benchmark, regardless of whether the
    scrub fallback fires."""
    # Compact log: only first/last call get the full prefix; middle calls log
    # at intervals to avoid spamming for benchmarks with many requests.
    if _SCRUB_CALL_COUNT == 1 or _SCRUB_CALL_COUNT % 50 == 0:
        preview = scrubbed_text[:60].replace("\n", " ")
        print(
            f"[bench-serving][prompt-gen] {fn_name} call #{_SCRUB_CALL_COUNT} "
            f"token_num={token_num} preview={preview!r}",
            flush=True,
        )


def log_special_token_filter_stats() -> None:
    """Print a summary of how many times the scrub fallback fired.

    Useful at the end of a benchmark run to confirm the bench-side guards
    against stray multimodal placeholder tokens are wired in. Zero hits is
    the expected case once the id-level filter is active; non-zero means the
    surface-form scrub successfully rescued some BPE-concat collisions.
    """
    print(
        f"[bench-serving][prompt-gen] scrub summary: calls={_SCRUB_CALL_COUNT}, "
        f"per_token_hits={_SCRUB_HIT_COUNT or '{}'}",
        flush=True,
    )


def gen_prompt(tokenizer, token_num):
    """Generate a random prompt of specified token length using tokenizer vocabulary."""
    all_available_tokens = get_available_tokens(tokenizer)
    selected_tokens = random.choices(all_available_tokens, k=token_num)
    text = _scrub_special_token_strings(tokenizer.decode(selected_tokens), tokenizer)
    _emit_gen_log("gen_prompt", token_num, text)
    return text


def gen_mm_prompt(tokenizer, image_pad_id, token_num):
    """Generate a random prompt of specified token length using tokenizer vocabulary.

    ``image_pad_id`` is kept for backward compatibility; it is implicitly
    excluded by :func:`get_available_tokens` along with every other
    special / multimodal placeholder token. The decoded text is also scrubbed
    of any literal special-token surface forms (e.g. ``"<|video_pad|>"``)
    that BPE sub-tokens could happen to spell out, since the server will
    re-tokenize our prompt and would otherwise misinterpret them as real
    multimodal placeholders.
    """
    del image_pad_id  # unused; handled by get_available_tokens
    all_available_tokens = get_available_tokens(tokenizer)
    selected_tokens = random.choices(all_available_tokens, k=token_num)
    text = _scrub_special_token_strings(tokenizer.decode(selected_tokens), tokenizer)
    _emit_gen_log("gen_mm_prompt", token_num, text)
    return text
