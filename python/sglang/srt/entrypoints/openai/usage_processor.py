from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, final

from sglang.srt.entrypoints.openai.protocol import PromptTokensDetails, UsageInfo


@final
class UsageProcessor:
    """Stateless helpers that turn raw token counts into a UsageInfo."""

    @staticmethod
    def _build_prompt_details(
        cached_total: int,
        l1_total: int = 0,
        l2_total: int = 0,
        l3_total: int = 0,
    ) -> Optional[PromptTokensDetails]:
        if cached_total <= 0 and l1_total <= 0 and l2_total <= 0 and l3_total <= 0:
            return None
        if cached_total > 0 and l1_total == 0 and l2_total == 0 and l3_total == 0:
            l1_total = cached_total
        return PromptTokensDetails(
            cached_tokens=cached_total,
            l1_cached_tokens=l1_total,
            l2_cached_tokens=l2_total,
            l3_cached_tokens=l3_total,
        )

    @staticmethod
    def _aggregate_cache_breakdown(
        responses: List[Dict[str, Any]],
        n_choices: int,
    ) -> tuple:
        """Aggregate l1/l2/l3 breakdown from cached_tokens_details across prompts."""
        l1 = l2 = l3 = 0
        for i in range(0, len(responses), n_choices):
            details = responses[i]["meta_info"].get("cached_tokens_details")
            if details:
                l1 += details.get("device", 0)
                l2 += details.get("host", 0)
                l3 += details.get("storage", 0)
        return l1, l2, l3

    @staticmethod
    def calculate_response_usage(
        responses: List[Dict[str, Any]],
        n_choices: int = 1,
        enable_cache_report: bool = False,
    ) -> UsageInfo:
        completion_tokens = sum(
            r["meta_info"].get("completion_tokens", 0) for r in responses
        )
        prompt_tokens = sum(
            responses[i]["meta_info"].get("prompt_tokens", 0)
            for i in range(0, len(responses), n_choices)
        )

        # some API don't have reasoning_tokens semantics
        reasoning_tokens = sum(
            r["meta_info"].get("reasoning_tokens", 0) for r in responses
        )

        cached_details = None
        if enable_cache_report:
            cached_total = sum(
                responses[i]["meta_info"].get("cached_tokens", 0)
                for i in range(0, len(responses), n_choices)
            )
            l1, l2, l3 = UsageProcessor._aggregate_cache_breakdown(
                responses, n_choices
            )
            cached_details = UsageProcessor._build_prompt_details(
                cached_total, l1, l2, l3
            )

        return UsageProcessor.calculate_token_usage(
            prompt_tokens=prompt_tokens,
            reasoning_tokens=reasoning_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_details,
        )

    @staticmethod
    def calculate_streaming_usage(
        prompt_tokens: Mapping[int, int],
        reasoning_tokens: Mapping[int, int],
        completion_tokens: Mapping[int, int],
        cached_tokens: Mapping[int, int],
        n_choices: int,
        enable_cache_report: bool = False,
        cached_tokens_details: Optional[Mapping[int, Optional[Dict]]] = None,
    ) -> UsageInfo:
        total_prompt_tokens = sum(
            tok for idx, tok in prompt_tokens.items() if idx % n_choices == 0
        )
        total_reasoning_tokens = sum(reasoning_tokens.values())
        total_completion_tokens = sum(completion_tokens.values())

        cached_details = None
        if enable_cache_report:
            cached_total = sum(
                tok for idx, tok in cached_tokens.items() if idx % n_choices == 0
            )
            l1 = l2 = l3 = 0
            if cached_tokens_details:
                for idx, details in cached_tokens_details.items():
                    if idx % n_choices != 0 or not details:
                        continue
                    l1 += details.get("device", 0)
                    l2 += details.get("host", 0)
                    l3 += details.get("storage", 0)
            cached_details = UsageProcessor._build_prompt_details(
                cached_total, l1, l2, l3
            )

        return UsageProcessor.calculate_token_usage(
            prompt_tokens=total_prompt_tokens,
            reasoning_tokens=total_reasoning_tokens,
            completion_tokens=total_completion_tokens,
            cached_tokens=cached_details,
        )

    @staticmethod
    def calculate_token_usage(
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: Optional[int] = 0,
        cached_tokens: Optional[PromptTokensDetails] = None,
    ) -> UsageInfo:
        """Calculate token usage information"""
        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=cached_tokens,
            reasoning_tokens=reasoning_tokens,
        )
