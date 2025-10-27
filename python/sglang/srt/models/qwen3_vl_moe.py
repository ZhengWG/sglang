# Copyright 2025 Qwen Team
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only Qwen3-VL model compatible with HuggingFace weights."""
import logging
from functools import lru_cache, partial
from typing import Callable, Iterable, List, Literal, Optional, Tuple, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BatchFeature
from transformers.activations import ACT2FN
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionRotaryEmbedding,
)

from sglang.srt.configs.qwen3_vl import Qwen3VLMoeConfig, Qwen3VLMoeVisionConfig
from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_rank,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_moe import Qwen3MoeModel, Qwen3MoeWeightLoaderMixin
from sglang.srt.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLForConditionalGeneration,
)
from sglang.srt.utils import add_prefix
from sglang.srt.utils.hf_transformers_utils import get_processor

logger = logging.getLogger(__name__)

cached_get_processor = lru_cache(get_processor)


class Qwen3VLMoeForConditionalGeneration(
    Qwen3VLForConditionalGeneration, Qwen3MoeWeightLoaderMixin
):
    def __init__(
        self,
        *,
        config: Qwen3VLMoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super(Qwen3VLForConditionalGeneration, self).__init__()
        self.config = config

        self.visual = Qwen3_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            # NOTE: Qwen3-VL vision encoder currently supports BitsAndBytes 4-bit quantization.
            # Other quantization methods (e.g., GPTQ, AWQ) are untested and may not be supported.
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
        )

        self.is_multimodal_embedding = False
        if (
            hasattr(config, "is_multimodal_embedding")
            and config.is_multimodal_embedding
        ):
            # build a dummy model for text embedding
            self.model = nn.Module()
            model_prefix = add_prefix("model", prefix)
            self.model.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", model_prefix),
            )
            setattr(self.model, "get_input_embeddings", lambda: self.model.embed_tokens)
            self.is_multimodal_embedding = True
        else:
            self.model = Qwen3MoeModel(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("model", prefix),
            )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # deepstack
        self.deepstack_visual_indexes = self.visual.deepstack_visual_indexes
        self.num_deepstack_embeddings = len(self.deepstack_visual_indexes)

    @property
    def use_deepstack(self) -> bool:
        return hasattr(self, "deepstack_visual_indexes")

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in qwen-vl, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        get_multimodal_embedding: bool = False,
    ):
        """Run forward pass for Qwen3-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen2-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
                (Use input_metadata.mrope_positions to replace it)
        """
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        if not (
            forward_batch.forward_mode.is_decode()
            or not forward_batch.contains_image_inputs()
        ):
            if self.is_mrope_enabled:
                assert positions.ndim == 2 and positions.size(0) == 3, (
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.size()}"
                )

        mm_embeds = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            use_deepstack=self.use_deepstack,
            get_multimodal_embedding=get_multimodal_embedding,
        )

        if get_multimodal_embedding:
            if self.use_deepstack:
                hidden_states, deepstack_embeds = mm_embeds
            else:
                hidden_states = mm_embeds
            return EmbeddingPoolerOutput(
                embeddings=hidden_states,
                deepstack_embeddings=deepstack_embeds if self.use_deepstack else None,
            )
        hidden_states = mm_embeds

        if not get_embedding:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights using the shared implementation from Qwen3MoeWeightLoaderMixin."""
        self.load_weights_impl(weights)


EntryClass = Qwen3VLMoeForConditionalGeneration
