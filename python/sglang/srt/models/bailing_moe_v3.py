# coding=utf-8
# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
from __future__ import annotations

import copy
import logging
import os
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.configs import KimiLinearConfig
from sglang.srt.distributed import (
    divide,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.nsa.utils import nsa_use_prefill_cp
from sglang.srt.layers.communicator import (
    AttentionInputs,
    LayerScatterModes,
    get_attn_tp_context,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.ngpt import ScaleUpNorm, apply_hyper_spherical_fusion, l2norm
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    is_fp8_fnuz,
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    block_quant_to_tensor_quant,
    channel_quant_to_tensor_quant,
    normalize_e4m3fn_to_e4m3fnuz,
    requant_weight_ue8m0_inplace,
)
from sglang.srt.layers.quantization.int8_utils import (
    block_dequant as int8_block_dequant,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA, DeepseekV2MLP, _is_hip
from sglang.srt.models.kimi_linear import KimiDeltaAttention
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    BumpAllocator,
    add_prefix,
    bind_or_assign,
    cpu_has_amx_support,
    get_bool_env_var,
    get_device_sm,
    is_cpu,
    is_cuda,
    is_flashinfer_available,
    is_gfx95_supported,
    is_hip,
    is_npu,
    is_nvidia_cublas_cu12_version_ge_12_9,
    is_sm100_supported,
    make_layers,
    set_weight_attrs,
    use_intel_amx_backend,
)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_device_sm = get_device_sm()
_is_gfx95_supported = is_gfx95_supported()

_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported

if _use_aiter_gfx95:

    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
    )
    from aiter.ops.triton.fused_fp8_quant import (
        fused_flatten_fp8_group_quant,
        fused_rms_fp8_group_quant,
    )

    from sglang.srt.layers.quantization.rocm_mxfp4_utils import (
        batched_gemm_afp4wfp4_pre_quant,
        fused_flatten_mxfp4_quant,
        fused_rms_mxfp4_quant,
    )
    from sglang.srt.layers.rocm_linear_utils import fused_qk_rope_cat_and_cache_mla

if _is_cuda:
    from sgl_kernel import awq_dequantize, bmm_fp8
elif _is_cpu and _is_cpu_amx_available:
    pass
elif _is_hip:
    from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope import (
        decode_attention_fwd_grouped_rope,
    )
    from sglang.srt.layers.quantization.awq_triton import (
        awq_dequantize_triton as awq_dequantize,
    )

else:
    from vllm._custom_ops import awq_dequantize

if _is_hip:
    pass

_is_cublas_ge_129 = is_nvidia_cublas_cu12_version_ge_12_9()
_is_flashinfer_available = is_flashinfer_available()
_is_sm100_supported = is_cuda() and is_sm100_supported()

FORWARD_ABSORB_CORE_ATTENTION_BACKENDS = [
    "fa3",
    "nsa",
    "flashinfer",
    "cutlass_mla",
    "trtllm_mla",
    "ascend",
]


class DsV3MLA(DeepseekV2AttentionMLA):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        layer_id: int = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        skip_rope: bool = False,
    ) -> None:
        super().__init__(
            config,
            hidden_size,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
            rope_theta,
            rope_scaling,
            max_position_embeddings,
            quant_config,
            reduce_results,
            layer_id,
            prefix,
            alt_stream,
            skip_rope,
        )
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        self.gated_attention_proj_granularity_type = getattr(
            config, "gated_attention_proj_granularity_type", None
        )
        # gated_attn now not support NPU
        if self.gated_attention_proj_granularity_type == "head_wise":
            self.g_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.output_gate",
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
        elif self.gated_attention_proj_granularity_type == "element_wise":
            self.g_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.v_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.output_gate",
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
        else:
            self.g_proj = None

    def forward_normal_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        if self.q_lora_rank is not None:
            q, latent_cache = (
                get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            )

            # NSA Indexer: cache quantized keys, auto-skip topk for sequences <= nsa_index_topk

            if self.use_nsa:
                q_lora = self.q_a_layernorm(q)
                q = self.q_b_proj(q_lora)[0].view(
                    -1, self.num_local_heads, self.qk_head_dim
                )
                _ = self.indexer(
                    x=hidden_states,
                    q_lora=q_lora,
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=self.layer_id,
                    return_indices=False,
                )
            elif _use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.uint8:
                # MXFP4: fused RMSNorm + quant
                q, _, _, _ = fused_rms_mxfp4_quant(
                    q,
                    self.q_a_layernorm.weight,
                    self.q_a_layernorm.variance_epsilon,
                    None,
                    None,
                    None,
                )
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            elif _use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.float8_e4m3fn:

                q, _, _, _ = fused_rms_fp8_group_quant(
                    q,
                    self.q_a_layernorm.weight,
                    self.q_a_layernorm.variance_epsilon,
                    None,
                    None,
                    None,
                    group_size=128,
                    dtype_quant=torch.float8_e4m3fn,
                    res1=None,
                    output_unquantized_inp1=False,
                )
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            else:
                q = self.q_a_layernorm(q)
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)

        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)

        if _use_aiter_gfx95 and self.kv_b_proj.weight.dtype == torch.float8_e4m3fn:

            kv_a_quanted, kv_a, _, _ = fused_rms_fp8_group_quant(
                kv_a,
                self.kv_a_layernorm.weight,
                self.kv_a_layernorm.variance_epsilon,
                None,
                None,
                None,
                group_size=128,
                dtype_quant=torch.float8_e4m3fn,
                res1=None,
                output_unquantized_inp1=True,  # return unqaunt kv_a
            )

        else:
            kv_a = self.kv_a_layernorm(kv_a)

        # kv_a = self.kv_a_layernorm(kv_a)

        k_pe = latent_cache[:, :, self.kv_lora_rank :]
        if self.rotary_emb is not None:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe

        self._set_mla_kv_buffer(latent_cache, kv_a, k_pe, forward_batch)
        if (
            forward_batch.mha_one_shot
            and sum(forward_batch.extend_prefix_lens_cpu) != 0
        ):
            if self.use_nsa and self.kv_cache_dtype == "fp8_e4m3":
                # FP8 path: dequantize NSA-specific FP8 format to BF16
                kv_a, k_pe = self._get_mla_kv_buffer_from_fp8(forward_batch)
            else:
                # BF16/FP16 path: directly fetch from cache
                kv_a, k_pe = self._get_mla_kv_buffer(
                    forward_batch.fetch_mha_one_shot_kv_indices(),
                    q.dtype,
                    forward_batch,
                )
        if _use_aiter_gfx95 and self.kv_b_proj.weight.dtype == torch.float8_e4m3fn:
            kv = self.kv_b_proj(
                kv_a_quanted,
            )[0]
        else:
            kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]

        k = self._concat_and_cast_mha_k(k_nope, k_pe, forward_batch)
        return q, k, v, forward_batch, hidden_states

    def forward_normal_core(self, q, k, v, forward_batch, hidden_states):
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)

        if self.g_proj is not None:
            gate, _ = self.g_proj(hidden_states)
            gate = F.sigmoid(gate.float()).type_as(hidden_states)
            if self.gated_attention_proj_granularity_type == "head_wise":
                # gate shape: [seq_len, head_num]
                attn_output = (
                    attn_output.view(-1, self.num_local_heads, self.v_head_dim)
                    * gate[:, :, None]
                )
                attn_output = attn_output.view(
                    -1, self.num_local_heads * self.v_head_dim
                )
            else:
                # gate shape: [seq_len, head_num*head_dim]
                attn_output = attn_output * gate

        output, _ = self.o_proj(attn_output)
        return output

    def forward_absorb_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        q_lora = None
        if self.q_lora_rank is not None:
            q, latent_cache = (
                get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            )
            k_nope = latent_cache[..., : self.kv_lora_rank]

            # overlap qk norm
            if self.alt_stream is not None and get_is_capture_mode():
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                q = self.q_a_layernorm(q)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = self.kv_a_layernorm(k_nope)
                current_stream.wait_stream(self.alt_stream)
            else:
                if _use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.uint8:
                    q, _, k_nope, *_ = fused_rms_mxfp4_quant(
                        q,
                        self.q_a_layernorm.weight,
                        self.q_a_layernorm.variance_epsilon,
                        k_nope,
                        self.kv_a_layernorm.weight,
                        self.kv_a_layernorm.variance_epsilon,
                    )
                else:
                    if (
                        _use_aiter_gfx95
                        and self.q_b_proj.weight.dtype == torch.float8_e4m3fn
                    ):

                        q, _, k_nope, _ = fused_rms_fp8_group_quant(
                            q,
                            self.q_a_layernorm.weight,
                            self.q_a_layernorm.variance_epsilon,
                            k_nope,
                            self.kv_a_layernorm.weight,
                            self.kv_a_layernorm.variance_epsilon,
                            group_size=128,
                            dtype_quant=torch.float8_e4m3fn,
                            res1=None,
                            output_unquantized_inp1=False,
                        )

                    else:
                        q = self.q_a_layernorm(q)
                        k_nope = self.kv_a_layernorm(k_nope)

            # q_lora needed by indexer
            if self.use_nsa:
                q_lora = q

            k_nope = k_nope.unsqueeze(1)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

        if self.use_deep_gemm_bmm:
            q_nope_val, q_nope_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(q_nope.transpose(0, 1))
            )
            q_nope_out = q_nope.new_empty(
                (self.num_local_heads, aligned_m, self.kv_lora_rank)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (q_nope_val, q_nope_scale),
                (self.w_kc, self.w_scale_k),
                q_nope_out,
                masked_m,
                expected_m,
            )
            q_nope_out = q_nope_out[:, :expected_m, :]
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if _use_aiter_gfx95 and self.w_kc.dtype == torch.uint8:
                x = q_nope.transpose(0, 1)
                q_nope_out = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_kc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_kc.transpose(-2, -1),
                    self.w_scale_k.transpose(-2, -1),
                    torch.bfloat16,
                    q_nope_out,
                )
            else:
                if _use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn:

                    q_nope_out = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                        X=q_nope,
                        WQ=self.w_kc.transpose(-1, -2),
                        w_scale=self.w_scale,
                        group_size=128,
                        YQ=None,  # allocate (B, M, N)
                        transpose_bm=False,  # (B, M, N)
                        transpose_bm_in=True,  # (M, B, K)
                        dtype=torch.bfloat16,
                    )

                else:
                    q_nope_out = torch.bmm(
                        q_nope.to(torch.bfloat16).transpose(0, 1),
                        self.w_kc.to(torch.bfloat16) * self.w_scale,
                    )

        elif self.w_kc.dtype == torch.float8_e4m3fn:
            # fix bmm_fp8 error under cublas12.9 caused by bumpallocator, detail in pr#11612
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                q_nope.transpose(0, 1),
                (
                    torch.zeros((1,), dtype=torch.float32, device=q_nope.device)
                    if _is_cublas_ge_129
                    else zero_allocator.allocate(1)
                ),
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)

        if (
            self.rotary_emb is not None
            and (not self._fuse_rope_for_trtllm_mla(forward_batch))
            and (not _use_aiter or not _is_gfx95_supported or self.use_nsa)
        ):
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        if nsa_use_prefill_cp(forward_batch):
            # support allgather+rerrange
            k_nope, k_pe = self.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )
        topk_indices = None
        if q_lora is not None:
            topk_indices = self.indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=self.layer_id,
            )

        return (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
            llama_4_scaling,
            hidden_states,
        )

    def forward_absorb_core(
        self,
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
        hidden_states,
    ):
        save_kv_cache = True

        if self.current_attention_backend in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
            extra_args = {}
            if self._fuse_rope_for_trtllm_mla(forward_batch):
                extra_args = {
                    "cos_sin_cache": self.rotary_emb.cos_sin_cache,
                    "is_neox": self.rotary_emb.is_neox_style,
                    "llama_4_scaling": llama_4_scaling,
                }

            attn_output = self.attn_mqa(
                q_nope_out,
                k_nope,
                k_nope,
                forward_batch,
                q_rope=q_pe,
                k_rope=k_pe,
                **extra_args,
                **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
            )
        else:
            if _use_aiter_gfx95:
                cos = self.rotary_emb.cos_cache
                sin = self.rotary_emb.sin_cache

                kv_cache_dtype = (
                    fp8_dtype if self.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
                )

                q, _, _, k = fused_qk_rope_cat_and_cache_mla(
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    forward_batch.token_to_kv_pool.get_key_buffer(
                        self.attn_mqa.layer_id
                    ),
                    forward_batch.out_cache_loc,
                    positions,
                    cos,
                    sin,
                    self.attn_mqa.k_scale,
                    self.rotary_emb.is_neox_style,
                    q_out_dtype=kv_cache_dtype,
                )

                save_kv_cache = False
            else:
                q = torch.cat([q_nope_out, q_pe], dim=-1)
                k = torch.cat([k_nope, k_pe], dim=-1)

            # Apply llama 4 scaling if provided
            if llama_4_scaling is not None:
                q *= llama_4_scaling

            attn_output = self.attn_mqa(
                q,
                k,
                k_nope,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
            )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if self.use_deep_gemm_bmm:
            attn_output_val, attn_output_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(
                    attn_output.transpose(0, 1)
                )
            )
            attn_bmm_output = attn_output.new_empty(
                (self.num_local_heads, aligned_m, self.v_head_dim)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (attn_output_val, attn_output_scale),
                (self.w_vc, self.w_scale_v),
                attn_bmm_output,
                masked_m,
                expected_m,
            )
            attn_bmm_output = (
                attn_bmm_output[:, :expected_m, :].transpose(0, 1).flatten(1, 2)
            )
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if _use_aiter_gfx95 and self.w_vc.dtype == torch.uint8:
                x = attn_output.transpose(0, 1)
                attn_bmm_output = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_vc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_vc.transpose(-2, -1),
                    self.w_scale_v.transpose(-2, -1),
                    torch.bfloat16,
                    attn_bmm_output,
                )
            else:
                if _use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn:
                    attn_bmm_output = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                        X=attn_output,
                        WQ=self.w_vc.transpose(-1, -2),
                        w_scale=self.w_scale,
                        group_size=128,
                        YQ=None,
                        transpose_bm=False,
                        transpose_bm_in=True,
                        dtype=torch.bfloat16,
                    )
                else:
                    attn_bmm_output = torch.bmm(
                        attn_output.to(torch.bfloat16).transpose(0, 1),
                        self.w_vc.to(torch.bfloat16) * self.w_scale,
                    )

            if self.o_proj.weight.dtype == torch.uint8:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_mxfp4_quant(attn_bmm_output)
            elif self.o_proj.weight.dtype == torch.float8_e4m3fn:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_fp8_group_quant(
                    attn_bmm_output, group_size=128, dtype_quant=torch.float8_e4m3fn
                )
            else:
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                (
                    torch.zeros((1,), dtype=torch.float32, device=attn_output.device)
                    if _is_cublas_ge_129
                    else zero_allocator.allocate(1)
                ),
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
            attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        else:
            if is_in_piecewise_cuda_graph():
                # torch dynamo requires out= op was called where output tensor was non-contiguous
                attn_bmm_output = (
                    torch.bmm(attn_output.transpose(0, 1), self.w_vc)
                    .transpose(0, 1)
                    .flatten(1, 2)
                )
            else:
                attn_bmm_output = torch.empty(
                    (attn_output.shape[0], self.num_local_heads * self.v_head_dim),
                    dtype=attn_output.dtype,
                    device=attn_output.device,
                )
                torch.bmm(
                    attn_output.transpose(0, 1),
                    self.w_vc,
                    out=attn_bmm_output.view(
                        -1, self.num_local_heads, self.v_head_dim
                    ).transpose(0, 1),
                )

        if self.g_proj is not None:
            gate, _ = self.g_proj(hidden_states)
            gate = F.sigmoid(gate.float()).type_as(hidden_states)
            if self.gated_attention_proj_granularity_type == "head_wise":
                # gate shape: [seq_len, head_num]
                attn_bmm_output = (
                    attn_bmm_output.view(-1, self.num_local_heads, self.v_head_dim)
                    * gate[:, :, None]
                )
                attn_bmm_output = attn_bmm_output.view(
                    -1, self.num_local_heads * self.v_head_dim
                )
            else:
                # gate shape: [seq_len, head_num*head_dim]
                attn_bmm_output = attn_bmm_output * gate

        output, _ = self.o_proj(attn_bmm_output)

        return output

    def forward_absorb_fused_mla_rope_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        enable_rope_fusion = (
            os.getenv("SGLANG_FUSED_MLA_ENABLE_ROPE_FUSION", "1") == "1"
        )
        # NOTE: hidden_states can be a tuple for some quantization paths.
        # For shape/device/dtype, use the first tensor; still pass the original
        # hidden_states through linear ops which may accept tuple inputs.
        hidden_states_tensor = (
            hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        )

        q_len = hidden_states_tensor.shape[0]
        q_input = hidden_states_tensor.new_empty(
            q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim
        )
        if self.q_lora_rank is not None:
            q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                self.w_kc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                q_nope.transpose(0, 1),
                zero_allocator.allocate(1),
                dtype=torch.float8_e4m3fn,
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
        q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)
        v_input = latent_cache[..., : self.kv_lora_rank]
        v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)
        k_input = latent_cache.unsqueeze(1)
        k_input[..., : self.kv_lora_rank] = v_input

        if not enable_rope_fusion:
            k_pe = k_input[..., self.kv_lora_rank :]
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            q_input[..., self.kv_lora_rank :] = q_pe
            k_input[..., self.kv_lora_rank :] = k_pe
            k_pe_output = None
        else:
            k_pe_output = torch.empty_like(k_input[..., self.kv_lora_rank :])

        q_input[..., self.kv_lora_rank :] = q_pe

        # attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        # Use Fused ROPE with use_rope=OFF.
        attn_output = torch.empty(
            (q_len, self.num_local_heads, self.kv_lora_rank),
            dtype=q.dtype,
            device=q.device,
        )
        attn_logits, _, kv_indptr, kv_indices, _, _, _ = (
            forward_batch.attn_backend.forward_metadata
        )
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        num_kv_split = forward_batch.attn_backend.num_kv_splits
        sm_scale = self.attn_mqa.scaling
        if attn_logits is None:
            attn_logits = torch.empty(
                (
                    forward_batch.batch_size,
                    self.num_local_heads,
                    num_kv_split,
                    self.kv_lora_rank + 1,
                ),
                dtype=torch.float32,
                device=q.device,
            )

        # save current latent cache.
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mqa, forward_batch.out_cache_loc, k_input, None
        )
        key_cache_buf = forward_batch.token_to_kv_pool.get_key_buffer(
            self.attn_mqa.layer_id
        )
        val_cache_buf = key_cache_buf[..., : self.kv_lora_rank]

        return (
            q_input,
            key_cache_buf,
            val_cache_buf,
            attn_output,
            kv_indptr,
            kv_indices,
            k_pe_output,
            cos_sin_cache,
            positions,
            attn_logits,
            num_kv_split,
            sm_scale,
            enable_rope_fusion,
            k_input,
            forward_batch,
            zero_allocator,
            hidden_states,
        )

    def forward_absorb_fused_mla_rope_cpu_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        assert self.q_lora_rank is not None and use_intel_amx_backend(
            self
        ), "forward_absorb_fused_mla_rope_cpu_prepare requires q_lora_rank is not None and use_intel_amx_backend"

        q_input, k_input, v_input = (
            torch.ops.sgl_kernel.qkv_proj_with_rope_fused_weight(
                hidden_states,
                self.fused_qkv_a_proj_with_mqa.weight,
                self.q_b_proj.weight,
                self.w_kc,
                self.q_a_layernorm.weight,
                self.kv_a_layernorm.weight,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.kv_a_layernorm.variance_epsilon,
                self.qkv_proj_with_rope_is_int8,
                self.qkv_proj_with_rope_is_fp8,
                (
                    self.fused_qkv_a_proj_with_mqa.weight_scale
                    if self.qkv_proj_with_rope_is_int8
                    else (
                        self.fused_qkv_a_proj_with_mqa.weight_scale_inv
                        if self.qkv_proj_with_rope_is_fp8
                        else None
                    )
                ),
                (
                    self.q_b_proj.weight_scale
                    if self.qkv_proj_with_rope_is_int8
                    else (
                        self.q_b_proj.weight_scale_inv
                        if self.qkv_proj_with_rope_is_fp8
                        else None
                    )
                ),
                True,  # is_vnni
                self.weight_block_size,
                self.q_lora_rank,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            )
        )
        return (q_input, k_input, v_input, forward_batch, zero_allocator, hidden_states)

    def forward_absorb_fused_mla_rope_core(
        self,
        q_input,
        key_cache_buf,
        val_cache_buf,
        attn_output,
        kv_indptr,
        kv_indices,
        k_pe_output,
        cos_sin_cache,
        positions,
        attn_logits,
        num_kv_split,
        sm_scale,
        enable_rope_fusion,
        k_input,
        forward_batch,
        zero_allocator,
        hidden_states,
    ):
        decode_attention_fwd_grouped_rope(
            q_input,
            key_cache_buf,
            val_cache_buf,
            attn_output,
            kv_indptr,
            kv_indices,
            k_pe_output,
            self.kv_lora_rank,
            self.rotary_emb.rotary_dim,
            cos_sin_cache,
            positions,
            attn_logits,
            num_kv_split,
            sm_scale,
            logit_cap=self.attn_mqa.logit_cap,
            use_rope=enable_rope_fusion,
            is_neox_style=self.rotary_emb.is_neox_style,
        )

        if enable_rope_fusion:
            k_input[..., self.kv_lora_rank :] = k_pe_output
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self.attn_mqa, forward_batch.out_cache_loc, k_input, None
            )

        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                self.w_vc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                zero_allocator.allocate(1),
                dtype=torch.float8_e4m3fn,
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
        else:
            attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

        if self.g_proj is not None:
            gate, _ = self.g_proj(hidden_states)
            gate = F.sigmoid(gate.float()).type_as(hidden_states)
            if self.gated_attention_proj_granularity_type == "head_wise":
                # gate shape: [seq_len, head_num]
                attn_output = (
                    attn_output.view(-1, self.num_local_heads, self.v_head_dim)
                    * gate[:, :, None]
                )
                attn_output = attn_output.view(
                    -1, self.num_local_heads * self.v_head_dim
                )
            else:
                # gate shape: [seq_len, head_num*head_dim]
                attn_output = attn_output * gate

        output, _ = self.o_proj(attn_output)

        return output

    def forward_absorb_fused_mla_rope_cpu_core(
        self, q_input, k_input, v_input, forward_batch, zero_allocator, hidden_states
    ):
        assert self.q_lora_rank is not None and use_intel_amx_backend(
            self
        ), "forward_absorb_fused_mla_rope_cpu_core requires q_lora_rank is not None and use_intel_amx_backend"

        attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        # [Note] Align shapes of bmm inputs.
        # Shapes of inputs:
        #   q_nope: [M, B, K]
        #   original self.w_kc: [B, K, N]
        #   current self.w_kc (which has been converted in PackWeightMethod): [B, N, K]

        # Shapes of inputs to sgl_kernel.cpu.bmm:
        #   out: [B, M, N]
        #   mat1: [B, M, K]
        #   mat2: [B, N, K]
        B = self.w_vc.size(0)
        N = self.w_vc.size(1)
        M = attn_output.size(0)
        output = torch.empty([M, int(B * N)], dtype=attn_output.dtype)
        attn_bmm_output = output.view([M, B, N]).transpose_(0, 1)
        torch.ops.sgl_kernel.bmm_cpu(
            attn_bmm_output,
            attn_output.transpose(0, 1),
            self.w_vc,
            True,  # is_vnni
            None,  # scale
        )
        attn_output = output

        if self.g_proj is not None:
            gate, _ = self.g_proj(hidden_states)
            gate = F.sigmoid(gate.float()).type_as(hidden_states)
            if self.gated_attention_proj_granularity_type == "head_wise":
                # gate shape: [seq_len, head_num]
                attn_output = (
                    attn_output.view(-1, self.num_local_heads, self.v_head_dim)
                    * gate[:, :, None]
                )
                attn_output = attn_output.view(
                    -1, self.num_local_heads * self.v_head_dim
                )
            else:
                # gate shape: [seq_len, head_num*head_dim]
                attn_output = attn_output * gate

        output, _ = self.o_proj(attn_output)

        return output

    def forward_normal_chunked_kv_core(self, q, k, v, forward_batch, hidden_states):
        has_extend_prefix = forward_batch.extend_prefix_lens_cpu is not None and any(
            forward_batch.extend_prefix_lens_cpu
        )
        # Only initialize the info once
        if has_extend_prefix and forward_batch.num_prefix_chunks is None:
            forward_batch.prepare_chunked_prefix_cache_info(q.device)
            if hasattr(forward_batch.attn_backend, "init_mha_chunk_metadata"):
                forward_batch.attn_backend.init_mha_chunk_metadata(forward_batch)

        forward_batch.mha_return_lse = has_extend_prefix
        # Do mha for extended part without prefix
        forward_batch.set_attn_attend_prefix_cache(False)
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)

        # Do mha attention with chunked prefix cache if there are any sequence with prefix
        if has_extend_prefix:
            attn_output, lse = attn_output
            forward_batch.set_attn_attend_prefix_cache(True)
            attn_output = self._chunked_prefix_attn_mha(
                q=q,
                accum_output=attn_output,
                accum_lse=lse,
                forward_batch=forward_batch,
            )

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)

        if self.g_proj is not None:
            gate, _ = self.g_proj(hidden_states)
            gate = F.sigmoid(gate.float()).type_as(hidden_states)
            if self.gated_attention_proj_granularity_type == "head_wise":
                # gate shape: [seq_len, head_num]
                attn_output = (
                    attn_output.view(-1, self.num_local_heads, self.v_head_dim)
                    * gate[:, :, None]
                )
                attn_output = attn_output.view(
                    -1, self.num_local_heads * self.v_head_dim
                )
            else:
                # gate shape: [seq_len, head_num*head_dim]
                attn_output = attn_output * gate

        output, _ = self.o_proj(attn_output)

        return output

    def forward_normal_one_shot_core(self, q, k, v, forward_batch, hidden_states):
        has_extend_prefix = any(forward_batch.extend_prefix_lens_cpu)
        # Only initialize the info once
        if has_extend_prefix and forward_batch.num_prefix_chunks is None:
            forward_batch.num_prefix_chunks = 0
            if hasattr(forward_batch.attn_backend, "init_mha_chunk_metadata"):
                forward_batch.attn_backend.init_mha_chunk_metadata(forward_batch)
        forward_batch.mha_return_lse = False
        # Do mha for extended part without prefix
        forward_batch.set_attn_attend_prefix_cache(False)
        return self.forward_normal_core(q, k, v, forward_batch, hidden_states)


LoraConfig = None
logger = logging.getLogger(__name__)
_is_cpu = is_cpu()


def is_linear_layer(layer_idx, layer_group_size):
    if layer_idx is None:
        return False
    if isinstance(layer_group_size, list):
        return layer_group_size[layer_idx] == 1
    if layer_group_size > 0:
        return (layer_idx + 1) % layer_group_size != 0
    else:
        return False


def is_pp_missing_parameter(
    name: str,
    model: torch.nn.Module,
) -> bool:
    if isinstance(model, PPMissingLayer):
        return True
    return False


def weight_loader_with_alias(alias: str):
    def wrapper(func: Callable):
        def inner_func(
            param: torch.Tensor,
            loaded_weight: torch.Tensor,
            *args,
            prefix: str = None,
            **kwargs,
        ):
            # pf = "[vLLM][load]" + " " if prefix is None else f"[{prefix}] "
            value = func(param, loaded_weight, *args, **kwargs)
            return value

        return inner_func

    return wrapper


class BailingMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        config: PretrainedConfig,
        reduce_results=True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.use_nGPT = getattr(config, "use_nGPT", False)
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )

        if self.use_nGPT:
            self.hidden_size = config.hidden_size
            self.swv = ScaleUpNorm(
                intermediate_size=intermediate_size,
                layernorm_epsilon=getattr(config, "layernorm_epsilon", 1e-6),
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
            )

        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        x, _ = self.gate_up_proj(x)

        if self.use_nGPT:
            x = self.swv(x)

        x = self.act_fn(x)
        x, _ = self.down_proj(
            x,
            skip_all_reduce=use_reduce_scatter or should_allreduce_fusion,
        )
        return x


class BailingMoEGate(nn.Module):
    def __init__(
        self,
        config,
        params_dtype: Optional[torch.dtype] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.use_nGPT = getattr(config, "use_nGPT", False)
        self.scale_router_input = getattr(config, "scale_router_input", False)

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.weight = nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                dtype=self.params_dtype,
            ),
        )

        if self.use_nGPT and self.scale_router_input:
            self.router_init_value = 1.0
            self.router_init_scaling = config.hidden_size**-0.5
            self.router_scaling = torch.nn.Parameter(torch.ones(config.hidden_size))

        if getattr(config, "moe_router_enable_expert_bias", False):
            self.expert_bias = nn.Parameter(
                torch.empty((config.num_experts,), dtype=torch.float32),
            )
        else:
            self.expert_bias = None

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.to(self.weight.dtype), self.weight, None).to(
            hidden_states.dtype
        )
        return logits


class BailingMoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id: int = 0,
        prefix: str = "moe",
    ):
        super().__init__()

        self.use_nGPT = getattr(config, "use_nGPT", False)

        self.layer_id = layer_id

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.top_k = config.num_experts_per_tok
        self.norm_expert_prob = getattr(config, "norm_topk_prob", False)
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.num_shared_experts = getattr(config, "num_shared_experts", 0)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.score_function = getattr(config, "score_function", None)

        # Gate always runs at half / full precision for now.
        router_dtype = getattr(config, "router_dtype", None)
        if router_dtype is None:
            self.router_dtype = torch.float32
        elif router_dtype == "fp32":
            self.router_dtype = torch.float32
        else:
            self.router_dtype = torch.bfloat16

        # check group topk
        self.num_expert_group = getattr(config, "n_group", 0)
        self.topk_group = getattr(config, "topk_group", 0)
        if self.num_expert_group > 0 or self.topk_group > 0:
            assert (
                self.num_expert_group > 0
                and 0 < self.topk_group <= self.num_expert_group
            )
            self.use_grouped_topk = True
        else:
            self.num_expert_group = self.topk_group = None
            self.use_grouped_topk = False

        self.num_experts = config.num_experts

        self.gate = BailingMoEGate(
            config=config,
            params_dtype=self.router_dtype,
            prefix=add_prefix("gate", prefix),
        )
        self.correction_bias = (
            self.gate.expert_bias.data if self.gate.expert_bias is not None else None
        )

        if self.score_function is not None:
            assert (
                self.score_function == "softmax" and self.correction_bias is None
            ) or (
                self.score_function == "sigmoid" and self.correction_bias is not None
            ), "score_function and correction_bias should be in 2 combination (softmax, None) or (sigmoid, not None)"

        self.topk = TopK(
            top_k=self.top_k,
            use_grouped_topk=self.use_grouped_topk,
            renormalize=self.norm_expert_prob,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            correction_bias=self.correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            layer_id=self.layer_id,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            prefix=f"{prefix}.experts",
            use_nGPT=self.use_nGPT,
            layernorm_epsilon=getattr(config, "layernorm_epsilon", 1e-6),
        )

        if self.num_shared_experts > 0:
            intermediate_size = getattr(
                config,
                "moe_shared_expert_intermediate_size",
                self.intermediate_size * self.num_shared_experts,
            )
            self.shared_experts = BailingMLP(
                hidden_size=self.hidden_size,
                intermediate_size=intermediate_size,
                config=config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        if self.num_shared_experts > 0:
            shared_output = self.shared_experts(hidden_states)

        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = self.experts(hidden_states, topk_output)

        if self.num_shared_experts > 0:
            final_hidden_states = final_hidden_states + shared_output

        if self.tp_size > 1 and not use_reduce_scatter and not should_allreduce_fusion:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states


class BailingKDA(KimiDeltaAttention):
    def __init__(
        self,
        layer_id: int,
        hidden_size: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        kimi_linear_config = KimiLinearConfig(
            linear_attn_config={
                "head_dim": getattr(config, "head_dim", None),
                "num_heads": config.num_attention_heads,
                "short_conv_kernel_size": config.short_conv_kernel_size,
                "kda_layers": [],
                "full_attn_layers": [],
                "use_nGPT": getattr(config, "use_nGPT", False),
                "value_norm": getattr(config, "value_norm", False),
            }
        )
        super().__init__(
            layer_id,
            hidden_size,
            kimi_linear_config,
            quant_config,
            prefix=prefix,
            rms_norm_eps=1e-6,
            kda_use_lora=config.use_kda_lora,
        )


class BailingMoEAttention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id: int = None,
        prefix: str = "mha",
    ) -> None:
        super().__init__()
        self.use_nGPT = getattr(config, "use_nGPT", False)
        self.value_norm = getattr(config, "value_norm", False)
        self.config = config
        self.layer_id = layer_id

        self.hidden_size = config.hidden_size
        tp_size = get_attention_tp_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(config, "head_dim", None)
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.total_num_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**0.5 if self.use_nGPT else self.head_dim**-0.5

        self.split_qkv = getattr(config, "using_split_qkv_in_self_attention", False)
        assert not self.split_qkv, "split_qkv is not supported for now"
        self.use_qk_norm = getattr(config, "use_qk_norm", False)

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=(config.use_bias or config.use_qkv_bias),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if self.use_nGPT and self.value_norm:
                self.value_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        if hasattr(config, "rotary_dim"):
            self.rotary_dim = config.rotary_dim
        elif hasattr(config, "partial_rotary_factor"):
            self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        else:
            self.rotary_dim = self.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 600000)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=config.rope_scaling,
            dtype=torch.float32,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_by_head = q.reshape(-1, self.head_dim)
        q_by_head = self.query_layernorm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.key_layernorm(k_by_head)
        k = k_by_head.view(k.shape)
        return q, k

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.use_qk_norm:
            q, k = self._apply_qk_norm(q, k)
            if self.use_nGPT and self.value_norm:
                v_by_head = v.reshape(-1, self.head_dim)
                v_by_head = self.value_layernorm(v_by_head)
                v = v_by_head.view(v.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.dense(attn_output)
        return output


class BailingMoELinearDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id: int = 0,
        prefix: str = "layer",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.use_mla = getattr(config, "full_attention_type", "mla") == "mla"
        alt_stream = None  # tptest
        self.attention_type = config.attention_type
        # todo nextn

        is_kda = True
        self.use_nGPT = getattr(config, "use_nGPT", False)
        if self.use_nGPT:
            self.layer_group_size = getattr(config, "layer_group_size", None)
            if self.layer_group_size is not None:
                self.attn_alpha_init_value = (
                    0.5
                    if (is_kda and (layer_id + 1) < self.layer_group_size)
                    else 0.70716 / config.num_hidden_layers
                )
            else:
                self.attn_alpha_init_value = 0.70716 / config.num_hidden_layers
            self.attn_alpha_init_scaling = config.hidden_size**-0.5
            self.attn_alpha = torch.nn.Parameter(torch.ones(config.hidden_size))
            self.mlp_alpha_init_value = 0.70716 / config.num_hidden_layers
            self.mlp_alpha_init_scaling = config.hidden_size**-0.5
            self.mlp_alpha = torch.nn.Parameter(torch.ones(config.hidden_size))
        self.config = config

        if config.attention_type == 0:  # Linear layer
            self.attention = BailingKDA(
                layer_id=self.layer_id,
                hidden_size=config.hidden_size,
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        elif config.attention_type == 1:  # softmax layer
            if self.use_mla:
                self.attention = DsV3MLA(
                    config=config,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    v_head_dim=config.v_head_dim,
                    q_lora_rank=(
                        config.q_lora_rank if hasattr(config, "q_lora_rank") else None
                    ),
                    kv_lora_rank=config.kv_lora_rank,
                    rope_theta=getattr(config, "rope_theta", 600000),
                    rope_scaling=config.rope_scaling,
                    max_position_embeddings=262144,
                    quant_config=quant_config,
                    layer_id=layer_id,
                    reduce_results=True,
                    prefix=add_prefix("attention", prefix),
                    alt_stream=alt_stream,
                    skip_rope=(
                        getattr(config, "use_mla_nope", False)
                        or config.qk_rope_head_dim == 0
                    ),
                )
            else:
                logger.info(f"==={layer_id=} use gqa")
                self.attention = BailingMoEAttention(
                    config,
                    quant_config=quant_config,
                    layer_id=self.layer_id,
                    prefix=prefix + ".attention",
                )
        else:
            raise ValueError(f"Unsupported attention type: {config.attention_type}")

        self.expert_num = config.num_experts
        self.hidden_size = config.hidden_size
        is_moe_layer = not (self.expert_num == 1) and (
            self.layer_id >= config.first_k_dense_replace
        )
        is_previous_moe_layer = not (self.expert_num == 1) and (
            self.layer_id - 1 >= config.first_k_dense_replace
        )
        is_next_layer_sparse = not (self.expert_num == 1) and (
            self.layer_id + 1 >= config.first_k_dense_replace
        )
        if self.expert_num == 1:
            self.mlp = BailingMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            )
        else:
            if self.layer_id >= config.first_k_dense_replace:
                # MoE layer
                self.mlp = BailingMoE(
                    config,
                    quant_config=quant_config,
                    layer_id=self.layer_id,
                    prefix=prefix,
                )
            else:
                # dense layer
                self.mlp = BailingMLP(
                    hidden_size=self.hidden_size,
                    intermediate_size=config.intermediate_size,
                    config=config,
                    quant_config=quant_config,
                    prefix=prefix,
                )
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-5))
        self.input_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=is_moe_layer,
            is_previous_layer_sparse=is_previous_moe_layer,
            is_next_layer_sparse=is_next_layer_sparse,
        )

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if not forward_batch.forward_mode.is_idle():
            if self.use_mla:
                if self.attention_type == 1:
                    attn_inputs = AttentionInputs(
                        hidden_states, forward_batch, self.attention.prepare_qkv_latent
                    )
                    get_attn_tp_context().set_attn_inputs(attn_inputs)
                hidden_states = self.attention(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    zero_allocator=zero_allocator,
                )
            else:
                hidden_states = self.attention(
                    hidden_states=hidden_states,
                    positions=positions,
                    forward_batch=forward_batch,
                )
        if self.use_nGPT:
            hidden_states, residual = apply_hyper_spherical_fusion(
                hidden_states,
                residual,
                self.attn_alpha,
                getattr(self.config, "layernorm_epsilon", 1e-6),
            )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if self.use_nGPT:
            hidden_states, residual = apply_hyper_spherical_fusion(
                hidden_states,
                residual,
                self.mlp_alpha,
                getattr(self.config, "layernorm_epsilon", 1e-6),
            )

        return hidden_states, residual

    @staticmethod
    def shared_moe_coefficient_loader(
        param: torch.Tensor, loaded_weight: torch.Tensor
    ) -> None:
        assert param.size() == loaded_weight.size()

        param.data.copy_(loaded_weight.to(torch.float32))
        return


class BailingMoELinearModel(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.use_nGPT = getattr(config, "use_nGPT", False)
        self.pp_group = get_pp_group()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size
        self.num_layers = config.num_hidden_layers

        self.layer_group_size = getattr(config, "layer_group_size", 1)
        self.decoder_attention_types = [
            0 if is_linear_layer(i, self.layer_group_size) else 1
            for i in range(self.num_layers)
        ]
        logger.info(
            f"attention type of layers:{self.decoder_attention_types}, 0 is linear layer and 1 is softmax layer!"
        )

        assert (
            isinstance(self.layer_group_size, list)
            or self.num_layers % self.layer_group_size == 0
        ), f"num_layers={self.num_layers} must be divided by layer_group_size={self.layer_group_size}"

        if self.pp_group.is_first_rank:
            self.word_embeddings = VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
                enable_tp=not is_dp_attention_enabled(),
                org_num_embeddings=self.vocab_size,
            )
        else:
            self.word_embeddings = PPMissingLayer()

        def layer_fn(idx, prefix):
            layer_idx = idx
            layer_config = copy.deepcopy(config)
            layer_config.attention_type = self.decoder_attention_types[layer_idx]

            decoder_kwargs = {"quant_config": quant_config, "layer_id": layer_idx}
            return BailingMoELinearDecoderLayer(
                layer_config, **decoder_kwargs, prefix=prefix
            )

        self.layers, self.start_layer, self.end_layer = make_layers(
            self.num_layers,
            layer_fn,
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
        )

        linear_layer_nums = sum(
            1 for i in range(self.num_layers) if self.decoder_attention_types[i] == 0
        )
        logger.info(f"linear_layer_nums={linear_layer_nums}")

        norm_kwargs = {}
        if hasattr(config, "rms_norm_eps"):
            norm_kwargs["eps"] = config.rms_norm_eps
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, **norm_kwargs)
        else:
            self.norm = PPMissingLayer()
        self.embed_scale = 1.0
        return

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if inputs_embeds is None:
                hidden_states = self.word_embeddings(input_ids)
            else:
                hidden_states = inputs_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        total_num_layers = self.end_layer - self.start_layer
        device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )

        for i in range(self.start_layer, self.end_layer):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.layers[i]
                hidden_states, residual = layer(
                    hidden_states=hidden_states,
                    positions=positions,
                    forward_batch=forward_batch,
                    residual=residual,
                    zero_allocator=zero_allocator,
                )
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        else:
            if not forward_batch.forward_mode.is_idle():
                if self.use_nGPT:
                    hidden_states = hidden_states + residual
                    hidden_states = l2norm(
                        hidden_states, getattr(self.config, "layernorm_epsilon", 1e-6)
                    )
                else:
                    if residual is None:
                        hidden_states = self.norm(hidden_states)
                    else:
                        hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states


class BailingMoeV3ForCausalLM(nn.Module):

    def __init__(
        self,
        *,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.use_nGPT = getattr(config, "use_nGPT", False)
        if self.use_nGPT:
            self.finalnorm_init_value = 1.0
            self.finalnorm_init_scaling = config.hidden_size**-0.5
            tp_size = get_tensor_model_parallel_world_size()
            self.finalnorm = torch.nn.Parameter(
                torch.ones(divide(config.vocab_size, tp_size), dtype=torch.float32)
            )
            set_weight_attrs(
                self.finalnorm, {"weight_loader": sharded_weight_loader(0)}
            )

        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = BailingMoELinearModel(
            self.config, quant_config, prefix=add_prefix("model", prefix)
        )

        if self.pp_group.is_last_rank:
            self.lm_head = (
                self.model.word_embeddings
                if config.tie_word_embeddings
                else ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    params_dtype=torch.float32,
                    quant_config=quant_config,
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                )
            )
            self.logits_processor = LogitsProcessor(config)
        else:
            self.lm_head = PPMissingLayer()

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def post_load_weights(self, is_nextn=False, weight_names=None):
        # nGPT fusion
        if self.use_nGPT:
            # fuse finalnorm into lm_head weight
            if self.pp_group.is_last_rank and self.use_nGPT:
                self.finalnorm.data.mul_(
                    self.finalnorm_init_value / self.finalnorm_init_scaling
                )
                self.lm_head.weight.data.mul_(
                    self.finalnorm.unsqueeze(1).to(self.lm_head.weight.dtype)
                )
                logger.info("Fused finalnorm into lm_head weight.")

            for i in range(self.start_layer, self.end_layer):
                layer = self.model.layers[i]

                # fuse router_scaling into expert weights
                mlp = layer.mlp
                if isinstance(mlp, BailingMoE):
                    gate = mlp.gate
                    if gate.scale_router_input:
                        router_scaling = (
                            gate.router_scaling
                            * (gate.router_init_value / gate.router_init_scaling)
                        ).to(gate.weight.dtype)
                        gate.weight.data.mul_(router_scaling.unsqueeze(0))

                # calculate attn_alpha and mlp_alpha
                attn_alpha = layer.attn_alpha
                attn_alpha.data.mul_(
                    layer.attn_alpha_init_value / layer.attn_alpha_init_scaling
                )
                mlp_alpha = layer.mlp_alpha
                mlp_alpha.data.mul_(
                    layer.mlp_alpha_init_value / layer.mlp_alpha_init_scaling
                )

        # Perform post-processing after loading weights
        if is_nextn:
            layer_ids = [self.config.num_hidden_layers]
        else:
            if weight_names is None:
                layer_ids = range(self.model.start_layer, self.model.end_layer)
            else:
                layer_ids = set()
                for name in weight_names:
                    if "kv_b_proj" in name:
                        layer_id = int(name.split(".")[2])
                        if (
                            layer_id < self.model.end_layer
                            and layer_id >= self.model.start_layer
                        ):
                            layer_ids.add(layer_id)
        logger.info(f"=====layer_ids {layer_ids}")

        for layer_id in layer_ids:
            self_attn = (
                self.model.layers[layer_id].attention
                if not is_nextn
                else self.model.decoder.attention
            )
            if not hasattr(self_attn, "kv_b_proj"):
                continue
            if hasattr(self_attn.kv_b_proj, "qweight"):
                # AWQ compatible
                if _is_cuda or _is_hip:
                    w = awq_dequantize(
                        self_attn.kv_b_proj.qweight,
                        self_attn.kv_b_proj.scales,
                        self_attn.kv_b_proj.qzeros,
                    ).T
                else:
                    w = awq_dequantize(
                        self_attn.kv_b_proj.qweight,
                        self_attn.kv_b_proj.scales,
                        self_attn.kv_b_proj.qzeros,
                        0,
                        0,
                        0,
                    ).T
            else:
                w = self_attn.kv_b_proj.weight
            # NOTE(HandH1998): Since `bmm_fp8` only supports per-tensor scale, we have to requantize `self_attn.kv_b_proj`.
            # This may affect the accuracy of fp8 model.
            # Fix deepseek v3 blockwise bmm by using deep_gemm
            use_deep_gemm_bmm = False

            if w.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
            ):
                if (
                    hasattr(self.quant_config, "weight_block_size")
                    and self.quant_config.weight_block_size is not None
                ):
                    weight_block_size = self.quant_config.weight_block_size
                    assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                    if _is_fp8_fnuz:
                        weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                            weight=w,
                            weight_scale=self_attn.kv_b_proj.weight_scale_inv,
                            input_scale=None,
                        )
                    else:
                        weight = w
                        weight_scale = self_attn.kv_b_proj.weight_scale_inv

                    if (
                        _is_cuda
                        and weight_block_size[0] == 128
                        and weight_block_size[1] == 128
                    ):
                        if (
                            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                            and not deep_gemm_wrapper.DEEPGEMM_BLACKWELL
                            and get_bool_env_var("SGL_USE_DEEPGEMM_BMM", "false")
                        ):
                            block_scale = weight_scale
                            use_deep_gemm_bmm = True
                        else:
                            w = block_quant_dequant(
                                weight,
                                weight_scale,
                                weight_block_size,
                                torch.bfloat16,
                            )
                    else:
                        w, scale = block_quant_to_tensor_quant(
                            weight, weight_scale, weight_block_size
                        )
                        self_attn.w_scale = scale
                else:
                    if _is_fp8_fnuz:
                        weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                            weight=w,
                            weight_scale=self_attn.kv_b_proj.weight_scale,
                            input_scale=None,
                        )
                    else:
                        weight = w
                        weight_scale = self_attn.kv_b_proj.weight_scale

                    w, scale = channel_quant_to_tensor_quant(weight, weight_scale)
                    self_attn.w_scale = scale

            if w.dtype == torch.int8:
                if hasattr(self.quant_config, "weight_block_size"):
                    # block-wise int8 need it
                    weight_block_size = self.quant_config.weight_block_size
                    if weight_block_size is not None:
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                        weight = w
                        weight_scale = self_attn.kv_b_proj.weight_scale_inv
                        w = int8_block_dequant(
                            weight, weight_scale, weight_block_size
                        ).to(torch.bfloat16)
                else:
                    # channel-wise int8 need it
                    w = w.to(torch.bfloat16) * self_attn.kv_b_proj.weight_scale.to(
                        torch.bfloat16
                    )

            w_kc, w_vc = w.unflatten(
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            if not use_deep_gemm_bmm:
                self_attn.w_kc = bind_or_assign(
                    self_attn.w_kc, w_kc.transpose(1, 2).contiguous().transpose(1, 2)
                )
                self_attn.w_vc = bind_or_assign(
                    self_attn.w_vc, w_vc.contiguous().transpose(1, 2)
                )
                if (
                    hasattr(self_attn.kv_b_proj, "weight_scale")
                    and self_attn.w_scale is None
                ):
                    self_attn.w_scale = bind_or_assign(
                        self_attn.w_scale, self_attn.kv_b_proj.weight_scale
                    )
                    if _is_hip:
                        self_attn.w_scale *= 2.0
                # TODO: remove this after adding FP8 support in bmm cpu kernel
                if _is_cpu and _is_cpu_amx_available and w.dtype == torch.float8_e4m3fn:
                    self_attn.w_kc = (
                        self_attn.w_kc.to(torch.bfloat16) * self_attn.w_scale
                    )
                    self_attn.w_vc = (
                        self_attn.w_vc.to(torch.bfloat16) * self_attn.w_scale
                    )
            else:
                num_tiles_k = self_attn.qk_nope_head_dim // weight_block_size[1]
                num_tiles_n = self_attn.v_head_dim // weight_block_size[0]
                ws_kc, ws_vc = block_scale.unflatten(
                    0, (-1, (num_tiles_k + num_tiles_n))
                ).split([num_tiles_k, num_tiles_n], dim=1)
                self_attn.w_scale_k = bind_or_assign(
                    self_attn.w_scale_k, ws_kc.transpose(1, 2).contiguous()
                )
                self_attn.w_scale_v = bind_or_assign(
                    self_attn.w_scale_v, ws_vc.contiguous()
                )
                self_attn.w_kc = bind_or_assign(
                    self_attn.w_kc, w_kc.transpose(1, 2).contiguous()
                )
                self_attn.w_vc = bind_or_assign(self_attn.w_vc, w_vc.contiguous())
                self_attn.use_deep_gemm_bmm = True

        if (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
            and hasattr(self.quant_config, "weight_block_size")
            and self.quant_config.weight_block_size is not None
        ):
            self._weight_requant_ue8m0(is_nextn)

    def _weight_requant_ue8m0(self, is_nextn=False):
        weight_block_size = self.quant_config.weight_block_size

        moe_layers = list(
            range(
                self.config.first_k_dense_replace,
                self.config.num_hidden_layers,
                self.config.moe_layer_freq,
            )
        )

        num_hidden_layers = 1 if is_nextn else self.config.num_hidden_layers

        for layer_id in range(num_hidden_layers):
            if is_nextn:
                layer = self.model.decoder
            else:
                layer = self.model.layers[layer_id]

            module_list = [
                layer.self_attn.kv_b_proj,
                layer.self_attn.o_proj,
            ]

            if self.config.q_lora_rank is not None:
                module_list.append(layer.self_attn.fused_qkv_a_proj_with_mqa)
                module_list.append(layer.self_attn.q_b_proj)
            else:
                module_list.append(layer.self_attn.kv_a_proj_with_mqa)
                module_list.append(layer.self_attn.q_proj)

            for module in module_list:
                requant_weight_ue8m0_inplace(
                    module.weight, module.weight_scale_inv, weight_block_size
                )

            if layer_id in moe_layers or is_nextn:
                shared_experts = getattr(layer.mlp, "shared_experts", None)
                if shared_experts is not None:
                    for module in [
                        shared_experts.gate_up_proj,
                        shared_experts.down_proj,
                    ]:
                        requant_weight_ue8m0_inplace(
                            module.weight, module.weight_scale_inv, weight_block_size
                        )

                experts = layer.mlp.experts
                if isinstance(experts, DeepEPMoE):
                    for w in [
                        experts.w13_weight_fp8,
                        experts.w2_weight_fp8,
                    ]:
                        requant_weight_ue8m0_inplace(w[0], w[1], weight_block_size)
            else:
                mlp = layer.mlp
                assert isinstance(mlp, DeepseekV2MLP)
                for module in [
                    mlp.gate_up_proj,
                    mlp.down_proj,
                ]:
                    requant_weight_ue8m0_inplace(
                        module.weight, module.weight_scale_inv, weight_block_size
                    )

    def get_decoder_attention_types(self):
        return self.model.decoder_attention_types

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            forward_batch=forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states.float(), self.lm_head, forward_batch
            )
        else:
            return hidden_states

    @staticmethod
    def weight_direct_load(param: torch.Tensor, loaded_weight: torch.Tensor):
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        num_groups = getattr(config, "n_group", 0)
        from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation

        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None if num_groups == 0 else num_groups,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        def load_linear_attn_weight(
            name: str, loaded_weight: torch.Tensor, self
        ) -> None:
            if is_pp_missing_parameter(name, self):
                return
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", self.weight_direct_load)
            weight_loader = weight_loader_with_alias(name)(weight_loader)
            if "A_log" in name:
                # Temporary use this way
                # As our A_log param's shape is different from kimi's
                loaded_weight = loaded_weight[None, None, :, None]
            weight_loader(param, loaded_weight)
            return

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        weight_names = []
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (
            self.config.q_lora_rank is not None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None

        for name, loaded_weight in weights:
            try:
                found = False
                name0 = name
                if name.startswith("model.mtp"):
                    continue
                layer_idx = None
                if "model.layers." in name:
                    layer_idx = int(name.split(".")[2])
                if (
                    ("v_head" in name)
                    or ("inv_freq" in name)
                    or (self.config.tie_word_embeddings and "lm_head" in name)
                ):
                    continue

                weight_names.append(name)

                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    if "mlp.experts" in name:
                        continue

                    name = name.replace(weight_name, param_name)
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    found = True
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:

                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)

                        if name not in params_dict:
                            continue
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        found = True
                        weight_loader(
                            param,
                            loaded_weight,
                            name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                        break
                    else:

                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        if "slope" in name:
                            continue

                        if fuse_qkv_a_proj and (
                            "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                        ):
                            found = True
                            cached_a_proj[name] = loaded_weight
                            q_a_proj_name = (
                                name
                                if "q_a_proj" in name
                                else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                            )
                            kv_a_proj_name = (
                                name
                                if "kv_a_proj_with_mqa" in name
                                else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                            )

                            # When both q_a_proj and kv_a_proj_with_mqa has been cached, load the fused weight to parameter
                            if (
                                q_a_proj_name in cached_a_proj
                                and kv_a_proj_name in cached_a_proj
                            ):
                                q_a_proj_weight = cached_a_proj[q_a_proj_name]
                                kv_a_proj_weight = cached_a_proj[kv_a_proj_name]
                                cat_dim = 0
                                if self.quant_config is not None and (
                                    self.quant_config.get_name() == "awq"
                                    or self.quant_config.get_name() == "awq_marlin"
                                    or self.quant_config.get_name() == "moe_wna16"
                                ):
                                    cat_dim = 1
                                fused_weight = torch.cat(
                                    [q_a_proj_weight, kv_a_proj_weight], dim=cat_dim
                                )
                                param_name = (
                                    name.replace(
                                        "q_a_proj", "fused_qkv_a_proj_with_mqa"
                                    )
                                    if "q_a_proj" in name
                                    else name.replace(
                                        "kv_a_proj_with_mqa",
                                        "fused_qkv_a_proj_with_mqa",
                                    )
                                )
                                if param_name not in params_dict:
                                    continue
                                param = params_dict[param_name]
                                weight_loader = getattr(
                                    param, "weight_loader", default_weight_loader
                                )

                                weight_loader(param, fused_weight)
                                cached_a_proj.pop(q_a_proj_name)
                                cached_a_proj.pop(kv_a_proj_name)
                        else:

                            if name not in params_dict:
                                name = name.replace(".dense.", ".o_proj.")
                                if name not in params_dict:
                                    continue
                            if is_pp_missing_parameter(name, self):
                                continue
                            if (
                                "attention" in name
                                and "slope" not in name
                                and is_linear_layer(
                                    layer_idx, self.model.layer_group_size
                                )
                            ):
                                load_linear_attn_weight(name, loaded_weight, self)
                                loaded_params.add(name)
                                found = True
                                continue

                            param = params_dict[name]
                            found = True
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, loaded_weight)
            finally:
                if not found:
                    # print(f"fail load: {name0}")
                    pass
            loaded_params.add(name)
        self.post_load_weights(is_nextn=False, weight_names=weight_names)

        return loaded_params


EntryClass = [
    BailingMoeV3ForCausalLM,
]
