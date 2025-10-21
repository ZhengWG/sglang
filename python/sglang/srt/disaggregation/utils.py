from __future__ import annotations

import logging
import os
import random
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Type

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

#########################
# Constants & Enums
#########################
FAKE_BOOTSTRAP_HOST = "2.2.2.2"


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"
    ENCODE = "encode"
    LANGUAGE = "language"


#########################
# Synchronization
#########################

# env var for testing failure, convert to float explicitly
FAILURE_PROB = float(os.getenv("DISAGGREGATION_TEST_FAILURE_PROB", 0))


def poll_and_all_reduce(pollers, gloo_group):
    # at a certain prob, the poll is failed to simulate failure
    if FAILURE_PROB > 0:
        from sglang.srt.disaggregation.base import KVPoll

        polls = [
            int(KVPoll.Failed) if random.random() < FAILURE_PROB else int(poller.poll())
            for poller in pollers
        ]
    else:
        polls = [int(poller.poll()) for poller in pollers]
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
    return tensor_to_reduce.tolist()


#########################
# Metadata Buffers
#########################


@dataclass
class MetadataAllocation:
    """Represents a metadata buffer allocation using blocks."""
    block_indices: List[int]  # Allocated block indices (may not be contiguous)
    num_tokens: int  # Actual number of tokens needed
    
    def get_contiguous_ranges(self, block_size: int) -> List[Tuple[int, int]]:
        """
        Merge contiguous blocks into ranges.
        
        Returns:
            List of (start_token, num_tokens) for each contiguous range.
        
        Example:
            block_indices=[8,9,3,4,5], block_size=128
            -> sorted: [3,4,5,8,9]
            -> ranges: [(3*128, 3*128), (8*128, 2*128)]
                      = [(384, 384), (1024, 256)]
        """
        if not self.block_indices:
            return []
        
        sorted_blocks = sorted(self.block_indices)
        ranges = []
        
        range_start = sorted_blocks[0]
        range_len = 1
        
        for i in range(1, len(sorted_blocks)):
            if sorted_blocks[i] == sorted_blocks[i-1] + 1:
                # Contiguous, extend current range
                range_len += 1
            else:
                # Gap found, save current range and start new one
                ranges.append((range_start * block_size, range_len * block_size))
                range_start = sorted_blocks[i]
                range_len = 1
        
        # Don't forget the last range
        ranges.append((range_start * block_size, range_len * block_size))
        
        return ranges


class ReqToMetadataIdxAllocator:
    """A memory pool that maps a request to its first output token location."""

    def __init__(
        self,
        size: int,
    ):
        self.size = size
        self.free_slots = deque(list(range(size)))

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, fake: bool = False) -> Optional[int]:
        if fake:
            return random.randint(0, self.size - 1)

        if len(self.free_slots) == 0:
            return None

        return self.free_slots.popleft()

    def free(self, free_index: int, fake: bool = False):
        if fake:
            return

        self.free_slots.append(free_index)

    def free_with_req(self, req: Req):
        """
        This function is used to free slot and reset the metadata buffer index of the request.
        NOTE: Only used in the disaggregation language mode: \
              since transfer buffer need to be freed after the prefill is done. \
        TODO: Need to refactor the code to keep interface consistent.
        """
        free_index = req.metadata_buffer_index
        fake = req.bootstrap_host == FAKE_BOOTSTRAP_HOST
        self.free(free_index, fake=fake)
        req.metadata_buffer_index = -1


class ReqToMetadataBlockAllocator:
    """Block-based metadata buffer allocator (simplified)."""

    def __init__(self, total_tokens: int, block_size: int = 128):
        self.block_size = block_size
        self.num_blocks = (total_tokens + block_size - 1) // block_size
        self.free_blocks = deque(list(range(self.num_blocks)))
        self.allocations: Dict[int, MetadataAllocation] = {}
        # Default blocks for Language side initial allocation
        self.default_num_blocks = int(os.getenv("SGLANG_DEFAULT_MULTIMODAL_BLOCKS", "8"))

    def available_blocks(self) -> int:
        return len(self.free_blocks)

    def alloc_blocks(self, num_blocks: int, num_tokens: int, req_id: int = None, fake: bool = False) -> Optional[MetadataAllocation]:
        """
        Allocate specified number of blocks.
        
        Args:
            num_blocks: Number of blocks to allocate
            num_tokens: Actual tokens needed (for validation)
            req_id: Request ID for tracking
            fake: Fake allocation for testing
        
        Note:
            Blocks may not be contiguous due to fragmentation.
            Use get_contiguous_ranges() to get merged contiguous chunks.
        """
        if fake:
            return MetadataAllocation([0], num_tokens)
        
        if len(self.free_blocks) < num_blocks:
            return None

        # Allocate blocks (may not be contiguous due to free order)
        blocks = [self.free_blocks.popleft() for _ in range(num_blocks)]
        
        allocation = MetadataAllocation(blocks, num_tokens)
        
        if req_id is not None:
            self.allocations[req_id] = allocation
        return allocation

    def alloc(self, num_tokens: int, req_id: int = None, fake: bool = False) -> Optional[MetadataAllocation]:
        """Allocate blocks based on num_tokens (for Embedding side)."""
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        return self.alloc_blocks(num_blocks, num_tokens, req_id, fake)

    def alloc_default(self, req_id: int = None, fake: bool = False) -> Optional[MetadataAllocation]:
        """Allocate default blocks (for Language side initial allocation)."""
        num_tokens = self.default_num_blocks * self.block_size
        return self.alloc_blocks(self.default_num_blocks, num_tokens, req_id, fake)

    def free(self, allocation: MetadataAllocation, req_id: int = None, fake: bool = False):
        """Free allocated blocks."""
        if fake:
            return
        # Return blocks to free pool (may become non-contiguous)
        for block_idx in allocation.block_indices:
            self.free_blocks.append(block_idx)
        if req_id and req_id in self.allocations:
            del self.allocations[req_id]


class MetadataBuffers:
    def __init__(
        self,
        size: int,
        hidden_size: int,
        hidden_states_dtype: torch.dtype,
        max_top_logprobs_num: int = 128,
        custom_mem_pool: torch.cuda.MemPool = None,
    ):
        self.custom_mem_pool = custom_mem_pool
        device = "cpu"
        if is_npu():
            # For ascend backend, output tokens are placed in the NPU and will be transferred by D2D channel.
            device = "npu"
        elif self.custom_mem_pool:
            # TODO(shangming): Fix me (use 'cuda') when nvlink_transport of Mooncake is bug-free
            device = "cpu"
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            # TODO: abort top_logprobs_num > 128 in PD

            # We transfer the metadata of first output token to decode
            # The minimal size for RDMA is 64Bytes, so we pad it to > 64Bytes
            self.output_ids = torch.zeros((size, 16), dtype=torch.int32, device=device)
            self.cached_tokens = torch.zeros(
                (size, 16), dtype=torch.int32, device=device
            )
            self.output_token_logprobs_val = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
            self.output_token_logprobs_idx = torch.zeros(
                (size, 16), dtype=torch.int32, device=device
            )
            self.output_top_logprobs_val = torch.zeros(
                (size, max_top_logprobs_num), dtype=torch.float32, device=device
            )
            self.output_top_logprobs_idx = torch.zeros(
                (size, max_top_logprobs_num), dtype=torch.int32, device=device
            )
            # For PD + spec decode
            self.output_topk_p = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
            self.output_topk_index = torch.zeros(
                (size, 16), dtype=torch.int64, device=device
            )
            self.output_hidden_states = torch.zeros(
                (size, hidden_size), dtype=hidden_states_dtype, device=device
            )

    def get_buf_infos(self):
        ptrs = [
            self.output_ids.data_ptr(),
            self.cached_tokens.data_ptr(),
            self.output_token_logprobs_val.data_ptr(),
            self.output_token_logprobs_idx.data_ptr(),
            self.output_top_logprobs_val.data_ptr(),
            self.output_top_logprobs_idx.data_ptr(),
            self.output_topk_p.data_ptr(),
            self.output_topk_index.data_ptr(),
            self.output_hidden_states.data_ptr(),
        ]
        data_lens = [
            self.output_ids.nbytes,
            self.cached_tokens.nbytes,
            self.output_token_logprobs_val.nbytes,
            self.output_token_logprobs_idx.nbytes,
            self.output_top_logprobs_val.nbytes,
            self.output_top_logprobs_idx.nbytes,
            self.output_topk_p.nbytes,
            self.output_topk_index.nbytes,
            self.output_hidden_states.nbytes,
        ]
        item_lens = [
            self.output_ids[0].nbytes,
            self.cached_tokens[0].nbytes,
            self.output_token_logprobs_val[0].nbytes,
            self.output_token_logprobs_idx[0].nbytes,
            self.output_top_logprobs_val[0].nbytes,
            self.output_top_logprobs_idx[0].nbytes,
            self.output_topk_p[0].nbytes,
            self.output_topk_index[0].nbytes,
            self.output_hidden_states[0].nbytes,
        ]
        return ptrs, data_lens, item_lens

    def get_buf(self, idx: int):
        return (
            self.output_ids[idx],
            self.cached_tokens[idx],
            self.output_token_logprobs_val[idx],
            self.output_token_logprobs_idx[idx],
            self.output_top_logprobs_val[idx],
            self.output_top_logprobs_idx[idx],
            self.output_topk_p[idx],
            self.output_topk_index[idx],
            self.output_hidden_states[idx],
        )

    def set_buf(self, req: Req):

        self.output_ids[req.metadata_buffer_index][0] = req.output_ids[0]
        self.cached_tokens[req.metadata_buffer_index][0] = req.cached_tokens
        if req.return_logprob:
            if req.output_token_logprobs_val:  # not none or empty list
                self.output_token_logprobs_val[req.metadata_buffer_index][0] = (
                    req.output_token_logprobs_val[0]
                )
            if req.output_token_logprobs_idx:  # not none or empty list
                self.output_token_logprobs_idx[req.metadata_buffer_index][0] = (
                    req.output_token_logprobs_idx[0]
                )

            if req.output_top_logprobs_val:  # not none or empty list
                self.output_top_logprobs_val[req.metadata_buffer_index][
                    : len(req.output_top_logprobs_val[0])
                ] = torch.tensor(
                    req.output_top_logprobs_val[0], dtype=torch.float32, device="cpu"
                )
            if req.output_top_logprobs_idx:  # not none or empty list
                self.output_top_logprobs_idx[req.metadata_buffer_index][
                    : len(req.output_top_logprobs_idx[0])
                ] = torch.tensor(
                    req.output_top_logprobs_idx[0], dtype=torch.int32, device="cpu"
                )
        # For PD + spec decode
        if req.hidden_states_tensor is not None:
            # speculative_eagle_topk should not be greater than 16 currently
            topk = req.output_topk_p.size(0)

            self.output_topk_p[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_p
            )
            self.output_topk_index[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_index
            )
            self.output_hidden_states[req.metadata_buffer_index].copy_(
                req.hidden_states_tensor
            )


#########################
# Transfer Backend
#########################


class TransferBackend(Enum):
    MOONCAKE = "mooncake"
    NIXL = "nixl"
    ASCEND = "ascend"
    FAKE = "fake"


class KVClassType(Enum):
    KVARGS = "kvargs"
    MANAGER = "manager"
    SENDER = "sender"
    RECEIVER = "receiver"
    BOOTSTRAP_SERVER = "bootstrap_server"


def get_kv_class(
    transfer_backend: TransferBackend,
    class_type: KVClassType,
    is_multimodal: bool = False,
) -> Optional[Type]:
    from sglang.srt.disaggregation.fake import FakeKVReceiver, FakeKVSender

    if transfer_backend == TransferBackend.MOONCAKE:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.mooncake import (
            MooncakeKVBootstrapServer,
            MooncakeKVManager,
            MooncakeKVReceiver,
            MooncakeKVSender,
        )
        from sglang.srt.disaggregation.mooncake.conn_multimodal import (
            MooncakeEmbeddingBootstrapServer,
            MooncakeEmbeddingManager,
            MooncakeEmbeddingReceiver,
            MooncakeEmbeddingSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: (
                MooncakeKVManager if not is_multimodal else MooncakeEmbeddingManager
            ),
            KVClassType.SENDER: (
                MooncakeKVSender if not is_multimodal else MooncakeEmbeddingSender
            ),
            KVClassType.RECEIVER: (
                (MooncakeKVReceiver) if not is_multimodal else MooncakeEmbeddingReceiver
            ),
            KVClassType.BOOTSTRAP_SERVER: (
                MooncakeKVBootstrapServer
                if not is_multimodal
                else MooncakeEmbeddingBootstrapServer
            ),
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.ASCEND:
        from sglang.srt.disaggregation.ascend import (
            AscendKVBootstrapServer,
            AscendKVManager,
            AscendKVReceiver,
            AscendKVSender,
        )
        from sglang.srt.disaggregation.base import KVArgs

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: AscendKVManager,
            KVClassType.SENDER: AscendKVSender,
            KVClassType.RECEIVER: (AscendKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: AscendKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.NIXL:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.nixl import (
            NixlKVBootstrapServer,
            NixlKVManager,
            NixlKVReceiver,
            NixlKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: NixlKVManager,
            KVClassType.SENDER: NixlKVSender,
            KVClassType.RECEIVER: (NixlKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: NixlKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.FAKE:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.fake import FakeKVReceiver, FakeKVSender

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.SENDER: FakeKVSender,
            KVClassType.RECEIVER: (FakeKVReceiver),
        }
        return class_mapping.get(class_type)

    raise ValueError(f"Unsupported transfer backend: {transfer_backend}")


#########################
# KV Pages
#########################


def kv_to_page_indices(kv_indices: np.ndarray, page_size: int):
    # 1. The page is guaranteed to be full except the last page.
    # 2. page index = kv_index // page_size
    # The return vector is kv_indices[::page_size] // page_size
    if page_size == 1:  # shortcut
        return kv_indices

    return kv_indices[::page_size] // page_size


def kv_to_page_num(num_kv_indices: int, page_size: int):
    # ceil(num_kv_indices / page_size)
    return (num_kv_indices + page_size - 1) // page_size


#########################
# Misc
#########################


def is_mla_backend(target_kv_pool) -> bool:
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    return isinstance(target_kv_pool, MLATokenToKVPool)


def prepare_abort(req: Req, error_message: str, status_code=None):
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    # populate finish metadata and stream output
    req.finished_reason = FINISH_ABORT(error_message, status_code)

    if req.return_logprob:
        req.input_token_logprobs_val = []
        req.input_token_logprobs_idx = []
        req.input_top_logprobs_val = []
        req.input_top_logprobs_idx = []
        req.input_token_ids_logprobs_val = []
        req.input_token_ids_logprobs_idx = []


class MultimodalDataBuffers:
    def __init__(self, size: int, max_prefill_tokens: int, embedding_dim: int = 8192, block_size: int = 128):
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.total_capacity_tokens = size * max_prefill_tokens
        self.num_blocks = (self.total_capacity_tokens + block_size - 1) // block_size
        self.default_buffer_tokens = int(os.getenv("SGLANG_DEFAULT_MULTIMODAL_BUFFER_TOKENS", "1024"))

        # Contiguous buffers
        self.input_embeddings = torch.zeros((self.total_capacity_tokens * embedding_dim,), dtype=torch.bfloat16, device="cpu")
        self.fill_ids = torch.zeros((self.total_capacity_tokens,), dtype=torch.int32, device="cpu")
        self.mrope_positions = torch.zeros((3 * self.total_capacity_tokens,), dtype=torch.int32, device="cpu")
        self.aux_datas = torch.zeros((self.num_blocks, 16), dtype=torch.int32, device="cpu")

    def get_buf_chunk_info(self, allocation: MetadataAllocation, offset_tokens: int = 0, max_tokens: int = None):
        """
        Calculate chunk info for transfer.
        
        Returns list of chunks for each buffer type, where each buffer type has
        multiple chunks corresponding to contiguous block ranges.
        
        Note:
            Data is stored in blocks according to block_indices.
            Contiguous blocks are merged into single chunks for efficiency.
        """
        actual_tokens = allocation.num_tokens - offset_tokens
        if max_tokens:
            actual_tokens = min(actual_tokens, max_tokens)
        
        # Get contiguous ranges in (start_token, length_tokens) format
        ranges = allocation.get_contiguous_ranges(self.block_size)
        
        # Apply offset and limit
        chunks_embeddings = []
        chunks_fill_ids = []
        chunks_mrope = []
        
        tokens_consumed = 0
        tokens_skipped = 0
        
        for start_token, range_tokens in ranges:
            # Skip ranges before offset
            if tokens_skipped + range_tokens <= offset_tokens:
                tokens_skipped += range_tokens
                continue
            
            # Adjust for offset within this range
            range_offset = max(0, offset_tokens - tokens_skipped)
            range_start = start_token + range_offset
            range_len = range_tokens - range_offset
            
            # Limit by max_tokens
            remaining = actual_tokens - tokens_consumed
            range_len = min(range_len, remaining)
            
            if range_len > 0:
                chunks_embeddings.append((range_start * self.embedding_dim * 2, range_len * self.embedding_dim * 2))
                chunks_fill_ids.append((range_start * 4, range_len * 4))
                chunks_mrope.append((range_start * 3 * 4, range_len * 3 * 4))
                
                tokens_consumed += range_len
                if tokens_consumed >= actual_tokens:
                    break
            
            tokens_skipped += range_tokens
        
        # For simplicity, return single chunk info (merge all ranges)
        # TODO: Support multiple chunks in RDMA transfer
        if chunks_embeddings:
            # For now, return the first contiguous range
            # In future, RDMA layer should support scatter-gather
            return [
                chunks_embeddings[0],
                chunks_fill_ids[0],
                chunks_mrope[0],
                (allocation.block_indices[0] * 64, 64),  # aux_datas
            ]
        else:
            return [
                (0, 0),
                (0, 0),
                (0, 0),
                (allocation.block_indices[0] * 64, 64),
            ]

    def get_buf_infos(self):
        """Return buffer pointers, lengths, and item sizes for RDMA registration."""
        return (
            [self.input_embeddings.data_ptr(), self.fill_ids.data_ptr(), 
             self.mrope_positions.data_ptr(), self.aux_datas.data_ptr()],
            [self.input_embeddings.nbytes, self.fill_ids.nbytes, 
             self.mrope_positions.nbytes, self.aux_datas.nbytes],
            [self.block_size * self.embedding_dim * 2, self.block_size * 4, 
             3 * self.block_size * 4, 64],  # item_lens per block
        )

    def get_buf(self, allocation: MetadataAllocation):
        """
        Get buffer data for allocation.
        
        Data is gathered from potentially non-contiguous blocks.
        """
        ranges = allocation.get_contiguous_ranges(self.block_size)
        
        embeddings_list = []
        fill_ids_list = []
        mrope_list = []
        
        tokens_collected = 0
        
        for start_token, range_tokens in ranges:
            # Limit to actual num_tokens
            range_tokens = min(range_tokens, allocation.num_tokens - tokens_collected)
            if range_tokens <= 0:
                break
            
            end_token = start_token + range_tokens
            
            embeddings_list.append(
                self.input_embeddings[start_token * self.embedding_dim : end_token * self.embedding_dim]
            )
            fill_ids_list.append(
                self.fill_ids[start_token : end_token]
            )
            mrope_list.append(
                self.mrope_positions[start_token * 3 : end_token * 3]
            )
            
            tokens_collected += range_tokens
        
        # Concatenate all ranges
        embeddings = torch.cat(embeddings_list).reshape(allocation.num_tokens, self.embedding_dim)
        fill_ids = torch.cat(fill_ids_list)
        mrope = torch.cat(mrope_list)
        aux = self.aux_datas[allocation.block_indices[0]]
        
        return embeddings, fill_ids, mrope, aux

    def set_buf(self, req: Req, allocation: MetadataAllocation):
        """
        Write request data to buffer.
        
        Data is scattered to potentially non-contiguous blocks.
        """
        embed_length = req.embedding.shape[0]
        ranges = allocation.get_contiguous_ranges(self.block_size)
        
        data_offset = 0
        
        for start_token, range_tokens in ranges:
            # Limit to actual embed_length
            range_tokens = min(range_tokens, embed_length - data_offset)
            if range_tokens <= 0:
                break
            
            end_token = start_token + range_tokens
            
            # Write to this contiguous range
            self.fill_ids[start_token:end_token] = torch.tensor(
                req.fill_ids[data_offset : data_offset + range_tokens]
            )
            
            self.input_embeddings[start_token * self.embedding_dim : end_token * self.embedding_dim] = \
                req.embedding[data_offset : data_offset + range_tokens].flatten()
            
            if req.multimodal_inputs and req.multimodal_inputs.mrope_positions is not None:
                self.mrope_positions[start_token * 3 : end_token * 3] = \
                    req.multimodal_inputs.mrope_positions[:, data_offset : data_offset + range_tokens].flatten().detach().cpu()
            
            data_offset += range_tokens
        
        # aux_datas stored in first block
        self.aux_datas[allocation.block_indices[0]][0] = embed_length
        if req.multimodal_inputs and req.multimodal_inputs.mrope_position_delta is not None:
            self.aux_datas[allocation.block_indices[0]][1] = req.multimodal_inputs.mrope_position_delta[0][0]
