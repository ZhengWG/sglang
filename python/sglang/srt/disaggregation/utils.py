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

    block_indices: List[int]  # List of allocated block indices
    num_tokens: int  # Actual number of tokens needed
    start_offset: int = 0  # Starting offset within the first block (usually 0)


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
    """
    Block-based metadata buffer allocator.
    Divides the buffer into fixed-size blocks and allocates based on actual token requirements.
    """

    def __init__(
        self,
        total_tokens: int,  # Total token capacity
        block_size: int = 128,  # Tokens per block
    ):
        self.total_tokens = total_tokens
        self.block_size = block_size
        self.num_blocks = (total_tokens + block_size - 1) // block_size

        # Use deque to maintain free blocks
        self.free_blocks = deque(list(range(self.num_blocks)))

        # Track allocations: req_id -> MetadataAllocation
        self.allocations: Dict[int, MetadataAllocation] = {}

        logger.info(
            f"Initialized ReqToMetadataBlockAllocator: "
            f"total_tokens={total_tokens}, block_size={block_size}, "
            f"num_blocks={self.num_blocks}"
        )

    def available_size(self) -> int:
        """Returns available tokens (approximate)."""
        return len(self.free_blocks) * self.block_size

    def available_blocks(self) -> int:
        """Returns number of available blocks."""
        return len(self.free_blocks)

    def alloc(
        self,
        num_tokens: int = None,  # Number of tokens needed, None = one block
        fake: bool = False,
        req_id: int = None,  # Request ID for tracking
    ) -> Optional[MetadataAllocation]:
        """
        Allocate blocks for specified number of tokens.

        Args:
            num_tokens: Number of tokens to allocate. If None, allocate one block.
            fake: Whether this is a fake allocation (for testing).
            req_id: Request ID for later freeing.

        Returns:
            MetadataAllocation object with block info, or None if allocation fails.
        """
        if fake:
            # Fake allocation, return random allocation
            fake_blocks = [random.randint(0, self.num_blocks - 1)]
            return MetadataAllocation(
                block_indices=fake_blocks,
                num_tokens=num_tokens or self.block_size,
                start_offset=0,
            )

        # Default to one block if num_tokens not specified
        if num_tokens is None:
            num_tokens = self.block_size

        # Calculate number of blocks needed
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            logger.warning(
                f"Allocation failed: need {num_blocks_needed} blocks "
                f"({num_tokens} tokens), but only {len(self.free_blocks)} blocks available"
            )
            return None

        # Allocate blocks (remove from free_blocks)
        allocated_blocks = []
        for _ in range(num_blocks_needed):
            allocated_blocks.append(self.free_blocks.popleft())

        allocation = MetadataAllocation(
            block_indices=allocated_blocks,
            num_tokens=num_tokens,
            start_offset=0,
        )

        # Record allocation
        if req_id is not None:
            self.allocations[req_id] = allocation

        logger.debug(
            f"Allocated {num_blocks_needed} blocks for {num_tokens} tokens: "
            f"blocks={allocated_blocks}"
        )

        return allocation

    def alloc_default(
        self, default_tokens: int, fake: bool = False, req_id: int = None
    ) -> Optional[MetadataAllocation]:
        """Allocate default size buffer (used by Language side for initial allocation)."""
        return self.alloc(num_tokens=default_tokens, fake=fake, req_id=req_id)

    def free(
        self, allocation: MetadataAllocation, fake: bool = False, req_id: int = None
    ):
        """
        Free allocated blocks.

        Args:
            allocation: Allocation to free.
            fake: Whether this is a fake free.
            req_id: Request ID for cleanup.
        """
        if fake:
            return

        # Return blocks to free_blocks
        for block_idx in allocation.block_indices:
            self.free_blocks.append(block_idx)

        # Cleanup record
        if req_id is not None and req_id in self.allocations:
            del self.allocations[req_id]

        logger.debug(
            f"Freed {len(allocation.block_indices)} blocks: {allocation.block_indices}"
        )

    def free_by_req_id(self, req_id: int, fake: bool = False):
        """Free allocation by request ID."""
        if fake:
            return

        if req_id in self.allocations:
            allocation = self.allocations[req_id]
            self.free(allocation, fake=False, req_id=req_id)
        else:
            logger.warning(f"Attempted to free unknown req_id: {req_id}")

    def get_allocation(self, req_id: int) -> Optional[MetadataAllocation]:
        """Get allocation info for a request."""
        return self.allocations.get(req_id)


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
    def __init__(
        self,
        size: int,
        max_prefill_tokens: int,
        embedding_dim: int = 8192,
        block_size: int = 128,  # New: tokens per block
        use_block_allocator: bool = False,  # New: enable block-based allocation
    ) -> None:
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.use_block_allocator = use_block_allocator

        if use_block_allocator:
            # Block-based allocation: single contiguous buffer
            self.total_capacity_tokens = size * max_prefill_tokens
            self.num_blocks = (self.total_capacity_tokens + block_size - 1) // block_size

            # Buffers are contiguous memory, logically managed as blocks
            self.input_embeddings = torch.zeros(
                (self.total_capacity_tokens * embedding_dim,),
                dtype=torch.bfloat16,
                device="cpu",
            )
            self.fill_ids = torch.zeros(
                (self.total_capacity_tokens,), dtype=torch.int32, device="cpu"
            )
            self.mrope_positions = torch.zeros(
                (3 * self.total_capacity_tokens,), dtype=torch.int32, device="cpu"
            )
            # aux_datas: one per block
            self.aux_datas = torch.zeros((self.num_blocks, 16), dtype=torch.int32, device="cpu")

            logger.info(
                f"Initialized MultimodalDataBuffers (block-based): "
                f"capacity={self.total_capacity_tokens} tokens, "
                f"block_size={block_size}, num_blocks={self.num_blocks}"
            )
        else:
            # Original index-based allocation
            self.input_embeddings = torch.zeros(
                (size, max_prefill_tokens * embedding_dim),
                dtype=torch.bfloat16,
                device="cpu",
            )
            self.fill_ids = torch.zeros(
                (size, max_prefill_tokens), dtype=torch.int32, device="cpu"
            )
            self.mrope_positions = torch.zeros(
                (size, 3 * max_prefill_tokens), dtype=torch.int32, device="cpu"
            )
            self.aux_datas = torch.zeros((size, 16), dtype=torch.int32, device="cpu")

        self.max_prefill_tokens = max_prefill_tokens
        # Default buffer size for Language side initial allocation
        self.default_buffer_tokens = int(
            os.getenv("SGLANG_DEFAULT_MULTIMODAL_BUFFER_TOKENS", "1024")
        )

    def get_buf_chunk_info(
        self,
        req: Req = None,
        allocation: MetadataAllocation = None,
        offset_tokens: int = 0,
        max_tokens: int = None,
    ):
        """
        Calculate chunk info for transfer.

        Args:
            req: Request object (for index-based allocation)
            allocation: MetadataAllocation (for block-based allocation)
            offset_tokens: Token offset for continuation
            max_tokens: Max tokens to transfer (for first transfer limit)

        Returns:
            [(chunk_offset_bytes, chunk_size_bytes), ...] for each buffer type
        """
        if self.use_block_allocator:
            # Block-based allocation
            assert allocation is not None, "allocation required for block-based mode"
            block_indices = allocation.block_indices
            num_tokens = allocation.num_tokens

            # Calculate actual tokens to transfer
            actual_tokens = num_tokens - offset_tokens
            if max_tokens is not None:
                actual_tokens = min(actual_tokens, max_tokens)

            # Calculate starting position
            start_token = min(block_indices) * self.block_size + offset_tokens

            return [
                # input_embeddings
                (
                    start_token * self.embedding_dim * self.input_embeddings.itemsize,
                    actual_tokens * self.embedding_dim * self.input_embeddings.itemsize,
                ),
                # fill_ids
                (start_token * self.fill_ids.itemsize, actual_tokens * self.fill_ids.itemsize),
                # mrope_positions
                (
                    start_token * 3 * self.mrope_positions.itemsize,
                    actual_tokens * 3 * self.mrope_positions.itemsize,
                ),
                # aux_datas: always transfer first block's aux_data
                (block_indices[0] * self.aux_datas[0].nbytes, self.aux_datas[0].nbytes),
            ]
        else:
            # Original index-based allocation
            assert req is not None, "req required for index-based mode"
            return [
                (
                    0,
                    len(req.fill_ids) * self.embedding_dim * self.input_embeddings.itemsize,
                ),
                (0, len(req.fill_ids) * self.fill_ids.itemsize),
                (0, len(req.fill_ids) * 3 * self.mrope_positions.itemsize),
                (0, self.aux_datas.shape[1] * self.aux_datas.itemsize),
            ]

    def get_buf_infos(self):
        ptrs = [
            self.input_embeddings.data_ptr(),
            self.fill_ids.data_ptr(),
            self.mrope_positions.data_ptr(),
            self.aux_datas.data_ptr(),
        ]
        data_lens = [
            self.input_embeddings.nbytes,
            self.fill_ids.nbytes,
            self.mrope_positions.nbytes,
            self.aux_datas.nbytes,
        ]

        if self.use_block_allocator:
            # item_lens: per block
            item_lens = [
                self.block_size * self.embedding_dim * self.input_embeddings.itemsize,
                self.block_size * self.fill_ids.itemsize,
                3 * self.block_size * self.mrope_positions.itemsize,
                self.aux_datas[0].nbytes,
            ]
        else:
            # item_lens: per index
            item_lens = [
                self.input_embeddings[0].nbytes,
                self.fill_ids[0].nbytes,
                self.mrope_positions[0].nbytes,
                self.aux_datas[0].nbytes,
            ]

        return ptrs, data_lens, item_lens

    def get_buf(self, idx: int = None, allocation: MetadataAllocation = None):
        """
        Get buffer data.

        Args:
            idx: Buffer index (for index-based allocation)
            allocation: MetadataAllocation (for block-based allocation)

        Returns:
            (input_embeddings, fill_ids, mrope_positions, aux_datas)
        """
        if self.use_block_allocator:
            assert allocation is not None, "allocation required for block-based mode"
            block_indices = allocation.block_indices
            num_tokens = allocation.num_tokens

            # Calculate token range
            start_token = min(block_indices) * self.block_size
            end_token = start_token + num_tokens

            # Extract data
            input_embeddings = (
                self.input_embeddings[
                    start_token * self.embedding_dim : end_token * self.embedding_dim
                ]
                .reshape(num_tokens, self.embedding_dim)
            )
            fill_ids = self.fill_ids[start_token:end_token]
            mrope_positions = self.mrope_positions[start_token * 3 : end_token * 3]
            # aux_datas: use first block's
            aux_datas = self.aux_datas[block_indices[0]]

            return input_embeddings, fill_ids, mrope_positions, aux_datas
        else:
            # Original index-based allocation
            assert idx is not None, "idx required for index-based mode"
            input_embeddings = self.input_embeddings[idx].reshape(
                self.max_prefill_tokens, self.embedding_dim
            )
            fill_ids = self.fill_ids[idx]
            mrope_positions = self.mrope_positions[idx]
            aux_datas = self.aux_datas[idx]
            return input_embeddings, fill_ids, mrope_positions, aux_datas

    def set_buf(self, req: Req, allocation: MetadataAllocation = None):
        """
        Write request data to buffer.

        Args:
            req: Request object
            allocation: MetadataAllocation (for block-based allocation)
        """
        embed_length = req.embedding.shape[0]

        if self.use_block_allocator:
            assert allocation is not None, "allocation required for block-based mode"
            block_indices = allocation.block_indices

            # Calculate starting position
            start_token = min(block_indices) * self.block_size
            end_token = start_token + embed_length

            # Write data
            self.fill_ids[start_token:end_token] = torch.tensor(
                req.fill_ids[:embed_length]
            )

            if (
                req.multimodal_inputs is not None
                and req.multimodal_inputs.mrope_positions is not None
            ):
                self.mrope_positions[start_token * 3 : end_token * 3] = (
                    req.multimodal_inputs.mrope_positions[:, :embed_length]
                    .flatten()
                    .detach()
                    .cpu()
                )

            self.input_embeddings[
                start_token * self.embedding_dim : end_token * self.embedding_dim
            ] = req.embedding.flatten()

            # Write aux_datas to first block
            self.aux_datas[block_indices[0]][0] = embed_length
            if (
                req.multimodal_inputs is not None
                and req.multimodal_inputs.mrope_position_delta is not None
            ):
                assert req.multimodal_inputs.mrope_position_delta.numel() == 1
                self.aux_datas[block_indices[0]][1] = (
                    req.multimodal_inputs.mrope_position_delta[0][0]
                )
        else:
            # Original index-based allocation
            idx = req.metadata_buffer_index
            self.fill_ids[idx, : len(req.fill_ids)] = torch.tensor(req.fill_ids)

            if (
                req.multimodal_inputs is not None
                and req.multimodal_inputs.mrope_positions is not None
            ):
                self.mrope_positions[idx, : 3 * embed_length] = (
                    req.multimodal_inputs.mrope_positions.flatten().detach().cpu()
                )

            self.input_embeddings[idx, : embed_length * self.embedding_dim] = (
                req.embedding.flatten()
            )

            self.aux_datas[idx][0] = embed_length
            if (
                req.multimodal_inputs is not None
                and req.multimodal_inputs.mrope_position_delta is not None
            ):
                assert req.multimodal_inputs.mrope_position_delta.numel() == 1
                self.aux_datas[idx][1] = req.multimodal_inputs.mrope_position_delta[0][0]
