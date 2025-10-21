from __future__ import annotations

import os
import random
from collections import deque
from contextlib import nullcontext
from enum import Enum
from typing import TYPE_CHECKING, Optional, Type

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

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


class ReqToMetadataIdxAllocator:
    """Block-based allocator for variable-length metadata buffers.
    
    Allocates blocks based on actual sequence length instead of fixed indices.
    Supports scatter/gather operations for efficient memory usage.
    """

    def __init__(
        self,
        size: int,
        block_size: int = None,
    ):
        """
        Args:
            size: Total number of blocks available
            block_size: Number of tokens per block (get from env var if not provided)
        """
        self.total_blocks = size
        self.block_size = block_size or int(
            os.getenv("SGLANG_MULTIMODAL_BLOCK_SIZE", "256")
        )
        self.free_blocks = deque(list(range(size)))
        self.req_to_blocks = {}  # req_id -> list of block indices

    def available_size(self):
        """Returns number of available blocks"""
        return len(self.free_blocks)

    def alloc(
        self, num_tokens: int = None, req_id: str = None, fake: bool = False
    ) -> Optional[list]:
        """Allocate blocks based on actual token length.
        
        Args:
            num_tokens: Actual number of tokens needed (if None, allocate 1 block)
            req_id: Request ID for tracking
            fake: Whether this is a fake allocation for warmup
            
        Returns:
            List of allocated block indices, or None if not enough blocks
        """
        if fake:
            # For fake requests, return a dummy block list
            return [random.randint(0, self.total_blocks - 1)]

        # Calculate required blocks
        if num_tokens is None:
            num_blocks_needed = 1
        else:
            num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            return None

        # Allocate consecutive blocks from free list
        allocated_blocks = [
            self.free_blocks.popleft() for _ in range(num_blocks_needed)
        ]

        if req_id is not None:
            self.req_to_blocks[req_id] = allocated_blocks

        return allocated_blocks

    def free(self, block_indices: list = None, req_id: str = None, fake: bool = False):
        """Free allocated blocks.
        
        Args:
            block_indices: List of block indices to free (used for backward compatibility)
            req_id: Request ID to free blocks for
            fake: Whether this is a fake deallocation
        """
        if fake:
            return

        # Support both old interface (single index) and new interface (block list)
        if block_indices is not None:
            if isinstance(block_indices, int):
                # Old interface compatibility
                self.free_blocks.append(block_indices)
            else:
                # New interface: list of blocks
                self.free_blocks.extend(block_indices)
        elif req_id is not None and req_id in self.req_to_blocks:
            blocks = self.req_to_blocks.pop(req_id)
            self.free_blocks.extend(blocks)

    def free_with_req(self, req: Req):
        """Free blocks associated with a request.
        
        This function is used to free blocks and reset the metadata buffer index of the request.
        NOTE: Only used in the disaggregation language mode:
              since transfer buffer need to be freed after the prefill is done.
        """
        fake = req.bootstrap_host == FAKE_BOOTSTRAP_HOST
        
        # Support both old (metadata_buffer_index) and new (metadata_block_indices) interfaces
        if hasattr(req, "metadata_block_indices") and req.metadata_block_indices:
            self.free(block_indices=req.metadata_block_indices, fake=fake)
            req.metadata_block_indices = None
        elif hasattr(req, "metadata_buffer_index") and req.metadata_buffer_index != -1:
            # Old interface compatibility
            self.free(block_indices=req.metadata_buffer_index, fake=fake)
            req.metadata_buffer_index = -1
        elif hasattr(req, "rid"):
            self.free(req_id=req.rid, fake=fake)


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
        self, size: int, max_prefill_tokens: int, embedding_dim: int = 8192
    ) -> None:
        """Initialize block-based multimodal data buffers.
        
        Args:
            size: Total number of blocks
            max_prefill_tokens: Block size (tokens per block)
            embedding_dim: Embedding dimension
        """
        # Treat size as total_blocks and max_prefill_tokens as block_size
        self.total_blocks = size
        self.block_size = max_prefill_tokens
        self.embedding_dim = embedding_dim
        
        # Block-based storage: each "row" is a block
        self.input_embeddings = torch.zeros(
            (size, max_prefill_tokens * embedding_dim),
            dtype=torch.bfloat16,
            device="cpu",
        )
        self.fill_ids = torch.zeros(
            (size, max_prefill_tokens), dtype=torch.int32, device="cpu"
        )
        # The minimal size for RDMA is 64Bytes, so we pad it to > 64Bytes
        self.mrope_positions = torch.zeros(
            (size, 3 * max_prefill_tokens), dtype=torch.int32, device="cpu"
        )
        # aux_datas: embedding_length, mrope_position_delta, to speedup transfer
        self.aux_datas = torch.zeros((size, 16), dtype=torch.int32, device="cpu")
        
        # For backward compatibility
        self.max_prefill_tokens = max_prefill_tokens

    def get_buf_chunk_info(self, req: Req):
        """Get chunk info for block-based transfer.
        
        Returns a list of chunk_info for each block, where each chunk_info is a list of
        (offset, size) tuples for each buffer type.
        
        Args:
            req: Request with metadata_block_indices and fill_ids
            
        Returns:
            List of chunk_info per block, where each chunk_info contains:
            [(embedding_offset, embedding_size), (fill_ids_offset, fill_ids_size), 
             (mrope_offset, mrope_size), (aux_offset, aux_size)]
        """
        block_indices = req.metadata_block_indices
        total_tokens = len(req.fill_ids)
        chunk_infos = []
        
        for block_idx, start_pos in enumerate(range(0, total_tokens, self.block_size)):
            end_pos = min(start_pos + self.block_size, total_tokens)
            block_len = end_pos - start_pos
            
            # For each block, return (offset_in_block, size_for_this_block)
            chunk_info = [
                # input_embeddings: offset=0, size=block_len*embedding_dim*itemsize
                (0, block_len * self.embedding_dim * self.input_embeddings.itemsize),
                # fill_ids: offset=0, size=block_len*itemsize
                (0, block_len * self.fill_ids.itemsize),
                # mrope_positions: offset=0, size=block_len*3*itemsize
                (0, block_len * 3 * self.mrope_positions.itemsize),
                # aux_datas: only transfer in first block
                (0, self.aux_datas.shape[1] * self.aux_datas.itemsize)
                if block_idx == 0
                else (0, 0),
            ]
            chunk_infos.append(chunk_info)
        
        return chunk_infos

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
        item_lens = [
            self.input_embeddings[0].nbytes,
            self.fill_ids[0].nbytes,
            self.mrope_positions[0].nbytes,
            self.aux_datas[0].nbytes,
        ]
        return ptrs, data_lens, item_lens

    def get_buf(self, idx: int = None, block_indices: list = None):
        """Get buffer data using either single index (old) or block indices (new).
        
        Args:
            idx: Single buffer index (for backward compatibility)
            block_indices: List of block indices for block-based access
            
        Returns:
            Tuple of (input_embeddings, fill_ids, mrope_positions, aux_datas)
        """
        # Backward compatibility: single index
        if idx is not None:
            input_embeddings = self.input_embeddings[idx].reshape(
                self.max_prefill_tokens, self.embedding_dim
            )
            fill_ids = self.fill_ids[idx]
            mrope_positions = self.mrope_positions[idx]
            aux_datas = self.aux_datas[idx]
            return input_embeddings, fill_ids, mrope_positions, aux_datas
        
        # New block-based access: gather from multiple blocks
        if block_indices is None or len(block_indices) == 0:
            raise ValueError("Either idx or block_indices must be provided")
        
        # Get total length from aux_datas in first block
        aux_datas = self.aux_datas[block_indices[0]]
        total_length = int(aux_datas[0])
        
        gathered_embeddings = []
        gathered_fill_ids = []
        gathered_mrope_positions = []
        
        tokens_gathered = 0
        for block_idx in block_indices:
            tokens_in_block = min(self.block_size, total_length - tokens_gathered)
            
            # Gather embeddings
            block_embed = self.input_embeddings[block_idx, : tokens_in_block * self.embedding_dim]
            gathered_embeddings.append(block_embed.reshape(tokens_in_block, self.embedding_dim))
            
            # Gather fill_ids
            gathered_fill_ids.append(self.fill_ids[block_idx, :tokens_in_block])
            
            # Gather mrope_positions
            gathered_mrope_positions.append(
                self.mrope_positions[block_idx, : 3 * tokens_in_block]
            )
            
            tokens_gathered += tokens_in_block
            if tokens_gathered >= total_length:
                break
        
        # Concatenate gathered data
        input_embeddings = torch.cat(gathered_embeddings, dim=0)
        fill_ids = torch.cat(gathered_fill_ids)
        mrope_positions = torch.cat(gathered_mrope_positions)
        
        return input_embeddings, fill_ids, mrope_positions, aux_datas

    def set_buf(self, req: Req):
        """Set buffer data using block-based scatter operation.
        
        Args:
            req: Request with metadata_block_indices, embedding, and fill_ids
        """
        embed_length = req.embedding.shape[0]
        total_tokens = len(req.fill_ids)
        
        # Support both old (metadata_buffer_index) and new (metadata_block_indices) interfaces
        if hasattr(req, "metadata_block_indices") and req.metadata_block_indices:
            block_indices = req.metadata_block_indices
        elif hasattr(req, "metadata_buffer_index") and req.metadata_buffer_index != -1:
            # Backward compatibility: treat single index as single-block list
            block_indices = [req.metadata_buffer_index]
        else:
            raise ValueError("Request must have metadata_block_indices or metadata_buffer_index")
        
        # Scatter data across allocated blocks
        tokens_written = 0
        for block_idx_pos, block_id in enumerate(block_indices):
            start_pos = tokens_written
            end_pos = min(start_pos + self.block_size, total_tokens)
            block_len = end_pos - start_pos
            
            if block_len <= 0:
                break
            
            # Scatter fill_ids
            self.fill_ids[block_id, :block_len] = torch.tensor(
                req.fill_ids[start_pos:end_pos]
            )
            
            # Scatter embeddings
            embed_start = min(start_pos, embed_length)
            embed_end = min(end_pos, embed_length)
            if embed_end > embed_start:
                self.input_embeddings[block_id, : (embed_end - embed_start) * self.embedding_dim] = (
                    req.embedding[embed_start:embed_end].flatten()
                )
            
            # Scatter mrope_positions
            if (
                req.multimodal_inputs is not None
                and req.multimodal_inputs.mrope_positions is not None
            ):
                mrope_start = min(start_pos, embed_length)
                mrope_end = min(end_pos, embed_length)
                if mrope_end > mrope_start:
                    self.mrope_positions[block_id, : 3 * (mrope_end - mrope_start)] = (
                        req.multimodal_inputs.mrope_positions[:, mrope_start:mrope_end]
                        .flatten()
                        .detach()
                        .cpu()
                    )
            
            # Store metadata in first block
            if block_idx_pos == 0:
                self.aux_datas[block_id][0] = embed_length
                if (
                    req.multimodal_inputs is not None
                    and req.multimodal_inputs.mrope_position_delta is not None
                ):
                    assert req.multimodal_inputs.mrope_position_delta.numel() == 1
                    self.aux_datas[block_id][1] = (
                        req.multimodal_inputs.mrope_position_delta[0][0]
                    )
            
            tokens_written += block_len
