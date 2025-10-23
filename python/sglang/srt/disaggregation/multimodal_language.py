"""
1. PreallocQueue:

2. TransferQueue:

3. WaitingQueue:

4. RunningBatch:
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import threading
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Deque, List, Tuple

import numpy as np
import torch
from torch.distributed import ProcessGroup

from sglang.srt.disaggregation.base import BaseKVReceiver, KVArgs, KVPoll
from sglang.srt.disaggregation.fake.conn import FakeKVReceiver
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MultimodalDataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    poll_and_all_reduce,
    prepare_abort,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    ScheduleBatch,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import DynamicGradMode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler

logger = logging.getLogger(__name__)


@dataclass
class MultimodalLanguageRequest:
    req: Req
    embedding_receiver: BaseKVReceiver
    waiting_for_input: bool = False
    embedding_indices: List[int] = None


class MultimodalLanguagePreallocQueue:
    def __init__(
        self,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MultimodalDataBuffers,
        tp_rank: int,
        tp_size: int,
        scheduler: Scheduler,
        transfer_backend: TransferBackend,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
    ):
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.transfer_backend = transfer_backend
        self.data_manager = self._init_data_manager()
        self.bootstrap_port = bootstrap_port
        self.queue: List[MultimodalLanguageRequest] = []
        self.gloo_group = gloo_group

        # Get default buffer size from environment variable
        # Language side only sees text, not the full embedding length from encode side
        # So we use a default buffer size (can be configured via env var)
        self.default_allocate_tokens = int(
            os.getenv(
                "SGLANG_EMBEDDING_DEFAULT_ALLOCATE_BUFFER_SIZE",
                "8192",
            )
        )

    def _init_data_manager(self):
        kv_args = KVArgs()
        kv_args.engine_rank = self.tp_rank
        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id
        data_manager_class = get_kv_class(
            self.transfer_backend, KVClassType.MANAGER, is_multimodal=True
        )
        data_manager = data_manager_class(
            kv_args,
            DisaggregationMode.LANGUAGE,
            self.scheduler.server_args,
        )
        return data_manager

    def add(self, req: Req):
        if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
            # Fake transfer for warmup reqs
            embedding_receiver_class = get_kv_class(
                TransferBackend.FAKE, KVClassType.RECEIVER, is_multimodal=True
            )
        else:
            embedding_receiver_class = get_kv_class(
                self.transfer_backend, KVClassType.RECEIVER, is_multimodal=True
            )
        embedding_receiver = embedding_receiver_class(
            mgr=self.data_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            prefill_dp_rank=req.data_parallel_rank,
        )
        self.queue.append(
            MultimodalLanguageRequest(req=req, embedding_receiver=embedding_receiver)
        )

    def extend(self, reqs: List[Req]):
        for req in reqs:
            self.add(req)

    def _update_handshake_waiters(self) -> None:
        if not self.queue:
            return

        if all(decode_req.waiting_for_input for decode_req in self.queue):
            return

        polls = poll_and_all_reduce(
            [language_req.embedding_receiver for language_req in self.queue],
            self.gloo_group,
        )

        for i, (language_req, poll) in enumerate(zip(self.queue, polls)):
            if poll == KVPoll.Bootstrapping:
                pass
            elif poll == KVPoll.WaitingForInput:
                language_req.waiting_for_input = True
            elif poll == KVPoll.Failed:
                error_message = f"MultimodalLanguage handshake failed for request rank={self.tp_rank} {language_req.req.rid=} {language_req.req.bootstrap_room=}"
                try:
                    language_req.embedding_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    language_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    def pop_preallocated(self):
        self._update_handshake_waiters()

        preallocated_reqs = []
        indices_to_remove = set()

        for i, language_req in enumerate(self.queue):
            if isinstance(language_req.req.finished_reason, FINISH_ABORT):
                self.scheduler.stream_output(
                    [language_req.req], language_req.req.return_logprob
                )
                indices_to_remove.add(i)

        for i, language_req in enumerate(self.queue):
            if i in indices_to_remove:
                continue

            if not language_req.waiting_for_input:
                continue

            if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
                break

            # Language side: allocate blocks based on default buffer size
            # Since we only have text here, not the full embedding from encode side
            language_req.embedding_indices = (
                self.req_to_metadata_buffer_idx_allocator.alloc(
                    num_tokens=self.default_allocate_tokens,
                    req_id=language_req.req.rid,
                    fake=isinstance(language_req.embedding_receiver, FakeKVReceiver),
                )
            )

            if language_req.embedding_indices is None:
                break

            # Calculate actual allocated tokens from allocated blocks
            # This ensures proper alignment with block_size
            actual_allocated_tokens = len(language_req.embedding_indices) * self.metadata_buffers.block_size

            # Initialize receiver with block_indices and actual allocated_tokens
            language_req.embedding_receiver.init(
                embedding_indices=language_req.embedding_indices,
                allocated_tokens=actual_allocated_tokens,
            )
            preallocated_reqs.append(language_req)
            indices_to_remove.add(i)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return preallocated_reqs


class MultimodalLanguageTransferQueue:
    def __init__(
        self,
        gloo_group: ProcessGroup,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MultimodalDataBuffers,
        scheduler: Scheduler,
        tree_cache: BasePrefixCache,
    ):
        self.queue: List[MultimodalLanguageRequest] = []
        self.gloo_group = gloo_group
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        self.tree_cache = tree_cache

    def add(self, req):
        self.queue.append(req)

    def extend(self, reqs):
        self.queue.extend(reqs)

    def _handle_failed_request(self, language_req: MultimodalLanguageRequest):
        error_message = f"MultiModalLanguage transfer failed for request rank={self.scheduler.tp_rank} {language_req.req.rid=} {language_req.req.bootstrap_room=}"
        try:
            language_req.embedding_receiver.failure_exception()
        except Exception as e:
            error_message += f" with exception {e}"
        logger.error(error_message)
        prepare_abort(
            language_req.req,
            error_message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        self.scheduler.stream_output(
            [language_req.req], language_req.req.return_logprob
        )
        # unlock the kv cache or it will have memory leak
        self.req_to_metadata_buffer_idx_allocator.free(
            block_indices=language_req.embedding_indices,
            req_id=language_req.req.rid,
            fake=isinstance(language_req.embedding_receiver, FakeKVReceiver),
        )
        if self.scheduler.enable_metrics:
            self.scheduler.metrics_collector.increment_transfer_failed_reqs()

    def pop_transferred(self):
        if not self.queue:
            return []

        polls = poll_and_all_reduce(
            [language_req.embedding_receiver for language_req in self.queue],
            self.gloo_group,
        )

        transferred_reqs = []
        indices_to_remove = set()
        for i, (language_req, poll) in enumerate(zip(self.queue, polls)):
            if poll == KVPoll.Failed:
                self._handle_failed_request(language_req)
                # unlock the kv cache or it will have memory leak
                indices_to_remove.add(i)
                continue
            elif poll == KVPoll.Success:
                # Use block_indices instead of single index
                block_indices = language_req.embedding_indices
                if not isinstance(language_req.embedding_receiver, FakeKVReceiver):
                    # Check if this is a resumed transfer (has partial data)
                    if hasattr(language_req.req, 'partial_input_embeds'):
                        # Resume transfer: need special handling for aux_datas
                        # The new blocks' aux_datas[0] is not set by Embedding side,
                        # so we can't use get_buf directly. Instead, we calculate
                        # the expected tokens from allocation.
                        
                        # Calculate expected tokens in resume transfer
                        block_size = self.metadata_buffers.block_size
                        partial_sent = language_req.req.partial_sent_tokens
                        total_expected = int(language_req.req.partial_aux_datas[0])
                        remaining_expected = total_expected - partial_sent
                        
                        logger.info(
                            f"Resume transfer for rid={language_req.req.rid}: "
                            f"partial_sent={partial_sent}, total_expected={total_expected}, "
                            f"remaining_expected={remaining_expected}, allocated_blocks={len(block_indices)}"
                        )
                        
                        # Gather data from blocks manually with correct token count
                        gathered_embeddings = []
                        gathered_fill_ids = []
                        gathered_mrope_positions = []
                        
                        tokens_gathered = 0
                        for block_idx in block_indices:
                            tokens_in_block = min(block_size, remaining_expected - tokens_gathered)
                            if tokens_in_block <= 0:
                                break
                            
                            # Gather embeddings
                            block_embed = self.metadata_buffers.input_embeddings[
                                block_idx, : tokens_in_block * self.metadata_buffers.embedding_dim
                            ]
                            gathered_embeddings.append(
                                block_embed.reshape(tokens_in_block, self.metadata_buffers.embedding_dim)
                            )
                            
                            # Gather fill_ids
                            gathered_fill_ids.append(
                                self.metadata_buffers.fill_ids[block_idx, :tokens_in_block]
                            )
                            
                            # Gather mrope_positions
                            gathered_mrope_positions.append(
                                self.metadata_buffers.mrope_positions[block_idx, : 3 * tokens_in_block].reshape(3, -1)
                            )
                            
                            tokens_gathered += tokens_in_block
                        
                        # Concatenate gathered data
                        embedding_data = torch.cat(gathered_embeddings, dim=0) if gathered_embeddings else torch.empty(0, self.metadata_buffers.embedding_dim)
                        fill_ids = torch.cat(gathered_fill_ids) if gathered_fill_ids else torch.empty(0, dtype=torch.int32)
                        mrope_positions = torch.cat(gathered_mrope_positions, dim=-1) if gathered_mrope_positions else torch.empty(3, 0, dtype=torch.int32)
                        
                        # Use cached aux_datas
                        aux_datas = language_req.req.partial_aux_datas
                    else:
                        # First time transfer: use normal get_buf
                        embedding_data, fill_ids, mrope_positions, aux_datas = (
                            self.metadata_buffers.get_buf(block_indices=block_indices)
                        )
                    
                    # Merge resumed data if applicable
                    if hasattr(language_req.req, 'partial_input_embeds'):
                        # Merge partial data with new data
                        logger.info(
                            f"Merging resumed transfer data for rid={language_req.req.rid}: "
                            f"partial_embeds_shape={language_req.req.partial_input_embeds.shape}, "
                            f"new_embeds_shape={embedding_data.shape}, "
                            f"partial_fill_ids_len={len(language_req.req.partial_fill_ids)}, "
                            f"new_fill_ids_len={len(fill_ids)}, "
                            f"partial_mrope_shape={language_req.req.partial_mrope_positions.shape}, "
                            f"new_mrope_shape={mrope_positions.shape}"
                        )
                        
                        # Concatenate embeddings
                        embedding_data = torch.cat([
                            language_req.req.partial_input_embeds,
                            embedding_data
                        ])
                        
                        # Concatenate fill_ids
                        fill_ids = torch.cat([
                            torch.tensor(language_req.req.partial_fill_ids),
                            fill_ids
                        ])
                        
                        # Concatenate mrope_positions (handle empty tensors)
                        # mrope_positions shape: (3, num_tokens) or could be empty
                        partial_mrope = language_req.req.partial_mrope_positions
                        if partial_mrope.numel() > 0 and mrope_positions.numel() > 0:
                            # Both have data, concatenate along last dimension
                            logger.info(
                                f"Concatenating mrope_positions: partial shape={partial_mrope.shape}, "
                                f"new shape={mrope_positions.shape}"
                            )
                            mrope_positions = torch.cat([partial_mrope, mrope_positions], dim=-1)
                        elif partial_mrope.numel() > 0:
                            # Only first part has data
                            logger.info(
                                f"Using only partial mrope_positions: shape={partial_mrope.shape}"
                            )
                            mrope_positions = partial_mrope
                        else:
                            # Use new mrope_positions (or empty)
                            logger.info(
                                f"Using only new mrope_positions: shape={mrope_positions.shape}"
                            )
                        
                        # Use aux_datas from first transfer (contains total length)
                        aux_datas = language_req.req.partial_aux_datas
                        
                        # Clean up partial data
                        del language_req.req.partial_input_embeds
                        del language_req.req.partial_fill_ids
                        del language_req.req.partial_mrope_positions
                        del language_req.req.partial_aux_datas
                        del language_req.req.partial_sent_tokens
                    
                    embedding_length = int(aux_datas[0])
                    mrope_position_delta = aux_datas[1]
                    language_req.req.input_embeds = embedding_data
                    language_req.req.origin_input_ids = fill_ids.tolist()
                    mm_inputs = None
                    ori_input_length = len(language_req.req.origin_input_ids)

                    if ori_input_length == embedding_length:
                        mm_inputs = None
                    elif ori_input_length < embedding_length:
                        # NOTE: mock mm_inputs to make mm_inputs not None
                        # need to be checked carefully for modality-attributes
                        mm_inputs = MultimodalInputs(
                            mm_items=[
                                MultimodalDataItem(
                                    modality=Modality.IMAGE, model_specific_data={}
                                ),
                            ]
                        )
                        mm_inputs.mrope_positions = mrope_positions
                        mm_inputs.mrope_position_delta = torch.tensor(
                            [mrope_position_delta]
                        ).unsqueeze(1)
                    else:
                        # take as transfer failed case
                        self._handle_failed_request(language_req)
                        indices_to_remove.add(i)
                        continue
                    language_req.req.multimodal_inputs = mm_inputs
                    # NOTE: we need to set the metadata block indices to the request
                    # because the metadata buffer should be freed after the request prefill forward finished
                    language_req.req.embedding_indices = language_req.embedding_indices
                else:
                    self.req_to_metadata_buffer_idx_allocator.free(
                        block_indices=block_indices, fake=True
                    )

                transferred_reqs.append(language_req.req)
                indices_to_remove.add(i)
            elif poll == KVPoll.Transferring:
                # Partial transfer complete, need to resume with remaining data
                block_indices = language_req.embedding_indices
                if not isinstance(language_req.embedding_receiver, FakeKVReceiver):
                    # Get partial data and actual total length from aux_datas
                    embedding_data, fill_ids, mrope_positions, aux_datas = (
                        self.metadata_buffers.get_buf(block_indices=block_indices)
                    )
                    
                    # In multi-TP scenario, some ranks may not have received data yet
                    # or may be dummy ranks. We need to sync aux_datas across all ranks.
                    actual_total_length = int(aux_datas[0])  # Actual total length (may be 0 on some ranks)
                    sent_tokens = len(fill_ids)  # Tokens already sent (may be 0 on some ranks)
                    
                    # Sync actual_total_length and sent_tokens across all ranks using max
                    # (the rank that has data will have non-zero values)
                    import torch.distributed as dist
                    if self.gloo_group is not None:
                        actual_total_length_tensor = torch.tensor([actual_total_length], dtype=torch.int64)
                        sent_tokens_tensor = torch.tensor([sent_tokens], dtype=torch.int64)
                        
                        dist.all_reduce(actual_total_length_tensor, op=dist.ReduceOp.MAX, group=self.gloo_group)
                        dist.all_reduce(sent_tokens_tensor, op=dist.ReduceOp.MAX, group=self.gloo_group)
                        
                        actual_total_length = int(actual_total_length_tensor.item())
                        sent_tokens = int(sent_tokens_tensor.item())
                        
                        logger.info(
                            f"Synced aux_datas for rid={language_req.req.rid}: "
                            f"actual_total_length={actual_total_length}, sent_tokens={sent_tokens}"
                        )
                    
                    if actual_total_length > sent_tokens:
                        # Need to resume transfer
                        remaining_tokens = actual_total_length - sent_tokens
                        
                        logger.info(
                            f"Partial transfer detected for rid={language_req.req.rid}: "
                            f"received {sent_tokens}/{actual_total_length} tokens, "
                            f"need to resume for {remaining_tokens} more tokens"
                        )
                        
                        # Cache received partial data (only if this rank has data)
                        # Some ranks may be dummy ranks and don't have data
                        if not hasattr(language_req.req, 'partial_input_embeds'):
                            # Only cache if we actually have data on this rank
                            has_data = (len(fill_ids) > 0)
                            
                            if has_data:
                                language_req.req.partial_input_embeds = embedding_data
                                language_req.req.partial_fill_ids = fill_ids.tolist()
                                language_req.req.partial_mrope_positions = mrope_positions
                                language_req.req.partial_aux_datas = torch.tensor([actual_total_length, aux_datas[1]])  # Use synced value
                                language_req.req.partial_sent_tokens = sent_tokens
                            else:
                                # Dummy rank: create placeholder partial data using synced values
                                language_req.req.partial_input_embeds = torch.empty(0, self.metadata_buffers.embedding_dim)
                                language_req.req.partial_fill_ids = []
                                language_req.req.partial_mrope_positions = torch.empty(3, 0, dtype=torch.int32)
                                language_req.req.partial_aux_datas = torch.tensor([actual_total_length, 0])  # Use synced value
                                language_req.req.partial_sent_tokens = sent_tokens
                        
                        # Free old allocation
                        self.req_to_metadata_buffer_idx_allocator.free(
                            block_indices=block_indices,
                            req_id=language_req.req.rid,
                            fake=isinstance(language_req.embedding_receiver, FakeKVReceiver),
                        )
                        
                        # Allocate new space for remaining tokens
                        new_allocation = self.req_to_metadata_buffer_idx_allocator.alloc(
                            num_tokens=remaining_tokens,
                            req_id=language_req.req.rid,
                            fake=isinstance(language_req.embedding_receiver, FakeKVReceiver),
                        )
                        
                        if new_allocation is None:
                            # Not enough memory to resume, mark as failed
                            logger.error(
                                f"Not enough memory to resume transfer for rid={language_req.req.rid}, "
                                f"need {remaining_tokens} tokens"
                            )
                            self._handle_failed_request(language_req)
                            indices_to_remove.add(i)
                            continue
                        
                        # Update embedding_indices
                        language_req.embedding_indices = new_allocation
                        
                        # Calculate allocated_tokens from new allocation
                        block_size = self.metadata_buffers.block_size
                        allocated_tokens = len(new_allocation) * block_size
                        
                        # Send resume request
                        language_req.embedding_receiver.resume_transfer(
                            embedding_indices=new_allocation,
                            sent_tokens=sent_tokens,
                            allocated_tokens=allocated_tokens,
                        )
                        
                        logger.info(
                            f"Resume transfer initiated for rid={language_req.req.rid}: "
                            f"allocated {len(new_allocation)} blocks ({allocated_tokens} tokens)"
                        )
                    else:
                        # This shouldn't happen - Transferring status but all data received
                        logger.warning(
                            f"Unexpected: Transferring status but sent_tokens={sent_tokens} >= "
                            f"actual_total_length={actual_total_length}"
                        )
                # Continue waiting for transfer to complete
                pass
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
            ]:
                pass
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        for i in indices_to_remove:
            block_indices = self.queue[i].embedding_indices
            assert block_indices is not None and len(block_indices) > 0

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]
        return transferred_reqs


class SchedulerDisaggregationMultiModalLanguageMixin:

    @DynamicGradMode()
    def event_loop_normal_disagg_multimodal_language(self: Scheduler):
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # polling and allocating kv cache
            self.process_multimodal_language_queue()
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch

    @DynamicGradMode()
    def event_loop_overlap_disagg_multimodal_language(self: Scheduler):
        self.result_queue: Deque[Tuple[ScheduleBatch, GenerationBatchResult]] = deque()

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.process_multimodal_language_queue()
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            batch_result = None
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))

            if self.last_batch:
                # Process the results of the last batch
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()

            self.launch_batch_sample_if_needed(batch_result)
            self.last_batch = batch

    def process_multimodal_language_queue(self: Scheduler):
        req_conns = self.disagg_language_prealloc_queue.pop_preallocated()
        self.disagg_language_transfer_queue.extend(req_conns)
        alloc_reqs = self.disagg_language_transfer_queue.pop_transferred()
        self.waiting_queue.extend(alloc_reqs)
