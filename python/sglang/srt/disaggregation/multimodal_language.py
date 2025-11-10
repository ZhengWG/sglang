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
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple

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
    embedding_indices: List[int] = None
    requested_seq_len: Optional[int] = None
    requested_block_size: Optional[int] = None
    allocation_received: bool = False


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
        self.requests_by_room: Dict[int, MultimodalLanguageRequest] = {}
        self.pending_remote_events: Dict[int, Dict[str, int]] = {}
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
        language_req = MultimodalLanguageRequest(
            req=req, embedding_receiver=embedding_receiver
        )
        self.queue.append(language_req)
        self.requests_by_room[req.bootstrap_room] = language_req
        if req.bootstrap_room in self.pending_remote_events:
            event = self.pending_remote_events.pop(req.bootstrap_room)
            language_req.requested_seq_len = event["seq_len"]
            language_req.requested_block_size = event["block_size"]
            language_req.allocation_received = True

    def extend(self, reqs: List[Req]):
        for req in reqs:
            self.add(req)

    def pop_preallocated(self):
        self._drain_remote_allocations()

        preallocated_reqs = []
        indices_to_remove = set()

        if not self.queue:
            return preallocated_reqs

        polls = poll_and_all_reduce(
            [language_req.embedding_receiver for language_req in self.queue],
            self.gloo_group,
        )

        for idx, (language_req, poll) in enumerate(zip(self.queue, polls)):
            if isinstance(language_req.req.finished_reason, FINISH_ABORT):
                self.scheduler.stream_output(
                    [language_req.req], language_req.req.return_logprob
                )
                indices_to_remove.add(idx)
                self.requests_by_room.pop(language_req.req.bootstrap_room, None)
                continue

            if poll == KVPoll.Failed:
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
                indices_to_remove.add(idx)
                self.requests_by_room.pop(language_req.req.bootstrap_room, None)
                continue

            if not language_req.allocation_received:
                continue

            if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
                break

            requested_tokens = (
                language_req.requested_seq_len
                if language_req.requested_seq_len is not None
                else self.default_allocate_tokens
            )

            language_req.embedding_indices = (
                self.req_to_metadata_buffer_idx_allocator.alloc(
                    num_tokens=requested_tokens,
                    req_id=language_req.req.rid,
                    fake=isinstance(language_req.embedding_receiver, FakeKVReceiver),
                )
            )

            if language_req.embedding_indices is None:
                # Not enough blocks, wait for next iteration
                continue

            actual_allocated_tokens = (
                len(language_req.embedding_indices) * self.metadata_buffers.block_size
            )
            language_req.embedding_receiver.init(
                embedding_indices=language_req.embedding_indices,
                allocated_tokens=actual_allocated_tokens,
            )
            language_req.allocation_received = False
            preallocated_reqs.append(language_req)
            indices_to_remove.add(idx)
            self.requests_by_room.pop(language_req.req.bootstrap_room, None)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return preallocated_reqs

    def _drain_remote_allocations(self):
        events = self.data_manager.drain_remote_allocations()
        for event in events:
            room = event["room"]
            language_req = self.requests_by_room.get(room)
            if language_req is None:
                self.pending_remote_events[room] = event
                continue
            language_req.requested_seq_len = event["seq_len"]
            language_req.requested_block_size = event["block_size"]
            language_req.allocation_received = True


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
                    (
                        embedding_data,
                        fill_ids,
                        mrope_positions,
                        aux_datas,
                        deepstack_embedding,
                    ) = self.metadata_buffers.get_buf(block_indices=block_indices)

                    embedding_length = int(aux_datas[0])
                    mrope_position_delta = aux_datas[1]
                    mm_inputs = None
                    ori_input_length = len(language_req.req.origin_input_ids)
                    language_req.req.origin_input_ids = fill_ids.tolist()

                    if deepstack_embedding is not None:
                        # NOTE: merge input_embeds and deepstack_embedding to input_embeds to
                        # simplify the model forward pass
                        language_req.req.input_embeds = torch.cat(
                            [embedding_data, deepstack_embedding],
                            dim=-1,
                        ).contiguous()
                    else:
                        language_req.req.input_embeds = embedding_data

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
                # Still transferring; wait for next iteration
                continue
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
