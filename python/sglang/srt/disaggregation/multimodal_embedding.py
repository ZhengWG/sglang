"""
1. Bootstrap Queue

2. Waiting Queue

3. Inflight Queue

"""

from __future__ import annotations

import logging
import time
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional

import torch
from torch.distributed import ProcessGroup

from sglang.srt.disaggregation.base import BaseKVManager, KVArgs, KVPoll
from sglang.srt.disaggregation.fake.conn import FakeKVSender
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
    FINISH_LENGTH,
    Req,
    ScheduleBatch,
    get_global_server_args,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler


logger = logging.getLogger(__name__)


class MultimodalEmbeddingBootstrapQueue:
    def __init__(
        self,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MultimodalDataBuffers,
        tp_rank: int,
        tp_size: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        transfer_backend: TransferBackend,
        scheduler: Scheduler,
    ):
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.metadata_buffers = metadata_buffers
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.transfer_backend = transfer_backend
        self.scheduler = scheduler
        self.data_manager = self._init_data_manager()
        self.queue: List[Req] = []
        self.gloo_group = gloo_group
        self.bootstrap_port = bootstrap_port

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
            DisaggregationMode.ENCODE,
            self.scheduler.server_args,
        )
        return data_manager

    def add(self, req: Req):
        if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
            # Fake transfer for warmup reqs
            embedding_sender_class = get_kv_class(
                TransferBackend.FAKE, KVClassType.SENDER, is_multimodal=True
            )
        else:
            embedding_sender_class = get_kv_class(
                self.transfer_backend, KVClassType.SENDER, is_multimodal=True
            )
        dest_tp_ranks = [self.tp_rank]
        req.disagg_embedding_sender = embedding_sender_class(
            mgr=self.data_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            dest_tp_ranks=dest_tp_ranks,
            pp_rank=0,
        )
        self.queue.append(req)

    def extend(self, reqs: List[Req]):
        for req in reqs:
            self.add(req)

    def pop_bootstrapped(self):
        """pop the reqs which has finished bootstrapping"""
        bootstrapped_reqs = []
        indices_to_remove = set()

        if len(self.queue) == 0:
            return []

        polls = poll_and_all_reduce(
            [req.disagg_embedding_sender for req in self.queue], self.gloo_group
        )

        for i, (req, poll) in enumerate(zip(self.queue, polls)):
            if poll == KVPoll.Bootstrapping:
                continue
            elif poll == KVPoll.Failed:
                error_message = f"MultimodalEmbedding bootstrap failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                try:
                    req.disagg_embedding_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                # TODO: check embedding return_logprob
                self.scheduler.stream_output([req], req.return_logprob)
                indices_to_remove.add(i)
                continue

            # For block-based allocator, defer allocation until we know actual length
            # For index-based allocator, allocate now
            if self.metadata_buffers.use_block_allocator:
                # Defer allocation: will allocate in process_batch_result when actual length is known
                req.metadata_buffer_index = -1
                req.metadata_allocation = None
                # Still call init to establish connection with Language side
                req.disagg_embedding_sender.init(embedding_index=0)  # Placeholder index
            else:
                # Legacy index-based: allocate now
                if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
                    break

                req.metadata_buffer_index = self.req_to_metadata_buffer_idx_allocator.alloc(
                    fake=isinstance(req.disagg_embedding_sender, FakeKVSender)
                )

                assert req.metadata_buffer_index is not None
                req.disagg_embedding_sender.init(embedding_index=req.metadata_buffer_index)
            
            bootstrapped_reqs.append(req)
            indices_to_remove.add(i)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]
        return bootstrapped_reqs


class SchedulerDisaggregationMultimodalEmbeddingMixin:

    @torch.no_grad()
    def event_loop_normal_disagg_multimodal_embedding(self: Scheduler):
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_embedding_bootstrap_queue.pop_bootstrapped()
            )
            self.process_embedding_chunk()
            batch = self.get_new_batch_prefill()

            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                logger.debug(
                    f"End to run batch {[req.bootstrap_room for req in batch.reqs]}"
                )
                self.process_batch_result_disagg_multimodal_embedding(batch, result)

            if len(self.disagg_embedding_inflight_queue) > 0:
                self.process_multimodal_embedding_inflight_queue()

            if batch is None and len(self.disagg_embedding_inflight_queue) == 0:
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch
            self.running_batch.batch_is_full = False

    def process_batch_result_disagg_multimodal_embedding(
        self: Scheduler, batch: ScheduleBatch, result: EmbeddingBatchResult
    ):
        """
        Transfer embedding results for completed requests and add it into disagg_multimodal_embedding_inflight_queue
        """
        embeddings = result.embeddings

        embedding_offsets = 0
        dummy_output_ids = []
        for i, req in enumerate(batch.reqs):
            if req.is_retracted:
                continue

            embedding = embeddings[
                embedding_offsets : embedding_offsets + req.extend_input_len
            ]
            if req.embedding is None:
                req.embedding = embedding
            else:
                req.embedding = torch.cat([req.embedding, embedding])
            embedding_offsets += req.extend_input_len
            if req.is_chunked <= 0:
                # Dummy output token for embedding models
                req.output_ids.append(0)
                dummy_output_ids.append(0)
                
                # For block-based allocator: allocate buffer based on actual length
                if self.disagg_metadata_buffers.use_block_allocator:
                    actual_length = req.embedding.shape[0]
                    logger.debug(
                        f"Embedding request {req.rid} room={req.bootstrap_room} "
                        f"actual length={actual_length} tokens"
                    )
                    
                    # Allocate buffer based on actual length
                    allocation = self.req_to_metadata_buffer_idx_allocator.alloc(
                        num_tokens=actual_length,
                        fake=isinstance(req.disagg_embedding_sender, FakeKVSender),
                        req_id=req.rid
                    )
                    
                    if allocation is None:
                        logger.error(
                            f"Failed to allocate buffer for {actual_length} tokens, "
                            f"request {req.rid} will fail"
                        )
                        # TODO: handle allocation failure
                        continue
                    
                    req.metadata_allocation = allocation
                    req.metadata_buffer_index = allocation.block_indices[0]  # Use first block index
                
                # release kv cache immediately for embedding models
                # in order to speed up for batch forward
                self.tree_cache.cache_finished_req(req)
                self.disagg_embedding_inflight_queue.append(req)
                self.send_embedding_chunk(req, last_chunk=False)  # May not be last if continuation needed
            else:
                req.is_chunked -= 1

        if len(dummy_output_ids) > 0:
            batch.output_ids = torch.tensor(dummy_output_ids).to(
                self.device, non_blocking=True
            )

    def process_multimodal_embedding_inflight_queue(self: Scheduler):
        """
        Poll the requests in the middle of transfer. If done, return the request.
        """
        assert len(self.disagg_embedding_inflight_queue) > 0

        done_reqs = []

        # NOTE: no need poll_and_all_reduce for vision embedding, because all data is same
        polls = poll_and_all_reduce(
            [
                req.disagg_embedding_sender
                for req in self.disagg_embedding_inflight_queue
            ],
            self.attn_tp_cpu_group,
        )

        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in disagg_embedding_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_embedding_inflight_queue, polls):
            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                # Still transferring, or waiting for continuation request from Language side
                undone_reqs.append(req)
                
                # Log for debugging continuation scenario
                if poll == KVPoll.Transferring:
                    logger.debug(
                        f"Request {req.rid} room={req.bootstrap_room} in Transferring state, "
                        f"waiting for completion or continuation request"
                    )
            elif poll == KVPoll.Success:  # transfer done
                req.finished_reason = FINISH_LENGTH(length=0)
                # dummy embedding for embedding models
                req.embedding = [0]
                # FIXME: clean up req's data in transfer engine
                if hasattr(req.disagg_embedding_sender, "clear"):
                    req.disagg_embedding_sender.clear()
                done_reqs.append(req)
                logger.debug(
                    f"Request {req.rid} room={req.bootstrap_room} transfer completed"
                )
            elif poll == KVPoll.Failed:
                error_message = f"MultimodalEmbedding transfer failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                try:
                    req.disagg_embedding_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.warning(error_message)
                self.tree_cache.cache_finished_req(req)
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                done_reqs.append(req)

        # keep generation response format here
        self.stream_output_generation(
            done_reqs, any(req.return_logprob for req in done_reqs), None
        )

        for req in done_reqs:
            if self.disagg_metadata_buffers.use_block_allocator:
                # Block-based: free by allocation
                allocation = getattr(req, 'metadata_allocation', None)
                if allocation is not None:
                    self.req_to_metadata_buffer_idx_allocator.free(
                        allocation,
                        fake=isinstance(req.disagg_embedding_sender, FakeKVSender),
                        req_id=req.rid
                    )
            else:
                # Index-based: legacy
                self.req_to_metadata_buffer_idx_allocator.free(
                    req.metadata_buffer_index,
                    fake=isinstance(req.disagg_embedding_sender, FakeKVSender),
                )
            req.metadata_buffer_index = -1
        self.disagg_embedding_inflight_queue = undone_reqs

        return done_reqs

    def process_embedding_chunk(self: Scheduler):
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.chunked_req:
                # keep same logic with prefill mode
                self.last_batch.filter_batch(chunked_req_to_exclude=self.chunked_req)
                self.tree_cache.cache_unfinished_req(self.chunked_req)
                # only send the last chunk
                # self.send_embedding_chunk(self.chunked_req, last_chunk=False)
                self.req_to_token_pool.free(self.chunked_req.req_pool_idx)
                self.running_batch.batch_is_full = False

    def send_embedding_chunk(
        self: Scheduler,
        req: Req,
        last_chunk: bool = False,
    ):
        """
        Send embedding chunk to Language side.
        For block-based allocation, may send in multiple batches if actual_length > default_buffer_tokens.
        """
        # Set buffer data
        if self.disagg_metadata_buffers.use_block_allocator:
            # Block-based: use allocation
            allocation = getattr(req, 'metadata_allocation', None)
            self.disagg_metadata_buffers.set_buf(req, allocation=allocation)
        else:
            # Index-based: legacy mode
            self.disagg_metadata_buffers.set_buf(req)
        
        # Get actual embedding length
        actual_length = req.embedding.shape[0]
        default_tokens = self.disagg_metadata_buffers.default_buffer_tokens
        
        # Determine sent_tokens from transfer_infos
        sent_tokens = 0
        if req.bootstrap_room in self.data_manager.transfer_infos:
            for transfer_info in self.data_manager.transfer_infos[req.bootstrap_room].values():
                sent_tokens = transfer_info.sent_tokens
                break
        
        # Calculate chunk_info
        if self.disagg_metadata_buffers.use_block_allocator:
            if sent_tokens == 0:
                # First transfer
                is_last = (actual_length <= default_tokens)
                chunk_info = self.disagg_metadata_buffers.get_buf_chunk_info(
                    allocation=allocation,
                    offset_tokens=0,
                    max_tokens=default_tokens  # Limit first transfer to default_tokens
                )
                logger.debug(
                    f"Sending first chunk for request {req.rid} room={req.bootstrap_room}: "
                    f"{min(actual_length, default_tokens)}/{actual_length} tokens, is_last={is_last}"
                )
            else:
                # Continuation transfer
                is_last = True  # Continuation should send all remaining data
                chunk_info = self.disagg_metadata_buffers.get_buf_chunk_info(
                    allocation=allocation,
                    offset_tokens=sent_tokens
                )
                remaining = actual_length - sent_tokens
                logger.debug(
                    f"Sending continuation for request {req.rid} room={req.bootstrap_room}: "
                    f"offset={sent_tokens}, remaining={remaining} tokens"
                )
        else:
            # Legacy index-based mode
            chunk_info = self.disagg_metadata_buffers.get_buf_chunk_info(req)
            is_last = True
        
        req.disagg_embedding_sender.send_embedding(
            req.metadata_buffer_index, is_last, chunk_info
        )

    def get_num_allocatable_reqs(self: Scheduler, running_bs: int):
        return get_global_server_args().max_micro_batch_size - running_bs
