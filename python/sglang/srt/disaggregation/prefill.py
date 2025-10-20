"""
Life cycle of a request in the prefill server

1. Bootstrap Queue
    a. Initialize a sender for each request
    b. Use the queue to store requests whose bootstrap (handshake and preallocation) has not finished
    c. Poll senders to check bootstrap state
    d. Once bootstrap is complete, move request to Waiting Queue

2. Waiting Queue
    a. Use PrefillAdder to pop requests
    b. Run forward
    c. Add the request to Inflight Queue

3. Inflight Queue
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional, Type

import torch

from sglang.srt.disaggregation.base import BaseKVManager, BaseKVReceiver, KVArgs, KVPoll
from sglang.srt.disaggregation.fake.conn import FakeKVReceiver
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    MultimodalDataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    is_mla_backend,
    kv_to_page_indices,
    kv_to_page_num,
    poll_and_all_reduce,
    prepare_abort,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
    RequestStage,
    ScheduleBatch,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.utils import (
    DynamicGradMode,
    broadcast_pyobj,
    point_to_point_pyobj,
    require_mlp_sync,
)

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


class PrefillBootstrapQueue:
    """
    Unified Bootstrap Queue.
    
    Supports two modes:
    - Send mode (PREFILL): Send KV cache to DECODE
    - Receive mode (LANGUAGE): Receive embedding from ENCODE
    """

    def __init__(
        self,
        token_to_kv_pool: KVCache,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        tp_rank: int,
        tp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        max_total_num_tokens: int,
        decode_tp_size: int,
        decode_dp_size: int,
        scheduler: Scheduler,
        pp_rank: int,
        pp_size: int,
        transfer_backend: TransferBackend,
        multimodal_data_buffers: Optional[MultimodalDataBuffers] = None,
        support_embedding_receive: bool = False,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.decode_tp_size = decode_tp_size
        self.decode_dp_size = decode_dp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.queue: List[Req] = []  # KV sender requests
        self.gloo_group = gloo_group
        self.max_total_num_tokens = max_total_num_tokens
        self.scheduler = scheduler
        self.transfer_backend = transfer_backend
        self.kv_manager = self._init_kv_manager()
        
        # Support for embedding receive mode
        self.support_embedding_receive = support_embedding_receive
        if support_embedding_receive:
            self.multimodal_data_buffers = multimodal_data_buffers
            self.embedding_receiver_manager = self._init_embedding_receiver_manager()
            self.embedding_requests: List[MultimodalLanguageRequest] = []

    def _init_kv_manager(self) -> BaseKVManager:
        """Initialize KV manager for sending KV cache (PREFILL mode)."""
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        kv_args.engine_rank = self.tp_rank
        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.dp_rank
        kv_args.decode_tp_size = self.decode_tp_size // self.decode_dp_size
        kv_args.prefill_pp_size = self.pp_size
        kv_args.prefill_start_layer = self.token_to_kv_pool.start_layer
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.token_to_kv_pool.get_contiguous_buf_infos()
        )

        if self.draft_token_to_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        if not self.is_mla_backend:
            kv_args.kv_head_num = self.token_to_kv_pool.head_num
        kv_args.page_size = self.token_to_kv_pool.page_size

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id

        kv_manager_class: Type[BaseKVManager] = get_kv_class(
            self.transfer_backend, KVClassType.MANAGER
        )
        kv_manager: BaseKVManager = kv_manager_class(
            kv_args,
            DisaggregationMode.PREFILL,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        return kv_manager
    
    def _init_embedding_receiver_manager(self) -> BaseKVManager:
        """Initialize embedding receiver manager (LANGUAGE mode)."""
        from sglang.srt.disaggregation.base import KVArgs
        
        kv_args = KVArgs()
        kv_args.engine_rank = self.tp_rank
        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.multimodal_data_buffers.get_buf_infos()
        )
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id
        
        manager_class = get_kv_class(
            self.transfer_backend, KVClassType.MANAGER, is_multimodal=True
        )
        return manager_class(
            kv_args,
            DisaggregationMode.LANGUAGE,
            self.scheduler.server_args,
        )

    def add(self, req: Req, num_kv_heads: int) -> None:
        """Add KV sender request (PREFILL mode)."""
        if self._check_if_req_exceed_kv_capacity(req):
            return

        if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
            kv_sender_class = get_kv_class(TransferBackend.FAKE, KVClassType.SENDER)
        else:
            kv_sender_class = get_kv_class(self.transfer_backend, KVClassType.SENDER)

        dest_tp_ranks = [self.tp_rank]

        req.disagg_kv_sender = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            dest_tp_ranks=dest_tp_ranks,
            pp_rank=self.pp_rank,
        )
        self._process_req(req)
        req.add_latency(RequestStage.PREFILL_PREPARE)
        self.queue.append(req)

    def extend(self, reqs: List[Req], num_kv_heads: int) -> None:
        """Add multiple KV sender requests."""
        for req in reqs:
            self.add(req, num_kv_heads)
    
    def add_embedding_receiver(self, req: Req) -> None:
        """Add embedding receiver request (LANGUAGE mode)."""
        if not self.support_embedding_receive:
            raise RuntimeError(
                "Embedding receive mode is not enabled. "
                "Set support_embedding_receive=True in __init__."
            )
        
        if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
            receiver_class = get_kv_class(
                TransferBackend.FAKE, KVClassType.RECEIVER, is_multimodal=True
            )
        else:
            receiver_class = get_kv_class(
                self.transfer_backend, KVClassType.RECEIVER, is_multimodal=True
            )
        
        embedding_receiver = receiver_class(
            mgr=self.embedding_receiver_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            prefill_dp_rank=req.data_parallel_rank,
        )
        
        self.embedding_requests.append(
            MultimodalLanguageRequest(req=req, embedding_receiver=embedding_receiver)
        )
    
    def extend_embedding_receivers(self, reqs: List[Req]) -> None:
        """Add multiple embedding receiver requests."""
        for req in reqs:
            self.add_embedding_receiver(req)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.stream_output([req], req.return_logprob)
            return True
        return False

    def _process_req(self, req: Req) -> None:
        """
        Set max_new_tokens = 1, so PrefillAdder memory estimation is accurate
        """
        req.sampling_params.max_new_tokens = 1

    def _update_embedding_handshake_waiters(self) -> None:
        """Poll embedding receivers for handshake completion."""
        if not self.embedding_requests:
            return
        
        if all(req.waiting_for_input for req in self.embedding_requests):
            return
        
        polls = poll_and_all_reduce(
            [req.embedding_receiver for req in self.embedding_requests],
            self.gloo_group,
        )
        
        for language_req, poll in zip(self.embedding_requests, polls):
            if poll == KVPoll.Bootstrapping:
                pass
            elif poll == KVPoll.WaitingForInput:
                language_req.waiting_for_input = True
            elif poll == KVPoll.Failed:
                error_message = (
                    f"MultimodalLanguage handshake failed for request "
                    f"rank={self.tp_rank} {language_req.req.rid=} "
                    f"{language_req.req.bootstrap_room=}"
                )
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
    
    def _pop_embedding_bootstrapped(self) -> List[Req]:
        """Pop embedding receiver requests that have finished bootstrapping."""
        if not self.support_embedding_receive:
            return []
        
        self._update_embedding_handshake_waiters()
        
        bootstrapped_reqs = []
        indices_to_remove = set()
        
        # Remove aborted requests
        for i, language_req in enumerate(self.embedding_requests):
            if isinstance(language_req.req.finished_reason, FINISH_ABORT):
                self.scheduler.stream_output(
                    [language_req.req], language_req.req.return_logprob
                )
                indices_to_remove.add(i)
        
        # Process bootstrapped requests
        for i, language_req in enumerate(self.embedding_requests):
            if i in indices_to_remove:
                continue
            
            if not language_req.waiting_for_input:
                continue
            
            if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
                break
            
            from sglang.srt.disaggregation.fake.conn import FakeKVReceiver
            
            language_req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc(
                    fake=isinstance(language_req.embedding_receiver, FakeKVReceiver)
                )
            )
            
            assert language_req.metadata_buffer_index is not None
            
            language_req.embedding_receiver.init(
                embedding_index=language_req.metadata_buffer_index
            )
            
            # Store metadata_buffer_index in req for later use
            language_req.req.metadata_buffer_index = language_req.metadata_buffer_index
            language_req.req.embedding_receiver = language_req.embedding_receiver
            
            bootstrapped_reqs.append(language_req.req)
            indices_to_remove.add(i)
        
        self.embedding_requests = [
            entry for i, entry in enumerate(self.embedding_requests)
            if i not in indices_to_remove
        ]
        
        return bootstrapped_reqs
    
    def pop_bootstrapped(
        self,
        return_failed_reqs: bool = False,
        rids_to_check: Optional[List[str]] = None,
    ) -> List[Req]:
        """
        Pop requests that have finished bootstrapping (both KV and embedding).

        return_failed_reqs: For PP, on rank 0, also return the failed reqs to notify the next rank
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """
        # Pop KV sender requests
        kv_bootstrapped_reqs = []
        kv_failed_reqs = []
        indices_to_remove = set()

        if len(self.queue) == 0 and not self.support_embedding_receive:
            if return_failed_reqs is False:
                return []
            else:
                return [], []

        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.queue], self.gloo_group
        )

        for i, (req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None:
                # if req not in reqs_info_to_check, skip
                if req.rid not in rids_to_check:
                    continue
                # Either waiting for input or failed
                assert poll == KVPoll.WaitingForInput or poll == KVPoll.Failed

            if poll == KVPoll.Bootstrapping:
                continue
            elif poll == KVPoll.Failed:
                error_message = f"Prefill bootstrap failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                self.scheduler.stream_output([req], req.return_logprob)
                indices_to_remove.add(i)
                failed_reqs.append(req)
                continue

            # KV.WaitingForInput - init here
            num_kv_indices = len(req.origin_input_ids)
            if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
                break

            req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert req.metadata_buffer_index is not None

            num_pages = kv_to_page_num(num_kv_indices, self.token_to_kv_pool.page_size)
            req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)

            bootstrapped_reqs.append(req)
            indices_to_remove.add(i)
            req.time_stats.wait_queue_entry_time = time.perf_counter()
            req.add_latency(RequestStage.PREFILL_BOOTSTRAP)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]
        
        kv_bootstrapped_reqs = bootstrapped_reqs
        kv_failed_reqs = failed_reqs
        
        # Pop embedding receiver requests
        embedding_bootstrapped_reqs = self._pop_embedding_bootstrapped()
        
        # Combine results
        all_bootstrapped_reqs = kv_bootstrapped_reqs + embedding_bootstrapped_reqs

        if return_failed_reqs is False:
            return all_bootstrapped_reqs
        else:
            return all_bootstrapped_reqs, kv_failed_reqs


class SchedulerDisaggregationPrefillMixin:
    """
    Mixin for Scheduler to handle disaggregation prefill
    """

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self: Scheduler) -> None:
        """
        Unified scheduler loop for prefill worker in disaggregation mode.
        
        Supports:
        - KV sending mode (PREFILL): Send KV cache to DECODE
        - Embedding receiving mode (LANGUAGE): Receive embedding from ENCODE
        """

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            
            # Pop bootstrapped requests (both KV senders and embedding receivers)
            bootstrapped_reqs = self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            
            # Separate embedding receiver requests from KV sender requests
            for req in bootstrapped_reqs:
                if hasattr(req, 'embedding_receiver'):
                    # Embedding receiver request - add to embedding inflight queue
                    if not hasattr(self, 'disagg_embedding_inflight_queue'):
                        self.disagg_embedding_inflight_queue = []
                    self.disagg_embedding_inflight_queue.append(req)
                else:
                    # KV sender request - add to waiting queue for prefill
                    self.waiting_queue.append(req)
            
            self.process_prefill_chunk()
            batch = self.get_new_batch_prefill()

            if require_mlp_sync(self.server_args):
                batch = self.prepare_mlp_sync_batch(batch)
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result_disagg_prefill(batch, result)

            # Process both KV and embedding inflight queues
            if len(self.disagg_prefill_inflight_queue) > 0 or (
                hasattr(self, 'disagg_embedding_inflight_queue') and 
                len(self.disagg_embedding_inflight_queue) > 0
            ):
                done_reqs = self.process_disagg_prefill_inflight_queue()
                # Add embedding receiver requests that finished to waiting queue
                for req in done_reqs:
                    if hasattr(req, 'embedding_receiver'):
                        self.waiting_queue.append(req)

            if batch is None and len(self.disagg_prefill_inflight_queue) == 0 and (
                not hasattr(self, 'disagg_embedding_inflight_queue') or 
                len(self.disagg_embedding_inflight_queue) == 0
            ):
                self.self_check_during_idle()

            self.last_batch = batch
            # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
            # Otherwise, it hangs under high concurrency
            self.running_batch.batch_is_full = False

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler) -> None:
        """
        Overlap scheduler loop for prefill worker in disaggregation mode.
        
        Supports:
        - KV sending mode (PREFILL): Send KV cache to DECODE
        - Embedding receiving mode (LANGUAGE): Receive embedding from ENCODE
        """
        self.result_queue = deque()

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            
            # Pop bootstrapped requests (both KV senders and embedding receivers)
            bootstrapped_reqs = self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            
            # Separate embedding receiver requests from KV sender requests
            for req in bootstrapped_reqs:
                if hasattr(req, 'embedding_receiver'):
                    # Embedding receiver request - add to embedding inflight queue
                    if not hasattr(self, 'disagg_embedding_inflight_queue'):
                        self.disagg_embedding_inflight_queue = []
                    self.disagg_embedding_inflight_queue.append(req)
                else:
                    # KV sender request - add to waiting queue for prefill
                    self.waiting_queue.append(req)
            
            self.process_prefill_chunk()
            batch = self.get_new_batch_prefill()

            if require_mlp_sync(self.server_args):
                batch = self.prepare_mlp_sync_batch(batch)
            self.cur_batch = batch

            batch_result = None
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))

            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result_disagg_prefill(tmp_batch, tmp_result)

            # Process both KV and embedding inflight queues
            if len(self.disagg_prefill_inflight_queue) > 0 or (
                hasattr(self, 'disagg_embedding_inflight_queue') and 
                len(self.disagg_embedding_inflight_queue) > 0
            ):
                done_reqs = self.process_disagg_prefill_inflight_queue()
                # Add embedding receiver requests that finished to waiting queue
                for req in done_reqs:
                    if hasattr(req, 'embedding_receiver'):
                        self.waiting_queue.append(req)

            self.launch_batch_sample_if_needed(batch_result)

            if batch is None and len(self.disagg_prefill_inflight_queue) == 0 and (
                not hasattr(self, 'disagg_embedding_inflight_queue') or 
                len(self.disagg_embedding_inflight_queue) == 0
            ):
                self.self_check_during_idle()

            self.last_batch = batch
            # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
            # Otherwise, it hangs under high concurrency
            self.running_batch.batch_is_full = False

    def process_batch_result_disagg_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        """
        Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        Adapted from process_batch_result_prefill
        """
        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
            copy_done,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
            result.copy_done,
        )

        if copy_done is not None:
            copy_done.synchronize()

        logprob_pt = 0
        # Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        next_token_ids = result.next_token_ids.tolist()
        if batch.return_logprob:
            if logits_output.next_token_logprobs is not None:
                logits_output.next_token_logprobs = (
                    logits_output.next_token_logprobs.tolist()
                )
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = tuple(
                    logits_output.input_token_logprobs.tolist()
                )

        hidden_state_offset = 0
        for i, (req, next_token_id) in enumerate(
            zip(batch.reqs, next_token_ids, strict=True)
        ):
            if req.is_chunked <= 0:
                # There is no output_ids for prefill
                req.output_ids.append(next_token_id)
                self.tree_cache.cache_unfinished_req(req)  # update the tree and lock
                req.add_latency(RequestStage.PREFILL_FORWARD)
                self.disagg_prefill_inflight_queue.append(req)
                if (
                    logits_output is not None
                    and logits_output.hidden_states is not None
                ):
                    last_hidden_index = (
                        hidden_state_offset + extend_input_len_per_req[i] - 1
                    )
                    req.output_topk_p = batch.spec_info.topk_p[i]
                    req.output_topk_index = batch.spec_info.topk_index[i]
                    if self.spec_algorithm.is_eagle3():
                        req.hidden_states_tensor = (
                            batch.spec_info.hidden_states[i].cpu().clone()
                        )
                    else:
                        req.hidden_states_tensor = (
                            logits_output.hidden_states[last_hidden_index].cpu().clone()
                        )
                    hidden_state_offset += extend_input_len_per_req[i]
                else:
                    req.hidden_states_tensor = None
                if req.return_logprob:
                    assert extend_logprob_start_len_per_req is not None
                    assert extend_input_len_per_req is not None
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    num_input_logprobs = extend_input_len - extend_logprob_start_len
                    self.add_logprob_return_values(
                        i,
                        req,
                        logprob_pt,
                        next_token_ids,
                        num_input_logprobs,
                        logits_output,
                    )
                    logprob_pt += num_input_logprobs
                self.send_kv_chunk(req, last_chunk=True)
                req.time_stats.prefill_transfer_queue_entry_time = time.perf_counter()

                if req.grammar is not None:
                    # FIXME: this try-except block is for handling unexpected xgrammar issue.
                    try:
                        req.grammar.accept_token(next_token_id)
                    except ValueError as e:
                        # Grammar accept_token can raise ValueError if the token is not in the grammar.
                        # This can happen if the grammar is not set correctly or the token is invalid.
                        error_message = f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                        self.tree_cache.cache_finished_req(req)
                        prepare_abort(
                            req,
                            error_message,
                            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        )
                    req.grammar.finished = req.finished()
            else:
                # being chunked reqs' prefill is not finished
                req.is_chunked -= 1

                if req.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        # Update input logprobs.
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.add_input_logprob_return_values(
                            i,
                            req,
                            logits_output,
                            logprob_pt,
                            num_input_logprobs,
                            last_prefill_chunk=False,
                        )
                        logprob_pt += num_input_logprobs

                if self.enable_overlap:
                    self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)

        self.maybe_send_health_check_signal()

    def _process_embedding_inflight_queue(
        self: Scheduler,
    ) -> List[Req]:
        """
        Poll embedding receivers in the inflight queue.
        Returns requests that have finished receiving embedding data.
        """
        if not hasattr(self, 'disagg_embedding_inflight_queue'):
            return []
        
        if len(self.disagg_embedding_inflight_queue) == 0:
            return []
        
        done_reqs = []
        
        polls = poll_and_all_reduce(
            [req.embedding_receiver for req in self.disagg_embedding_inflight_queue],
            self.attn_tp_cpu_group,
        )
        
        undone_reqs: List[Req] = []
        
        for req, poll in zip(self.disagg_embedding_inflight_queue, polls):
            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring, KVPoll.Bootstrapping]:
                undone_reqs.append(req)
            elif poll == KVPoll.Success:
                # Handle embedding transfer success
                from sglang.srt.disaggregation.fake.conn import FakeKVReceiver
                
                idx = req.metadata_buffer_index
                if not isinstance(req.embedding_receiver, FakeKVReceiver):
                    embedding_data, fill_ids, mrope_positions, aux_datas = (
                        self.disagg_multimodal_data_buffers.get_buf(idx)
                    )
                    embedding_length = aux_datas[0]
                    mrope_position_delta = aux_datas[1]
                    req.input_embeds = embedding_data[:embedding_length, :]
                    mrope_positions = mrope_positions[: 3 * embedding_length].reshape(
                        3, embedding_length
                    )
                    ori_input_length = len(req.origin_input_ids)
                    req.origin_input_ids = fill_ids[:embedding_length].tolist()
                    
                    mm_inputs = None
                    if ori_input_length == embedding_length:
                        mm_inputs = None
                    elif ori_input_length < embedding_length:
                        # Mock mm_inputs for multimodal requests
                        from sglang.srt.managers.schedule_batch import (
                            Modality,
                            MultimodalDataItem,
                            MultimodalInputs,
                        )
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
                        # Transfer failed case
                        error_message = (
                            f"Embedding transfer failed: ori_input_length > embedding_length "
                            f"for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                        )
                        logger.error(error_message)
                        prepare_abort(
                            req,
                            error_message,
                            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        )
                        self.stream_output([req], req.return_logprob)
                        continue
                    
                    req.multimodal_inputs = mm_inputs
                else:
                    # Fake receiver - just free the buffer
                    self.req_to_metadata_buffer_idx_allocator.free(idx, fake=True)
                
                done_reqs.append(req)
                
            elif poll == KVPoll.Failed:
                error_message = (
                    f"Embedding transfer failed for request rank={self.tp_rank} "
                    f"{req.rid=} {req.bootstrap_room=}"
                )
                try:
                    req.embedding_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                self.stream_output([req], req.return_logprob)
                if hasattr(req, 'metadata_buffer_index') and req.metadata_buffer_index >= 0:
                    from sglang.srt.disaggregation.fake.conn import FakeKVReceiver
                    self.req_to_metadata_buffer_idx_allocator.free(
                        req.metadata_buffer_index,
                        fake=isinstance(req.embedding_receiver, FakeKVReceiver),
                    )
            else:
                assert False, f"Unexpected polling state {poll=}"
        
        self.disagg_embedding_inflight_queue = undone_reqs
        return done_reqs
    
    def process_disagg_prefill_inflight_queue(
        self: Scheduler, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        """
        Poll the requests in the middle of transfer. If done, return the request.
        Handles both KV senders and embedding receivers.
        
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """
        # Process KV sender requests
        if len(self.disagg_prefill_inflight_queue) == 0:
            kv_done_reqs = []
        else:
            kv_done_reqs = self._process_kv_inflight_queue(rids_to_check)
        
        # Process embedding receiver requests
        embedding_done_reqs = self._process_embedding_inflight_queue()
        
        return kv_done_reqs + embedding_done_reqs
    
    def _process_kv_inflight_queue(
        self: Scheduler, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        """
        Poll KV senders in the inflight queue.
        Returns requests that have finished transferring KV cache.
        """
        done_reqs = []

        # This is the existing KV sender processing logic
        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_tp_cpu_group,
        )

        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):

            if rids_to_check is not None:
                if req.rid not in rids_to_check:
                    undone_reqs.append(req)
                    continue

                assert poll == KVPoll.Success or poll == KVPoll.Failed

            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                self.tree_cache.cache_finished_req(req)  # unlock the tree
                req.finished_reason = FINISH_LENGTH(length=0)
                # FIXME: clean up req's data in transfer engine
                if hasattr(req.disagg_kv_sender, "clear"):
                    req.disagg_kv_sender.clear()
                done_reqs.append(req)
            elif poll == KVPoll.Failed:
                error_message = f"Prefill transfer failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.warning(error_message)
                self.tree_cache.cache_finished_req(req)  # unlock the tree
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                done_reqs.append(req)
            else:
                assert False, f"Unexpected polling state {poll=}"

        for req in done_reqs:
            req.time_stats.completion_time = time.perf_counter()

        # Stream requests which have finished transfer
        self.stream_output(
            done_reqs,
            any(req.return_logprob for req in done_reqs),
            None,
        )
        for req in done_reqs:
            req: Req
            req.add_latency(RequestStage.PREFILL_TRANSFER_KV_CACHE)
            self.req_to_metadata_buffer_idx_allocator.free(req.metadata_buffer_index)
            req.metadata_buffer_index = -1

        self.disagg_prefill_inflight_queue = undone_reqs

        return done_reqs

    def get_transferred_rids(self: Scheduler) -> List[str]:
        """
        Used by PP, get the transferred rids but **do not pop**
        """
        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.tp_worker.get_tp_group().cpu_group,
        )

        transferred_rids: List[str] = []

        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if poll == KVPoll.Success or poll == KVPoll.Failed:
                transferred_rids.append(req.rid)

        return transferred_rids

    def process_prefill_chunk(self: Scheduler) -> None:
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.chunked_req:
                # Move the chunked request out of the batch so that we can merge
                # only finished requests to running_batch.
                self.last_batch.filter_batch(chunked_req_to_exclude=self.chunked_req)
                self.tree_cache.cache_unfinished_req(self.chunked_req, chunked=True)
                if self.enable_overlap:
                    # Delay KV transfer to process_batch_result_disagg_prefill when overlap is enabled to ensure results are resolved
                    self.chunked_req.tmp_end_idx = min(
                        len(self.chunked_req.fill_ids),
                        len(self.chunked_req.origin_input_ids),
                    )
                else:
                    self.send_kv_chunk(self.chunked_req)
                # chunked request keeps its rid but will get a new req_pool_idx
                self.req_to_token_pool.free(self.chunked_req.req_pool_idx)
                self.running_batch.batch_is_full = False

    def send_kv_chunk(
        self: Scheduler,
        req: Req,
        last_chunk: bool = False,
        end_idx: Optional[int] = None,
    ) -> None:
        """
        Send a prefilled chunk to the decode server
        """
        page_size = self.token_to_kv_pool_allocator.page_size
        start_idx = req.start_send_idx
        end_idx = (
            end_idx
            if end_idx is not None
            else min(len(req.fill_ids), len(req.origin_input_ids))
        )

        if not last_chunk:
            # if not the last chunk and the last page is partial, delay the last partial page to the next send
            end_idx = end_idx - end_idx % page_size

        kv_indices = (
            self.req_to_token_pool.req_to_token[req.req_pool_idx, start_idx:end_idx]
            .cpu()
            .numpy()
        )
        req.start_send_idx = end_idx
        if last_chunk:
            self.disagg_metadata_buffers.set_buf(req)
        page_indices = kv_to_page_indices(kv_indices, page_size)
        if len(page_indices) == 0:
            logger.info(
                f"Skip sending kv chunk for request {req.rid=} {req.bootstrap_room=} because page_indices is empty"
            )
            return
        req.disagg_kv_sender.send(page_indices)

    # PP
    @DynamicGradMode()
    def event_loop_pp_disagg_prefill(self: Scheduler):
        """
        An event loop for the prefill server in pipeline parallelism.

        Rules:
        1. Each stage runs in the same order and is notified by the previous stage.
        2. Each send/recv operation is blocking and matched by the neighboring stage.

        Regular Schedule:
        ====================================================================
        Stage i                   | Stage i+1
        send ith req              | recv ith req
        send ith proxy            | recv ith proxy
        send prev (i+1)th carry   | recv prev (i+1)th carry
        ====================================================================

        Prefill Server Schedule:
        ====================================================================
        Stage i                        | Stage i+1
        send ith req                   | recv ith req
        send ith bootstrap req         | recv ith bootstrap req
        send ith transferred req       | recv ith transferred req
        send ith proxy                 | recv ith proxy
        send prev (i+1)th carry        | recv prev (i+1)th carry
        send prev (i+1)th release req  | recv prev (i+1)th release req
        ====================================================================

        There are two additional elements compared to the regular schedule:

        1. Bootstrap Requests:
            a. Instead of polling the status on the current workers, we should wait for the previous stage to notify to avoid desynchronization.
            b. The first stage polls the status and propagates the bootstrapped requests down to all other stages.
            c. If the first stage polls successfully, by nature, other ranks are also successful because they performed a handshake together.

        2. Transferred Requests + Release Requests:
            a. The first stage polls the transfer finished requests, performs an intersection with the next stage's finished requests, and propagates down to the last stage.
            b. The last stage receives the requests that have finished transfer on all stages (consensus), then sends them to the first stage to release the memory.
            c. The first stage receives the release requests, releases the memory, and then propagates the release requests down to the last stage.
        """
        from sglang.srt.managers.scheduler import GenerationBatchResult

        mbs = [None] * self.pp_size
        last_mbs = [None] * self.pp_size
        self.running_mbs = [
            ScheduleBatch(reqs=[], batch_is_full=False) for _ in range(self.pp_size)
        ]
        pp_outputs: Optional[PPProxyTensors] = None

        # Either success or failed
        bootstrapped_rids: List[str] = []
        transferred_rids: List[str] = []
        release_rids: Optional[List[str]] = None

        # transferred microbatch
        tmbs = [None] * self.pp_size

        ENABLE_RELEASE = True  # For debug

        while True:
            server_is_idle = True

            for mb_id in range(self.pp_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = last_mbs[mb_id]

                recv_reqs = self.recv_requests()

                self.process_input_requests(recv_reqs)

                if self.pp_group.is_first_rank:
                    # First rank, pop the bootstrap reqs from the bootstrap queue
                    bootstrapped_reqs, failed_reqs = (
                        self.disagg_prefill_bootstrap_queue.pop_bootstrapped(
                            return_failed_reqs=True
                        )
                    )
                    bootstrapped_rids = [req.rid for req in bootstrapped_reqs] + [
                        req.rid for req in failed_reqs
                    ]
                    self.waiting_queue.extend(bootstrapped_reqs)
                else:
                    # Other ranks, receive the bootstrap reqs info from the previous rank and ensure the consensus
                    bootstrapped_rids = self.recv_pyobj_from_prev_stage()
                    bootstrapped_reqs = (
                        self.disagg_prefill_bootstrap_queue.pop_bootstrapped(
                            rids_to_check=bootstrapped_rids
                        )
                    )
                    self.waiting_queue.extend(bootstrapped_reqs)

                if self.pp_group.is_first_rank:
                    transferred_rids = self.get_transferred_rids()
                # if other ranks,
                else:
                    # 1. recv previous stage's transferred reqs info
                    prev_transferred_rids = self.recv_pyobj_from_prev_stage()
                    # 2. get the current stage's transferred reqs info
                    curr_transferred_rids = self.get_transferred_rids()
                    # 3. new consensus rids = intersection(previous consensus rids, transfer finished rids)
                    transferred_rids = list(
                        set(prev_transferred_rids) & set(curr_transferred_rids)
                    )

                tmbs[mb_id] = transferred_rids

                self.process_prefill_chunk()
                mbs[mb_id] = self.get_new_batch_prefill()
                self.running_mbs[mb_id] = self.running_batch

                self.cur_batch = mbs[mb_id]
                if self.cur_batch:
                    server_is_idle = False
                    result = self.run_batch(self.cur_batch)

                # send the outputs to the next step
                if self.pp_group.is_last_rank:
                    if self.cur_batch:
                        next_token_ids = result.next_token_ids
                        pp_outputs = PPProxyTensors(
                            {
                                "next_token_ids": next_token_ids,
                            }
                        )
                        # send the output from the last round to let the next stage worker run post processing
                        self.pp_group.send_tensor_dict(
                            pp_outputs.tensors,
                            all_gather_group=self.attn_tp_group,
                        )

                if ENABLE_RELEASE:
                    if self.pp_group.is_last_rank:
                        # At the last stage, all stages has reached the consensus to release memory for transferred_rids
                        release_rids = transferred_rids
                        # send to the first rank
                        self.send_pyobj_to_next_stage(release_rids)

                # receive outputs and post-process (filter finished reqs) the coming microbatch
                next_mb_id = (mb_id + 1) % self.pp_size
                next_pp_outputs = None
                next_release_rids = None

                if mbs[next_mb_id] is not None:
                    next_pp_outputs: Optional[PPProxyTensors] = PPProxyTensors(
                        self.pp_group.recv_tensor_dict(
                            all_gather_group=self.attn_tp_group
                        )
                    )
                    mbs[next_mb_id].output_ids = next_pp_outputs["next_token_ids"]
                    output_result = GenerationBatchResult(
                        logits_output=None,
                        pp_hidden_states_proxy_tensors=None,
                        next_token_ids=next_pp_outputs["next_token_ids"],
                        extend_input_len_per_req=None,
                        extend_logprob_start_len_per_req=None,
                        can_run_cuda_graph=result.can_run_cuda_graph,
                    )
                    self.process_batch_result_disagg_prefill(
                        mbs[next_mb_id], output_result
                    )

                    last_mbs[next_mb_id] = mbs[next_mb_id]

                if ENABLE_RELEASE:
                    if tmbs[next_mb_id] is not None:
                        # recv consensus rids from the previous rank
                        next_release_rids = self.recv_pyobj_from_prev_stage()
                        self.process_disagg_prefill_inflight_queue(next_release_rids)

                # carry the outputs to the next stage
                if not self.pp_group.is_last_rank:
                    if pp_outputs:
                        # send the outputs from the last round to let the next stage worker run post processing
                        self.pp_group.send_tensor_dict(
                            pp_outputs.tensors,
                            all_gather_group=self.attn_tp_group,
                        )
                    if ENABLE_RELEASE:
                        if release_rids is not None:
                            self.send_pyobj_to_next_stage(release_rids)

                if not self.pp_group.is_last_rank:
                    # send out reqs to the next stage
                    self.send_pyobj_to_next_stage(recv_reqs)
                    self.send_pyobj_to_next_stage(bootstrapped_rids)
                    self.send_pyobj_to_next_stage(transferred_rids)

                    # send out proxy tensors to the next stage
                    if self.cur_batch:
                        # FIXME(lsyin): remove this assert
                        assert result.pp_hidden_states_proxy_tensors.tensors is not None
                        self.pp_group.send_tensor_dict(
                            result.pp_hidden_states_proxy_tensors.tensors,
                            all_gather_group=self.attn_tp_group,
                        )

                pp_outputs = next_pp_outputs
                release_rids = next_release_rids

                self.running_batch.batch_is_full = False

            if not ENABLE_RELEASE:
                if len(self.disagg_prefill_inflight_queue) > 0:
                    self.process_disagg_prefill_inflight_queue()

            # When the server is idle, self-check and re-init some states
            if server_is_idle and len(self.disagg_prefill_inflight_queue) == 0:
                self.check_memory()
                self.check_tree_cache()
                self.new_token_ratio = self.init_new_token_ratio

    def send_pyobj_to_next_stage(self, data):
        if self.attn_tp_rank == 0:
            dp_offset = self.attn_dp_rank * self.attn_tp_size
            point_to_point_pyobj(
                data,
                self.pp_rank * self.tp_size + dp_offset,
                self.world_group.device_group,
                self.pp_rank * self.tp_size + dp_offset,
                ((self.pp_rank + 1) % self.pp_size) * self.tp_size + dp_offset,
            )

    def recv_pyobj_from_prev_stage(self):
        if self.attn_tp_rank == 0:
            dp_offset = self.attn_dp_rank * self.attn_tp_size
            data = point_to_point_pyobj(
                [],
                self.pp_rank * self.tp_size + dp_offset,
                self.world_group.device_group,
                ((self.pp_rank - 1) % self.pp_size) * self.tp_size + dp_offset,
                self.pp_rank * self.tp_size + dp_offset,
            )
        else:
            data = None

        if self.tp_size != 1:
            data = broadcast_pyobj(
                data, self.tp_group.rank, self.tp_cpu_group, src=self.tp_group.ranks[0]
            )
        return data


# ==================== Multimodal Language Classes ====================
# Note: These classes are now deprecated and integrated into PrefillBootstrapQueue.
# They are kept here for backward compatibility during migration.


@dataclass
class MultimodalLanguageRequest:
    """Data class for multimodal language requests."""
    req: Req
    embedding_receiver: BaseKVReceiver
    waiting_for_input: bool = False
    metadata_buffer_index: int = -1


class MultimodalLanguageBootstrapQueue:
    """
    DEPRECATED: Use PrefillBootstrapQueue with support_embedding_receive=True instead.
    
    This class is kept for backward compatibility. New code should use:
    PrefillBootstrapQueue(..., support_embedding_receive=True)
    """
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

    def pop_bootstrapped(self):
        """pop the reqs which has finished bootstrapping"""
        self._update_handshake_waiters()

        bootstrapped_reqs = []
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

            language_req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc(
                    fake=isinstance(language_req.embedding_receiver, FakeKVReceiver)
                )
            )

            assert language_req.metadata_buffer_index is not None

            language_req.embedding_receiver.init(
                embedding_index=language_req.metadata_buffer_index
            )
            bootstrapped_reqs.append(language_req)
            indices_to_remove.add(i)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return bootstrapped_reqs


class MultimodalLanguageInflightQueue:
    """
    DEPRECATED: Embedding inflight processing is now integrated into
    SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue().
    
    This class is kept for backward compatibility.
    """
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
            language_req.metadata_buffer_index,
            fake=isinstance(language_req.embedding_receiver, FakeKVReceiver),
        )
        if self.scheduler.enable_metrics:
            self.scheduler.metrics_collector.increment_transfer_failed_reqs()

    def pop_done(self):
        """Pop the requests which have finished transfer"""
        if not self.queue:
            return []

        polls = poll_and_all_reduce(
            [language_req.embedding_receiver for language_req in self.queue],
            self.gloo_group,
        )

        done_reqs = []
        indices_to_remove = set()
        for i, (language_req, poll) in enumerate(zip(self.queue, polls)):
            if poll == KVPoll.Failed:
                self._handle_failed_request(language_req)
                # unlock the kv cache or it will have memory leak
                indices_to_remove.add(i)
                continue
            elif poll == KVPoll.Success:
                idx = language_req.metadata_buffer_index
                if not isinstance(language_req.embedding_receiver, FakeKVReceiver):
                    embedding_data, fill_ids, mrope_positions, aux_datas = (
                        self.metadata_buffers.get_buf(idx)
                    )
                    embedding_length = aux_datas[0]
                    mrope_position_delta = aux_datas[1]
                    language_req.req.input_embeds = embedding_data[:embedding_length, :]
                    mrope_positions = mrope_positions[: 3 * embedding_length].reshape(
                        3, embedding_length
                    )
                    ori_input_length = len(language_req.req.origin_input_ids)
                    language_req.req.origin_input_ids = fill_ids[
                        :embedding_length
                    ].tolist()
                    mm_inputs = None
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
                    # NOTE: we need to set the metadata buffer index to the request
                    # because the metadata buffer index will be freed after the request is done
                    # to avoid embedding buffer is freed before the request is done
                    language_req.req.metadata_buffer_index = (
                        language_req.metadata_buffer_index
                    )
                else:
                    self.req_to_metadata_buffer_idx_allocator.free(idx, fake=True)

                done_reqs.append(language_req.req)
                indices_to_remove.add(i)
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
                KVPoll.Transferring,
            ]:
                pass
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        for i in indices_to_remove:
            idx = self.queue[i].metadata_buffer_index
            assert idx != -1

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]
        return done_reqs


class SchedulerDisaggregationMultiModalLanguageMixin:
    """
    DEPRECATED: Multimodal language support is now integrated into
    SchedulerDisaggregationPrefillMixin.
    
    Use SchedulerDisaggregationPrefillMixin.event_loop_normal_disagg_prefill() instead,
    with PrefillBootstrapQueue configured with support_embedding_receive=True.
    
    This class is kept for backward compatibility during migration.
    """

    @torch.no_grad()
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

    @torch.no_grad()
    def event_loop_overlap_disagg_multimodal_language(self: Scheduler):
        self.result_queue = deque()

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.process_multimodal_language_queue()
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                batch.launch_done = threading.Event()
                result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), result))

                if self.last_batch is None:
                    # Create a dummy first batch to start the pipeline for overlap schedule.
                    # It is now used for triggering the sampling_info_done event.
                    tmp_batch = ScheduleBatch(
                        reqs=None,
                        forward_mode=ForwardMode.DUMMY_FIRST,
                        next_batch_sampling_info=self.tp_worker.cur_sampling_info,
                    )
                    self.process_batch_result(tmp_batch, None, batch.launch_done)

            if self.last_batch:
                # Process the results of the last batch
                tmp_batch, tmp_result = self.result_queue.popleft()
                tmp_batch.next_batch_sampling_info = (
                    self.tp_worker.cur_sampling_info if batch else None
                )
                self.process_batch_result(
                    tmp_batch, tmp_result, batch.launch_done if batch else None
                )
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch

    def process_multimodal_language_queue(self: Scheduler):
        """Process multimodal language queues: bootstrap -> inflight -> waiting"""
        bootstrapped_reqs = self.disagg_language_bootstrap_queue.pop_bootstrapped()
        self.disagg_language_inflight_queue.extend(bootstrapped_reqs)
        done_reqs = self.disagg_language_inflight_queue.pop_done()
        self.waiting_queue.extend(done_reqs)
