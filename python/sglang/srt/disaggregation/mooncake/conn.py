from __future__ import annotations

import asyncio
import concurrent.futures
import ctypes
import dataclasses
import logging
import os
import queue
import socket
import struct
import threading
import time
from collections import defaultdict
from functools import cache
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
import requests
import zmq
from aiohttp import web

from sglang.srt.disaggregation.base.conn import (
    BaseKVBootstrapServer,
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
)
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    group_concurrent_contiguous,
)
from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    format_tcp_address,
    get_bool_env_var,
    get_free_port,
    get_int_env_var,
    get_ip,
    get_local_ip_by_remote,
    is_valid_ipv6_address,
)

logger = logging.getLogger(__name__)


class KVTransferError(Exception):
    def __init__(self, bootstrap_room: int, failure_reason: str):
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason

    def __str__(self):
        return f"KVTransferError(bootstrap_room={self.bootstrap_room}): {self.failure_reason}"


# prefill
@dataclasses.dataclass
class TransferKVChunk:
    room: int
    prefill_kv_indices: npt.NDArray[np.int32]
    index_slice: slice
    is_last: bool
    prefill_aux_index: Optional[int]


# decode
@dataclasses.dataclass
class TransferInfo:
    room: int
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    required_dst_info_num: int
    is_dummy: bool

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        if msg[4] == b"" and msg[5] == b"":
            is_dummy = True
            dst_kv_indices = np.array([], dtype=np.int32)
            dst_aux_index = None
        else:
            dst_kv_indices = np.frombuffer(msg[4], dtype=np.int32)
            dst_aux_index = int(msg[5].decode("ascii"))
            is_dummy = False
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            required_dst_info_num=int(msg[6].decode("ascii")),
            is_dummy=is_dummy,
        )


# decode
@dataclasses.dataclass
class KVArgsRegisterInfo:
    room: str
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_ptrs: list[int]
    dst_aux_ptrs: list[int]
    dst_tp_rank: int
    dst_attn_tp_size: int
    dst_kv_item_len: int

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[5])//8}Q", msg[5])),
            dst_tp_rank=int(msg[6].decode("ascii")),
            dst_attn_tp_size=int(msg[7].decode("ascii")),
            dst_kv_item_len=int(msg[8].decode("ascii")),
        )


class AuxDataCodec:
    """Handles serialization and deserialization of auxiliary data buffers"""

    @staticmethod
    def serialize_data_from_buffer(src_addr, data_length):
        """Serialize data from memory buffer to bytes"""
        buffer = (ctypes.c_byte * data_length).from_address(src_addr)
        return bytes(buffer)

    @staticmethod
    def deserialize_data_to_buffer(kv_args, buffer_index, aux_index, data):
        """Deserialize bytes into target memory buffer"""
        dst_aux_ptr = kv_args.aux_data_ptrs[buffer_index]
        item_len = kv_args.aux_item_lens[buffer_index]
        dst_addr = dst_aux_ptr + item_len * aux_index
        buffer = (ctypes.c_byte * len(data)).from_address(dst_addr)
        buffer[:] = data
        return


class MooncakeKVManager(CommonKVManager):
    AUX_DATA_HEADER = b"AUX_DATA"

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.init_engine()
        self.register_buffer_to_engine()
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.start_prefill_thread()
            self.session_failures = defaultdict(int)
            self.failed_sessions = set()
            self.session_lock = threading.Lock()
            # Determine the number of threads to use for kv sender
            cpu_count = os.cpu_count()
            transfer_thread_pool_size = get_int_env_var(
                "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE",
                min(max(4, int(0.75 * cpu_count) // 8), 12),
            )
            transfer_queue_size = get_int_env_var("SGLANG_DISAGGREGATION_QUEUE_SIZE", 4)
            self.transfer_queues: List[FastQueue] = [
                FastQueue() for _ in range(transfer_queue_size)
            ]
            assert transfer_thread_pool_size >= transfer_queue_size, (
                f"The environment variable SGLANG_DISAGGREGATION_THREAD_POOL_SIZE={transfer_thread_pool_size} must be "
                f"greater than or equal to SGLANG_DISAGGREGATION_QUEUE_SIZE={transfer_queue_size}."
            )
            self.executors = [
                concurrent.futures.ThreadPoolExecutor(
                    transfer_thread_pool_size // transfer_queue_size
                )
                for _ in range(transfer_queue_size)
            ]
            for queue, executor in zip(self.transfer_queues, self.executors):
                threading.Thread(
                    target=self.transfer_worker, args=(queue, executor), daemon=True
                ).start()
            # If a timeout happens on the prefill side, it means prefill instances
            # fail to receive the KV indices from the decode instance of this request.
            # These timeout requests should be aborted to release the tree cache.
            self.bootstrap_timeout = get_int_env_var(
                "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT", 300
            )

            self.enable_custom_mem_pool = get_bool_env_var(
                "SGLANG_MOONCAKE_CUSTOM_MEM_POOL", "false"
            )
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.heartbeat_failures = {}
            self.session_pool = defaultdict(requests.Session)
            self.session_pool_lock = threading.Lock()
            self.addr_to_rooms_tracker = defaultdict(set)
            self.prefill_response_tracker: Dict[int, Set[int]] = defaultdict(set)
            # Heartbeat interval should be at least 2 seconds
            self.heartbeat_interval = max(
                float(os.getenv("SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL", 5.0)), 2.0
            )
            # Heartbeat failure should be at least 1
            self.max_failures = max(
                get_int_env_var("SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE", 2), 1
            )
            self.start_decode_thread()
            # If a timeout happens on the decode side, it means decode instances
            # fail to receive the KV Cache transfer done signal after bootstrapping.
            # These timeout requests should be aborted to release the tree cache.
            self.waiting_timeout = get_int_env_var(
                "SGLANG_DISAGGREGATION_WAITING_TIMEOUT", 300
            )

        self.failure_records: Dict[int, str] = {}
        self.failure_lock = threading.Lock()

    def init_engine(self):
        self.engine = MooncakeTransferEngine(
            hostname=self.local_ip,
            gpu_id=self.kv_args.gpu_id,
            ib_device=self.kv_args.ib_device,
        )

    def register_buffer_to_engine(self):
        # Batch register KV data buffers
        if self.kv_args.kv_data_ptrs and self.kv_args.kv_data_lens:
            self.engine.batch_register(
                self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
            )

        # Batch register auxiliary data buffers
        if self.kv_args.aux_data_ptrs and self.kv_args.aux_data_lens:
            self.engine.batch_register(
                self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
            )

    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        if not transfer_blocks:
            return 0

        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
        return self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        layers_params = None

        # pp is not supported on the decode side yet
        if self.is_mla_backend:
            src_kv_ptrs, dst_kv_ptrs, layers_current_pp_stage = (
                self.get_mla_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
            )
            kv_item_len = self.kv_args.kv_item_lens[0]
            layers_params = [
                (
                    src_kv_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    kv_item_len,
                )
                for layer_id in range(layers_current_pp_stage)
            ]
        else:
            src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
                self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
            )
            kv_item_len = self.kv_args.kv_item_lens[0]
            layers_params = [
                (
                    src_k_ptrs[layer_id],
                    dst_k_ptrs[layer_id],
                    kv_item_len,
                )
                for layer_id in range(layers_current_pp_stage)
            ] + [
                (
                    src_v_ptrs[layer_id],
                    dst_v_ptrs[layer_id],
                    kv_item_len,
                )
                for layer_id in range(layers_current_pp_stage)
            ]
        assert layers_params is not None

        def set_transfer_blocks(
            src_ptr: int, dst_ptr: int, item_len: int
        ) -> List[Tuple[int, int, int]]:
            transfer_blocks = []
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))
            return transfer_blocks

        # Worker function for processing a single layer
        def process_layer(src_ptr: int, dst_ptr: int, item_len: int) -> int:
            transfer_blocks = set_transfer_blocks(src_ptr, dst_ptr, item_len)
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        # Worker function for processing all layers in a batch
        def process_layers(layers_params: List[Tuple[int, int, int]]) -> int:
            transfer_blocks = []
            for src_ptr, dst_ptr, item_len in layers_params:
                transfer_blocks.extend(set_transfer_blocks(src_ptr, dst_ptr, item_len))
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        if self.enable_custom_mem_pool:
            futures = [
                executor.submit(
                    process_layer,
                    src_ptr,
                    dst_ptr,
                    item_len,
                )
                for (src_ptr, dst_ptr, item_len) in layers_params
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    for f in futures:
                        f.cancel()
                    return status
        else:
            # Combining all layers' params in one batch transfer is more efficient
            # compared to using multiple threads
            return process_layers(layers_params)

        return 0

    def send_kvcache_slice(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int64],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int64],
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """
        Sends KV cache slices from this Prefill rank to a target Decode rank,
        supporting generic M-to-N TP size configurations.

        NOTE: This implementation calls the transfer engine for each token slot within
        each page to ensure correctness for any page_size and head-slicing configuration.
        This may introduce performance overhead (increased TTFT) for long sequences.
        """
        # Extract configuration
        local_tp_rank_in_group = self.kv_args.engine_rank % self.attn_tp_size
        src_kv_item_len = self.kv_args.kv_item_lens[0]
        dst_tp_rank_in_group = dst_tp_rank % dst_attn_tp_size
        num_kv_heads = self.kv_args.kv_head_num
        num_layers = len(self.kv_args.kv_data_ptrs)
        page_size = self.kv_args.page_size

        # Calculate head distribution
        src_heads_per_rank = num_kv_heads
        dst_heads_per_rank = num_kv_heads * self.attn_tp_size // dst_attn_tp_size
        bytes_per_head_slice_to_send = (
            dst_kv_item_len // page_size // dst_heads_per_rank
        )

        # Determine slicing parameters based on TP configuration
        if self.attn_tp_size > dst_attn_tp_size:
            # Send KVCache from multiple prefill instances to 1 decode instance
            src_head_start_offset = 0
            num_heads_to_send = src_heads_per_rank
            dst_head_start_offset = local_tp_rank_in_group * src_heads_per_rank
        else:
            # Send KVCache from 1 prefill instance to multiple decode instances
            src_head_start_offset = (
                dst_tp_rank_in_group * dst_heads_per_rank
            ) % src_heads_per_rank
            num_heads_to_send = dst_heads_per_rank
            dst_head_start_offset = 0

        src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
            self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
        )

        # Calculate precise byte offset and length for the sub-slice within the token
        src_head_slice_offset = src_head_start_offset * bytes_per_head_slice_to_send
        dst_head_slice_offset = dst_head_start_offset * bytes_per_head_slice_to_send
        heads_bytes_per_token_to_send = num_heads_to_send * bytes_per_head_slice_to_send

        # Sanity check: The data sub-slice to be sent should fit into the dst buffer.
        # This means heads_bytes_per_token_to_send <= (dst_kv_item_len // page_size)
        if heads_bytes_per_token_to_send > (dst_kv_item_len // page_size):
            logger.error(
                f"[{mooncake_session_id}] slice size ({heads_bytes_per_token_to_send}) exceeds "
                f"target token slot size ({dst_kv_item_len // page_size})"
            )
            return -1

        layers_params = [
            (
                src_k_ptrs[layer_id],
                dst_k_ptrs[layer_id],
                src_kv_item_len,
                dst_kv_item_len,
                src_head_slice_offset,
                dst_head_slice_offset,
                heads_bytes_per_token_to_send,
            )
            for layer_id in range(layers_current_pp_stage)
        ] + [
            (
                src_v_ptrs[layer_id],
                dst_v_ptrs[layer_id],
                src_kv_item_len,
                dst_kv_item_len,
                src_head_slice_offset,
                dst_head_slice_offset,
                heads_bytes_per_token_to_send,
            )
            for layer_id in range(layers_current_pp_stage)
        ]

        def process_layer_tp_aware(layer_params):
            (
                src_ptr,
                dst_ptr,
                src_item_len,
                dst_item_len,
                src_head_slice_offset,
                dst_head_slice_offset,
                heads_bytes_per_token_to_send,
            ) = layer_params
            src_addr_list = []
            dst_addr_list = []
            length_list = []

            # Calculate strides for a single token slot
            bytes_per_token_on_prefill = src_item_len // page_size
            bytes_per_token_on_decode = dst_item_len // page_size

            for i in range(len(prefill_kv_indices)):
                prefill_page_idx = int(prefill_kv_indices[i])
                decode_page_idx = int(dst_kv_indices[i])

                # Get the starting addresses for the current src and dst pages
                src_page_start_addr = src_ptr + prefill_page_idx * src_item_len
                dst_page_start_addr = dst_ptr + decode_page_idx * dst_item_len

                # Iterate through each valid token slot within the current page
                for token_slot_in_page in range(page_size):
                    # Calculate the start address of the current token slot
                    src_token_slot_start_addr = (
                        src_page_start_addr
                        + token_slot_in_page * bytes_per_token_on_prefill
                    )
                    dst_token_slot_start_addr = (
                        dst_page_start_addr
                        + token_slot_in_page * bytes_per_token_on_decode
                    )

                    # Calculate final src and dst addresses by applying head-slice offsets
                    src_slice_addr = src_token_slot_start_addr + src_head_slice_offset
                    dst_slice_addr = dst_token_slot_start_addr + dst_head_slice_offset

                    src_addr_list.append(src_slice_addr)
                    dst_addr_list.append(dst_slice_addr)
                    length_list.append(heads_bytes_per_token_to_send)

            return self.engine.batch_transfer_sync(
                mooncake_session_id, src_addr_list, dst_addr_list, length_list
            )

        futures = [
            executor.submit(
                process_layer_tp_aware,
                layer_params,
            )
            for layer_params in layers_params
        ]

        for future in concurrent.futures.as_completed(futures):
            status = future.result()
            if status != 0:
                for f in futures:
                    f.cancel()
                return status

        return 0

    def send_aux(
        self,
        req: TransferInfo,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
    ):
        # TODO(shangming): Fix me when nvlink_transport of Mooncake is bug-free
        if self.enable_custom_mem_pool:
            return self.send_aux_tcp(req, prefill_aux_index, dst_aux_ptrs)

        transfer_blocks = []
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i, dst_aux_ptr in enumerate(dst_aux_ptrs):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            dst_addr = dst_aux_ptrs[i] + length * req.dst_aux_index
            transfer_blocks.append((src_addr, dst_addr, length))

        return self._transfer_data(req.mooncake_session_id, transfer_blocks)

    def send_aux_tcp(
        self,
        req: TransferInfo,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
    ):
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i in range(len(prefill_aux_ptrs)):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            data = AuxDataCodec.serialize_data_from_buffer(src_addr, length)

            self.send_aux_data_to_endpoint(
                remote=req.endpoint,
                dst_port=req.dst_port,
                room=req.room,
                buffer_index=i,
                aux_index=req.dst_aux_index,
                data=data,
            )

        return 0

    def send_aux_data_to_endpoint(
        self,
        remote: str,
        dst_port: int,
        room: int,
        buffer_index: int,
        aux_index: int,
        data: bytes,
    ):
        socket = self._connect(
            format_tcp_address(remote, dst_port), is_ipv6=is_valid_ipv6_address(remote)
        )

        socket.send_multipart(
            [
                MooncakeKVManager.AUX_DATA_HEADER,
                str(room).encode("ascii"),
                str(buffer_index).encode("ascii"),
                str(aux_index).encode("ascii"),
                struct.pack(">I", len(data)),
                data,
            ]
        )

    def _handle_aux_data(self, msg: List[bytes]):
        """Handle AUX_DATA messages received by the decode thread."""
        room = int(msg[1].decode("ascii"))
        buffer_index = int(msg[2].decode("ascii"))
        aux_index = int(msg[3].decode("ascii"))
        data_length = struct.unpack(">I", msg[4])[0]
        data = msg[5]

        if len(data) != data_length:
            logger.error(f"AUX_DATA length mismatch for bootstrap_room {room}")
            return

        AuxDataCodec.deserialize_data_to_buffer(
            self.kv_args, buffer_index, aux_index, data
        )

        logger.debug(
            f"Received AUX_DATA for bootstrap_room {room} with length:{len(data)}"
        )

    def sync_status_to_decode_endpoint(
        self, remote: str, dst_port: int, room: int, status: int, prefill_rank: int
    ):
        self._connect(
            format_tcp_address(remote, dst_port), is_ipv6=is_valid_ipv6_address(remote)
        ).send_multipart(
            [
                str(room).encode("ascii"),
                str(status).encode("ascii"),
                str(prefill_rank).encode("ascii"),
            ]
        )

    def transfer_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):
        while True:
            try:
                kv_chunk: TransferKVChunk = queue.get()
                reqs_to_be_processed = (
                    self.transfer_infos[kv_chunk.room].values()
                    if kv_chunk.room in self.transfer_infos
                    else []
                )
                polls = []
                dst_ranks_infos = []
                local_rank = self.attn_tp_rank * self.pp_size + self.pp_rank
                for req in reqs_to_be_processed:
                    if not req.is_dummy:
                        # Early exit if the request has failed
                        with self.session_lock:
                            if req.mooncake_session_id in self.failed_sessions:
                                self.record_failure(
                                    kv_chunk.room,
                                    f"Decode instance could be dead, remote mooncake session {req.mooncake_session_id} is not alive",
                                )
                                self.update_status(kv_chunk.room, KVPoll.Failed)
                                self.sync_status_to_decode_endpoint(
                                    req.endpoint,
                                    req.dst_port,
                                    req.room,
                                    KVPoll.Failed,
                                    local_rank,
                                )
                                break

                        chunked_dst_kv_indice = req.dst_kv_indices[kv_chunk.index_slice]

                        # NOTE: This is temporarily a workaround to deal with the case where the prefill_kv_indices
                        # is mismatched with the dst_kv_indices when page size > 1, this should never happen.
                        if len(chunked_dst_kv_indice) < len(
                            kv_chunk.prefill_kv_indices
                        ):
                            logger.warning(
                                f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"
                            )
                            kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
                                : len(chunked_dst_kv_indice)
                            ]

                        target_rank_registration_info: KVArgsRegisterInfo = (
                            self.decode_kv_args_table[req.mooncake_session_id]
                        )
                        if self.is_mla_backend or (
                            self.attn_tp_size
                            == target_rank_registration_info.dst_attn_tp_size
                        ):
                            ret = self.send_kvcache(
                                req.mooncake_session_id,
                                kv_chunk.prefill_kv_indices,
                                target_rank_registration_info.dst_kv_ptrs,
                                chunked_dst_kv_indice,
                                executor,
                            )
                        else:
                            ret = self.send_kvcache_slice(
                                req.mooncake_session_id,
                                kv_chunk.prefill_kv_indices,
                                target_rank_registration_info.dst_kv_ptrs,
                                chunked_dst_kv_indice,
                                target_rank_registration_info.dst_tp_rank,
                                target_rank_registration_info.dst_attn_tp_size,
                                target_rank_registration_info.dst_kv_item_len,
                                executor,
                            )
                        if ret != 0:
                            with self.session_lock:
                                self.session_failures[req.mooncake_session_id] += 1
                                # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
                                if self.session_failures[req.mooncake_session_id] >= 1:
                                    self.failed_sessions.add(req.mooncake_session_id)
                                    logger.error(
                                        f"Session {req.mooncake_session_id} failed."
                                    )
                            self.record_failure(
                                kv_chunk.room,
                                f"Failed to send kv chunk of {kv_chunk.room} to {req.endpoint}:{req.dst_port}",
                            )
                            self.update_status(kv_chunk.room, KVPoll.Failed)
                            self.sync_status_to_decode_endpoint(
                                req.endpoint,
                                req.dst_port,
                                req.room,
                                KVPoll.Failed,
                                local_rank,
                            )
                            break

                        if kv_chunk.is_last:
                            if self.pp_group.is_last_rank:
                                # Only the last chunk we need to send the aux data
                                ret = self.send_aux(
                                    req,
                                    kv_chunk.prefill_aux_index,
                                    target_rank_registration_info.dst_aux_ptrs,
                                )
                            polls.append(True if ret == 0 else False)
                            dst_ranks_infos.append(
                                (req.endpoint, req.dst_port, req.room)
                            )

                            # Only sync status when all the dst ranks have received the kvcache
                            if len(polls) == req.required_dst_info_num:
                                status = KVPoll.Success if all(polls) else KVPoll.Failed
                                self.update_status(req.room, status)
                                for endpoint, dst_port, room in dst_ranks_infos:
                                    self.sync_status_to_decode_endpoint(
                                        endpoint, dst_port, room, status, local_rank
                                    )
                    else:
                        # Dummy request means the decode instance is not used, so its status can be marked as success directly
                        # Dummy request does not need to sync status to decode endpoint
                        if kv_chunk.is_last and req.room in self.request_status:
                            self.update_status(req.room, KVPoll.Success)

                if (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Success
                ):
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)

            except Exception as e:
                # NOTE(shangming): Remove this when we make sure the transfer thread is bug-free
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                )

    def start_prefill_thread(self):
        self._bind_server_socket()

        def bootstrap_thread():
            """This thread recvs pre-alloc notification from the decode engine"""
            # KVPoll.Bootstrapping -> KVPoll.WaitingForInput
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                room = waiting_req_bytes[0].decode("ascii")
                mooncake_session_id = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    self.decode_kv_args_table[mooncake_session_id] = (
                        KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    )
                    with self.session_lock:
                        if mooncake_session_id in self.failed_sessions:
                            self.failed_sessions.remove(mooncake_session_id)
                        if mooncake_session_id in self.session_failures:
                            del self.session_failures[mooncake_session_id]
                    logger.debug(
                        f"Register KVArgs from {mooncake_session_id} successfully"
                    )
                    continue
                else:
                    required_dst_info_num = int(waiting_req_bytes[6].decode("ascii"))
                    room = int(room)
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}

                    self.transfer_infos[room][mooncake_session_id] = (
                        TransferInfo.from_zmq(waiting_req_bytes)
                    )
                    # NOTE: after bootstrapping we can mark the req as waiting for input
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread).start()

    def start_decode_thread(self):
        self._bind_server_socket()

        def decode_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                if msg[0] == MooncakeKVManager.AUX_DATA_HEADER:
                    self._handle_aux_data(msg)
                    continue

                (bootstrap_room, status, prefill_rank) = msg
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                prefill_rank = int(prefill_rank.decode("ascii"))

                if status == KVPoll.Success:
                    if bootstrap_room in self.request_status:
                        self.prefill_response_tracker[bootstrap_room].add(prefill_rank)
                        expected_response_num = (
                            self.required_prefill_response_num_table[bootstrap_room]
                        )
                        arrived_response_num = len(
                            self.prefill_response_tracker[bootstrap_room]
                        )
                        if arrived_response_num == expected_response_num:
                            self.update_status(bootstrap_room, KVPoll.Success)
                elif status == KVPoll.Failed:
                    self.record_failure(
                        bootstrap_room,
                        f"Failed to get kvcache from prefill instance, it might be dead",
                    )
                    self.update_status(bootstrap_room, status)

        def heartbeat_checker():
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.prefill_dp_size_table.keys())

                for bootstrap_addr in addresses:
                    session = None
                    try:
                        with self.session_pool_lock:
                            session = self.session_pool[bootstrap_addr]
                        response = session.get(
                            f"http://{bootstrap_addr}/health",
                            timeout=(2, 3),
                            headers={"Connection": "keep-alive"},
                        )
                        if response.status_code == 200:
                            self.heartbeat_failures[bootstrap_addr] = 0

                            current_rooms = self.addr_to_rooms_tracker[
                                bootstrap_addr
                            ].copy()

                            for bootstrap_room in current_rooms:
                                # Remove KVPoll.Success requests from the tracker
                                if bootstrap_room not in self.request_status:
                                    self.addr_to_rooms_tracker[bootstrap_addr].discard(
                                        bootstrap_room
                                    )
                        else:
                            logger.info(
                                f"Attempting to reconnect to {bootstrap_addr}..."
                            )
                            self.heartbeat_failures[bootstrap_addr] = (
                                self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                            )
                            with self.session_pool_lock:
                                if bootstrap_addr in self.session_pool:
                                    del self.session_pool[bootstrap_addr]
                    except Exception:
                        logger.info(f"Attempting to reconnect to {bootstrap_addr}...")
                        self.heartbeat_failures[bootstrap_addr] = (
                            self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                        )

                    if (
                        self.heartbeat_failures.get(bootstrap_addr, 0)
                        >= self.max_failures
                    ):
                        self._handle_node_failure(bootstrap_addr)
                        with self.session_pool_lock:
                            if bootstrap_addr in self.session_pool:
                                del self.session_pool[bootstrap_addr]

        threading.Thread(target=decode_thread).start()
        threading.Thread(target=heartbeat_checker).start()

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        aux_index: Optional[int] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)

        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
            return

        if bootstrap_room not in self.transfer_infos:
            # This means that the current rank is a dummy rank for this request,
            # and it has already been marked as success, so there is no need to
            # add further chunks into the transfer queue.
            return

        # NOTE(shangming): sharding according to the dst_infos to make sure
        # requests with the same dst_sessions will be added into the same
        # queue, which enables early abort with failed sessions.
        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.rsplit(":", 1)[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        self.transfer_queues[shard_idx].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last=is_last,
                prefill_aux_index=aux_index,
            )
        )

    def check_status(self, bootstrap_room: int):
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            # NOTE: status is only allowed to be incremented unless it is KVPoll.Failed
            if status == KVPoll.Failed:
                self.request_status[bootstrap_room] = KVPoll.Failed
            else:
                self.request_status[bootstrap_room] = max(
                    self.request_status[bootstrap_room], status
                )

    def record_failure(self, bootstrap_room: int, failure_reason: str):
        with self.failure_lock:
            self.failure_records[bootstrap_room] = failure_reason

    def get_session_id(self):
        return self.engine.get_session_id()

    def _handle_node_failure(self, failed_bootstrap_addr):
        with self.connection_lock:
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            for k in keys_to_remove:
                del self.connection_pool[k]
            if failed_bootstrap_addr in self.prefill_attn_tp_size_table:
                del self.prefill_attn_tp_size_table[failed_bootstrap_addr]
            if failed_bootstrap_addr in self.prefill_dp_size_table:
                del self.prefill_dp_size_table[failed_bootstrap_addr]
            if failed_bootstrap_addr in self.prefill_pp_size_table:
                del self.prefill_pp_size_table[failed_bootstrap_addr]

            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
            )
            if failed_bootstrap_addr in self.addr_to_rooms_tracker:
                del self.addr_to_rooms_tracker[failed_bootstrap_addr]

        # Report the requests associated with the failed bootstrap addr and mark their status as KVPoll.Failed
        affected_rooms = []
        for room in possible_affected_rooms:
            if (
                room in self.request_status
                and self.check_status(room) != KVPoll.Success
            ):
                self.record_failure(
                    room,
                    f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr})",
                )
                self.update_status(room, KVPoll.Failed)
                affected_rooms.append(room)
        logger.error(
            f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr}), {len(affected_rooms)} requests affected"
        )


class MooncakeKVSender(CommonKVSender):

    def __init__(
        self,
        mgr: MooncakeKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.conclude_state = None
        self.init_time = time.time()

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
    ):
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices

        if not is_last:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                False,
            )
        else:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                True,
                aux_index=self.aux_index,
            )

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.Bootstrapping:
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.kv_mgr.bootstrap_timeout:
                        logger.warning_once(
                            "Some requests timed out when bootstrapping, "
                            "which means prefill instances fail to receive the KV indices from the decode instance of this request. "
                            "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
                        )
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.Bootstrapping",
                        )
                        self.conclude_state = KVPoll.Failed
                        return KVPoll.Failed

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)

    def failure_exception(self):
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        raise KVTransferError(self.bootstrap_room, failure_reason)

    def abort(self):
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        # Explicitly set the status to failure since this request has been aborted
        self.conclude_state = KVPoll.Failed


class MooncakeKVReceiver(CommonKVReceiver):
    _ctx = zmq.Context()
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()

    def __init__(
        self,
        mgr: MooncakeKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        prefill_dp_rank: Optional[int] = None,
    ):
        self.session_id = mgr.get_session_id()
        self.conclude_state = None
        self.init_time = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)

        self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].add(self.bootstrap_room)
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def _get_bootstrap_info_from_server(
        self, engine_rank, target_dp_group, target_pp_rank
    ):
        """Fetch the bootstrap info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={engine_rank}&target_dp_group={target_dp_group}&target_pp_rank={target_pp_rank}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                bootstrap_info = response.json()
                return bootstrap_info
            else:
                logger.error(
                    f"Failed to get prefill server info: {response.status_code}, {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error fetching prefill info from bootstrap: {e}")
            return None

    def _register_kv_args(self):
        for bootstrap_info in self.bootstrap_infos:
            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
            )
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )
            # Note(shangming): No need to add pp rank here since pp is not supported on the decode side yet
            tp_rank = self.kv_mgr.kv_args.engine_rank
            kv_item_len = self.kv_mgr.kv_args.kv_item_lens[0]
            dst_tp_rank = str(tp_rank).encode("ascii")
            dst_attn_tp_size = str(self.kv_mgr.attn_tp_size).encode("ascii")
            dst_kv_item_len = str(kv_item_len).encode("ascii")

            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(
                    [
                        "None".encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        packed_kv_data_ptrs,
                        packed_aux_data_ptrs,
                        dst_tp_rank,
                        dst_attn_tp_size,
                        dst_kv_item_len,
                    ]
                )

    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info["is_dummy"]

            with lock:
                sock.send_multipart(
                    [
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii") if not is_dummy else b"",
                        str(self.required_dst_info_num).encode("ascii"),
                    ]
                )
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.WaitingForInput:
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.kv_mgr.waiting_timeout:
                        logger.warning_once(
                            "Some requests fail to receive KV Cache transfer done signal after bootstrapping. "
                            "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
                        )
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.WaitingForInput",
                        )
                        self.conclude_state = KVPoll.Failed
                        return KVPoll.Failed

            return status

        else:
            return self.conclude_state

    def clear(self) -> None:
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)

        if self.bootstrap_room in self.kv_mgr.required_prefill_response_num_table:
            self.kv_mgr.required_prefill_response_num_table.pop(self.bootstrap_room)

        if self.bootstrap_room in self.kv_mgr.prefill_response_tracker:
            self.kv_mgr.prefill_response_tracker.pop(self.bootstrap_room)

    def failure_exception(self):
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        raise KVTransferError(self.bootstrap_room, failure_reason)

    def abort(self):
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        # Explicitly set the status to failure since this request has been aborted
        self.conclude_state = KVPoll.Failed


class MooncakeKVBootstrapServer(CommonKVBootstrapServer):
    pass


# ==================== Multimodal Embedding Classes ====================


class EmbeddingTransferError(Exception):
    def __init__(self, bootstrap_room: int, failure_reason: str):
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason

    def __str__(self):
        return f"EmbeddingTransferError(bootstrap_room={self.bootstrap_room}): {self.failure_reason}"


@dataclasses.dataclass
class TransferEmbeddingChunk:
    room: int
    embedding_index: int
    is_last: bool
    chunk_info: List[Tuple[int, int]]
    transmission_id: int = 0  # Track which round of transmission this is


@dataclasses.dataclass
class EmbeddingTransmissionState:
    """Track the state of a multi-round embedding transmission"""
    room: int
    embedding_index: int
    total_size_per_buffer: List[int]  # Total size for each buffer
    transmitted_size_per_buffer: List[int]  # How much has been transmitted for each buffer
    transmission_count: int = 0  # Number of transmission rounds completed
    
    def is_complete(self) -> bool:
        """Check if all data has been transmitted"""
        return all(
            transmitted >= total 
            for transmitted, total in zip(self.transmitted_size_per_buffer, self.total_size_per_buffer)
        )
    
    def get_remaining_chunks(self) -> List[Tuple[int, int]]:
        """Get chunk info for remaining data"""
        return [
            (transmitted, total - transmitted)
            for transmitted, total in zip(self.transmitted_size_per_buffer, self.total_size_per_buffer)
        ]


@dataclasses.dataclass
class TransferEmbeddingInfo:
    room: int
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_embedding_index: int
    required_dst_info_num: int
    transmission_id: int = 0  # Track current transmission round

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        transmission_id = 0
        if len(msg) > 6:
            transmission_id = int(msg[6].decode("ascii")) if msg[6] else 0
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_embedding_index=int(msg[4].decode("ascii")),
            required_dst_info_num=int(msg[5].decode("ascii")),
            transmission_id=transmission_id,
        )


@dataclasses.dataclass
class RequestMoreCacheInfo:
    """Request from Language side to Embedding side for more cache allocation"""
    room: int
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_embedding_index: int
    new_chunk_info: List[Tuple[int, int]]  # Updated chunk allocation info
    transmission_id: int  # Which transmission round this is for
    
    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        import json
        chunk_info_json = msg[5].decode("ascii")
        new_chunk_info = json.loads(chunk_info_json) if chunk_info_json else []
        
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_embedding_index=int(msg[4].decode("ascii")),
            new_chunk_info=[(offset, size) for offset, size in new_chunk_info],
            transmission_id=int(msg[6].decode("ascii")),
        )


@dataclasses.dataclass
class EmbeddingArgsRegisterInfo:
    room: str
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_embedding_ptrs: list[int]

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_embedding_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
        )


class MooncakeEmbeddingManager(BaseKVManager):
    # Message headers for different types of communication
    REQUEST_MORE_CACHE_HEADER = b"REQUEST_MORE_CACHE"
    EMBEDDING_DATA_HEADER = b"EMBEDDING_DATA"
    
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
    ):
        self.data_args = args
        self.engine = MooncakeTransferEngine(
            hostname=get_local_ip_by_remote(),
            gpu_id=self.data_args.gpu_id,
            ib_device=self.data_args.ib_device,
        )
        self.disaggregation_mode = disaggregation_mode
        # for embedding/language model multi node infer
        self.bootstrap_port = server_args.disaggregation_bootstrap_port
        self.dist_init_addr = server_args.dist_init_addr
        self.tp_size = server_args.tp_size
        self.dp_size = server_args.dp_size
        self.request_status: Dict[int, KVPoll] = {}
        self.rank_port = None
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.register_buffer_to_engine()

        if self.disaggregation_mode == DisaggregationMode.ENCODE:
            self.transfer_infos: Dict[int, Dict[str, TransferEmbeddingInfo]] = {}
            self.language_args_table: Dict[str, EmbeddingArgsRegisterInfo] = {}
            # Track transmission state for multi-round transfers
            self.transmission_states: Dict[int, EmbeddingTransmissionState] = {}
            self.transmission_state_lock = threading.Lock()
            self.start_embedding_thread()
            self._register_to_bootstrap()
            self.session_failures = defaultdict(int)
            self.failed_sessions = set()
            self.session_lock = threading.Lock()

            # Determine the number of threads to use for aux sender
            cpu_count = os.cpu_count()
            transfer_thread_pool_size = get_int_env_var(
                "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE",
                min(max(4, int(0.75 * cpu_count) // 8), 12),
            )
            transfer_queue_size = get_int_env_var("SGLANG_DISAGGREGATION_QUEUE_SIZE", 4)
            self.transfer_queues: List[FastQueue] = [
                FastQueue() for _ in range(transfer_queue_size)
            ]
            assert transfer_thread_pool_size >= transfer_queue_size, (
                f"The environment variable SGLANG_DISAGGREGATION_THREAD_POOL_SIZE={transfer_thread_pool_size} must be "
                f"greater than or equal to SGLANG_DISAGGREGATION_QUEUE_SIZE={transfer_queue_size}."
            )
            self.executors = [
                concurrent.futures.ThreadPoolExecutor(
                    transfer_thread_pool_size // transfer_queue_size
                )
                for _ in range(transfer_queue_size)
            ]
            for queue, executor in zip(self.transfer_queues, self.executors):
                threading.Thread(
                    target=self.transfer_worker, args=(queue, executor), daemon=True
                ).start()

            self.bootstrap_time_out = get_int_env_var(
                "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT", 30
            )
        elif self.disaggregation_mode == DisaggregationMode.LANGUAGE:
            self.heartbeat_failures = {}
            self.session_pool = defaultdict(requests.Session)
            self.session_pool_lock = threading.Lock()
            self.addr_to_rooms_tracker = defaultdict(set)
            self.connection_lock = threading.Lock()
            # Track incomplete transmissions that need more cache
            self.pending_cache_requests: Dict[int, RequestMoreCacheInfo] = {}
            self.pending_cache_lock = threading.Lock()
            # Heartbeat interval should be at least 2 seconds
            self.heartbeat_interval = max(
                float(os.getenv("SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL", 5.0)), 2.0
            )
            # Heartbeat failure should be at least 1
            self.max_failures = max(
                get_int_env_var("SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE", 2), 1
            )
            self.start_language_thread()
            self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
            self.embedding_tp_size_table: Dict[str, int] = {}
            self.embedding_dp_size_table: Dict[str, int] = {}
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

        self.failure_records: Dict[int, str] = {}
        self.failure_lock = threading.Lock()

    def register_buffer_to_engine(self):
        # Only register aux data buffers, not kv data buffers
        for aux_data_ptr, aux_data_len in zip(
            self.data_args.aux_data_ptrs, self.data_args.aux_data_lens
        ):
            self.engine.register(aux_data_ptr, aux_data_len)

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

    def send_embedding(
        self,
        mooncake_session_id: str,
        embedding_index: int,
        dst_embedding_ptrs: list[int],
        dst_embedding_index: int,
        chunk_info: List[Tuple[int, int]],
    ):
        """
        Send embedding data according to chunk_info.
        Each tuple in chunk_info is (offset, size) for the corresponding buffer.
        """
        status_list = []

        for i in range(len(self.data_args.aux_item_lens)):
            chunk_offset, chunk_size = chunk_info[i]
            
            # Skip if chunk_size is 0 (nothing to send for this buffer)
            if chunk_size == 0:
                continue
                
            embedding_item_len = self.data_args.aux_item_lens[i]
            embedding_addr = (
                self.data_args.aux_data_ptrs[i]
                + embedding_index * embedding_item_len
                + chunk_offset
            )
            dst_embedding_addr = (
                dst_embedding_ptrs[i]
                + dst_embedding_index * embedding_item_len
                + chunk_offset
            )

            status = self.engine.transfer_sync(
                mooncake_session_id,
                embedding_addr,
                dst_embedding_addr,
                chunk_size,
            )
            status_list.append(status)

        return 0 if sum(status_list) == 0 else 1
    
    def send_request_more_cache_to_embedding(
        self,
        remote: str,
        dst_port: int,
        room: int,
        mooncake_session_id: str,
        dst_embedding_index: int,
        new_chunk_info: List[Tuple[int, int]],
        transmission_id: int,
    ):
        """Language side sends request to Embedding side for more cache allocation"""
        import json
        
        if ":" in remote:
            remote = remote.split(":")[0]
        
        chunk_info_json = json.dumps(new_chunk_info)
        
        self._connect("tcp://" + remote + ":" + str(dst_port)).send_multipart(
            [
                MooncakeEmbeddingManager.REQUEST_MORE_CACHE_HEADER,
                str(room).encode("ascii"),
                get_local_ip_by_remote().encode("ascii"),
                str(self.rank_port).encode("ascii"),
                mooncake_session_id.encode("ascii"),
                str(dst_embedding_index).encode("ascii"),
                chunk_info_json.encode("ascii"),
                str(transmission_id).encode("ascii"),
            ]
        )
        logger.debug(
            f"Sent REQUEST_MORE_CACHE for room {room}, transmission_id {transmission_id}"
        )
    
    def _handle_request_more_cache(self, msg: List[bytes]):
        """Embedding side handles request for more cache from Language side"""
        request_info = RequestMoreCacheInfo.from_zmq(msg[1:])
        
        logger.debug(
            f"Received REQUEST_MORE_CACHE for room {request_info.room}, "
            f"transmission_id {request_info.transmission_id}"
        )
        
        # Update the transfer info with new chunk allocation
        if request_info.room in self.transfer_infos:
            for session_id, transfer_info in self.transfer_infos[request_info.room].items():
                if session_id == request_info.mooncake_session_id:
                    # Update transmission_id to indicate this is a continuation
                    transfer_info.transmission_id = request_info.transmission_id
                    
                    # Add the continuation transfer to queue
                    with self.transmission_state_lock:
                        if request_info.room in self.transmission_states:
                            state = self.transmission_states[request_info.room]
                            state.transmission_count += 1
                            
                            # Schedule the next chunk transmission
                            self.add_transfer_request(
                                request_info.room,
                                state.embedding_index,
                                is_last=state.is_complete(),  # Check if this will be the last transmission
                                chunk_info=request_info.new_chunk_info,
                                transmission_id=request_info.transmission_id,
                            )
                    break

    def sync_status_to_language_endpoint(
        self, remote: str, dst_port: int, room: int, status: int
    ):
        if ":" in remote:
            remote = remote.split(":")[0]
        self._connect("tcp://" + remote + ":" + str(dst_port)).send_multipart(
            [
                str(room).encode("ascii"),
                str(status).encode("ascii"),
            ]
        )

    def transfer_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):
        while True:
            try:
                embedding_chunk: TransferEmbeddingChunk = queue.get()
                reqs_to_be_processed = (
                    self.transfer_infos[embedding_chunk.room].values()
                    if embedding_chunk.room in self.transfer_infos
                    else []
                )
                polls = []
                dst_ranks_infos = []

                for req in reqs_to_be_processed:
                    # Skip if this transfer info is for a different transmission round
                    if req.transmission_id != embedding_chunk.transmission_id:
                        continue
                    
                    # Early exit if the request has failed
                    with self.session_lock:
                        if req.mooncake_session_id in self.failed_sessions:
                            self.record_failure(
                                embedding_chunk.room,
                                f"Language instance could be dead, remote mooncake session {req.mooncake_session_id} is not alive",
                            )
                            self.update_status(embedding_chunk.room, KVPoll.Failed)
                            self.sync_status_to_language_endpoint(
                                req.endpoint,
                                req.dst_port,
                                req.room,
                                KVPoll.Failed,
                            )
                            break

                    ret = self.send_embedding(
                        req.mooncake_session_id,
                        embedding_chunk.embedding_index,
                        self.language_args_table[
                            req.mooncake_session_id
                        ].dst_embedding_ptrs,
                        req.dst_embedding_index,
                        embedding_chunk.chunk_info,
                    )
                    
                    # Update transmission state after successful send
                    if ret == 0 and embedding_chunk.room in self.transmission_states:
                        with self.transmission_state_lock:
                            state = self.transmission_states[embedding_chunk.room]
                            for i, (offset, size) in enumerate(embedding_chunk.chunk_info):
                                state.transmitted_size_per_buffer[i] += size
                    
                    if ret != 0:
                        with self.session_lock:
                            self.session_failures[req.mooncake_session_id] += 1
                            # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
                            if self.session_failures[req.mooncake_session_id] >= 1:
                                self.failed_sessions.add(req.mooncake_session_id)
                                logger.error(
                                    f"Session {req.mooncake_session_id} failed."
                                )
                            logger.error(
                                f"Session {req.mooncake_session_id} failed with {embedding_chunk.room=};{req.endpoint=};{req.dst_port=};{req.room=}"
                            )
                        self.record_failure(
                            embedding_chunk.room,
                            f"Failed to send embedding chunk of {embedding_chunk.room} to {req.endpoint}:{req.dst_port}",
                        )
                        self.update_status(embedding_chunk.room, KVPoll.Failed)
                        self.sync_status_to_language_endpoint(
                            req.endpoint, req.dst_port, req.room, KVPoll.Failed
                        )
                        break

                    polls.append(True if ret == 0 else False)
                    dst_ranks_infos.append((req.endpoint, req.dst_port, req.room))

                    # Only sync status when all the dst ranks have received the embedding data
                    if len(polls) == req.required_dst_info_num:
                        # Check if transmission is complete
                        is_transmission_complete = True
                        if embedding_chunk.room in self.transmission_states:
                            with self.transmission_state_lock:
                                state = self.transmission_states[embedding_chunk.room]
                                is_transmission_complete = state.is_complete()
                        
                        if is_transmission_complete:
                            status = KVPoll.Success if all(polls) else KVPoll.Failed
                        else:
                            # Partial transmission completed, but not final
                            status = KVPoll.Transferring
                        
                        self.update_status(req.room, status)
                        for endpoint, dst_port, room in dst_ranks_infos:
                            self.sync_status_to_language_endpoint(
                                endpoint, dst_port, room, status
                            )

                if (
                    embedding_chunk.room not in self.request_status
                    or self.check_status(embedding_chunk.room) == KVPoll.Success
                ):
                    if embedding_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(embedding_chunk.room)

            except Exception as e:
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Embedding instance with bootstrap_port={self.bootstrap_port} is dead."
                )

    def start_embedding_thread(self):
        self.rank_port = get_free_port()
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def embedding_thread():
            """This thread recvs pre-alloc notification from the language engine"""
            # KVPoll.Bootstrapping -> KVPoll.WaitingForInput
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                
                # Handle REQUEST_MORE_CACHE messages
                if waiting_req_bytes[0] == MooncakeEmbeddingManager.REQUEST_MORE_CACHE_HEADER:
                    self._handle_request_more_cache(waiting_req_bytes)
                    continue
                
                room = waiting_req_bytes[0].decode("ascii")
                mooncake_session_id = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    self.language_args_table[mooncake_session_id] = (
                        EmbeddingArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    )
                    with self.session_lock:
                        if mooncake_session_id in self.failed_sessions:
                            self.failed_sessions.remove(mooncake_session_id)
                        if mooncake_session_id in self.session_failures:
                            del self.session_failures[mooncake_session_id]
                    continue
                else:
                    required_dst_info_num = int(waiting_req_bytes[5].decode("ascii"))
                    room = int(room)
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}

                    self.transfer_infos[room][mooncake_session_id] = (
                        TransferEmbeddingInfo.from_zmq(waiting_req_bytes)
                    )
                    # NOTE: after bootstrapping we can mark the req as waiting for input
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=embedding_thread).start()

    def start_language_thread(self):
        self.rank_port = get_free_port()
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def language_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                bootstrap_room = int(msg[0].decode("ascii"))
                status = int(msg[1].decode("ascii"))
                
                if status == KVPoll.Failed:
                    self.record_failure(
                        bootstrap_room,
                        f"Failed to get embedding data from embedding instance, it might be dead",
                    )
                elif status == KVPoll.Transferring:
                    # Partial transmission completed, waiting for more cache allocation
                    logger.debug(
                        f"Partial transmission completed for room {bootstrap_room}, "
                        f"status: {status}"
                    )
                
                self.update_status(bootstrap_room, status)

        def heartbeat_checker():
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.embedding_dp_size_table.keys())

                for bootstrap_addr in addresses:
                    session = None
                    try:
                        with self.session_pool_lock:
                            session = self.session_pool[bootstrap_addr]
                        response = session.get(
                            f"http://{bootstrap_addr}/health",
                            timeout=(2, 3),
                            headers={"Connection": "keep-alive"},
                        )
                        if response.status_code == 200:
                            self.heartbeat_failures[bootstrap_addr] = 0

                            current_rooms = self.addr_to_rooms_tracker[
                                bootstrap_addr
                            ].copy()

                            for bootstrap_room in current_rooms:
                                # Remove KVPoll.Success requests from the tracker
                                if bootstrap_room not in self.request_status:
                                    self.addr_to_rooms_tracker[bootstrap_addr].discard(
                                        bootstrap_room
                                    )
                        else:
                            logger.info(
                                f"Attempting to reconnect to {bootstrap_addr}..."
                            )
                            self.heartbeat_failures[bootstrap_addr] = (
                                self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                            )
                            with self.session_pool_lock:
                                if bootstrap_addr in self.session_pool:
                                    del self.session_pool[bootstrap_addr]
                    except Exception:
                        logger.info(f"Attempting to reconnect to {bootstrap_addr}...")
                        self.heartbeat_failures[bootstrap_addr] = (
                            self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                        )

                    if (
                        self.heartbeat_failures.get(bootstrap_addr, 0)
                        >= self.max_failures
                    ):
                        self._handle_node_failure(bootstrap_addr)
                        with self.session_pool_lock:
                            if bootstrap_addr in self.session_pool:
                                del self.session_pool[bootstrap_addr]

        threading.Thread(target=language_thread).start()
        threading.Thread(target=heartbeat_checker).start()

    def add_transfer_request(
        self,
        bootstrap_room: int,
        embedding_index: int,
        is_last: bool,
        chunk_info: List[Tuple[int, int]],
        transmission_id: int = 0,
        total_sizes: Optional[List[int]] = None,
    ):
        """
        Add a transfer request to the queue.
        
        Args:
            bootstrap_room: Room ID for this request
            embedding_index: Index of the embedding data
            is_last: Whether this is the last chunk (deprecated for multi-round transmission)
            chunk_info: List of (offset, size) tuples for each buffer
            transmission_id: ID to track multi-round transmissions
            total_sizes: Total sizes for each buffer (only needed for first transmission)
        """
        assert self.disaggregation_mode == DisaggregationMode.ENCODE

        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            return

        if bootstrap_room not in self.transfer_infos:
            # This means that the current rank is a dummy rank for this request,
            # and it has already been marked as success, so there is no need to
            # add further chunks into the transfer queue.
            return

        # Initialize transmission state on first transmission
        if transmission_id == 0 and total_sizes is not None:
            with self.transmission_state_lock:
                self.transmission_states[bootstrap_room] = EmbeddingTransmissionState(
                    room=bootstrap_room,
                    embedding_index=embedding_index,
                    total_size_per_buffer=total_sizes,
                    transmitted_size_per_buffer=[0] * len(total_sizes),
                    transmission_count=0,
                )

        # NOTE(shangming): sharding according to the dst_infos to make sure
        # requests with the same dst_sessions will be added into the same
        # queue, which enables early abort with failed sessions.
        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.split(":")[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        self.transfer_queues[shard_idx].put(
            TransferEmbeddingChunk(
                room=bootstrap_room,
                embedding_index=embedding_index,
                is_last=is_last,
                chunk_info=chunk_info,
                transmission_id=transmission_id,
            )
        )

    def check_status(self, bootstrap_room: int):
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            # NOTE: status is only allowed to be incremented unless it is KVPoll.Failed
            if status == KVPoll.Failed:
                self.request_status[bootstrap_room] = KVPoll.Failed
            else:
                self.request_status[bootstrap_room] = max(
                    self.request_status[bootstrap_room], status
                )

    def record_failure(self, bootstrap_room: int, failure_reason: str):
        with self.failure_lock:
            self.failure_records[bootstrap_room] = failure_reason

    def get_session_id(self):
        return self.engine.get_session_id()

    def _register_to_bootstrap(self):
        """Register EmbeddingSender to bootstrap server via HTTP POST."""
        bootstrap_server_url = f"{get_local_ip_by_remote()}:{self.bootstrap_port}"
        url = f"http://{bootstrap_server_url}/route"
        payload = {
            "role": "Encode",
            "tp_size": self.tp_size,
            "dp_size": self.dp_size,
            "rank_ip": get_local_ip_by_remote(),
            "rank_port": self.rank_port,
            "engine_rank": self.data_args.engine_rank,
        }

        try:
            response = requests.put(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.debug("Embedding successfully registered to bootstrap server.")
            else:
                logger.error(
                    f"Embedding instance failed to connect to bootstrap server: {response.status_code}, {response.text}"
                )
        except Exception as e:
            logger.error(
                f"Embedding instance failed to register to bootstrap server: {e}"
            )

    def _handle_node_failure(self, failed_bootstrap_addr):
        with self.connection_lock:
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            for k in keys_to_remove:
                del self.connection_pool[k]
            if failed_bootstrap_addr in self.embedding_dp_size_table:
                del self.embedding_dp_size_table[failed_bootstrap_addr]

            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
            )
            if failed_bootstrap_addr in self.addr_to_rooms_tracker:
                del self.addr_to_rooms_tracker[failed_bootstrap_addr]

        # Report the requests associated with the failed bootstrap addr and mark their status as KVPoll.Failed
        affected_rooms = []
        for room in possible_affected_rooms:
            if (
                room in self.request_status
                and self.check_status(room) != KVPoll.Success
            ):
                self.record_failure(
                    room,
                    f"Losing connection with embedding instance (bootstrap_addr: {failed_bootstrap_addr})",
                )
                self.update_status(room, KVPoll.Failed)
                affected_rooms.append(room)
        logger.error(
            f"Losing connection with embedding instance (bootstrap_addr: {failed_bootstrap_addr}), affected {len(affected_rooms)} requests"
        )


class MooncakeEmbeddingSender(BaseKVSender):
    def __init__(
        self,
        mgr: MooncakeEmbeddingManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        self.embedding_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.embedding_mgr.update_status(bootstrap_room, KVPoll.Bootstrapping)
        self.embedding_index = None
        self.bootstrap_server_url = bootstrap_addr
        self.conclude_state = None
        self.init_time = None
        self.dest_tp_ranks = dest_tp_ranks
        self.pp_rank = pp_rank
        self.current_transmission_id = 0

    def init(self, embedding_index: Optional[int] = None):
        # For embedding data, we don't need num_kv_indices, but we keep the interface consistent
        self.embedding_index = embedding_index
        self.init_time = time.time()

    def send(
        self,
        kv_indices: npt.NDArray[np.int64],
    ):
        # For embedding data, we don't use kv_indices, but we keep the interface consistent
        # We only send embedding data once at the end
        pass

    def send_embedding(
        self, 
        embedding_index: int, 
        last_chunk: bool, 
        chunk_info: List[Tuple[int, int]],
        total_sizes: Optional[List[int]] = None,
    ):
        """
        Send embedding data to language instances.
        
        Args:
            embedding_index: Index of the embedding data
            last_chunk: Whether this is expected to be the last chunk (may not be if allocation is insufficient)
            chunk_info: List of (offset, size) tuples for each buffer
            total_sizes: Total sizes for each buffer (only needed for first transmission)
        """
        self.embedding_mgr.add_transfer_request(
            self.bootstrap_room, 
            embedding_index, 
            last_chunk, 
            chunk_info,
            transmission_id=self.current_transmission_id,
            total_sizes=total_sizes,
        )
    
    def continue_transmission(self, chunk_info: List[Tuple[int, int]]):
        """
        Continue transmission with new chunk allocation from Language side.
        Called after receiving notification that more cache has been allocated.
        """
        self.current_transmission_id += 1
        
        # Check if this will complete the transmission
        is_last = False
        if self.bootstrap_room in self.embedding_mgr.transmission_states:
            with self.embedding_mgr.transmission_state_lock:
                state = self.embedding_mgr.transmission_states[self.bootstrap_room]
                # Calculate if the new chunks will complete the transmission
                remaining = state.get_remaining_chunks()
                is_last = all(
                    chunk_info[i][1] >= remaining[i][1] 
                    for i in range(len(chunk_info))
                )
        
        self.embedding_mgr.add_transfer_request(
            self.bootstrap_room,
            self.embedding_index,
            is_last,
            chunk_info,
            transmission_id=self.current_transmission_id,
        )
        
        logger.debug(
            f"Continuing transmission for room {self.bootstrap_room}, "
            f"transmission_id {self.current_transmission_id}, is_last={is_last}"
        )

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.embedding_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.Bootstrapping:
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.embedding_mgr.bootstrap_time_out:
                        self.embedding_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.Bootstrapping",
                        )
                        self.conclude_state = KVPoll.Failed
                        return KVPoll.Failed

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        if self.bootstrap_room in self.embedding_mgr.request_status:
            self.embedding_mgr.request_status.pop(self.bootstrap_room)

        if self.bootstrap_room in self.embedding_mgr.transfer_infos:
            self.embedding_mgr.transfer_infos.pop(self.bootstrap_room)

    def failure_exception(self):
        self.clear()

        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        with self.embedding_mgr.failure_lock:
            failure_reason = self.embedding_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        raise EmbeddingTransferError(self.bootstrap_room, failure_reason)


class MooncakeEmbeddingReceiver(BaseKVReceiver):
    _ctx = zmq.Context()
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()

    def __init__(
        self,
        mgr: MooncakeEmbeddingManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        prefill_dp_rank: Optional[int] = None,
    ):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.embedding_mgr = mgr
        self.session_id = self.embedding_mgr.get_session_id()
        self.embedding_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)
        self.conclude_state = None
        self.data_parallel_rank = prefill_dp_rank
        self.current_transmission_id = 0
        self.embedding_index = None

        if self.bootstrap_addr not in self.embedding_mgr.embedding_dp_size_table:
            self.embedding_tp_size, self.embedding_dp_size = (
                self._get_embedding_parallel_info_from_server()
            )
            if self.embedding_tp_size is None or self.embedding_dp_size is None:
                self.embedding_mgr.record_failure(
                    self.bootstrap_room,
                    f"Could not fetch embedding parallel info from bootstrap_addr: {self.bootstrap_addr}",
                )
                self.embedding_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return
            else:
                self.embedding_mgr.embedding_dp_size_table[self.bootstrap_addr] = (
                    self.embedding_dp_size
                )
                self.embedding_mgr.embedding_tp_size_table[self.bootstrap_addr] = (
                    self.embedding_tp_size
                )
        else:
            self.embedding_tp_size = self.embedding_mgr.embedding_tp_size_table[
                self.bootstrap_addr
            ]
            self.embedding_dp_size = self.embedding_mgr.embedding_dp_size_table[
                self.bootstrap_addr
            ]

        local_tp_size_per_dp_rank = (
            self.embedding_mgr.tp_size // self.embedding_mgr.dp_size
        )
        if local_tp_size_per_dp_rank <= self.embedding_tp_size:
            self.target_tp_rank = (
                self.embedding_mgr.data_args.engine_rank % local_tp_size_per_dp_rank
            )
            self.required_dst_info_num = 1
            self.target_tp_ranks = [self.target_tp_rank]
        else:
            self.target_tp_rank = (
                self.embedding_mgr.data_args.engine_rank % self.embedding_tp_size
            )
            self.required_dst_info_num = (
                local_tp_size_per_dp_rank // self.embedding_tp_size
            )
            self.target_tp_ranks = [self.target_tp_rank]

        if self.data_parallel_rank is not None:
            self.target_dp_group = self.data_parallel_rank
        else:
            self.target_dp_group = bootstrap_room % self.embedding_dp_size

        # NOTE: key distinguished by bootstrap_addr, target_dp_group, and target_tp_rank
        bootstrap_key = (
            f"{self.bootstrap_addr}_{self.target_dp_group}_{self.target_tp_rank}"
        )

        if bootstrap_key not in self.embedding_mgr.connection_pool:
            bootstrap_infos = []
            for target_tp_rank in self.target_tp_ranks:
                bootstrap_info = self._get_bootstrap_info_from_server(
                    target_tp_rank,
                    self.target_dp_group,
                )
                if bootstrap_info is not None:
                    bootstrap_infos.append(bootstrap_info)
                else:
                    self.embedding_mgr.record_failure(
                        self.bootstrap_room,
                        f"Could not fetch bootstrap info for engine rank: {self.embedding_mgr.data_args.engine_rank} and target_dp_group: {self.target_dp_group}",
                    )
                    self.embedding_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                    return

            self.bootstrap_infos = bootstrap_infos
            self.embedding_mgr.connection_pool[bootstrap_key] = self.bootstrap_infos

            # Register aux_args only once to prefill AuxManager according to the info fetched from the bootstrap server
            self._register_embedding_args()
        else:
            self.bootstrap_infos = self.embedding_mgr.connection_pool[bootstrap_key]

        assert len(self.bootstrap_infos) > 0
        self.embedding_mgr.addr_to_rooms_tracker[self.bootstrap_addr].add(
            self.bootstrap_room
        )
        self.embedding_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def _get_bootstrap_info_from_server(self, engine_rank, target_dp_group):
        """Fetch the bootstrap info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={engine_rank}&target_dp_group={target_dp_group}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                bootstrap_info = response.json()
                return bootstrap_info
            else:
                logger.error(
                    f"Failed to get prefill server info: {response.status_code}, {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error fetching prefill info from bootstrap: {e}")
            return None

    def _get_embedding_parallel_info_from_server(self) -> Tuple[int, int]:
        """Fetch the embedding parallel info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={-1}&target_dp_group={-1}"
            response = requests.get(url)
            if response.status_code == 200:
                embedding_parallel_info = response.json()
                return int(embedding_parallel_info["embedding_tp_size"]), int(
                    embedding_parallel_info["embedding_dp_size"]
                )
            else:
                logger.error(
                    f"Failed to get embedding parallel info: {response.status_code}, {response.text}"
                )
                return None, None
        except Exception as e:
            logger.error(f"Error fetching embedding parallel info from bootstrap: {e}")
            return None, None

    def _register_embedding_args(self):
        for bootstrap_info in self.bootstrap_infos:
            self.embedding_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            packed_embedding_data_ptrs = b"".join(
                struct.pack("Q", ptr)
                for ptr in self.embedding_mgr.data_args.aux_data_ptrs
            )

            sock, lock = self._connect("tcp://" + self.embedding_server_url)
            with lock:
                sock.send_multipart(
                    [
                        "None".encode("ascii"),
                        get_local_ip_by_remote().encode("ascii"),
                        str(self.embedding_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        packed_embedding_data_ptrs,
                    ]
                )

    @classmethod
    def _connect(cls, endpoint: str):
        with cls._global_lock:
            if endpoint not in cls._socket_cache:
                sock = cls._ctx.socket(zmq.PUSH)
                sock.connect(endpoint)
                cls._socket_cache[endpoint] = sock
                cls._socket_locks[endpoint] = threading.Lock()
            return cls._socket_cache[endpoint], cls._socket_locks[endpoint]

    def init(self, embedding_index: Optional[int] = None):
        self.embedding_index = embedding_index
        for bootstrap_info in self.bootstrap_infos:
            self.embedding_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )

            sock, lock = self._connect("tcp://" + self.embedding_server_url)
            with lock:
                sock.send_multipart(
                    [
                        str(self.bootstrap_room).encode("ascii"),
                        get_local_ip_by_remote().encode("ascii"),
                        str(self.embedding_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        str(embedding_index).encode("ascii"),
                        str(self.required_dst_info_num).encode("ascii"),
                        str(self.current_transmission_id).encode("ascii"),
                    ]
                )
    
    def request_more_cache(self, new_chunk_info: List[Tuple[int, int]]):
        """
        Request more cache allocation from Embedding side.
        Called by Language side when allocated block is smaller than needed.
        
        Args:
            new_chunk_info: Updated allocation info with (offset, size) for each buffer
        """
        self.current_transmission_id += 1
        
        for bootstrap_info in self.bootstrap_infos:
            embedding_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            
            # Send request to Embedding side
            self.embedding_mgr.send_request_more_cache_to_embedding(
                remote=bootstrap_info['rank_ip'],
                dst_port=bootstrap_info['rank_port'],
                room=self.bootstrap_room,
                mooncake_session_id=self.session_id,
                dst_embedding_index=self.embedding_index,
                new_chunk_info=new_chunk_info,
                transmission_id=self.current_transmission_id,
            )
        
        # Reset status to wait for the next transmission
        self.embedding_mgr.update_status(self.bootstrap_room, KVPoll.Transferring)
        
        logger.debug(
            f"Requested more cache for room {self.bootstrap_room}, "
            f"transmission_id {self.current_transmission_id}"
        )

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.embedding_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.Transferring:
                # Partial transmission in progress, not final state yet
                pass

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        if self.bootstrap_room in self.embedding_mgr.request_status:
            self.embedding_mgr.request_status.pop(self.bootstrap_room)

    def failure_exception(self):
        self.clear()

        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        with self.embedding_mgr.failure_lock:
            failure_reason = self.embedding_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        raise EmbeddingTransferError(self.bootstrap_room, failure_reason)

    def abort(self):
        self.embedding_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        # Explicitly set the status to failure since this request has been aborted
        self.conclude_state = KVPoll.Failed


class MooncakeEmbeddingBootstrapServer(BaseKVBootstrapServer):
    def __init__(self, host: str, port: int):
        self.port = port
        self.host = host
        self.app = web.Application()
        self.store = dict()
        self.lock = asyncio.Lock()
        self._setup_routes()
        self.tp_size = None
        self.dp_size = None
        self.embedding_port_table: Dict[int, Dict[int, Dict[str, Union[str, int]]]] = {}

        # Start bootstrap server
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.run()

    def run(self):
        self.thread.start()

    def _setup_routes(self):
        self.app.router.add_route("*", "/route", self._handle_route)
        self.app.router.add_get("/health", self._handle_health_check)

    async def _handle_health_check(self, request):
        return web.Response(text="OK", status=200)

    async def _handle_route(self, request: web.Request):
        method = request.method
        if method == "PUT":
            return await self._handle_route_put(request)
        elif method == "GET":
            return await self._handle_route_get(request)
        else:
            return web.Response(
                text="Method not allowed", status=405, content_type="application/json"
            )

    async def _handle_route_put(self, request: web.Request):
        data = await request.json()
        role = data["role"]
        tp_size = data["tp_size"]
        dp_size = data["dp_size"]
        rank_ip = data["rank_ip"]
        rank_port = int(data["rank_port"])
        engine_rank = int(data["engine_rank"])

        if self.tp_size is None:
            self.tp_size = tp_size

        if self.dp_size is None:
            self.dp_size = dp_size

        if role == "Encode":
            dp_group = engine_rank // self.tp_size
            tp_rank_in_dp_group = engine_rank % self.tp_size

            # Add lock to make sure thread-safe
            async with self.lock:
                if dp_group not in self.embedding_port_table:
                    self.embedding_port_table[dp_group] = {}

            self.embedding_port_table[dp_group][tp_rank_in_dp_group] = {
                "rank_ip": rank_ip,
                "rank_port": rank_port,
            }

        return web.Response(text="OK", status=200)

    async def _handle_route_get(self, request: web.Request):
        engine_rank = request.query.get("engine_rank")
        target_dp_group = request.query.get("target_dp_group")
        if not engine_rank or not target_dp_group:
            return web.Response(text="Missing inputs for bootstrap server.", status=400)

        # Currently we use engine_rank == -1 and target_dp_group == -1 to sync dp size
        if int(engine_rank) == -1 and int(target_dp_group) == -1:
            embedding_parallel_info = {
                "embedding_tp_size": self.tp_size,
                "embedding_dp_size": self.dp_size,
            }
            return web.json_response(embedding_parallel_info, status=200)

        # Find corresponding prefill info
        async with self.lock:
            bootstrap_info = self.embedding_port_table[int(target_dp_group)][
                int(engine_rank)
            ]

        if bootstrap_info is not None:
            return web.json_response(bootstrap_info, status=200)
        else:
            return web.Response(text="Bootstrap info not Found", status=404)

    def _run_server(self):
        try:
            # Event Loop
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            access_log = None
            if logging.getLogger(__name__).getEffectiveLevel() <= logging.DEBUG:
                access_log = self.app.logger

            self._runner = web.AppRunner(self.app, access_log=access_log)
            self._loop.run_until_complete(self._runner.setup())

            site = web.TCPSite(self._runner, port=self.port)
            self._loop.run_until_complete(site.start())
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
        finally:
            # Cleanup
            self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()

    def close(self):
        """Shutdown"""
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            logger.info("Stopping server loop...")

        if self.thread.is_alive():
            self.thread.join(timeout=2)
            logger.info("Server thread stopped")

    def poll(self) -> KVPoll: ...
