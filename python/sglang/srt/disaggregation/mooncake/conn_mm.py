from __future__ import annotations

import ctypes
import dataclasses
import logging
import queue
import struct
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import format_tcp_address, get_int_env_var, is_valid_ipv6_address

logger = logging.getLogger(__name__)


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
    embedding_indices: List[int]
    is_last: bool
    total_tokens: int


@dataclasses.dataclass
class TransferEmbeddingInfo:
    room: int
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_embedding_indices: List[int]
    required_dst_info_num: int
    sent_tokens: int = 0
    allocated_tokens: int = 0
    src_embedding_indices: Optional[List[int]] = None
    total_tokens: int = 0
    resume_ready: bool = False

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        indices_str = msg[4].decode("ascii")
        if indices_str:
            dst_embedding_indices = [int(x) for x in indices_str.split(",")]
        else:
            dst_embedding_indices = []

        required_dst_info_num = int(msg[5].decode("ascii"))

        sent_tokens = 0
        allocated_tokens = 0

        if len(msg) >= 8:
            sent_tokens = int(msg[6].decode("ascii"))
            allocated_tokens = int(msg[7].decode("ascii"))
        elif len(msg) >= 7:
            allocated_tokens = int(msg[6].decode("ascii"))

        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_embedding_indices=dst_embedding_indices,
            required_dst_info_num=required_dst_info_num,
            sent_tokens=sent_tokens,
            allocated_tokens=allocated_tokens,
        )


@dataclasses.dataclass
class EmbeddingArgsRegisterInfo:
    room: str
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_embedding_ptrs: List[int]

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_embedding_ptrs=list(struct.unpack(f"{len(msg[4]) // 8}Q", msg[4])),
        )


class AuxDataCodec:
    @staticmethod
    def serialize_data_from_buffer(src_addr: int, data_length: int) -> bytes:
        buffer = (ctypes.c_byte * data_length).from_address(src_addr)
        return bytes(buffer)

    @staticmethod
    def deserialize_data_to_buffer(kv_args: KVArgs, buffer_index: int, aux_index: int, data: bytes):
        dst_aux_ptr = kv_args.aux_data_ptrs[buffer_index]
        item_len = kv_args.aux_item_lens[buffer_index]
        dst_addr = dst_aux_ptr + item_len * aux_index
        buffer = (ctypes.c_byte * len(data)).from_address(dst_addr)
        buffer[:] = data


class MooncakeEmbeddingManager(CommonKVManager):
    AUX_DATA_HEADER = b"AUX_DATA"

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
    ):
        if disaggregation_mode not in (DisaggregationMode.ENCODE, DisaggregationMode.LANGUAGE):
            raise ValueError(
                "MooncakeEmbeddingManager only supports ENCODE and LANGUAGE disaggregation modes"
            )

        super().__init__(args, disaggregation_mode, server_args, is_mla_backend=False)

        self.engine = MooncakeTransferEngine(
            hostname=self.local_ip,
            gpu_id=self.kv_args.gpu_id,
            ib_device=self.kv_args.ib_device,
        )

        self.register_buffer_to_engine()

        self.transfer_infos: Dict[int, Dict[str, TransferEmbeddingInfo]] = {}
        self.decode_embedding_args_table: Dict[str, EmbeddingArgsRegisterInfo] = {}
        self.failure_records: Dict[int, str] = {}
        self.failure_lock = threading.Lock()
        self.session_lock = threading.Lock()
        self.session_failures = defaultdict(int)
        self.failed_sessions = set()

        self.transfer_queue: "queue.Queue[TransferEmbeddingChunk]" = queue.Queue()
        self.transfer_thread = threading.Thread(
            target=self.transfer_worker, daemon=True
        )
        self.transfer_thread.start()

        self.bootstrap_timeout = get_int_env_var(
            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT", 300
        )
        self.waiting_timeout = get_int_env_var(
            "SGLANG_DISAGGREGATION_WAITING_TIMEOUT", 300
        )

        if self.disaggregation_mode == DisaggregationMode.ENCODE:
            self._bind_server_socket()
            self._start_encode_threads()
        else:
            self._bind_server_socket()
            self._start_language_threads()

    def register_buffer_to_engine(self):
        if self.kv_args.aux_data_ptrs and self.kv_args.aux_data_lens:
            self.engine.batch_register(
                self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
            )

    def _transfer_data(self, mooncake_session_id: str, transfer_blocks):
        if not transfer_blocks:
            return 0
        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
        return self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )

    def transfer_worker(self):
        while True:
            chunk = self.transfer_queue.get()
            try:
                self._transfer_worker_embedding(chunk)
            except Exception as exc:  # pragma: no cover - unexpected failure should surface
                logger.exception("Embedding transfer worker crashed: %s", exc)
                raise

    def _transfer_worker_embedding(self, embedding_chunk: TransferEmbeddingChunk):
        reqs_to_process = (
            self.transfer_infos[embedding_chunk.room].values()
            if embedding_chunk.room in self.transfer_infos
            else []
        )

        polls = []
        dst_ranks_infos = []
        local_rank = self.attn_tp_rank * self.pp_size + self.pp_rank

        for req in reqs_to_process:
            with self.session_lock:
                if req.mooncake_session_id in self.failed_sessions:
                    self.record_failure(
                        embedding_chunk.room,
                        "Language instance could be dead, remote mooncake session is not alive",
                    )
                    self.update_status(embedding_chunk.room, KVPoll.Failed)
                    self.sync_status_to_receiver_endpoint(
                        req.endpoint,
                        req.dst_port,
                        req.room,
                        KVPoll.Failed,
                        local_rank,
                    )
                    return

            if req.src_embedding_indices is None:
                req.src_embedding_indices = embedding_chunk.embedding_indices
                req.total_tokens = embedding_chunk.total_tokens

            block_size = self.kv_args.aux_item_lens[1] // 4
            sent_tokens = req.sent_tokens
            allocated_tokens = req.allocated_tokens

            ret, is_partial = self.send_embedding(
                req.mooncake_session_id,
                embedding_chunk.embedding_indices,
                self.decode_embedding_args_table[req.mooncake_session_id].dst_embedding_ptrs,
                req.dst_embedding_indices,
                embedding_chunk.total_tokens,
                block_size,
                sent_tokens,
                allocated_tokens,
            )

            if ret != 0:
                with self.session_lock:
                    self.session_failures[req.mooncake_session_id] += 1
                    if self.session_failures[req.mooncake_session_id] >= 1:
                        self.failed_sessions.add(req.mooncake_session_id)
                        logger.error(
                            "Session %s failed.", req.mooncake_session_id
                        )
                self.record_failure(
                    embedding_chunk.room,
                    f"Failed to send embedding chunk of {embedding_chunk.room} to {req.endpoint}:{req.dst_port}",
                )
                self.update_status(embedding_chunk.room, KVPoll.Failed)
                self.sync_status_to_receiver_endpoint(
                    req.endpoint,
                    req.dst_port,
                    req.room,
                    KVPoll.Failed,
                    local_rank,
                )
                return

            tokens_sent = min(
                embedding_chunk.total_tokens - sent_tokens, allocated_tokens
            )
            req.sent_tokens += tokens_sent

            polls.append(True if ret == 0 else False)
            dst_ranks_infos.append((req.endpoint, req.dst_port, req.room))

            if len(polls) == req.required_dst_info_num:
                if is_partial:
                    status = KVPoll.Transferring if all(polls) else KVPoll.Failed
                else:
                    status = KVPoll.Success if all(polls) else KVPoll.Failed

                self.update_status(req.room, status)
                for endpoint, dst_port, room in dst_ranks_infos:
                    self.sync_status_to_receiver_endpoint(
                        endpoint, dst_port, room, status, local_rank
                    )

        if (
            embedding_chunk.room not in self.request_status
            or self.check_status(embedding_chunk.room) == KVPoll.Success
        ):
            if embedding_chunk.room in self.transfer_infos:
                self.transfer_infos.pop(embedding_chunk.room)

    def send_embedding(
        self,
        mooncake_session_id: str,
        embedding_indices: List[int],
        dst_embedding_ptrs: List[int],
        dst_embedding_indices: List[int],
        total_tokens: int,
        block_size: int,
        sent_tokens: int = 0,
        allocated_tokens: Optional[int] = None,
    ):
        if allocated_tokens is not None:
            expected_block_size = allocated_tokens // len(dst_embedding_indices or [1])
            if dst_embedding_indices and expected_block_size != block_size:
                raise ValueError(
                    "Block size mismatch between encoder and language allocations"
                )
        else:
            allocated_tokens = len(dst_embedding_indices) * block_size

        remaining_tokens = total_tokens - sent_tokens

        if remaining_tokens > allocated_tokens:
            tokens_to_send = allocated_tokens
            is_partial = True
        else:
            tokens_to_send = remaining_tokens
            is_partial = False

        if block_size == 0:
            raise ValueError("Block size must be positive for embedding transfer")

        dst_blocks_needed = (tokens_to_send + block_size - 1) // block_size
        if dst_blocks_needed > len(dst_embedding_indices):
            raise ValueError(
                "Insufficient destination blocks for requested token transfer"
            )

        start_block = sent_tokens // block_size
        embedding_indices_to_send = embedding_indices[
            start_block : start_block + dst_blocks_needed
        ]
        dst_embedding_indices = dst_embedding_indices[:dst_blocks_needed]

        src_addrs = []
        dst_addrs = []
        lengths = []

        tokens_transferred = 0

        for block_idx, (src_block_idx, dst_block_idx) in enumerate(
            zip(embedding_indices_to_send, dst_embedding_indices)
        ):
            remaining_in_transfer = tokens_to_send - tokens_transferred
            tokens_in_block = min(block_size, remaining_in_transfer)
            if tokens_in_block <= 0:
                break

            for buffer_type_idx in range(len(self.kv_args.aux_item_lens)):
                embedding_item_len = self.kv_args.aux_item_lens[buffer_type_idx]

                if buffer_type_idx == 3:  # aux_datas
                    if sent_tokens == 0 and block_idx == 0:
                        chunk_size = embedding_item_len
                    else:
                        continue
                else:
                    chunk_size = (embedding_item_len * tokens_in_block) // block_size

                embedding_addr = (
                    self.kv_args.aux_data_ptrs[buffer_type_idx]
                    + src_block_idx * embedding_item_len
                )
                dst_embedding_addr = (
                    dst_embedding_ptrs[buffer_type_idx]
                    + dst_block_idx * embedding_item_len
                )

                src_addrs.append(embedding_addr)
                dst_addrs.append(dst_embedding_addr)
                lengths.append(chunk_size)

            tokens_transferred += tokens_in_block

        ret = self.engine.batch_transfer_sync(
            mooncake_session_id, src_addrs, dst_addrs, lengths
        )

        return ret, is_partial

    def add_transfer_request(
        self,
        bootstrap_room: int,
        embedding_indices: List[int],
        total_tokens: int,
        is_last: bool = True,
    ):
        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Embedding request with bootstrap_room=%s already failed",
                bootstrap_room,
            )
            return

        if bootstrap_room not in self.transfer_infos:
            return

        self.transfer_queue.put(
            TransferEmbeddingChunk(
                room=bootstrap_room,
                embedding_indices=embedding_indices,
                is_last=is_last,
                total_tokens=total_tokens,
            )
        )

    def check_status(self, bootstrap_room: int):
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
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

    def sync_status_to_receiver_endpoint(
        self, remote: str, dst_port: int, room: int, status: int, sender_rank: int = -1
    ):
        self._connect(
            format_tcp_address(remote, dst_port), is_ipv6=is_valid_ipv6_address(remote)
        ).send_multipart(
            [
                str(room).encode("ascii"),
                str(status).encode("ascii"),
                str(sender_rank).encode("ascii"),
            ]
        )

    def _start_encode_threads(self):
        def bootstrap_thread():
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                room = waiting_req_bytes[0].decode("ascii")
                mooncake_session_id = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    self.decode_embedding_args_table[mooncake_session_id] = (
                        EmbeddingArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    )
                    with self.session_lock:
                        if mooncake_session_id in self.failed_sessions:
                            self.failed_sessions.remove(mooncake_session_id)
                        if mooncake_session_id in self.session_failures:
                            del self.session_failures[mooncake_session_id]
                    logger.debug(
                        "Register EmbeddingArgs from %s successfully",
                        mooncake_session_id,
                    )
                    continue
                room = int(room)
                is_resume = len(waiting_req_bytes) >= 8

                if is_resume:
                    if (
                        room in self.transfer_infos
                        and mooncake_session_id in self.transfer_infos[room]
                    ):
                        transfer_info = TransferEmbeddingInfo.from_zmq(
                            waiting_req_bytes
                        )
                        req = self.transfer_infos[room][mooncake_session_id]
                        req.sent_tokens = transfer_info.sent_tokens
                        req.allocated_tokens = transfer_info.allocated_tokens
                        req.dst_embedding_indices = (
                            transfer_info.dst_embedding_indices
                        )
                        req.resume_ready = True

                        all_ready = all(
                            dst_req.resume_ready
                            for dst_req in self.transfer_infos[room].values()
                        )
                        if all_ready:
                            if (
                                req.src_embedding_indices is not None
                                and req.total_tokens > 0
                            ):
                                self.transfer_queue.put(
                                    TransferEmbeddingChunk(
                                        room=room,
                                        embedding_indices=req.src_embedding_indices,
                                        is_last=True,
                                        total_tokens=req.total_tokens,
                                    )
                                )
                                for dst_req in self.transfer_infos[room].values():
                                    dst_req.resume_ready = False
                            else:
                                logger.error(
                                    "Cannot trigger resume: missing src_embedding_indices or total_tokens for room=%s session=%s",
                                    room,
                                    mooncake_session_id,
                                )
                    else:
                        logger.error(
                            "Cannot resume: room=%s session=%s not found in transfer_infos",
                            room,
                            mooncake_session_id,
                        )
                else:
                    required_dst_info_num = int(waiting_req_bytes[5].decode("ascii"))
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}

                    self.transfer_infos[room][mooncake_session_id] = (
                        TransferEmbeddingInfo.from_zmq(waiting_req_bytes)
                    )
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread, daemon=True).start()

    def _start_language_threads(self):
        def decode_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                if msg[0] == MooncakeEmbeddingManager.AUX_DATA_HEADER:
                    self._handle_aux_data(msg)
                    continue

                (bootstrap_room, status, prefill_rank) = msg
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                prefill_rank = int(prefill_rank.decode("ascii"))

                if status in [KVPoll.Success, KVPoll.Transferring]:
                    if bootstrap_room in self.request_status:
                        self.update_status(bootstrap_room, status)
                elif status == KVPoll.Failed:
                    self.record_failure(
                        bootstrap_room,
                        "Failed to get embeddings from encode instance, it might be dead",
                    )
                    self.update_status(bootstrap_room, status)

        threading.Thread(target=decode_thread, daemon=True).start()

    def _handle_aux_data(self, msg: List[bytes]):
        room = int(msg[1].decode("ascii"))
        buffer_index = int(msg[2].decode("ascii"))
        aux_index = int(msg[3].decode("ascii"))
        data_length = struct.unpack(">I", msg[4])[0]
        data = msg[5]

        if len(data) != data_length:
            logger.error(
                "AUX_DATA length mismatch for bootstrap_room %s", room
            )
            return

        AuxDataCodec.deserialize_data_to_buffer(
            self.kv_args, buffer_index, aux_index, data
        )

        logger.debug(
            "Received AUX_DATA for bootstrap_room %s with length:%s",
            room,
            len(data),
        )


class MooncakeEmbeddingSender(CommonKVSender):
    def __init__(
        self,
        mgr: MooncakeEmbeddingManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.conclude_state = None
        self.init_time = time.time()
        self.embedding_indices: Optional[List[int]] = None

    def init(
        self,
        embedding_indices: Optional[List[int]] = None,
    ):
        self.embedding_indices = embedding_indices
        self.init_time = time.time()

    def send_embedding(
        self,
        embedding_indices: List[int],
        last_chunk: bool = True,
        total_tokens: int = None,
        block_size: int = None,
    ):
        self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            embedding_indices=embedding_indices,
            total_tokens=total_tokens,
            is_last=last_chunk,
        )

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.Bootstrapping and self.init_time is not None:
                elapsed = time.time() - self.init_time
                if elapsed >= self.kv_mgr.bootstrap_timeout:
                    logger.warning(
                        "Embedding request %s timed out during bootstrapping",
                        self.bootstrap_room,
                    )
                    self.kv_mgr.record_failure(
                        self.bootstrap_room,
                        f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.Bootstrapping",
                    )
                    self.conclude_state = KVPoll.Failed
                    return KVPoll.Failed
            return status
        return self.conclude_state

    def clear(self) -> None:
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)
        if self.bootstrap_room in self.kv_mgr.transfer_infos:
            self.kv_mgr.transfer_infos.pop(self.bootstrap_room)

    def failure_exception(self):
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )

        raise EmbeddingTransferError(self.bootstrap_room, failure_reason)

    def abort(self):
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        self.conclude_state = KVPoll.Failed


class MooncakeEmbeddingReceiver(CommonKVReceiver):
    _ctx = CommonKVReceiver._ctx
    _socket_cache = CommonKVReceiver._socket_cache
    _socket_locks = CommonKVReceiver._socket_locks
    _global_lock = CommonKVReceiver._global_lock

    def __init__(
        self,
        mgr: MooncakeEmbeddingManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        prefill_dp_rank: Optional[int] = None,
    ):
        self.session_id = mgr.get_session_id()
        self.conclude_state = None
        self.init_time = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)

        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def _register_kv_args(self):
        for bootstrap_info in self.bootstrap_infos:
            messages = [
                "None".encode("ascii"),
                self.kv_mgr.local_ip.encode("ascii"),
                str(self.kv_mgr.rank_port).encode("ascii"),
                self.session_id.encode("ascii"),
            ]

            packed_embedding_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )
            messages.append(packed_embedding_data_ptrs)

            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(messages)

    def init(
        self,
        embedding_indices: Optional[List[int]] = None,
        allocated_tokens: Optional[int] = None,
    ):
        for bootstrap_info in self.bootstrap_infos:
            messages = [
                str(self.bootstrap_room).encode("ascii"),
                self.kv_mgr.local_ip.encode("ascii"),
                str(self.kv_mgr.rank_port).encode("ascii"),
                self.session_id.encode("ascii"),
            ]

            embedding_indices_str = (
                ",".join(str(idx) for idx in embedding_indices)
                if embedding_indices is not None
                else ""
            )

            if allocated_tokens is None and embedding_indices is not None:
                block_size = self.kv_mgr.kv_args.aux_item_lens[1] // 4
                allocated_tokens = len(embedding_indices) * block_size

            messages.extend(
                [
                    embedding_indices_str.encode("ascii"),
                    str(self.required_dst_info_num).encode("ascii"),
                    str(allocated_tokens or 0).encode("ascii"),
                ]
            )

            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(messages)
        self.init_time = time.time()

    def resume_transfer(
        self,
        embedding_indices: List[int],
        sent_tokens: int,
        allocated_tokens: int,
    ):
        embedding_indices_str = ",".join(str(idx) for idx in embedding_indices)
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(
                    [
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        embedding_indices_str.encode("ascii"),
                        str(self.required_dst_info_num).encode("ascii"),
                        str(sent_tokens).encode("ascii"),
                        str(allocated_tokens).encode("ascii"),
                    ]
                )

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.WaitingForInput and self.init_time is not None:
                elapsed = time.time() - self.init_time
                if elapsed >= self.kv_mgr.waiting_timeout:
                    logger.warning(
                        "Embedding request %s timed out waiting for input",
                        self.bootstrap_room,
                    )
                    self.kv_mgr.record_failure(
                        self.bootstrap_room,
                        f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.WaitingForInput",
                    )
                    self.conclude_state = KVPoll.Failed
                    return KVPoll.Failed
            return status
        return self.conclude_state

    def clear(self) -> None:
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)

    def failure_exception(self):
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )

        raise EmbeddingTransferError(self.bootstrap_room, failure_reason)

    def abort(self):
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        self.conclude_state = KVPoll.Failed


class MooncakeEmbeddingBootstrapServer(CommonKVBootstrapServer):
    pass


__all__ = [
    "EmbeddingTransferError",
    "MooncakeEmbeddingBootstrapServer",
    "MooncakeEmbeddingManager",
    "MooncakeEmbeddingReceiver",
    "MooncakeEmbeddingSender",
]
