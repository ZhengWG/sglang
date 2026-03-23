import asyncio
import itertools
import logging
import pickle
import random
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from enum import IntEnum
from http import HTTPStatus
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import aiohttp
import torch
import zmq
import zmq.asyncio

from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    get_mooncake_transfer_engine,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.schedule_batch import Modality, Req
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import ImageData
from sglang.srt.utils.hf_transformers_utils import get_processor
from sglang.srt.utils.network import get_local_ip_auto, get_zmq_socket_on_host
from transformers import PretrainedConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


def _grpc_target(url: str) -> str:
    if url.startswith("grpc://"):
        return url[len("grpc://") :]
    if url.startswith("grpcs://"):
        raise ValueError("grpcs:// is not supported; use grpc://")
    return url


def _normalize_embedding_ports(embedding_port):
    if embedding_port is None:
        return []
    if isinstance(embedding_port, list):
        return embedding_port
    return [embedding_port]


def _grpc_scheduler_receive_url(target, req_id, receive_url, receive_count):
    import grpc
    from smg_grpc_proto import sglang_encoder_pb2, sglang_encoder_pb2_grpc

    timeout_secs = envs.SGLANG_ENCODER_GRPC_TIMEOUT_SECS.get()
    channel = grpc.insecure_channel(target)
    stub = sglang_encoder_pb2_grpc.SglangEncoderStub(channel)
    try:
        stub.SchedulerReceiveUrl(
            sglang_encoder_pb2.SchedulerReceiveUrlRequest(
                req_id=req_id,
                receive_url=receive_url,
                receive_count=receive_count,
            ),
            timeout=timeout_secs,
        )
    finally:
        channel.close()


def _grpc_encode_request(target, encode_request):
    import grpc
    from smg_grpc_proto import sglang_encoder_pb2, sglang_encoder_pb2_grpc

    timeout_secs = envs.SGLANG_ENCODER_GRPC_TIMEOUT_SECS.get()
    channel = grpc.insecure_channel(target)
    stub = sglang_encoder_pb2_grpc.SglangEncoderStub(channel)
    try:
        response = stub.Encode(
            sglang_encoder_pb2.EncodeRequest(
                mm_items=encode_request["mm_items"],
                req_id=encode_request["req_id"],
                num_parts=encode_request["num_parts"],
                part_idx=encode_request["part_idx"],
                prefill_host=encode_request["prefill_host"],
                embedding_port=_normalize_embedding_ports(
                    encode_request["embedding_port"]
                ),
            ),
            timeout=timeout_secs,
        )
        return response
    finally:
        channel.close()


def _grpc_send_request(target, request_json):
    import grpc
    from smg_grpc_proto import sglang_encoder_pb2, sglang_encoder_pb2_grpc

    timeout_secs = envs.SGLANG_ENCODER_GRPC_TIMEOUT_SECS.get()
    channel = grpc.insecure_channel(target)
    stub = sglang_encoder_pb2_grpc.SglangEncoderStub(channel)
    try:
        stub.Send(
            sglang_encoder_pb2.SendRequest(
                req_id=request_json["req_id"],
                prefill_host=request_json["prefill_host"],
                embedding_port=request_json["embedding_port"],
                session_id=request_json["session_id"],
                buffer_address=request_json["buffer_address"],
            ),
            timeout=timeout_secs,
        )
    finally:
        channel.close()


class EmbeddingData:
    def __init__(
        self,
        req_id,
        num_parts,
        part_idx,
        grid_dim,
        modality,
        embedding=None,
        embedding_shape=None,
        error_msg=None,
        error_code=None,
        **kwargs,
    ):
        self.req_id = req_id
        self.num_parts = num_parts
        self.part_idx = part_idx
        self.grid_dim = grid_dim
        self.modality = modality
        self.embedding = embedding
        self.send_time = None
        self.dtype = embedding.dtype if embedding is not None else None
        if embedding_shape is not None:
            self.shape = embedding_shape
        else:
            self.shape = list(embedding.shape) if embedding is not None else None
        self.error_msg = error_msg
        self.error_code = error_code
        # Store additional metadata (e.g., video_timestamps for qwen3_vl)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_grid(self):
        """Get the grid dimension of the embedding, used for image/video/audio."""
        return self.grid_dim

    def get_embedding(self):
        return self.embedding

    def __repr__(self):
        return f"EmbeddingData(req_id={self.req_id}, num_parts={self.num_parts}, part_idx={self.part_idx}) error_msg={self.error_msg}"

    def copy_without_embedding(self):
        new_data = EmbeddingData(
            req_id=self.req_id,
            num_parts=self.num_parts,
            part_idx=self.part_idx,
            grid_dim=self.grid_dim,
            modality=self.modality,
            embedding=None,
            embedding_shape=self.shape,
            error_msg=self.error_msg,
            error_code=self.error_code,
        )
        for key, value in self.__dict__.items():
            if key.startswith("_") or key == "embedding":
                continue
            setattr(new_data, key, value)
        return new_data


# Modality -> (list attr name, whether to flatten grid for that list)
_MODALITY_GRID_ATTRS = {
    Modality.IMAGE: ("img_grid_thw", False),
    Modality.VIDEO: ("video_grid_thw", False),
    Modality.AUDIO: ("audio_feature_lens", True),
}
_VIDEO_META_ATTRS = ("video_timestamps", "second_per_grid_ts")


def _cat_grid(dims, flatten_items=False):
    """Concatenate non-None tensors from a list; optionally flatten each before cat."""
    valid = (
        [g.flatten() for g in dims if g is not None]
        if flatten_items
        else [g for g in dims if g is not None]
    )
    return torch.cat(valid, dim=0) if valid else None


class MultiModalEmbeddingData(EmbeddingData):
    def __init__(
        self,
        part_idx,
        num_parts,
        req_id,
        grid_dim,
        modality,
        embedding,
        embedding_shape,
        **kwargs,
    ):
        super().__init__(
            req_id,
            num_parts,
            part_idx,
            grid_dim,
            modality,
            embedding,
            embedding_shape,
            **kwargs,
        )
        self.img_grid_thw = [None] * num_parts
        self.video_grid_thw = [None] * num_parts
        self.audio_feature_lens = [None] * num_parts
        self.modality_list = [
            modality if part_idx == i else None for i in range(num_parts)
        ]
        self.ready_list = [i == part_idx for i in range(num_parts)]
        self.embedding_list = [
            embedding if i == part_idx else None for i in range(num_parts)
        ]
        self.embedding_shape_list = [
            embedding_shape if i == part_idx else None for i in range(num_parts)
        ]
        self.video_timestamps = [None] * num_parts
        self.second_per_grid_ts = [None] * num_parts

        self._set_part_grid(part_idx, modality, self.get_grid())
        if modality == Modality.VIDEO:
            self._set_video_meta_for_part(part_idx, kwargs)

    def _set_part_grid(self, part_idx, modality, grid):
        """Set the grid for one part according to modality (IMAGE/VIDEO/AUDIO)."""
        spec = _MODALITY_GRID_ATTRS.get(modality)
        if spec is None:
            raise ValueError(f"Invalid modality: {modality}")
        attr_name, flatten = spec
        value = grid.flatten() if flatten else grid
        getattr(self, attr_name)[part_idx] = value

    def _set_video_meta_for_part(self, part_idx, source):
        """Copy video_timestamps and second_per_grid_ts from source (dict or object)."""
        for attr_name in _VIDEO_META_ATTRS:
            val = (
                source.get(attr_name)
                if isinstance(source, dict)
                else getattr(source, attr_name, None)
            )
            if val is not None:
                getattr(self, attr_name)[part_idx] = val

    @classmethod
    def from_embedding_data(cls, embedding_data: EmbeddingData):
        """Create MultiModalEmbeddingData from an EmbeddingData instance."""
        # Only forward known optional attrs (e.g. video metadata) so they land on the instance
        extra = {}
        for attr in _VIDEO_META_ATTRS:
            val = getattr(embedding_data, attr, None)
            if val is not None:
                extra[attr] = val
        mm_data = cls(
            part_idx=embedding_data.part_idx,
            num_parts=embedding_data.num_parts,
            req_id=embedding_data.req_id,
            grid_dim=embedding_data.grid_dim,
            modality=embedding_data.modality,
            embedding=embedding_data.embedding,
            embedding_shape=embedding_data.shape,
            **extra,
        )
        mm_data.send_time = embedding_data.send_time
        return mm_data

    def __repr__(self):
        return f"MultiModalEmbeddingData(req_id={self.req_id}, num_parts={self.num_parts}, part_idx={self.part_idx}, modality={self.modality})"

    def get_embedding(self, is_concat=False):
        if is_concat:
            groups = defaultdict(list)
            for i, e in enumerate(self.embedding_list):
                if e is not None:
                    groups[self.modality_list[i]].append(e.cuda())
            return {
                mod: torch.concat(tensors).to("cpu", non_blocking=True)
                for mod, tensors in groups.items()
            }
        return self.embedding_list

    @property
    def ready(self):
        return sum(self.ready_list) == self.num_parts

    def get_mm_extra_meta(self):
        """Build kwargs for mm_processor.get_mm_data() from grid and optional video meta."""
        kwargs = {
            "img_grid_thw": _cat_grid(self.img_grid_thw),
            "video_grid_thw": _cat_grid(self.video_grid_thw),
            "audio_feature_lens": _cat_grid(
                self.audio_feature_lens, flatten_items=True
            ),
        }
        for attr in _VIDEO_META_ATTRS:
            lst = getattr(self, attr, None)
            if not lst:
                continue
            valid = [a for a in lst if a is not None]
            if valid:
                kwargs[attr] = list(itertools.chain(*valid))
        return kwargs

    def add(self, embedding_data: EmbeddingData):
        assert self.req_id == embedding_data.req_id
        assert not self.ready_list[embedding_data.part_idx]
        pid = embedding_data.part_idx
        self.ready_list[pid] = True
        self.modality_list[pid] = embedding_data.modality
        self.embedding_list[pid] = embedding_data.get_embedding()
        self.embedding_shape_list[pid] = embedding_data.shape
        self._set_part_grid(pid, embedding_data.modality, embedding_data.get_grid())
        if embedding_data.modality == Modality.VIDEO:
            self._set_video_meta_for_part(pid, embedding_data)


class WaitingImageRequestStatus(IntEnum):
    FAIL = -1
    PENDING = 0
    SUCCESS = 1
    TIMEOUT = -2


def create_part_req_id(original_req_id: str, part_idx: int) -> str:
    """Create a unique part request ID by appending part index suffix."""
    return f"{original_req_id}_local_part_{part_idx}"


def extract_original_req_id(part_req_id: str) -> str:
    """Extract the original request ID from a part request ID."""
    if "_local_part_" in part_req_id:
        return part_req_id.rsplit("_local_part_", 1)[0]
    return part_req_id


def calculate_modality_num_parts(modalities, num_items_assigned):
    """
    Calculate total number of parts and number of parts per modality.

    Args:
        modalities: List of modalities in order
        num_items_assigned: Dictionary mapping modality to list of assignment counts per encoder

    Returns:
        Tuple of (total_num_parts, modality_num_parts_dict)
        - total_num_parts: Total number of parts across all modalities
        - modality_num_parts: Dictionary mapping modality to number of parts for that modality
    """
    total_num_parts = 0
    modality_num_parts = {}
    for modality in modalities:
        num_items_assigned_modality = num_items_assigned.get(modality)
        num_parts = sum(1 for x in num_items_assigned_modality if x != 0)
        modality_num_parts[modality] = num_parts
        total_num_parts += num_parts
    return total_num_parts, modality_num_parts


# For zmq backend
class WaitingImageRequest:
    def __init__(
        self,
        rid: str,
        recv_req: TokenizedGenerateReqInput,
        mm_processor,
        encoder_urls,
        host_name,
        receive_count,
    ):
        self.rid = rid
        self.recv_req = recv_req
        self.mm_inputs = None
        self.error = None
        self.thread = None
        self.mm_processor = mm_processor
        self.encoder_urls = encoder_urls
        self.host_name = host_name
        self.receive_count = receive_count
        self.num_items_assigned = recv_req.num_items_assigned
        self.embedding_port, self.recv_socket = get_zmq_socket_on_host(
            zmq.Context(), zmq.PULL
        )
        logger.info(f"Waiting for input {self.embedding_port = }")
        self.recv_embedding_data = None
        # ok=1 pending=0 fail=-1
        self.status = WaitingImageRequestStatus.PENDING
        self.error_msg = None
        self.error_code = None
        self.start_time = time.time()

    def send_encode_request(self):
        async def _send_single_request(session, url, payload):
            try:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    return await response.text()
            except Exception as e:
                logger.error(f"Failed to send request to {url}: {e}")
                raise

        async def send_embedding_port(req_id, receive_count, host_name, embedding_port):
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=1800)
            ) as session:
                tasks = []
                logger.info(f"{self.num_items_assigned = } ")

                # Calculate part_idx_offset similar to encode() method
                modalities = list(self.num_items_assigned.keys())
                _, modality_num_parts = calculate_modality_num_parts(
                    modalities, self.num_items_assigned
                )

                part_idx_offset = 0
                for modality in modalities:
                    assigned_nums = self.num_items_assigned[modality]
                    num_parts = modality_num_parts[modality]
                    cum_idx = 0
                    for idx, assigned_num in enumerate(assigned_nums):
                        if assigned_num == 0:
                            continue
                        part_idx = part_idx_offset + cum_idx
                        part_req_id = create_part_req_id(req_id, part_idx)
                        encoder_url = self.encoder_urls[idx]
                        target_url = f"{encoder_url}/scheduler_receive_url"
                        payload = {
                            "req_id": part_req_id,  # use part_req_id to match encode request
                            "receive_count": receive_count,
                            "receive_url": f"{host_name}:{embedding_port}",
                            "modality": modality.name,
                        }
                        logger.info(
                            f"Preparing to send to {target_url} with part_req_id={part_req_id}"
                        )
                        task = _send_single_request(session, target_url, payload)
                        tasks.append(task)
                        cum_idx += 1
                    part_idx_offset += num_parts

                if not tasks:
                    logger.info("No tasks to send.")
                    return
                logger.info(f"Concurrently sending {len(tasks)} requests...")
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Request {i} failed: {result}")
                    else:
                        logger.debug(f"Request {i} succeeded.")

        asyncio.run(
            send_embedding_port(
                self.recv_req.rid,
                self.receive_count,
                self.host_name,
                self.embedding_port,
            )
        )

    def _try_recv_mm_data(self):
        if self.status != WaitingImageRequestStatus.PENDING:
            return
        while self.recv_embedding_data is None or not self.recv_embedding_data.ready:
            try:
                parts = self.recv_socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
            except zmq.Again:
                # No data available yet, wait a bit and retry
                return
            recv_obj: EmbeddingData = pickle.loads(parts[0])
            if getattr(recv_obj, "error_msg", None) is not None:
                logger.warning(
                    f"Received error signal from encoder for {self.rid}: {recv_obj.error_msg} {recv_obj.error_code = }"
                )
                self.error_msg = recv_obj.error_msg
                self.error_code = recv_obj.error_code
                self.status = WaitingImageRequestStatus.FAIL
                self.recv_socket.close()
                return

            buffer = parts[1].buffer if hasattr(parts[1], "buffer") else parts[1]
            recv_obj.embedding = (
                torch.frombuffer(buffer, dtype=recv_obj.dtype)
                .reshape(recv_obj.shape)
                .clone()
            )

            # Extract original req_id from part_req_id
            part_req_id = recv_obj.req_id
            original_req_id = extract_original_req_id(part_req_id)
            # Update recv_obj.req_id to original for aggregation
            recv_obj.req_id = original_req_id

            if self.recv_embedding_data is None:
                self.recv_embedding_data = MultiModalEmbeddingData.from_embedding_data(
                    recv_obj
                )
            else:
                self.recv_embedding_data.add(recv_obj)

        recv_embedding = self.recv_embedding_data.get_embedding(is_concat=True)
        mm_inputs = self.mm_processor.get_mm_data(
            self.recv_req.input_text,
            recv_embedding,
            **self.recv_embedding_data.get_mm_extra_meta(),
        )
        self.recv_req.mm_inputs = mm_inputs
        self.recv_req.input_ids = mm_inputs["input_ids"]
        self.status = WaitingImageRequestStatus.SUCCESS
        self.recv_socket.close()


class WaitingImageRequestGrpc(WaitingImageRequest):
    def send_encode_request(self):
        async def send_embedding_port(req_id, receive_count, host_name, embedding_port):
            tasks = []
            # gRPC image-only: flatten modality dict to flat list
            assigned = list(self.num_items_assigned.values())[0]
            logger.info(f"num_items_assigned={assigned}")

            for idx, assigned_num in enumerate(assigned):
                if assigned_num == 0:
                    continue
                encoder_url = self.encoder_urls[idx]
                receive_url = f"{host_name}:{embedding_port}"
                target_url = f"{encoder_url}/SchedulerReceiveUrl"
                logger.info(f"Preparing to send to {target_url}")
                tasks.append(
                    asyncio.to_thread(
                        _grpc_scheduler_receive_url,
                        _grpc_target(encoder_url),
                        req_id,
                        receive_url,
                        receive_count,
                    )
                )

            if not tasks:
                logger.info("No tasks to send.")
                return
            logger.info(f"Concurrently sending {len(tasks)} requests...")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Request {i} failed: {result}")
                else:
                    logger.debug(f"Request {i} succeeded.")

        asyncio.run(
            send_embedding_port(
                self.recv_req.rid,
                self.receive_count,
                self.host_name,
                self.embedding_port,
            )
        )


class WaitingImageRequestMooncake:
    """Mooncake-based waiting request: GPU-direct RDMA transfer, no D2H/H2D overhead.

    For the mooncake encoder transfer backend, the scheduler pre-allocates a GPU
    buffer, registers it with Mooncake, and calls /send on the encoder to initiate
    GPU-direct RDMA. This overlaps VIT computation with buffer allocation.
    """

    def __init__(
        self,
        rid: str,
        recv_req: "TokenizedGenerateReqInput",
        mm_processor,
        encoder_urls,
        host_name,
        receive_count,
        tp_rank: int = 0,
        gpu_id: int = 0,
        embeddings_engine=None,
        dtype=None,
        embedding_meta=None,
        embeddings_buffer_dict=None,
    ):
        self.rid = rid
        self.recv_req = recv_req
        self.mm_processor = mm_processor
        self.encoder_urls = encoder_urls
        self.host_name = host_name
        self.receive_count = receive_count
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.embeddings_engine = embeddings_engine
        self.embeddings_buffer_dict = embeddings_buffer_dict
        self.dtype = dtype
        self.embedding_meta = embedding_meta or []
        self.recv_socket = None
        self.recv_embedding_data = None
        self.status = WaitingImageRequestStatus.PENDING
        self.error_msg = None
        self.error_code = None
        self.start_time = time.time()
        self._send_done = False

    def send_encode_request(self):
        """Start background thread to allocate GPU buffer, register, and send /send."""
        threading.Thread(
            target=self._run_send_embedding_port,
            daemon=True,
        ).start()

    def _run_send_embedding_port(self):
        """Allocate GPU buffer, register with mooncake, send /send to encoders."""
        try:
            torch.cuda.set_device(self.gpu_id)
            asyncio.run(self._send_embedding_port())
        except Exception as e:
            logger.error(f"Mooncake send failed for {self.rid}: {e}", exc_info=True)
            self._cleanup()
            self.error_msg = str(e)
            self.error_code = HTTPStatus.INTERNAL_SERVER_ERROR
            self.status = WaitingImageRequestStatus.FAIL

    async def _send_embedding_port(self):
        meta_list = self.embedding_meta
        num_parts = len(meta_list)
        if num_parts == 0:
            logger.warning(f"No embedding metadata for {self.rid}, skipping send")
            self._send_done = True
            return

        embedding_length_tot = sum(m["embedding_len"] for m in meta_list)
        embedding_dim = meta_list[0]["embedding_dim"]
        embedding_size_list = [m["embedding_size"] for m in meta_list]

        embeddings = torch.zeros(
            (embedding_length_tot, embedding_dim),
            dtype=self.dtype,
            device=f"cuda:{self.gpu_id}",
        )
        self.embeddings_engine.register(embeddings.data_ptr(), embeddings.nbytes)
        self.embeddings_buffer_dict[self.rid] = embeddings
        buffer_address = embeddings.data_ptr()

        zmq_port, zmq_socket = get_zmq_socket_on_host(zmq.Context(), zmq.PULL)
        self.recv_socket = zmq_socket

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=1800)
        ) as session:
            offset = 0
            send_tasks = []
            for idx in range(num_parts):
                meta = meta_list[idx]
                send_payload = dict(meta)
                send_payload.update(
                    {
                        "session_id": self.embeddings_engine.session_id,
                        "buffer_address": offset + buffer_address,
                        "embedding_port": zmq_port,
                        "prefill_host": self.host_name,
                        "receive_count": self.receive_count,
                    }
                )
                send_tasks.append(
                    session.post(
                        f"{self.encoder_urls[meta['encoder_idx']]}/send",
                        json=send_payload,
                    )
                )
                offset += embedding_size_list[idx]
            await asyncio.gather(*send_tasks)

        self._send_done = True

    def _try_recv_mm_data(self):
        """Poll for ZMQ ack from encoder (metadata only, tensor already on GPU via RDMA)."""
        if self.status != WaitingImageRequestStatus.PENDING:
            return
        if not self._send_done or self.recv_socket is None:
            return
        if self.rid not in self.embeddings_buffer_dict:
            return
        while self.recv_embedding_data is None or not self.recv_embedding_data.ready:
            try:
                parts = self.recv_socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
            except zmq.Again:
                return
            recv_obj: EmbeddingData = pickle.loads(parts[0])
            if getattr(recv_obj, "error_msg", None) is not None:
                logger.warning(
                    f"Received error from encoder for {self.rid}: "
                    f"{recv_obj.error_msg} {recv_obj.error_code = }"
                )
                self.error_msg = recv_obj.error_msg
                self.error_code = recv_obj.error_code
                self.status = WaitingImageRequestStatus.FAIL
                self._cleanup()
                return

            # Normalize req_id (part req_id → original req_id)
            recv_obj.req_id = extract_original_req_id(recv_obj.req_id)

            if self.recv_embedding_data is None:
                self.recv_embedding_data = MultiModalEmbeddingData.from_embedding_data(
                    recv_obj
                )
            else:
                self.recv_embedding_data.add(recv_obj)

        recv_embedding = self.embeddings_buffer_dict.pop(self.rid)
        self.embeddings_engine.deregister(recv_embedding.data_ptr())

        # Set GPU embedding slices into the aggregated metadata
        # recv_embedding is shaped (total_tokens, embedding_dim) with the correct dtype
        token_offset = 0
        for i in range(self.recv_embedding_data.num_parts):
            shape = self.recv_embedding_data.embedding_shape_list[i]
            if shape is None:
                continue
            num_tokens = shape[0]
            self.recv_embedding_data.embedding_list[i] = recv_embedding[
                token_offset : token_offset + num_tokens
            ]
            token_offset += num_tokens

        # Concat all parts on GPU
        recv_embedding_final = torch.concat(
            [t for t in self.recv_embedding_data.embedding_list if t is not None]
        )

        mm_inputs = self.mm_processor.get_mm_data(
            self.recv_req.input_text,
            recv_embedding_final,
            **self.recv_embedding_data.get_mm_extra_meta(),
        )
        self.recv_req.mm_inputs = mm_inputs
        self.recv_req.input_ids = mm_inputs["input_ids"]

        self.status = WaitingImageRequestStatus.SUCCESS
        self.recv_socket.close()
        self.recv_embedding_data = None
        self.embeddings_engine = None
        self.embeddings_buffer_dict = None
        self.mm_processor = None

    def _cleanup(self):
        if self.embeddings_buffer_dict and self.rid in self.embeddings_buffer_dict:
            buf = self.embeddings_buffer_dict.pop(self.rid)
            self.embeddings_engine.deregister(buf.data_ptr())
        if self.recv_socket is not None:
            self.recv_socket.close()


def _determine_tensor_transport_mode(server_args):
    is_cross_node = server_args.dist_init_addr

    if is_cross_node:
        # Fallback to default CPU transport for multi-node
        return "default"
    else:
        return "cuda_ipc"


class MMReceiverBase(ABC):
    def __init__(
        self,
        server_args: ServerArgs,
        dtype: Optional[torch.dtype] = None,
        hf_config: Optional[PretrainedConfig] = None,
        pp_rank: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_group: Optional[GroupCoordinator] = None,
        scheduler: Optional["Scheduler"] = None,
    ):
        self.context = zmq.asyncio.Context(20)
        self.encoder_transfer_backend = server_args.encoder_transfer_backend
        self.encode_urls = server_args.encoder_urls
        self.host = get_local_ip_auto(server_args.host)
        if self.encoder_transfer_backend in ("zmq", "mooncake"):
            self.pp_rank = pp_rank
            self.tp_rank = tp_rank
            self.tp_size = server_args.tp_size
            self.tp_group = tp_group
            self.nnodes = server_args.nnodes
            self.hostname = get_local_ip_auto()
            self.waiting_list: List[
                Union[WaitingImageRequest, WaitingImageRequestMooncake]
            ] = []
            self.scheduler = scheduler
            self.wait_timeout = envs.SGLANG_ENCODER_RECV_TIMEOUT.get()
            if self.encoder_transfer_backend == "mooncake":
                self.dtype = dtype
                self.embeddings_engine = get_mooncake_transfer_engine()
                self.embeddings_buffer = dict()
            if hf_config is not None:
                transport_mode = _determine_tensor_transport_mode(server_args)
                import_processors("sglang.srt.multimodal.processors")
                _processor = None
                try:
                    _processor = get_processor(
                        server_args.tokenizer_path,
                        tokenizer_mode=server_args.tokenizer_mode,
                        trust_remote_code=server_args.trust_remote_code,
                        revision=server_args.revision,
                        use_fast=not server_args.disable_fast_image_processor,
                    )
                except ValueError as e:
                    error_message = str(e)
                    if "does not have a slow version" in error_message:
                        logger.info(
                            f"Processor {server_args.tokenizer_path} does not have a slow version. Automatically use fast version"
                        )
                        _processor = get_processor(
                            server_args.tokenizer_path,
                            tokenizer_mode=server_args.tokenizer_mode,
                            trust_remote_code=server_args.trust_remote_code,
                            revision=server_args.revision,
                            use_fast=True,
                        )
                    else:
                        raise e

                # Skip mm_pool if not adaptive dispatch to encoder
                enable_adaptive_dispatch_to_encoder = (
                    server_args.enable_adaptive_dispatch_to_encoder
                )
                self.mm_processor = get_mm_processor(
                    hf_config,
                    server_args,
                    _processor,
                    transport_mode,
                    skip_mm_pool=not enable_adaptive_dispatch_to_encoder,
                )

    @abstractmethod
    def process_waiting_requests(self, recv_reqs):
        pass

    def send_encode_request(self, obj):
        self._send_encode_request(obj)

    def _send_encode_request(self, obj):
        mm_data = self._extract_url_data(obj)
        if obj.rid is None:
            obj.rid = uuid.uuid4().hex
        if mm_data and self.encode_urls:
            logger.info(f"Processing {len(mm_data)} mm items for request {obj.rid}")
            obj.need_wait_for_mm_inputs = True

            num_items_assigned = self._assign_items_by_modality(
                mm_data, len(self.encode_urls)
            )
            obj.num_items_assigned = num_items_assigned
            if self.encoder_transfer_backend == "zmq":
                encode_thread = threading.Thread(
                    target=self._run_encode_in_thread,
                    args=(
                        obj.rid,
                        mm_data,
                        "encode",
                        num_items_assigned,
                        None,
                    ),
                    daemon=True,
                )
                encode_thread.start()
            # For mooncake backend, metadata is fetched separately via wait_mm_metadata

    # For zmq and mooncake backends
    def _process_waiting_requests(self, recv_reqs):
        new_recv_reqs = []
        for recv_req in recv_reqs:
            if (
                isinstance(recv_req, TokenizedGenerateReqInput)
                and recv_req.need_wait_for_mm_inputs is True
            ):
                if self.encoder_transfer_backend == "mooncake":
                    waiting_req = WaitingImageRequestMooncake(
                        rid=recv_req.rid,
                        recv_req=recv_req,
                        mm_processor=self.mm_processor,
                        encoder_urls=self.encode_urls,
                        host_name=self.hostname,
                        receive_count=self.tp_size,
                        tp_rank=self.tp_rank,
                        gpu_id=self.scheduler.gpu_id,
                        embeddings_engine=self.embeddings_engine,
                        dtype=self.dtype,
                        embedding_meta=recv_req.mm_metadata,
                        embeddings_buffer_dict=self.embeddings_buffer,
                    )
                else:
                    waiting_req = WaitingImageRequest(
                        rid=recv_req.rid,
                        recv_req=recv_req,
                        mm_processor=self.mm_processor,
                        encoder_urls=self.encode_urls,
                        host_name=self.hostname,
                        receive_count=self.tp_size,
                    )
                waiting_req.send_encode_request()
                self.waiting_list.append(waiting_req)
            else:
                new_recv_reqs.append(recv_req)

        if len(self.waiting_list) == 0:
            return new_recv_reqs, []

        current_time = time.time()
        local_status = []
        for waiting_req in self.waiting_list:
            waiting_req._try_recv_mm_data()
            if current_time - waiting_req.start_time > self.wait_timeout:
                waiting_req.status = WaitingImageRequestStatus.TIMEOUT
            local_status.append(waiting_req.status)

        local_status = torch.tensor(local_status, device="cpu", dtype=torch.int32)

        torch.distributed.all_reduce(
            local_status,
            op=torch.distributed.ReduceOp.MIN,
            group=self.tp_group.cpu_group,
        )

        new_waiting = []
        abort_reqs = []
        for i, waiting_req in enumerate(self.waiting_list):
            status_value = local_status[i].item()
            if status_value == WaitingImageRequestStatus.SUCCESS:
                new_recv_reqs.append(waiting_req.recv_req)
            elif status_value == WaitingImageRequestStatus.FAIL:
                logger.error(
                    f"Waiting request {waiting_req.rid} failed: {waiting_req.error_msg} {waiting_req.error_code = }"
                )
                if hasattr(waiting_req, "_cleanup"):
                    waiting_req._cleanup()
                abort_reqs.append(
                    (
                        self.create_req(waiting_req.recv_req),
                        waiting_req.error_msg,
                        waiting_req.error_code,
                    )
                )
            elif status_value == WaitingImageRequestStatus.TIMEOUT:
                logger.error(
                    f"Timed out waiting for image embeddings for request {waiting_req.rid}"
                )
                if hasattr(waiting_req, "_cleanup"):
                    waiting_req._cleanup()
                abort_reqs.append(
                    (
                        self.create_req(waiting_req.recv_req),
                        f"Timeout waiting for image embedding after {self.wait_timeout}s",
                        HTTPStatus.REQUEST_TIMEOUT,
                    )
                )
            else:  # status_value == WaitingImageRequestStatus.PENDING
                new_waiting.append(waiting_req)

        self.waiting_list = new_waiting
        return new_recv_reqs, abort_reqs

    def _run_encode_in_thread(
        self, req_id, mm_data, endpoint_encode, num_items_assigned, embedding_port
    ):
        try:
            asyncio.run(
                self.encode(
                    req_id=req_id,
                    mm_data=mm_data,
                    embedding_port=embedding_port,
                    endpoint_encode=endpoint_encode,
                    endpoint_send=None,
                    num_items_assigned=num_items_assigned,
                )
            )
        except Exception as e:
            logger.error(f"Encode failed for request {req_id}: {e}", exc_info=True)

    def create_req(self, recv_req: TokenizedGenerateReqInput):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            return_logprob=recv_req.return_logprob,
            top_logprobs_num=recv_req.top_logprobs_num,
            token_ids_logprob=recv_req.token_ids_logprob,
            stream=recv_req.stream,
            lora_id=recv_req.lora_id,
            input_embeds=recv_req.input_embeds,
            custom_logit_processor=recv_req.custom_logit_processor,
            require_reasoning=recv_req.require_reasoning,
            return_hidden_states=recv_req.return_hidden_states,
            return_routed_experts=recv_req.return_routed_experts,
            eos_token_ids=self.scheduler.model_config.hf_eos_token_id,
            bootstrap_host=recv_req.bootstrap_host,
            bootstrap_port=recv_req.bootstrap_port,
            bootstrap_room=recv_req.bootstrap_room,
            disagg_mode=self.scheduler.disaggregation_mode,
            routed_dp_rank=recv_req.routed_dp_rank,
            disagg_prefill_dp_rank=recv_req.disagg_prefill_dp_rank,
            vocab_size=self.scheduler.model_config.vocab_size,
            priority=recv_req.priority,
            metrics_collector=(
                self.scheduler.metrics_collector
                if self.scheduler.enable_metrics
                else None
            ),
            http_worker_ipc=recv_req.http_worker_ipc,
            dllm_config=self.scheduler.dllm_config,
        )
        req.tokenizer = self.scheduler.tokenizer
        return req

    def _assign_items_by_modality(
        self, mm_data, encoder_num, random_shuffle=True
    ) -> Dict:
        """
        Assign multimodal items across encoders by modality with cross-modality load balancing.

        Args:
            mm_data: List of multimodal data items, each with a "modality" key
            encoder_num: Number of encoders
            random_shuffle: Whether to shuffle the encoder indices

        Returns:
            Dictionary mapping modality to list of assignment counts per encoder
            Format: {modality: [count_for_encoder_0, count_for_encoder_1, ...]}
        """
        encode_idx = list(range(encoder_num))
        if random_shuffle:
            random.shuffle(encode_idx)
        # Get unique modalities with order preserved
        modalities = list(dict.fromkeys(mm_item.get("modality") for mm_item in mm_data))
        # Use OrderedDict to explicitly maintain modality order
        num_items_assigned = OrderedDict()
        current_offset = 0

        for modality in modalities:
            mm_data_modality = [
                mm_item for mm_item in mm_data if mm_item.get("modality") == modality
            ]
            num_items = len(mm_data_modality)
            if num_items == 0:
                continue

            base = num_items // len(encode_idx)
            remainder = num_items % len(encode_idx)
            # Rotate assignments based on current_offset to balance load across modalities
            assignments = [0] * len(encode_idx)
            for i in range(len(encode_idx)):
                # keep shuffle order when assigning items to encoders
                pos_in_shuffled = (current_offset + i) % len(encode_idx)
                actual_encoder_idx = encode_idx[pos_in_shuffled]
                assignments[actual_encoder_idx] = base + (1 if i < remainder else 0)
            num_items_assigned[modality] = assignments
            current_offset = (current_offset + remainder) % len(encode_idx)

        return num_items_assigned

    def _extract_url_data(self, request_obj) -> List[Dict]:
        mm_data = []
        for attr, modality in [
            ("image_data", Modality.IMAGE),
            ("video_data", Modality.VIDEO),
            ("audio_data", Modality.AUDIO),
        ]:
            mm_items = getattr(request_obj, attr, None)
            if mm_items:
                if not isinstance(mm_items, list):
                    mm_items = [mm_items]
                for mm_item in mm_items:
                    mm_data.append(
                        {
                            "url": (
                                mm_item.url
                                if isinstance(mm_item, ImageData)
                                else mm_item
                            ),
                            "modality": modality,
                        }
                    )
        return mm_data


class MMReceiverHTTP(MMReceiverBase):
    def __init__(
        self,
        server_args: ServerArgs,
        dtype: Optional[torch.dtype] = None,
        hf_config: Optional[PretrainedConfig] = None,
        pp_rank: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_group: Optional[GroupCoordinator] = None,
        scheduler: Optional["Scheduler"] = None,
    ):
        super().__init__(
            server_args,
            dtype=dtype,
            hf_config=hf_config,
            pp_rank=pp_rank,
            tp_rank=tp_rank,
            tp_group=tp_group,
            scheduler=scheduler,
        )

    # For zmq and mooncake backends (scheduler-side)
    def process_waiting_requests(self, recv_reqs):
        return self._process_waiting_requests(recv_reqs)

    async def encode(
        self,
        req_id,
        mm_data,
        embedding_port,
        endpoint_encode,
        endpoint_send,
        num_items_assigned=None,
    ):
        if len(mm_data) == 0:
            return

        # get unique modalities with order preserved
        modalities = [mm_item.get("modality") for mm_item in mm_data]
        modalities = list(dict.fromkeys(modalities))
        encode_requests = []

        if num_items_assigned is None:
            num_items_assigned = self._assign_items_by_modality(
                mm_data, len(self.encode_urls)
            )

        # Calculate total num_parts across all modalities
        total_num_parts, modality_num_parts = calculate_modality_num_parts(
            modalities, num_items_assigned
        )

        part_idx_offset = 0
        for modality in modalities:
            num_items_assigned_modality = num_items_assigned.get(modality)
            mm_data_modality = [
                mm_item for mm_item in mm_data if mm_item.get("modality") == modality
            ]

            num_parts = modality_num_parts[modality]
            cum_num_items = 0
            cum_idx = 0
            for idx, assigned_num in enumerate(num_items_assigned_modality):
                if assigned_num == 0:
                    continue
                part_idx = part_idx_offset + cum_idx
                part_req_id = create_part_req_id(req_id, part_idx)
                encode_requests.append(
                    {
                        "encoder_idx": idx,
                        "mm_items": [
                            mm_item.get("url")
                            for mm_item in mm_data_modality[
                                cum_num_items : cum_num_items + assigned_num
                            ]
                        ],
                        "num_parts": total_num_parts,
                        "part_idx": part_idx,
                        "req_id": part_req_id,  # use part_req_id to avoid key collision
                        "modality": modality.name,  # convert enum to string for json serialization
                        "prefill_host": self.host,
                        "embedding_port": embedding_port,
                    }
                )
                cum_idx += 1
                cum_num_items += assigned_num
            part_idx_offset += num_parts

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=1800
            )  # Add timeout for request reliability
        ) as session:
            # Send encode requests

            tasks = [
                session.post(
                    f"{self.encode_urls[encode_request['encoder_idx']]}/{endpoint_encode}",
                    json=encode_request,
                )
                for encode_request in encode_requests
            ]

            responses = await asyncio.gather(*tasks)
            for response in responses:
                if response.status != 200:
                    try:
                        err_data = await response.json()
                        msg = err_data.get("message", "Unknown encoder error")
                    except:
                        msg = await response.text()

                    logger.error(f"Encoder returned error {response.status}: {msg}")
                    return
            response_json_list_unsort = [
                await response.json() for response in responses
            ]

            # zmq backend: returns None (send is handled server-side)
            if None in response_json_list_unsort:
                return

    async def wait_mm_metadata(self, obj) -> Optional[str]:
        """For mooncake backend: send /encode to all encoders and collect embedding metadata.

        The metadata is used by the scheduler to allocate GPU buffers for RDMA transfer.
        This is called in the tokenizer_manager's async context (no GPU ops).

        Returns None on success, or an error message string on failure.
        """
        num_items_assigned = obj.num_items_assigned
        mm_data = self._extract_url_data(obj)

        # Get modalities in order
        modalities = [mm_item.get("modality") for mm_item in mm_data]
        modalities = list(dict.fromkeys(modalities))
        total_num_parts, modality_num_parts = calculate_modality_num_parts(
            modalities, num_items_assigned
        )

        encode_requests = []
        part_idx_offset = 0
        for modality in modalities:
            num_items_assigned_modality = num_items_assigned.get(modality)
            mm_data_modality = [
                mm_item for mm_item in mm_data if mm_item.get("modality") == modality
            ]
            num_parts = modality_num_parts[modality]
            cum_num_items = 0
            cum_idx = 0
            for idx, assigned_num in enumerate(num_items_assigned_modality):
                if assigned_num == 0:
                    continue
                part_idx = part_idx_offset + cum_idx
                part_req_id = create_part_req_id(obj.rid, part_idx)
                encode_requests.append(
                    {
                        "encoder_idx": idx,
                        "mm_items": [
                            mm_item.get("url")
                            for mm_item in mm_data_modality[
                                cum_num_items : cum_num_items + assigned_num
                            ]
                        ],
                        "num_parts": total_num_parts,
                        "part_idx": part_idx,
                        "req_id": part_req_id,
                        "modality": modality.name,
                        "prefill_host": self.host,
                        "embedding_port": None,
                    }
                )
                cum_idx += 1
                cum_num_items += assigned_num
            part_idx_offset += num_parts

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=1800)
        ) as session:
            tasks = [
                session.post(
                    f"{self.encode_urls[req['encoder_idx']]}/encode",
                    json=req,
                )
                for req in encode_requests
            ]
            responses = await asyncio.gather(*tasks)
            for response in responses:
                if response.status != 200:
                    try:
                        err_data = await response.json()
                        msg = err_data.get("message", "Unknown encoder error")
                    except Exception:
                        msg = await response.text()
                    logger.error(f"Encoder returned error {response.status}: {msg}")
                    return msg

            response_json_list = [await response.json() for response in responses]

        meta_sorted = [None] * total_num_parts
        for rj in response_json_list:
            meta_sorted[rj["part_idx"]] = rj

        obj.mm_metadata = meta_sorted
        return None


class MMReceiverGrpc(MMReceiverBase):
    def __init__(
        self,
        server_args: ServerArgs,
        dtype: Optional[torch.dtype] = None,
        hf_config: Optional[PretrainedConfig] = None,
        pp_rank: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_group: Optional[GroupCoordinator] = None,
        scheduler: Optional["Scheduler"] = None,
    ):
        super().__init__(
            server_args,
            dtype=dtype,
            hf_config=hf_config,
            pp_rank=pp_rank,
            tp_rank=tp_rank,
            tp_group=tp_group,
            scheduler=scheduler,
        )

    def build_and_send_encode_request(self, image_urls, rid):
        encode_req = GenerateReqInput(
            image_data=[ImageData(url=url) for url in image_urls],
            rid=rid,
        )
        self.send_encode_request(encode_req)
        return encode_req

    # For zmq and mooncake backends (scheduler-side)
    def process_waiting_requests(self, recv_reqs):
        return self._process_waiting_requests(recv_reqs)

    async def encode(
        self,
        req_id,
        mm_data,
        embedding_port,
        endpoint_encode,
        endpoint_send,
        num_items_assigned=None,
    ):
        if not mm_data:
            return

        # gRPC currently only supports image; flatten new dict formats to simple lists
        if mm_data and isinstance(mm_data[0], dict):
            non_image = [
                item.get("modality")
                for item in mm_data
                if item.get("modality") != Modality.IMAGE
            ]
            if non_image:
                raise NotImplementedError(
                    f"gRPC encode only supports IMAGE modality, got: {non_image}"
                )
            img_data = [item.get("url") for item in mm_data]
        else:
            img_data = mm_data
        if isinstance(num_items_assigned, dict):
            num_items_assigned = list(num_items_assigned.values())[0]

        encode_requests = []
        if num_items_assigned is None:
            encode_idx = list(range(len(self.encode_urls)))
            random.shuffle(encode_idx)
            num_items_assigned = [
                (idx + len(img_data)) // len(self.encode_urls) for idx in encode_idx
            ]
        num_parts = sum(1 for x in num_items_assigned if x != 0)
        cum_num_items = 0
        cum_idx = 0
        for idx, assigned_num in enumerate(num_items_assigned):
            if assigned_num == 0:
                continue
            start = cum_num_items
            end = cum_num_items + assigned_num
            encode_requests.append(
                {
                    "encoder_idx": idx,
                    "mm_items": img_data[start:end],
                    "num_parts": num_parts,
                    "part_idx": cum_idx,
                    "req_id": req_id,
                    "prefill_host": self.host,
                    "embedding_port": embedding_port,
                }
            )
            cum_idx += 1
            cum_num_items += assigned_num

        grpc_tasks = [
            asyncio.to_thread(
                _grpc_encode_request,
                _grpc_target(self.encode_urls[encode_request["encoder_idx"]]),
                encode_request,
            )
            for encode_request in encode_requests
        ]
        grpc_responses = await asyncio.gather(*grpc_tasks)
        # zmq backend: send is handled server-side
        for encode_request, response in zip(encode_requests, grpc_responses):
            if self.encoder_transfer_backend == "zmq":
                return


def _validate_transport_mode(transport_mode: str, encoder_urls):
    if transport_mode == "grpc":
        invalid_prefix = "http://"
        error_msg = (
            "EPD MMReceiver: grpc mode requires grpc:// encoder URLs. "
            "Set SGLANG_ENCODER_MM_RECEIVER_MODE=http for http:// URLs."
        )
    elif transport_mode == "http":
        invalid_prefix = "grpc://"
        error_msg = (
            "EPD MMReceiver: http mode requires http:// encoder URLs. "
            "Set SGLANG_ENCODER_MM_RECEIVER_MODE=grpc for grpc:// URLs."
        )
    else:
        return

    if any(url.startswith(invalid_prefix) for url in encoder_urls):
        raise ValueError(error_msg)


_MM_RECEIVER_BY_MODE = {
    "grpc": MMReceiverGrpc,
    "http": MMReceiverHTTP,
}


def create_mm_receiver(
    server_args: ServerArgs,
    dtype: Optional[torch.dtype] = None,
    hf_config: Optional[PretrainedConfig] = None,
    pp_rank: Optional[int] = None,
    tp_rank: Optional[int] = None,
    tp_group: Optional[GroupCoordinator] = None,
    scheduler: Optional["Scheduler"] = None,
    transport_mode: Optional[str] = None,
):
    if transport_mode is None:
        transport_mode = envs.SGLANG_ENCODER_MM_RECEIVER_MODE.get()
        logger.debug(f"MMReceiver transport_mode from env: {transport_mode}")

    _validate_transport_mode(transport_mode, server_args.encoder_urls)
    logger.info(f"EPD MMReceiver: using transport_mode={transport_mode}")

    receiver_cls = _MM_RECEIVER_BY_MODE.get(transport_mode)
    if receiver_cls is None:
        raise ValueError(f"Unsupported transport_mode: {transport_mode}")
    return receiver_cls(
        server_args,
        dtype=dtype,
        hf_config=hf_config,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        tp_group=tp_group,
        scheduler=scheduler,
    )
