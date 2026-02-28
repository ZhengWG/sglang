# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""package for sglang requests tracing"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

from sglang.srt.utils import get_int_env_var

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Req

from sglang.utils import run_once

logger = logging.getLogger(__name__)
opentelemetry_imported = False
opentelemetry_initialized = False
_trace_context_propagator = None
tracer: Optional[trace.Tracer] = None

global_trace_level = 1

NORMAL_TRACE_LEVEL = 1
ENHANCED_TRACE_LEVEL = 2
APP_NAME = 'sglang'

TRACE_HEADERS = ["traceparent", "tracestate", "SOFA-TraceId", "SOFA-RpcId", "X-Request-ID", "X-AIGW-APP-KeyId"]

try:
    from opentelemetry import context, propagate, trace
    from opentelemetry.context.context import Context
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GRPCSpanExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HTTPSpanExporter,
    )
    from opentelemetry.sdk.environment_variables import (
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
    )
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider, id_generator
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    _trace_context_propagator = TraceContextTextMapPropagator()

    opentelemetry_imported = True
except ImportError:

    class id_generator:
        class IdGenerator:
            pass

    logger.debug("opentelemetry package is not installed, tracing disabled")


def contains_trace_headers(headers: Mapping[str, str]) -> bool:
    return any(h in headers for h in TRACE_HEADERS)


@run_once
def log_tracing_disabled_warning() -> None:
    logger.warning(
        "Received a request with trace context but tracing is disabled")


def is_tracing_enabled() -> bool:
    return opentelemetry_initialized


def extract_trace_headers(headers: Mapping[str, str]) -> Optional[Dict]:
    return {h: headers[h] for h in TRACE_HEADERS if h in headers}


def set_global_trace_level(level: int):
    global global_trace_level
    global_trace_level = level


@dataclass
class TraceThreadInfo:
    host_id: str
    pid: int
    thread_label: str
    tp_rank: int
    dp_rank: int


@dataclass
class TraceEvent:
    event_name: str
    ts: int
    attrs: Dict[str, Any]


@dataclass
class TraceSliceContext:
    slice_name: str
    start_time_ns: int
    end_time_ns: Optional[int] = None
    span: Optional[trace.span.Span] = None
    level: int = 1
    attrs: Optional[Dict[str, Any]] = None
    events: Optional[List[TraceEvent]] = None
    # When True, defers slice_name assignment until trace_slice_end()
    anonymous: bool = False


@dataclass
class TraceThreadContext:
    thread_info: TraceThreadInfo
    cur_slice_stack: Optional[List[TraceSliceContext]] = None
    thread_span: Optional[trace.span.Span] = None
    # Record the most recently completed span as the previous span for the next span to be created.
    last_span_context: Optional[trace.span.SpanContext] = None


class TraceCustomIdGenerator(id_generator.IdGenerator):
    """
    The default IdGenerator may produce duplicate trace IDs across multiple TP scheduler processes,
    hence a custom IdGenerator is implemented.
    """

    def __init__(self):
        super().__init__()
        self.local_random = random.Random()
        self.local_random.seed(time.time())

    def generate_trace_id(self) -> int:
        return self.local_random.getrandbits(64)

    def generate_span_id(self) -> int:
        return self.local_random.getrandbits(64)


# global variables
remote_trace_contexts: Dict[str, "TracePropagateContext"] = {}
threads_info: Dict[int, TraceThreadInfo] = {}

get_cur_time_ns = lambda: int(time.time() * 1e9)
if hasattr(time, "time_ns"):
    get_cur_time_ns = lambda: int(time.time_ns())


def __get_host_id() -> str:
    """
    In distributed tracing systems, obtain a unique node identifier
    and inject it into all subsequently generated spans
    to prevent PID conflicts between threads on different nodes.
    """
    if os.path.exists("/etc/machine-id"):
        try:
            with open("/etc/machine-id", "r") as f:
                return f.read().strip()
        except:
            pass

    mac = uuid.getnode()
    if mac != 0:
        return uuid.UUID(int=mac).hex

    return "unknown"


# Should be called by each tracked process.
def process_tracing_init(otlp_endpoint, server_name):
    global opentelemetry_initialized
    global get_cur_time_ns
    global tracer
    if not opentelemetry_imported:
        opentelemetry_initialized = False
        raise RuntimeError(
            "opentelemetry package is not installed!!! Please not enable tracing or install opentelemetry"
        )

    try:
        resource = Resource.create(
            attributes={
                SERVICE_NAME: server_name,
            }
        )
        tracer_provider = TracerProvider(
            resource=resource, id_generator=TraceCustomIdGenerator()
        )

        schedule_delay_millis = get_int_env_var(
            "SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS", 500
        )
        max_export_batch_size = get_int_env_var(
            "SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE", 64
        )

        processor = BatchSpanProcessor(
            span_exporter=get_otlp_span_exporter(otlp_endpoint),
            schedule_delay_millis=schedule_delay_millis,
            max_export_batch_size=max_export_batch_size,
        )
        tracer_provider.add_span_processor(processor)
        trace.set_tracer_provider(tracer_provider)
    except Exception as e:
        opentelemetry_initialized = False
        raise RuntimeError(
            f"initialize opentelemetry error:{e}. Please set correct otlp endpoint."
        )

    opentelemetry_initialized = True
    tracer = trace.get_tracer("sglang server")


def get_global_tracing_enabled():
    return opentelemetry_initialized


def get_otlp_span_exporter(endpoint):
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "http/protobuf")
    supported_protocols = {"grpc", "http/protobuf"}

    if protocol not in supported_protocols:
        raise ValueError(
            f"Unsupported OTLP protocol '{protocol}' configured. "
            f"Supported protocols are: {', '.join(sorted(supported_protocols))}"
        )

    if protocol == "grpc":
        return GRPCSpanExporter(endpoint=endpoint, insecure=True)
    elif protocol == "http/protobuf":
        return HTTPSpanExporter(endpoint=endpoint)


# Should be called by each tracked thread.
def trace_set_thread_info(
    thread_label: str, tp_rank: Optional[int] = None, dp_rank: Optional[int] = None
):
    if not opentelemetry_initialized:
        return

    pid = threading.get_native_id()
    if pid in threads_info:
        return

    threads_info[pid] = TraceThreadInfo(
        host_id=__get_host_id(),
        pid=pid,
        thread_label=thread_label,
        tp_rank=tp_rank,
        dp_rank=dp_rank,
    )


@dataclass
class TracePropagateContext:
    root_span_context: context.Context
    prev_span_context: Optional[trace.span.SpanContext]

    def to_dict(self):
        carrier: dict[str, str] = {}
        propagate.inject(carrier, self.root_span_context)

        if self.prev_span_context:
            return {
                "root_span": carrier,
                "prev_span": {
                    "span_id": self.prev_span_context.span_id,
                    "trace_id": self.prev_span_context.trace_id,
                },
            }
        else:
            return {"root_span": carrier, "prev_span": "None"}

    @classmethod
    def instance_from_dict(cls, d):
        if "root_span" not in d or "prev_span" not in d:
            return None

        carrier = d["root_span"]
        root_span_context = propagate.extract(carrier)

        if d["prev_span"] == "None":
            prev_span_context = None
        else:
            prev_span_context = trace.span.SpanContext(
                trace_id=d["prev_span"]["trace_id"],
                span_id=d["prev_span"]["span_id"],
                is_remote=True,
            )

        return cls(root_span_context, prev_span_context)


class TraceReqContext:
    def __init__(
        self,
        rid,
        bootstrap_room=None,
        role="unified",
        module_name="",
        external_trace_header: Optional[Dict[str, str]] = None,
    ):
        self.rid: str = str(rid)
        self.trace_level = global_trace_level
        self.tracing_enable: bool = opentelemetry_initialized and self.trace_level > 0

        if not self.tracing_enable:
            return

        self.start_time_ns: Optional[int] = None
        self.thread_context: Optional[TraceThreadContext] = None
        self.bootstrap_room: Optional[int] = bootstrap_room
        self.role: str = role
        self.module_name = module_name

        # Indicates whether this instance is a replica from the main process.
        # When True, root_span is None and only root_span_context is preserved.
        self.is_copy: bool = False
        self.root_span: Optional[trace.span.Span] = None
        self.root_span_context: Optional[context.Context] = None
        self.bootstrap_room_span: Optional[trace.span.Span] = None
        self.bootstrap_room_span_context: Optional[context.Context] = None
        # Record the most recently completed span as the previous span for the next span to be created.
        self.last_span_context: Optional[trace.span.SpanContext] = None
        self.external_trace_header: Optional[Dict[str, str]] = external_trace_header

        self.events_cache: List[TraceEvent] = []

        self.pid: int = threading.get_native_id()

    def is_tracing_enabled(self) -> bool:
        return self.tracing_enable

    def __create_thread_context(self, ts: int):
        if self.pid not in threads_info:
            trace_set_thread_info("unknown")

        thread_info = threads_info[self.pid]
        thread_context = TraceThreadContext(
            thread_info=thread_info,
            cur_slice_stack=[],
        )

        thread_name = f"{thread_info.thread_label}"
        if thread_info.tp_rank is not None:
            thread_name += f" [TP {thread_info.tp_rank}] "
        thread_name += f"(host:{thread_info.host_id[:8]} | pid:{self.pid})"

        if self.tracing_enable == 1:
            return thread_context

        thread_context.thread_span = tracer.start_span(
            name=thread_name,
            start_time=ts,
            context=self.root_span_context,
        )

        if thread_info.tp_rank is not None:
            thread_context.thread_span.set_attributes({"tp_rank": thread_info.tp_rank})

        thread_context.thread_span.set_attributes(
            {
                "host_id": thread_info.host_id,
                "pid": thread_info.pid,
                "thread_label": thread_info.thread_label,
            }
        )

        return thread_context

    def __getstate__(self) -> Optional[Dict[str, Any]]:
        if not self.tracing_enable:
            return {"tracing_enable": False}

        if not self.root_span_context:
            return {"tracing_enable": False}

        state = {
            "tracing_enable": self.tracing_enable,
            "rid": self.rid,
            "bootstrap_room": self.bootstrap_room,
            "start_time_ns": self.start_time_ns,
            "role": self.role,
            "trace_level": self.trace_level,
            "module_name": self.module_name,
            "is_copy": self.is_copy,
            "pid": self.pid,
            "thread_context": None,
            "root_span": None,
            "last_span_context": None,
        }

        carrier: dict[str, str] = {}
        propagate.inject(carrier, self.root_span_context)
        state["root_span_context"] = carrier

        prev_span_context = self.last_span_context
        if self.thread_context and self.thread_context.cur_slice_stack:
            cur_slice = self.thread_context.cur_slice_stack[0]
            if cur_slice.span:
                prev_span_context = cur_slice.span.get_span_context()

        if prev_span_context:
            state["last_span_context"] = {
                "span_id": prev_span_context.span_id,
                "trace_id": prev_span_context.trace_id,
            }

        return state

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)
        if not opentelemetry_initialized:
            self.tracing_enable = False
        if not self.tracing_enable:
            return

        self.is_copy = True
        self.pid = threading.get_native_id()
        self.root_span_context = propagate.extract(self.root_span_context)
        if self.last_span_context:
            self.last_span_context = trace.span.SpanContext(
                trace_id=self.last_span_context["trace_id"],
                span_id=self.last_span_context["span_id"],
                is_remote=True,
            )
        self.events_cache = []

    def rebuild_thread_context(self, ts: Optional[int] = None):
        if not self.tracing_enable:
            return

        ts = ts or get_cur_time_ns()
        self.thread_context = self.__create_thread_context(ts)

    def trace_req_start(
        self,
        ts: Optional[int] = None,
    ):
        if not self.tracing_enable:
            return

        ts = ts or get_cur_time_ns()

        # create req context and root span
        self.start_time_ns = ts

        bootstrap_room_span_context = None
        external_trace_context = _trace_context_propagator.extract(
            self.external_trace_header or {}
        )

        # create bootstrap room span for multispan
        if self.trace_level > 1:
            bootstrap_room = self.bootstrap_room or 0
            if str(bootstrap_room) not in remote_trace_contexts:
                attrs = {"bootstrap_room": str(hex(bootstrap_room))}
                bootstrap_room_span = tracer.start_span(
                    name=f"Bootstrap Room {hex(bootstrap_room)}",
                    start_time=ts,
                    attributes=attrs,
                    context=external_trace_context,
                )
                self.bootstrap_room_span = bootstrap_room_span
                bootstrap_room_span_context = trace.set_span_in_context(bootstrap_room_span)
            else:
                bootstrap_room_span_context = remote_trace_contexts[
                    str(bootstrap_room)
                ].root_span_context

        # Drop the worker_id added by MultiTokenizer
        orig_rid = self.rid.split("_")[-1]
        role = "" if self.role == "unified" else self.role
        attrs = {"rid": orig_rid, "module": f"sglang::{self.module_name}"}
        if self.bootstrap_room:
            attrs["bootstrap_room"] = str(hex(self.bootstrap_room))
        root_span = tracer.start_span(
            name="llm_request",
            start_time=ts,
            context=bootstrap_room_span_context if self.trace_level > 1 else external_trace_context,
            attributes=attrs,
            kind=trace.SpanKind.SERVER,
        )

        self.root_span = root_span
        self.root_span_context = trace.set_span_in_context(root_span)
        self.bootstrap_room_span_context = bootstrap_room_span_context

        # create thread context and thread span
        self.thread_context = self.__create_thread_context(ts)

        # Set last_span_context from remote context if available
        if self.trace_level > 1 and self.bootstrap_room and str(self.bootstrap_room) in remote_trace_contexts:
            self.last_span_context = remote_trace_contexts[str(self.bootstrap_room)].prev_span_context

    def trace_req_finish(
        self, ts: Optional[int] = None, attrs: Optional[Dict[str, Any]] = None
    ):
        if not self.tracing_enable:
            return

        if not self.root_span:
            return

        ts = ts or get_cur_time_ns()

        # End all unclosed thread spans.
        self.abort()

        if attrs:
            self.root_span.set_attributes(attrs)

        self.root_span.set_status(Status(StatusCode.OK))
        self.root_span.end(end_time=ts)

        # Clean up multispan resources
        if self.trace_level > 1:
            if self.bootstrap_room and str(self.bootstrap_room) in remote_trace_contexts:
                del remote_trace_contexts[str(self.bootstrap_room)]
            elif self.bootstrap_room_span:
                self.bootstrap_room_span.end(end_time=ts)

        self.root_span = None

    def __check_fast_return(self, level=None):
        if not self.tracing_enable:
            return True

        if not self.thread_context:
            return True

        if level and level > self.trace_level:
            return True

        return False

    def trace_slice_start(
        self,
        name: str,
        level: int,
        ts: Optional[int] = None,
        anonymous: bool = False,
    ):
        if self.__check_fast_return(level):
            return

        ts = ts or get_cur_time_ns()

        cur_slice = TraceSliceContext(
            slice_name=name,
            start_time_ns=ts,
            level=level,
            attrs={},
            events=[],
            anonymous=anonymous,
        )

        parent_span = self.thread_context.thread_span
        prev_span_context = None
        if not self.thread_context.cur_slice_stack:
            if self.last_span_context:
                prev_span_context = self.last_span_context
        else:
            parent_span = self.thread_context.cur_slice_stack[-1].span

        parent_span_context = trace.set_span_in_context(parent_span)

        span = tracer.start_span(
            name=cur_slice.slice_name,
            start_time=cur_slice.start_time_ns,
            context=parent_span_context,
        )
        cur_slice.span = span

        if prev_span_context:
            span.add_link(prev_span_context)

        self.thread_context.cur_slice_stack.append(cur_slice)

    def trace_slice_end(
        self,
        name: str,
        level: int,
        ts: Optional[int] = None,
        attrs: Optional[Dict[str, Any]] = None,
        auto_next_anon: bool = False,
        thread_finish_flag: bool = False,
    ):
        if self.__check_fast_return(level):
            return

        if not self.thread_context.cur_slice_stack:
            logger.warning(
                f"No matching with the SLICE_START event {name} is required."
            )
            return

        cur_slice = self.thread_context.cur_slice_stack[-1]
        ts = ts or get_cur_time_ns()

        # Handle anonymous slice name
        if cur_slice.anonymous:
            cur_slice.span.update_name(name)
        elif cur_slice.slice_name != name or cur_slice.level != level:
            logger.warning(
                f"Slice name mismatch: {name} != {cur_slice.slice_name} or level mismatch: {level} != {cur_slice.level}"
            )
            cur_slice.span.set_status(Status(StatusCode.ERROR))
            self.thread_context.cur_slice_stack.pop()
            return

        span = cur_slice.span

        if attrs:
            span.set_attributes(attrs)

        if self.events_cache:
            new_events_cache = []
            for event in self.events_cache:
                if event.ts >= cur_slice.start_time_ns and event.ts < ts:
                    span.add_event(
                        name=event.event_name,
                        timestamp=event.ts,
                        attributes=event.attrs,
                    )
                else:
                    new_events_cache.append(event)
            self.events_cache = new_events_cache

        span.end(end_time=ts)

        self.thread_context.cur_slice_stack.pop()
        # only for first level slice
        if not self.thread_context.cur_slice_stack:
            self.last_span_context = span.get_span_context()

        if thread_finish_flag:
            self.abort(ts)
            if self.is_copy:
                return

        if auto_next_anon:
            self.trace_slice_start("", level, ts, anonymous=True)

    def trace_slice(
        self,
        slice: TraceSliceContext,
        thread_finish_flag: bool = False,
    ):
        if self.__check_fast_return(slice.level):
            return

        parent_span = self.thread_context.thread_span
        prev_span_context = None
        if not self.thread_context.cur_slice_stack:
            if self.last_span_context:
                prev_span_context = self.last_span_context
        else:
            parent_span = self.thread_context.cur_slice_stack[-1].span

        parent_span_context = trace.set_span_in_context(parent_span)

        span = tracer.start_span(
            name=slice.slice_name,
            start_time=slice.start_time_ns,
            context=parent_span_context,
        )

        if prev_span_context:
            span.add_link(prev_span_context)

        if slice.attrs:
            span.set_attributes(slice.attrs)

        if slice.events:
            for event in slice.events:
                span.add_event(
                    name=event.event_name, timestamp=event.ts, attributes=event.attrs
                )

        if self.events_cache:
            new_events_cache = []
            for event in self.events_cache:
                if event.ts >= slice.start_time_ns and event.ts < slice.end_time_ns:
                    span.add_event(
                        name=event.event_name,
                        timestamp=event.ts,
                        attributes=event.attrs,
                    )
                else:
                    new_events_cache.append(event)
            self.events_cache = new_events_cache

        span.end(end_time=slice.end_time_ns)

        # only for first level slice
        if not self.thread_context.cur_slice_stack:
            self.last_span_context = span.get_span_context()

        if thread_finish_flag:
            self.abort(slice.end_time_ns)

    # Add event to the current slice on the same thread with the same rid.
    def trace_event(
        self,
        name: str,
        level: int,
        ts: Optional[int] = None,
        attrs: Dict[str, Any] = None,
    ):
        if self.__check_fast_return(level):
            return

        ts = ts or get_cur_time_ns()

        if attrs is None:
            attrs = {}
        self.events_cache.append(TraceEvent(name, ts, attrs))

    def trace_set_root_attrs(self, attrs: Dict[str, Any]):
        if not self.tracing_enable:
            return

        if self.root_span:
            self.root_span.set_attributes(attrs)

    def trace_set_thread_attrs(self, attrs: Dict[str, Any]):
        if self.__check_fast_return():
            return

        if self.thread_context.thread_span:
            self.thread_context.thread_span.set_attributes(attrs)

    def abort(self, ts=None, abort_info: Optional[Dict] = None):
        if self.__check_fast_return():
            return

        # close all slice spans (unlikely, except error API usage)
        ts = ts or get_cur_time_ns()
        while len(self.thread_context.cur_slice_stack) > 0:
            if self.thread_context.cur_slice_stack[-1].span:
                self.thread_context.cur_slice_stack[-1].span.end(end_time=ts)
            self.thread_context.cur_slice_stack.pop()

        # set abort info into thread span
        if self.thread_context.thread_span:
            if abort_info:
                from sglang.srt.managers.schedule_batch import BaseFinishReason

                if isinstance(abort_info, BaseFinishReason):
                    abort_info = abort_info.to_json()
                self.thread_context.thread_span.set_status(Status(StatusCode.ERROR))
                self.thread_context.thread_span.set_attributes(abort_info)

            if self.events_cache:
                for event in self.events_cache:
                    self.thread_context.thread_span.add_event(
                        name=event.event_name,
                        timestamp=event.ts,
                        attributes=event.attrs,
                    )
                self.events_cache = []

            self.thread_context.thread_span.end(end_time=ts)
        self.thread_context = None

    def __del__(self):
        self.abort(abort_info={"reason": "have unclosed span, auto closed"})


@dataclass
class TraceNullContext:
    tracing_enable: bool = False

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self


class SofaTraceInfo:
    def __init__(self,
                 sofa_trace_id: Optional[str] = None,
                 sofa_rpc_id: Optional[str] = None,
                 request_id: Optional[str] = None,
                 aigw_app_key_id: Optional[str] = None):
        self.sofa_trace_id = sofa_trace_id
        self.sofa_rpc_id = sofa_rpc_id
        self.request_id = request_id
        self.aigw_app_key_id = aigw_app_key_id


@dataclass
class EnvInfo:
    pod_ip: Optional[str] = None
    idc: Optional[str] = None
    model_service_id: Optional[str] = None
    model_instance_id: Optional[str] = None
    pod_name: Optional[str] = None
    hostname: Optional[str] = None
    model_instance_name: Optional[str] = None


def get_env_info() -> EnvInfo:
    """
    Extract metadata from environment
    """
    env_info = EnvInfo()
    if ip := os.getenv("POD_IP"):
        env_info.pod_ip = ip
    if idc := os.getenv("ALIPAY_APP_IDC"):
        env_info.idc = idc
    if model_service_id := os.getenv("MODEL_SERVICE_ID"):
        env_info.model_service_id = model_service_id
    if model_instance_id := os.getenv("MODEL_INSTANCE_NAME"):
        env_info.model_instance_id = model_instance_id
    if pod_name := os.getenv("ALIPAY_POD_NAME"):
        env_info.pod_name = pod_name
    if hostname := os.getenv("HOSTNAME"):
        env_info.hostname = hostname
    if model_instance_name := os.getenv("MODEL_INSTANCE_NAME"):
        env_info.model_instance_name = model_instance_name
    return env_info


def get_sofa_trace_info(parent_trace_headers: Mapping[str, str]) -> Optional[SofaTraceInfo]:
    """
    Get SOFA trace id and RPC id from headers
    """
    sofa_trace_info = SofaTraceInfo()
    for (k, v) in parent_trace_headers.items():
        if k == "SOFA-TraceId":
            sofa_trace_info.sofa_trace_id = v
        if k == "SOFA-RpcId":
            sofa_trace_info.sofa_rpc_id = v
        if k == "X-Request-ID":
            sofa_trace_info.request_id = v
        if k == "X-AIGW-APP-KeyId":
            sofa_trace_info.aigw_app_key_id = v
    return sofa_trace_info


# Global request contexts for multispan support
reqs_context: Dict[str, TraceReqContext] = {}


def trace_get_proc_propagate_context(
    rid, remote_propagate=False
) -> Optional[Dict[str, Any]]:
    if not opentelemetry_initialized:
        return None

    if not global_trace_level > 1:
        return None

    rid = str(rid)
    if rid not in reqs_context or not reqs_context[rid].root_span_context:
        return None

    pid = threading.get_native_id()
    prev_span_context = None
    if reqs_context[rid].thread_context and reqs_context[rid].thread_context.cur_slice_stack:
        cur_slice = reqs_context[rid].thread_context.cur_slice_stack[0]
        prev_span_context = cur_slice.span.get_span_context()
    elif reqs_context[rid].last_span_context:
        prev_span_context = reqs_context[rid].last_span_context

    root_span_context = reqs_context[rid].root_span_context
    if remote_propagate and reqs_context[rid].bootstrap_room_span_context:
        root_span_context = reqs_context[rid].bootstrap_room_span_context

    trace_context = TracePropagateContext(root_span_context, prev_span_context)
    return trace_context.to_dict()


def trace_set_proc_propagate_context(rid, trace_context: Optional[Dict[str, Any]]):
    if not opentelemetry_initialized:
        return
    if not trace_context:
        return

    if not global_trace_level > 1:
        return None

    trace_context = TracePropagateContext.instance_from_dict(trace_context)
    if not trace_context:
        return

    rid = str(rid)
    # Create a copy of the request context
    if rid not in reqs_context:
        reqs_context[rid] = TraceReqContext(
            rid=rid,
        )
        reqs_context[rid].start_time_ns = get_cur_time_ns()
        reqs_context[rid].root_span_context = trace_context.root_span_context
        reqs_context[rid].is_copy = True
        reqs_context[rid].tracing_enable = True

    pid = threading.get_native_id()

    if reqs_context[rid].thread_context:
        return

    # Create new thread context.
    ts = reqs_context[rid].start_time_ns
    if pid not in threads_info:
        trace_set_thread_info("unknown")
    thread_info = threads_info[pid]
    thread_context = TraceThreadContext(
        thread_info=thread_info,
        cur_slice_stack=[],
    )

    thread_name = f"{thread_info.thread_label}"
    if thread_info.tp_rank is not None:
        thread_name += f" [TP {thread_info.tp_rank}] "
    thread_name += f"(host:{thread_info.host_id[:8]} | pid:{pid})"
    thread_context.thread_span = tracer.start_span(
        name=thread_name,
        start_time=ts,
        context=reqs_context[rid].root_span_context,
    )

    if thread_info.tp_rank is not None:
        thread_context.thread_span.set_attributes({"tp_rank": thread_info.tp_rank})

    thread_context.thread_span.set_attributes(
        {
            "host_id": thread_info.host_id,
            "pid": thread_info.pid,
            "thread_label": thread_info.thread_label,
        }
    )

    reqs_context[rid].thread_context = thread_context
    reqs_context[rid].last_span_context = trace_context.prev_span_context


def trace_get_remote_propagate_context(bootstrap_room_list: List[str]):
    if not opentelemetry_initialized:
        return ""

    if not global_trace_level > 1:
        return ""

    reqs_trace_contexts = {}
    for bootstrap_room in bootstrap_room_list:
        # In the router, rid is also the bootstrap room.
        bootstrap_room = str(bootstrap_room)

        if bootstrap_room not in reqs_context:
            continue

        _context = trace_get_proc_propagate_context(
            bootstrap_room, remote_propagate=True
        )
        reqs_trace_contexts[bootstrap_room] = _context

    json_str = json.dumps(reqs_trace_contexts, ensure_ascii=False)
    return base64.b64encode(json_str.encode("utf-8")).decode("utf-8")


def trace_set_remote_propagate_context(base64_str):
    if not opentelemetry_initialized:
        return

    if not global_trace_level > 1:
        return

    if base64_str is None or base64_str == "" or base64_str == "None":
        return

    base64_bytes = base64.b64decode(base64_str)
    json_str = base64_bytes.decode("utf-8")
    remote_reqs_trace_contexts = json.loads(json_str)

    for bootstrap_room in remote_reqs_trace_contexts:
        if bootstrap_room in remote_trace_contexts:
            continue

        remote_trace_contexts[bootstrap_room] = TracePropagateContext.instance_from_dict(
            remote_reqs_trace_contexts[bootstrap_room]
        )


# Legacy function-style API for backward compatibility
def trace_req_start_legacy(
    rid: str,
    bootstrap_room: Optional[int] = None,
    ts: Optional[int] = None,
    role: Optional[str] = "null",
    external_trace_header: Optional[Dict[str, str]] = None,
):
    if not opentelemetry_initialized:
        return

    rid = str(rid)
    ts = ts or get_cur_time_ns()

    # Create TraceReqContext and store in global dict
    reqs_context[rid] = TraceReqContext(
        rid=rid,
        bootstrap_room=bootstrap_room,
        role=role,
        external_trace_header=external_trace_header,
    )
    reqs_context[rid].trace_req_start(ts)


def trace_req_finish_legacy(
    rid: str, ts: Optional[int] = None, attrs: Optional[Dict[str, Any]] = None
):
    if not opentelemetry_initialized:
        return

    rid = str(rid)
    if rid not in reqs_context:
        return

    reqs_context[rid].trace_req_finish(ts, attrs)
    del reqs_context[rid]


def trace_slice_start_legacy(
    name: str,
    rid: str,
    ts: Optional[int] = None,
    anonymous: bool = False,
):
    if not opentelemetry_initialized or not global_trace_level > 1:
        return

    rid = str(rid)
    if rid not in reqs_context:
        return

    reqs_context[rid].trace_slice_start(name, 1, ts, anonymous)


def trace_slice_end_legacy(
    name: str,
    rid: str,
    ts: Optional[int] = None,
    attrs: Optional[Dict[str, Any]] = None,
    auto_next_anon: bool = False,
    thread_finish_flag: bool = False,
):
    if not opentelemetry_initialized or not global_trace_level > 1:
        return

    rid = str(rid)
    if rid not in reqs_context:
        return

    reqs_context[rid].trace_slice_end(name, 1, ts, attrs, auto_next_anon, thread_finish_flag)

    if thread_finish_flag and reqs_context[rid].is_copy and not reqs_context[rid].thread_context:
        del reqs_context[rid]


# alias
trace_slice = trace_slice_end_legacy


def trace_event_legacy(
    name: str, rid: str, ts: Optional[int] = None, attrs: Dict[str, Any] = None
):
    if not opentelemetry_initialized or not global_trace_level > 1:
        return

    rid = str(rid)
    if rid not in reqs_context:
        return

    reqs_context[rid].trace_event(name, 1, ts, attrs)


def trace_slice_batch(
    name: str,
    reqs: List[Req],
):
    if not opentelemetry_initialized:
        return

    for req in reqs:
        trace_slice(
            name,
            req.rid,
            auto_next_anon=not req.finished(),
            thread_finish_flag=req.finished(),
        )


def extract_trace_context(
        headers: Optional[Mapping[str, str]]) -> Optional[Context]:
    if opentelemetry_initialized:
        headers = headers or {}
        return TraceContextTextMapPropagator().extract(headers)
    else:
        return None


def trace_event_batch(
    name: str,
    reqs: List[Req],
    ts: Optional[int] = None,
    attrs: Dict[str, Any] = {},
):
    if not opentelemetry_initialized or not global_trace_level > 1:
        return

    bid = uuid.uuid4().hex[:8]
    _attrs = {"bid": bid, "batch_size": len(reqs)}
    _attrs.update(attrs)

    for req in reqs:
        trace_event_legacy(name, req.rid, ts=ts, attrs=_attrs)


class SpanAttributes:
    # Attribute names copied from here to avoid version conflicts:
    # https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    GEN_AI_USAGE_CACHED_TOKENS = "gen_ai.usage.cached_tokens"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    GEN_AI_REQUEST_ID = "gen_ai.request.id"
    GEN_AI_REQUEST_N = "gen_ai.request.n"
    GEN_AI_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    GEN_AI_LATENCY_E2E = "gen_ai.latency.e2e"
    GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL = "gen_ai.latency.time_in_model_prefill"
    GEN_AI_LATENCY_TIME_IN_MODEL_DECODE = "gen_ai.latency.time_in_model_decode"
    GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE = "gen_ai.latency.time_in_model_inference"

    # The following attributes are excluded from the open-source version.
    GEN_AI_REQUEST_MIN_P = "gen_ai.request.min_p"
    GEN_AI_REQUEST_REPETITION_PENALTY = "gen_ai.request.repetition_penalty"
    GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD = (
         "gen_ai.latency.time_in_model_forward")
    GEN_AI_REQUEST_TRACE_LEVEL = "gen_ai.request.trace_level"

    # trace_level_2
    GEN_AI_LATENCY_PER_TOKEN_GENERATION_TIME = "gen_ai.latency.per_token_generation_time"
    GEN_AI_LATENCY_PER_TOKEN_SCHEDULED_TIME = "gen_ai.latency.per_token_scheduled_time"
    GEN_AI_ITERATION_PER_TOKEN_BATCH_SIZE = "gen_ai.iteration.per_token_batch_size"
    GEN_AI_ITERATION_PER_TOKEN_WAITING_SIZE = "gen_ai.iteration.per_token_waiting_size"
    GEN_AI_ITERATION_PER_TOKEN_TOTAL_TOKENS = "gen_ai.iteration.per_token_total_tokens"
    GEN_AI_ITERATION_PER_TOKEN_CACHED_TOKENS = "gen_ai.iteration.per_token_cached_tokens"
    GEN_AI_RESPONSE_PER_TOKEN_CANDIDATE_DECODED_TOKENS = "gen_ai.response.per_token_candidate_decoded_tokens"
    GEN_AI_RESPONSE_PER_TOKEN_CANDIDATE_TOKEN_IDS = "gen_ai.response.per_token_candidate_token_ids"
    GEN_AI_RESPONSE_PER_TOKEN_CANDIDATE_TOKENS_LOGPROBS = "gen_ai.response.per_token_candidate_tokens_logprobs"

    SOFA_TRACE_ID = "Parent-TraceId"
    SOFA_RPC_ID = "Parent-RpcId"
    REQUEST_ID = "alipay.aicloud.request_id"
    API_KEY_ID = "alipay.aicloud.api_key_id"
    POD_IP = "alipay.base.ip"
    POD_NAME = "alipay.base.pod_name"
    HOSTNAME = "alipay.base.host"
    IDC = "alipay.base.idc"
    MODEL_SERVICE_ID = "alipay.aicloud.model_service_id"
    MODEL_INSTANCE_ID = "alipay.aicloud.model_instance_id"
    MODEL_INSTANCE_NAME = "alipay.aicloud.model_instance_name"
    APP_NAME = "alipay.aicloud.app_name"
    ALIPAY_LATENCY_TIME_IN_API_SERVER = "alipay.aicloud.time_in_api_server"
    ALIPAY_LATENCY_TIME_IN_INPUT_PROCESSING = "alipay.aicloud.time_in_input_processing"
    ALIPAY_LATENCY_TIME_IN_OUTPUT_QUEUE = "alipay.aicloud.time_in_output_queue"
    ALIPAY_LATENCY_TIME_IN_OUTPUT_PROCESSING = "alipay.aicloud.time_in_output_processing"
    ALIPAY_REQUEST_PARAMS = "alipay.aicloud.request_params"
    ALIPAY_REQ_METRIC = "alipay.aicloud.req_metric"
