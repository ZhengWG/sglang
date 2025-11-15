"""
Minimal HTTP load balancer for prefill and decode servers for testing.
"""

import asyncio
import copy
import ipaddress
import logging
import random
import urllib
from dataclasses import dataclass
from http import HTTPStatus
from itertools import chain
from typing import List, Optional

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, Response, StreamingResponse
from sglang_router.router_args import RouterArgs

try:
    from sglang.srt.tracing.trace import (
        process_tracing_init,
        trace_get_remote_propagate_context,
        trace_req_finish,
        trace_req_start,
        trace_set_thread_info,
        trace_slice_end,
        trace_slice_start,
    )

    trace_package_imported = True
except ImportError:
    trace_package_imported = False

logger = logging.getLogger(__name__)

AIOHTTP_STREAM_READ_CHUNK_SIZE = (
    1024 * 64
)  # 64KB, to prevent aiohttp's "Chunk too big" error


@dataclass
class PrefillConfig:
    url: str
    bootstrap_port: Optional[int]


@dataclass
class VisionConfig:
    url: str
    bootstrap_port: Optional[int]


def maybe_wrap_ipv6_address(address: str) -> str:
    try:
        ipaddress.IPv6Address(address)
        return f"[{address}]"
    except ValueError:
        return address


class MiniLoadBalancer:
    def __init__(
        self,
        router_args: RouterArgs,
    ):
        self._validate_router_args(router_args)

        self.host = router_args.host
        self.port = router_args.port
        self.timeout = router_args.request_timeout_secs
        self.prefill_configs: List[PrefillConfig] = [
            PrefillConfig(url=url, bootstrap_port=bootstrap_port)
            for url, bootstrap_port in router_args.prefill_urls
        ]
        self.prefill_urls = [cfg.url for cfg in self.prefill_configs]
        self.prefill_bootstrap_ports = [cfg.bootstrap_port for cfg in self.prefill_configs]
        self.decode_urls = router_args.decode_urls
        self.vision_configs: List[VisionConfig] = [
            VisionConfig(url=url, bootstrap_port=bootstrap_port)
            for url, bootstrap_port in router_args.vision_urls
        ]
        self.vision_urls = [cfg.url for cfg in self.vision_configs]
        self.vision_bootstrap_ports = [cfg.bootstrap_port for cfg in self.vision_configs]
        self.enable_multimodal_disagg = router_args.enable_multimodal_disagg
        self._vision_index = 0
        self._prefill_index = 0
        self.otlp_traces_endpoint = router_args.otlp_traces_endpoint
        self.enable_trace = router_args.enable_trace
        if self.enable_trace and not trace_package_imported:
            logger.warning(
                "Tracing is not supported in this environment. Please install sglang."
            )
            self.enable_trace = False

    def _validate_router_args(self, router_args: RouterArgs):
        logger.warning(
            "\x1b[33mMiniLB is only for debugging purposes, it only supports random policy!\033[0m"
        )

        # NOTE: too many arguments unsupported, just validate some important ones
        if router_args.policy != "random":
            logger.warning("[MiniLB] Overriding policy to random")
            router_args.policy = "random"

        if not router_args.pd_disaggregation:
            raise ValueError("MiniLB only supports PD disaggregation mode")

        if len(router_args.prefill_urls) == 0:
            raise ValueError("MiniLB requires at least one prefill server")
        if (
            not router_args.enable_multimodal_disagg
            and len(router_args.decode_urls) == 0
        ):
            raise ValueError("MiniLB requires at least one decode server")
        if router_args.enable_multimodal_disagg and len(router_args.vision_urls) == 0:
            raise ValueError(
                "MiniLB multimodal mode requires at least one vision server"
            )

    def start(self):
        global lb
        lb = self
        if self.enable_trace:
            process_tracing_init(self.otlp_traces_endpoint, "sglang")
            trace_set_thread_info("Mini lb")
        uvicorn.run(app, host=self.host, port=self.port)

    def select_pair(self):
        assert len(self.prefill_urls) > 0, "No prefill servers available"
        if self.enable_multimodal_disagg:
            if len(self.vision_urls) == 0:
                raise HTTPException(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    detail="No vision servers available",
                )
            vision_config = self.vision_configs[self._vision_index]
            prefill_config = self.prefill_configs[self._prefill_index]
            self._vision_index = (self._vision_index + 1) % len(self.vision_configs)
            self._prefill_index = (self._prefill_index + 1) % len(self.prefill_configs)
            return (
                vision_config.url,
                vision_config.bootstrap_port,
                prefill_config.url,
            )

        if len(self.decode_urls) == 0:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="No decode servers available",
            )

        pidx = random.randint(0, len(self.prefill_urls) - 1)
        didx = random.randint(0, len(self.decode_urls) - 1)
        return (
            self.prefill_urls[pidx],
            self.prefill_bootstrap_ports[pidx],
            self.decode_urls[didx],
        )

    async def generate(
        self, modified_request, prefill_server, decode_server, endpoint
    ) -> ORJSONResponse:
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=self.timeout
            )  # Add timeout for request reliability
        ) as session:
            headers = {}
            bootstrap_room_list = []
            if self.enable_trace:
                bootstrap_room_list = (
                    modified_request["bootstrap_room"]
                    if isinstance(modified_request["bootstrap_room"], list)
                    else [modified_request["bootstrap_room"]]
                )
                trace_context = trace_get_remote_propagate_context(bootstrap_room_list)
                headers = {"trace_context": trace_context}

            tasks = [
                session.post(
                    f"{prefill_server}/{endpoint}",
                    json=modified_request,
                    headers=headers,
                ),
                session.post(
                    f"{decode_server}/{endpoint}",
                    json=modified_request,
                    headers=headers,
                ),
            ]

            for bootstrap_room in bootstrap_room_list:
                trace_slice_end("mini_lb_launch", bootstrap_room, auto_next_anon=True)

            # Wait for both responses to complete. Prefill should end first.
            prefill_response, decode_response = await asyncio.gather(*tasks)

            if "return_logprob" in modified_request:

                prefill_json = await prefill_response.json()
                ret_json = await decode_response.json()

                # merge `meta_info.input_token_logprobs` from prefill to decode
                if "meta_info" in ret_json:
                    if "input_token_logprobs" in ret_json["meta_info"]:
                        ret_json["meta_info"]["input_token_logprobs"] = (
                            prefill_json["meta_info"]["input_token_logprobs"]
                            + ret_json["meta_info"]["input_token_logprobs"]
                        )
            else:
                ret_json = await decode_response.json()

            for bootstrap_room in bootstrap_room_list:
                trace_slice_end(
                    "wait_PD_finish",
                    bootstrap_room,
                    thread_finish_flag=True,
                )
                trace_req_finish(bootstrap_room)

            return ORJSONResponse(
                content=ret_json,
                status_code=decode_response.status,
            )

    async def generate_stream(
        self, modified_request, prefill_server, decode_server, endpoint="generate"
    ):
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async def stream_results():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self.timeout
                )  # Add timeout for request reliability
            ) as session:
                # Create the tasks for both prefill and decode requests
                headers = {}
                bootstrap_room_list = []
                if self.enable_trace:
                    bootstrap_room_list = (
                        modified_request["bootstrap_room"]
                        if isinstance(modified_request["bootstrap_room"], list)
                        else [modified_request["bootstrap_room"]]
                    )
                    trace_context = trace_get_remote_propagate_context(
                        bootstrap_room_list
                    )
                    headers = {"trace_context": trace_context}

                tasks = [
                    session.post(
                        f"{prefill_server}/{endpoint}",
                        json=modified_request,
                        headers=headers,
                    ),
                    session.post(
                        f"{decode_server}/{endpoint}",
                        json=modified_request,
                        headers=headers,
                    ),
                ]

                for bootstrap_room in bootstrap_room_list:
                    trace_slice_end(
                        "mini_lb_launch", bootstrap_room, auto_next_anon=True
                    )
                # Wait for both responses to complete. Since this is streaming, they return immediately.
                prefill_response, decode_response = await asyncio.gather(*tasks)

                if modified_request.get("return_logprob", False):
                    prefill_chunks = []
                    async for chunk in prefill_response.content:
                        prefill_chunks.append(chunk)

                    first_prefill_chunk = (
                        prefill_chunks[0].decode("utf-8")[5:].strip("\n")
                    )
                    first_prefill_chunk_json = orjson.loads(first_prefill_chunk)

                    async for chunk in decode_response.content:
                        # Note: This is inefficient
                        # merge prefill input_token_logprobs, output_token_logprobs to decode
                        decoded_chunk = chunk.decode("utf-8")
                        if (
                            decoded_chunk
                            and decoded_chunk.startswith("data:")
                            and "[DONE]" not in decoded_chunk
                        ):
                            ret_json = orjson.loads(decoded_chunk[5:].strip("\n"))
                            ret_json["meta_info"]["input_token_logprobs"] = (
                                first_prefill_chunk_json["meta_info"][
                                    "input_token_logprobs"
                                ]
                                + ret_json["meta_info"]["input_token_logprobs"]
                            )

                            yield b"data: " + orjson.dumps(ret_json) + b"\n\n"
                        else:
                            yield chunk
                else:
                    async for chunk in decode_response.content.iter_chunked(
                        AIOHTTP_STREAM_READ_CHUNK_SIZE
                    ):
                        yield chunk

            for bootstrap_room in bootstrap_room_list:
                trace_slice_end(
                    "wait_PD_finish",
                    bootstrap_room,
                    thread_finish_flag=True,
                )
                trace_req_finish(bootstrap_room)

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )

    async def _check_single_response(self, response: aiohttp.ClientResponse):
        try:
            response_json = await response.json()
            if response.status != 200:
                return False, response_json
            return True, None
        except Exception as e:
            return False, f"Response check failed: {e}"

    async def multimodal_generate(
        self,
        vision_modified_request,
        prefill_modified_request,
        vision_server,
        prefill_server,
        endpoint="v1/chat/completions",
    ) -> ORJSONResponse:
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=max(self.timeout, 3600))
        ) as session:
            headers = {}
            bootstrap_room_list = []
            if self.enable_trace:
                bootstrap_room_list = (
                    prefill_modified_request["bootstrap_room"]
                    if isinstance(prefill_modified_request["bootstrap_room"], list)
                    else [prefill_modified_request["bootstrap_room"]]
                )
                trace_context = trace_get_remote_propagate_context(bootstrap_room_list)
                headers = {"trace_context": trace_context}

            vision_task = asyncio.create_task(
                session.post(
                    f"{vision_server}/{endpoint}",
                    json=vision_modified_request,
                    headers=headers,
                )
            )
            ret_task = asyncio.create_task(
                session.post(
                    f"{prefill_server}/{endpoint}",
                    json=prefill_modified_request,
                    headers=headers,
                )
            )

            for bootstrap_room in bootstrap_room_list:
                trace_slice_end("mini_lb_launch", bootstrap_room, auto_next_anon=True)

            vision_response = await vision_task
            is_success, error_message = await self._check_single_response(
                vision_response
            )
            if not is_success:
                await session.post(
                    f"{prefill_server}/abort_request",
                    json={"rid": prefill_modified_request["rid"]},
                )
                logger.info(
                    "Abort prefill request due to failed vision response: %s",
                    prefill_modified_request["rid"],
                )
                raise HTTPException(
                    status_code=vision_response.status, detail=error_message
                )

            ret_response = await ret_task

            for bootstrap_room in bootstrap_room_list:
                trace_slice_end(
                    "wait_PD_finish",
                    bootstrap_room,
                    thread_finish_flag=True,
                )
                trace_req_finish(bootstrap_room)

            return ORJSONResponse(
                content=await ret_response.json(),
                status_code=ret_response.status,
            )

    async def multimodal_generate_stream(
        self,
        vision_modified_request,
        prefill_modified_request,
        vision_server,
        prefill_server,
        endpoint="v1/chat/completions",
    ):
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async def stream_results():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=max(self.timeout, 3600))
            ) as session:
                headers = {}
                bootstrap_room_list = []
                if self.enable_trace:
                    bootstrap_room_list = (
                        prefill_modified_request["bootstrap_room"]
                        if isinstance(prefill_modified_request["bootstrap_room"], list)
                        else [prefill_modified_request["bootstrap_room"]]
                    )
                    trace_context = trace_get_remote_propagate_context(
                        bootstrap_room_list
                    )
                    headers = {"trace_context": trace_context}

                vision_task = asyncio.create_task(
                    session.post(
                        f"{vision_server}/{endpoint}",
                        json=vision_modified_request,
                        headers=headers,
                    )
                )
                ret_task = asyncio.create_task(
                    session.post(
                        f"{prefill_server}/{endpoint}",
                        json=prefill_modified_request,
                        headers=headers,
                    )
                )

                for bootstrap_room in bootstrap_room_list:
                    trace_slice_end("mini_lb_launch", bootstrap_room, auto_next_anon=True)

                vision_response = await vision_task
                is_success, error_message = await self._check_single_response(
                    vision_response
                )
                if not is_success:
                    await session.post(
                        f"{prefill_server}/abort_request",
                        json={"rid": prefill_modified_request["rid"]},
                    )
                    logger.info(
                        "Abort prefill request due to failed vision response: %s",
                        prefill_modified_request["rid"],
                    )
                    raise HTTPException(
                        status_code=vision_response.status, detail=error_message
                    )

                ret_response = await ret_task

                if prefill_modified_request.get("return_logprob", False):
                    async for chunk in ret_response.content:
                        decoded_chunk = chunk.decode("utf-8")
                        if (
                            decoded_chunk
                            and decoded_chunk.startswith("data:")
                            and "[DONE]" not in decoded_chunk
                        ):
                            ret_json = orjson.loads(decoded_chunk[5:].strip("\n"))
                            yield b"data: " + orjson.dumps(ret_json) + b"\n\n"
                        else:
                            yield chunk
                else:
                    async for chunk in ret_response.content:
                        yield chunk

            for bootstrap_room in bootstrap_room_list:
                trace_slice_end(
                    "wait_PD_finish",
                    bootstrap_room,
                    thread_finish_flag=True,
                )
                trace_req_finish(bootstrap_room)

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )


app = FastAPI()
lb: Optional[MiniLoadBalancer] = None


@app.get("/health")
async def health_check():
    return Response(status_code=200)


@app.get("/health_generate")
async def health_generate():
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(lb.prefill_urls, lb.decode_urls):
            tasks.append(session.get(f"{server}/health_generate"))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.post("/flush_cache")
async def flush_cache():
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(lb.prefill_urls, lb.decode_urls):
            tasks.append(session.post(f"{server}/flush_cache"))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.get("/get_server_info")
async def get_server_info():
    prefill_infos = []
    decode_infos = []
    all_internal_states = []

    async with aiohttp.ClientSession() as session:
        for server in lb.prefill_urls:
            server_info = await session.get(f"{server}/get_server_info")
            prefill_infos.append(await server_info.json())
        for server in lb.decode_urls:
            server_info = await session.get(f"{server}/get_server_info")
            info_json = await server_info.json()
            decode_infos.append(info_json)
            # Extract internal_states from decode servers
            if "internal_states" in info_json:
                all_internal_states.extend(info_json["internal_states"])

    # Return format expected by bench_one_batch_server.py
    if all_internal_states:
        return {
            "internal_states": all_internal_states,
            "prefill": prefill_infos,
            "decode": decode_infos,
        }
    else:
        # Fallback with dummy data if no internal states found
        return {
            "internal_states": [
                {
                    "last_gen_throughput": 0.0,
                    "avg_spec_accept_length": None,
                }
            ],
            "prefill": prefill_infos,
            "decode": decode_infos,
        }


@app.get("/get_model_info")
async def get_model_info():
    if not lb or not lb.prefill_urls:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="There is no server registered",
        )

    target_server_url = lb.prefill_urls[0]
    endpoint_url = f"{target_server_url}/get_model_info"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint_url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=HTTPStatus.BAD_GATEWAY,
                        detail=(
                            f"Failed to get model info from {target_server_url}"
                            f"Status: {response.status}, Response: {error_text}"
                        ),
                    )

                model_info_json = await response.json()
                return ORJSONResponse(content=model_info_json)

        except aiohttp.ClientError as e:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail=f"Failed to get model info from backend",
            )


@app.post("/generate")
async def handle_generate_request(request_data: dict):
    prefill_server, bootstrap_port, decode_server = lb.select_pair()

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
    modified_request = request_data.copy()

    batch_size = _get_request_batch_size(modified_request)
    if batch_size is not None:
        modified_request.update(
            {
                "bootstrap_host": [hostname] * batch_size,
                "bootstrap_port": [bootstrap_port] * batch_size,
                "bootstrap_room": [
                    _generate_bootstrap_room() for _ in range(batch_size)
                ],
            }
        )
    else:
        modified_request.update(
            {
                "bootstrap_host": hostname,
                "bootstrap_port": bootstrap_port,
                "bootstrap_room": _generate_bootstrap_room(),
            }
        )

    if request_data.get("stream", False):
        return await lb.generate_stream(
            modified_request, prefill_server, decode_server, "generate"
        )
    else:
        return await lb.generate(
            modified_request, prefill_server, decode_server, "generate"
        )


async def _forward_to_backend(request_data: dict, endpoint_name: str):
    prefill_server, bootstrap_port, decode_server = lb.select_pair()

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
    modified_request = request_data.copy()
    modified_request.update(
        {
            "bootstrap_host": hostname,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": _generate_bootstrap_room(),
        }
    )

    if request_data.get("stream", False):
        return await lb.generate_stream(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
        )
    else:
        return await lb.generate(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
        )


async def _forward_to_backend_multimodal(request_data: dict, endpoint_name: str):
    if endpoint_name != "v1/chat/completions":
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Multimodal disaggregation only supports /v1/chat/completions",
        )

    vision_server, bootstrap_port, prefill_server = lb.select_pair()

    parsed_url = urllib.parse.urlparse(vision_server)
    hostname = maybe_wrap_ipv6_address(parsed_url.hostname)

    vision_modified_request = copy.deepcopy(request_data)
    language_modified_request = copy.deepcopy(request_data)

    bootstrap_room = request_data.get("bootstrap_room")
    if bootstrap_room is None:
        bootstrap_room = _generate_bootstrap_room()

    rid = request_data["rid"] if "rid" in request_data else str(bootstrap_room)

    vision_modified_request.update(
        {
            "bootstrap_host": hostname,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": bootstrap_room,
            "rid": rid,
            "stream": False,
        }
    )
    language_modified_request.update(
        {
            "bootstrap_host": hostname,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": bootstrap_room,
            "rid": rid,
        }
    )

    for message in language_modified_request.get("messages", []):
        if isinstance(message.get("content"), list):
            text_content = [
                content for content in message["content"] if content.get("type") == "text"
            ]
            message["content"] = text_content

    if request_data.get("stream", False):
        return await lb.multimodal_generate_stream(
            vision_modified_request,
            language_modified_request,
            vision_server,
            prefill_server,
            endpoint=endpoint_name,
        )
    return await lb.multimodal_generate(
        vision_modified_request,
        language_modified_request,
        vision_server,
        prefill_server,
        endpoint=endpoint_name,
    )


@app.post("/v1/chat/completions")
async def handle_chat_completion_request(request_data: dict):
    if lb.enable_multimodal_disagg:
        return await _forward_to_backend_multimodal(
            request_data, "v1/chat/completions"
        )
    return await _forward_to_backend(request_data, "v1/chat/completions")


@app.post("/v1/completions")
async def handle_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/completions")


def _generate_bootstrap_room():
    bootstrap_room = random.randint(0, 2**63 - 1)
    if lb.enable_trace:
        trace_req_start(bootstrap_room, bootstrap_room, role="router")
        trace_slice_start("mini_lb_launch", bootstrap_room)
    return bootstrap_room


# We may utilize `GenerateReqInput`'s logic later
def _get_request_batch_size(request):
    if (text := request.get("text")) is not None:
        return None if isinstance(text, str) else len(text)
    if (input_ids := request.get("input_ids")) is not None:
        return None if isinstance(input_ids[0], int) else len(input_ids)
    return None


@app.get("/v1/models")
async def get_models():
    prefill_server = lb.prefill_urls[0]  # Get the first prefill server
    async with aiohttp.ClientSession() as session:
        try:
            response = await session.get(f"{prefill_server}/v1/models")
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Prefill server error: Status {response.status}",
                )
            return ORJSONResponse(content=await response.json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
