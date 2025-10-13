# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/backend_request_func.py
# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/benchmark_serving.py

"""
Benchmark online serving with dynamic requests.

Usage:
python3 -m sglang.bench_serving --backend sglang --num-prompt 10

python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 1024 --random-range-ratio 0.5
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate-range 1,2,4,8,16,32 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --multi
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
import traceback
import warnings
from argparse import ArgumentParser
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import requests
from data_processing import MsgContent, SampleOutput, get_dataset
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from sglang.bench_serving import get_tokenizer, remove_prefix, set_ulimit

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)

global args

# Global request id counter for tracing
_REQUEST_ID_COUNTER = 0


def next_request_id() -> int:
    """Generate a monotonically increasing request id for trace labeling."""
    global _REQUEST_ID_COUNTER
    _REQUEST_ID_COUNTER += 1
    return _REQUEST_ID_COUNTER


@dataclass
class RequestFuncInput:
    prompts: List[Tuple[MsgContent, int, int]]
    api_url: str
    model: str
    lora_name: str
    extra_request_body: Dict[str, Any]

    # For multiturn chat, store the context
    prev_messages: List = field(default_factory=list)
    finished_prompts: int = 0
    # Trace metadata, assigned when queued and yielded
    request_id: int = -1
    yield_ts: float = 0.0


@dataclass
class RequestFuncOutput:
    generated_text: List[str] = field(default_factory=list)
    prompt_len: List[int] = field(default_factory=list)
    output_len: List[int] = field(default_factory=list)
    latency: List[float] = field(default_factory=list)
    ttft: List[float] = field(default_factory=list)
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies

    success: bool = False
    error: str = ""
    # Per-request stage timestamps: {"request_id", "yield", "post", "ttft", "end"}
    trace: Dict[str, float] = field(default_factory=dict)


# set ignore_eos True by default
async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    queue: asyncio.Queue,
    tokenizer: PreTrainedTokenizerBase,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "completions"
    ), "OpenAI Completions API URL must end with 'completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "temperature": 0.0,
            "best_of": 1,
            "stream": not args.disable_stream,
            "stream_options": {"include_usage": True},
            "ignore_eos": not args.disable_ignore_eos,
            **request_func_input.extra_request_body,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()

        prompt_idx = request_func_input.finished_prompts
        messages = request_func_input.prev_messages
        prompt, input_len, max_tokens = request_func_input.prompts[prompt_idx]
        prompt_len = sum(
            prompt[1] + prompt[2]  # input_len + output_len
            for prompt in request_func_input.prompts[:prompt_idx]
        )
        prompt_len += input_len

        # Messages
        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )
        payload["messages"] = messages
        payload["max_tokens"] = max_tokens

        # output.prompt_len = request_func_input.prompt_len
        # print(payload)

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        # Trace timestamps for this request
        rid = request_func_input.request_id
        yield_ts = getattr(request_func_input, "yield_ts", 0.0)
        post_ts: Optional[float] = None
        ttft_ts: Optional[float] = None
        end_ts: Optional[float] = None
        try:
            post_ts = time.perf_counter()
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    actual_prompt_len = prompt_len - 1
                    actual_output_len = 0
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)
                            timestamp = time.perf_counter()
                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["usage"] is not None and len(data["usage"]) > 0:
                                actual_prompt_len = data["usage"]["prompt_tokens"]
                                actual_output_len = data["usage"]["completion_tokens"]
                                continue
                            delta = data["choices"][0]["delta"]

                            if delta.get("content", None):
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    ttft_ts = timestamp
                                    output.ttft.append(ttft)

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                generated_text += delta["content"]
                            most_recent_timestamp = timestamp

                    output.prompt_len.append(actual_prompt_len)  # truncate <s>
                    output.output_len.append(actual_output_len)
                    output.generated_text.append(generated_text)
                    output.success = True
                    output.latency.append(latency)
                    end_ts = st + latency

                    # Prepare for the new request
                    request_func_input.prompts[prompt_idx] = (
                        prompt,
                        input_len,
                        actual_output_len,  # changes from max_tokens to output_len
                    )
                    prompt_idx += 1
                    messages.append(
                        {
                            "role": "assistant",
                            "content": generated_text,
                        }
                    )

                    # Move the new request to the end of the queue
                    if prompt_idx < len(request_func_input.prompts):
                        request_func_input.finished_prompts = prompt_idx
                        request_func_input.prev_messages = messages
                        # Assign a new request id for the next prompt in this conversation
                        request_func_input.request_id = next_request_id()
                        request_func_input.yield_ts = 0.0
                        await queue.put(request_func_input)
                else:
                    output.error = response.reason or ""
                    output.success = False
                    end_ts = time.perf_counter()
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            end_ts = time.perf_counter()

        # Record trace timestamps for this request
        output.trace = {
            "request_id": float(rid) if rid is not None else -1.0,
            "yield": float(yield_ts) if yield_ts else 0.0,
            "post": float(post_ts) if post_ts else 0.0,
            "ttft": float(ttft_ts) if ttft_ts else 0.0,
            "end": float(end_ts) if end_ts else 0.0,
        }

    if pbar:
        pbar.update(1)
    return output


async def async_request_profile(api_url: str) -> RequestFuncOutput:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        output = RequestFuncOutput()
        try:
            async with session.post(url=api_url) as response:
                if response.status == 200:
                    output.success = True
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    return output


ASYNC_REQUEST_FUNCS = {
    "sglang": async_request_openai_completions,
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
}


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    total_output_retokenized: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    output_throughput_retokenized: float
    total_throughput: float
    total_throughput_retokenized: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p90_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p90_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p90_itl_ms: float
    p99_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    std_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    concurrency: float


async def get_requests(
    input_requests_queue: asyncio.Queue,
    request_rate: float,
    num_actual_requests: int,
) -> AsyncGenerator[RequestFuncInput, None]:
    for _ in range(num_actual_requests):
        try:
            request = await asyncio.wait_for(
                input_requests_queue.get(), timeout=300
            )  # Wait for 5 minites then abort
        except Exception as e:
            print(f"exception: {e}")
            break

        # Mark the time when this request is yielded to the worker
        request.yield_ts = time.perf_counter()
        yield request

        if request_rate == float("inf"):
            continue

        interval = np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(interval)


def calculate_metrics(
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    backend: str,
) -> Tuple[BenchmarkMetrics, List[int]]:
    output_lens: List[int] = []
    retokenized_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2e_latencies: List[float] = []
    output_success = 0
    for i in range(len(outputs)):
        if outputs[i].success:
            output_success += 1
            assert len(outputs[i].generated_text) == len(outputs[i].latency)
            assert len(outputs[i].generated_text) == len(outputs[i].ttft)
            for j in range(len(outputs[i].generated_text)):
                output_len = outputs[i].output_len[j]
                output_lens.append(output_len)
                retokenized_output_len = len(
                    tokenizer.encode(
                        outputs[i].generated_text[j], add_special_tokens=False
                    )
                )
                retokenized_output_lens.append(retokenized_output_len)
                total_input += outputs[i].prompt_len[j]
                if output_len > 1:
                    tpots.append(
                        (outputs[i].latency[j] - outputs[i].ttft[j]) / (output_len - 1)
                    )

                completed += 1
            itls += outputs[i].itl
            ttfts += outputs[i].ttft
            e2e_latencies += outputs[i].latency

        else:
            output_lens.append(0)
            retokenized_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(output_lens),
        total_output_retokenized=sum(retokenized_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(output_lens) / dur_s,
        output_throughput_retokenized=sum(retokenized_output_lens) / dur_s,
        total_throughput=(total_input + sum(output_lens)) / dur_s,
        total_throughput_retokenized=(total_input + sum(retokenized_output_lens))
        / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p90_ttft_ms=np.percentile(ttfts or 0, 90) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p90_tpot_ms=np.percentile(tpots or 0, 90) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p90_itl_ms=np.percentile(itls or 0, 90) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000,
        median_e2e_latency_ms=np.median(e2e_latencies) * 1000,
        std_e2e_latency_ms=np.std(e2e_latencies) * 1000,
        p99_e2e_latency_ms=np.percentile(e2e_latencies, 99) * 1000,
        concurrency=np.sum(e2e_latencies) / dur_s,
    )

    return metrics, output_lens


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: SampleOutput,
    request_rate: float,
    max_concurrency: Optional[int],
    disable_tqdm: bool,
    lora_name: str,
    extra_request_body: Dict[str, Any],
    profile: bool,
    enable_shared_prefix: bool,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Limit concurrency
    # From https://github.com/vllm-project/vllm/pull/9390
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, queue, tokenizer, pbar):
        if semaphore is None:
            return await request_func(
                request_func_input=request_func_input,
                queue=queue,
                tokenizer=tokenizer,
                pbar=pbar,
            )
        async with semaphore:
            return await request_func(
                request_func_input=request_func_input,
                queue=queue,
                tokenizer=tokenizer,
                pbar=pbar,
            )

    num_actual_requests = sum(len(r) for r in input_requests)
    print(f"Num of shared prefixes or conversations: {len(input_requests)}")
    print(f"Num of total requests: {num_actual_requests}")

    # flatten the requests for shared prefix
    if enable_shared_prefix:
        input_requests = [[r] for requests in input_requests for r in requests]
    inputs_requests_queue = asyncio.Queue(maxsize=len(input_requests))
    print("Starting initial single prompt test run...")
    # NOTE: Just use the first request of the first conversation for warmup
    test_input = RequestFuncInput(
        model=model_id,
        prompts=input_requests[0][:1],
        api_url=api_url,
        lora_name=lora_name,
        extra_request_body=extra_request_body,
    )
    test_output = await request_func(
        request_func_input=test_input, queue=inputs_requests_queue, tokenizer=tokenizer
    )
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        print("Initial test run completed. Starting main benchmark run...")

    # Check the states
    assert inputs_requests_queue.empty()

    # Flush cache
    if "sglang" in backend:
        requests.post(base_url + "/flush_cache")

    time.sleep(1.0)

    # Start profiler
    if profile:
        print("Starting profiler...")
        profile_output = await async_request_profile(
            api_url=base_url + "/start_profile"
        )
        if profile_output.success:
            print("Profiler started")

    for request in input_requests:
        request_func_input = RequestFuncInput(
            model=model_id,
            prompts=request,
            api_url=api_url,
            lora_name=lora_name,
            extra_request_body=extra_request_body,
        )
        # Assign initial id for this request (per prompt)
        request_func_input.request_id = next_request_id()
        inputs_requests_queue.put_nowait(request_func_input)
    if (
        not args.enable_multiturn
        and not args.enable_shared_prefix
        and not args.dataset_name == "generated-shared-prefix"
    ):
        assert len(input_requests) == num_actual_requests

    pbar = None if disable_tqdm else tqdm(total=num_actual_requests)

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for request in get_requests(
        inputs_requests_queue, request_rate, num_actual_requests
    ):
        tasks.append(
            asyncio.create_task(
                limited_request_func(
                    request_func_input=request,
                    queue=inputs_requests_queue,
                    tokenizer=tokenizer,
                    pbar=pbar,
                )
            )
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    # Stop profiler
    if profile:
        print("Stopping profiler...")
        profile_output = await async_request_profile(api_url=base_url + "/stop_profile")
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    # Compute metrics and print results
    benchmark_duration = time.perf_counter() - benchmark_start_time
    metrics, output_lens = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        backend=backend,
    )

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", backend))
    print("{:<40} {:<10}".format("Traffic request rate:", request_rate))
    print(
        "{:<40} {:<10}".format(
            "Max reqeuest concurrency:",
            max_concurrency if max_concurrency else "not set",
        )
    )
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10}".format(
            "Total generated tokens (retokenized):", metrics.total_output_retokenized
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", metrics.input_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total token throughput (tok/s):", metrics.total_throughput
        )
    )
    print("{:<40} {:<10.2f}".format("Concurrency:", metrics.concurrency))
    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=50, c="-"))
    print(
        "{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", metrics.mean_e2e_latency_ms)
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Median E2E Latency (ms):", metrics.median_e2e_latency_ms
        )
    )
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P90 TTFT (ms):", metrics.p90_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print(
        "{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-")
    )
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P90 TPOT (ms):", metrics.p90_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P90 ITL (ms):", metrics.p90_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("=" * 50)

    if (
        metrics.median_ttft_ms is not None
        and metrics.mean_itl_ms is not None
        and metrics.output_throughput is not None
    ):
        result = {
            # Arguments
            "backend": args.backend,
            "dataset_name": args.dataset_name,
            "request_rate": request_rate,
            "max_concurrency": max_concurrency,
            "fixed_output_len": args.fixed_output_len,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "random_range_ratio": args.random_range_ratio,
            # Results
            "duration": benchmark_duration,
            "completed": metrics.completed,
            "total_input_tokens": metrics.total_input,
            "total_output_tokens": metrics.total_output,
            "total_output_tokens_retokenized": metrics.total_output_retokenized,
            "request_throughput": metrics.request_throughput,
            "input_throughput": metrics.input_throughput,
            "output_throughput": metrics.output_throughput,
            "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
            "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
            "std_e2e_latency_ms": metrics.std_e2e_latency_ms,
            "p99_e2e_latency_ms": metrics.p99_e2e_latency_ms,
            "mean_ttft_ms": metrics.mean_ttft_ms,
            "median_ttft_ms": metrics.median_ttft_ms,
            "std_ttft_ms": metrics.std_ttft_ms,
            "p99_ttft_ms": metrics.p99_ttft_ms,
            "mean_tpot_ms": metrics.mean_tpot_ms,
            "median_tpot_ms": metrics.median_tpot_ms,
            "std_tpot_ms": metrics.std_tpot_ms,
            "p99_tpot_ms": metrics.p99_tpot_ms,
            "mean_itl_ms": metrics.mean_itl_ms,
            "median_itl_ms": metrics.median_itl_ms,
            "std_itl_ms": metrics.std_itl_ms,
            "p99_itl_ms": metrics.p99_itl_ms,
            "concurrency": metrics.concurrency,
            "input_throughput": metrics.input_throughput,
            "output_throughput": metrics.output_throughput,
            "fixed_output_len": args.fixed_output_len,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "random_range_ratio": args.random_range_ratio,
            "duration": benchmark_duration,
            "completed": metrics.completed,
        }
    else:
        print(f"Error running benchmark for request rate: {request_rate}")
        print("-" * 30)

    # Determine output file name
    if args.output_file:
        output_file_name = args.output_file
    else:
        now = datetime.now().strftime("%m%d")
        if args.dataset_name == "random":
            output_file_name = f"{args.backend}_{now}_{args.num_prompts}_{args.random_input_len}_{args.random_output_len}.jsonl"
        else:
            output_file_name = (
                f"{args.backend}_{now}_{args.num_prompts}_{args.dataset_name}.jsonl"
            )

    # Append results to a JSONL file
    with open(output_file_name, "a") as file:
        file.write(json.dumps(result) + "\n")

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "total_output_tokens_retokenized": metrics.total_output_retokenized,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "std_ttft_ms": metrics.std_ttft_ms,
        "p90_ttft_ms": metrics.p90_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "std_tpot_ms": metrics.std_tpot_ms,
        "p90_tpot_ms": metrics.p90_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "std_itl_ms": metrics.std_itl_ms,
        "p90_itl_ms": metrics.p90_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
        "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
    }

    # Collect per-request traces and plot a timeline
    try:
        traces = [o.trace for o in outputs if getattr(o, "trace", None)]
        if traces:
            def _plot_request_traces(traces_list: List[Dict[str, float]], output_path: str) -> None:
                try:
                    import matplotlib.pyplot as plt
                except Exception as e:
                    print(f"matplotlib is not available, skip plotting trace. err={e}")
                    return

                # Normalize time to the first yield
                valid_yields = [t.get("yield", 0.0) for t in traces_list if t.get("yield", 0.0) > 0.0]
                if not valid_yields:
                    print("No valid yield timestamps; skip plotting trace.")
                    return
                t0 = min(valid_yields)

                # Sort by request id then by yield time
                traces_sorted = sorted(traces_list, key=lambda t: (t.get("request_id", -1), t.get("yield", 0.0)))

                fig, ax = plt.subplots(figsize=(12, max(4, len(traces_sorted) * 0.12)))
                y_ticks = []
                y_labels = []
                for idx, t in enumerate(traces_sorted):
                    rid = int(t.get("request_id", -1))
                    y = idx
                    y_ticks.append(y)
                    y_labels.append(str(rid))

                    yld = t.get("yield", 0.0)
                    pst = t.get("post", 0.0)
                    ftt = t.get("ttft", 0.0)
                    end = t.get("end", 0.0)

                    # Convert to relative times
                    yld_r = yld - t0 if yld > 0 else None
                    pst_r = pst - t0 if pst > 0 else None
                    ftt_r = ftt - t0 if ftt > 0 else None
                    end_r = end - t0 if end > 0 else None

                    # Segments: yield->post, post->ttft, ttft->end
                    if yld_r is not None and pst_r is not None and pst_r >= yld_r:
                        ax.barh(y, pst_r - yld_r, left=yld_r, height=0.6, color="#b0b0b0", label="yield→post" if idx == 0 else None)
                    if pst_r is not None and ftt_r is not None and ftt_r >= pst_r:
                        ax.barh(y, ftt_r - pst_r, left=pst_r, height=0.6, color="#4ea5d9", label="post→ttft" if idx == 0 else None)
                    if ftt_r is not None and end_r is not None and end_r >= ftt_r:
                        ax.barh(y, end_r - ftt_r, left=ftt_r, height=0.6, color="#66bb6a", label="ttft→end" if idx == 0 else None)

                ax.set_xlabel("Time since first yield (s)")
                ax.set_ylabel("Request ID")
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels)
                ax.legend(loc="upper right")
                ax.set_title("Request timeline trace")
                fig.tight_layout()
                plt.savefig(output_path, dpi=150)
                plt.close(fig)

            base_name, _ = os.path.splitext(output_file_name)
            trace_png = f"{base_name}.trace.png"
            _plot_request_traces(traces, trace_png)
            result["trace_file"] = trace_png
            result["traces"] = traces
            print(f"Saved request trace chart to {trace_png}")
    except Exception as e:
        print(f"Failed to generate request trace plot: {e}")
    return result


def run_benchmark(args_: argparse.Namespace):
    global args
    args = args_

    # Set default value for max_concurrency if not present
    if not hasattr(args, "max_concurrency"):
        args.max_concurrency = None

    # Set global environments
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    extra_request_body = {}
    if args.extra_request_body:
        extra_request_body = json.loads(args.extra_request_body)

    # Set url
    if args.port is None:
        args.port = {
            "sglang": 30000,
            "lmdeploy": 23333,
            "vllm": 8000,
        }.get(args.backend, 30000)

    model_url = (
        f"{args.base_url}/v1/models"
        if args.base_url
        else f"http://{args.host}:{args.port}/v1/models"
    )

    if args.backend in ["sglang", "vllm", "lmdeploy"]:
        api_url = (
            f"{args.base_url}/v1/chat/completions"
            if args.base_url
            else f"http://{args.host}:{args.port}/v1/chat/completions"
        )
    base_url = (
        f"http://{args.host}:{args.port}" if args.base_url is None else args.base_url
    )

    # Get model name
    if args.model is None:
        if args.backend == "truss":
            print(
                "Please provide a model with `--model` when using truss backend. e.g. --model meta-llama/Llama-3.1-8B-Instruct"
            )
            sys.exit(1)
        try:
            response = requests.get(model_url)
            model_list = response.json().get("data", [])
            args.model = model_list[0]["id"] if model_list else None
        except Exception as e:
            print(f"Failed to fetch model from {model_url}. Error: {e}")
            print(
                "Please specify the correct host and port using `--host` and `--port`."
            )
            sys.exit(1)

    if args.model is None:
        print("No model specified or found. Please provide a model using `--model`.")
        sys.exit(1)

    # Dataset compatibility check
    if args.enable_multiturn:
        # TODO: Support multiturn for random
        if args.dataset_name not in ["sharegpt", "ultrachat", "loogle", "nextqa"]:
            print(
                "Multiturn conversation is only supported for sharegpt, ultrachat, loogle, and nextqa datasets."
            )
            sys.exit(1)

    if args.enable_shared_prefix:
        if args.dataset_name not in ["loogle", "nextqa"]:
            print("Shared prefix is only supported for loogle and nextqa datasets.")
            sys.exit(1)

    print(f"{args}\n")

    # Read dataset
    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    tokenizer = get_tokenizer(tokenizer_id)

    input_requests = get_dataset(args, tokenizer)

    return asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=args.request_rate,
            max_concurrency=args.max_concurrency,
            disable_tqdm=args.disable_tqdm,
            lora_name=args.lora_name,
            extra_request_body=extra_request_body,
            profile=args.profile,
            enable_shared_prefix=args.enable_shared_prefix,
        )
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        default="sglang",
        help="Must specify a backend, depending on the LLM Inference Engine.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Default host is 0.0.0.0."
    )
    parser.add_argument(
        "--port",
        type=int,
        help="If not set, the default port is configured according to its default value for different LLM Inference Engines.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=[
            "sharegpt",
            "random",
            "generated-shared-prefix",
            "ultrachat",
            "loogle",
            "nextqa",
        ],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--dataset-path", type=str, default="", help="Path to the dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name or path of the model. If not set, the default model will request /v1/models for conf.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer. If not set, using the model conf.",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        help="The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process. Default is 1000.",
    )
    parser.add_argument(
        "--fixed-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length from the dataset.",
    )
    parser.add_argument(
        "--sharegpt-context-len",
        type=int,
        default=None,
        help="The context length of the model for the ShareGPT dataset. Requests longer than the context length will be dropped.",
    )
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random dataset.",
    )
    parser.add_argument(
        "--random-output-len",
        default=1024,
        type=int,
        help="Number of output tokens per request, used only for random dataset.",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random dataset.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Use request rate range rather than single value.",
    )
    parser.add_argument(
        "--request-rate-range",
        type=str,
        default="2,34,2",
        help="Range of request rates in the format start,stop,step. Default is 2,34,2. It also supports a list of request rates, requiring the parameters to not equal three.",
    )
    parser.add_argument("--output-file", type=str, help="Output JSONL file name.")
    parser.add_argument(
        "--enable-multiturn",
        action="store_true",
        help="Enable multiturn chat for online serving benchmarking. "
        "This option is effective on the following datasets: "
        "sharegpt, ultrachat, loogle, nextqa",
    )
    parser.add_argument(
        "--enable-shared-prefix",
        action="store_true",
        help="Enable shared prefix for online serving benchmarking. "
        "This option is effective on the following datasets: "
        "loogle, nextqa",
    )

    parser.add_argument(
        "--disable-shuffle",
        action="store_true",
        help="Disable shuffling datasets. This is useful to generate stable output "
        "in benchmarking",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--disable-stream",
        action="store_true",
        help="Disable streaming mode.",
    )
    parser.add_argument(
        "--return-logprob",
        action="store_true",
        help="Return logprob.",
    )
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    parser.add_argument(
        "--disable-ignore-eos",
        action="store_true",
        help="Disable ignoring EOS.",
    )
    parser.add_argument(
        "--extra-request-body",
        metavar='{"key1": "value1", "key2": "value2"}',
        type=str,
        help="Append given JSON object to the request payload. You can use this to specify"
        "additional generate params like sampling params.",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply chat template",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "SGLANG_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--lora-name",
        type=str,
        default=None,
        help="The name of LoRA adapter",
    )

    group = parser.add_argument_group("generated-shared-prefix dataset arguments")
    group.add_argument(
        "--gsp-num-groups",
        type=int,
        default=64,
        help="Number of system prompt groups for generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-prompts-per-group",
        type=int,
        default=16,
        help="Number of prompts per system prompt group for generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-system-prompt-len",
        type=int,
        default=2048,
        help="Target length in tokens for system prompts in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-question-len",
        type=int,
        default=128,
        help="Target length in tokens for questions in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-output-len",
        type=int,
        default=256,
        help="Target length in tokens for outputs in generated-shared-prefix dataset",
    )
    # videos specific
    parser.add_argument(
        "--max-frames",
        type=int,
        default=sys.maxsize,
        help="The maximum number of frames to extract from each video. "
        "This option is specific to the nextqa dataset (video benchmark). ",
    )
    args = parser.parse_args()

    if args.enable_multiturn and args.enable_shared_prefix:
        parser.error(
            "--enable-multiturn and --enable-shared-prefix cannot be set at the same time."
        )

    run_benchmark(args)
