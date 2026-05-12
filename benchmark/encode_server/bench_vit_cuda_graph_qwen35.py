"""Benchmark: ViT CUDA Graph for Qwen3.5 / Qwen3-VL encoder.

Times ``Qwen3VLMoeVisionModel.forward(...)`` (the same vision module that
``encode_server.py`` invokes via ``model.get_image_feature``) with and
without ``SGLANG_VIT_ENABLE_CUDA_GRAPH``. The point is to validate that
the CUDA-graph-safe Triton vision RoPE introduced in
``triton_apply_rotary_pos_emb`` actually unlocks the existing
``ViTCudaGraphRunner`` path for Qwen3.5 and that doing so produces a
measurable end-to-end speedup.

The script:

1. Builds the vision module ONCE per env-var setting (random weights —
   we do not need ground-truth correctness, only baseline-vs-CG
   numerical closeness and per-call latency).
2. Warms up, then times N iterations per image-token bucket.
3. Captures first-graph-build cost separately for the CG run.
4. Writes a Markdown summary to ``bench_results.md``.

Requires a CUDA GPU + Triton. Skips with a clear message otherwise.

Example::

    python3 benchmark/encode_server/bench_vit_cuda_graph_qwen35.py \\
        --buckets 256,1024,4096 --warmup 30 --iters 200
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile(values, p):
    """Lightweight percentile (no numpy dep)."""
    if not values:
        return float("nan")
    s = sorted(values)
    if p <= 0:
        return s[0]
    if p >= 100:
        return s[-1]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


@dataclass
class BucketResult:
    name: str
    s_tokens: int
    median_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    capture_ms: float = 0.0


# ---------------------------------------------------------------------------
# Environment / SGLang init (lightweight — TP=1, no full launch_server)
# ---------------------------------------------------------------------------


def _init_distributed_single_process():
    """Bring up SGLang's distributed primitives in a single-process TP=1 setup.

    The vision model uses ``ColumnParallelLinear`` / ``RowParallelLinear``
    which require an initialized distributed environment even at TP=1.

    Mirrors the pattern used in
    ``test/registered/quant/test_gptqmodel_dynamic.py``.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29555))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        monkey_patch_vllm_parallel_state,
    )
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.server_args import (
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    try:
        if not torch.distributed.is_initialized():
            init_distributed_environment(
                backend="nccl",
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method="tcp://127.0.0.1:29555",
            )
            initialize_model_parallel(tensor_model_parallel_size=1)
            monkey_patch_vllm_parallel_state()
    except AssertionError:
        # Already initialized — ignore (consistent with sglang test conventions)
        pass

    # Some module constructors expect a global ServerArgs to be registered.
    sa = ServerArgs(model_path="dummy")
    set_global_server_args_for_scheduler(sa)
    try:
        initialize_dp_attention(sa, None)
    except Exception:
        # Older trees use a slightly different signature; the vision module
        # still constructs at TP=1 if DP-attention init is skipped.
        pass


def _make_vision_config():
    """Build a Qwen3-VL-style vision config that exercises real depths and
    head shapes. Defaults match Qwen3-VL-8B-Instruct's vision tower so the
    measurement is representative; can be overridden via env vars for quick
    runs.
    """
    from sglang.srt.configs.qwen3_vl import Qwen3VLVisionConfig

    cfg = Qwen3VLVisionConfig(
        depth=int(os.getenv("BENCH_VIT_DEPTH", "32")),
        hidden_size=int(os.getenv("BENCH_VIT_HIDDEN", "1152")),
        intermediate_size=int(os.getenv("BENCH_VIT_INTER", "4304")),
        num_heads=int(os.getenv("BENCH_VIT_HEADS", "16")),
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=int(os.getenv("BENCH_VIT_OUT_HIDDEN", "3584")),
        num_position_embeddings=2304,  # 48*48
        deepstack_visual_indexes=[7, 15, 23],
        hidden_act="silu",
    )
    return cfg


def _build_vit(dtype: torch.dtype):
    from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel

    cfg = _make_vision_config()
    vit = Qwen3VLMoeVisionModel(cfg, norm_eps=1e-6).to("cuda").to(dtype)
    vit.eval()
    return vit


def _make_grid_for_tokens(s_tokens: int, vit) -> torch.Tensor:
    """Pick a (1, t, h, w) grid yielding ~s_tokens patches."""
    msz = vit.spatial_merge_size
    # We want t*h*w ~= s_tokens; keep t=1 (image case), square-ish h,w.
    side = max(msz * 2, int((s_tokens) ** 0.5))
    side = (side // msz) * msz  # multiple of merge size
    # Recompute s
    h = w = side
    return torch.tensor([[1, h, w]], dtype=torch.int32, device="cuda")


def _make_pixel_values(grid_thw: torch.Tensor, vit) -> torch.Tensor:
    """Random pixel_values matching the grid contract used by patch_embed."""
    cfg = vit.patch_embed
    # patch_embed does view(-1, C, T, P, P), so total elements per token is
    # in_channels * temporal_patch_size * patch_size**2.
    elems_per_patch = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size ** 2
    n_patches = int(grid_thw[:, 0].prod() * grid_thw[:, 1] * grid_thw[:, 2])
    return torch.randn(n_patches, elems_per_patch, device="cuda", dtype=vit.dtype)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _time_one_call(vit, x, grid_thw) -> float:
    """Time a single forward in ms (CUDA-event accurate)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    with torch.inference_mode():
        out = vit(x, grid_thw)
    end.record()
    end.synchronize()
    # Force materialization
    out_size = int(out.shape[0])
    return start.elapsed_time(end), out_size, out


def _bench_bucket(
    vit, s_tokens: int, name: str, warmup: int, iters: int
) -> Tuple[BucketResult, torch.Tensor]:
    grid_thw = _make_grid_for_tokens(s_tokens, vit)
    x = _make_pixel_values(grid_thw, vit)

    # Warmup
    capture_ms = 0.0
    times = []
    for i in range(warmup):
        t0 = time.perf_counter()
        ms, _, last_out = _time_one_call(vit, x, grid_thw)
        t1 = time.perf_counter()
        if i == 0:
            # On the CG path the first call captures the graph; record wall-clock.
            capture_ms = (t1 - t0) * 1000.0
    # Timed iterations
    for _ in range(iters):
        ms, _, last_out = _time_one_call(vit, x, grid_thw)
        times.append(ms)
    res = BucketResult(
        name=name,
        s_tokens=s_tokens,
        median_ms=_percentile(times, 50),
        p95_ms=_percentile(times, 95),
        p99_ms=_percentile(times, 99),
        mean_ms=sum(times) / len(times),
        capture_ms=capture_ms,
    )
    return res, last_out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    if not torch.cuda.is_available():
        print(
            "[SKIP] No CUDA device found; this benchmark requires a CUDA GPU "
            "and Triton. See benchmark/encode_server/README.md for details."
        )
        return 0

    p = argparse.ArgumentParser()
    p.add_argument(
        "--buckets",
        default="256,1024,4096",
        help="Comma-separated patch-token counts to benchmark (small,medium,large).",
    )
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16"],
    )
    p.add_argument(
        "--output",
        default="bench_results.md",
        help="Where to write the Markdown summary.",
    )
    args = p.parse_args()

    bucket_sizes = [int(x.strip()) for x in args.buckets.split(",") if x.strip()]
    bucket_names = ["small", "medium", "large"]
    if len(bucket_sizes) != 3:
        bucket_names = [f"b{i}" for i in range(len(bucket_sizes))]

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"[bench] Initializing single-process distributed env...")
    _init_distributed_single_process()

    # ----- Baseline (no CG) -----
    os.environ["SGLANG_VIT_ENABLE_CUDA_GRAPH"] = "0"
    # Force re-import so that envs.SGLANG_VIT_ENABLE_CUDA_GRAPH is re-read and
    # any cached vision-rope-fn resolution is reset.
    for m in [
        "sglang.srt.layers.attention.vision",
        "sglang.srt.models.qwen3_vl",
        "sglang.srt.environ",
    ]:
        sys.modules.pop(m, None)
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[bench] Building ViT (baseline; CG OFF) dtype={args.dtype}...")
    vit_base = _build_vit(dtype)
    base_results: List[BucketResult] = []
    base_outputs = []
    for name, s in zip(bucket_names, bucket_sizes):
        print(f"[bench]  baseline {name} S={s} ...")
        res, out = _bench_bucket(vit_base, s, name, args.warmup, args.iters)
        base_results.append(res)
        base_outputs.append(out.detach().clone())

    del vit_base
    gc.collect()
    torch.cuda.empty_cache()

    # ----- CG ON -----
    os.environ["SGLANG_VIT_ENABLE_CUDA_GRAPH"] = "1"
    for m in [
        "sglang.srt.layers.attention.vision",
        "sglang.srt.models.qwen3_vl",
        "sglang.srt.environ",
    ]:
        sys.modules.pop(m, None)
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[bench] Building ViT (CG ON) dtype={args.dtype}...")
    vit_cg = _build_vit(dtype)
    cg_results: List[BucketResult] = []
    cg_outputs = []
    for name, s in zip(bucket_names, bucket_sizes):
        print(f"[bench]  CG-ON   {name} S={s} ...")
        res, out = _bench_bucket(vit_cg, s, name, args.warmup, args.iters)
        cg_results.append(res)
        cg_outputs.append(out.detach().clone())

    # ----- Numerical closeness -----
    diffs = []
    for base_out, cg_out in zip(base_outputs, cg_outputs):
        d = (base_out.float() - cg_out.float()).abs().max().item()
        diffs.append(d)

    # ----- Emit Markdown -----
    lines = []
    lines.append(
        f"# ViT CUDA Graph bench — Qwen3.5 / Qwen3-VL ViT ({args.dtype})\n"
    )
    lines.append(
        "| size | S | baseline P50 (ms) | CG P50 (ms) | speedup | "
        "baseline P99 (ms) | CG P99 (ms) | first-capture (ms) | "
        "max-abs-diff |"
    )
    lines.append(
        "|------|---|-------------------|-------------|---------|"
        "-------------------|-------------|--------------------|--------------|"
    )
    for b, c, d in zip(base_results, cg_results, diffs):
        speedup = b.median_ms / c.median_ms if c.median_ms > 0 else float("nan")
        lines.append(
            f"| {b.name} | {b.s_tokens} | {b.median_ms:.3f} | {c.median_ms:.3f} | "
            f"{speedup:.2f}× | {b.p99_ms:.3f} | {c.p99_ms:.3f} | "
            f"{c.capture_ms:.1f} | {d:.2e} |"
        )

    out_path = args.output
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[bench] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
