# ViT CUDA Graph bench — Qwen3.5 / Qwen3-VL ViT — Results

> **Status: NOT YET RUN ON GPU.**
>
> The development VM that authored this PR is a CPU-only sandbox
> (`Intel(R) Xeon(R) Processor`, no `nvidia-smi`, no `lspci`,
> `torch.cuda.is_available() = False`). The benchmark script
> `bench_vit_cuda_graph_qwen35.py` correctly skips on this VM with the
> message
>
> ```
> [SKIP] No CUDA device found; this benchmark requires a CUDA GPU and
> Triton. See benchmark/encode_server/README.md for details.
> ```
>
> The script, acceptance thresholds, and run-book are in place. The
> reviewer should run them on a CUDA box and paste the resulting
> Markdown table below.

## How to populate this file

```
git fetch origin cursor/qwen35-vit-cg-rope-perf-f62e
git checkout cursor/qwen35-vit-cg-rope-perf-f62e

# On a CUDA host with sglang installed and a Qwen3.5 / Qwen3-VL
# checkpoint accessible (the bench uses random weights; no checkpoint
# download required):
python3 benchmark/encode_server/bench_vit_cuda_graph_qwen35.py \
    --buckets 256,1024,4096 \
    --warmup 30 --iters 200 \
    --output benchmark/encode_server/bench_results.md
```

The script overwrites this file with the actual measured table (one
row per bucket: P50, P95, P99, first-capture cost, baseline-vs-CG
max-abs-diff).

## Acceptance thresholds (from PLAN §5.2)

| size  | S    | min P50 speedup           |
|-------|------|---------------------------|
| small |  256 | ≥ 1.20× (CG ≤ 0.80×base)  |
| medium| 1024 | ≥ 1.10× (CG ≤ 0.90×base)  |
| large | 4096 | ≥ 1.00× (no regression)   |

Plus:
- max-abs-diff ≤ 5e-3 (fp16) per bucket
- first-capture cost per new `S` ≤ 500 ms

If any threshold fails, capture a profiler trace via `/start_profile`
and confirm:
- The captured ViT region appears as a single `cudaGraphLaunch`.
- The Triton vision-rope kernel appears as one launch per layer.
- No `random_*` / `torch.compile` host activity inside the captured
  region.

## What WAS validated locally on the CPU sandbox

1. **Eager-vs-native parity (fp32 / fp16 / bf16, 9 shape combos).**
   `apply_rotary_pos_emb_eager` is bit-equivalent to
   `apply_rotary_pos_emb_native` (max diff `0.000e+00` for all 9 cases).
   This is the safety fallback used when Triton is unavailable.
2. **Lint** clean across all changed files.
3. **Wrapper rejects 4D q/k** (negative test passes on CPU).
4. **No-CUDA branch** of the benchmark script exits cleanly.

## What CANNOT be validated on this CPU sandbox

These require a CUDA GPU and Triton 3.x and are guarded by
`@pytest.mark.skipif(not torch.cuda.is_available())`:

- `test_triton_vs_native_parity_cuda` (parity of the Triton kernel
  against the native rope across dtypes / shapes — 9 cases)
- `test_triton_inplace_semantics_cuda` (in-place `data_ptr` preserved)
- `test_triton_cuda_graph_capture_replay` (the headline scenario:
  `torch.cuda.graph(...)` capture + replay match eager output)
- The end-to-end perf benchmark above
