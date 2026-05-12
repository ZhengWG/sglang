# Encoder server benchmarks

Performance probes for `python/sglang/srt/disaggregation/encode_server.py`
and the underlying ViT path used by Qwen3.5 / Qwen3.5-MoE.

## bench_vit_cuda_graph_qwen35.py

Measures the per-image ViT latency of the encoder visual model with and
without ViT CUDA Graph capture (`SGLANG_VIT_ENABLE_CUDA_GRAPH`). It is
designed to validate the perf claim of the
"CUDA-graph-safe Triton vision RoPE" change for Qwen3.5: the rope
operator now uses a Triton kernel inside `VisionAttention` so that
`torch.cuda.graph(...)` can capture the full ViT block sequence (which
the previous `@torch.compile`'d native rope blocked with a
`random_rng` failure).

### Requirements

- A CUDA GPU and a Triton install matching torch.
- A Qwen3.5 / Qwen3.5-MoE checkpoint **or** Qwen3-VL as a proxy
  (both use the same `Qwen3VLMoeVisionModel`).
- `--mm-attention-backend triton_attn` or `fa3` (anything else will
  raise inside `ViTCudaGraphRunner._create_graph`).

### Run

```
python3 benchmark/encode_server/bench_vit_cuda_graph_qwen35.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --buckets 256,1024,4096 \
  --warmup 30 --iters 200
```

The script:
1. Builds a `MMEncoder` once with `SGLANG_VIT_ENABLE_CUDA_GRAPH=0` and
   times `encoder.encode(...)` for each image-token bucket (small /
   medium / large).
2. Tears it down and rebuilds with `SGLANG_VIT_ENABLE_CUDA_GRAPH=1`,
   re-times, and also reports the first-capture cost per `S`.
3. Compares fp16 max-abs-diff between the two output tensors per
   bucket.
4. Emits a Markdown table to `bench_results.md`.

### Acceptance thresholds (gating the PR — see PLAN.md §5.2)

- Small (S≈256):  CG P50 ≤ 0.80 × baseline P50 (≥ 20% faster)
- Medium (S≈1k):  CG P50 ≤ 0.90 × baseline P50 (≥ 10% faster)
- Large (S≈4k):   CG P50 ≤ baseline P50 (no regression)
- Output max-abs-diff ≤ 5e-3 (fp16) every bucket
- First-capture cost per new `S` ≤ 500 ms

If the perf delta is smaller than expected, capture a torch profiler
trace via the existing `/start_profile` endpoint and verify the
captured region replays as a single `cudaGraphLaunch`.
