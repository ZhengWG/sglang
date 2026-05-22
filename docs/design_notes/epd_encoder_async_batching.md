# EPD Encoder Cross-Request Batching — Design Notes & Async Best Practices

> Reference: [sgl-project/sglang#25964](https://github.com/sgl-project/sglang/pull/25964)
> Scope: `python/sglang/srt/disaggregation/encode_server.py`
> Audience: contributors writing asyncio + multi-process + TP-collective code

This note distills the design choices behind the cross-request encoder
batching introduced in PR #25964 into a reusable pattern, with an emphasis
on **how async/sync/parallel execution domains compose** under a multi-rank
TP topology.

---

## 1. Problem

The EPD encoder server (`encode_server.py`) receives one HTTP `/encode`
request per multimodal item. Without batching:

- N concurrent requests ⇒ N sequential preprocessor + ViT forward passes.
- GPU is under-utilised; per-request latency is bounded by serial scheduling.
- TP ranks > 0 still need to be kept in lockstep, so naive client-side
  batching is not enough — the scheduler must broadcast a *coherent* batch
  to every rank.

Goal: aggregate concurrent requests into bounded batches, fan them out to
every TP rank, run a single batched preprocess + ViT forward, and return
per-request results — all while preserving the original HTTP error and
timeout semantics.

---

## 2. Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│  Rank 0 process  (FastAPI / uvicorn, single asyncio event loop)         │
│                                                                          │
│    N × HTTP handler coroutines                                          │
│         │  submit(req) → pending_queue.put                              │
│         ▼                                                                │
│    ┌──────────────────────────────────────────┐                        │
│    │  pending_queue (asyncio.Queue)            │                        │
│    └──────────────────────────────────────────┘                        │
│         ▲ put              │ get                                         │
│         │                  ▼                                             │
│         │       ┌──────────────────────────────┐                        │
│         │       │  _batch_worker (1 coroutine) │                        │
│         │       │  collect → group → dispatch  │                        │
│         │       └──────────────────────────────┘                        │
│         │                  │                                             │
│         │  set_result()    ├── ZMQ PUSH ── ▶ worker ranks                │
│         │                  └── await batch_encode (rank-0 in-process)   │
└─────────┴────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
            ┌──────────────────────────────────────┐
            │  Worker rank N processes              │
            │   run_encoder loop (recv → process)   │
            │   batch_encode (mirrors rank 0)       │
            └──────────────────────────────────────┘
                                  │
                                  ▼
              torch.distributed collective synchronisation
```

Three responsibilities are cleanly separated:

| Layer | Component | Responsibility |
|-------|-----------|----------------|
| Ingress | HTTP handler coroutine | Accept one request, submit, await future, return |
| Scheduling | `EncoderScheduler._batch_worker` | Collect, group by modality, fan out, resolve futures |
| Execution | `MMEncoder.batch_encode` (per rank) | Preprocess + collective + ViT + assemble |

---

## 3. Five Core Design Principles

### 3.1 Cross-coroutine completion via `Future`, not `Event`

```python
class PendingRequest:
    __slots__ = ("request", "future", "submit_time")
    def __init__(self, request, loop):
        self.future = loop.create_future()
```

`asyncio.Event` only signals completion; `asyncio.Future` carries either a
result (`set_result`) or an exception (`set_exception`). Always prefer
`loop.create_future()` over `asyncio.Future()` — the former binds to the
running loop and fails loud on cross-loop misuse.

### 3.2 Single scheduler coroutine for determinism

```python
async def _batch_worker(self):
    while True:
        batch = await self._collect_batch()
        for modality, group in groups.items():
            await self._dispatch_group(group, modality)
```

A coroutine *pool* would let multiple workers race for items from the
queue, producing non-deterministic batch composition. Worker ranks consume
ZMQ messages strictly in order, so if rank 0 dispatches `[A,B]` and `[C,D]`
in one ordering while a parallel scheduler dispatches `[A,C]` and `[B,D]`
in another, NCCL collectives will mis-pair across ranks and deadlock.

**Rule**: anything whose downstream requires strict global ordering should
be funneled through a single scheduler coroutine.

### 3.3 Continuous-batching flywheel: accumulation window is implicit

```python
async def _collect_batch(self):
    batch = [await self.pending_queue.get()]    # blocks when idle
    while len(batch) < self.max_batch_size:
        try:
            batch.append(self.pending_queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return batch
```

This looks like it never waits for additional requests, which would
defeat batching. The trick is that **the previous batch's processing
already yields the event loop** multiple times:

- `_flatten_and_load_*` → `asyncio.gather` over `io_executor` futures.
- `cache.batch_is_exist` (mooncake RPC).
- `asyncio.wait_for(_wait_prefetch)` polling at 5ms intervals.

During those yields, HTTP handler coroutines run and push new requests
into `pending_queue`. When `_batch_worker` loops back, `await get()`
returns immediately and `get_nowait()` drains the rest.

The result is **load-adaptive batching for free**:

- Higher load ⇒ longer per-batch processing ⇒ larger accumulation window
  ⇒ larger next batch.
- Lower load ⇒ shorter processing ⇒ smaller batches (which is fine, no
  benefit to batch a single request).

This is the same intuition as token-level continuous batching in
vLLM / SGLang, applied at request granularity. Adding an explicit
`batch_window_ms` is *not* necessary in the steady state.

### 3.4 Multi-rank lockstep: broadcast before await

```python
async def _dispatch_group(self, group, modality):
    requests = [p.request for p in group]
    for sock in self.send_sockets:                       # ① fan-out first
        sock.send_pyobj({"type": "batch_encode", "requests": requests, ...})
    results = await self.encoder.batch_encode(requests, modality)   # ② then await
```

Reversing the order would deadlock: rank 0 enters `batch_encode` → blocks
on `torch.distributed.broadcast` → worker ranks have not received the ZMQ
message yet → no one to broadcast with.

**Rule**: any "leader broadcasts and then participates in collective"
pattern must dispatch *before* entering the collective.

### 3.5 Collective-aware error boundaries

```python
try:
    mm_inputs, get_feat = await self._process_mm_items(flat_items, modality)
except Exception as e:
    return self._batch_set_error(requests, modality, ...)   # convert to error tuples
```

Inside `batch_encode`, exceptions are *never* propagated up — they are
converted to error tuples and written to `embedding_to_send`. The reason:
if rank 0 raises, it skips the subsequent `torch.distributed.broadcast`
calls. Worker ranks will sit forever waiting for that collective.

`_dispatch_group` adds a second safety net with an explicit comment:

```python
# If batch_encode raised, rank-0 may have skipped a collective,
# leaving TP workers stuck. Don't try to recover — fail every pending
# future and let the client retry. Re-broadcasting would risk a deadlock.
```

**Rule**: when code participates in a multi-rank collective, the golden
exception policy is *"either all ranks agree to fail, or all ranks
complete"*. Never silently skip a collective.

---

## 4. Execution Pipeline — Four Concurrency Domains

The pipeline mixes four kinds of "parallel":

| Domain | Mechanism | Yields to event loop | Notes |
|--------|-----------|---------------------|-------|
| 🟢 Coroutine | Multiple coroutines on one event loop | Only at `await` | HTTP handlers + scheduler + background tasks |
| 🟦 Thread | `ThreadPoolExecutor` via `run_in_executor` | Yes (caller awaits future) | `io_executor` (4T) for IO, `self.executor` (10T) for ZMQ send |
| 🟪 Process | TP worker ranks as separate OS processes | N/A (different process) | Synchronised at `torch.distributed` collectives |
| 🔴 Sync | Plain Python or sync C call | **No — blocks the loop** | HF processor, `torch.distributed.broadcast`, ViT forward |

Critical observation: **`await` does not guarantee yielding the event
loop**. `torch.distributed.broadcast(mask, src=0)` is a synchronous C
function — wrapping it in `async def` does not make it non-blocking.

### 4.1 Step-by-step classification

| Phase | Step | Domain | Latency hint |
|-------|------|--------|--------------|
| Ingress | uvicorn accept → handler spawn | 🟢 | µs |
| Ingress | `submit()` → `put()` → `wait_for(future)` | 🟢 (handler suspends) | µs |
| Scheduling | `_collect_batch`: `await get()` | 🟢 yields when empty | µs–ms |
| Scheduling | `_collect_batch`: `get_nowait` drain | 🔴 sync | µs |
| Dispatch | `sock.send_pyobj` × N_workers | 🔴 sync pickle + send | ~1 ms |
| Dispatch | `await batch_encode(...)` | 🟢 + 🟪 starts cross-rank | — |
| Preproc | `_flatten_and_load_*` → `asyncio.gather` | 🟢 yield + 🟦 4T parallel | 30–100 ms |
| Preproc | `self.image_processor(...)` | 🔴 sync CPU | 20–80 ms |
| Cache | `await cache.batch_is_exist` | 🟢 + 🟦 RPC | a few ms |
| Sync | `torch.distributed.broadcast(mask)` | 🔴🔴 **blocks loop** | <1 ms (deadly if hang) |
| Encode | `_encode_missing` → ViT forward | 🔴🔴 sync GPU + 🟪 NCCL collective | 50–500 ms |
| Cache | `cache.prefetch` + `wait_for(_wait, 60s)` | 🔴 + 🟢 5ms polling | variable |
| Sync | `torch.distributed.broadcast(status)` | 🔴🔴 blocks loop | <1 ms |
| Assemble | Build `EmbeddingData`, write `embedding_to_send` | 🔴 sync | µs |
| Cache | `asyncio.to_thread(cache.insert_batch, ...)` | 🟦 fire-and-forget | offloaded |
| Wakeup | `future.set_result(...)` × N | 🔴 sync (µs) | µs |
| Egress | `await encoder.send(...)` → `run_in_executor` | 🟢 + 🟦 10T parallel | 5–10 ms |

### 4.2 Worked example — 4 concurrent requests, TP=2

Realistic timings (ms) for 4 concurrent image requests with `max_batch_size=8`,
no global cache, `zmq_to_tokenizer` backend:

```
t (ms)  │ Rank 0 event loop                  │ Rank 0 io threads   │ Rank 0 GPU │ Rank 1 event loop          │ Rank 1 GPU
────────┼────────────────────────────────────┼──────────────────────┼────────────┼────────────────────────────┼──────────
 0.0    │ W: await get() [suspended]          │ idle                 │ idle       │ R: await recv [suspended]  │ idle
 0.1    │ H_A..H_D spawned, all ready          │                      │            │                            │
 0.2-0.4│ H_A..H_D each: submit → put → wait  │                      │            │                            │
 0.4    │ W: get → A; get_nowait → B,C,D       │                      │            │                            │
 0.5    │ W: send_pyobj({batch_encode,..})    │                      │            │                            │
 1.5    │ W: await batch_encode                │                      │            │ R: recv → batch_encode      │
 1.6    │   await _flatten_and_load_images    │ T_0..T_3: download  │ idle       │   (same logic)              │ idle
        │   [event loop yields here]          │ A,B,C,D urls         │            │                            │
 52     │   ◄── gather returns                 │ all idle             │            │ ◄── gather returns          │
        │   image_processor(...)              │                      │            │   image_processor(...)      │
        │   [🔴 BLOCKS 40ms]                  │                      │            │   [🔴 BLOCKS 40ms]          │
 92     │   _encode_missing → ViT(...)        │                      │ ────────►  │   _encode_missing → ViT     │ ────────►
        │   [🔴🔴 BLOCKS 100ms]               │                      │ ViT +      │   [🔴🔴 BLOCKS 100ms]        │ ViT +
        │                                     │                      │ NCCL       │                            │ NCCL
193     │   Build EmbeddingData × 4            │                      │            │   (rank>0 early return)     │
196     │ W: set_result × 4 → wake H_A..H_D   │                      │            │ R: await recv [suspended]  │
        │ H_A..H_D: await encoder.send         │ T_send_0..3: ZMQ    │ idle       │                            │
        │   [yield to thread pool]            │ push embeddings      │            │                            │
201     │ Handlers resume → 200 OK            │                      │            │                            │
```

End-to-end latency ≈ 200 ms for 4 requests. Running them serially
(no batching) would cost ~400 ms on the GPU alone. Batching saves ~50%
on this workload.

### 4.3 What is *actually parallel* vs *only interleaved*

| Action | Rank 0 loop | Rank 0 io_executor | Rank 0 GPU | Rank 1 loop | Rank 1 GPU |
|--------|-------------|--------------------|------------|--------------|------------|
| Download images | idle (yielded) | 🟦 4T parallel | — | idle (yielded) | — |
| `image_processor` | 🔴 busy | idle | — | 🔴 busy | — |
| ViT forward | 🔴 busy (blocking) | — | 🔴🔴 active | 🔴 busy | 🔴🔴 active |
| ZMQ embedding push | 🟢 yielded | — | — | (rank-0-only path) | — |

Two notable consequences:

1. **Download is duplicated across ranks** — each rank fetches the same
   URL into its own process. Large media URLs amplify network usage by
   `tp_size`. Worth flagging in deployments.
2. **The ViT block (100 ms here) freezes the rank-0 event loop completely.**
   New TCP connections can be accepted by the OS but uvicorn cannot
   process them until the GIL/loop returns.

---

## 5. Reusable Skeleton

A scaffold capturing every principle above. Drop in your own
`executor.run_batch` to apply this pattern to other request-level
batching scenarios.

```python
import asyncio
import contextlib
import time
from typing import List, Optional, Set


class PendingRequest:
    __slots__ = ("payload", "future", "submit_time")
    def __init__(self, payload, loop: asyncio.AbstractEventLoop):
        self.payload = payload
        self.future: asyncio.Future = loop.create_future()
        self.submit_time = time.monotonic()


class BatchScheduler:
    def __init__(self, executor, max_batch_size: int = 8,
                 request_timeout: float = 180.0):
        self.executor = executor
        self.max_batch_size = max(1, max_batch_size)
        self.request_timeout = max(1.0, request_timeout)
        self.queue: "asyncio.Queue[PendingRequest]" = asyncio.Queue()
        self._worker: Optional[asyncio.Task] = None
        self._bg_tasks: Set[asyncio.Task] = set()

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
        while True:
            try:
                p = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not p.future.done():
                p.future.set_exception(RuntimeError("scheduler stopped"))

    async def submit(self, payload):
        p = PendingRequest(payload, asyncio.get_running_loop())
        await self.queue.put(p)
        try:
            return await asyncio.wait_for(p.future,
                                          timeout=self.request_timeout)
        except asyncio.TimeoutError:
            if not p.future.done():
                p.future.cancel()
            raise

    async def _collect(self) -> List[PendingRequest]:
        batch = [await self.queue.get()]
        while len(batch) < self.max_batch_size:
            try:
                batch.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return batch

    async def _loop(self) -> None:
        while True:
            batch: List[PendingRequest] = []
            try:
                batch = await self._collect()
                payloads = [p.payload for p in batch]
                # The executor must catch its own exceptions and return
                # error results — see principle 3.5.
                results = await self.executor.run_batch(payloads)
                if len(results) != len(batch):
                    raise RuntimeError(
                        f"executor returned {len(results)} results "
                        f"for batch of {len(batch)}"
                    )
                for p, r in zip(batch, results):
                    if not p.future.done():
                        p.future.set_result(r)
            except asyncio.CancelledError:
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(
                            RuntimeError("scheduler stopped"))
                raise
            except Exception as e:
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(e)
```

---

## 6. Anti-Patterns

| ❌ Anti-pattern | Consequence | ✅ Replace with |
|----------------|-------------|----------------|
| `asyncio.Event` to pass results across coroutines | No place for exceptions; need side-channel error store | `asyncio.Future` with `set_result`/`set_exception` |
| Multiple worker coroutines pulling from a shared scheduling queue | Non-deterministic grouping → collective desync across ranks | Single scheduler coroutine |
| `asyncio.create_task(...)` with no reference held | GC can collect the task before it runs | `set.add(task)` + `task.add_done_callback(set.discard)` |
| Letting exceptions bubble out of a function that calls a collective | Other ranks block forever waiting for the collective | Catch and convert to error result, ensure all ranks finish |
| Trusting `asyncio.wait_for` to recover from NCCL hangs | Sync C calls block the loop; `wait_for` cannot cancel them | Process-level health check + restart |
| Initialising `asyncio.Queue / Lock / create_task` at module import | Event loop may not exist or may differ from the runtime loop | Move into `__init__` or FastAPI `lifespan` |
| Tuning `max_batch_size` without measuring `await`-link yields | Implicit accumulation window may already saturate batches | Trace yields first, then decide |
| Treating each TP rank as a peer that can self-schedule | Local timing differences ⇒ collective desync | Rank 0 decides, broadcasts to others |

---

## 7. Design Checklist

When porting this pattern to another component:

- [ ] HTTP/RPC handler uses `Future` (not `Event`) to receive results.
- [ ] Exactly one scheduler coroutine; `while True` body wrapped in
      `try/except CancelledError/Exception`.
- [ ] `_collect_batch` blocks once at the head, then non-blocking drain.
- [ ] All cross-rank fan-out happens *before* the rank participates in a
      collective.
- [ ] Executor catches exceptions internally and returns error results.
      Never let an exception skip a collective.
- [ ] All `future.set_result / set_exception / cancel` guarded by
      `if not future.done()`.
- [ ] Background tasks tracked with a `set` + `add_done_callback(discard)`.
- [ ] Event-loop-bound resources created in `__init__` or FastAPI
      `lifespan`, not at module import.
- [ ] `stop()` order: cancel worker → await worker → drain queue → fail
      futures.
- [ ] Sync C calls > 10 ms identified and considered for thread offload
      (`run_in_executor` / `asyncio.to_thread`).
- [ ] Watchdog timeouts documented with what they *can* and *cannot* catch
      (e.g. cannot interrupt sync TP collectives).
- [ ] Observability: histograms for queue depth, batch size,
      `submit_time → dispatch_time → complete_time`.

---

## 8. References

- PR #25964 — Cross-request batching for image/audio encoder (the
  implementation this note is based on).
- PR #25669 — Async image preprocessing and `_vit_queue` batching (an
  alternative approach at the `_encode` layer; complementary preprocessing
  improvements).
- vLLM / SGLang token-level continuous batching — same flywheel intuition
  applied at the token granularity.
- Python `asyncio` documentation: `loop.create_future`, `asyncio.shield`,
  `asyncio.to_thread`.
- `torch.distributed` collective semantics — all participating ranks must
  call the collective in the same order.
