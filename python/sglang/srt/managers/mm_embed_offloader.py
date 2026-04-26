"""Async D2H offloader for multimodal embeddings used by EPD prefill.

This module provides a true-async CPU offload path for multimodal features
and precomputed embeddings produced on the language-only worker in EPD
disaggregation. The goal is to remove the synchronization point that a
plain ``tensor.to("cpu", non_blocking=True)`` introduces (which silently
falls back to a synchronous copy when the host buffer is not pinned),
while still releasing GPU memory immediately after each forward chunk.

Design

- A small pool of pinned host buffers, bucketed by ``(dtype, nbytes)``,
  reused across calls so that we do not pay ``cudaHostAlloc`` on every
  offload.
- A dedicated CUDA stream issues the D2H copy. ``record_stream`` is called
  on the source GPU tensor so the caching allocator is free to reuse its
  memory as soon as the copy on that stream finishes.
- A :class:`LazyHostTensor` wraps the pinned buffer together with the
  recording event, exposing a torch.Tensor-like surface; consumers wait on
  the event lazily, on the first access.

This path is intentionally scoped to EPD (``--language-only``); regular
multimodal inference keeps its existing offload behavior.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

import torch


_PoolKey = Tuple[torch.dtype, int]


class _PinnedHostPool:
    """Thread-safe pool of pinned host tensors keyed by (dtype, nbytes)."""

    def __init__(self, max_buffers_per_bucket: int = 8) -> None:
        self._free: Dict[_PoolKey, List[torch.Tensor]] = {}
        self._lock = threading.Lock()
        self._max_buffers_per_bucket = max_buffers_per_bucket

    @staticmethod
    def _nbytes(shape: Tuple[int, ...], dtype: torch.dtype) -> int:
        elem = torch.tensor([], dtype=dtype).element_size()
        n = 1
        for s in shape:
            n *= int(s)
        return n * elem

    def acquire(self, shape, dtype: torch.dtype) -> torch.Tensor:
        shape_t = tuple(int(s) for s in shape)
        nbytes = self._nbytes(shape_t, dtype)
        key: _PoolKey = (dtype, nbytes)
        with self._lock:
            bucket = self._free.get(key)
            if bucket:
                buf = bucket.pop()
                return buf.view(shape_t)
        return torch.empty(shape_t, dtype=dtype, pin_memory=True)

    def release(self, tensor: torch.Tensor) -> None:
        if not tensor.is_pinned():
            return
        nbytes = tensor.numel() * tensor.element_size()
        key: _PoolKey = (tensor.dtype, nbytes)
        flat = tensor.flatten()
        with self._lock:
            bucket = self._free.setdefault(key, [])
            if len(bucket) < self._max_buffers_per_bucket:
                bucket.append(flat)


_HOST_POOL = _PinnedHostPool()
_OFFLOAD_STREAM: Optional["torch.cuda.Stream"] = None
_STREAM_LOCK = threading.Lock()


def _get_offload_stream() -> "torch.cuda.Stream":
    global _OFFLOAD_STREAM
    if _OFFLOAD_STREAM is None:
        with _STREAM_LOCK:
            if _OFFLOAD_STREAM is None:
                _OFFLOAD_STREAM = torch.cuda.Stream()
    return _OFFLOAD_STREAM


class LazyHostTensor:
    """A pinned-host tensor whose D2H copy may still be in flight.

    Behaves like a CPU ``torch.Tensor`` for the operations that the
    multimodal embedding pipeline actually performs: ``.to(device)``,
    ``.cpu()``, ``.cuda()``, indexing, ``.numel()``, ``.shape``,
    ``.dtype``, ``.is_cuda`` (always False), ``.device`` (always
    ``cpu``), and an explicit :meth:`materialize` to obtain the
    underlying ``torch.Tensor`` after synchronization.

    The first access automatically waits for the recording event.
    """

    __slots__ = ("_host", "_event", "_synced")

    def __init__(self, host: torch.Tensor, event: "torch.cuda.Event") -> None:
        self._host = host
        self._event = event
        self._synced = False

    def _wait(self) -> None:
        if not self._synced:
            self._event.synchronize()
            self._synced = True

    def materialize(self) -> torch.Tensor:
        self._wait()
        return self._host

    def cpu(self) -> torch.Tensor:
        return self.materialize()

    def cuda(self, non_blocking: bool = True) -> torch.Tensor:
        self._wait()
        return self._host.to("cuda", non_blocking=non_blocking)

    def to(self, *args, **kwargs) -> torch.Tensor:
        self._wait()
        return self._host.to(*args, **kwargs)

    def reshape(self, *shape) -> "LazyHostTensor":
        self._wait()
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return LazyHostTensor(self._host.reshape(shape), self._event)

    def __getitem__(self, item) -> torch.Tensor:
        self._wait()
        return self._host[item]

    def numel(self) -> int:
        return self._host.numel()

    @property
    def shape(self):
        return self._host.shape

    @property
    def dtype(self):
        return self._host.dtype

    @property
    def device(self):
        return self._host.device

    @property
    def is_cuda(self) -> bool:
        return False

    def __repr__(self) -> str:
        return (
            f"LazyHostTensor(shape={tuple(self._host.shape)}, "
            f"dtype={self._host.dtype}, synced={self._synced})"
        )


def async_offload_to_host(feature: torch.Tensor) -> LazyHostTensor:
    """Issue an async D2H copy on a side stream; return a lazy host view.

    The source tensor's GPU memory becomes reclaimable by the caching
    allocator as soon as the copy on the offload stream completes
    (enforced via :func:`torch.Tensor.record_stream`). The returned
    :class:`LazyHostTensor` lazily synchronizes on first access.
    """
    if not isinstance(feature, torch.Tensor) or not feature.is_cuda:
        raise TypeError(
            "async_offload_to_host expects a CUDA torch.Tensor, "
            f"got {type(feature).__name__}"
        )

    host = _HOST_POOL.acquire(feature.shape, feature.dtype)
    stream = _get_offload_stream()
    cur = torch.cuda.current_stream(device=feature.device)
    stream.wait_stream(cur)
    with torch.cuda.stream(stream):
        host.copy_(feature, non_blocking=True)
        feature.record_stream(stream)
    event = torch.cuda.Event()
    event.record(stream)
    return LazyHostTensor(host, event)


def maybe_async_offload(feature):
    """Best-effort async offload helper.

    Returns ``feature`` unchanged when it is not a CUDA tensor; otherwise
    schedules an async D2H copy and returns a :class:`LazyHostTensor`.
    Falls back to a plain (potentially synchronous) ``.to('cpu')`` if CUDA
    is unavailable for any reason.
    """
    if isinstance(feature, LazyHostTensor):
        return feature
    if not isinstance(feature, torch.Tensor):
        return feature
    if not feature.is_cuda:
        return feature
    try:
        return async_offload_to_host(feature)
    except Exception:
        return feature.to("cpu", non_blocking=True)


def to_device_for_concat(
    tensors: List, target_device: Optional[torch.device] = None
) -> Tuple[List[torch.Tensor], Optional[torch.device]]:
    """Materialize lazy host tensors and unify device for ``torch.concat``.

    Filters out empty tensors, materializes any :class:`LazyHostTensor`,
    then casts everything to ``target_device`` (defaulting to the device
    of the first non-empty tensor). Returns the cleaned list and the
    chosen device. The list is empty when there is nothing to concat.
    """
    cleaned: List[torch.Tensor] = []
    for t in tensors:
        if t is None:
            continue
        if isinstance(t, LazyHostTensor):
            t = t.materialize()
        if not isinstance(t, torch.Tensor):
            continue
        if t.numel() == 0:
            continue
        cleaned.append(t)

    if not cleaned:
        return cleaned, None

    if target_device is None:
        target_device = cleaned[0].device

    if any(t.device != target_device for t in cleaned):
        cleaned = [t.to(target_device, non_blocking=True) for t in cleaned]

    return cleaned, target_device
