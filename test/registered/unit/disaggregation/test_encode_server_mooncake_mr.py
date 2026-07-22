"""Unit tests for the Mooncake embedding-transfer MR lifecycle in
srt/disaggregation/encode_server.py.

Covers the encoder-side /send path with concurrent sibling-TP callers:
the source-embedding Memory Region must be registered exactly once per
request, must stay registered while any transfer is in flight (an
early-finishing sibling must not invalidate it), and register /
transfer_sync failures must propagate instead of being ACKed as success.
"""

import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.encode_receiver import EmbeddingData
from sglang.srt.disaggregation.encode_server import InternalError, MMEncoder
from sglang.srt.managers.schedule_batch import Modality
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeMooncakeEngine:
    """Mimics MooncakeTransferEngine MR semantics: registering an
    already-registered address fails (overlapping MR), and transfers
    require an active MR covering the source pointer."""

    def __init__(self, transfer_delay: float = 0.0, transfer_ret: int = 0):
        self.active_mrs = {}  # ptr -> length
        self.register_calls = []
        self.deregister_calls = []
        self.transfer_calls = []
        self.transfer_without_mr = 0
        self.transfer_delay = transfer_delay
        self.transfer_ret = transfer_ret

    def register(self, ptr, length) -> int:
        self.register_calls.append((ptr, length))
        if ptr in self.active_mrs:
            return -1  # ERR_ADDRESS_OVERLAPPED
        self.active_mrs[ptr] = length
        return 0

    def deregister(self, ptr) -> int:
        self.deregister_calls.append(ptr)
        if ptr not in self.active_mrs:
            return -1
        del self.active_mrs[ptr]
        return 0

    def transfer_sync(self, session_id, src_ptr, dst_ptr, length) -> int:
        import time

        if self.transfer_delay:
            time.sleep(self.transfer_delay)
        # The source MR must still be active when the transfer runs.
        if src_ptr not in self.active_mrs:
            self.transfer_without_mr += 1
            return -1
        self.transfer_calls.append((session_id, src_ptr, dst_ptr, length))
        return self.transfer_ret


def _make_encoder(engine: FakeMooncakeEngine) -> MMEncoder:
    enc = object.__new__(MMEncoder)
    enc.server_args = SimpleNamespace(encoder_transfer_backend="mooncake")
    enc.engine = engine
    enc._forward_ready_events = {}
    enc._forward_results = {}
    enc._element_size = 2  # float16
    enc.executor = ThreadPoolExecutor(max_workers=4)
    # ZMQ ack path: fake context so no real socket is opened.
    enc.sync_context = MagicMock()
    return enc


def _make_mm_data(req_id: str, embedding: torch.Tensor) -> EmbeddingData:
    return EmbeddingData(
        req_id,
        1,
        0,
        [[1, 4, 4]],
        Modality.IMAGE,
        embedding,
    )


def _embedding(rows: int = 8, dim: int = 16) -> torch.Tensor:
    return torch.randn(rows, dim, dtype=torch.float16)


class TestMooncakeSendMRLifecycle(CustomTestCase):
    def test_concurrent_sibling_sends_share_single_mr(self):
        """Two sibling-TP /send calls for the same request must share one MR;
        the first to finish must not deregister it while the second transfer
        is still in flight."""
        engine = FakeMooncakeEngine(transfer_delay=0.05)
        enc = _make_encoder(engine)
        emb = _embedding()
        mm_data = _make_mm_data("req-concurrent", emb)

        async def run():
            await asyncio.gather(
                enc._send(
                    mm_data.embedding,
                    mm_data,
                    session_id="s1",
                    buffer_address=0x1000,
                    prefill_host="127.0.0.1",
                    embedding_port=12345,
                ),
                enc._send(
                    mm_data.embedding,
                    mm_data,
                    session_id="s2",
                    buffer_address=0x2000,
                    prefill_host="127.0.0.1",
                    embedding_port=12346,
                ),
            )

        asyncio.run(run())

        self.assertEqual(
            len(engine.register_calls), 1, "MR must be registered exactly once"
        )
        self.assertEqual(
            engine.deregister_calls,
            [],
            "MR must not be deregistered inside /send (deferred to request cleanup)",
        )
        self.assertEqual(
            engine.transfer_without_mr,
            0,
            "no transfer may run after its source MR was deregistered",
        )
        self.assertEqual(len(engine.transfer_calls), 2)
        # The MR is still active, awaiting request-level cleanup.
        self.assertIn(emb.data_ptr(), engine.active_mrs)

    def test_sequential_sibling_send_reuses_cached_embedding_and_mr(self):
        """After the first /send completes, mm_data.embedding is cleared;
        a later sibling /send must reuse cached_embedding and the shared MR."""
        engine = FakeMooncakeEngine()
        enc = _make_encoder(engine)
        emb = _embedding()
        mm_data = _make_mm_data("req-sequential", emb)

        async def run():
            await enc._send(
                mm_data.embedding,
                mm_data,
                session_id="s1",
                buffer_address=0x1000,
                prefill_host="127.0.0.1",
                embedding_port=12345,
            )
            self.assertIsNone(mm_data.embedding)
            # Mimics MMEncoder.send() passing mm_data.embedding (now None).
            await enc._send(
                mm_data.embedding,
                mm_data,
                session_id="s2",
                buffer_address=0x2000,
                prefill_host="127.0.0.1",
                embedding_port=12346,
            )

        asyncio.run(run())

        self.assertEqual(len(engine.register_calls), 1)
        self.assertEqual(len(engine.transfer_calls), 2)
        self.assertEqual(engine.deregister_calls, [])

    def test_register_failure_raises(self):
        engine = FakeMooncakeEngine()
        enc = _make_encoder(engine)
        emb = _embedding()
        mm_data = _make_mm_data("req-reg-fail", emb)
        # Force overlap rejection on the first register.
        engine.active_mrs[emb.data_ptr()] = emb.nbytes

        async def run():
            await enc._send(
                mm_data.embedding,
                mm_data,
                session_id="s1",
                buffer_address=0x1000,
                prefill_host="127.0.0.1",
                embedding_port=12345,
            )

        with self.assertRaises(InternalError):
            asyncio.run(run())
        self.assertEqual(
            engine.transfer_calls, [], "no transfer may run after register failed"
        )

    def test_transfer_failure_raises(self):
        engine = FakeMooncakeEngine(transfer_ret=-1)
        enc = _make_encoder(engine)
        emb = _embedding()
        mm_data = _make_mm_data("req-xfer-fail", emb)

        sent_via_zmq = []
        enc.sync_context.socket.side_effect = (
            lambda *a, **k: sent_via_zmq.append(1) or MagicMock()
        )

        async def run():
            await enc._send(
                mm_data.embedding,
                mm_data,
                session_id="s1",
                buffer_address=0x1000,
                prefill_host="127.0.0.1",
                embedding_port=12345,
            )

        with self.assertRaises(InternalError):
            asyncio.run(run())
        self.assertEqual(
            sent_via_zmq, [], "no ZMQ success ACK may be sent after a failed transfer"
        )

    def test_cleanup_deregisters_shared_mr(self):
        """Request-level cleanup must release the MR registered by /send."""
        engine = FakeMooncakeEngine()
        enc = _make_encoder(engine)
        emb = _embedding()
        mm_data = _make_mm_data("req-cleanup", emb)
        enc._inflight_encode_lock = asyncio.Lock()
        enc._inflight_encode_events = {}
        enc._inflight_encode_meta = {}
        enc._inflight_encode_cleanup_tasks = {}
        enc.embedding_to_send = {"req-cleanup": mm_data}

        async def run():
            await enc._send(
                mm_data.embedding,
                mm_data,
                session_id="s1",
                buffer_address=0x1000,
                prefill_host="127.0.0.1",
                embedding_port=12345,
            )
            self.assertIn(emb.data_ptr(), engine.active_mrs)
            await enc._cleanup_inflight_encode_state("req-cleanup")

        asyncio.run(run())

        self.assertEqual(engine.deregister_calls, [emb.data_ptr()])
        self.assertNotIn(emb.data_ptr(), engine.active_mrs)
        self.assertIsNone(mm_data.cached_embedding)
        self.assertEqual(enc.embedding_to_send, {})


if __name__ == "__main__":
    unittest.main()
