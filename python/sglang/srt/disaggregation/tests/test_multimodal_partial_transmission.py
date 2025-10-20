"""
Unit tests for multi-round embedding transmission feature.

Tests the incomplete block transmission handling when Language side
cannot allocate sufficient cache in a single round.
"""

import unittest
from typing import List, Tuple
from unittest.mock import MagicMock, Mock, patch

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))


class TestEmbeddingTransmissionState(unittest.TestCase):
    """Test the EmbeddingTransmissionState class."""
    
    def test_initialization(self):
        """Test state initialization."""
        from sglang.srt.disaggregation.mooncake.conn import EmbeddingTransmissionState
        
        state = EmbeddingTransmissionState(
            room=1,
            embedding_index=0,
            total_size_per_buffer=[1000, 2000, 3000],
            transmitted_size_per_buffer=[0, 0, 0],
        )
        
        self.assertEqual(state.room, 1)
        self.assertEqual(state.embedding_index, 0)
        self.assertEqual(state.transmission_count, 0)
        self.assertFalse(state.is_complete())
    
    def test_is_complete(self):
        """Test completion detection."""
        from sglang.srt.disaggregation.mooncake.conn import EmbeddingTransmissionState
        
        # Not complete
        state = EmbeddingTransmissionState(
            room=1,
            embedding_index=0,
            total_size_per_buffer=[1000, 2000],
            transmitted_size_per_buffer=[500, 1000],
        )
        self.assertFalse(state.is_complete())
        
        # Complete
        state.transmitted_size_per_buffer = [1000, 2000]
        self.assertTrue(state.is_complete())
        
        # Over-transmitted (edge case)
        state.transmitted_size_per_buffer = [1100, 2100]
        self.assertTrue(state.is_complete())
    
    def test_get_remaining_chunks(self):
        """Test remaining chunk calculation."""
        from sglang.srt.disaggregation.mooncake.conn import EmbeddingTransmissionState
        
        state = EmbeddingTransmissionState(
            room=1,
            embedding_index=0,
            total_size_per_buffer=[1000, 2000, 3000],
            transmitted_size_per_buffer=[400, 800, 1200],
        )
        
        remaining = state.get_remaining_chunks()
        expected = [
            (400, 600),   # buffer 0: transmitted 400, remaining 600
            (800, 1200),  # buffer 1: transmitted 800, remaining 1200
            (1200, 1800), # buffer 2: transmitted 1200, remaining 1800
        ]
        self.assertEqual(remaining, expected)


class TestMultiRoundTransmission(unittest.TestCase):
    """Test multi-round transmission protocol."""
    
    def test_chunk_calculation(self):
        """Test chunk calculation for multi-round transmission."""
        total_size = 1000 * 1024 * 1024  # 1000 MB
        max_chunk = 400 * 1024 * 1024   # 400 MB per round
        
        chunks: List[Tuple[int, int]] = []
        offset = 0
        
        while offset < total_size:
            remaining = total_size - offset
            chunk_size = min(max_chunk, remaining)
            chunks.append((offset, chunk_size))
            offset += chunk_size
        
        # Should need 3 rounds: 400MB + 400MB + 200MB
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], (0, 400 * 1024 * 1024))
        self.assertEqual(chunks[1], (400 * 1024 * 1024, 400 * 1024 * 1024))
        self.assertEqual(chunks[2], (800 * 1024 * 1024, 200 * 1024 * 1024))
        
        # Verify total
        total_transmitted = sum(size for _, size in chunks)
        self.assertEqual(total_transmitted, total_size)
    
    def test_transmission_id_increment(self):
        """Test that transmission_id increments correctly."""
        from sglang.srt.disaggregation.mooncake.conn import TransferEmbeddingChunk
        
        chunk1 = TransferEmbeddingChunk(
            room=1,
            embedding_index=0,
            is_last=False,
            chunk_info=[(0, 400)],
            transmission_id=0,
        )
        
        chunk2 = TransferEmbeddingChunk(
            room=1,
            embedding_index=0,
            is_last=False,
            chunk_info=[(400, 400)],
            transmission_id=1,
        )
        
        chunk3 = TransferEmbeddingChunk(
            room=1,
            embedding_index=0,
            is_last=True,
            chunk_info=[(800, 200)],
            transmission_id=2,
        )
        
        self.assertEqual(chunk1.transmission_id, 0)
        self.assertEqual(chunk2.transmission_id, 1)
        self.assertEqual(chunk3.transmission_id, 2)


class TestRequestMoreCacheInfo(unittest.TestCase):
    """Test RequestMoreCacheInfo message parsing."""
    
    def test_from_zmq(self):
        """Test parsing from ZMQ message."""
        from sglang.srt.disaggregation.mooncake.conn import RequestMoreCacheInfo
        import json
        
        chunk_info = [(400 * 1024 * 1024, 400 * 1024 * 1024)]
        chunk_info_json = json.dumps(chunk_info)
        
        msg = [
            b"1",                          # room
            b"192.168.1.100",              # endpoint
            b"8080",                       # dst_port
            b"session-123",                # mooncake_session_id
            b"0",                          # dst_embedding_index
            chunk_info_json.encode("ascii"),  # new_chunk_info
            b"1",                          # transmission_id
        ]
        
        info = RequestMoreCacheInfo.from_zmq(msg)
        
        self.assertEqual(info.room, 1)
        self.assertEqual(info.endpoint, "192.168.1.100")
        self.assertEqual(info.dst_port, 8080)
        self.assertEqual(info.mooncake_session_id, "session-123")
        self.assertEqual(info.dst_embedding_index, 0)
        self.assertEqual(info.new_chunk_info, chunk_info)
        self.assertEqual(info.transmission_id, 1)


class TestTransferEmbeddingInfo(unittest.TestCase):
    """Test TransferEmbeddingInfo message parsing."""
    
    def test_from_zmq_without_transmission_id(self):
        """Test parsing from ZMQ message without transmission_id."""
        from sglang.srt.disaggregation.mooncake.conn import TransferEmbeddingInfo
        
        msg = [
            b"1",            # room
            b"192.168.1.100",  # endpoint
            b"8080",         # dst_port
            b"session-123",  # mooncake_session_id
            b"0",            # dst_embedding_index
            b"1",            # required_dst_info_num
        ]
        
        info = TransferEmbeddingInfo.from_zmq(msg)
        
        self.assertEqual(info.room, 1)
        self.assertEqual(info.transmission_id, 0)  # Default
    
    def test_from_zmq_with_transmission_id(self):
        """Test parsing from ZMQ message with transmission_id."""
        from sglang.srt.disaggregation.mooncake.conn import TransferEmbeddingInfo
        
        msg = [
            b"1",            # room
            b"192.168.1.100",  # endpoint
            b"8080",         # dst_port
            b"session-123",  # mooncake_session_id
            b"0",            # dst_embedding_index
            b"1",            # required_dst_info_num
            b"2",            # transmission_id
        ]
        
        info = TransferEmbeddingInfo.from_zmq(msg)
        
        self.assertEqual(info.room, 1)
        self.assertEqual(info.transmission_id, 2)


class TestSendEmbeddingLogic(unittest.TestCase):
    """Test the send_embedding logic."""
    
    @patch('sglang.srt.disaggregation.mooncake.conn.MooncakeTransferEngine')
    def test_send_embedding_skips_zero_size(self, mock_engine_class):
        """Test that send_embedding skips buffers with size 0."""
        from sglang.srt.disaggregation.mooncake.conn import MooncakeEmbeddingManager
        from sglang.srt.disaggregation.base.conn import KVArgs
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.server_args import ServerArgs
        
        # Create mock arguments
        mock_args = Mock(spec=KVArgs)
        mock_args.aux_data_ptrs = [1000, 2000, 3000]
        mock_args.aux_data_lens = [100, 200, 300]
        mock_args.aux_item_lens = [10, 20, 30]
        mock_args.gpu_id = 0
        mock_args.ib_device = "mlx5_0"
        mock_args.engine_rank = 0
        
        mock_server_args = Mock(spec=ServerArgs)
        mock_server_args.disaggregation_bootstrap_port = 8080
        mock_server_args.dist_init_addr = "localhost"
        mock_server_args.tp_size = 1
        mock_server_args.dp_size = 1
        
        # Create manager with mocked engine
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        mock_engine.transfer_sync.return_value = 0
        
        manager = MooncakeEmbeddingManager(
            mock_args,
            DisaggregationMode.ENCODE,
            mock_server_args,
        )
        
        # Test sending with some zero-size chunks
        chunk_info = [
            (0, 10),   # Send buffer 0
            (0, 0),    # Skip buffer 1 (size 0)
            (0, 30),   # Send buffer 2
        ]
        
        result = manager.send_embedding(
            mooncake_session_id="test-session",
            embedding_index=0,
            dst_embedding_ptrs=[5000, 6000, 7000],
            dst_embedding_index=0,
            chunk_info=chunk_info,
        )
        
        # Should only call transfer_sync twice (skipping buffer 1)
        self.assertEqual(mock_engine.transfer_sync.call_count, 2)
        self.assertEqual(result, 0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingTransmissionState))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiRoundTransmission))
    suite.addTests(loader.loadTestsFromTestCase(TestRequestMoreCacheInfo))
    suite.addTests(loader.loadTestsFromTestCase(TestTransferEmbeddingInfo))
    suite.addTests(loader.loadTestsFromTestCase(TestSendEmbeddingLogic))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
