"""
Test cases for multimodal embedding continuation feature.

This module tests the block-based allocation and continuation mechanism
for multimodal embedding transfers when actual data length exceeds default buffer size.
"""

import os
import torch
import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

# Set environment variables before imports
os.environ["SGLANG_USE_BLOCK_ALLOCATOR"] = "true"
os.environ["SGLANG_MULTIMODAL_BLOCK_SIZE"] = "128"
os.environ["SGLANG_DEFAULT_MULTIMODAL_BUFFER_TOKENS"] = "1024"

from sglang.srt.disaggregation.utils import (
    MetadataAllocation,
    ReqToMetadataBlockAllocator,
    MultimodalDataBuffers,
)


class TestBlockAllocator:
    """Test suite for ReqToMetadataBlockAllocator."""

    def test_allocator_initialization(self):
        """Test allocator is initialized correctly."""
        allocator = ReqToMetadataBlockAllocator(
            total_tokens=8192,
            block_size=128,
        )
        
        assert allocator.total_tokens == 8192
        assert allocator.block_size == 128
        assert allocator.num_blocks == 64  # 8192 / 128
        assert allocator.available_blocks() == 64

    def test_alloc_single_block(self):
        """Test allocating a single block."""
        allocator = ReqToMetadataBlockAllocator(
            total_tokens=8192,
            block_size=128,
        )
        
        allocation = allocator.alloc(num_tokens=100, req_id=1)
        
        assert allocation is not None
        assert len(allocation.block_indices) == 1
        assert allocation.num_tokens == 100
        assert allocator.available_blocks() == 63

    def test_alloc_multiple_blocks(self):
        """Test allocating multiple blocks."""
        allocator = ReqToMetadataBlockAllocator(
            total_tokens=8192,
            block_size=128,
        )
        
        # Allocate 300 tokens (needs 3 blocks: 300/128 = 2.34 -> 3)
        allocation = allocator.alloc(num_tokens=300, req_id=1)
        
        assert allocation is not None
        assert len(allocation.block_indices) == 3
        assert allocation.num_tokens == 300
        assert allocator.available_blocks() == 61

    def test_alloc_default(self):
        """Test default allocation (for Language side initial request)."""
        allocator = ReqToMetadataBlockAllocator(
            total_tokens=8192,
            block_size=128,
        )
        
        allocation = allocator.alloc_default(default_tokens=1024, req_id=1)
        
        assert allocation is not None
        assert len(allocation.block_indices) == 8  # 1024 / 128
        assert allocation.num_tokens == 1024

    def test_alloc_insufficient_blocks(self):
        """Test allocation fails when insufficient blocks available."""
        allocator = ReqToMetadataBlockAllocator(
            total_tokens=256,  # Only 2 blocks
            block_size=128,
        )
        
        # Try to allocate 3 blocks
        allocation = allocator.alloc(num_tokens=300, req_id=1)
        
        assert allocation is None
        assert allocator.available_blocks() == 2

    def test_free_allocation(self):
        """Test freeing allocated blocks."""
        allocator = ReqToMetadataBlockAllocator(
            total_tokens=8192,
            block_size=128,
        )
        
        allocation = allocator.alloc(num_tokens=300, req_id=1)
        assert allocator.available_blocks() == 61
        
        allocator.free(allocation, req_id=1)
        assert allocator.available_blocks() == 64

    def test_free_by_req_id(self):
        """Test freeing allocation by request ID."""
        allocator = ReqToMetadataBlockAllocator(
            total_tokens=8192,
            block_size=128,
        )
        
        allocation = allocator.alloc(num_tokens=300, req_id=1)
        assert allocator.available_blocks() == 61
        
        allocator.free_by_req_id(req_id=1)
        assert allocator.available_blocks() == 64

    def test_get_allocation(self):
        """Test retrieving allocation by request ID."""
        allocator = ReqToMetadataBlockAllocator(
            total_tokens=8192,
            block_size=128,
        )
        
        allocation = allocator.alloc(num_tokens=300, req_id=1)
        retrieved = allocator.get_allocation(req_id=1)
        
        assert retrieved is allocation
        assert retrieved.num_tokens == 300


class TestMultimodalDataBuffers:
    """Test suite for MultimodalDataBuffers with block-based allocation."""

    def test_buffer_initialization_block_mode(self):
        """Test buffer initialization in block-based mode."""
        buffers = MultimodalDataBuffers(
            size=8,
            max_prefill_tokens=8192,
            embedding_dim=4096,
            block_size=128,
            use_block_allocator=True,
        )
        
        assert buffers.use_block_allocator is True
        assert buffers.block_size == 128
        assert buffers.total_capacity_tokens == 8 * 8192
        assert buffers.num_blocks == (8 * 8192 + 127) // 128

    def test_buffer_initialization_index_mode(self):
        """Test buffer initialization in legacy index-based mode."""
        buffers = MultimodalDataBuffers(
            size=8,
            max_prefill_tokens=8192,
            embedding_dim=4096,
            use_block_allocator=False,
        )
        
        assert buffers.use_block_allocator is False
        assert buffers.max_prefill_tokens == 8192

    def test_get_buf_chunk_info_first_transfer(self):
        """Test chunk info calculation for first transfer (with max_tokens limit)."""
        buffers = MultimodalDataBuffers(
            size=8,
            max_prefill_tokens=8192,
            embedding_dim=4096,
            block_size=128,
            use_block_allocator=True,
        )
        
        # Create allocation for 2000 tokens
        allocation = MetadataAllocation(
            block_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            num_tokens=2000,
            start_offset=0,
        )
        
        # First transfer: limit to 1024 tokens
        chunk_info = buffers.get_buf_chunk_info(
            allocation=allocation,
            offset_tokens=0,
            max_tokens=1024,
        )
        
        # Check input_embeddings chunk
        offset_bytes, size_bytes = chunk_info[0]
        expected_size = 1024 * 4096 * 2  # tokens * embedding_dim * itemsize(bf16=2)
        assert size_bytes == expected_size

    def test_get_buf_chunk_info_continuation(self):
        """Test chunk info calculation for continuation transfer."""
        buffers = MultimodalDataBuffers(
            size=8,
            max_prefill_tokens=8192,
            embedding_dim=4096,
            block_size=128,
            use_block_allocator=True,
        )
        
        # Create allocation for remaining 976 tokens
        allocation = MetadataAllocation(
            block_indices=[16, 17, 18, 19, 20, 21, 22, 23],
            num_tokens=976,
            start_offset=0,
        )
        
        # Continuation transfer: from offset 1024, no max_tokens limit
        chunk_info = buffers.get_buf_chunk_info(
            allocation=allocation,
            offset_tokens=0,  # offset is relative to this new allocation
            max_tokens=None,
        )
        
        # Check input_embeddings chunk
        offset_bytes, size_bytes = chunk_info[0]
        expected_size = 976 * 4096 * 2  # remaining tokens
        assert size_bytes == expected_size


class TestContinuationScenario:
    """Integration tests for continuation scenario."""

    def test_scenario_one_time_transfer(self):
        """
        Test scenario: actual length (800) <= default buffer (1024)
        Expected: One-time transfer completes
        """
        allocator = ReqToMetadataBlockAllocator(
            total_tokens=8192,
            block_size=128,
        )
        
        buffers = MultimodalDataBuffers(
            size=8,
            max_prefill_tokens=8192,
            embedding_dim=4096,
            block_size=128,
            use_block_allocator=True,
        )
        
        # Language side: allocate default buffer (1024 tokens)
        default_tokens = 1024
        lang_allocation = allocator.alloc_default(
            default_tokens=default_tokens,
            req_id=1
        )
        assert lang_allocation is not None
        assert lang_allocation.num_tokens == 1024
        
        # Embedding side: actual length is 800 tokens (fits in one transfer)
        actual_length = 800
        
        # Calculate if continuation needed
        needs_continuation = actual_length > default_tokens
        assert needs_continuation is False  # Should complete in one transfer

    def test_scenario_continuation_needed(self):
        """
        Test scenario: actual length (2000) > default buffer (1024)
        Expected: Two-step transfer (first 1024, then remaining 976)
        """
        allocator = ReqToMetadataBlockAllocator(
            total_tokens=16384,
            block_size=128,
        )
        
        # Step 1: Language side allocates default buffer
        default_tokens = 1024
        lang_allocation_1 = allocator.alloc_default(
            default_tokens=default_tokens,
            req_id=1
        )
        assert lang_allocation_1 is not None
        assert lang_allocation_1.num_tokens == 1024
        
        # Embedding side knows actual length is 2000
        actual_length = 2000
        
        # Check if continuation needed
        needs_continuation = actual_length > default_tokens
        assert needs_continuation is True
        
        # First transfer: send 1024 tokens
        first_batch_size = min(actual_length, default_tokens)
        assert first_batch_size == 1024
        
        # Language side receives first batch and reads aux_data
        # Determines remaining tokens
        received_tokens = first_batch_size
        remaining_tokens = actual_length - received_tokens
        assert remaining_tokens == 976
        
        # Step 2: Language side frees first allocation and requests new one
        allocator.free(lang_allocation_1, req_id=1)
        
        # Allocate buffer for remaining tokens
        lang_allocation_2 = allocator.alloc(
            num_tokens=remaining_tokens,
            req_id=1
        )
        assert lang_allocation_2 is not None
        assert lang_allocation_2.num_tokens == 976
        
        # Second transfer: send remaining 976 tokens
        # Language side merges: 1024 + 976 = 2000 tokens
        total_received = received_tokens + remaining_tokens
        assert total_received == actual_length

    def test_scenario_buffer_shortage(self):
        """
        Test scenario: Buffer shortage during continuation request
        Expected: Request waits until buffer becomes available
        """
        allocator = ReqToMetadataBlockAllocator(
            total_tokens=2048,  # Limited capacity: 16 blocks
            block_size=128,
        )
        
        # Allocate 8 blocks for first transfer (1024 tokens)
        allocation_1 = allocator.alloc_default(default_tokens=1024, req_id=1)
        assert allocation_1 is not None
        assert allocator.available_blocks() == 8
        
        # Allocate remaining 8 blocks for other requests
        allocation_2 = allocator.alloc(num_tokens=1024, req_id=2)
        assert allocation_2 is not None
        assert allocator.available_blocks() == 0
        
        # Try to allocate for continuation (needs 8 blocks for 976 tokens)
        allocation_3 = allocator.alloc(num_tokens=976, req_id=1)
        assert allocation_3 is None  # Allocation fails - no buffer available
        
        # Free allocation_2 to make room
        allocator.free(allocation_2, req_id=2)
        assert allocator.available_blocks() == 8
        
        # Now continuation allocation succeeds
        allocation_3 = allocator.alloc(num_tokens=976, req_id=1)
        assert allocation_3 is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_exact_block_boundary(self):
        """Test allocation when tokens exactly match block boundaries."""
        allocator = ReqToMetadataBlockAllocator(
            total_tokens=8192,
            block_size=128,
        )
        
        # Allocate exactly 256 tokens (2 blocks)
        allocation = allocator.alloc(num_tokens=256, req_id=1)
        
        assert allocation is not None
        assert len(allocation.block_indices) == 2
        assert allocation.num_tokens == 256

    def test_single_token_allocation(self):
        """Test allocating a single token."""
        allocator = ReqToMetadataBlockAllocator(
            total_tokens=8192,
            block_size=128,
        )
        
        # Allocate 1 token (still needs 1 block)
        allocation = allocator.alloc(num_tokens=1, req_id=1)
        
        assert allocation is not None
        assert len(allocation.block_indices) == 1
        assert allocation.num_tokens == 1

    def test_fake_allocation(self):
        """Test fake allocation (for testing mode)."""
        allocator = ReqToMetadataBlockAllocator(
            total_tokens=8192,
            block_size=128,
        )
        
        allocation = allocator.alloc(num_tokens=500, fake=True, req_id=1)
        
        assert allocation is not None
        assert allocator.available_blocks() == 64  # No blocks actually allocated

    def test_fake_free(self):
        """Test fake free (for testing mode)."""
        allocator = ReqToMetadataBlockAllocator(
            total_tokens=8192,
            block_size=128,
        )
        
        allocation = allocator.alloc(num_tokens=500, req_id=1)
        initial_available = allocator.available_blocks()
        
        allocator.free(allocation, fake=True, req_id=1)
        
        # Fake free shouldn't change available blocks
        assert allocator.available_blocks() == initial_available


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
