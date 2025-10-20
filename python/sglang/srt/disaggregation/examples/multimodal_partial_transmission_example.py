"""
Example demonstrating multi-round embedding transmission when cache allocation is insufficient.

This example shows how to handle scenarios where the Language side cannot allocate
enough cache in a single round to receive the full embedding data from the Embedding side.
"""

import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class PartialTransmissionExample:
    """
    Example demonstrating the multi-round transmission protocol.
    
    Scenario: 
    - Embedding data size: 1000 MB
    - Language side can only allocate 400 MB per round
    - Requires 3 rounds to complete transmission
    """
    
    def __init__(self):
        self.embedding_total_size = 1000 * 1024 * 1024  # 1000 MB
        self.cache_allocation_limit = 400 * 1024 * 1024  # 400 MB per round
        self.transmitted_bytes = 0
        
    def embedding_side_workflow(self, sender):
        """
        Embedding side workflow for multi-round transmission.
        
        Args:
            sender: MooncakeEmbeddingSender instance
        """
        logger.info("=== Embedding Side Workflow ===")
        
        # Round 1: Initial transmission
        logger.info("Round 1: Sending initial chunk")
        embedding_index = 0
        
        # Calculate how much can be sent in first round
        first_chunk_size = min(self.cache_allocation_limit, self.embedding_total_size)
        chunk_info = [(0, first_chunk_size)]  # (offset, size)
        
        sender.init(embedding_index=embedding_index)
        sender.send_embedding(
            embedding_index=embedding_index,
            last_chunk=False,  # More data remains
            chunk_info=chunk_info,
            total_sizes=[self.embedding_total_size],  # Total size info
        )
        
        logger.info(f"Sent {first_chunk_size / 1024 / 1024:.0f} MB in round 1")
        self.transmitted_bytes += first_chunk_size
        
        # Round 2: Continue transmission after receiving REQUEST_MORE_CACHE
        logger.info("Round 2: Continuing transmission")
        second_chunk_size = min(
            self.cache_allocation_limit, 
            self.embedding_total_size - self.transmitted_bytes
        )
        chunk_info = [(self.transmitted_bytes, second_chunk_size)]
        
        sender.continue_transmission(chunk_info=chunk_info)
        
        logger.info(f"Sent {second_chunk_size / 1024 / 1024:.0f} MB in round 2")
        self.transmitted_bytes += second_chunk_size
        
        # Round 3: Final transmission
        logger.info("Round 3: Final chunk")
        remaining_size = self.embedding_total_size - self.transmitted_bytes
        chunk_info = [(self.transmitted_bytes, remaining_size)]
        
        sender.continue_transmission(chunk_info=chunk_info)
        
        logger.info(f"Sent {remaining_size / 1024 / 1024:.0f} MB in round 3")
        logger.info(f"Total transmitted: {self.embedding_total_size / 1024 / 1024:.0f} MB")
        
    def language_side_workflow(self, receiver):
        """
        Language side workflow for requesting more cache and receiving data.
        
        Args:
            receiver: MooncakeEmbeddingReceiver instance
        """
        logger.info("=== Language Side Workflow ===")
        
        # Round 1: Initial cache allocation
        logger.info("Round 1: Allocating initial cache (400 MB)")
        embedding_index = 0
        receiver.init(embedding_index=embedding_index)
        
        # Simulate cache allocation
        allocated_size = self.cache_allocation_limit
        logger.info(f"Allocated {allocated_size / 1024 / 1024:.0f} MB cache")
        
        # Wait for initial data
        received_bytes = allocated_size
        logger.info(f"Received {received_bytes / 1024 / 1024:.0f} MB")
        
        # Round 2: Request more cache
        logger.info("Round 2: Requesting more cache")
        offset = received_bytes
        new_allocation = min(self.cache_allocation_limit, self.embedding_total_size - received_bytes)
        new_chunk_info = [(offset, new_allocation)]
        
        receiver.request_more_cache(new_chunk_info=new_chunk_info)
        logger.info(f"Requested {new_allocation / 1024 / 1024:.0f} MB more cache at offset {offset / 1024 / 1024:.0f} MB")
        
        received_bytes += new_allocation
        logger.info(f"Received {new_allocation / 1024 / 1024:.0f} MB")
        
        # Round 3: Request final cache
        logger.info("Round 3: Requesting final cache")
        offset = received_bytes
        final_allocation = self.embedding_total_size - received_bytes
        new_chunk_info = [(offset, final_allocation)]
        
        receiver.request_more_cache(new_chunk_info=new_chunk_info)
        logger.info(f"Requested {final_allocation / 1024 / 1024:.0f} MB more cache at offset {offset / 1024 / 1024:.0f} MB")
        
        received_bytes += final_allocation
        logger.info(f"Received {final_allocation / 1024 / 1024:.0f} MB")
        logger.info(f"Total received: {received_bytes / 1024 / 1024:.0f} MB")
        
    def demonstrate_chunk_calculation(self):
        """
        Demonstrate how to calculate chunk_info for multi-round transmission.
        """
        logger.info("\n=== Chunk Calculation Example ===")
        
        total_size = self.embedding_total_size
        max_chunk = self.cache_allocation_limit
        
        chunks: List[Tuple[int, int]] = []
        offset = 0
        
        while offset < total_size:
            remaining = total_size - offset
            chunk_size = min(max_chunk, remaining)
            chunks.append((offset, chunk_size))
            offset += chunk_size
        
        logger.info(f"Total size: {total_size / 1024 / 1024:.0f} MB")
        logger.info(f"Max chunk: {max_chunk / 1024 / 1024:.0f} MB")
        logger.info(f"Number of rounds needed: {len(chunks)}")
        
        for i, (offset, size) in enumerate(chunks, 1):
            logger.info(
                f"Round {i}: offset={offset / 1024 / 1024:.0f} MB, "
                f"size={size / 1024 / 1024:.0f} MB"
            )
        
        return chunks


def main():
    """
    Main function to run the example.
    """
    example = PartialTransmissionExample()
    
    # Demonstrate chunk calculation
    chunks = example.demonstrate_chunk_calculation()
    
    logger.info("\n" + "=" * 60)
    logger.info("Multi-Round Transmission Protocol Example")
    logger.info("=" * 60 + "\n")
    
    logger.info("Scenario:")
    logger.info(f"  - Total embedding size: {example.embedding_total_size / 1024 / 1024:.0f} MB")
    logger.info(f"  - Cache allocation limit: {example.cache_allocation_limit / 1024 / 1024:.0f} MB")
    logger.info(f"  - Rounds needed: {len(chunks)}\n")
    
    # Note: In real usage, you would create actual sender/receiver instances
    logger.info("NOTE: This is a demonstration of the workflow.")
    logger.info("In production, replace with actual MooncakeEmbeddingSender and")
    logger.info("MooncakeEmbeddingReceiver instances.\n")
    
    # Show workflow steps
    logger.info("Workflow Summary:")
    for i, (offset, size) in enumerate(chunks, 1):
        logger.info(f"  Round {i}:")
        logger.info(f"    - Embedding side: Send chunk at offset {offset / 1024 / 1024:.0f} MB, size {size / 1024 / 1024:.0f} MB")
        if i < len(chunks):
            logger.info(f"    - Language side: Request more cache for next round")
        else:
            logger.info(f"    - Language side: Transmission complete")
        logger.info("")


if __name__ == "__main__":
    main()
