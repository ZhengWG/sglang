# Multimodal Embedding Transmission Feature

## Overview

This document describes the **incomplete embedding block transmission** feature for the multimodal disaggregation system. This feature handles cases where the allocated cache block on the Language side is smaller than the embedding data that needs to be transmitted from the Embedding side.

## Problem Statement

Previously, the system assumed that a single cache allocation would be sufficient for the entire embedding data. However, in practice:

1. The Language side may allocate a cache block smaller than the total embedding size
2. This could lead to data truncation or transmission failures
3. No mechanism existed to handle multi-round transmissions

## Solution Design

The new feature implements a **multi-round transmission protocol** with the following components:

### 1. Transmission State Tracking

**`EmbeddingTransmissionState`** tracks the progress of multi-round transmissions:

```python
@dataclasses.dataclass
class EmbeddingTransmissionState:
    room: int
    embedding_index: int
    total_size_per_buffer: List[int]        # Total size for each buffer
    transmitted_size_per_buffer: List[int]  # How much has been transmitted
    transmission_count: int                 # Number of rounds completed
    
    def is_complete(self) -> bool:
        """Check if all data has been transmitted"""
        
    def get_remaining_chunks(self) -> List[Tuple[int, int]]:
        """Get chunk info for remaining data"""
```

### 2. Message Types

- **`REQUEST_MORE_CACHE`**: Language → Embedding request for additional cache allocation
- **`EMBEDDING_DATA`**: Embedding → Language data transmission with transmission_id tracking

### 3. Key Components

#### Embedding Side (`MooncakeEmbeddingSender`)

```python
def send_embedding(
    self, 
    embedding_index: int, 
    last_chunk: bool, 
    chunk_info: List[Tuple[int, int]],
    total_sizes: Optional[List[int]] = None,
):
    """
    Send embedding data to language instances.
    
    Args:
        embedding_index: Index of the embedding data
        last_chunk: Whether this is expected to be the last chunk
        chunk_info: List of (offset, size) tuples for each buffer
        total_sizes: Total sizes for each buffer (first transmission only)
    """

def continue_transmission(self, chunk_info: List[Tuple[int, int]]):
    """
    Continue transmission with new chunk allocation from Language side.
    Called after receiving notification that more cache has been allocated.
    """
```

#### Language Side (`MooncakeEmbeddingReceiver`)

```python
def request_more_cache(self, new_chunk_info: List[Tuple[int, int]]):
    """
    Request more cache allocation from Embedding side.
    Called when allocated block is smaller than needed.
    
    Args:
        new_chunk_info: Updated allocation with (offset, size) for each buffer
    """
```

## Usage Example

### Scenario: Embedding data is 1000MB, but Language side can only allocate 400MB at a time

#### Round 1: Initial Transmission

**Language Side:**
```python
# Allocate initial cache (400MB available)
receiver = MooncakeEmbeddingReceiver(mgr, bootstrap_addr, room_id)
receiver.init(embedding_index=0)

# Initial chunk_info: [(0, 400MB)] for first buffer
```

**Embedding Side:**
```python
# Send first chunk
sender = MooncakeEmbeddingSender(mgr, bootstrap_addr, room_id, dest_ranks, pp_rank)
sender.init(embedding_index=0)

# total_sizes = [1000MB] - full size of embedding
# chunk_info = [(0, 400MB)] - what Language side allocated
sender.send_embedding(
    embedding_index=0,
    last_chunk=False,  # More data remains
    chunk_info=[(0, 400 * 1024 * 1024)],
    total_sizes=[1000 * 1024 * 1024]
)

# Status becomes KVPoll.Transferring (not Success yet)
```

#### Round 2: Request More Cache

**Language Side:**
```python
# Check status
status = receiver.poll()
if status == KVPoll.Transferring:
    # Allocate more cache (another 400MB)
    # new_chunk_info: [(400MB, 400MB)] - continue from offset 400MB
    receiver.request_more_cache(
        new_chunk_info=[(400 * 1024 * 1024, 400 * 1024 * 1024)]
    )
```

**Embedding Side:**
```python
# Automatically triggered by request_more_cache
# Manager receives REQUEST_MORE_CACHE and continues transmission
# chunk_info = [(400MB, 400MB)] - send next chunk
```

#### Round 3: Final Transmission

**Language Side:**
```python
# Allocate final cache (200MB needed)
status = receiver.poll()
if status == KVPoll.Transferring:
    receiver.request_more_cache(
        new_chunk_info=[(800 * 1024 * 1024, 200 * 1024 * 1024)]
    )
```

**Embedding Side:**
```python
# Send final chunk
# chunk_info = [(800MB, 200MB)]
# transmission_state.is_complete() returns True
# Status becomes KVPoll.Success
```

## Workflow Diagram

```
Embedding Side                          Language Side
     |                                       |
     |  1. Bootstrap & Init                  |
     |<------------------------------------->|
     |                                       |
     |  2. Send First Chunk (400MB)          |
     |-------------------------------------->|
     |   Status: Transferring                |
     |                                       |
     |  3. REQUEST_MORE_CACHE                |
     |<--------------------------------------|
     |   new_chunk_info: [(400MB, 400MB)]    |
     |                                       |
     |  4. Send Second Chunk (400MB)         |
     |-------------------------------------->|
     |   Status: Transferring                |
     |                                       |
     |  5. REQUEST_MORE_CACHE                |
     |<--------------------------------------|
     |   new_chunk_info: [(800MB, 200MB)]    |
     |                                       |
     |  6. Send Final Chunk (200MB)          |
     |-------------------------------------->|
     |   Status: Success                     |
     |                                       |
```

## API Reference

### MooncakeEmbeddingManager

#### Methods

**`send_embedding(mooncake_session_id, embedding_index, dst_embedding_ptrs, dst_embedding_index, chunk_info)`**
- Sends embedding data according to chunk_info
- Each tuple in chunk_info is (offset, size) for the corresponding buffer
- Skips buffers with size 0

**`send_request_more_cache_to_embedding(remote, dst_port, room, mooncake_session_id, dst_embedding_index, new_chunk_info, transmission_id)`**
- Language side sends request to Embedding side for more cache
- Includes updated chunk allocation info

**`_handle_request_more_cache(msg)`**
- Embedding side handles request for more cache from Language side
- Updates transfer info and schedules continuation

**`add_transfer_request(bootstrap_room, embedding_index, is_last, chunk_info, transmission_id, total_sizes)`**
- Adds a transfer request to the queue
- Initializes transmission state on first transmission
- Tracks multi-round transmissions with transmission_id

### MooncakeEmbeddingSender

**`send_embedding(embedding_index, last_chunk, chunk_info, total_sizes)`**
- Send embedding data to language instances
- total_sizes only needed for first transmission

**`continue_transmission(chunk_info)`**
- Continue transmission with new chunk allocation
- Automatically increments transmission_id

### MooncakeEmbeddingReceiver

**`request_more_cache(new_chunk_info)`**
- Request more cache allocation from Embedding side
- Updates transmission_id and resets status to Transferring

## Status States

| State | Description |
|-------|-------------|
| `KVPoll.Bootstrapping` | Initial connection setup |
| `KVPoll.WaitingForInput` | Ready to receive data |
| `KVPoll.Transferring` | Partial transmission in progress |
| `KVPoll.Success` | All data transmitted successfully |
| `KVPoll.Failed` | Transmission failed |

## Implementation Details

### Thread Safety

- `transmission_state_lock`: Protects transmission state updates
- `session_lock`: Protects session failure tracking
- `pending_cache_lock`: Protects pending cache request tracking

### Error Handling

1. **Failed Sessions**: Tracked and early-exited to avoid retries
2. **Timeouts**: Bootstrap and waiting timeouts configurable via environment variables
3. **Node Failures**: Heartbeat checker detects and handles dead nodes

### Performance Considerations

- Thread pool for parallel transfers
- Queue sharding to prevent head-of-line blocking
- Zero-copy transfers using Mooncake engine

## Configuration

Environment variables:

- `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE`: Thread pool size (default: based on CPU count)
- `SGLANG_DISAGGREGATION_QUEUE_SIZE`: Number of transfer queues (default: 4)
- `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT`: Bootstrap timeout in seconds (default: 30)
- `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL`: Heartbeat interval (default: 5.0)
- `SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE`: Max heartbeat failures (default: 2)

## Testing

To test the multi-round transmission:

1. Configure Language side to allocate smaller cache blocks
2. Use large embedding data (> available cache)
3. Monitor logs for "REQUEST_MORE_CACHE" and "Continuing transmission" messages
4. Verify final status is `KVPoll.Success`
5. Check that all data was transmitted correctly

## Limitations

- Currently supports homogeneous buffer types (all buffers must be of same type)
- Requires Mooncake transfer engine
- Network latency may impact multi-round transmission performance

## Future Enhancements

1. Adaptive chunk sizing based on available memory
2. Compression for large embeddings
3. Prefetching to reduce round-trip delays
4. Support for heterogeneous buffer types
