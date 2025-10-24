# Qwen3-MoE-VL Disaggregation with DeepStack Embedding Support

## Architecture Overview

The disaggregation architecture separates multimodal model processing into two stages:
- **Encode Side**: Processes images/videos, generates embeddings and deepstack embeddings
- **Language Side**: Receives embeddings and runs language model with deepstack support

## Current State Analysis

### ✅ Already Implemented
1. **Model Support** (`qwen3_vl_moe.py`):
   - `Qwen3MoeLLMModel.forward()` accepts `input_deepstack_embeds` parameter
   - Deepstack embeddings are added to hidden states in first 3 layers (lines 122-127)
   - Model has `use_deepstack` property and `deepstack_visual_indexes`

2. **MM Utils** (`mm_utils.py`):
   - `general_mm_embed_routine()` extracts deepstack embeddings using `use_deepstack` flag
   - Separates regular embeddings and deepstack embeddings via `separate_deepstack_embeds()`
   - Creates `input_deepstack_embeds` tensor for model forward

3. **Transfer Infrastructure**:
   - Block-based allocation system (`ReqToMetadataBlockAllocator`)
   - Multi-buffer transfer protocol (embeddings, fill_ids, mrope_positions, aux_datas)
   - Resume transfer support for partial data

### ❌ Missing Components

1. **Buffer Structure** (`utils.py`):
   - No `deepstack_embeddings` buffer in `MultimodalDataBuffers`
   - Need to extend buffer info methods

2. **Encode Side** (`multimodal_embedding.py`):
   - No deepstack extraction during embedding processing
   - No deepstack storage in buffer before transfer

3. **Language Side** (`multimodal_language.py`):
   - No deepstack retrieval from buffer after transfer
   - No deepstack passing to model forward

4. **Transfer Protocol** (`conn_multimodal.py`):
   - Transfer logic doesn't include deepstack buffer
   - Block size calculations don't account for deepstack data

## Implementation Plan

### ✅ Phase 0: Model Layer Refactoring (COMPLETED)

**Goal**: Enable Language side to use pure text model `Qwen3MoeForCausalLM` with deepstack support

**Motivation**: 
- Language side should use pure text model, not VL model with visual encoder
- Only `Qwen3MoeLLMModel` (in qwen3_vl_moe.py) originally supported deepstack
- Need to add deepstack support to base `Qwen2MoeModel` and `Qwen3MoeForCausalLM`

**Changes Made**:

1. **`qwen2_moe.py` - `Qwen2MoeModel`**:
   - Added `self.hidden_size = config.hidden_size` in `__init__`
   - Added `input_deepstack_embeds` parameter to `forward()`
   - Added deepstack processing after each layer forward (layers 0-2 only):
     ```python
     if input_deepstack_embeds is not None and i in range(3):
         sep = self.hidden_size * i
         hidden_states.add_(
             input_deepstack_embeds[:, sep : sep + self.hidden_size]
         )
     ```

2. **`qwen2_moe.py` - `Qwen2MoeForCausalLM`**:
   - Added `input_deepstack_embeds` parameter to `forward()`
   - Pass parameter through to `self.model()`

3. **`qwen3_moe.py` - `Qwen3MoeForCausalLM`**:
   - Added `input_deepstack_embeds` parameter to `forward()`
   - Pass parameter through to `self.model()`

**Benefits**:
- Language side can use `Qwen3MoeForCausalLM` (pure text model)
- No visual encoder overhead on Language side
- Unified deepstack interface across all Qwen MoE models
- Fully backward compatible (parameter is optional, defaults to None)

**Documentation**: See `REFACTORING_SUMMARY.md` for detailed changes

---

### Phase 1: Extend Buffer Structure (`utils.py`)

**Goal**: Add deepstack_embeddings buffer to MultimodalDataBuffers

**Changes**:
```python
class MultimodalDataBuffers:
    def __init__(self, size: int, block_size: int, embedding_dim: int = 8192, 
                 num_deepstack_embeddings: int = 3):
        # Add new buffer
        self.deepstack_embeddings = torch.zeros(
            (size, block_size * embedding_dim * num_deepstack_embeddings),
            dtype=torch.bfloat16,
            device="cpu",
        )
        
    def get_buf_infos(self):
        # Add deepstack_embeddings to ptrs, data_lens, item_lens
        
    def get_buf(self, block_indices, actual_total_length=None):
        # Add deepstack_embeddings gathering logic
        
    def set_buf(self, req):
        # Add deepstack_embeddings scattering logic
```

**Key Considerations**:
- DeepStack dimension: `embedding_dim * num_deepstack_embeddings` (e.g., 8192 * 3 = 24576)
- Block-based storage: scatter/gather across blocks like regular embeddings
- Only transmitted once during initial transfer (not in resume)

### Phase 2: Encode Side Updates (`multimodal_embedding.py`)

**Goal**: Extract and store deepstack embeddings before transfer

**Changes**:

1. **Extract deepstack in forward**:
```python
def process_batch_result_disagg_multimodal_embedding(self, batch, result):
    embeddings = result.embeddings
    
    for i, req in enumerate(batch.reqs):
        embedding = embeddings[...]
        
        # NEW: Extract deepstack if model supports it
        if hasattr(self.model, 'use_deepstack') and self.model.use_deepstack:
            embedding, deepstack_embedding = (
                self.model.separate_deepstack_embeds(embedding)
            )
            req.deepstack_embedding = deepstack_embedding
        
        req.embedding = embedding
```

2. **Store deepstack in buffer**:
```python
def send_embedding_chunk(self, req, last_chunk=True):
    if last_chunk:
        self.disagg_metadata_buffers.set_buf(req)  # Will include deepstack
```

**Key Considerations**:
- Check if model has `use_deepstack` property
- Handle case where deepstack is None (non-deepstack models)
- Store deepstack in req for buffer access

### Phase 3: Language Side Updates (`multimodal_language.py`)

**Goal**: Receive deepstack embeddings and pass to model

**Changes**:

1. **Retrieve deepstack from buffer**:
```python
def pop_transferred(self):
    # In the Success case
    if poll == KVPoll.Success:
        embedding_data, fill_ids, mrope_positions, aux_datas, deepstack_data = (
            self.metadata_buffers.get_buf(block_indices=block_indices)
        )
        
        language_req.req.input_embeds = embedding_data
        language_req.req.origin_input_ids = fill_ids.tolist()
        
        # NEW: Store deepstack embeddings
        if deepstack_data is not None:
            language_req.req.input_deepstack_embeds = deepstack_data
```

2. **Pass deepstack to model forward**:
The scheduler will automatically pass `req.input_deepstack_embeds` to model forward through
the existing mechanism (already handled in `general_mm_embed_routine`).

**Key Considerations**:
- Handle resume transfers (deepstack only in initial transfer)
- Check if deepstack is None for backward compatibility
- Store in req for scheduler access

### Phase 4: Transfer Protocol Updates (`conn_multimodal.py`)

**Goal**: Include deepstack buffer in transfer operations

**Changes**:

1. **Update buffer registration**:
```python
def register_buffer_to_engine(self):
    for aux_data_ptr, aux_data_len in zip(
        self.data_args.aux_data_ptrs, 
        self.data_args.aux_data_lens
    ):
        self.engine.register(aux_data_ptr, aux_data_len)
    # aux_data_ptrs now includes: [embeddings, fill_ids, mrope, aux_datas, deepstack]
```

2. **Update transfer logic**:
```python
def send_embedding(self, ...):
    # Loop through buffer types
    for buffer_type_idx in range(len(self.data_args.aux_item_lens)):
        # buffer_type_idx: 0=embeddings, 1=fill_ids, 2=mrope, 3=aux_datas, 4=deepstack
        
        if buffer_type_idx == 4:  # deepstack
            if sent_tokens == 0 and block_idx == 0:
                # Only transfer deepstack in initial transfer, first block
                chunk_size = embedding_item_len
            else:
                continue  # Skip deepstack in resume/other blocks
        elif buffer_type_idx == 3:  # aux_datas
            # Existing logic
        else:
            # Regular buffers: scale by tokens_in_block
```

**Key Considerations**:
- DeepStack only transferred in initial transfer (like aux_datas)
- Resume transfers skip deepstack (already received)
- Buffer ordering: [embeddings, fill_ids, mrope, aux_datas, deepstack]

### Phase 5: Buffer Dimension Configuration

**Goal**: Properly configure deepstack dimensions

**Changes**:

1. **Get num_deepstack_embeddings from model**:
```python
# In scheduler initialization
if hasattr(model, 'num_deepstack_embeddings'):
    num_deepstack = model.num_deepstack_embeddings
else:
    num_deepstack = 0

metadata_buffers = MultimodalDataBuffers(
    size=buffer_size,
    block_size=block_size,
    embedding_dim=hidden_size,
    num_deepstack_embeddings=num_deepstack,
)
```

**Key Considerations**:
- For Qwen3-VL-MoE: `num_deepstack_embeddings = 3`
- For models without deepstack: `num_deepstack_embeddings = 0`
- Buffer size scales with num_deepstack

## Data Flow Diagram

```
Encode Side:
  1. Forward pass → embeddings (hidden_size)
  2. separate_deepstack_embeds() → regular_embedding + deepstack_embedding
     - regular_embedding: (seq_len, hidden_size)
     - deepstack_embedding: (seq_len, hidden_size * 3)
  3. Store in buffer blocks
  4. Transfer via Mooncake

Language Side:
  1. Receive data in buffer blocks
  2. Gather from blocks → regular_embedding + deepstack_embedding
  3. Pass to model forward:
     - input_embeds = regular_embedding
     - input_deepstack_embeds = deepstack_embedding
  4. Model adds deepstack to layers 0-2
```

## Memory Layout

```
Block Structure (per block):
┌─────────────────────────────────────┐
│ embeddings (block_size * embed_dim) │  → Regular token embeddings
├─────────────────────────────────────┤
│ fill_ids (block_size)               │  → Original token IDs
├─────────────────────────────────────┤
│ mrope_positions (3 * block_size)    │  → M-RoPE position info
├─────────────────────────────────────┤
│ aux_datas (16)                      │  → Metadata (length, delta, etc.)
├─────────────────────────────────────┤
│ deepstack_embeddings                │  → DeepStack features
│ (block_size * embed_dim * 3)        │     (3 layers worth)
└─────────────────────────────────────┘
```

## Testing Strategy

1. **Unit Tests**:
   - Test buffer allocation/deallocation with deepstack
   - Test scatter/gather operations
   - Test transfer with/without deepstack

2. **Integration Tests**:
   - End-to-end disaggregation with Qwen3-VL-MoE
   - Verify deepstack values match non-disaggregated mode
   - Test resume transfers (deepstack should not be re-sent)

3. **Compatibility Tests**:
   - Models without deepstack should work (num_deepstack=0)
   - Backward compatibility with existing transfers

## Rollout Plan

1. ✅ Create implementation plan (this document)
2. Implement Phase 1: Buffer structure
3. Implement Phase 2: Encode side
4. Implement Phase 3: Language side
5. Implement Phase 4: Transfer protocol
6. Implement Phase 5: Configuration
7. Test and validate
8. Code review and merge

## Open Questions

1. **Q**: Should deepstack be sent in every block or only first block?
   **A**: Only in first block of initial transfer (like aux_datas). Resume transfers skip it.

2. **Q**: What if embedding and deepstack have different lengths?
   **A**: They should always have same seq_len dimension. Deepstack is wider (3x hidden_size).

3. **Q**: How to handle models without deepstack support?
   **A**: Set `num_deepstack_embeddings=0`, buffer size=0, skip all deepstack logic.

4. **Q**: Performance impact of larger buffer size?
   **A**: Deepstack adds 3x memory for embeddings. Monitor and adjust buffer_size if needed.

## References

- Model implementation: `python/sglang/srt/models/qwen3_vl_moe.py`
- MM utilities: `python/sglang/srt/managers/mm_utils.py`
- Buffer utils: `python/sglang/srt/disaggregation/utils.py`
- Encode side: `python/sglang/srt/disaggregation/multimodal_embedding.py`
- Language side: `python/sglang/srt/disaggregation/multimodal_language.py`
- Transfer: `python/sglang/srt/disaggregation/mooncake/conn_multimodal.py`
