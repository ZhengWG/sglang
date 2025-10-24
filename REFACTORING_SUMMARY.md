# Qwen3-MoE-VL DeepStack Refactoring Summary

## ✅ Completed: Model Layer Refactoring

### Problem Statement
In disaggregation architecture, the Language side should use pure text model `Qwen3MoeForCausalLM` (from `qwen3_moe.py`) instead of the VL model. However, only `Qwen3MoeLLMModel` (in `qwen3_vl_moe.py`) supported `input_deepstack_embeds` parameter.

### Solution
Added `input_deepstack_embeds` support to the base model classes, making deepstack functionality available for pure text models used in Language side disaggregation.

## Modified Files

### 1. `python/sglang/srt/models/qwen2_moe.py`

#### Changes to `Qwen2MoeModel.__init__`:
```python
# ADDED: Store hidden_size for deepstack processing
self.hidden_size = config.hidden_size
```

#### Changes to `Qwen2MoeModel.forward()`:
```python
# ADDED: New parameter
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds: torch.Tensor = None,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
    input_deepstack_embeds: Optional[torch.Tensor] = None,  # NEW
) -> Union[torch.Tensor, PPProxyTensors]:
```

```python
# ADDED: DeepStack processing after each layer forward
for i in range(self.start_layer, self.end_layer):
    # ... existing layer forward ...
    hidden_states, residual = layer(...)
    
    # Process deepstack embeddings for first 3 layers
    if input_deepstack_embeds is not None and i in range(3):
        sep = self.hidden_size * i
        hidden_states.add_(
            input_deepstack_embeds[:, sep : sep + self.hidden_size]
        )
```

#### Changes to `Qwen2MoeForCausalLM.forward()`:
```python
# ADDED: New parameter and pass-through
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds: torch.Tensor = None,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
    input_deepstack_embeds: Optional[torch.Tensor] = None,  # NEW
) -> torch.Tensor:
    hidden_states = self.model(
        input_ids,
        positions,
        forward_batch,
        input_embeds,
        pp_proxy_tensors=pp_proxy_tensors,
        input_deepstack_embeds=input_deepstack_embeds,  # NEW: Pass to model
    )
```

### 2. `python/sglang/srt/models/qwen3_moe.py`

#### Changes to `Qwen3MoeForCausalLM.forward()`:
```python
# ADDED: New parameter and pass-through
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds: torch.Tensor = None,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
    input_deepstack_embeds: Optional[torch.Tensor] = None,  # NEW
) -> torch.Tensor:
    hidden_states = self.model(
        input_ids,
        positions,
        forward_batch,
        input_embeds,
        pp_proxy_tensors=pp_proxy_tensors,
        input_deepstack_embeds=input_deepstack_embeds,  # NEW: Pass to model
    )
```

## Architecture Impact

### Before Refactoring:
```
Encode Side (qwen3_vl_moe.py):
  Qwen3VLMoeForConditionalGeneration
    ├── visual (Qwen3_VisionTransformer)
    └── model (Qwen3MoeLLMModel) ✓ Has deepstack support
        └── Qwen3MoeModel

Language Side:
  ❌ Must use Qwen3VLMoeForConditionalGeneration (includes unnecessary visual encoder)
  OR
  ❌ Cannot use Qwen3MoeForCausalLM (no deepstack support)
```

### After Refactoring:
```
Encode Side (qwen3_vl_moe.py):
  Qwen3VLMoeForConditionalGeneration
    ├── visual (Qwen3_VisionTransformer)
    └── model (Qwen3MoeLLMModel) ✓ Has deepstack support
        └── Qwen3MoeModel ✓ NOW has deepstack support

Language Side (qwen3_moe.py):
  ✅ Qwen3MoeForCausalLM ✓ NOW has deepstack support
      └── Qwen3MoeModel ✓ NOW has deepstack support
```

## DeepStack Processing Logic

The deepstack embeddings are added to the hidden states of the **first 3 layers only**:

```python
# Layer 0: Add deepstack[:, 0:hidden_size]
# Layer 1: Add deepstack[:, hidden_size:hidden_size*2]
# Layer 2: Add deepstack[:, hidden_size*2:hidden_size*3]
# Layer 3+: No deepstack addition
```

This matches the behavior in `Qwen3MoeLLMModel` from `qwen3_vl_moe.py`.

## Backward Compatibility

✅ **Fully backward compatible**:
- `input_deepstack_embeds` is an optional parameter (default: `None`)
- When `None`, the models behave exactly as before
- No changes to existing model loading or inference paths
- Models without deepstack support (non-VL models) simply pass `None`

## Benefits

1. **Clean Separation**: Language side can use pure text model without visual encoder overhead
2. **Unified Interface**: All Qwen MoE models now have consistent deepstack support
3. **Maintainability**: Single implementation in base class instead of duplicated in VL variant
4. **Flexibility**: Easy to add deepstack to other model architectures by following same pattern

## Next Steps

With model layer refactoring complete, the remaining tasks are:

1. **Phase 1**: Extend `MultimodalDataBuffers` to support `deepstack_embeddings` storage
2. **Phase 2**: Update Encode side to extract and transmit deepstack embeddings
3. **Phase 3**: Update Language side to receive and pass deepstack embeddings to model ✓ (model ready)
4. **Phase 4**: Update transfer protocol to handle deepstack embeddings blocks
5. **Phase 5**: Test and validate end-to-end

## Testing Recommendations

1. **Unit Tests**:
   - Test forward pass with `input_deepstack_embeds=None` (backward compatibility)
   - Test forward pass with valid deepstack tensors
   - Verify deepstack is only added to layers 0-2

2. **Integration Tests**:
   - Compare outputs between:
     - Qwen3VLMoeForConditionalGeneration with deepstack
     - Qwen3MoeForCausalLM with same deepstack input
   - Should produce identical results (for language-only forward)

3. **Disaggregation Tests**:
   - End-to-end test with Encode/Language split
   - Verify deepstack values transmitted correctly
   - Check performance metrics
