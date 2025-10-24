# Qwen3-MoE-VL Disaggregation Implementation Status

## 🎯 Objective
实现 qwen3-moe-vl 的 encode/language 分离，支持 deepstack_embedding 的传输和处理。

## ✅ Phase 0: 模型层重构与简化 (已完成)

### 问题分析
- **原问题**: Language侧应该使用纯文本模型 `Qwen3MoeForCausalLM`，但它不支持 `input_deepstack_embeds`
- **原因**: 只有 `Qwen3MoeLLMModel` (在 qwen3_vl_moe.py) 支持 deepstack
- **影响**: Language侧被迫使用包含 visual encoder 的完整 VL 模型，造成不必要的开销

### 解决方案
为基础模型类添加 deepstack 支持，使纯文本模型可以在 Language 侧使用。

### 修改文件

#### 1. `python/sglang/srt/models/qwen2_moe.py`

**`Qwen2MoeModel` 修改**:
```python
# __init__ 添加
self.hidden_size = config.hidden_size

# forward() 添加参数
def forward(
    ...,
    input_deepstack_embeds: Optional[torch.Tensor] = None,
)

# forward() 层循环后添加 deepstack 处理
for i in range(self.start_layer, self.end_layer):
    hidden_states, residual = layer(...)
    
    # 仅在前3层添加 deepstack
    if input_deepstack_embeds is not None and i in range(3):
        sep = self.hidden_size * i
        hidden_states.add_(
            input_deepstack_embeds[:, sep : sep + self.hidden_size]
        )
```

**`Qwen2MoeForCausalLM` 修改**:
```python
def forward(
    ...,
    input_deepstack_embeds: Optional[torch.Tensor] = None,
):
    hidden_states = self.model(
        ...,
        input_deepstack_embeds=input_deepstack_embeds,
    )
```

#### 2. `python/sglang/srt/models/qwen3_moe.py`

**`Qwen3MoeForCausalLM` 修改**:
```python
def forward(
    ...,
    input_deepstack_embeds: Optional[torch.Tensor] = None,
):
    hidden_states = self.model(
        ...,
        input_deepstack_embeds=input_deepstack_embeds,
    )
```

### 架构变化

**重构前**:
```
Language Side 只能选择:
❌ Qwen3VLMoeForConditionalGeneration (包含不必要的 visual encoder)
❌ Qwen3MoeForCausalLM (不支持 deepstack)
```

**重构后**:
```
Language Side 现在可以使用:
✅ Qwen3MoeForCausalLM (纯文本模型 + deepstack 支持)
```

### DeepStack 处理逻辑

```python
# input_deepstack_embeds shape: (seq_len, hidden_size * 3)
#
# Layer 0: 添加 deepstack[:, 0:hidden_size]
# Layer 1: 添加 deepstack[:, hidden_size:hidden_size*2]  
# Layer 2: 添加 deepstack[:, hidden_size*2:hidden_size*3]
# Layer 3+: 不添加
```

### 向后兼容性
✅ 完全向后兼容:
- `input_deepstack_embeds` 是可选参数（默认 `None`）
- 传入 `None` 时，模型行为与之前完全一致
- 不支持 deepstack 的模型直接传 `None` 即可

### 验证
- ✅ Git diff 检查通过
- ✅ 无 linter errors
- ✅ 语法正确

### Phase 0.2: 简化 qwen3_vl_moe.py (已完成)

**删除重复代码**:
- ❌ 删除了整个 `Qwen3MoeLLMModel` 类 (90行)
- ✅ `Qwen3VLMoeForConditionalGeneration` 直接使用 `Qwen3MoeModel`
- ✅ 移动 `get_image_feature()` 到正确位置

**架构改进**:
```
Before: Qwen3VLMoeForConditionalGeneration → Qwen3MoeLLMModel → Qwen3MoeModel
After:  Qwen3VLMoeForConditionalGeneration → Qwen3MoeModel (直接使用)
```

**净减少**: 90 行重复代码

详见: `SIMPLIFICATION_SUMMARY.md`

---

## 📋 待实现阶段

### Phase 1: 扩展缓冲区结构 (`utils.py`)
- [ ] 在 `MultimodalDataBuffers` 添加 `deepstack_embeddings` 缓冲区
- [ ] 更新 `get_buf_infos()` 包含 deepstack buffer 信息
- [ ] 更新 `get_buf()` 支持 deepstack 数据聚合
- [ ] 更新 `set_buf()` 支持 deepstack 数据分散存储

**关键参数**:
- `num_deepstack_embeddings = 3` (对于 Qwen3-VL-MoE)
- Buffer 大小: `(size, block_size * embedding_dim * 3)`

### Phase 2: Encode侧更新 (`multimodal_embedding.py`)
- [ ] 在 `process_batch_result_disagg_multimodal_embedding` 提取 deepstack
- [ ] 使用 `model.separate_deepstack_embeds()` 分离 embeddings
- [ ] 存储到 `req.deepstack_embedding`
- [ ] 在 `send_embedding_chunk` 中通过 buffer 传输

### Phase 3: Language侧更新 (`multimodal_language.py`)
- [ ] 在 `pop_transferred` 从 buffer 获取 deepstack
- [ ] 存储到 `req.input_deepstack_embeds`
- [ ] ✅ 模型 forward 已支持 (Phase 0 完成)

### Phase 4: 传输协议更新 (`conn_multimodal.py`)
- [ ] 在 `register_buffer_to_engine` 注册 deepstack buffer
- [ ] 在 `send_embedding` 添加 deepstack 传输逻辑
- [ ] 仅在初始传输的第一个块传输 deepstack（类似 aux_datas）
- [ ] 断点续传时跳过 deepstack（已接收）

**Buffer 顺序**: 
`[embeddings, fill_ids, mrope_positions, aux_datas, deepstack_embeddings]`

### Phase 5: 测试验证
- [ ] 单元测试: buffer 分配/释放
- [ ] 单元测试: scatter/gather 操作
- [ ] 集成测试: 端到端 disaggregation
- [ ] 集成测试: 验证 deepstack 值正确性
- [ ] 集成测试: 断点续传测试
- [ ] 兼容性测试: 不支持 deepstack 的模型

---

## 📊 实现进度

| Phase | 任务 | 状态 |
|-------|------|------|
| 0 | 模型层重构 | ✅ 完成 |
| 1 | 缓冲区结构 | ⏳ 待实现 |
| 2 | Encode侧更新 | ⏳ 待实现 |
| 3 | Language侧更新 | 🟡 部分完成 (模型已支持) |
| 4 | 传输协议更新 | ⏳ 待实现 |
| 5 | 测试验证 | ⏳ 待实现 |

**总体进度**: 20% (1/5 完成)

---

## 📚 相关文档

- **实现计划**: `IMPLEMENTATION_PLAN_QWEN3_MOE_VL_DEEPSTACK.md`
- **重构详情**: `REFACTORING_SUMMARY.md`
- **模型文件**: 
  - `python/sglang/srt/models/qwen2_moe.py`
  - `python/sglang/srt/models/qwen3_moe.py`
  - `python/sglang/srt/models/qwen3_vl_moe.py`
- **Disaggregation文件**:
  - `python/sglang/srt/disaggregation/utils.py`
  - `python/sglang/srt/disaggregation/multimodal_embedding.py`
  - `python/sglang/srt/disaggregation/multimodal_language.py`
  - `python/sglang/srt/disaggregation/mooncake/conn_multimodal.py`

---

## 🔄 下一步行动

建议按以下顺序实施剩余阶段：

1. **Phase 1**: 扩展 `MultimodalDataBuffers` (最基础，其他阶段依赖此)
2. **Phase 4**: 更新传输协议 (定义如何传输)
3. **Phase 2**: Encode侧实现 (数据发送方)
4. **Phase 3**: Language侧实现 (数据接收方，模型部分已完成)
5. **Phase 5**: 端到端测试

---

## ✅ Phase 0 检查清单

- [x] `Qwen2MoeModel.__init__` 添加 `self.hidden_size`
- [x] `Qwen2MoeModel.forward()` 添加 `input_deepstack_embeds` 参数
- [x] `Qwen2MoeModel.forward()` 在前3层处理 deepstack
- [x] `Qwen2MoeForCausalLM.forward()` 添加 `input_deepstack_embeds` 参数
- [x] `Qwen3MoeForCausalLM.forward()` 添加 `input_deepstack_embeds` 参数
- [x] Git diff 验证修改正确
- [x] Linter 检查通过
- [x] 文档更新完成
