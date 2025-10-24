# 🏗️ DeepStack 架构重构 - 更清晰的设计

## 🎯 重构目标

将 deepstack 支持从通用基类移到 VL 模型专用类，使架构更加清晰易懂。

## ❌ 旧架构问题

### Before (有问题的设计)
```
Qwen2MoeModel (基类)
  ├─ ✅ hidden_size
  ├─ ❌ input_deepstack_embeds parameter  ← 所有模型都有,但只有VL需要
  └─ ❌ deepstack processing logic        ← 在基类中,混淆语义

Qwen3MoeForCausalLM (纯文本模型)
  └─ ❌ 继承了 deepstack 参数             ← 不需要但被迫接受
```

**问题**:
1. 纯文本模型 (Qwen2Moe, Qwen3Moe) 不需要 deepstack
2. 放在基类让人困惑：这是通用功能还是 VL 特有？
3. 算法层面不清晰：为什么文本模型有 VL 功能？

## ✅ 新架构设计

### After (清晰的设计)
```
Qwen2MoeModel (基类 - 纯文本)
  ├─ ✅ 只包含通用功能
  └─ ❌ 没有 deepstack (符合语义)

Qwen3MoeModel (继承 Qwen2MoeModel)
  ├─ ✅ 纯文本 MoE 模型
  └─ ❌ 没有 deepstack

Qwen3MoeForCausalLM (纯文本 CausalLM)
  └─ model: Qwen3MoeModel
  └─ ❌ 没有 deepstack

┌─────────────────────────────────────┐
│  VL-Specific Models with DeepStack  │
└─────────────────────────────────────┘

Qwen3MoeLLMModel (VL专用)
  ├─ extends: Qwen3MoeModel
  ├─ ✅ input_deepstack_embeds parameter
  ├─ ✅ deepstack processing in forward()
  └─ ✅ 明确标注为 VL-specific

Qwen3VLMoeForConditionalGeneration (VL模型)
  ├─ visual: Qwen3_VisionTransformer
  └─ model: Qwen3MoeLLMModel ✅ (使用VL专用model)

┌────────────────────────────────────────────┐
│  Disaggregation Language Side (VL分离场景) │
└────────────────────────────────────────────┘

Qwen3MoeModelWithDeepStack (专用)
  ├─ extends: Qwen3MoeModel
  ├─ ✅ input_deepstack_embeds parameter
  ├─ ✅ deepstack processing
  └─ 📝 Docstring: "For disaggregation language side"

Qwen3MoeForCausalLMWithDeepStack (专用)
  ├─ model: Qwen3MoeModelWithDeepStack
  ├─ ✅ forward(..., input_deepstack_embeds=...)
  └─ 📝 Docstring: "For disaggregation with deepstack"
```

## 📊 修改对比

### 文件1: `qwen2_moe.py` (基类 - 移除 deepstack)

#### Before
```python
class Qwen2MoeModel(nn.Module):
    def __init__(self, ...):
        self.hidden_size = config.hidden_size  # ← 为 deepstack 添加的
    
    def forward(self, ..., input_deepstack_embeds=None):  # ← 不应该在基类
        ...
        for i in range(self.start_layer, self.end_layer):
            hidden_states, residual = layer(...)
            
            # ❌ VL-specific logic in base class
            if input_deepstack_embeds is not None and i in range(3):
                hidden_states.add_(input_deepstack_embeds[...])
```

#### After
```python
class Qwen2MoeModel(nn.Module):
    def __init__(self, ...):
        # ✅ 不再有 self.hidden_size (不需要)
    
    def forward(self, ..., input_embeds=None):  # ← 干净的接口
        ...
        for i in range(self.start_layer, self.end_layer):
            hidden_states, residual = layer(...)
            # ✅ 没有 deepstack 逻辑
```

### 文件2: `qwen3_vl_moe.py` (VL模型 - 恢复专用类)

#### Before (之前简化掉的)
```python
class Qwen3VLMoeForConditionalGeneration(...):
    self.model = Qwen3MoeModel(...)  # ← 使用基类，但需要 deepstack
```

#### After (恢复但更清晰)
```python
class Qwen3MoeLLMModel(Qwen3MoeModel):
    """Qwen3 MoE model with DeepStack support for VL models.
    
    This class extends Qwen3MoeModel to add deepstack embedding support,
    which is specific to vision-language models.
    """
    
    def __init__(self, *, config, ...):
        super().__init__(config=config, ...)
        self.hidden_size = config.hidden_size  # ✅ 只在需要的地方
    
    def forward(self, ..., input_deepstack_embeds=None):
        ...
        for layer_idx in range(...):
            hidden_states, residual = layer(...)
            
            # ✅ VL-specific: deepstack for first 3 layers
            if input_deepstack_embeds is not None and layer_idx in range(3):
                sep = self.hidden_size * layer_idx
                hidden_states.add_(
                    input_deepstack_embeds[:, sep : sep + self.hidden_size]
                )

class Qwen3VLMoeForConditionalGeneration(...):
    self.model = Qwen3MoeLLMModel(...)  # ✅ 使用 VL 专用 model
```

### 文件3: `qwen3_moe.py` (纯文本 + Disaggregation专用)

#### 新增类1: Qwen3MoeModelWithDeepStack
```python
class Qwen3MoeModelWithDeepStack(Qwen3MoeModel):
    """Qwen3 MoE model with DeepStack support for disaggregation language side.
    
    This is a specialized variant used in disaggregation scenarios where the
    language side receives deepstack embeddings from the encode side.
    
    Note: This is only used in disaggregation mode. Regular inference should 
    use Qwen3MoeModel.
    """
    
    def __init__(self, config, ...):
        super().__init__(config=config, ...)
        self.hidden_size = config.hidden_size
    
    def forward(self, ..., input_deepstack_embeds=None):
        ...
        for i in range(self.start_layer, self.end_layer):
            hidden_states, residual = layer(...)
            
            # ✅ For VL disaggregation: add deepstack to first 3 layers
            if input_deepstack_embeds is not None and i in range(3):
                sep = self.hidden_size * i
                hidden_states.add_(
                    input_deepstack_embeds[:, sep : sep + self.hidden_size]
                )
```

#### 新增类2: Qwen3MoeForCausalLMWithDeepStack
```python
class Qwen3MoeForCausalLMWithDeepStack(nn.Module):
    """Qwen3 MoE for Causal LM with DeepStack support for disaggregation.
    
    This variant is used in disaggregation language side when receiving
    deepstack embeddings from the encode side.
    
    Usage:
        # In disaggregation language mode
        model = Qwen3MoeForCausalLMWithDeepStack(config)
        output = model.forward(..., input_deepstack_embeds=deepstack_data)
    """
    
    def __init__(self, config, ...):
        super().__init__()
        self.model = Qwen3MoeModelWithDeepStack(...)  # ✅ 使用带 deepstack 的版本
        ...
    
    def forward(self, ..., input_deepstack_embeds=None):
        hidden_states = self.model(
            ...,
            input_deepstack_embeds=input_deepstack_embeds,  # ✅ 传递 deepstack
        )
        ...
```

#### 保持不变: Qwen3MoeForCausalLM
```python
class Qwen3MoeForCausalLM(nn.Module):
    """Regular Qwen3 MoE for Causal LM (no deepstack)."""
    
    def __init__(self, config, ...):
        self.model = Qwen3MoeModel(...)  # ✅ 使用纯文本 model
    
    def forward(self, ..., input_embeds=None):  # ✅ 没有 deepstack 参数
        hidden_states = self.model(...)
        ...
```

## 🎓 使用场景

### 1. 纯文本推理 (Qwen3-MoE)
```python
model = Qwen3MoeForCausalLM(config)
output = model.forward(input_ids, positions, forward_batch)
# ✅ 干净的接口，没有不需要的 deepstack 参数
```

### 2. VL 推理 (Qwen3-VL-MoE)
```python
model = Qwen3VLMoeForConditionalGeneration(config)
# model.model is Qwen3MoeLLMModel (with deepstack support)

# Encode side: extract deepstack
output = model.forward(input_ids, positions, forward_batch)
# ✅ 内部自动处理 deepstack
```

### 3. Disaggregation - Encode Side (VL)
```python
# Uses regular VL model
model = Qwen3VLMoeForConditionalGeneration(config)

# Extract deepstack
embedding, deepstack = model.separate_deepstack_embeds(full_embedding)

# Transfer both to language side
transfer(embedding, deepstack)
```

### 4. Disaggregation - Language Side (接收 deepstack)
```python
# ✅ 使用专门的 deepstack 版本
model = Qwen3MoeForCausalLMWithDeepStack(config)

# Receive from encode side
embedding_data = receive_embedding()
deepstack_data = receive_deepstack()

# Forward with deepstack
output = model.forward(
    input_ids,
    positions,
    forward_batch,
    input_embeds=embedding_data,
    input_deepstack_embeds=deepstack_data,  # ✅ 支持 deepstack
)
```

## ✅ 优势

### 1. **语义清晰**
- ✅ 纯文本模型：没有 VL 特有功能
- ✅ VL 模型：明确标注 deepstack 支持
- ✅ Disaggregation：专门的类，明确用途

### 2. **算法理解**
- ✅ 看到 `Qwen3MoeModel` → 纯文本，无 deepstack
- ✅ 看到 `Qwen3MoeLLMModel` → VL 模型，有 deepstack
- ✅ 看到 `Qwen3MoeModelWithDeepStack` → Disaggregation 专用

### 3. **代码组织**
```
qwen2_moe.py (基础层)
  └─ Qwen2MoeModel: 纯文本基类

qwen3_moe.py (纯文本 + Disagg)
  ├─ Qwen3MoeModel: 纯文本
  ├─ Qwen3MoeForCausalLM: 纯文本推理
  ├─ Qwen3MoeModelWithDeepStack: Disagg专用 ✅
  └─ Qwen3MoeForCausalLMWithDeepStack: Disagg推理 ✅

qwen3_vl_moe.py (VL层)
  ├─ Qwen3MoeLLMModel: VL专用model ✅
  └─ Qwen3VLMoeForConditionalGeneration: VL推理
```

### 4. **维护性**
- ✅ deepstack 逻辑集中在 VL 相关类
- ✅ 修改 deepstack 不影响纯文本模型
- ✅ 新增 VL 功能时清楚应该改哪里

### 5. **向后兼容**
- ✅ 纯文本模型接口没变
- ✅ VL 模型功能没变
- ✅ 只是重新组织了代码结构

## 📊 代码变化统计

| 文件 | 变化 | 说明 |
|------|------|------|
| qwen2_moe.py | -11 行 | 移除基类中的 deepstack |
| qwen3_moe.py | +225 行 | 添加 Disagg 专用类 |
| qwen3_vl_moe.py | +87 行 | 恢复 VL 专用 LLMModel |
| **总计** | +301 行 | 净增加（更清晰的设计） |

## 🎯 关键改进

1. **基类保持纯净** ✅
   - `Qwen2MoeModel` 只有通用功能
   - 不包含 VL 特有的 deepstack

2. **VL 功能明确标注** ✅
   - `Qwen3MoeLLMModel` 有清晰的 docstring
   - 说明这是 "for VL models"

3. **Disaggregation 专用类** ✅
   - `Qwen3MoeModelWithDeepStack` 明确用途
   - `Qwen3MoeForCausalLMWithDeepStack` 完整实现
   - Docstring 说明 "for disaggregation"

4. **灵活的使用方式** ✅
   - 纯文本：用 `Qwen3MoeForCausalLM`
   - VL：用 `Qwen3VLMoeForConditionalGeneration`
   - Disagg Language：用 `Qwen3MoeForCausalLMWithDeepStack`

## 📝 文档说明

每个类都添加了清晰的 docstring：

```python
class Qwen3MoeLLMModel(Qwen3MoeModel):
    """Qwen3 MoE model with DeepStack support for VL models.
    
    This class extends Qwen3MoeModel to add deepstack embedding support,
    which is specific to vision-language models. The deepstack embeddings
    are added to the first 3 layers during forward pass.
    """

class Qwen3MoeModelWithDeepStack(Qwen3MoeModel):
    """Qwen3 MoE model with DeepStack support for disaggregation language side.
    
    This is a specialized variant used in disaggregation scenarios where the
    language side receives deepstack embeddings from the encode side.
    
    Note: This is only used in disaggregation mode. Regular inference should 
    use Qwen3MoeModel.
    """
```

## 🎉 总结

这次重构让架构更加：
- ✅ **清晰**: 一眼看出哪个类有 deepstack
- ✅ **合理**: VL 功能在 VL 类中
- ✅ **灵活**: 根据场景选择合适的类
- ✅ **易懂**: 算法层面更容易理解

**核心原则**: DeepStack 是 VL 特有功能，应该只出现在 VL 相关的类中，而不是通用基类。

---

**重构完成**: 2025-10-24  
**状态**: ✅ 代码重构完成，0 linter errors
