# 多模态Embedding分批传输功能实现总结

## 概述

实现了支持多模态Embedding数据分批传输的功能，解决当实际数据长度超过默认buffer大小时的传输问题。

## 核心设计

### 问题场景
- **Embedding侧**：处理多模态数据后知道实际长度（如2000 tokens）
- **Language侧**：请求时不知道实际长度，只能申请默认buffer（如1024 tokens）
- **需求**：当实际长度 > 默认buffer时，需要分批传输

### 解决方案

**方案名称**：Block-based 分配 + Continuation 传输

**关键特性**：
1. ✅ **Block-based内存分配**：按128 tokens/block的粒度管理buffer
2. ✅ **动态大小适配**：Language侧申请默认大小，Embedding侧按实际大小分配
3. ✅ **Continuation机制**：第一批传输完成后，Language侧读取总长度，请求发送剩余数据
4. ✅ **状态复用**：复用`Transferring`状态，无需新增状态枚举
5. ✅ **Room不变**：整个传输过程保持同一个room ID
6. ✅ **向后兼容**：支持新旧协议共存

## 文件修改清单

### 1. 基础数据结构 (`python/sglang/srt/disaggregation/utils.py`)

**新增**：
- `MetadataAllocation`: 记录block分配信息
- `ReqToMetadataBlockAllocator`: Block-based分配器

**修改**：
- `MultimodalDataBuffers`: 
  - 支持block-based和index-based两种模式
  - 新增`default_buffer_tokens`配置
  - 新增`get_buf_chunk_info()`支持offset和max_tokens参数

**配置参数**：
```python
SGLANG_MULTIMODAL_BLOCK_SIZE=128  # 每个block的token数
SGLANG_DEFAULT_MULTIMODAL_BUFFER_TOKENS=1024  # 默认申请大小
```

### 2. 传输协议 (`python/sglang/srt/disaggregation/mooncake/conn_multimodal.py`)

**修改数据结构**：
```python
@dataclass
class TransferEmbeddingInfo:
    # ... existing fields ...
    sent_tokens: int = 0  # 新增：已发送的token数量

@dataclass
class TransferEmbeddingChunk:
    # ... existing fields ...
    sent_tokens: int = 0  # 新增：已发送的token数量
```

**修改方法**：
- `send_embedding()`: 增加`sent_tokens`参数
- `embedding_thread()`: 处理continuation请求（`sent_tokens > 0`）
- `add_transfer_request()`: 从transfer_info获取sent_tokens并传递
- `MooncakeEmbeddingReceiver.init_continuation()`: 新增方法，发送continuation请求

**ZMQ消息格式**：
```
首次请求:  [room, endpoint, port, session_id, block_idx, required_num, 0]
Continuation: [room, endpoint, port, session_id, new_block_idx, required_num, 1024]
                                                                           ^^^^^^
                                                                        sent_tokens
```

### 3. Embedding侧 (`python/sglang/srt/disaggregation/multimodal_embedding.py`)

**关键修改**：

**Bootstrap阶段**：
- Block模式：延迟分配buffer（在知道实际长度后分配）
- Index模式：保持原有行为

**Process batch result**：
```python
# 按实际长度分配buffer
actual_length = req.embedding.shape[0]
allocation = allocator.alloc(num_tokens=actual_length, req_id=req.rid)
req.metadata_allocation = allocation
```

**send_embedding_chunk()**：
```python
if sent_tokens == 0:
    # 首次发送：限制为 min(actual_length, default_tokens)
    is_last = (actual_length <= default_tokens)
    chunk_info = get_buf_chunk_info(offset=0, max_tokens=default_tokens)
else:
    # Continuation发送：发送剩余所有数据
    is_last = True
    chunk_info = get_buf_chunk_info(offset=sent_tokens)
```

**inflight queue处理**：
- 在`Transferring`状态下等待continuation请求
- `Success`状态表示全部传输完成

### 4. Language侧 (`python/sglang/srt/disaggregation/multimodal_language.py`)

**数据结构扩展**：
```python
@dataclass
class MultimodalLanguageRequest:
    # ... existing fields ...
    current_allocation: Optional[MetadataAllocation] = None
    total_embedding_length: int = -1
    received_tokens: int = 0
    partial_data: Optional[dict] = None
    needs_continuation: bool = False
```

**pop_preallocated()**：
```python
# Block模式：按default_buffer_tokens分配
allocation = allocator.alloc_default(default_tokens=1024, req_id=req.rid)
receiver.init(allocation=allocation)
```

**pop_transferred() - 核心逻辑**：

**Transferring状态处理**：
```python
# 检查第一批数据是否到达
total_length = int(aux_datas[0])

if total_length > 0 and not received:
    if total_length > default_tokens:
        # 需要continuation
        # 1. 保存第一批数据到 partial_data
        # 2. 释放旧buffer
        # 3. 申请新buffer（remaining_tokens大小）
        # 4. 发送continuation请求
        receiver.init_continuation(allocation=new_allocation, sent_tokens=received_length)
```

**Success状态处理**：
```python
if received_tokens == 0:
    # 一次性传输完成
    req.input_embeds = embedding_data[:total_length, :]
else:
    # Continuation完成，拼接数据
    full_embeddings = torch.cat([partial_data["embeddings"], new_data])
    req.input_embeds = full_embeddings
```

**process_multimodal_language_queue()**：
```python
# 处理等待continuation buffer的请求
for language_req in queue:
    if language_req.needs_continuation and buffer_available:
        new_allocation = allocator.alloc(num_tokens=remaining)
        receiver.init_continuation(allocation, sent_tokens=received)
```

## 传输流程示例

### 场景：实际长度2000 tokens，默认buffer 1024 tokens

```
第一次传输：
-----------
Language:
  - 申请 1024 tokens (8 blocks)
  - 发送 init(allocation, sent_tokens=0)

Embedding:
  - 收到请求，处理多模态数据，得到实际长度 2000
  - 分配 2000 tokens (16 blocks)
  - 发送前 1024 tokens + aux_data[0]=2000
  - 状态: Transferring (is_last=False)

Language:
  - 接收 1024 tokens
  - 读取 aux_data[0] = 2000
  - 判断：2000 > 1024，需要continuation
  - 保存前1024 tokens到 partial_data
  - 释放8 blocks

第二次传输（Continuation）：
--------------------------
Language:
  - 申请新buffer：976 tokens (8 blocks)
  - 发送 init_continuation(new_allocation, sent_tokens=1024)

Embedding:
  - 收到continuation请求
  - 更新 transfer_info: sent_tokens=1024, dst_buffer=new_allocation
  - 发送剩余 976 tokens（从offset=1024开始）
  - 状态: Success (is_last=True)

Language:
  - 接收剩余 976 tokens
  - 拼接：[partial_data(1024) + new_data(976)] = 2000 tokens
  - 释放8 blocks
  - 完成 ✓
```

## 关键设计点

### 1. 为什么使用 `sent_tokens` 而不是 `continuation_offset`？
- ✅ 更直观：表示"已发送了多少个tokens"
- ✅ 自解释：看到变量名就知道含义
- ✅ 便于日志：`sent_tokens=1024` 比 `continuation_offset=1024` 更易读

### 2. 为什么复用 `Transferring` 状态？
- ✅ 简洁：无需新增 `PartialComplete` 状态
- ✅ 通用：`Transferring` 本身就表示"传输进行中"
- ✅ 兼容：不影响其他loop的状态判断

### 3. Block-based vs Index-based
| 特性 | Block-based | Index-based |
|------|-------------|-------------|
| 内存利用率 | 高（按需分配） | 低（固定大小） |
| 灵活性 | 高（动态大小） | 低（固定大小） |
| 实现复杂度 | 中 | 低 |
| 适用场景 | 变长数据 | 固定长度数据 |

## 配置建议

```python
# 根据典型多模态输入调整
DEFAULT_MULTIMODAL_BUFFER_TOKENS = 1024  # 默认值

# 建议值：
# - 小图片：512-1024 tokens
# - 中等图片：1024-2048 tokens
# - 大图片/视频：2048-4096 tokens

MULTIMODAL_BLOCK_SIZE = 128  # 推荐值，平衡内存碎片和分配效率

# 总容量 = size * max_prefill_tokens
# 例如：size=8, max_prefill_tokens=8192 -> 65536 tokens total
```

## 测试场景

建议测试以下场景：

1. ✅ **一次传输完成**：`actual_length <= default_buffer_tokens`
2. ✅ **两次传输完成**：`actual_length > default_buffer_tokens`
3. ⚠️ **Buffer不足等待**：多个请求同时需要continuation，buffer不足
4. ⚠️ **首次传输失败**：在第一批传输失败的错误处理
5. ⚠️ **Continuation传输失败**：第二次传输失败，partial_data清理
6. ⚠️ **并发请求**：多个请求同时进行，有的一次完成，有的需要continuation

## 向后兼容性

✅ 通过检测消息长度实现向后兼容：
```python
sent_tokens = int(msg[6].decode("ascii")) if len(msg) > 6 else 0
```

✅ 支持旧版本协议（6个字段）和新版本协议（7个字段）共存

## 已知限制

1. **Block-based模式需要显式启用**：通过 `use_block_allocator=True`
2. **假设blocks连续分配**：当前实现简化处理，假设分配的blocks是连续的
3. **仅支持两次传输**：首次 + continuation（实际场景足够）

## 未来优化方向

1. **动态block大小**：根据实际数据分布自适应调整block大小
2. **多次continuation**：支持超大数据的多次分批传输
3. **优先级调度**：continuation请求优先级高于新请求
4. **预分配优化**：预测常见长度，提前分配buffer

---

**实现完成时间**：2025-10-20  
**测试状态**：待测试  
**文档版本**：v1.0
