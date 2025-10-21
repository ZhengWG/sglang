# 多模态Embedding分批传输设计文档

## 🎯 核心问题

**场景**：
- Embedding侧处理多模态数据后知道实际长度（如2000 tokens）
- Language侧请求时不知道实际长度，只能预分配默认buffer
- 当实际长度 > 默认buffer时，需要分批传输（Resume机制）

---

## ✅ 核心设计

### 1. Block-based 内存管理

```python
总容量 = 8192 tokens
Block大小 = 128 tokens
Block数量 = 64 blocks

连续分配：blocks总是从start_block开始连续分配
例如：start_block=5, num_blocks=3 -> 使用blocks [5, 6, 7]
```

### 2. 数据结构

```python
@dataclass
class MetadataAllocation:
    start_block: int  # 起始block索引
    num_blocks: int   # 连续的block数量
    num_tokens: int   # 实际需要的token数
```

### 3. 分配策略

**Language侧**（不知道实际长度）：
```python
allocation = allocator.alloc_default(req_id)
# 分配固定数量的blocks（如8个）= 1024 tokens
```

**Embedding侧**（知道实际长度）：
```python
actual_length = 2000
allocation = allocator.alloc(num_tokens=2000, req_id)
# 按实际长度分配：2000/128 = 16 blocks
```

**Resume阶段**（Language侧知道剩余长度）：
```python
remaining = 2000 - 1024  # 976 tokens
allocation = allocator.alloc(num_tokens=976, req_id)
# 按剩余长度分配：976/128 = 8 blocks
```

---

## 🔄 完整传输流程

```
场景：实际长度2000 tokens，默认8 blocks (1024 tokens)

第一次传输：
-----------
Language: alloc_default() -> 8 blocks (1024 tokens)
         ↓ init(allocation)
Embedding: alloc(2000) -> 16 blocks
          发送前1024 tokens + aux_data[0]=2000
         ↓ status: Transferring (is_last=False)
Language: 读取 aux_data[0]=2000
          判断：2000 > 1024，需要resume
          缓存前1024 tokens到 buffered_chunks
          free(8 blocks)

Resume传输：
-----------
Language: alloc(976) -> 8 blocks
         ↓ resume_transfer(allocation, sent_tokens=1024)
Embedding: 更新 transfer_info.sent_tokens=1024
          发送剩余976 tokens（从offset=1024开始）
         ↓ status: Success (is_last=True)
Language: 拼接数据：[buffered_chunks(1024) + new(976)] = 2000 ✓
```

---

## 📊 关键方法

### ReqToMetadataBlockAllocator

```python
# 初始化
allocator = ReqToMetadataBlockAllocator(
    total_tokens=8192,  # 总容量
    block_size=128      # 每block大小
)
# 自动计算：num_blocks = 64
# 从环境变量读取：default_num_blocks = 8

# Language侧：按默认block数分配
allocation = allocator.alloc_default(req_id=1)
# -> start_block=0, num_blocks=8, num_tokens=1024

# Embedding侧：按实际tokens分配
allocation = allocator.alloc(num_tokens=2000, req_id=2)
# -> start_block=8, num_blocks=16, num_tokens=2000

# 释放
allocator.free(allocation, req_id=1)
# 归还 blocks [start_block, start_block+1, ..., start_block+num_blocks-1]
```

### MultimodalDataBuffers

```python
# 计算chunk信息
chunk_info = buffers.get_buf_chunk_info(
    allocation,
    offset_tokens=0,      # 从第0个token开始
    max_tokens=1024       # 最多传输1024 tokens
)
# 返回：[(offset_bytes, size_bytes), ...] for [embeddings, fill_ids, mrope, aux]

# 读取数据
embeddings, fill_ids, mrope, aux = buffers.get_buf(allocation)

# 写入数据
buffers.set_buf(req, allocation)
```

---

## 🔧 配置参数

```bash
# Block大小（tokens per block）
export SGLANG_MULTIMODAL_BLOCK_SIZE=128

# Language侧默认分配的block数量（新增）
export SGLANG_DEFAULT_MULTIMODAL_BLOCKS=8

# Buffer总数量
export SGLANG_EMBEDDING_CACHE_BUFFER_SIZE=64
```

**计算关系**：
```
总容量 = EMBEDDING_CACHE_BUFFER_SIZE * max_req_len
Block数量 = 总容量 / BLOCK_SIZE
默认buffer大小 = DEFAULT_MULTIMODAL_BLOCKS * BLOCK_SIZE
```

**示例**：
```
buffer_size=64, max_req_len=8192
-> 总容量 = 524288 tokens
-> Block数量 = 4096 blocks

default_blocks=8, block_size=128
-> 默认buffer = 1024 tokens
```

---

## 🎯 核心优势

1. **连续分配** ✅
   - Blocks总是连续的：[start, start+1, ..., start+num-1]
   - 简化计算，避免min()错误

2. **语义清晰** ✅
   - `alloc_default()` 明确用于Language侧
   - `alloc(num_tokens)` 明确用于Embedding侧

3. **配置灵活** ✅
   - 通过 `DEFAULT_MULTIMODAL_BLOCKS` 控制Language侧buffer
   - 通过 `BLOCK_SIZE` 控制粒度

4. **性能优化** ✅
   - O(1) 计算 start_token
   - 减少内存碎片

---

## 📝 代码示例

### Language 侧使用

```python
# 首次分配
allocation = allocator.alloc_default(req_id=req.rid)
receiver.init(allocation)

# Resume分配  
remaining = total_length - transferred_tokens
new_allocation = allocator.alloc(num_tokens=remaining, req_id=req.rid)
receiver.resume_transfer(new_allocation, sent_tokens=transferred_tokens)
```

### Embedding 侧使用

```python
# 按实际长度分配
actual_length = req.embedding.shape[0]
allocation = allocator.alloc(num_tokens=actual_length, req_id=req.rid)
buffers.set_buf(req, allocation)
```

---

**设计修正完成！代码更简洁、更准确、更高效！** ✅

---

**最后更新**: 2025-10-20  
**设计版本**: v5.0-final
