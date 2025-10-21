# 多模态Embedding分批传输功能

## 🎯 功能说明

支持多模态Embedding数据分批传输，解决实际数据长度超过默认buffer时的传输问题。

---

## 🚀 快速开始

### 配置

```bash
# 设置环境变量
export SGLANG_MULTIMODAL_BLOCK_SIZE=128          # Block大小
export SGLANG_DEFAULT_MULTIMODAL_BLOCKS=8        # Language侧默认申请8个blocks
export SGLANG_EMBEDDING_CACHE_BUFFER_SIZE=64     # Buffer总数量
```

### 启动

```bash
# Embedding侧
python -m sglang.launch_server \
    --model-path /path/to/model \
    --disaggregation-mode encode \
    --disaggregation-bootstrap-port 8001

# Language侧
python -m sglang.launch_server \
    --model-path /path/to/model \
    --disaggregation-mode language \
    --disaggregation-bootstrap-addr localhost:8001
```

---

## 📊 工作原理

### 核心机制

**问题**：Language侧不知道实际长度 → 只能预分配默认buffer

**解决**：两阶段传输 + Resume机制

```
实际长度2000 > 默认1024 时：

第1次传输：发送1024 + aux_data[总长度=2000]
         ↓
Language: 读取总长度，判断需要resume
         ↓
第2次传输：发送剩余976 tokens
         ↓
拼接完成：1024 + 976 = 2000 ✓
```

### 核心组件

1. **ReqToMetadataBlockAllocator** - Block-based分配器
   - Language侧：`alloc_default()` 分配固定8个blocks
   - Embedding侧：`alloc(num_tokens)` 按实际长度分配
   - 保证连续分配：blocks = [start, start+1, ..., start+num-1]

2. **MultimodalDataBuffers** - Buffer管理
   - 连续内存，按block逻辑管理
   - 支持offset和max_tokens参数

3. **Resume机制** - 分批传输
   - `resume_transfer(allocation, sent_tokens)` - 恢复传输
   - `buffered_chunks` - 缓存第一批数据
   - `transferred_tokens` - 已传输token数

---

## 📝 代码示例

### Language侧

```python
# 1. 首次分配
allocation = allocator.alloc_default(req_id=req.rid)
receiver.init(allocation)

# 2. 检查是否需要resume
if total_length > default_buffer_tokens:
    # 保存第一批
    buffered_chunks = save_first_batch(...)
    transferred_tokens = default_buffer_tokens
    
    # 重新分配
    allocator.free(allocation, req_id)
    new_allocation = allocator.alloc(num_tokens=remaining, req_id)
    
    # Resume
    receiver.resume_transfer(new_allocation, sent_tokens=transferred_tokens)

# 3. 拼接数据
if transferred_tokens > 0:
    full_embeddings = torch.cat([buffered_chunks["embeddings"], new_embeddings])
```

### Embedding侧

```python
# 1. 按实际长度分配
actual_length = req.embedding.shape[0]  # 2000
allocation = allocator.alloc(num_tokens=actual_length, req_id=req.rid)

# 2. 发送数据
if sent_tokens == 0:
    # 首次：限制为1024
    is_last = actual_length <= 1024
    chunk_info = buffers.get_buf_chunk_info(allocation, 0, max_tokens=1024)
else:
    # Resume：发送剩余
    is_last = True
    chunk_info = buffers.get_buf_chunk_info(allocation, sent_tokens)
```

---

## 📚 详细文档

- `MULTIMODAL_CACHE_DESIGN.md` - 完整设计文档
- `DESIGN_FIX_SUMMARY.md` - 设计修正说明
- `FINAL_IMPLEMENTATION_SUMMARY.md` - 实现总结

---

## ✅ 状态

- ✅ 代码实现完成
- ✅ Linter检查通过（0错误）
- ✅ 设计问题已修正
- ✅ 命名已专业化
- ⚠️ 集成测试待执行

---

## 📞 问题排查

### 日志关键词

```bash
# 监控Resume相关日志
tail -f logs/*.log | grep -E "resume|transferred_tokens"

# 查看分配信息
tail -f logs/*.log | grep -E "alloc_default|start_block"
```

### 常见问题

**Q: Language侧频繁等待buffer？**
```bash
# 增加默认block数量
export SGLANG_DEFAULT_MULTIMODAL_BLOCKS=16
```

**Q: Resume比例过高？**
```bash
# 增加默认block数量，减少resume次数
export SGLANG_DEFAULT_MULTIMODAL_BLOCKS=16
```

---

**实现完成**: 2025-10-20  
**版本**: v5.0-final  
**状态**: ✅ Ready for Testing
