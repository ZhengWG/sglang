# 多模态Embedding分批传输配置指南

## 环境变量配置

### 必需配置

```bash
# 启用disaggregation模式
export SGLANG_DISAGGREGATION_MODE=encode  # 或 language

# Bootstrap服务器地址
export SGLANG_DISAGGREGATION_BOOTSTRAP_ADDR=your_bootstrap_server:port
```

### Block-based分配器配置（推荐）

```bash
# 启用block-based allocator（默认：true）
export SGLANG_USE_BLOCK_ALLOCATOR=true

# Block大小，单位：tokens（默认：128）
export SGLANG_MULTIMODAL_BLOCK_SIZE=128

# 默认buffer大小，单位：tokens（默认：1024）
export SGLANG_DEFAULT_MULTIMODAL_BUFFER_TOKENS=1024

# Embedding cache buffer数量（默认：64）
export SGLANG_EMBEDDING_CACHE_BUFFER_SIZE=64
```

### 高级配置（可选）

```bash
# 传输线程池大小
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=12

# 传输队列大小
export SGLANG_DISAGGREGATION_QUEUE_SIZE=4

# Bootstrap超时时间（秒）
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=30

# 心跳检测间隔（秒）
export SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL=5.0

# 心跳失败最大次数
export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=2
```

## 配置示例

### 场景1：小图片处理（平均500-1000 tokens）

```bash
# 使用较小的default buffer
export SGLANG_USE_BLOCK_ALLOCATOR=true
export SGLANG_MULTIMODAL_BLOCK_SIZE=128
export SGLANG_DEFAULT_MULTIMODAL_BUFFER_TOKENS=512
export SGLANG_EMBEDDING_CACHE_BUFFER_SIZE=128
```

**预期行为**：
- 大部分图片一次传输完成（≤512 tokens）
- 少数大图片需要continuation（>512 tokens）
- 内存利用率高，支持更多并发请求

### 场景2：中等图片处理（平均1000-2000 tokens）

```bash
# 使用标准default buffer
export SGLANG_USE_BLOCK_ALLOCATOR=true
export SGLANG_MULTIMODAL_BLOCK_SIZE=128
export SGLANG_DEFAULT_MULTIMODAL_BUFFER_TOKENS=1024
export SGLANG_EMBEDDING_CACHE_BUFFER_SIZE=64
```

**预期行为**：
- 中等图片一次传输完成（≤1024 tokens）
- 大图片/视频需要continuation（>1024 tokens）
- 平衡内存和性能

### 场景3：大图片/视频处理（平均2000-4000 tokens）

```bash
# 使用较大的default buffer
export SGLANG_USE_BLOCK_ALLOCATOR=true
export SGLANG_MULTIMODAL_BLOCK_SIZE=256
export SGLANG_DEFAULT_MULTIMODAL_BUFFER_TOKENS=2048
export SGLANG_EMBEDDING_CACHE_BUFFER_SIZE=32
```

**预期行为**：
- 大部分数据一次传输完成（≤2048 tokens）
- 极少需要continuation（>2048 tokens）
- 更高的内存占用，更少的continuation开销

### 场景4：禁用block-based allocator（向后兼容）

```bash
# 使用传统的index-based allocator
export SGLANG_USE_BLOCK_ALLOCATOR=false
export SGLANG_EMBEDDING_CACHE_BUFFER_SIZE=64
```

**预期行为**：
- 使用固定大小buffer（max_prefill_tokens）
- 不支持continuation，数据必须一次传输完成
- 与旧版本行为一致

## 启动命令示例

### Embedding侧（ENCODE模式）

```bash
python -m sglang.launch_server \
    --model-path /path/to/embedding/model \
    --disaggregation-mode encode \
    --disaggregation-bootstrap-port 8001 \
    --tp-size 4 \
    --dp-size 2 \
    --port 8000
```

### Language侧（LANGUAGE模式）

```bash
python -m sglang.launch_server \
    --model-path /path/to/language/model \
    --disaggregation-mode language \
    --disaggregation-bootstrap-addr embedding_host:8001 \
    --tp-size 8 \
    --dp-size 1 \
    --port 8100
```

## 性能调优建议

### 1. Block Size选择

| Block Size | 优势 | 劣势 | 适用场景 |
|-----------|------|------|---------|
| 64 tokens | 更少内存碎片 | 更多分配操作 | 变长数据多 |
| 128 tokens | **平衡** | - | **推荐默认** |
| 256 tokens | 更少分配操作 | 可能浪费内存 | 大数据为主 |
| 512 tokens | 最少分配操作 | 显著内存浪费 | 不推荐 |

### 2. Default Buffer Tokens选择

**计算公式**：
```
default_buffer_tokens = P75(actual_embedding_length)
```

其中 P75 表示75%分位数。即75%的请求可以一次传输完成。

**示例**：
- 如果75%的请求长度 ≤ 800 tokens，设置 `DEFAULT_MULTIMODAL_BUFFER_TOKENS=1024`
- 如果75%的请求长度 ≤ 1500 tokens，设置 `DEFAULT_MULTIMODAL_BUFFER_TOKENS=2048`

### 3. Buffer Size选择

**计算公式**：
```
buffer_size = concurrent_requests * safety_factor
```

**示例**：
- 预期并发20个请求，设置 `EMBEDDING_CACHE_BUFFER_SIZE=40`（safety_factor=2）
- 预期并发50个请求，设置 `EMBEDDING_CACHE_BUFFER_SIZE=100`（safety_factor=2）

### 4. 监控指标

建议监控以下指标优化配置：

```python
# 伪代码
continuation_rate = num_continuation_requests / total_requests
avg_embedding_length = sum(embedding_lengths) / total_requests
buffer_wait_time = avg_time_waiting_for_buffer
```

**优化目标**：
- `continuation_rate < 25%`：大部分请求一次传输完成
- `buffer_wait_time < 10ms`：buffer充足，无显著等待
- `memory_usage < 80%`：有足够的buffer headroom

## 故障排查

### 问题1：Continuation率过高（>50%）

**原因**：`DEFAULT_MULTIMODAL_BUFFER_TOKENS` 设置过小

**解决**：
```bash
# 增加default buffer大小
export SGLANG_DEFAULT_MULTIMODAL_BUFFER_TOKENS=2048
```

### 问题2：内存占用过高

**原因**：`EMBEDDING_CACHE_BUFFER_SIZE` 或 `DEFAULT_MULTIMODAL_BUFFER_TOKENS` 过大

**解决**：
```bash
# 减少buffer数量或大小
export SGLANG_EMBEDDING_CACHE_BUFFER_SIZE=32
export SGLANG_DEFAULT_MULTIMODAL_BUFFER_TOKENS=1024
```

### 问题3：频繁等待buffer

**日志示例**：
```
WARNING: No buffer for continuation: 976 tokens needed
```

**原因**：Buffer数量不足

**解决**：
```bash
# 增加buffer数量
export SGLANG_EMBEDDING_CACHE_BUFFER_SIZE=128
```

### 问题4：传输失败

**日志示例**：
```
ERROR: MultimodalEmbedding transfer failed for request
```

**排查步骤**：
1. 检查网络连接：`ping embedding_host`
2. 检查Bootstrap服务：`curl http://bootstrap_host:port/health`
3. 检查日志中的详细错误信息
4. 验证配置一致性（Embedding和Language侧）

## 最佳实践

1. **测试环境先验证**
   ```bash
   # 使用小配置测试
   export SGLANG_EMBEDDING_CACHE_BUFFER_SIZE=8
   export SGLANG_DEFAULT_MULTIMODAL_BUFFER_TOKENS=512
   ```

2. **生产环境渐进式调优**
   - 从保守配置开始（大buffer size，大default tokens）
   - 基于监控指标逐步调整
   - 每次调整一个参数

3. **预留headroom**
   ```bash
   # 实际需要40个buffer，配置80个（2x headroom）
   export SGLANG_EMBEDDING_CACHE_BUFFER_SIZE=80
   ```

4. **定期审查配置**
   - 每月review监控指标
   - 根据业务变化调整配置
   - 保存配置变更历史

## 参考资料

- [实现文档](../MULTIMODAL_EMBEDDING_CACHE_IMPLEMENTATION.md)
- [Disaggregation架构文档](../docs/disaggregation.md)
- [环境变量完整列表](../docs/env_vars.md)

---

**最后更新**：2025-10-20  
**适用版本**：SGLang v0.3.0+
