# Encode/Language 分离方案设计文档

## 1. 背景与动机

### 1.1 问题陈述

在多模态大模型（Vision-Language Models）的推理服务中，视觉编码（Visual Encoding）和语言生成（Language Generation）存在以下特点：

- **视觉编码**：计算密集，但只在prefill阶段执行一次，将图像/视频转换为embedding
- **语言生成**：需要持续的decode过程，但每个token的计算量相对较小

传统的EPD（Encode-Prefill-Decode）架构虽然分离了prefill和decode，但视觉编码仍然在prefill实例上执行，导致：
1. Prefill实例需要加载完整的视觉编码器模型（参数量大）
2. 视觉编码和文本prefill无法进一步解耦
3. 资源利用效率不够理想

### 1.2 设计目标

本方案旨在实现**Encode/Language分离**，将多模态处理解耦为：

- **Embedding实例（Encode）**：专门处理视觉编码，生成multimodal embeddings
- **Language实例**：接收embeddings，执行文本prefill和decode

**核心优势**：
1. 更细粒度的资源解耦
2. 可独立扩展视觉编码和语言生成能力
3. 支持Deepstack等高级特性
4. 为未来的优化提供架构基础

---

## 2. 整体架构

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Request                          │
│                    (Text + Image/Video)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Embedding Instance (Encode)                  │
├─────────────────────────────────────────────────────────────────┤
│  1. Receive request with multimodal inputs                      │
│  2. Visual Encoding (images/videos → embeddings)                │
│  3. Text Embedding (tokens → embeddings)                        │
│  4. Bootstrap handshake with Language instance                  │
│  5. Transfer embeddings via RDMA/Mooncake                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Embedding Transfer
                             │ (Block-based, supports resume)
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Language Instance (Prefill+Decode)            │
├─────────────────────────────────────────────────────────────────┤
│  1. Bootstrap handshake with Embedding instance                 │
│  2. Allocate buffer for incoming embeddings                     │
│  3. Receive embeddings via RDMA/Mooncake                        │
│  4. Execute Prefill forward with received embeddings            │
│  5. Execute Decode with KV cache                                │
│  6. Return generated text to client                             │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

#### 2.2.1 Embedding Instance
- **文件**：`python/sglang/srt/disaggregation/multimodal_embedding.py`
- **队列管理**：
  - `MultimodalEmbeddingBootstrapQueue`：处理bootstrap握手
  - `disagg_embedding_inflight_queue`：管理正在传输的请求
- **关键流程**：
  ```
  recv_requests → bootstrap → prefill → send_embedding → poll_transfer
  ```

#### 2.2.2 Language Instance
- **文件**：`python/sglang/srt/disaggregation/multimodal_language.py`
- **队列管理**：
  - `MultimodalLanguagePreallocQueue`：预分配buffer，完成bootstrap握手
  - `MultimodalLanguageTransferQueue`：等待embedding传输完成
- **关键流程**：
  ```
  handshake → preallocate_buffer → receive_embedding → prefill → decode
  ```

#### 2.2.3 Transfer Layer
- **文件**：`python/sglang/srt/disaggregation/mooncake/conn_multimodal.py`
- **核心类**：
  - `MooncakeEmbeddingSender`：发送embedding数据
  - `MooncakeEmbeddingReceiver`：接收embedding数据
  - `MooncakeEmbeddingManager`：管理传输引擎

---

## 3. 核心技术细节

### 3.1 Block-based Allocation（块分配机制）

#### 3.1.1 设计动机

在Encode/Language分离架构下，Language端无法预知Embedding端生成的实际embedding长度（因为图像token数量是动态的），但需要提前分配buffer。

**解决方案**：采用**块分配（Block-based Allocation）**机制

#### 3.1.2 实现细节

```python
class ReqToMetadataBlockAllocator:
    """Block-based allocator for variable-length metadata buffers."""
    
    def __init__(self, size: int, block_size: int):
        self.total_blocks = size
        self.block_size = block_size  # e.g., 1024 tokens per block
        self.free_blocks = deque(list(range(size)))
        self.req_to_blocks = {}
    
    def alloc(self, num_tokens: int, req_id: str) -> List[int]:
        """Allocate multiple blocks based on actual token count."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < num_blocks_needed:
            return None
        allocated_blocks = [self.free_blocks.popleft() for _ in range(num_blocks_needed)]
        self.req_to_blocks[req_id] = allocated_blocks
        return allocated_blocks
```

**关键特性**：
- **动态分配**：根据实际token数量按需分配块
- **Scatter/Gather**：支持非连续内存的高效读写
- **灵活性**：Language端可以先分配默认数量的块，不足时通过Resume机制追加

#### 3.1.3 Buffer Layout

```python
class MultimodalDataBuffers:
    def __init__(self, size: int, block_size: int, embedding_dim: int):
        # Each buffer is organized as [num_blocks, block_size * feature_dim]
        self.input_embeddings = torch.zeros(
            (size, block_size * embedding_dim), 
            dtype=torch.bfloat16
        )
        self.fill_ids = torch.zeros((size, block_size), dtype=torch.int32)
        self.mrope_positions = torch.zeros((size, 3 * block_size), dtype=torch.int32)
        self.aux_datas = torch.zeros((size, 16), dtype=torch.int32)
        # aux_datas[0] = embedding_length (total tokens)
        # aux_datas[1] = mrope_position_delta
```

### 3.2 Resume Transfer（断点续传）

#### 3.2.1 问题场景

当Embedding端生成的token数量超过Language端初始分配的buffer大小时，需要支持**分批传输**：

```
Scenario:
- Embedding generates: 10000 tokens
- Language allocates: 8192 tokens (default buffer)
→ Need to transfer in 2 chunks: 8192 + 1808
```

#### 3.2.2 实现流程

```
Embedding Side:
  1. Transfer first 8192 tokens → mark as partial (is_partial=True)
  2. Wait for Language side to request resume
  3. Receive resume request with new buffer allocation
  4. Transfer remaining 1808 tokens → mark as complete

Language Side:
  1. Receive first 8192 tokens
  2. Detect partial transfer (aux_datas[0] > received_tokens)
  3. Cache partial data in memory
  4. Allocate new buffer for remaining tokens
  5. Send resume request to Embedding side
  6. Receive remaining data and merge
```

#### 3.2.3 代码示例

**Embedding端发送**：
```python
def send_embedding(self, embedding_indices, total_tokens, block_size, 
                   sent_tokens=0, allocated_tokens=None):
    remaining_tokens = total_tokens - sent_tokens
    
    if remaining_tokens > allocated_tokens:
        # Partial transfer
        tokens_to_send = allocated_tokens
        is_partial = True
    else:
        # Final transfer
        tokens_to_send = remaining_tokens
        is_partial = False
    
    # Execute transfer...
    return is_partial
```

**Language端接收与恢复**：
```python
def pop_transferred(self):
    for language_req, poll in zip(self.queue, polls):
        if poll == KVPoll.Transferring:
            # Get partial data
            embedding_data, fill_ids, aux_datas = self.metadata_buffers.get_buf(...)
            actual_total_length = int(aux_datas[0])
            sent_tokens = len(fill_ids)
            
            if actual_total_length > sent_tokens:
                # Cache partial data
                language_req.partial_input_embeds = embedding_data
                language_req.partial_fill_ids = fill_ids
                language_req.partial_sent_tokens = sent_tokens
                
                # Allocate new buffer for remaining
                remaining_tokens = actual_total_length - sent_tokens
                new_allocation = self.allocator.alloc(num_tokens=remaining_tokens)
                
                # Send resume request
                language_req.embedding_receiver.resume_transfer(
                    embedding_indices=new_allocation,
                    sent_tokens=sent_tokens,
                    allocated_tokens=len(new_allocation) * block_size
                )
```

### 3.3 Deepstack Embedding Support

#### 3.3.1 What is Deepstack?

Deepstack是一种为MoE模型优化的技术，将中间层的hidden states作为额外的embedding注入到后续层中，提升模型性能。

#### 3.3.2 架构适配

在Encode/Language分离下，Deepstack需要特殊处理：

1. **Embedding端**：
   - 运行前N层（如前3层），捕获中间层的hidden states
   - 将`[input_embeds, layer0_hidden, layer1_hidden, layer2_hidden]`拼接

2. **Language端**：
   - 接收完整的embedding（包含deepstack部分）
   - 在相应的层中注入deepstack embeddings

```python
# Embedding side (qwen3_vl_moe.py)
if self.use_deepstack:
    # Capture hidden states at specific layers
    aux_hidden_states = []
    for layer_idx in self.layers_to_capture:
        aux_hidden_states.append(hidden_states)
    
    # Merge into single tensor for transfer
    merged_embeds = torch.cat([input_embeds] + aux_hidden_states, dim=-1)

# Language side
if input_deepstack_embeds is not None:
    for layer_idx in range(3):
        sep = self.hidden_size * layer_idx
        hidden_states.add_(
            input_deepstack_embeds[:, sep : sep + self.hidden_size]
        )
```

### 3.4 Bootstrap与握手协议

#### 3.4.1 Bootstrap Flow

```
Embedding Side:                          Language Side:
    │                                        │
    ├─[1]─ Connect to Bootstrap Server      │
    │                                        │
    │      [2]─ Connect to Bootstrap Server─┤
    │                                        │
    ├─[3]─ Send metadata ──────────────────>│
    │      (embedding_indices, total_tokens) │
    │                                        │
    │<─────────────────── Acknowledge ──[4]─┤
    │                                        │
    ├─[5]─ Begin Transfer ─────────────────>│
    │      (RDMA/Mooncake)                  │
```

#### 3.4.2 Message Format

**Init Message (Embedding → Language)**:
```python
msg = [
    room_id,                     # bootstrap room
    endpoint,                    # IP address
    dst_port,                    # port for transfer
    mooncake_session_id,         # session ID
    dst_embedding_indices_str,   # comma-separated block indices
    required_dst_info_num,       # number of destinations
    allocated_tokens,            # buffer size allocated
]
```

**Resume Message (Language → Embedding)**:
```python
msg = [
    room_id,
    endpoint,
    dst_port,
    mooncake_session_id,
    dst_embedding_indices_str,   # new block indices
    required_dst_info_num,
    sent_tokens,                 # tokens already received
    allocated_tokens,            # new buffer size
]
```

---

## 4. 模型适配

### 4.1 支持的模型

目前已适配的模型：
- **Qwen2.5-VL-MoE**
- **Qwen3-VL-MoE**

### 4.2 模型修改要点

#### 4.2.1 Embedding Instance配置

为Embedding实例添加特殊配置：
```python
config.is_multimodal_embedding = True
```

在模型初始化时，只构建必要的组件：
```python
if config.is_multimodal_embedding:
    # Build minimal model for embedding
    self.visual = Qwen2VLVisionTransformer(...)
    self.model = nn.Module()
    self.model.embed_tokens = VocabParallelEmbedding(...)
    # Skip building full language model
```

#### 4.2.2 Forward函数扩展

```python
def forward(self, input_ids, positions, forward_batch, 
            get_multimodal_embedding=False):
    # Process multimodal inputs and generate embeddings
    hidden_states = general_mm_embed_routine(
        input_ids=input_ids,
        forward_batch=forward_batch,
        multimodal_model=self,
        use_deepstack=self.use_deepstack,
        get_multimodal_embedding=get_multimodal_embedding,
    )
    
    if get_multimodal_embedding:
        # Return raw embeddings for transfer
        return EmbeddingPoolerOutput(embeddings=hidden_states)
    
    # Normal language model forward
    return self.logits_processor(...)
```

---

## 5. 使用示例

### 5.1 启动Embedding Instance

```bash
python -m sglang.launch_server \
  --model Qwen/Qwen2.5-VL-72B-Instruct-MoE \
  --disagg-mode encode \
  --disagg-transfer-backend mooncake \
  --disagg-bootstrap-host 0.0.0.0 \
  --disagg-bootstrap-port 29500 \
  --tp 8 \
  --enable-multimodal-embedding
```

**关键参数**：
- `--disagg-mode encode`：设置为Embedding模式
- `--enable-multimodal-embedding`：启用multimodal embedding特性
- `--disagg-transfer-backend mooncake`：使用Mooncake作为传输后端

### 5.2 启动Language Instance

```bash
python -m sglang.launch_server \
  --model Qwen/Qwen2.5-VL-72B-Instruct-MoE \
  --disagg-mode language \
  --disagg-transfer-backend mooncake \
  --disagg-bootstrap-host 0.0.0.0 \
  --disagg-bootstrap-port 29500 \
  --tp 8
```

**关键参数**：
- `--disagg-mode language`：设置为Language模式
- `--disagg-bootstrap-host/port`：连接到Embedding实例的bootstrap服务

### 5.3 环境变量配置

```bash
# Language端默认分配的buffer大小（tokens）
export SGLANG_EMBEDDING_DEFAULT_ALLOCATE_BUFFER_SIZE=8192

# Block大小（tokens per block）
export SGLANG_MULTIMODAL_BLOCK_SIZE=1024

# Mooncake传输配置
export MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json
```

### 5.4 客户端请求

客户端代码无需修改，正常发送multimodal请求即可：

```python
from sglang import RuntimeEndpoint

client = RuntimeEndpoint("http://language-instance:30000")

response = client.chat.completions.create(
    model="Qwen2.5-VL-72B-Instruct-MoE",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://..."}},
            {"type": "text", "text": "What's in this image?"}
        ]
    }]
)
```

---

## 6. 性能数据

> **待补充**：请在此处填写性能测试数据

### 6.1 吞吐量对比

| 配置 | Throughput (req/s) | Latency P50 (ms) | Latency P99 (ms) |
|------|-------------------|------------------|------------------|
| Baseline (EPD) | TBD | TBD | TBD |
| Encode/Language | TBD | TBD | TBD |

### 6.2 资源利用率

| 配置 | GPU Memory (Embedding) | GPU Memory (Language) | GPU Utilization |
|------|----------------------|---------------------|-----------------|
| Baseline (EPD) | TBD | TBD | TBD |
| Encode/Language | TBD | TBD | TBD |

### 6.3 Transfer开销分析

| Sequence Length | Transfer Time (ms) | Transfer Bandwidth (GB/s) |
|----------------|-------------------|--------------------------|
| 1K tokens | TBD | TBD |
| 4K tokens | TBD | TBD |
| 8K tokens | TBD | TBD |
| 16K tokens | TBD | TBD |

### 6.4 Resume Transfer效率

| Total Tokens | Initial Buffer | Resume Count | Total Time (ms) |
|--------------|----------------|--------------|-----------------|
| 10K | 8K | 1 | TBD |
| 20K | 8K | 2 | TBD |
| 32K | 8K | 3 | TBD |

---

## 7. 技术限制与已知问题

### 7.1 当前限制

1. **模型支持**：
   - 目前仅支持Qwen系列MoE模型
   - 需要模型显式适配（添加`is_multimodal_embedding`配置）

2. **传输后端**：
   - 需要RDMA支持（Mooncake）
   - 不支持本地测试（可使用FAKE backend）

3. **内存管理**：
   - Language端需要预留足够的buffer空间
   - 大序列可能需要多次resume（影响延迟）

### 7.2 已知问题

1. **Overlap模式兼容性**：
   - 目前在overlap模式下可能存在时序问题
   - 建议使用normal模式

2. **Expert权重加载**：
   - Qwen3-MoE的expert权重加载需要特殊处理
   - 详见commit `97c58812d`

---

## 8. 未来优化方向

### 8.1 短期优化

1. **自适应Buffer分配**：
   - 根据历史请求动态调整默认buffer大小
   - 减少resume次数

2. **Embedding缓存**：
   - 对相同图像的embedding进行缓存
   - 避免重复编码

3. **流式传输**：
   - 边编码边传输，降低首token延迟

### 8.2 长期规划

1. **更多模型适配**：
   - LLaVA系列
   - InternVL系列
   - CogVLM系列

2. **异构部署支持**：
   - Embedding端使用视觉优化硬件（如H20）
   - Language端使用通用GPU

3. **多Embedding实例负载均衡**：
   - 支持多个Embedding实例并行处理
   - 动态路由和负载均衡

---

## 9. 相关Commit记录

本设计基于以下commit实现（从 `5d9eb07` 到 `300c221`）：

1. **300c221** - fix: fix only-test with deepstack embedding
2. **b3f7032** - fix trigger resume transfer timing && fix reallocate logic for language
3. **97c5881** - fix: adapt fused_experts in qwen3-moe
4. **1de94f1** - fix: merge deepstack embeds into input_embeds
5. **fdc63c9** - fix: support deepstack-embediing for language instance
6. **19e40b3** - tmp support transfer for deepstackembedding
7. **60088ba** - refactor: move deepstack support to Qwen3MoeForCausalLM
8. **58fe20d** - feat: support for resume-transfer
9. **252ef6a** - Refactor multimodal embedding to use block-based allocation
10. **7bbc7ea** - fix: adapt overlap mode

**核心改动文件**：
- `python/sglang/srt/disaggregation/multimodal_embedding.py`
- `python/sglang/srt/disaggregation/multimodal_language.py`
- `python/sglang/srt/disaggregation/mooncake/conn_multimodal.py`
- `python/sglang/srt/disaggregation/utils.py`
- `python/sglang/srt/models/qwen3_moe.py`
- `python/sglang/srt/models/qwen3_vl_moe.py`

---

## 10. 参考资料

### 10.1 相关文档

- [SGLang Disaggregation Architecture](../references/disaggregation.md)
- [Mooncake Transfer Engine](https://github.com/kvcache-ai/Mooncake)

### 10.2 相关论文

- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670)

---

**文档版本**: v1.0  
**更新时间**: 2025-10-28  
**维护者**: SGLang Team
