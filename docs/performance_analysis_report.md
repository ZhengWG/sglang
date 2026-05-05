# SGLang 性能分析报告：EPD 多模态推理配置

## 目录

1. [配置概览](#1-配置概览)
2. [Model 层性能分析](#2-model-层性能分析)
3. [Scheduler 层性能分析](#3-scheduler-层性能分析)
4. [H2D (Host-to-Device) 传输分析](#4-h2d-host-to-device-传输分析)
5. [D2D (Device-to-Device / TP 通信) 分析](#5-d2d-device-to-device--tp-通信-分析)
6. [Processor / Tokenizer 性能分析](#6-processor--tokenizer-性能分析)
7. [多模态管线分析 (Encoder-Only / Language-Only)](#7-多模态管线分析-encoder-only--language-only)
8. [性能提升建议总结](#8-性能提升建议总结)

---

## 1. 配置概览

### Language Server (主推理服务器)

| 参数 | 值 | 说明 |
|------|------|------|
| `--tp-size` | 4 | 4 卡张量并行 |
| `--mem-fraction-static` | 0.85 (standalone 0.8) | 静态显存占比 |
| `--context-length` | 262144 | 256K 超长上下文 |
| `--speculative-algo` | NEXTN | 映射为 EAGLE 投机解码 |
| `--speculative-num-steps` | 3 | Draft 模型迭代 3 步 |
| `--speculative-eagle-topk` | 1 | 每步保留 top-1（退化为链式推测） |
| `--speculative-num-draft-tokens` | 4 | 每步验证 4 个 draft token |
| `--reasoning-parser` | qwen3 | Qwen3 推理模式解析器 |
| `--tool-call-parser` | qwen3_coder | 工具调用解析器 |
| `--default-thinking` | 启用 | 默认开启思考模式 |
| `--mamba-scheduler-strategy` | extra_buffer | Mamba 调度策略使用额外缓冲区 |
| `--enable-multimodal` | 启用 | 多模态支持 |
| `--language-only` | 启用 | 仅运行语言模型（编码器分离） |
| `--enable-adaptive-dispatch-to-encoder` | 启用 | 自适应分发到编码器 |
| `--encoder-urls` | 自动注入 | 编码器服务地址 |

### Encoder Server (编码器服务器)

| 参数 | 值 | 说明 |
|------|------|------|
| `--tp-size` | 1 | 单卡运行 |
| `--encoder-only` | 启用 | 仅运行视觉/音频编码器 |
| `--mm-attention-backend` | fa3 | FlashAttention-3 作为注意力后端 |
| `--mem-fraction-static` | 0.8 | 静态显存占比 |

---

## 2. Model 层性能分析

### 2.1 Forward Pass 热路径

SGLang 的模型执行核心在 `model_runner.py` 的 `_forward_raw` 方法中。关键调度逻辑：

```
_forward_raw
├── CUDA Graph 可用? → graph_runner.replay()  [decode 热路径]
├── 否则:
│   ├── prepare_mlp_sync_batch()  [DP token 对齐]
│   ├── forward_decode()          [decode 模式]
│   ├── forward_extend()          [prefill 模式]
│   ├── forward_split_prefill()   [分块 prefill]
│   └── forward_idle()            [空闲]
└── post_forward_mlp_sync_batch() [DP 后处理]
```

**性能瓶颈 1: CUDA Graph 之外的操作**

CUDA Graph `replay()` 路径是 decode 阶段最快的路径，但以下操作必须在 CUDA Graph 之外执行：

- `populate_from_forward_batch`: 每次 replay 前的 GPU→GPU buffer 拷贝（`_grouped_foreach_copy_`），包含 `input_ids`, `req_pool_indices`, `seq_lens`, `out_cache_loc`, `positions`
- `init_forward_metadata_replay_cuda_graph`: 注意力后端的元数据初始化
- `DeepEPCudaGraphRunnerAdapter.replay()`: MoE A2A dispatch 模式切换
- CPU 侧 `seq_lens_cpu.copy_`: 用于后续 Python 逻辑的 CPU 拷贝

**提升空间**: 减少 replay 前的 CPU-side 准备工作，考虑将更多元数据计算移入 CUDA Graph 或用 persistent buffer 减少每步拷贝。

**性能瓶颈 2: HiSparse 等待**

```python
# model_runner.py ~line 3115-3119
if self.hisparse_coordinator is not None and forward_batch.forward_mode.is_decode():
    self.hisparse_coordinator.wait_for_pending_backup()
```

如果使用 HiSparse，decode 阶段每步都会阻塞等待备份完成。

**性能瓶颈 3: MLP Sync Batch 准备 (DP 场景)**

`prepare_mlp_sync_batch` 在非 CUDA Graph 路径上执行 token 对齐、padding，调用多次 `torch.cat` 造成动态内存分配。对于 `DpPaddingMode.MAX_LEN` 模式，会将所有 DP rank 的 activation 膨胀到最大 rank 的 token 数。

### 2.2 投机解码 (NEXTN/EAGLE) 性能分析

`NEXTN` 在 `server_args.py` 中被标准化为 `EAGLE`：

```python
# server_args.py line 3260-3261
if self.speculative_algorithm == "NEXTN":
    self.speculative_algorithm = "EAGLE"
```

#### 投机解码 Decode 阶段的完整流程：

```
forward_batch_generation (decode)
├── 1. Draft 阶段 (3 steps)
│   ├── draft_forward() × 3 (或 CUDA Graph replay)
│   ├── 每步: top-k softmax (topk=1)
│   └── 总共生成 4 个 draft tokens
├── 2. Tree 构建
│   └── build_tree_kernel_efficient()
│       ├── 构建 tree_mask, position, retrieve_index
│       └── 分配 FULL_MASK bool tensor: O(seq_lens_sum × 4)
├── 3. Verify 阶段
│   ├── prepare_for_verify(): 分配 out_cache_loc
│   ├── Target forward (TARGET_VERIFY mode)
│   │   └── 4 tokens/req 的 batch forward
│   ├── verify(): argmax / tree_speculative_sampling
│   └── TP broadcast (accept decisions)
└── 4. Draft Extend (为下轮准备)
    └── forward_draft_extend_after_decode()
```

**性能瓶颈 4: 每个 Decode Step 需要多次 Forward**

在 `topk=1, steps=3, draft_tokens=4` 配置下，每个 decode 迭代包含：
- **Draft**: 3 步 draft model forward (可被 CUDA Graph 加速)
- **Verify**: 1 次 target model forward (4 tokens/req, TARGET_VERIFY 模式)
- **Draft Extend**: 1 次 draft model extend

共 **~5 次 model forward** 才生成 1-4 个有效 token。如果接受率较低，开销远大于标准自回归。

**性能瓶颈 5: Tree 构建开销**

即使 `topk=1` 使树退化为链，`build_tree_kernel_efficient` 仍然执行完整的树构建逻辑：
- 分配 `tree_mask`: `torch.full((seq_lens_sum * 4 + 16 * bs,), True, dtype=torch.bool)`
- 分配 `retrieve_index`, `retrieve_next_token`, `retrieve_next_sibling`, `draft_tokens`

对于 256K 上下文，`seq_lens_sum` 可能非常大，导致 `tree_mask` 分配量可观。

**提升空间**: 当 `topk=1` 时，可以特化为链式投机路径，跳过树构建，减少内存分配和 kernel 调用。

**性能瓶颈 6: Verify 阶段的 TP 广播**

```python
# eagle_info.py ~line 395-400
if tp_group.world_size > 1:
    tp_group.broadcast(predict, src=0)
    tp_group.broadcast(accept_index, src=0)
    tp_group.broadcast(accept_length, src=0)
```

在 `tp=4` 下，verify 结果需要 3 次 broadcast 同步，每次都是跨 4 卡的 NCCL 通信。

**性能瓶颈 7: Custom Mask 动态增长**

```python
# eagle_info.py ~line 202-218
if self.custom_mask.numel() < mask_numel:
    self.custom_mask = torch.cat([
        self.custom_mask,
        torch.full((mask_numel - self.custom_mask.numel(),), True, ...)
    ], dim=0)
```

当 batch 大小波动时，`custom_mask` 会通过 `torch.cat` 动态扩展，产生额外内存分配。

### 2.3 KV Cache 管理与 256K 上下文

**性能瓶颈 8: 超长上下文的 KV Cache 内存**

```python
# model_runner_kv_cache_mixin.py ~line 198-205
extra_max_context_len = 4
if self.server_args.speculative_num_draft_tokens is not None:
    extra_max_context_len += self.server_args.speculative_num_draft_tokens
```

- `ReqToTokenPool` 维度为 `max_num_reqs × (262144 + 8)`，int64 索引表
- 假设 `max_num_reqs=128`，仅索引表就需要 ~256MB
- `token_to_kv_pool` 需要 `max_total_num_tokens × layers × head_dims × dtype_size`
- 在 `mem_fraction_static=0.85` 下，可用显存大部分分配给 KV cache，导致 `max_running_requests` 较低

**提升空间**: 考虑 paged attention 的 page_size 优化、KV cache 量化 (FP8/INT8)、或 attention offloading 策略。

### 2.4 CUDA Graph 局限性

CUDA Graph 无法处理以下情况（`can_run` 判断）：

- 混合 encoder batch (`encoder_lens > 0`)
- Two-batch overlap 不支持时
- `capture_hidden_mode` 不匹配
- N-gram token layout 变化
- DP CUDA Graph 资格不满足

每次 CUDA Graph 不可用时，回退到 eager mode，性能显著下降。

---

## 3. Scheduler 层性能分析

### 3.1 调度主循环

Scheduler 的核心循环在 `event_loop_normal` / `event_loop_overlap` 中：

```
event_loop_overlap (推荐路径)
├── recv_requests()          [ZMQ 接收]
├── process_input_requests() [请求处理 + DP 广播]
├── get_next_batch_to_run()  [调度决策]
│   ├── get_new_batch_prefill()  [prefill 优先]
│   │   ├── policy.calc_priority()
│   │   └── PrefillAdder (chunked prefill, token budget)
│   └── update_running_batch()   [decode batch]
├── run_batch()              [执行 forward]
└── process_batch_result()   [结果处理，与下一步 overlap]
```

**性能瓶颈 9: ZMQ 接收 + Python pickle 反序列化**

```python
# scheduler.py ~line 1528-1544
recv_reqs = self.recv_requests()  # recv_pyobj(NOBLOCK) 循环
```

每个调度迭代都需要：
1. ZMQ `recv_pyobj(NOBLOCK)` 循环接收多条消息
2. Python `pickle.loads` 反序列化
3. TP rank 间 `broadcast_pyobj` (CPU group)

**性能瓶颈 10: 多模态输入的 TP 广播**

```python
# scheduler.py ~line 1768-1774
# _process_and_broadcast_mm_inputs
# 所有 TP scheduler rank 都会执行 CPU-heavy 的 MultimodalInputs 构建
```

在 TP=4 下，多模态输入需要在 4 个 scheduler rank 间广播。如果使用 SHM，还需要额外的 `barrier(tp_cpu_group)`。

**性能瓶颈 11: 调度策略的 Python 开销**

`PrefillAdder` 的 prefill token 预算分配、radix cache 匹配/插入、chunked prefill 切分等逻辑全部是 Python 实现，在高吞吐场景下可能成为 CPU 瓶颈。

### 3.2 Mamba Scheduler Strategy: `extra_buffer`

**配置**: `--mamba-scheduler-strategy extra_buffer`

**`extra_buffer` vs `no_buffer` 对比**:

| 特性 | `no_buffer` | `extra_buffer` |
|------|-------------|----------------|
| Radix Cache | 强制 `page_size=1` | 支持 `page_size > 1` |
| Overlap Schedule | **禁用** | 兼容 |
| 投机解码 | 不兼容 radix+spec | 兼容 (需 SPEC_V2) |
| Mamba 状态缓存 | 简单 `mamba_pool_idx` | Ping-pong track buffer |
| Cache 插入长度 | `len(token_ids)` | `mamba_last_track_seqlen` |

**性能瓶颈 12: `extra_buffer` 的额外 Tensor 构建**

每次 `prepare_for_extend` 和 `prepare_for_decode` 都需要额外构建 mamba track tensors：

```python
# schedule_batch.py ~line 1930-1945
self.mamba_track_indices = torch.tensor(mamba_track_indices_cpu, ...)
self.mamba_track_mask = torch.tensor(mamba_track_mask_cpu, ...)
self.mamba_track_seqlens = torch.tensor(mamba_track_seqlens_cpu, ...)
```

这些 tensor 从 CPU list 构建然后传到 GPU，每步额外增加 3 次小 H2D 传输。

**性能瓶颈 13: Ping-pong Buffer 管理**

```python
# mamba_radix_cache.py ~line 551-566
mamba_ping_pong_track_buffer_to_keep = (
    self.req_to_token_pool.get_mamba_ping_pong_other_idx(req.mamba_next_track_idx)
)
mamba_value = req.mamba_ping_pong_track_buffer[...].unsqueeze(-1).clone()
```

Radix cache 提交时需要 `.clone()` mamba 状态，额外的 GPU 内存分配和拷贝。

**提升空间**: 可预分配 mamba track tensor buffer，避免每步重复分配；Ping-pong buffer 的 clone 可考虑使用 double-buffer swap 替代。

### 3.3 Overlap Schedule

`event_loop_overlap` 将 `process_batch_result` 延迟到下一步执行，以 overlap CPU 调度和 GPU 计算：

```python
# scheduler.py ~line 1420-1505
# 但以下情况会 disable overlap:
disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)
# 包括: back-to-back extend, spec+grammar 等
```

**性能瓶颈 14: Overlap 被禁用的场景**

投机解码 + grammar 约束时 overlap 会被禁用，每步都变成串行的 CPU 处理 + GPU 计算。

### 3.4 DP Attention 的 Token 同步开销

**性能瓶颈 15: 每步 All-Gather 同步**

```python
# scheduler_dp_attn_mixin.py ~line 73-125
torch.distributed.all_gather_into_tensor(
    global_info_tensor.flatten(), local_info_tensor, group=group,
)
# + D2H copy:
cpu_data = tp0_info[:, :2].cpu()
```

每个调度步骤需要：
1. `all_gather_into_tensor`: 收集所有 DP rank 的 token 数
2. 一次 D2H 拷贝 (`.cpu()`)
3. Python `.tolist()` 转换
4. Forward 时的 `prepare_mlp_sync_batch` padding 对齐

---

## 4. H2D (Host-to-Device) 传输分析

### 4.1 每步 Decode 的 H2D 传输

在稳态 decode 中，H2D 传输相对较少（大部分数据已在 GPU 上），主要包括：

| 传输项 | 大小估算 | 路径 |
|--------|----------|------|
| `global_num_tokens` (DP sync) | ~64 bytes (pinned) | `forward_batch_info.py` `copy_` |
| `num_token_non_padded` | 4 bytes | `forward_batch_info.py` `.to()` |
| Mamba track tensors | ~3 × batch_size × 8 bytes | `schedule_batch.py` `torch.tensor().to()` |

### 4.2 Prefill/Extend 的 H2D 传输

Prefill 阶段 H2D 传输显著增加：

| 传输项 | 大小估算 | 是否 Pinned |
|--------|----------|-------------|
| `input_ids` | `extend_tokens × 8` bytes | Yes |
| `seq_lens` | `batch × 8` bytes | Yes |
| `orig_seq_lens` | `batch × 4` bytes | Yes |
| `token_type_ids` | `extend_tokens × 8` bytes | Yes |
| `encoder_lens` | `batch × 8` bytes | Yes |
| `input_embeds` | `tokens × hidden_size × dtype_size` | Yes |
| `replace_embeds` | 可变 | No (`.to()`) |
| `extend_seq_lens` | `batch × 4` bytes | No |
| `extend_prefix_lens` | `batch × 4` bytes | No |

**性能瓶颈 16: Pinned Memory 分配**

每次 prefill 调用 `torch.tensor(..., pin_memory=True)` 会触发 CUDA pinned memory 分配。虽然 PyTorch 有 caching allocator，但在高频场景下仍可能成为瓶颈。

**提升空间**: 预分配 pinned memory buffer pool，复用 CPU staging buffer 避免反复分配。

### 4.3 多模态特征的 H2D

在 `language_only` + 自适应分发模式下：

- 如果分发到 encoder: 编码器完成后，scheduler 通过 ZMQ 接收 embeddings，经过 `torch.frombuffer().clone()` (CPU)，然后在 `general_mm_embed_routine` 中传到 GPU
- 如果本地处理: `process_mm_data_async` 在 tokenizer 进程的 CPU/GPU 上运行 HF processor

**性能瓶颈 17: Encoder 回传的 Embedding 二次拷贝**

```python
# encode_receiver.py ~line 504-552
recv_obj.embedding = (
    torch.frombuffer(buffer, dtype=recv_obj.dtype)
    .reshape(recv_obj.shape)
    .clone()  # clone 确保 ZMQ buffer 可释放
)
```

```python
# encode_receiver.py ~line 311-320
# 合并多部分 embedding 时:
groups[mod].append(e.cuda())  # CPU→GPU
return {mod: torch.concat(tensors).to("cpu", non_blocking=True) ...}  # GPU→CPU
```

多模态 embedding 经历了: Encoder GPU → CPU (pickle) → ZMQ → CPU clone → GPU concat → CPU offload → GPU (最终使用)，存在多次冗余拷贝。

**提升空间**: 使用 CUDA IPC 或 RDMA 直接传输 GPU tensor，减少 CPU 中转。

### 4.4 LoRA Overlap Loading

```python
# lora_overlap_loader.py ~line 51-82
# 使用独立 CUDA stream 进行 H2D:
with self.load_stream_context:
    self.lora_manager.fetch_new_loras({lora_id}, loras_to_be_loaded)
    event = self.device_module.Event()
    event.record(self.load_stream)
```

LoRA 权重通过独立 stream 异步加载，通过 Event 同步。如果 LoRA 权重较大且切换频繁，H2D 带宽竞争可能影响主推理 stream。

---

## 5. D2D (Device-to-Device / TP 通信) 分析

### 5.1 TP=4 的通信模式

在 TP=4 配置下，每个 Transformer 层的通信开销：

#### 标准 Megatron 并行模式（每层）

```
Layer Forward:
├── Column Parallel Linear (QKV proj, Gate/Up proj)
│   └── gather_output=False → 无通信
├── Attention
│   └── 在本地 head 上计算
├── Row Parallel Linear (Attention output, Down proj)
│   └── all_reduce(hidden_states)  [TP=4]
│       └── 数据量: batch_tokens × hidden_size × dtype_size
├── LayerNorm (可融合 all-reduce + RMSNorm)
│   └── fused_allreduce_rmsnorm (FlashInfer/AITER)
└── MoE (如有)
    └── moe_tensor_model_parallel_all_reduce
```

#### 每层通信量估算 (TP=4)

假设 hidden_size=H, batch_tokens=T, dtype=BF16 (2 bytes):
- **Row parallel all-reduce (注意力输出)**: `T × H × 2 bytes`
- **Row parallel all-reduce (MLP 输出)**: `T × H × 2 bytes`
- **总计每层**: `~4 × T × H bytes` (ring all-reduce 实际传输量为 `2×(N-1)/N ≈ 1.5×` 倍)

对于 L 层模型，总通信量约 `L × 4 × T × H bytes`。

### 5.2 通信后端选择

SGLang 支持多种 all-reduce 实现，按优先级选择：

```python
# parallel_state.py ~line 557-634
if ca_comm.should_custom_ar(input_):       # 1. Custom All-Reduce
    outplace_all_reduce_method = "ca"
elif qr_comm.should_qr(input_):            # 2. Quick All-Reduce
    outplace_all_reduce_method = "qr"
else:                                      # 3. PyNCCL / PyTorch distributed
    inplace_all_reduce(input_)
```

**Custom All-Reduce** (`custom_all_reduce.py`): 针对小到中等 tensor 的 intra-node 优化，使用共享内存 + barrier，延迟更低。

**Quick All-Reduce** (`quick_all_reduce.py`): 基于 symmetric memory 的快速路径。

**性能瓶颈 18: All-Reduce 的选择路径开销**

每次 all-reduce 调用都需要检查 `should_custom_ar` / `should_qr`，包含 tensor 大小判断和设备检查。

### 5.3 Fused Communication (communicator.py)

`LayerCommunicator` 实现了多种融合通信模式：

#### 模式 A: Fused All-Reduce + RMSNorm

```python
# communicator.py ~line 514-532
if apply_flashinfer_allreduce_fusion(hidden_states.shape[0]):
    hidden_states, residual = self.input_layernorm.forward_with_allreduce_fusion(
        hidden_states, residual, use_attn_tp_group=False
    )
```

将 all-reduce 与 LayerNorm 融合为一个 kernel，减少一次 kernel launch 和一次全局内存读写。

**提升空间**: 确认 FlashInfer fused all-reduce 在当前硬件和配置下是否启用。如果未启用（如 SM 版本限制），可回退到 AITER 路径。

#### 模式 B: DP Attention Scatter/Gather

```python
# communicator.py ~line 1194-1214
if should_use_dp_reduce_scatterv():
    get_tp_group().reduce_scatterv(global_hidden_states, output=hidden_states, ...)
elif allow_reduce_scatter and forward_batch.dp_padding_mode.is_max_len():
    dp_reduce_scatter_tensor(hidden_states, global_hidden_states)
else:
    dp_scatter(hidden_states, global_hidden_states, forward_batch)
```

**性能瓶颈 19: DP Padding 导致的通信放大**

在 `MAX_LEN` padding 模式下，所有 DP rank 的 activation 被 pad 到最大 rank 的 token 数。如果 load 不均衡，通信量被放大到 `max_tokens_per_rank × dp_size`，而非 `sum(tokens)`。

### 5.4 Speculative Decoding 的额外通信

投机解码引入额外的 TP 通信：

1. **Draft model forward** (3步): 如果 draft model 也是 TP，每步额外 all-reduce
2. **Verify 结果广播** (3 次 broadcast): `predict`, `accept_index`, `accept_length`
3. **Draft extend**: 额外的 target/draft extend forward 中的 all-reduce

**性能瓶颈 20: 投机解码的通信开销放大**

标准 decode 每步 L 层 × 2 次 all-reduce = 2L 次通信。
投机解码每步可能变为: (3 × draft_layers + target_layers) × 2 + 3 次 broadcast。
如果 draft model 与 target model 共享 TP，通信量可能增加 3-4 倍。

### 5.5 DP Attention 的通信开销

```python
# dp_attention.py ~line 446-514
# _dp_gather_via_all_reduce: 全量 all-reduce (SUM_LEN mode)
# _dp_gather_via_all_gather: reduce_scatter + all_gather (MAX_LEN mode)
```

**性能瓶颈 21: DP Gather 的双阶段通信**

`_dp_gather_via_all_gather` 路径执行两阶段通信：
1. `attention_tp_group.reduce_scatter_tensor` (attention TP 内)
2. `tp_group.all_gather_into_tensor` (全局 TP 组)

这比单次 all-reduce 多一次 NCCL 调用。

---

## 6. Processor / Tokenizer 性能分析

### 6.1 Tokenization 路径

```
generate_request()
├── normalize_batch_and_arguments()
├── _handle_epd_disaggregation_encode_request()  [language_only]
├── _tokenize_one_request()
│   ├── _tokenize_texts()
│   │   ├── async_dynamic_batch_tokenizer.encode()  [单字符串优化路径]
│   │   └── tokenizer(...)                          [通用路径]
│   ├── mm_processor.process_mm_data_async()         [本地MM处理]
│   └── _create_tokenized_object()
└── _send_one_request()
```

**性能瓶颈 22: 同步 Tokenization**

默认路径下 `self.tokenizer(tokenizer_input, **tokenizer_kwargs)` 是同步调用，会阻塞 asyncio event loop。仅当满足以下条件时才使用异步路径：
- `async_dynamic_batch_tokenizer` 已初始化
- 输入为 `SINGLE_STRING` 格式

对于长文本 tokenization，同步调用可能引入显著延迟。

**提升空间**: 确保异步 tokenizer 路径启用，或将同步 tokenization 移入 `run_in_executor`。

### 6.2 Detokenization

```python
# detokenizer_manager.py ~line 179-216
# 分组 batch_decode: 按 (skip_special_tokens, spaces_between_special_tokens) 分组
def _grouped_batch_decode(self, ids_list, skip_list, space_list):
    if all(s == first_skip and sp == first_space ...):
        return self.tokenizer.batch_decode(ids_list, ...)  # 快速路径
    # 否则分组调用多次 batch_decode
```

**性能瓶颈 23: 增量 Detokenization 的双重解码**

流式输出时，每个 token 需要解码两次 (surr_ids + read_ids) 来计算 delta string：

```python
# 处理不完整 UTF-8 字符和 special token 边界
surr_text = decode(surr_ids)
all_text = decode(read_ids)
delta = all_text[len(surr_text):]
```

### 6.3 推理解析器开销

`--reasoning-parser qwen3` 和 `--tool-call-parser qwen3_coder` 在 detokenization 后运行，解析 `<think>` 标签和工具调用格式。

**性能瓶颈 24: 正则解析和 Grammar 约束**

如果启用 grammar-guided generation，每步 sampling 后需要运行 grammar 状态机更新，可能禁用 overlap schedule。

---

## 7. 多模态管线分析 (Encoder-Only / Language-Only)

### 7.1 EPD 架构概览

```
[Client Request]
       │
       ▼
[Tokenizer Manager (Language Server)]
       │
       ├─ 自适应分发决策 ──────────────┐
       │  total_mm_items >= 2?        │
       │                              │
       │ YES                   NO     │
       ▼                       ▼      │
[Encoder Server]     [本地处理]       │
  (TP=1, FA3)        process_mm_     │
       │             data_async()     │
       │                              │
       ▼                              │
[ZMQ to Scheduler] ◄─────────────────┘
       │
       ▼
[Scheduler (Language Server)]
       │
       ├── mm_receiver.process_waiting_requests()
       │   ├── _try_recv_mm_data() [非阻塞 ZMQ]
       │   └── all_reduce(MIN, status) [TP 同步]
       │
       ▼
[Model Forward (Language Only)]
```

### 7.2 自适应分发决策

```python
# tokenizer_manager.py ~line 2538-2572
def _should_dispatch_to_encoder(self, obj):
    total_mm_items = count(images) + count(videos) + count(audios)
    return total_mm_items >= SGLANG_ENCODER_DISPATCH_MIN_ITEMS  # default: 2
```

**性能影响**:
- 单图/单音频请求 (1 item): **本地处理**，tokenizer 进程承担全部视觉预处理开销
- 多图/视频+图 (≥2 items): **远程 encoder**，解放 language server CPU/GPU

**性能瓶颈 25: 单模态项的本地处理**

当 `total_mm_items < 2` 时，tokenizer 进程本地运行完整的 HF processor pipeline：
- 图像解码 (PIL)
- Resize + normalize
- 可能的 GPU fast image processor
- Tensor 打包

这会阻塞 tokenizer 的 asyncio loop，影响其他请求的 tokenization。

**提升空间**: 将阈值降低到 1，或为本地处理开启独立线程池。

### 7.3 Encoder Server 性能

Encoder server (`encode_server.py`) 的处理流程：

```
/encode (HTTP POST)
├── _process_mm_items()
│   ├── 线程池加载媒体 (ThreadPoolExecutor, workers=4)
│   └── 构建 processor inputs
├── get_feature_fn([mm_item])  [GPU ViT forward]
│   └── 使用 FA3 attention backend
├── mm_embedding.cpu()         [GPU→CPU]
└── ZMQ send_multipart([pickle, buffer])
```

**性能瓶颈 26: Encoder 的 GPU→CPU→网络 传输链**

```python
# encode_server.py ~line 1045-1093
mm_embedding = get_feature_fn([mm_item])
mm_embedding = mm_embedding.cpu()  # GPU→CPU: 可能阻塞
```

编码后的 embedding 强制拷贝到 CPU 进行 ZMQ 传输，引入一次完整的 GPU→CPU 同步传输。对于高分辨率图像或长视频，embedding tensor 可能很大。

**性能瓶颈 27: Encoder 的串行处理**

```python
# encode_server.py ~line 1045
# 每个 mm_item 串行处理
mm_embedding: torch.Tensor = get_feature_fn([mm_item])
```

多个模态项可能被串行编码，未充分利用 batch processing。

**提升空间**: 
- 考虑 CUDA IPC 替代 CPU 中转
- 启用 encoder batch processing
- 使用 RDMA/mooncake 替代 ZMQ 进行高性能传输

### 7.4 Scheduler 端的 MM 接收

**性能瓶颈 28: TP 状态同步等待**

```python
# encode_receiver.py ~line 891-897
# 每个调度步骤:
torch.distributed.all_reduce(
    local_status, op=torch.distributed.ReduceOp.MIN, group=self.tp_group.cpu_group,
)
```

所有 TP rank 必须对 MM 等待状态达成一致，使用 CPU group 的 `all_reduce(MIN)`。这意味着即使只有 rank 0 接收 embeddings，所有 rank 每步都参与同步。

**性能瓶颈 29: Embedding 的多次设备转移**

从 encoder 到最终使用，embedding 经历的路径：

```
Encoder GPU → CPU (mm_embedding.cpu())
→ ZMQ buffer (pickle + raw bytes)
→ Language Server CPU (torch.frombuffer().clone())
→ GPU (e.cuda() for concat)
→ CPU (concat result .to("cpu"))  [如果需要 offload]
→ GPU (最终 forward 使用)
```

总共可能 **5 次跨设备拷贝**，严重浪费 PCIe 带宽。

### 7.5 Encoder-Only Server 特性

Encoder server 使用 `--mm-attention-backend fa3` (FlashAttention-3)，TP=1：

**优势**:
- FA3 支持更好的长序列 attention 计算
- TP=1 无通信开销
- 独立 GPU 不与 language model 争夺资源

**性能瓶颈 30: Encoder Cache 效率**

```python
# encode_server.py ~line 249-255
embedding_cache_size = int(os.environ.get("SGLANG_VLM_CACHE_SIZE_MB", "4096"))
self.mm_cache = MultiModalStaticCache(embedding_cache_size * 1024 * 1024)
```

默认 4GB embedding cache，如果缓存命中率低，encoder 会反复编码相同内容。

---

## 8. 性能提升建议总结

### 优先级 P0: 高影响、相对易改

| # | 建议 | 影响模块 | 预期提升 |
|---|------|----------|----------|
| 1 | **投机解码 topk=1 特化**: 当 topk=1 时跳过完整的树构建 (`build_tree_kernel_efficient`)，使用简化的链式验证路径 | Model/Speculative | 减少 tree build kernel + FULL_MASK 分配，decode latency 减少 ~5-10% |
| 2 | **减少 Embedding 传输次数**: 在 encoder→language 通路中避免 GPU→CPU→GPU→CPU→GPU 的多次拷贝，使用 CUDA IPC 或统一 buffer | MM Pipeline | PCIe 带宽节省 50%+，大图/视频场景显著 |
| 3 | **预分配 Mamba Track Tensors**: 使用持久化 GPU buffer 替代每步 `torch.tensor().to()` | Scheduler/Mamba | 减少每 decode step 的小 H2D 传输和分配开销 |
| 4 | **Pinned Memory Buffer Pool**: 为 prefill 路径预分配 pinned CPU buffer，复用 staging 区域 | H2D | 减少 CUDA pinned memory 分配延迟 |

### 优先级 P1: 中高影响、中等改动

| # | 建议 | 影响模块 | 预期提升 |
|---|------|----------|----------|
| 5 | **DP Padding 优化**: 在 load 均衡场景下优先使用 `SUM_LEN` 模式避免 max-padding 放大 | D2D/Communicator | 通信量减少 (worst case: dp_size 倍 → 1倍) |
| 6 | **Fused All-Reduce + RMSNorm 确认启用**: 验证 FlashInfer/AITER fusion 在当前 SM 版本上是否生效 | D2D | 每层节省 1 次 kernel launch + 全局读写 |
| 7 | **Verify 结果 Broadcast 合并**: 将 3 次 broadcast (predict, accept_index, accept_length) 合并为 1 次 | Speculative/D2D | 减少 2 次 NCCL 调用延迟 (~μs 级，但每步累积) |
| 8 | **异步 Tokenization**: 确保长文本 tokenization 不阻塞 asyncio loop | Processor | 减少 TTFT (Time to First Token) |
| 9 | **Encoder Batch Processing**: 允许 encoder server 对多个请求的 MM items 进行 batch forward | MM Pipeline | 提高 encoder GPU 利用率 |
| 10 | **自适应分发阈值调优**: 将 `SGLANG_ENCODER_DISPATCH_MIN_ITEMS` 从 2 降到 1 | MM Pipeline | 减轻 language server tokenizer 进程的 CPU 负载 |

### 优先级 P2: 架构级优化

| # | 建议 | 影响模块 | 预期提升 |
|---|------|----------|----------|
| 11 | **KV Cache 量化 (FP8/INT8)**: 减少 KV cache 内存占用，增加 max_running_requests | Model/Memory | 256K 上下文下显著增加并发数 |
| 12 | **Encoder-Language RDMA 传输**: 替换 ZMQ 为 RDMA (mooncake backend) 进行 embedding 传输 | MM Pipeline | 消除 CPU 中转，延迟降低 1-2 个数量级 |
| 13 | **Custom Mask 预分配**: 根据 max_batch_size × max_draft_tokens 预分配 custom_mask buffer | Speculative | 消除运行时 `torch.cat` 扩展 |
| 14 | **Two-Batch Overlap for Speculative**: 支持 spec decode + overlap schedule 共存 | Scheduler/Speculative | 隐藏 CPU 调度时间 |
| 15 | **Draft Model TP 独立**: 如果 draft model 较小，考虑 draft model 使用 TP=1 减少通信 | Speculative/D2D | Draft 阶段通信降为 0 |

### 性能监控建议

为量化上述瓶颈的实际影响，建议开启以下监控：

1. **SGLang Tracing**: 通过 `set_time_batch` 已有的 trace 点监控各阶段耗时
   - `spec_draft_start_time` → `spec_draft_end_time`
   - `spec_verify_start_time` → `spec_verify_end_time`
   - `spec_draft_extend_start_time` → `spec_draft_extend_end_time`

2. **CUDA Profiling**: 使用 `nsys` 或 SGLang 内置的 `torch.profiler` 集成
   - 关注 NCCL 通信占比
   - H2D/D2H 传输占比
   - Kernel launch gap (bubble)

3. **Scheduler 指标**:
   - `num_running_reqs` / `max_running_requests` 利用率
   - prefill vs decode 调度比例
   - 投机解码接受率 (acceptance rate)

4. **Encoder 指标**:
   - Encoder cache hit rate
   - Encoding latency per modality
   - ZMQ transfer time

---

## 附录: 关键代码路径索引

| 组件 | 核心文件 |
|------|----------|
| Model Runner | `python/sglang/srt/model_executor/model_runner.py` |
| CUDA Graph | `python/sglang/srt/model_executor/cuda_graph_runner.py` |
| Forward Batch | `python/sglang/srt/model_executor/forward_batch_info.py` |
| KV Cache Mixin | `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` |
| Scheduler | `python/sglang/srt/managers/scheduler.py` |
| Schedule Batch | `python/sglang/srt/managers/schedule_batch.py` |
| DP Attention | `python/sglang/srt/layers/dp_attention.py` |
| Communicator | `python/sglang/srt/layers/communicator.py` |
| TP Parallel | `python/sglang/srt/distributed/parallel_state.py` |
| Custom All-Reduce | `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py` |
| EAGLE Worker | `python/sglang/srt/speculative/eagle_worker.py` |
| EAGLE Info | `python/sglang/srt/speculative/eagle_info.py` |
| EAGLE Utils | `python/sglang/srt/speculative/eagle_utils.py` |
| Draft CUDA Graph | `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py` |
| Tokenizer Manager | `python/sglang/srt/managers/tokenizer_manager.py` |
| Detokenizer | `python/sglang/srt/managers/detokenizer_manager.py` |
| Encode Server | `python/sglang/srt/disaggregation/encode_server.py` |
| Encode Receiver | `python/sglang/srt/disaggregation/encode_receiver.py` |
| MM Utils | `python/sglang/srt/managers/mm_utils.py` |
| Server Args | `python/sglang/srt/server_args.py` |
| Mamba Radix Cache | `python/sglang/srt/mem_cache/mamba_radix_cache.py` |
| Offloader | `python/sglang/srt/utils/offloader.py` |
