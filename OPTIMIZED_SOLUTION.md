# 优化方案：避免重复计算同时保持并发性能

## 🎯 核心思路

**在 Tokenizer 阶段就完成 `MultimodalInputs` 对象构造（包括hash计算），而不是传输dict让每个scheduler rank重复计算。**

## 📊 当前流程分析

### 现有流程
```
Tokenizer Manager:
  mm_inputs = mm_processor.process(...)  # 返回 dict
  ↓
  TokenizedGenerateReqInput(mm_inputs=mm_inputs)  # dict格式
  ↓
  send_to_scheduler.send_pyobj(req)
  ↓
Scheduler (broadcast_pyobj):
  recv_reqs = broadcast_pyobj(...)  # mm_inputs (dict) 已经广播了
  ↓
Scheduler (每个rank):
  image_inputs = MultimodalInputs.from_dict(recv_req.mm_inputs)  # ← 重复计算hash
  # 对于大tensor，每个rank都要hash一次，浪费！
```

### 问题
- `mm_inputs` dict 已经被 broadcast 过一次了
- 但每个scheduler rank都要重复执行 `from_dict()` 里的CPU密集hash计算
- 对于大tensor（高分辨率图像、长视频），hash计算很昂贵

## ✅ 优化方案1：在 Tokenizer 阶段完成对象构造（推荐）

### 核心改动

#### 1. 修改 Tokenizer Manager

```python
# tokenizer_manager.py

# 原来：返回dict
if self.mm_processor and obj.contains_mm_input():
    mm_inputs: Dict = await self.mm_data_processor.process(...)
else:
    mm_inputs = None

# 优化后：直接构造 MultimodalInputs 对象
if self.mm_processor and obj.contains_mm_input():
    mm_inputs_dict: Dict = await self.mm_data_processor.process(...)
    # 在tokenizer阶段就完成对象构造和hash计算（一次性）
    mm_inputs = MultimodalInputs.from_dict(mm_inputs_dict)
else:
    mm_inputs = None

tokenized_obj = TokenizedGenerateReqInput(
    input_text,
    input_ids,
    mm_inputs,  # 现在是 MultimodalInputs 对象，不是 dict
    ...
)
```

#### 2. 修改 TokenizedGenerateReqInput 类型定义

```python
# io_struct.py

@dataclass
class TokenizedGenerateReqInput(BaseReq):
    input_text: str
    input_ids: List[int]
    # 类型从 dict 改为 Optional[MultimodalInputs]
    mm_inputs: Optional[MultimodalInputs]  # 原来是 dict
    sampling_params: SamplingParams
    ...
```

#### 3. 修改 Scheduler (回滚 commit 17a57fd86)

```python
# scheduler.py

def handle_generate_request(self, recv_req: TokenizedGenerateReqInput):
    ...
    # 原来的复杂逻辑：
    # if recv_req.mm_inputs is not None:
    #     image_inputs = self._process_and_broadcast_mm_inputs(recv_req.mm_inputs)
    
    # 优化后：直接使用，无需from_dict
    if recv_req.mm_inputs is not None:
        image_inputs = recv_req.mm_inputs  # 已经是构造好的对象！
        
        # 只需要执行轻量级的pad操作
        req.origin_input_ids = self.pad_input_ids_func(
            req.origin_input_ids, image_inputs
        )
        req.extend_image_inputs(image_inputs)
    ...
```

### 优势分析

✅ **避免重复计算**
- hash计算只在tokenizer阶段执行一次
- 所有scheduler ranks收到的就是构造好的对象

✅ **保持并发性能**
- 没有引入额外的同步阻塞
- `broadcast_pyobj` 本身就会广播整个对象
- 各rank接收后直接使用，并行处理

✅ **兼容性好**
- 只需要修改类型定义，不改变整体架构
- `broadcast_pyobj` 会自动处理对象的pickle和广播

✅ **对大tensor特别有效**
- 大tensor的hash计算从 O(N × TP_size) 降到 O(N)
- 节省的CPU时间 = (TP_size - 1) × hash_time

### 性能对比

```
场景: 10MB tensor, TP_size=4, hash_time=20ms

原方案:
  Tokenizer: 生成dict (0ms)
  Broadcast: 传输dict + 对象结构 (~30ms)
  Scheduler: 4个ranks各自hash (4 × 20ms = 80ms总CPU)
  总延迟: ~30ms (并行执行)
  总CPU: 80ms

Commit方案（有问题）:
  Tokenizer: 生成dict (0ms)
  Broadcast: 传输dict (~30ms)
  Scheduler rank 0: hash (20ms) + pickle object (15ms)
  Broadcast again: 传输对象 (~25ms)
  其他ranks: unpickle (10ms)
  总延迟: 100ms (串行化！)
  总CPU: 20ms (hash) + 60ms (pickle/unpickle)

优化方案（本方案）:
  Tokenizer: hash一次 (20ms)
  Broadcast: 传输已构造对象 (~35ms)
  Scheduler: 直接使用 (0ms)
  总延迟: ~35ms (略增5ms，可接受)
  总CPU: 20ms (只hash一次！)

收益:
  vs 原方案: CPU减少 75% (80ms -> 20ms) ✓
  vs Commit方案: 延迟减少 65% (100ms -> 35ms) ✓
  vs 原方案: 延迟略增 17% (30ms -> 35ms, 可接受)
```

## ✅ 优化方案2：条件判断 + 缓存（备选）

如果不想改 tokenizer，可以在 scheduler 端优化：

```python
# scheduler.py

class Scheduler:
    def __init__(self, ...):
        ...
        # 添加对象缓存
        self.mm_inputs_cache = {}  # key: hash(dict), value: MultimodalInputs
        self.cache_max_size = 1000
    
    def _get_or_create_mm_inputs(self, raw_mm_inputs: dict) -> MultimodalInputs:
        """
        使用缓存避免重复计算
        对于相同的输入dict，只计算一次
        """
        if raw_mm_inputs is None:
            return None
        
        # 快速hash dict（不hash大tensor内容）
        cache_key = hash(tuple(sorted(raw_mm_inputs.keys())))
        
        # 检查缓存
        if cache_key in self.mm_inputs_cache:
            return self.mm_inputs_cache[cache_key]
        
        # Cache miss：执行from_dict
        mm_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
        
        # 更新缓存
        if len(self.mm_inputs_cache) < self.cache_max_size:
            self.mm_inputs_cache[cache_key] = mm_inputs
        
        return mm_inputs
    
    def handle_generate_request(self, recv_req: TokenizedGenerateReqInput):
        ...
        if recv_req.mm_inputs is not None:
            # 使用缓存版本
            image_inputs = self._get_or_create_mm_inputs(recv_req.mm_inputs)
            ...
```

**缺点**：
- 缓存key不准确（可能误命中）
- 仍然每个rank都有独立缓存
- 对于不重复的请求没有帮助

## ✅ 优化方案3：异步预处理队列（高级）

如果需要更精细的控制：

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class Scheduler:
    def __init__(self, ...):
        ...
        # 异步预处理池（只在rank 0）
        if self.is_entry_rank:
            self.mm_processor_pool = ThreadPoolExecutor(max_workers=4)
            self.pending_mm_tasks = {}  # rid -> Future
    
    async def _preprocess_mm_inputs_async(self, rid: str, raw_mm_inputs: dict):
        """在后台线程池中异步执行from_dict"""
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.mm_processor_pool,
            MultimodalInputs.from_dict,
            raw_mm_inputs
        )
        return await future
    
    def handle_generate_request(self, recv_req: TokenizedGenerateReqInput):
        ...
        if recv_req.mm_inputs is not None:
            if self.is_entry_rank:
                # Rank 0: 异步执行from_dict
                task = asyncio.create_task(
                    self._preprocess_mm_inputs_async(recv_req.rid, recv_req.mm_inputs)
                )
                self.pending_mm_tasks[recv_req.rid] = task
                # 继续处理其他请求，不阻塞
            else:
                # 其他ranks: 等待rank 0广播结果
                # (通过某种机制接收)
                ...
```

**优点**：
- 不阻塞主线程
- 可以并发处理多个请求

**缺点**：
- 实现复杂度高
- 需要处理同步和错误处理

## 🎯 推荐实施方案

### 短期（立即）：方案1 - 在Tokenizer阶段完成构造

**推荐理由**：
1. ✅ 最优性能：hash只计算一次
2. ✅ 保持并发：无同步阻塞
3. ✅ 改动简单：只需修改几个地方
4. ✅ 架构清晰：职责分明

**实施步骤**：
1. 修改 `tokenizer_manager.py`：在生成 mm_inputs 后立即调用 `from_dict`
2. 修改 `io_struct.py`：更新类型定义
3. 修改 `scheduler.py`：回滚 commit 17a57fd86，直接使用对象
4. 测试验证

### 中期：如果tokenizer修改有风险，使用方案2（缓存）

### 长期：如果需要极致性能，考虑方案3（异步）

## 📝 实施清单

- [ ] 修改 tokenizer_manager.py 的 mm_inputs 处理逻辑
- [ ] 修改 io_struct.py 的类型定义
- [ ] 回滚 scheduler.py 中的 commit 17a57fd86
- [ ] 添加单元测试验证正确性
- [ ] 性能测试对比
- [ ] 灰度发布验证

## 🔍 需要验证的点

1. **序列化大小**：
   - MultimodalInputs 对象 vs dict，哪个pickle后更大？
   - 如果对象更大，传输时间会增加

2. **兼容性**：
   - 确保所有使用 mm_inputs 的地方都兼容新类型

3. **错误处理**：
   - 如果 from_dict 在 tokenizer 失败，如何处理？

## 总结

**核心思想**：把计算移到更早的阶段（tokenizer），让后续的broadcast自然地传播已计算好的结果，而不是引入额外的同步机制。

这个方案：
- ✅ 避免了重复计算（对大tensor特别有效）
- ✅ 保持了并发性能（无同步阻塞）
- ✅ 实现简单（职责清晰）
- ✅ 兼容现有架构
