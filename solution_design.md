# 方案详细设计：解决高并发下QPS降低问题

## 方案1：异步通信 - 使用异步广播机制避免阻塞

### 1.1 设计思路

**核心思想**：将同步阻塞的 `broadcast_object_list` 改为异步操作，使用后台线程处理广播，主线程可以继续处理其他请求。

**关键改进**：
- Entry rank：后台线程执行 `from_dict` + `broadcast`，主线程立即返回，继续处理其他请求
- Non-entry ranks：后台线程等待接收广播，主线程可以处理其他请求
- 使用 Future/Promise 模式：请求在需要 `image_inputs` 时才等待结果

### 1.2 实现架构

```
主线程 (Scheduler)                   后台线程 (BroadcastWorker)
─────────────────                     ────────────────────────
handle_generate_request()             
  ↓                                   
  提交任务到队列                       
  ↓                                   
  继续处理其他请求                     
  ↓                                   
  ... (处理其他请求)                   
  ↓                                   
  需要image_inputs时                   
  ↓                                   
  等待Future完成 ←──────────────────→ from_dict() + broadcast()
                                         ↓
                                        完成，设置Future结果
```

### 1.3 代码实现

#### 1.3.1 添加异步广播管理器

```python
import threading
import queue
from typing import Optional, Dict
from concurrent.futures import Future, ThreadPoolExecutor
import torch.distributed

class AsyncMMInputsBroadcaster:
    """异步处理多模态输入的广播，避免阻塞主线程"""
    
    def __init__(self, cpu_group, is_entry_rank: bool, max_workers: int = 2):
        self.cpu_group = cpu_group
        self.is_entry_rank = is_entry_rank
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="MMBroadcast")
        self.pending_futures: Dict[str, Future] = {}  # req_id -> Future
        self.lock = threading.Lock()
        
    def submit_broadcast_task(
        self, 
        req_id: str, 
        raw_mm_inputs: Optional[dict]
    ) -> Future:
        """提交广播任务，返回Future对象"""
        if raw_mm_inputs is None:
            # 立即返回None结果
            future = Future()
            future.set_result(None)
            return future
            
        future = Future()
        
        with self.lock:
            self.pending_futures[req_id] = future
            
        # 提交到线程池异步执行
        self.executor.submit(
            self._process_and_broadcast_async,
            req_id,
            raw_mm_inputs,
            future
        )
        
        return future
    
    def _process_and_broadcast_async(
        self,
        req_id: str,
        raw_mm_inputs: dict,
        future: Future
    ):
        """在后台线程中执行from_dict和广播"""
        try:
            group_world_size = 1
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and self.cpu_group is not None
            ):
                group_world_size = torch.distributed.get_world_size(
                    group=self.cpu_group
                )
            
            if self.is_entry_rank:
                # Entry rank: 物化 + 广播
                image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
                if group_world_size > 1:
                    obj_list = [image_inputs]
                    torch.distributed.broadcast_object_list(
                        obj_list, src=0, group=self.cpu_group
                    )
                    image_inputs = obj_list[0]
                future.set_result(image_inputs)
            else:
                # Non-entry rank: 等待接收
                if group_world_size > 1:
                    obj_list = [None]
                    torch.distributed.broadcast_object_list(
                        obj_list, src=0, group=self.cpu_group
                    )
                    image_inputs = obj_list[0]
                else:
                    image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
                future.set_result(image_inputs)
                
        except Exception as e:
            future.set_exception(e)
        finally:
            with self.lock:
                self.pending_futures.pop(req_id, None)
    
    def cancel_pending(self, req_id: str):
        """取消pending的任务（如果请求被abort）"""
        with self.lock:
            future = self.pending_futures.pop(req_id, None)
            if future and not future.done():
                future.cancel()
    
    def shutdown(self):
        """关闭线程池"""
        self.executor.shutdown(wait=True)
```

#### 1.3.2 修改Scheduler类

```python
class Scheduler:
    def __init__(self, ...):
        # ... 现有初始化代码 ...
        
        # 初始化异步广播器
        self.async_mm_broadcaster = AsyncMMInputsBroadcaster(
            cpu_group=self.cpu_group,
            is_entry_rank=self.is_entry_rank,
            max_workers=2  # 可以配置
        )
    
    def _process_and_broadcast_mm_inputs(
        self,
        raw_mm_inputs: Optional[dict],
        req_id: str,
        blocking: bool = False  # 新增参数：是否阻塞等待
    ):
        """异步处理多模态输入广播"""
        if raw_mm_inputs is None:
            return None
        
        # 提交异步任务
        future = self.async_mm_broadcaster.submit_broadcast_task(
            req_id, raw_mm_inputs
        )
        
        if blocking:
            # 阻塞等待结果（用于需要立即使用image_inputs的场景）
            return future.result()
        else:
            # 返回Future，让调用者决定何时等待
            return future
    
    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        # ... 前面的代码保持不变 ...
        
        # Handle multimodal inputs - 异步方式
        if recv_req.mm_inputs is not None:
            # 提交异步任务，不阻塞
            mm_inputs_future = self._process_and_broadcast_mm_inputs(
                recv_req.mm_inputs, 
                recv_req.rid,
                blocking=False
            )
            
            # 在需要image_inputs时才等待（这里需要立即使用，所以阻塞）
            # 但此时主线程已经可以处理其他消息了
            image_inputs = mm_inputs_future.result() if isinstance(mm_inputs_future, Future) else mm_inputs_future
            
            # ... 后续处理保持不变 ...
```

#### 1.3.3 优化：延迟等待

更进一步的优化：将等待推迟到真正需要 `image_inputs` 的时候：

```python
class Req:
    def __init__(self, ...):
        # ... 现有字段 ...
        self._mm_inputs_future: Optional[Future] = None  # 延迟等待的Future
    
    def get_image_inputs(self) -> Optional[MultimodalInputs]:
        """延迟获取image_inputs，只在需要时才等待"""
        if self._mm_inputs_future is None:
            return None
        if isinstance(self._mm_inputs_future, Future):
            # 等待完成
            image_inputs = self._mm_inputs_future.result()
            self._mm_inputs_future = image_inputs  # 缓存结果
            return image_inputs
        return self._mm_inputs_future

def handle_generate_request(self, recv_req: TokenizedGenerateReqInput):
    # ... 创建req ...
    
    if recv_req.mm_inputs is not None:
        # 只提交任务，不等待
        req._mm_inputs_future = self._process_and_broadcast_mm_inputs(
            recv_req.mm_inputs, recv_req.rid, blocking=False
        )
    
    # ... 其他处理 ...
    
    # 在真正需要时才等待
    if req._mm_inputs_future is not None:
        image_inputs = req.get_image_inputs()
        req.origin_input_ids = self.pad_input_ids_func(
            req.origin_input_ids, image_inputs
        )
        req.extend_image_inputs(image_inputs)
```

### 1.4 优点

1. **非阻塞**：主线程可以继续处理其他请求，提高并发处理能力
2. **并行处理**：多个请求的广播可以并行执行（通过线程池）
3. **灵活性**：可以选择立即等待或延迟等待
4. **向后兼容**：可以保留同步模式作为fallback

### 1.5 缺点和注意事项

1. **线程开销**：需要额外的线程和线程池管理
2. **内存开销**：pending的Future和任务队列占用内存
3. **错误处理**：需要确保异常能正确传播
4. **请求取消**：如果请求被abort，需要取消pending的广播任务
5. **线程安全**：需要确保线程安全（使用lock保护共享状态）

### 1.6 性能预期

- **低并发场景**：开销略增（线程管理），但影响不大
- **高并发场景**：显著提升QPS，因为主线程不再被阻塞
- **延迟**：单个请求的延迟可能略增（线程切换），但整体吞吐量提升

---

## 方案2：延迟物化 - 只广播原始dict，各rank独立物化

### 2.1 设计思路

**核心思想**：避免序列化已物化的对象，只广播原始的dict数据，让各rank独立执行 `from_dict`。

**关键改进**：
- Entry rank：只广播 `raw_mm_inputs`（dict），不执行 `from_dict`
- Non-entry ranks：接收dict后，本地执行 `from_dict`
- 这样避免了序列化大型numpy数组/PIL.Image的开销

### 2.2 实现架构

```
Entry Rank                          Non-entry Ranks
──────────                          ───────────────
raw_mm_inputs (dict)                
  ↓                                 
broadcast dict (小体积) ───────────→ 接收 dict
  ↓                                 ↓
本地 from_dict()                    本地 from_dict()
  ↓                                 ↓
MultimodalInputs                    MultimodalInputs
```

**优势**：
- 广播的是原始dict（base64字符串），体积小，序列化快
- 各rank并行执行 `from_dict`，充分利用CPU
- 避免了序列化大型numpy数组的开销

### 2.3 代码实现

#### 2.3.1 修改 `_process_and_broadcast_mm_inputs`

```python
def _process_and_broadcast_mm_inputs(
    self,
    raw_mm_inputs: Optional[dict],
):
    """只广播原始dict，各rank独立物化"""
    if raw_mm_inputs is None:
        return None

    group_world_size = 1
    try:
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and self.cpu_group is not None
        ):
            group_world_size = torch.distributed.get_world_size(
                group=self.cpu_group
            )
    except Exception as e:
        logger.warning(
            f"Failed to get world size in mm_inputs handling with {e}, fallback to 1."
        )

    # 关键改进：只广播原始dict，不物化
    if group_world_size > 1:
        if self.is_entry_rank:
            # Entry rank: 广播原始dict
            obj_list = [raw_mm_inputs]
            torch.distributed.broadcast_object_list(
                obj_list, src=0, group=self.cpu_group
            )
            # Entry rank也使用广播后的dict（保持一致）
            raw_mm_inputs = obj_list[0]
        else:
            # Non-entry ranks: 接收dict
            obj_list = [None]
            torch.distributed.broadcast_object_list(
                obj_list, src=0, group=self.cpu_group
            )
            raw_mm_inputs = obj_list[0]
    
    # 所有ranks独立执行from_dict（并行）
    image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
    
    return image_inputs
```

#### 2.3.2 进一步优化：条件物化

可以根据数据大小决定是否物化：

```python
def _process_and_broadcast_mm_inputs(
    self,
    raw_mm_inputs: Optional[dict],
    materialize_on_entry: bool = False,  # 新增参数
):
    """延迟物化：可选择是否在entry rank物化"""
    if raw_mm_inputs is None:
        return None

    group_world_size = 1
    try:
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and self.cpu_group is not None
        ):
            group_world_size = torch.distributed.get_world_size(
                group=self.cpu_group
            )
    except Exception as e:
        logger.warning(
            f"Failed to get world size in mm_inputs handling with {e}, fallback to 1."
        )

    # 广播原始dict
    if group_world_size > 1:
        if self.is_entry_rank:
            obj_list = [raw_mm_inputs]
            torch.distributed.broadcast_object_list(
                obj_list, src=0, group=self.cpu_group
            )
            raw_mm_inputs = obj_list[0]
        else:
            obj_list = [None]
            torch.distributed.broadcast_object_list(
                obj_list, src=0, group=self.cpu_group
            )
            raw_mm_inputs = obj_list[0]
    
    # 所有ranks并行执行from_dict
    # 这样避免了序列化大型对象的开销，同时各rank可以并行计算
    image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
    
    return image_inputs
```

#### 2.3.3 优化：估算数据大小

可以根据数据大小动态选择策略：

```python
def _estimate_mm_inputs_size(raw_mm_inputs: dict) -> int:
    """估算mm_inputs的大小（字节）"""
    import sys
    import pickle
    
    try:
        # 估算序列化后的大小
        return sys.getsizeof(pickle.dumps(raw_mm_inputs))
    except:
        # fallback: 简单估算
        return sys.getsizeof(str(raw_mm_inputs))

def _process_and_broadcast_mm_inputs(
    self,
    raw_mm_inputs: Optional[dict],
    size_threshold: int = 10 * 1024 * 1024,  # 10MB阈值
):
    """根据数据大小选择策略"""
    if raw_mm_inputs is None:
        return None

    group_world_size = 1
    try:
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and self.cpu_group is not None
        ):
            group_world_size = torch.distributed.get_world_size(
                group=self.cpu_group
            )
    except Exception as e:
        logger.warning(
            f"Failed to get world size in mm_inputs handling with {e}, fallback to 1."
        )

    # 估算数据大小
    estimated_size = _estimate_mm_inputs_size(raw_mm_inputs)
    
    # 如果数据很大，使用延迟物化策略（只广播dict）
    # 如果数据很小，可以考虑在entry rank物化后广播（减少重复计算）
    use_delayed_materialization = estimated_size > size_threshold
    
    if use_delayed_materialization or group_world_size == 1:
        # 策略1：延迟物化 - 只广播dict，各rank并行from_dict
        if group_world_size > 1:
            if self.is_entry_rank:
                obj_list = [raw_mm_inputs]
                torch.distributed.broadcast_object_list(
                    obj_list, src=0, group=self.cpu_group
                )
                raw_mm_inputs = obj_list[0]
            else:
                obj_list = [None]
                torch.distributed.broadcast_object_list(
                    obj_list, src=0, group=self.cpu_group
                )
                raw_mm_inputs = obj_list[0]
        
        # 所有ranks并行执行from_dict
        image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
        return image_inputs
    else:
        # 策略2：原始策略 - entry rank物化后广播（适用于小数据）
        if self.is_entry_rank:
            image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
            if group_world_size > 1:
                obj_list = [image_inputs]
                torch.distributed.broadcast_object_list(
                    obj_list, src=0, group=self.cpu_group
                )
                image_inputs = obj_list[0]
            return image_inputs
        else:
            obj_list = [None]
            torch.distributed.broadcast_object_list(
                obj_list, src=0, group=self.cpu_group
            )
            return obj_list[0]
```

### 2.4 优点

1. **避免序列化开销**：只广播原始dict（base64字符串），体积小，序列化快
2. **并行计算**：各rank并行执行 `from_dict`，充分利用CPU资源
3. **实现简单**：修改量小，不需要额外的线程管理
4. **向后兼容**：可以保留条件逻辑，根据数据大小选择策略

### 2.5 缺点和注意事项

1. **仍有重复计算**：各rank都需要执行 `from_dict`，CPU占用仍然较高
2. **同步阻塞**：虽然序列化开销小了，但 `broadcast_object_list` 仍然是同步的
3. **数据大小依赖**：对于非常大的数据，广播dict也可能有开销

### 2.6 性能预期

- **低并发场景**：CPU占用仍然较高（各rank重复计算），但序列化开销大幅降低
- **高并发场景**：QPS提升，因为：
  - 广播dict的开销远小于广播物化对象
  - 各rank可以并行处理不同请求（虽然每个请求仍有重复计算）
- **延迟**：单个请求的延迟可能略增（各rank都需要from_dict），但整体吞吐量提升

### 2.7 与方案1的对比

| 特性 | 方案1（异步通信） | 方案2（延迟物化） |
|------|-----------------|------------------|
| 实现复杂度 | 高（需要线程管理） | 低（简单修改） |
| CPU占用 | 低（避免重复计算） | 高（各rank重复计算） |
| 序列化开销 | 高（物化对象） | 低（原始dict） |
| 阻塞问题 | 解决（异步） | 部分解决（序列化快） |
| 高并发QPS | 高 | 中等 |
| 内存开销 | 中等（Future队列） | 低 |

---

## 方案3：组合方案（推荐）

结合方案1和方案2的优点：

1. **使用延迟物化**：只广播原始dict，避免序列化开销
2. **使用异步通信**：异步处理广播，避免阻塞主线程
3. **条件优化**：根据数据大小和并发度动态选择策略

### 3.1 实现示例

```python
class AsyncDelayedMMInputsBroadcaster:
    """异步 + 延迟物化的组合方案"""
    
    def __init__(self, cpu_group, is_entry_rank: bool, max_workers: int = 2):
        self.cpu_group = cpu_group
        self.is_entry_rank = is_entry_rank
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_futures: Dict[str, Future] = {}
        self.lock = threading.Lock()
    
    def submit_broadcast_task(
        self, 
        req_id: str, 
        raw_mm_inputs: Optional[dict]
    ) -> Future:
        """提交任务：异步广播dict + 并行from_dict"""
        if raw_mm_inputs is None:
            future = Future()
            future.set_result(None)
            return future
            
        future = Future()
        with self.lock:
            self.pending_futures[req_id] = future
            
        self.executor.submit(
            self._async_broadcast_dict_and_materialize,
            req_id,
            raw_mm_inputs,
            future
        )
        
        return future
    
    def _async_broadcast_dict_and_materialize(
        self,
        req_id: str,
        raw_mm_inputs: dict,
        future: Future
    ):
        """异步广播dict，然后各rank并行from_dict"""
        try:
            group_world_size = 1
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and self.cpu_group is not None
            ):
                group_world_size = torch.distributed.get_world_size(
                    group=self.cpu_group
                )
            
            # 只广播原始dict（小体积，快速）
            if group_world_size > 1:
                if self.is_entry_rank:
                    obj_list = [raw_mm_inputs]
                    torch.distributed.broadcast_object_list(
                        obj_list, src=0, group=self.cpu_group
                    )
                    raw_mm_inputs = obj_list[0]
                else:
                    obj_list = [None]
                    torch.distributed.broadcast_object_list(
                        obj_list, src=0, group=self.cpu_group
                    )
                    raw_mm_inputs = obj_list[0]
            
            # 各rank并行执行from_dict（在后台线程中，不阻塞主线程）
            image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
            future.set_result(image_inputs)
            
        except Exception as e:
            future.set_exception(e)
        finally:
            with self.lock:
                self.pending_futures.pop(req_id, None)
```

### 3.2 优势

- **最佳性能**：结合两种方案的优点
- **非阻塞**：主线程可以继续处理其他请求
- **低序列化开销**：只广播dict
- **灵活性**：可以根据场景调整策略

---

## 总结

### 推荐方案

**优先推荐方案2（延迟物化）**：
- 实现简单，修改量小
- 立即解决序列化开销问题
- 在高并发下能显著提升QPS

**如果方案2不够，再考虑方案3（组合方案）**：
- 进一步解决阻塞问题
- 但实现复杂度更高

### 实施建议

1. **第一阶段**：实施方案2（延迟物化）
   - 快速验证效果
   - 如果QPS提升满足需求，停止

2. **第二阶段**（如需要）：实施方案3（组合方案）
   - 如果方案2后仍有阻塞问题
   - 进一步优化高并发性能

3. **监控指标**：
   - QPS变化
   - CPU占用率
   - 请求延迟（p50/p99）
   - 广播时间 vs from_dict时间
