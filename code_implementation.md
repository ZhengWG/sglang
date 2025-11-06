# 代码实现示例

## 方案2实现：延迟物化（推荐，简单有效）

### 修改 scheduler.py

```python
# 在 scheduler.py 中修改 _process_and_broadcast_mm_inputs 方法

def _process_and_broadcast_mm_inputs(
    self,
    raw_mm_inputs: Optional[dict],
):
    """Materialize MultimodalInputs once on the entry rank and broadcast to others.
    
    OPTIMIZATION: Only broadcast raw dict, let each rank materialize independently.
    This avoids serializing large numpy arrays/PIL.Image objects.
    
    Entry rank:
    - broadcasts raw_mm_inputs (dict) to other ranks
    - then materializes locally
    
    Non-entry ranks:
    - receive raw_mm_inputs (dict) via broadcast
    - then materialize locally
    
    All ranks execute from_dict in parallel, avoiding serialization overhead.
    """
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

    # OPTIMIZATION: Broadcast raw dict instead of materialized object
    # This avoids serializing large numpy arrays/PIL.Image objects.
    # The dict contains base64-encoded strings which are much smaller.
    if group_world_size > 1:
        if self.is_entry_rank:
            # Entry rank: broadcast raw dict
            obj_list = [raw_mm_inputs]
            torch.distributed.broadcast_object_list(
                obj_list, src=0, group=self.cpu_group
            )
            # Use broadcasted dict (for consistency)
            raw_mm_inputs = obj_list[0]
        else:
            # Non-entry ranks: receive dict
            obj_list = [None]
            torch.distributed.broadcast_object_list(
                obj_list, src=0, group=self.cpu_group
            )
            raw_mm_inputs = obj_list[0]
    
    # All ranks execute from_dict in parallel
    # This is faster than serializing/deserializing materialized objects
    image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
    
    return image_inputs
```

### 关键改动说明

**修改前**：
```python
if self.is_entry_rank:
    image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)  # 物化
    obj_list = [image_inputs]  # 序列化物化对象（大）
    torch.distributed.broadcast_object_list(...)
else:
    obj_list = [None]
    torch.distributed.broadcast_object_list(...)  # 接收物化对象（大）
    image_inputs = obj_list[0]
```

**修改后**：
```python
if group_world_size > 1:
    if self.is_entry_rank:
        obj_list = [raw_mm_inputs]  # 只广播dict（小）
        torch.distributed.broadcast_object_list(...)
        raw_mm_inputs = obj_list[0]
    else:
        obj_list = [None]
        torch.distributed.broadcast_object_list(...)  # 接收dict（小）
        raw_mm_inputs = obj_list[0]

# 所有ranks并行执行from_dict
image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
```

---

## 方案1实现：异步通信（如果需要进一步优化）

### 添加异步广播器类

```python
# 在 scheduler.py 文件开头添加导入
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional

# 添加异步广播器类（可以在 scheduler.py 中，或者单独文件）
class AsyncMMInputsBroadcaster:
    """异步处理多模态输入的广播，避免阻塞主线程"""
    
    def __init__(self, cpu_group, is_entry_rank: bool, max_workers: int = 2):
        self.cpu_group = cpu_group
        self.is_entry_rank = is_entry_rank
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, 
            thread_name_prefix="MMBroadcast"
        )
        self.pending_futures: Dict[str, Future] = {}
        self.lock = threading.Lock()
        
    def submit_broadcast_task(
        self, 
        req_id: str, 
        raw_mm_inputs: Optional[dict]
    ) -> Future:
        """提交广播任务，返回Future对象"""
        if raw_mm_inputs is None:
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
        """在后台线程中执行广播和物化"""
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
            
            # 使用延迟物化策略：只广播dict
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
            
            # 各rank并行执行from_dict
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

### 修改 Scheduler 类

```python
class Scheduler:
    def __init__(self, ...):
        # ... 现有初始化代码 ...
        
        # 初始化异步广播器
        self.async_mm_broadcaster = AsyncMMInputsBroadcaster(
            cpu_group=self.cpu_group,
            is_entry_rank=self.is_entry_rank,
            max_workers=2
        )
    
    def _process_and_broadcast_mm_inputs(
        self,
        raw_mm_inputs: Optional[dict],
        req_id: Optional[str] = None,
        use_async: bool = True,  # 新增参数：是否使用异步
    ):
        """处理多模态输入广播（支持同步和异步模式）"""
        if raw_mm_inputs is None:
            return None
        
        if use_async and req_id is not None:
            # 异步模式：提交任务，返回Future
            return self.async_mm_broadcaster.submit_broadcast_task(
                req_id, raw_mm_inputs
            )
        else:
            # 同步模式：立即执行（fallback）
            return self._process_and_broadcast_mm_inputs_sync(raw_mm_inputs)
    
    def _process_and_broadcast_mm_inputs_sync(
        self,
        raw_mm_inputs: Optional[dict],
    ):
        """同步处理（延迟物化策略）"""
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

        # 只广播原始dict
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
    
    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        # ... 前面的代码保持不变 ...
        
        # Handle multimodal inputs - 异步方式
        if recv_req.mm_inputs is not None:
            # 提交异步任务
            mm_inputs_result = self._process_and_broadcast_mm_inputs(
                recv_req.mm_inputs, 
                recv_req.rid,
                use_async=True
            )
            
            # 如果返回Future，等待结果
            if isinstance(mm_inputs_result, Future):
                image_inputs = mm_inputs_result.result()
            else:
                image_inputs = mm_inputs_result
            
            # ... 后续处理保持不变 ...
    
    def abort_request(self, req: Req):
        # ... 现有代码 ...
        
        # 取消pending的广播任务
        if hasattr(self, 'async_mm_broadcaster'):
            self.async_mm_broadcaster.cancel_pending(req.rid)
        
        # ... 其他abort逻辑 ...
```

---

## 测试建议

### 1. 性能测试

```python
# 测试脚本示例
import time
import statistics

def benchmark_mm_inputs_processing():
    """对比修改前后的性能"""
    
    # 准备测试数据（2MB视频文件）
    raw_mm_inputs = create_test_mm_inputs(size_mb=2)
    
    # 测试1：原始方案（物化后广播）
    times_original = []
    for _ in range(10):
        start = time.perf_counter()
        # 原始实现
        image_inputs = original_process_and_broadcast(raw_mm_inputs)
        times_original.append(time.perf_counter() - start)
    
    # 测试2：延迟物化方案
    times_delayed = []
    for _ in range(10):
        start = time.perf_counter()
        # 新实现
        image_inputs = delayed_materialization_process(raw_mm_inputs)
        times_delayed.append(time.perf_counter() - start)
    
    print(f"Original: mean={statistics.mean(times_original):.3f}s, "
          f"p99={statistics.quantiles(times_original, n=100)[98]:.3f}s")
    print(f"Delayed:  mean={statistics.mean(times_delayed):.3f}s, "
          f"p99={statistics.quantiles(times_delayed, n=100)[98]:.3f}s")
```

### 2. 并发测试

```python
# 模拟高并发场景
def test_concurrent_requests():
    """测试高并发下的QPS"""
    
    # 模拟100个并发请求
    requests = [create_test_request() for _ in range(100)]
    
    # 测试原始方案
    start = time.perf_counter()
    for req in requests:
        original_handle_request(req)
    time_original = time.perf_counter() - start
    
    # 测试新方案
    start = time.perf_counter()
    for req in requests:
        new_handle_request(req)
    time_new = time.perf_counter() - start
    
    qps_original = len(requests) / time_original
    qps_new = len(requests) / time_new
    
    print(f"QPS improvement: {qps_new / qps_original:.2f}x")
```

### 3. 监控指标

- **广播时间**：dict广播 vs 物化对象广播
- **from_dict时间**：各rank的执行时间
- **CPU占用率**：是否降低
- **QPS**：整体吞吐量
- **延迟**：p50/p99延迟

---

## 部署建议

### 阶段1：实施方案2（延迟物化）

1. **修改代码**：按照方案2的实现修改 `_process_and_broadcast_mm_inputs`
2. **测试验证**：
   - 单元测试：确保功能正确
   - 性能测试：对比QPS和延迟
3. **灰度发布**：先在小规模环境验证
4. **监控观察**：观察QPS、CPU、延迟等指标

### 阶段2（如需要）：实施方案1（异步通信）

如果方案2后仍有阻塞问题，再实施方案1：
1. 添加 `AsyncMMInputsBroadcaster` 类
2. 修改 `Scheduler` 类集成异步广播器
3. 添加配置开关，可以动态切换同步/异步模式
4. 测试验证后逐步启用

### 回滚方案

保留原始代码作为fallback：
```python
def _process_and_broadcast_mm_inputs(
    self,
    raw_mm_inputs: Optional[dict],
    use_optimization: bool = True,  # 配置开关
):
    if not use_optimization:
        # 回退到原始实现
        return self._process_and_broadcast_mm_inputs_original(raw_mm_inputs)
    
    # 新实现...
```
