# 平衡方案：保留PR优化 + 解决QPS问题

## 问题重新理解

### PR的优化目标（需要保留）
- **减少重复计算**：`from_dict` 包含解码base64、size检查、normalization等，CPU开销大
- **降低CPU占用**：对于2MB视频文件，from_dict需要~500ms CPU时间
- **避免CPU-overload**：在TP8场景下，如果所有ranks都执行from_dict，CPU占用会很高

### PR引入的问题
- **序列化开销巨大**：broadcast物化后的对象（包含numpy数组/PIL.Image）需要数百毫秒
- **同步阻塞**：broadcast_object_list是同步的，阻塞主线程
- **高并发QPS下降**：阻塞导致吞吐量下降

### 关键发现
- 各rank已经通过 `recv_requests()` 的 `broadcast_pyobj` 收到了 `raw_mm_inputs`（dict）

## 最佳方案：条件优化 + 异步通信

### 核心思路

1. **小文件**：各rank并行执行from_dict（避免broadcast开销）
2. **大文件**：只在entry rank执行from_dict，异步broadcast dict（保留PR优化，避免重复计算）
3. **使用异步通信**：避免阻塞主线程

### 实现方案

```python
def _process_and_broadcast_mm_inputs(
    self,
    raw_mm_inputs: Optional[dict],
    req_id: Optional[str] = None,
    use_async: bool = True,
):
    """平衡方案：根据数据大小选择策略
    
    - 小文件：各rank并行from_dict（避免broadcast开销）
    - 大文件：entry rank from_dict + 异步broadcast dict（避免重复计算）
    """
    if raw_mm_inputs is None:
        return None

    # 估算数据大小
    estimated_size = self._estimate_mm_inputs_size(raw_mm_inputs)
    size_threshold = 1 * 1024 * 1024  # 1MB阈值，可配置
    
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
    
    # 策略选择
    if group_world_size == 1:
        # 单rank：直接物化
        return MultimodalInputs.from_dict(raw_mm_inputs)
    
    if estimated_size <= size_threshold:
        # 小文件：各rank并行from_dict（避免broadcast开销）
        # 虽然有小量重复计算，但broadcast的开销更小
        return MultimodalInputs.from_dict(raw_mm_inputs)
    else:
        # 大文件：保留PR优化，避免重复计算
        if use_async and req_id is not None:
            # 异步模式：后台线程处理
            return self._async_process_large_mm_inputs(
                raw_mm_inputs, req_id, group_world_size
            )
        else:
            # 同步模式：entry rank物化，broadcast dict
            return self._sync_process_large_mm_inputs(
                raw_mm_inputs, group_world_size
            )

def _sync_process_large_mm_inputs(
    self,
    raw_mm_inputs: dict,
    group_world_size: int,
):
    """同步处理大文件：entry rank物化，broadcast dict"""
    if self.is_entry_rank:
        # Entry rank: 物化一次
        image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
        
        # Broadcast原始dict（小体积），让其他ranks也能物化
        # 注意：这里broadcast dict是为了让其他ranks也能物化（如果需要）
        # 但实际上，如果entry rank已经物化了，其他ranks可以直接使用
        # 所以这里可以选择：broadcast dict 或 broadcast物化对象
        
        # 选项1：broadcast dict（避免序列化大型对象）
        if group_world_size > 1:
            obj_list = [raw_mm_inputs]
            torch.distributed.broadcast_object_list(
                obj_list, src=0, group=self.cpu_group
            )
        return image_inputs
    else:
        # Non-entry ranks: 接收dict后物化
        if group_world_size > 1:
            obj_list = [None]
            torch.distributed.broadcast_object_list(
                obj_list, src=0, group=self.cpu_group
            )
            raw_mm_inputs = obj_list[0]
        return MultimodalInputs.from_dict(raw_mm_inputs)

def _async_process_large_mm_inputs(
    self,
    raw_mm_inputs: dict,
    req_id: str,
    group_world_size: int,
):
    """异步处理大文件：后台线程处理，避免阻塞"""
    if not hasattr(self, 'async_mm_broadcaster'):
        # Fallback to sync if async broadcaster not initialized
        return self._sync_process_large_mm_inputs(raw_mm_inputs, group_world_size)
    
    future = self.async_mm_broadcaster.submit_task(
        req_id, raw_mm_inputs, group_world_size, self.is_entry_rank
    )
    return future

def _estimate_mm_inputs_size(self, raw_mm_inputs: dict) -> int:
    """估算mm_inputs的大小（字节）"""
    import sys
    try:
        # 简单估算：检查mm_items中的数据大小
        total_size = 0
        if "mm_items" in raw_mm_inputs:
            for item in raw_mm_inputs["mm_items"]:
                if isinstance(item, dict):
                    # 估算base64编码数据的大小
                    for key, value in item.items():
                        if isinstance(value, (str, bytes)):
                            total_size += len(value) if isinstance(value, bytes) else len(value.encode())
                        elif isinstance(value, list):
                            total_size += sum(
                                len(v) if isinstance(v, (str, bytes)) 
                                else len(str(v).encode()) 
                                for v in value
                            )
        return total_size
    except:
        # Fallback: 使用pickle估算
        import pickle
        try:
            return len(pickle.dumps(raw_mm_inputs))
        except:
            return 0
```

### 异步广播器实现

```python
class AsyncMMInputsBroadcaster:
    """异步处理大文件的多模态输入，避免阻塞主线程"""
    
    def __init__(self, cpu_group, is_entry_rank: bool, max_workers: int = 2):
        self.cpu_group = cpu_group
        self.is_entry_rank = is_entry_rank
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="MMBroadcast"
        )
        self.pending_futures: Dict[str, Future] = {}
        self.lock = threading.Lock()
    
    def submit_task(
        self,
        req_id: str,
        raw_mm_inputs: dict,
        group_world_size: int,
        is_entry_rank: bool,
    ) -> Future:
        """提交异步任务"""
        future = Future()
        
        with self.lock:
            self.pending_futures[req_id] = future
        
        self.executor.submit(
            self._process_large_mm_inputs_async,
            req_id,
            raw_mm_inputs,
            group_world_size,
            is_entry_rank,
            future
        )
        
        return future
    
    def _process_large_mm_inputs_async(
        self,
        req_id: str,
        raw_mm_inputs: dict,
        group_world_size: int,
        is_entry_rank: bool,
        future: Future,
    ):
        """在后台线程中处理大文件"""
        try:
            if is_entry_rank:
                # Entry rank: 物化一次
                image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
                
                # Broadcast dict（小体积）
                if group_world_size > 1:
                    obj_list = [raw_mm_inputs]
                    torch.distributed.broadcast_object_list(
                        obj_list, src=0, group=self.cpu_group
                    )
                future.set_result(image_inputs)
            else:
                # Non-entry ranks: 接收dict后物化
                if group_world_size > 1:
                    obj_list = [None]
                    torch.distributed.broadcast_object_list(
                        obj_list, src=0, group=self.cpu_group
                    )
                    raw_mm_inputs = obj_list[0]
                image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
                future.set_result(image_inputs)
        except Exception as e:
            future.set_exception(e)
        finally:
            with self.lock:
                self.pending_futures.pop(req_id, None)
    
    def cancel_pending(self, req_id: str):
        """取消pending的任务"""
        with self.lock:
            future = self.pending_futures.pop(req_id, None)
            if future and not future.done():
                future.cancel()
    
    def shutdown(self):
        """关闭线程池"""
        self.executor.shutdown(wait=True)
```

## 方案对比

| 方案 | CPU占用 | 序列化开销 | 阻塞问题 | QPS | 实现复杂度 |
|------|---------|-----------|---------|-----|-----------|
| **完全revert** | 高（所有rank重复计算） | 无 | 无 | 高 | 低 |
| **PR原方案** | 低（只entry rank计算） | 高（序列化物化对象） | 有（同步阻塞） | 低 | 中 |
| **平衡方案** | 低（大文件只entry rank） | 低（broadcast dict） | 无（异步） | 高 | 中 |

## 优势

1. **保留PR优化**：大文件时避免重复计算，降低CPU占用
2. **解决QPS问题**：使用异步通信，避免阻塞主线程
3. **智能选择**：根据数据大小自动选择最优策略
4. **向后兼容**：可以配置阈值和是否使用异步

## 配置建议

```python
# 在Scheduler初始化时
self.mm_inputs_size_threshold = 1 * 1024 * 1024  # 1MB
self.enable_async_mm_broadcast = True  # 启用异步
```

## 实施步骤

1. **第一阶段**：实现条件优化（同步版本）
   - 小文件：各rank并行from_dict
   - 大文件：entry rank from_dict + broadcast dict
   - 验证CPU占用和QPS

2. **第二阶段**：添加异步支持
   - 实现AsyncMMInputsBroadcaster
   - 大文件使用异步处理
   - 进一步优化QPS
