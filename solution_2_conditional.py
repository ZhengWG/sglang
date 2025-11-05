"""
解决方案2: 条件优化 - 根据数据大小动态选择策略

核心思路:
1. 小数据(<100KB): 直接本地执行 from_dict，避免广播开销
2. 大数据(>=100KB): 使用广播优化，避免重复计算
3. 单卡模式: 始终本地执行
"""

from typing import Optional
import sys
import torch
import torch.distributed


def _estimate_mm_inputs_size(raw_mm_inputs: dict) -> int:
    """
    快速估算 mm_inputs 序列化后的大小
    
    Args:
        raw_mm_inputs: 原始的mm_inputs字典
        
    Returns:
        估算的字节大小
    """
    if not raw_mm_inputs:
        return 0
    
    total_size = 0
    
    # 估算 mm_items 的大小
    if 'mm_items' in raw_mm_inputs:
        mm_items = raw_mm_inputs['mm_items']
        if isinstance(mm_items, list):
            for item in mm_items:
                if not isinstance(item, dict):
                    continue
                    
                # 估算 feature 字段的大小（通常是最大的）
                if 'feature' in item:
                    feature = item['feature']
                    
                    if isinstance(feature, torch.Tensor):
                        # Tensor: numel * element_size
                        total_size += feature.numel() * feature.element_size()
                    elif isinstance(feature, list):
                        # 列表：估算为字符串长度（粗略估计）
                        total_size += sys.getsizeof(feature)
                    elif hasattr(feature, '__len__'):
                        # numpy array或其他有长度的对象
                        try:
                            total_size += sys.getsizeof(feature)
                        except:
                            total_size += 1024  # fallback
                
                # 估算 offsets 的大小
                if 'offsets' in item and isinstance(item['offsets'], list):
                    total_size += len(item['offsets']) * 8  # 假设每个offset是8字节
    
    # 其他字段的基础大小（通常很小）
    total_size += 1024  # 元数据开销
    
    return total_size


def _process_and_broadcast_mm_inputs(
    self,
    raw_mm_inputs: Optional[dict],
):
    """
    优化版本：根据数据大小动态选择处理策略
    
    策略选择:
    1. raw_mm_inputs is None -> 直接返回 None
    2. tp_size == 1 (单卡) -> 本地执行 from_dict
    3. 数据大小 < 阈值 -> 本地执行 from_dict (避免pickle开销)
    4. 数据大小 >= 阈值 -> 使用广播优化 (避免重复计算)
    
    Args:
        raw_mm_inputs: 原始的mm_inputs字典
        
    Returns:
        MultimodalInputs对象或None
    """
    from sglang.srt.managers.schedule_batch import MultimodalInputs
    
    if raw_mm_inputs is None:
        return None
    
    # 单卡模式：直接本地执行
    if self.tp_size == 1:
        return MultimodalInputs.from_dict(raw_mm_inputs)
    
    # 估算数据大小
    estimated_size = _estimate_mm_inputs_size(raw_mm_inputs)
    
    # 阈值配置（可通过环境变量调整）
    import os
    SIZE_THRESHOLD = int(os.environ.get(
        'SGLANG_MM_BROADCAST_THRESHOLD', 
        100 * 1024  # 默认 100KB
    ))
    
    # 小数据：本地执行更快（避免pickle + 网络 + unpickle的开销）
    if estimated_size < SIZE_THRESHOLD:
        return MultimodalInputs.from_dict(raw_mm_inputs)
    
    # 大数据：使用广播优化（避免重复计算）
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
        # 发生异常时fallback到本地执行
        from sglang.srt.logger import logger
        logger.warning(
            f"Failed to get world size in mm_inputs handling: {e}, "
            f"fallback to local processing."
        )
        return MultimodalInputs.from_dict(raw_mm_inputs)
    
    if self.is_entry_rank:
        # Entry rank: 执行 from_dict 并广播
        image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
        
        if group_world_size > 1:
            try:
                obj_list = [image_inputs]
                torch.distributed.broadcast_object_list(
                    obj_list, src=0, group=self.cpu_group
                )
                image_inputs = obj_list[0]
            except Exception as e:
                from sglang.srt.logger import logger
                logger.warning(
                    f"Failed to broadcast mm_inputs: {e}, "
                    f"continue with local result."
                )
    else:
        # Non-entry ranks: 接收广播
        if group_world_size > 1:
            try:
                obj_list = [None]
                torch.distributed.broadcast_object_list(
                    obj_list, src=0, group=self.cpu_group
                )
                image_inputs = obj_list[0]
            except Exception as e:
                from sglang.srt.logger import logger
                logger.warning(
                    f"Failed to receive broadcasted mm_inputs: {e}, "
                    f"fallback to local processing."
                )
                image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
        else:
            # 单进程多线程等特殊情况
            image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
    
    return image_inputs


# 使用说明:
# 1. 将此方法替换 scheduler.py 中的 _process_and_broadcast_mm_inputs 方法
# 2. 确保 _estimate_mm_inputs_size 函数也被添加到 scheduler.py
# 3. 可通过环境变量调整阈值: 
#    export SGLANG_MM_BROADCAST_THRESHOLD=50000  # 50KB
#    export SGLANG_MM_BROADCAST_THRESHOLD=200000 # 200KB
#
# 调优建议:
# - 测试不同阈值下的性能表现
# - 使用 profile 工具找到最优阈值
# - 考虑添加监控指标来跟踪选择的策略分布

"""
性能对比（示例数据）:

数据大小    原始方案        当前方案         条件优化方案
10KB       50μs/req       500μs/req       50μs/req     ✓
100KB      500μs/req      1ms/req         500μs/req    ✓
1MB        5ms/req        3ms/req         3ms/req      ✓
10MB       50ms/req       10ms/req        10ms/req     ✓

说明:
- 小数据: 条件优化=原始方案，避免了广播开销
- 大数据: 条件优化=广播方案，避免了重复计算
- 在高并发下，小数据场景的改善最为显著
"""
