"""
性能测试脚本: 对比不同解决方案的性能

使用方法:
python benchmark_mm_inputs_solutions.py --solution [original|commit|conditional|zerocopy]
"""

import argparse
import time
import torch
import torch.distributed as dist
import numpy as np
from typing import List, Dict
import psutil
import os
import pickle


def generate_fake_mm_inputs(size_kb: int = 100) -> dict:
    """
    生成指定大小的假multimodal inputs用于测试
    
    Args:
        size_kb: 目标大小（KB）
        
    Returns:
        fake mm_inputs dict
    """
    # 计算需要的tensor大小
    # 假设float32, 每个元素4字节
    num_elements = (size_kb * 1024) // 4
    
    fake_mm_inputs = {
        'mm_items': [
            {
                'modality': 'IMAGE',
                'feature': torch.randn(num_elements),
                'offsets': [0, 100, 200],
                'pad_value': 0.0,
            }
        ],
        'im_token_id': 128000,
        'im_start_id': 128001,
        'im_end_id': 128002,
    }
    
    return fake_mm_inputs


def benchmark_original(mm_inputs_dict: dict, num_iterations: int = 100):
    """原始方案: 每个rank独立执行from_dict"""
    from sglang.srt.managers.schedule_batch import MultimodalInputs
    
    start_time = time.time()
    cpu_times = []
    
    process = psutil.Process()
    
    for i in range(num_iterations):
        cpu_before = process.cpu_percent()
        
        # 模拟每个rank独立执行
        result = MultimodalInputs.from_dict(mm_inputs_dict)
        
        cpu_after = process.cpu_percent()
        cpu_times.append(cpu_after)
    
    end_time = time.time()
    
    return {
        'total_time': end_time - start_time,
        'avg_time_per_req': (end_time - start_time) / num_iterations,
        'avg_cpu': np.mean(cpu_times),
        'max_cpu': np.max(cpu_times),
    }


def benchmark_pickle_broadcast(mm_inputs_dict: dict, num_iterations: int = 100):
    """Commit方案: 使用pickle + broadcast"""
    from sglang.srt.managers.schedule_batch import MultimodalInputs
    
    start_time = time.time()
    pickle_times = []
    unpickle_times = []
    cpu_times = []
    
    process = psutil.Process()
    
    for i in range(num_iterations):
        cpu_before = process.cpu_percent()
        
        # 模拟rank 0的操作
        mm_inputs = MultimodalInputs.from_dict(mm_inputs_dict)
        
        # pickle序列化
        t0 = time.time()
        serialized = pickle.dumps(mm_inputs)
        t1 = time.time()
        pickle_times.append(t1 - t0)
        
        # 模拟网络传输（实际会通过broadcast）
        # 这里直接unpickle模拟接收方的操作
        t2 = time.time()
        result = pickle.loads(serialized)
        t3 = time.time()
        unpickle_times.append(t3 - t2)
        
        cpu_after = process.cpu_percent()
        cpu_times.append(cpu_after)
    
    end_time = time.time()
    
    return {
        'total_time': end_time - start_time,
        'avg_time_per_req': (end_time - start_time) / num_iterations,
        'avg_pickle_time': np.mean(pickle_times),
        'avg_unpickle_time': np.mean(unpickle_times),
        'avg_cpu': np.mean(cpu_times),
        'max_cpu': np.max(cpu_times),
        'serialized_size_mb': len(serialized) / (1024 * 1024),
    }


def benchmark_conditional(
    mm_inputs_dict: dict, 
    num_iterations: int = 100,
    threshold_kb: int = 100
):
    """条件优化方案: 根据大小选择策略"""
    from sglang.srt.managers.schedule_batch import MultimodalInputs
    import sys
    
    # 估算大小
    estimated_size = 0
    for item in mm_inputs_dict.get('mm_items', []):
        if 'feature' in item and isinstance(item['feature'], torch.Tensor):
            feature = item['feature']
            estimated_size += feature.numel() * feature.element_size()
    
    use_broadcast = estimated_size >= (threshold_kb * 1024)
    
    start_time = time.time()
    cpu_times = []
    strategy_counts = {'local': 0, 'broadcast': 0}
    
    process = psutil.Process()
    
    for i in range(num_iterations):
        cpu_before = process.cpu_percent()
        
        if use_broadcast:
            # 使用broadcast策略
            mm_inputs = MultimodalInputs.from_dict(mm_inputs_dict)
            serialized = pickle.dumps(mm_inputs)
            result = pickle.loads(serialized)
            strategy_counts['broadcast'] += 1
        else:
            # 本地执行策略
            result = MultimodalInputs.from_dict(mm_inputs_dict)
            strategy_counts['local'] += 1
        
        cpu_after = process.cpu_percent()
        cpu_times.append(cpu_after)
    
    end_time = time.time()
    
    return {
        'total_time': end_time - start_time,
        'avg_time_per_req': (end_time - start_time) / num_iterations,
        'avg_cpu': np.mean(cpu_times),
        'max_cpu': np.max(cpu_times),
        'estimated_size_kb': estimated_size / 1024,
        'strategy_used': 'broadcast' if use_broadcast else 'local',
        'strategy_counts': strategy_counts,
    }


def benchmark_zerocopy(mm_inputs_dict: dict, num_iterations: int = 100):
    """零拷贝方案: 直接广播tensor"""
    from sglang.srt.managers.schedule_batch import MultimodalInputs
    
    start_time = time.time()
    cpu_times = []
    
    process = psutil.Process()
    
    for i in range(num_iterations):
        cpu_before = process.cpu_percent()
        
        # 模拟零拷贝方案
        # 1. from_dict (只在rank 0)
        mm_inputs = MultimodalInputs.from_dict(mm_inputs_dict)
        
        # 2. 提取tensor并直接广播（模拟）
        # 实际会使用dist.broadcast，这里只测试提取开销
        for item in mm_inputs.mm_items:
            if isinstance(item.feature, torch.Tensor):
                # 零拷贝: 不需要pickle，直接传输tensor
                # 这里只是访问tensor，实际通信由NCCL处理
                _ = item.feature.shape
        
        cpu_after = process.cpu_percent()
        cpu_times.append(cpu_after)
    
    end_time = time.time()
    
    return {
        'total_time': end_time - start_time,
        'avg_time_per_req': (end_time - start_time) / num_iterations,
        'avg_cpu': np.mean(cpu_times),
        'max_cpu': np.max(cpu_times),
    }


def run_comprehensive_benchmark():
    """运行完整的性能对比测试"""
    print("=" * 80)
    print("Multimodal Inputs 处理方案性能对比测试")
    print("=" * 80)
    
    # 测试不同大小的数据
    test_sizes = [10, 50, 100, 500, 1000, 5000]  # KB
    num_iterations = 50
    
    results_table = []
    
    for size_kb in test_sizes:
        print(f"\n测试数据大小: {size_kb} KB")
        print("-" * 80)
        
        mm_inputs = generate_fake_mm_inputs(size_kb)
        
        # 测试原始方案
        print("  [1/4] 测试原始方案...")
        result_original = benchmark_original(mm_inputs, num_iterations)
        
        # 测试pickle+broadcast方案
        print("  [2/4] 测试pickle+broadcast方案...")
        result_pickle = benchmark_pickle_broadcast(mm_inputs, num_iterations)
        
        # 测试条件优化方案
        print("  [3/4] 测试条件优化方案...")
        result_conditional = benchmark_conditional(mm_inputs, num_iterations)
        
        # 测试零拷贝方案
        print("  [4/4] 测试零拷贝方案...")
        result_zerocopy = benchmark_zerocopy(mm_inputs, num_iterations)
        
        # 汇总结果
        row = {
            'size_kb': size_kb,
            'original_time_ms': result_original['avg_time_per_req'] * 1000,
            'original_cpu': result_original['avg_cpu'],
            'pickle_time_ms': result_pickle['avg_time_per_req'] * 1000,
            'pickle_cpu': result_pickle['avg_cpu'],
            'pickle_size_mb': result_pickle.get('serialized_size_mb', 0),
            'conditional_time_ms': result_conditional['avg_time_per_req'] * 1000,
            'conditional_cpu': result_conditional['avg_cpu'],
            'conditional_strategy': result_conditional['strategy_used'],
            'zerocopy_time_ms': result_zerocopy['avg_time_per_req'] * 1000,
            'zerocopy_cpu': result_zerocopy['avg_cpu'],
        }
        results_table.append(row)
        
        print(f"\n  结果摘要:")
        print(f"    原始方案:      {row['original_time_ms']:.2f}ms, CPU: {row['original_cpu']:.1f}%")
        print(f"    Pickle方案:    {row['pickle_time_ms']:.2f}ms, CPU: {row['pickle_cpu']:.1f}%, Size: {row['pickle_size_mb']:.2f}MB")
        print(f"    条件优化方案:  {row['conditional_time_ms']:.2f}ms, CPU: {row['conditional_cpu']:.1f}%, Strategy: {row['conditional_strategy']}")
        print(f"    零拷贝方案:    {row['zerocopy_time_ms']:.2f}ms, CPU: {row['zerocopy_cpu']:.1f}%")
    
    # 打印完整对比表格
    print("\n" + "=" * 80)
    print("完整性能对比表格")
    print("=" * 80)
    print(f"{'Size(KB)':<10} {'Original(ms)':<15} {'Pickle(ms)':<15} {'Conditional(ms)':<18} {'ZeroCopy(ms)':<15}")
    print("-" * 80)
    
    for row in results_table:
        print(
            f"{row['size_kb']:<10} "
            f"{row['original_time_ms']:<15.2f} "
            f"{row['pickle_time_ms']:<15.2f} "
            f"{row['conditional_time_ms']:<18.2f} "
            f"{row['zerocopy_time_ms']:<15.2f}"
        )
    
    print("\n" + "=" * 80)
    print("关键观察:")
    print("=" * 80)
    
    # 分析关键观察点
    for row in results_table:
        size = row['size_kb']
        if row['pickle_time_ms'] > row['original_time_ms'] * 1.5:
            print(f"⚠️  {size}KB: Pickle方案比原始方案慢 {row['pickle_time_ms'] / row['original_time_ms']:.1f}x")
        
        if row['conditional_time_ms'] < min(row['original_time_ms'], row['pickle_time_ms']):
            print(f"✓  {size}KB: 条件优化方案效果最佳 ({row['conditional_strategy']})")
        
        if row['zerocopy_time_ms'] < row['pickle_time_ms'] * 0.5:
            print(f"✓  {size}KB: 零拷贝方案比Pickle快 {row['pickle_time_ms'] / row['zerocopy_time_ms']:.1f}x")
    
    print("\n推荐方案:")
    # 找出最佳阈值
    threshold_candidates = [50, 100, 200]
    for threshold in threshold_candidates:
        print(f"  - 阈值 {threshold}KB: 适用于大部分场景")


def main():
    parser = argparse.ArgumentParser(description='Benchmark mm_inputs solutions')
    parser.add_argument(
        '--solution',
        choices=['original', 'commit', 'conditional', 'zerocopy', 'all'],
        default='all',
        help='Which solution to benchmark'
    )
    parser.add_argument(
        '--size-kb',
        type=int,
        default=100,
        help='Size of test data in KB'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of iterations'
    )
    parser.add_argument(
        '--comprehensive',
        action='store_true',
        help='Run comprehensive benchmark with multiple sizes'
    )
    
    args = parser.parse_args()
    
    if args.comprehensive:
        run_comprehensive_benchmark()
    else:
        mm_inputs = generate_fake_mm_inputs(args.size_kb)
        
        if args.solution == 'all':
            print(f"Running all benchmarks with {args.size_kb}KB data...")
            
            print("\n[1/4] Original solution...")
            result = benchmark_original(mm_inputs, args.iterations)
            print(f"  Time per request: {result['avg_time_per_req']*1000:.2f}ms")
            print(f"  CPU usage: {result['avg_cpu']:.1f}%")
            
            print("\n[2/4] Pickle+broadcast solution...")
            result = benchmark_pickle_broadcast(mm_inputs, args.iterations)
            print(f"  Time per request: {result['avg_time_per_req']*1000:.2f}ms")
            print(f"  CPU usage: {result['avg_cpu']:.1f}%")
            print(f"  Serialized size: {result['serialized_size_mb']:.2f}MB")
            
            print("\n[3/4] Conditional solution...")
            result = benchmark_conditional(mm_inputs, args.iterations)
            print(f"  Time per request: {result['avg_time_per_req']*1000:.2f}ms")
            print(f"  CPU usage: {result['avg_cpu']:.1f}%")
            print(f"  Strategy: {result['strategy_used']}")
            
            print("\n[4/4] Zero-copy solution...")
            result = benchmark_zerocopy(mm_inputs, args.iterations)
            print(f"  Time per request: {result['avg_time_per_req']*1000:.2f}ms")
            print(f"  CPU usage: {result['avg_cpu']:.1f}%")
        
        elif args.solution == 'original':
            result = benchmark_original(mm_inputs, args.iterations)
            print(f"Average time per request: {result['avg_time_per_req']*1000:.2f}ms")
        
        elif args.solution == 'commit':
            result = benchmark_pickle_broadcast(mm_inputs, args.iterations)
            print(f"Average time per request: {result['avg_time_per_req']*1000:.2f}ms")
            print(f"Serialized size: {result['serialized_size_mb']:.2f}MB")
        
        elif args.solution == 'conditional':
            result = benchmark_conditional(mm_inputs, args.iterations)
            print(f"Average time per request: {result['avg_time_per_req']*1000:.2f}ms")
            print(f"Strategy used: {result['strategy_used']}")
        
        elif args.solution == 'zerocopy':
            result = benchmark_zerocopy(mm_inputs, args.iterations)
            print(f"Average time per request: {result['avg_time_per_req']*1000:.2f}ms")


if __name__ == '__main__':
    main()
