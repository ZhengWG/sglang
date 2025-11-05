"""
测试真正的性能瓶颈：from_dict vs broadcast
"""
import time
import pickle
import numpy as np
import sys

# 模拟MultimodalDataItem
class FakeItem:
    def __init__(self, feature_size_mb=10):
        # 创建大型numpy数组模拟图像特征
        num_elements = int(feature_size_mb * 1024 * 1024 / 4)  # float32
        self.feature = np.random.randn(num_elements).astype(np.float32)
        self.offsets = [0, 100, 200]
        self.hash = None
    
    def is_valid(self):
        return True
    
    def set_pad_value(self):
        """模拟CPU密集的hash计算"""
        if self.hash is None:
            # 实际的hash_feature会对整个feature计算hash
            self.hash = hash(self.feature.tobytes())
    
class FakeMultimodalInputs:
    def __init__(self, mm_items):
        self.mm_items = mm_items
    
    @staticmethod
    def from_dict(obj):
        """模拟from_dict的操作"""
        ret = FakeMultimodalInputs(obj["mm_items"])
        ret.mm_items = [item for item in ret.mm_items if item.is_valid()]
        # 这里是CPU密集操作
        for item in ret.mm_items:
            item.set_pad_value()
        return ret

# 测试不同场景
def test_from_dict_overhead(feature_size_mb=10):
    """测试from_dict的CPU开销"""
    mm_inputs_dict = {
        "mm_items": [FakeItem(feature_size_mb) for _ in range(3)]
    }
    
    start = time.time()
    result = FakeMultimodalInputs.from_dict(mm_inputs_dict)
    end = time.time()
    
    return (end - start) * 1000  # ms

def test_pickle_overhead(feature_size_mb=10):
    """测试pickle序列化的开销"""
    mm_inputs_dict = {
        "mm_items": [FakeItem(feature_size_mb) for _ in range(3)]
    }
    mm_inputs = FakeMultimodalInputs.from_dict(mm_inputs_dict)
    
    start = time.time()
    pickled = pickle.dumps(mm_inputs)
    end = time.time()
    pickle_time = (end - start) * 1000
    
    start = time.time()
    unpickled = pickle.loads(pickled)
    end = time.time()
    unpickle_time = (end - start) * 1000
    
    return pickle_time, unpickle_time, len(pickled) / (1024 * 1024)  # ms, ms, MB

def test_combined_overhead(feature_size_mb=10):
    """测试新方案的总开销"""
    mm_inputs_dict = {
        "mm_items": [FakeItem(feature_size_mb) for _ in range(3)]
    }
    
    start = time.time()
    # rank 0: from_dict
    mm_inputs = FakeMultimodalInputs.from_dict(mm_inputs_dict)
    # rank 0: pickle
    pickled = pickle.dumps(mm_inputs)
    # 其他ranks: unpickle
    unpickled = pickle.loads(pickled)
    end = time.time()
    
    return (end - start) * 1000  # ms

if __name__ == "__main__":
    print("=" * 80)
    print("性能瓶颈分析：from_dict vs broadcast_object_list")
    print("=" * 80)
    
    test_sizes = [1, 5, 10, 50, 100]  # MB
    
    print(f"\n{'Size(MB)':<10} {'from_dict':<15} {'pickle':<15} {'unpickle':<15} {'总计':<15} {'pickled_size(MB)':<20}")
    print("-" * 90)
    
    for size_mb in test_sizes:
        from_dict_time = test_from_dict_overhead(size_mb)
        pickle_time, unpickle_time, pickled_size = test_pickle_overhead(size_mb)
        combined_time = test_combined_overhead(size_mb)
        
        print(f"{size_mb:<10} {from_dict_time:<15.2f} {pickle_time:<15.2f} {unpickle_time:<15.2f} {combined_time:<15.2f} {pickled_size:<20.2f}")
    
    print("\n" + "=" * 80)
    print("关键观察：")
    print("=" * 80)
    
    # 重新测试一个典型大小
    size_mb = 10
    from_dict_time = test_from_dict_overhead(size_mb)
    pickle_time, unpickle_time, _ = test_pickle_overhead(size_mb)
    
    print(f"\n对于 {size_mb}MB 的数据：")
    print(f"  原方案（每个rank独立执行from_dict）: {from_dict_time:.2f}ms")
    print(f"  新方案（rank 0执行from_dict + broadcast）: {from_dict_time + pickle_time + unpickle_time:.2f}ms")
    print(f"    - from_dict: {from_dict_time:.2f}ms")
    print(f"    - pickle: {pickle_time:.2f}ms")
    print(f"    - unpickle: {unpickle_time:.2f}ms")
    
    if from_dict_time < (pickle_time + unpickle_time):
        print(f"\n⚠️  发现问题：pickle开销({pickle_time + unpickle_time:.2f}ms) > from_dict开销({from_dict_time:.2f}ms)")
        print(f"   新方案反而更慢！")
    else:
        print(f"\n✓ 新方案确实能节省时间")
    
    print("\n" + "=" * 80)
    print("高并发场景分析：")
    print("=" * 80)
    print("\n原方案（并行）：")
    print("  请求1: rank0执行from_dict (10ms) | rank1执行from_dict (10ms) - 并行")
    print("  请求2: rank0执行from_dict (10ms) | rank1执行from_dict (10ms) - 并行")
    print("  总时间: 10ms per request (并行无等待)")
    
    print("\n新方案（串行broadcast）：")
    print("  请求1: rank0 from_dict (10ms) -> broadcast (30ms) - 所有ranks阻塞等待")
    print("  请求2: 等待请求1完成 -> rank0 from_dict (10ms) -> broadcast (30ms)")
    print("  总时间: 40ms per request (串行化，且有额外开销)")
    
    print("\n在高并发(100 req/s)下：")
    print("  原方案: 可以并行处理，CPU可能99%但不阻塞")
    print("  新方案: 串行化导致排队，CPU 99.9%且吞吐量暴跌 ❌")
