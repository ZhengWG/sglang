# Scatter-Gather Block分配设计

## 🎯 核心改进

**之前的错误**：假设数据连续存储在start_offset位置（还是连续假设）

**现在的正确实现**：
- ✅ 数据真正分散存储在各个blocks中
- ✅ 合并连续blocks优化IO
- ✅ Scatter写入 + Gather读取

---

## 📊 设计对比

### 错误设计（之前）

```python
# ❌ 还是假设连续
allocation = MetadataAllocation(
    block_indices=[8,9,3,4,5],
    start_offset=384  # = min([8,9,3,4,5]) * 128
)

# 写入：数据写到[384:1024)（连续区域）
buffer[384:1024] = data  # ❌ 只用了blocks [3,4,5]，浪费了[8,9]

# 读取：从[384:1024)读取
data = buffer[384:1024]  # ❌ 没有真正利用所有blocks
```

**问题**：
- 只使用了min(block_indices)开始的连续区域
- 其他blocks被浪费
- 还是假设连续性

### 正确设计（现在）

```python
# ✅ 真正的scatter-gather
allocation = MetadataAllocation(
    block_indices=[8,9,3,4,5],  # 乱序
    num_tokens=640
)

# 1. 合并连续blocks
ranges = allocation.get_contiguous_ranges(block_size=128)
# -> [(384, 384), (1024, 256)]
#    ↓             ↓
#    blocks[3,4,5] blocks[8,9]

# 2. Scatter写入：分散到各个range
set_buf():
  buffer[384:768] = data[0:384]      # blocks [3,4,5]
  buffer[1024:1280] = data[384:640]  # blocks [8,9]

# 3. Gather读取：从各个range收集
get_buf():
  chunk1 = buffer[384:768]    # 384 tokens
  chunk2 = buffer[1024:1280]  # 256 tokens
  result = concat([chunk1, chunk2])  # 640 tokens
```

**优势**：
- ✅ 真正利用所有分配的blocks
- ✅ 数据分散存储，无浪费
- ✅ 自动合并连续blocks优化IO

---

## 🔧 核心实现

### 1. 合并连续Blocks

```python
class MetadataAllocation:
    block_indices: List[int]  # [8,9,3,4,5]
    num_tokens: int
    
    def get_contiguous_ranges(self, block_size: int):
        """
        合并连续blocks为ranges。
        
        Example:
            block_indices=[8,9,3,4,5], block_size=128
            -> sorted: [3,4,5,8,9]
            -> ranges: [(384, 384), (1024, 256)]
        """
        sorted_blocks = sorted(self.block_indices)
        ranges = []
        
        range_start = sorted_blocks[0]
        range_len = 1
        
        for i in range(1, len(sorted_blocks)):
            if sorted_blocks[i] == sorted_blocks[i-1] + 1:
                range_len += 1  # 连续，扩展
            else:
                # 不连续，保存当前range
                ranges.append((range_start * block_size, range_len * block_size))
                range_start = sorted_blocks[i]
                range_len = 1
        
        ranges.append((range_start * block_size, range_len * block_size))
        return ranges
```

**示例**：

| block_indices | sorted | ranges | 说明 |
|---------------|--------|--------|------|
| [0,1,2,3,4] | [0,1,2,3,4] | [(0,640)] | 全连续 |
| [0,2,4,6,8] | [0,2,4,6,8] | [(0,128),(256,128),...] | 全不连续 |
| [8,9,3,4,5] | [3,4,5,8,9] | [(384,384),(1024,256)] | 2段连续 |
| [15,14,8,7,3,2] | [2,3,7,8,14,15] | [(256,256),(896,256),(1792,256)] | 3段 |

### 2. Scatter写入

```python
def set_buf(self, req, allocation):
    """将数据分散写入到各个blocks"""
    ranges = allocation.get_contiguous_ranges(self.block_size)
    
    data_offset = 0
    for start_token, range_tokens in ranges:
        end_token = start_token + range_tokens
        
        # 写入这个连续range
        self.buffer[start_token:end_token] = \
            req.data[data_offset:data_offset+range_tokens]
        
        data_offset += range_tokens
```

**示例**（block_indices=[8,9,3,4,5], 640 tokens）：

```
ranges = [(384, 384), (1024, 256)]

Scatter:
  buffer[384:768] = data[0:384]      # 写入384 tokens到blocks [3,4,5]
  buffer[1024:1280] = data[384:640]  # 写入256 tokens到blocks [8,9]

Buffer布局:
  [0    - 383 ] 其他数据
  [384  - 767 ] 本请求 (data[0:384])   ✅
  [768  - 1023] 其他数据
  [1024 - 1279] 本请求 (data[384:640]) ✅
  [1280 - ...] 其他数据
```

### 3. Gather读取

```python
def get_buf(self, allocation):
    """从各个blocks收集数据"""
    ranges = allocation.get_contiguous_ranges(self.block_size)
    
    chunks = []
    for start_token, range_tokens in ranges:
        end_token = start_token + range_tokens
        chunks.append(self.buffer[start_token:end_token])
    
    # 拼接所有chunks
    return concat(chunks)
```

**示例**：

```
ranges = [(384, 384), (1024, 256)]

Gather:
  chunk1 = buffer[384:768]    # 384 tokens
  chunk2 = buffer[1024:1280]  # 256 tokens
  result = concat([chunk1, chunk2])  # 640 tokens ✅

完美还原原始数据！
```

---

## 🧪 验证结果

### 测试1：合并连续blocks

```python
block_indices = [8,9,3,4,5]
ranges = get_contiguous_ranges(128)
# -> [(384, 384), (1024, 256)]

✅ 自动识别2段连续：[3,4,5] 和 [8,9]
```

### 测试2：Scatter-Gather

```python
# Scatter写入
data = [0,1,2,...,767]  # 768 tokens
block_indices = [15,14,8,7,3,2]
ranges = [(256,256), (896,256), (1792,256)]

buffer[256:512] = data[0:256]      # blocks [2,3]
buffer[896:1152] = data[256:512]   # blocks [7,8]
buffer[1792:2048] = data[512:768]  # blocks [14,15]

# Gather读取
chunk1 = buffer[256:512]    # [0,1,...,255]
chunk2 = buffer[896:1152]   # [256,257,...,511]
chunk3 = buffer[1792:2048]  # [512,513,...,767]
result = concat([chunk1, chunk2, chunk3])

✅ result == data  # 完全一致！
```

### 测试3：真实场景

```
初始: free_blocks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

分配A(5 blocks): [0,1,2,3,4]
分配B(5 blocks): [5,6,7,8,9]
分配C(5 blocks): [10,11,12,13,14]

释放C: free_blocks=[15,14,13,12,11,10]  # 倒序归还
释放A: free_blocks=[15,14,13,12,11,10,4,3,2,1,0]

分配D(10 blocks): [15,14,13,12,11,10,4,3,2,1]
sorted: [1,2,3,4,10,11,12,13,14,15]
ranges: [(128,512), (1280,768)]  # 2段

✅ 真正利用所有10个blocks分散存储数据
```

---

## 📈 性能优化

### 合并连续blocks的好处

**不合并**（5个IO）：
```python
blocks = [2,3,7,8,14,15]
# 6次IO:
read(block[2])   # 128 tokens
read(block[3])   # 128 tokens
read(block[7])   # 128 tokens
read(block[8])   # 128 tokens
read(block[14])  # 128 tokens
read(block[15])  # 128 tokens
```

**合并后**（3个IO）：
```python
ranges = [(256,256), (896,256), (1792,256)]
# 3次IO:
read([256:512])    # 256 tokens
read([896:1152])   # 256 tokens
read([1792:2048])  # 256 tokens
```

**性能提升**：
- IO次数减半
- 每次IO更大（更高效）
- 对RDMA/网络传输友好

---

## 🎯 关键要点

### 与之前的区别

| 维度 | 之前（错误） | 现在（正确） |
|------|-------------|-------------|
| 假设 | 数据连续存储 | 数据分散存储 |
| 位置 | start_offset | ranges |
| blocks利用 | 只用min开始的区域 | 全部利用 |
| 浪费 | 高（部分blocks未用） | 低（所有blocks都用） |
| IO优化 | 无 | 合并连续blocks |

### 核心公式

```python
# 旧设计（错误）
start_offset = min(block_indices) * block_size
data_range = [start_offset, start_offset + num_tokens)

# 新设计（正确）
ranges = merge_contiguous_blocks(sorted(block_indices))
data = gather(buffer, ranges)
```

---

## ✅ 总结

### 核心改进

1. **真正的Scatter-Gather**
   - 数据分散存储在各个blocks
   - 读取时从多个位置gather
   - 完全利用所有分配的blocks

2. **自动优化**
   - 合并连续blocks减少IO
   - 最差情况：N个blocks = N次IO
   - 最好情况：N个连续blocks = 1次IO

3. **正确性**
   - Scatter-Gather验证通过
   - 数据完整性保证
   - 无浪费，无重叠

### 代码变更

```
修改：
- MetadataAllocation: 移除start_offset，新增get_contiguous_ranges()
- set_buf(): Scatter写入到多个ranges
- get_buf(): Gather读取并拼接
- get_buf_chunk_info(): 基于ranges计算
```

---

**这才是真正支持非连续blocks的实现！** ✅

---

**版本**: v7.0-scatter-gather  
**完成时间**: 2025-10-20
