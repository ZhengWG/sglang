# Block对齐问题修复

## 🐛 问题描述

**发现者**：用户反馈  
**问题**：Language侧传递的`allocated_tokens`可能与实际分配的blocks不对齐

### 问题场景

```python
# Language侧配置
default_allocate_tokens = 8192  # 或其他不能被block_size整除的数字

# 分配逻辑
embedding_indices = allocator.alloc(num_tokens=8192)
# Allocator实际分配：ceil(8192 / 128) = 64 blocks
# 实际tokens：64 * 128 = 8192 ✅

# 但如果default_allocate_tokens = 8000:
embedding_indices = allocator.alloc(num_tokens=8000)
# Allocator实际分配：ceil(8000 / 128) = 63 blocks
# 实际tokens：63 * 128 = 8064 ≠ 8000 ❌

# 原实现传递的是：
allocated_tokens = self.default_allocate_tokens  # 8000 ❌
```

### 问题影响

Embedding侧会进行block_size一致性验证：
```python
expected_block_size = allocated_tokens // len(dst_embedding_indices)
# 8000 // 63 = 126.98... ≠ 128 ❌
# 验证失败！
```

---

## ✅ 修复方案

### 核心原则

**传递实际分配的token数量，而不是请求的token数量**

### 修复代码

#### 1. 初始化时（`pop_preallocated`）

**修复前**：
```python
language_req.embedding_receiver.init(
    embedding_indices=language_req.embedding_indices,
    allocated_tokens=self.default_allocate_tokens,  # ❌ 请求的数量
)
```

**修复后**：
```python
# Calculate actual allocated tokens from allocated blocks
# This ensures proper alignment with block_size
actual_allocated_tokens = len(language_req.embedding_indices) * self.metadata_buffers.block_size

language_req.embedding_receiver.init(
    embedding_indices=language_req.embedding_indices,
    allocated_tokens=actual_allocated_tokens,  # ✅ 实际分配的数量
)
```

#### 2. Resume时（`pop_transferred`）

**已经是正确的**：
```python
# Calculate allocated_tokens from new allocation
block_size = self.metadata_buffers.block_size
allocated_tokens = len(new_allocation) * block_size  # ✅ 正确

language_req.embedding_receiver.resume_transfer(
    embedding_indices=new_allocation,
    sent_tokens=sent_tokens,
    allocated_tokens=allocated_tokens,
)
```

---

## 🔍 验证

### 场景1：正好整除

```python
default_allocate_tokens = 8192
block_size = 128

# 分配
blocks = alloc(8192) → 64 blocks
actual_allocated = 64 * 128 = 8192 ✅

# Embedding侧验证
expected_block_size = 8192 // 64 = 128 ✅
```

### 场景2：不能整除

```python
default_allocate_tokens = 8000
block_size = 128

# 分配
blocks = alloc(8000) → 63 blocks (向上取整)
actual_allocated = 63 * 128 = 8064 ✅

# 修复前
allocated_tokens = 8000 ❌
expected_block_size = 8000 // 63 = 126.98... ❌ 验证失败

# 修复后
allocated_tokens = 8064 ✅
expected_block_size = 8064 // 63 = 128 ✅ 验证通过
```

### 场景3：Resume时

```python
remaining_tokens = 1000
block_size = 128

# 分配
blocks = alloc(1000) → 8 blocks
actual_allocated = 8 * 128 = 1024 ✅

# Resume逻辑（已经是正确的）
allocated_tokens = 8 * 128 = 1024 ✅
expected_block_size = 1024 // 8 = 128 ✅
```

---

## 📊 对比总结

| 情况 | 修复前 | 修复后 |
|------|--------|--------|
| **传递的值** | `default_allocate_tokens` | `len(blocks) * block_size` |
| **对齐保证** | ❌ 不保证 | ✅ 保证 |
| **验证结果** | ❌ 可能失败 | ✅ 总是通过 |

---

## 🎯 关键优势

1. **正确性**：总是传递实际分配的token数量
2. **对齐保证**：`allocated_tokens`保证是`block_size`的整数倍
3. **验证通过**：Embedding侧的block_size验证总是能通过
4. **简单明确**：逻辑清晰，易于理解和维护

---

## 📝 修改文件

| 文件 | 修改内容 | 位置 |
|------|---------|------|
| `multimodal_language.py` | Init时计算actual_allocated_tokens | `pop_preallocated()` |
| `multimodal_language.py` | Resume时已正确（无需修改） | `pop_transferred()` |

---

## ✅ 验证结果

```bash
✅ No linter errors
✅ Block对齐保证
✅ 验证逻辑通过
```

---

## 🎉 总结

通过传递**实际分配的token数量**而非**请求的token数量**，确保了：

1. Language侧和Embedding侧的block_size对齐
2. Embedding侧的验证逻辑总是能通过
3. 不会因为配置的`default_allocate_tokens`不能整除`block_size`而失败

感谢用户的细致发现！这个修复确保了系统的鲁棒性。
