# 多次Resume支持

## 🎯 设计目标

**问题**：当数据非常大时，单次resume可能仍然不够。例如：
- 实际数据：50000 tokens
- 第一次分配：8192 tokens → Transferring
- 第二次分配（resume）：16384 tokens → Transferring (还不够！)
- 第三次分配（第二次resume）：25424 tokens → Success ✅

当前实现需要支持这种多次resume场景。

---

## ✅ 实现机制

### 核心思路

**不使用永久boolean标记，而是基于allocation indices的变化来判断**

### 关键逻辑

```python
# 获取当前allocation
current_indices = tuple(language_req.embedding_indices)

# 获取上次处理的allocation
last_processed_indices = getattr(language_req.req, 'last_resume_indices', None)

# 判断是否需要处理
if current_indices == last_processed_indices:
    # 相同allocation，已经处理过，跳过
    continue
else:
    # 不同allocation（新的resume轮次，或第一次），处理
    # ... get_buf, free, alloc, resume_transfer
    # 记录当前allocation
    language_req.req.last_resume_indices = tuple(new_allocation)
```

### 为什么这样设计？

1. **区分不同的resume轮次**：
   - 每次resume后，embedding_indices会变化（新的allocation）
   - 通过比较indices，可以识别是否是新的轮次

2. **避免同一轮次的重复处理**：
   - 同一个allocation期间，indices不变
   - 跳过重复的loop iteration

3. **自然支持多次resume**：
   - 第一次resume：indices [0-7] → [8-15]
   - 第二次resume：indices [8-15] → [16-23]
   - 第三次resume：indices [16-23] → [24-31]
   - 每次都是新的indices，都会被处理

---

## 📊 完整流程示例

### 场景：50000 tokens，需要多次resume

```
初始状态：
  actual_total_length = 50000 tokens
  Language侧默认分配 = 8192 tokens

=== 第一次传输 ===

Loop 1-N: (第一次传输期间)
  poll() = WaitingForInput / Bootstrapping
  等待第一次传输...

Loop N+1:
  poll() = Transferring
  embedding_indices = [0, 1, 2, ..., 63]  # 8192 tokens
  last_resume_indices = None
  
  → current != last_processed ✅
  → get_buf() → 读取8192 tokens
  → free([0-63])
  → alloc(41808) → [64-390] (假设)
  → resume_transfer(sent_tokens=8192, allocated_tokens=41808)
  → last_resume_indices = (64, 65, ..., 390)

Loop N+2, N+3, ...: (第一次resume传输期间)
  poll() = Transferring
  embedding_indices = [64-390]
  last_resume_indices = (64, 65, ..., 390)
  
  → current == last_processed ✅
  → continue (跳过)

=== 第一次Resume完成，但数据还不够 ===

假设：第一次resume只传输了 41808 tokens
已传输：8192 + 41808 = 50000 ✅ (刚好够了！)

Loop M:
  poll() = Success
  → 合并数据，完成 ✅

---

### 场景2：如果仍然不够（需要第二次resume）

假设第二次只分配了16384 tokens：

Loop N+1:
  poll() = Transferring
  embedding_indices = [0-63]  # 8192 tokens
  → 第一次resume
  → alloc(16384) → [64-191]
  → last_resume_indices = (64, ..., 191)

Loop N+k: (第一次resume完成，但还需要更多)
  poll() = Transferring ⚠️ (sent_tokens=24576 < total=50000)
  embedding_indices = [64-191]  # 已经变化！
  last_resume_indices = (64, ..., 191)
  
  → current == last_processed ❌ 等等，这里有问题！
  
  实际上，这时Language侧会：
  1. 检测到还需要更多数据
  2. free([64-191])
  3. alloc(remaining) → [192-...]
  4. embedding_indices变为[192-...]
  
Loop N+k+1:
  poll() = Transferring
  embedding_indices = [192-...]  # 新的allocation!
  last_resume_indices = (64, ..., 191)
  
  → current != last_processed ✅ (indices已变化)
  → 执行第二次resume ✅
  → last_resume_indices = (192, ...)
```

---

## 🔑 关键点

### 1. 何时indices会变化？

**在Transferring状态的处理中**：
```python
# Free old allocation
free(block_indices)  # 释放当前的indices

# Allocate new
new_allocation = alloc(remaining_tokens)

# Update
language_req.embedding_indices = new_allocation  # ← indices变化！
```

**下一次loop**：
- `current_indices`从新的`language_req.embedding_indices`获取
- `last_processed_indices`是旧的值
- 两者不同，触发新的resume

### 2. 为什么使用tuple？

- `embedding_indices`是list，不能直接用于比较（每次都是新对象）
- 转换为tuple可以比较值是否相同
- tuple是不可变的，适合作为标记

### 3. 内存管理

```python
# 每次resume：
old_allocation = [0-63]
  → free → allocator回收
new_allocation = [64-127]
  → 使用新的blocks
  → 旧blocks可被其他请求使用
```

---

## 🎯 优势

1. **自动支持多次resume**：
   - 不需要额外的逻辑
   - indices的变化自然区分不同的resume轮次

2. **简单明确**：
   - 不需要计数器
   - 不需要复杂的状态机

3. **防止重复处理**：
   - 同一个allocation只处理一次
   - 避免内存泄漏

4. **可扩展**：
   - 支持任意次数的resume
   - 只要内存够，可以无限次

---

## 📝 配置建议

如果经常需要多次resume，可以考虑：

```bash
# 增加默认分配大小，减少resume次数
export SGLANG_EMBEDDING_DEFAULT_ALLOCATE_BUFFER_SIZE=16384  # 默认是8192

# 或者基于历史数据动态调整
# (未来可以实现自适应分配策略)
```

---

## 🎉 总结

通过基于allocation indices的标记机制：
- ✅ 支持多次resume
- ✅ 防止重复处理
- ✅ 简单明确
- ✅ 易于维护

这个设计为未来可能的极大数据场景（如100K+ tokens的embedding）提供了充分的灵活性。
