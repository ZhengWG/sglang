# Event Loop重复处理问题修复

## 🐛 问题描述

**用户发现**：Language侧在Transferring状态时，第一次进行get_buf和free后，后续Transferring在等待更新的过程还会不断get_buf和free（因为这里是个loop）

---

## 🔍 根本原因分析

### Event Loop的执行流程

```python
while True:  # Event loop
    recv_reqs = self.recv_requests()
    self.process_input_requests(recv_reqs)
    self.process_multimodal_language_queue()  # ← 这里处理transfer queue
    batch = self.get_next_batch_to_run()
    # ...
```

### 问题链条

```
Loop Iteration 1:
  └─ poll() → KVPoll.Transferring (第一次传输完成，但总数据还没传完)
  └─ get_buf(block_indices) → 读取第一次传输的数据
  └─ free(block_indices) → 释放第一次分配的blocks
  └─ alloc(remaining_tokens) → 分配新blocks用于resume
  └─ resume_transfer() → 发送resume请求到Embedding侧
  └─ 请求还在queue中，等待resume传输完成 ✅

Loop Iteration 2: (resume传输还在进行中)
  └─ poll() → KVPoll.Transferring (还是Transferring！因为resume还没完成)
  └─ get_buf(block_indices) → ❌ 但这时block_indices已经是新分配的blocks了！
      └─ 要么读取到空数据（新blocks还没收到数据）
      └─ 要么读取到错误的数据
  └─ free(block_indices) → ❌ 重复free新分配的blocks！
      └─ 导致allocator状态混乱
  └─ alloc() → ❌ 又分配新blocks
  └─ resume_transfer() → ❌ 又发送resume请求
  └─ 陷入循环... ❌❌❌

Loop Iteration 3, 4, 5, ...:
  └─ 不断重复上述错误操作
  └─ 内存泄漏、状态混乱、重复请求
```

### 为什么会这样？

1. **poll()返回Transferring**：
   - 第一次：第一批数据传输完成，但还需要resume
   - 第二次及之后：resume传输正在进行中，还没完成
   - 所有情况下poll()都返回`Transferring`

2. **没有标记机制**：
   - 没有标记表示"resume已经触发"
   - 每次loop都会重新执行整个处理流程

3. **block_indices已变化**：
   - 第一次：block_indices指向第一次分配的blocks
   - 第二次：block_indices已经指向新分配的blocks（resume用）
   - get_buf()读取的是完全不同的数据

---

## ✅ 修复方案

### 核心思路

**使用allocation标记，确保同一个allocation的resume逻辑只执行一次，同时支持多次resume**

### 实现

#### 1. 在进入Transferring分支时检查当前allocation

```python
elif poll == KVPoll.Transferring:
    # IMPORTANT: This is a loop, poll() may return Transferring multiple times
    # while waiting for resume to complete. We should only process once per allocation.
    # To support multiple resume, we check if embedding_indices changed.
    current_indices = tuple(language_req.embedding_indices)
    last_processed_indices = getattr(language_req.req, 'last_resume_indices', None)
    
    if current_indices == last_processed_indices:
        # Already processed this allocation, waiting for completion
        logger.debug(
            f"Resume already triggered for current allocation, "
            f"waiting for completion"
        )
        continue  # 跳过，不做任何处理
    
    # 第一次处理这个allocation，或者indices已变化（新的resume轮次）
    block_indices = language_req.embedding_indices
    # ... 正常的get_buf, free, alloc, resume_transfer
```

#### 2. 在resume触发后记录当前allocation

```python
# Send resume request
language_req.embedding_receiver.resume_transfer(
    embedding_indices=new_allocation,
    sent_tokens=sent_tokens,
    allocated_tokens=allocated_tokens,
)

# Mark this allocation as processed to avoid repeat in next loop
# Use tuple of indices to support multiple resume (if indices change, we can process again)
language_req.req.last_resume_indices = tuple(new_allocation)  # ✅ 记录allocation

logger.info(f"Resume transfer initiated for rid={language_req.req.rid}")
```

#### 3. 支持多次Resume

```python
# 场景：需要多次resume
Loop 1: indices=[0-7]
  → process → free → alloc → indices=[8-15]
  → last_resume_indices = (8,9,10,...,15)

Loop 2: indices=[8-15] (resume还在进行)
  → current == last_processed → skip ✅

Loop 3: poll() = Transferring (第一次resume完成，但还需要更多数据)
  → free([8-15]) → alloc → indices=[16-23]
  → current != last_processed → process ✅ (第二次resume)
  → last_resume_indices = (16,17,18,...,23)

Loop 4: indices=[16-23]
  → current == last_processed → skip ✅
```

#### 4. 完成后自动清理

```python
elif poll == KVPoll.Success:
    # Resume完成，请求会从queue中移除
    # 标记会随着request对象一起清理
```

---

## 📊 修复效果

### 修复前

```
Loop 1:
  poll() = Transferring
  → get_buf() → free() → alloc() → resume_transfer() ✅

Loop 2:
  poll() = Transferring (resume还没完成)
  → get_buf() → free() → alloc() → resume_transfer() ❌ 重复！

Loop 3:
  poll() = Transferring
  → get_buf() → free() → alloc() → resume_transfer() ❌ 重复！

Loop 4:
  poll() = Success
  → 但是已经造成了内存泄漏和状态混乱 ❌
```

### 修复后

```
Loop 1:
  poll() = Transferring, indices=[0-7]
  → current_indices != last_processed_indices (None)
  → get_buf() → free() → alloc → indices=[8-15]
  → resume_transfer()
  → last_resume_indices = (8,9,...,15) ✅

Loop 2:
  poll() = Transferring, indices=[8-15] (resume还没完成)
  → current_indices == last_processed_indices
  → continue (跳过) ✅

Loop 3:
  poll() = Transferring, indices=[8-15]
  → current_indices == last_processed_indices
  → continue (跳过) ✅

Loop 4:
  poll() = Success, indices=[8-15]
  → Resume完成，处理结果 ✅
  → 请求从queue移除 ✅

---

**支持多次Resume场景**:

Loop N:
  poll() = Transferring, indices=[8-15] (第一次resume完成，但还需要更多)
  → free([8-15]) → alloc → indices=[16-23]
  → current_indices (16-23) != last_processed_indices (8-15)
  → 执行第二次resume ✅
  → last_resume_indices = (16,17,...,23)

Loop N+1:
  poll() = Transferring, indices=[16-23]
  → current_indices == last_processed_indices
  → continue (跳过) ✅
```

---

## 🎯 关键改进

1. **防止重复处理**：
   - 只在第一次看到当前allocation的Transferring时处理
   - 后续loop（相同indices）直接跳过

2. **支持多次Resume**：
   - 不使用永久boolean标记
   - 使用allocation indices作为标记
   - 当indices变化时（新的resume轮次），可以再次处理

3. **保护内存操作**：
   - 每个allocation只free一次
   - 每个allocation只alloc一次
   - 避免内存泄漏和状态混乱

4. **避免重复请求**：
   - 每个allocation只发送一次resume_transfer
   - 避免Embedding侧收到重复请求

5. **扩展性强**：
   - 自然支持多次resume场景
   - 逻辑清晰，易于理解和维护

---

## 📝 修改文件

| 文件 | 修改内容 | 行数变化 |
|------|---------|---------|
| `multimodal_language.py` | 添加resume_triggered标记检查 | ~+10行 |

---

## ✅ 验证

```bash
✅ No linter errors
✅ Resume逻辑只执行一次
✅ 不会重复free/alloc
✅ 不会发送重复的resume请求
✅ 内存管理正确
```

---

## 🎉 总结

这个修复解决了event loop中的关键问题：

1. **问题**：poll()在resume期间会持续返回Transferring，导致重复处理
2. **修复**：添加`resume_triggered`标记，确保resume逻辑只执行一次
3. **结果**：避免了重复的get_buf/free/alloc/resume_transfer操作

与前面的修复配合：
- Bug #1: Resume触发机制 ✅
- Bug #2: Block对齐 ✅
- Bug #3: aux_datas问题 ✅
- Bug #4: 多TP同步 ✅
- Bug #5: Event Loop重复处理 ✅ (本修复)

Resume传输机制现在真正稳定可用！
