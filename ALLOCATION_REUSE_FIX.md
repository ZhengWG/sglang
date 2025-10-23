# Allocation重用问题修复

## 🐛 潜在问题

**用户发现**：存在`last_resume_indices`和`current_indices`相同的可能吗？就是两次分配的allocation刚好一样。

---

## 🔍 问题分析

### 场景重现

```python
第一次Resume:
  └─ embedding_indices = [0, 1, 2, ..., 63]  # 8192 tokens
  └─ free([0-63])  # 释放这些blocks
  └─ alloc(remaining) → ???

第二次分配:
  └─ Allocator可能重用刚释放的blocks！
  └─ alloc() → [0, 1, 2, ..., 63]  # ❌ 相同的blocks！
  
比较:
  current_indices = (0, 1, 2, ..., 63)
  last_resume_indices = (0, 1, 2, ..., 63)
  current == last → skip ❌ 错误地跳过了新的resume！
```

### 为什么会重用？

**Allocator的行为**：
1. **FIFO策略**：先释放的先分配
   ```
   free([0-63]) → 加入free_list头部
   alloc() → 从free_list头部取 → [0-63]
   ```

2. **内存局部性优化**：
   ```
   重用最近释放的blocks可能有更好的cache locality
   ```

3. **简单的空闲链表管理**：
   ```
   released_blocks.push([0-63])
   alloc() → released_blocks.pop() → [0-63]
   ```

### 问题影响

```
Loop 1:
  poll() = Transferring
  indices = [0-63]
  → process → free → alloc → indices = [0-63]  # 重用！
  → resume_transfer()
  → last_resume_indices = (0,1,...,63)

Loop 2:
  poll() = Transferring (resume还没完成)
  indices = [0-63]
  → current == last
  → skip ✅ (正确，等待resume完成)

Loop 3:
  poll() = Transferring (假设第一次resume完成，但还需要更多)
  indices = [0-63]  # 还是相同的！
  → current == last
  → skip ❌ (错误！应该触发第二次resume)
```

---

## ✅ 修复方案

### 核心思路

**使用sent_tokens而非indices或generation counter**

- `sent_tokens`准确反映传输进度
- 单调递增，不会重复
- 即使allocator重用相同blocks也能正确判断

### 实现

#### 1. 基于sent_tokens计算当前进度

```python
elif poll == KVPoll.Transferring:
    # Read buffer
    embedding_data, fill_ids, mrope_positions, aux_datas = get_buf(block_indices)
    
    # Calculate sent_tokens
    if hasattr(req, 'partial_aux_datas'):
        # Has cached data from previous resume
        actual_total_length = cached_total
        previous_sent = req.partial_sent_tokens
        sent_tokens = previous_sent + len(fill_ids)  # Accumulate
    else:
        # First Transferring
        actual_total_length = aux_datas[0]
        sent_tokens = len(fill_ids)
        # Sync across ranks...
```

#### 2. 检查是否已触发resume

```python
# Check if we already triggered resume at this sent_tokens value
last_resume_at_sent_tokens = getattr(req, 'last_resume_at_sent_tokens', -1)

if sent_tokens == last_resume_at_sent_tokens and sent_tokens > 0:
    # Resume already triggered, waiting for completion
    continue  # Skip
```

#### 3. 触发resume并记录sent_tokens

```python
# Resume logic: free, alloc, resume_transfer...

# Update partial_sent_tokens for next round
req.partial_sent_tokens = sent_tokens

# Record sent_tokens at which we triggered resume
req.last_resume_at_sent_tokens = sent_tokens
```

---

## 📊 修复效果

### 修复前（基于indices）

```
Loop 1: gen=0, last_gen=-1
  indices=[0-63]
  → gen != last_gen → process
  → resume → indices=[0-63] (重用!)
  → last_resume_indices=(0,...,63)

Loop 2: gen=0, last_gen=0
  indices=[0-63]
  → gen == last_gen → skip ✅

Loop 3: gen=0, last_gen=0 (第一次resume完成，但还需要更多)
  indices=[0-63]
  → current == last
  → skip ❌ (错误！应该触发第二次resume)
```

### 修复后（基于generation）

```
Loop 1: gen=0, last_gen=-1
  indices=[0-63]
  → gen != last_gen → process ✅
  → last_processed_gen=0
  → resume
  → gen++ → gen=1
  → indices=[0-63] (重用也没关系)

Loop 2: gen=1, last_gen=0
  indices=[0-63]
  → gen != last_gen
  → skip? 不，gen已经变了！
  
  等等，这里逻辑有问题...

实际上应该是：

Loop 1: gen=0, last_processed=-1
  → process
  → last_processed=0
  → resume
  → gen=1 (increment for next round)

Loop 2: gen=1, last_processed=0
  → gen != last_processed → process?
  
不对，这样resume还没完成就又process了...

让我重新思考：

正确的逻辑应该是：
- 进入Transferring时，检查当前generation是否已处理
- 如果未处理，处理并标记为已处理
- 处理完成后，不立即增加generation
- 只有在真正需要下一次resume时才增加generation

实际上，generation应该在什么时候增加？

应该是：free+alloc完成后，更新indices后，增加generation。

但是这样的话，同一个loop iteration内：
- 开始：gen=0, last=-1
- 检查：gen != last → process
- 标记：last=0
- 执行：free, alloc
- 更新：indices变化，gen++

下一个loop:
- gen=1, last=0
- gen != last → process again? ❌

问题在于：我们需要区分"正在等待当前resume完成"和"新的resume轮次"。

更好的方案：
1. 检查时不增加generation
2. Resume触发后，标记当前indices为"processing"
3. 下次loop检查indices是否和processing相同
4. 如果相同且正在processing，跳过
5. 如果indices变化了（新allocation），可以处理

或者：
- 在alloc成功后立即增加generation
- 检查：current_gen > last_processed_gen → process

wait, 让我重新看一下代码逻辑：

当前的修复：
1. current_resume_generation = getattr(req, 'resume_generation', 0)
2. last_processed_generation = getattr(req, 'last_processed_generation', -1)
3. if current == last: skip
4. last_processed = current
5. ... resume logic
6. resume_generation = current + 1

所以流程是：
Loop 1: resume_gen=0 (初始), last_processed=-1
  → 0 != -1 → process
  → last_processed = 0
  → resume
  → resume_gen = 1

Loop 2: resume_gen=1, last_processed=0
  → 1 != 0 → process? ❌ 这不对！

问题在于：resume_generation在resume触发后就增加了，但下一次loop不应该再process。

应该改为：
- 只在indices真正变化时才增加generation
- 或者在resume完成（Success）时增加generation

实际上，最简单的方案是：
- 使用一个boolean标记"resume_in_progress"
- Resume触发时设置为True
- Resume完成（下次进入Transferring，或Success）时检查是否需要新的resume
- 如果需要，清除标记，触发新resume

或者更简单：
- 在free之前记录old_indices
- 在alloc之后检查new_indices是否和old_indices不同
- 只有当不同时才更新last_resume_indices

让我采用更简单的方案：在resume逻辑的最开始就记录当前indices，在结束时才更新为新indices。
</thinking>

非常好的发现！确实存在这个问题。让我修复：

<function_calls>
<invoke name="StrReplace">
<parameter name="file_path">/workspace/python/sglang/srt/disaggregation/multimodal_language.py