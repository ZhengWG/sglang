# 多模态Embedding Resume传输设计方案

## 📋 问题描述

### 当前问题
Language侧每次申请block是按照**默认值**申请的（目前是8192 tokens），但Embedding侧能获取**实际的req_len**，这导致`dst_embedding_indices`数目可能会**小于**Embedding侧`embedding_indices`数目。

### 示例场景
```
实际情况: Embedding侧需要发送2000 tokens的数据
Language侧: 默认分配1024 tokens (8 blocks * 128 tokens/block)
问题: 1024 < 2000，目标缓冲区不足
```

当前行为（`conn_multimodal.py` 233-238行）:
```python
if len(embedding_indices) > len(dst_embedding_indices):
    raise ValueError(
        f"Source blocks ({len(embedding_indices)}) cannot be greater than "
        f"destination blocks ({len(dst_embedding_indices)}). "
        f"Language side allocated insufficient buffer."
    )
```

---

## 🎯 解决方案：Resume传输机制

### 核心思路
实现**两阶段传输**机制：
1. **第一阶段**: Language侧按默认值分配 → Embedding侧发送部分数据
2. **通知阶段**: Embedding侧告知Language实际长度
3. **第二阶段**: Language侧重新分配剩余空间 → Embedding侧发送剩余数据

---

## 🏗️ 设计方案

### 1. 数据结构修改

#### 1.1 `TransferEmbeddingInfo` (conn_multimodal.py)
添加字段跟踪传输进度：

```python
@dataclasses.dataclass
class TransferEmbeddingInfo:
    room: int
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_embedding_indices: List[int]
    required_dst_info_num: int
    sent_tokens: int = 0           # 新增：已发送的token数
    allocated_tokens: int = 0      # 新增：Language侧分配的token数
```

#### 1.2 Resume请求消息格式
在ZMQ消息中添加resume信息：

```python
# 初始化消息（init）
[
    str(room).encode("ascii"),
    endpoint.encode("ascii"),
    str(dst_port).encode("ascii"),
    session_id.encode("ascii"),
    embedding_indices_str.encode("ascii"),
    str(required_dst_info_num).encode("ascii"),
    str(allocated_tokens).encode("ascii"),  # 新增
]

# Resume消息（resume_transfer）
[
    str(room).encode("ascii"),
    endpoint.encode("ascii"),
    str(dst_port).encode("ascii"),
    session_id.encode("ascii"),
    embedding_indices_str.encode("ascii"),
    str(required_dst_info_num).encode("ascii"),
    str(sent_tokens).encode("ascii"),       # 新增：已发送的token数
    str(allocated_tokens).encode("ascii"),  # 新增：新分配的token数
]
```

---

### 2. Language侧修改 (multimodal_language.py)

#### 2.1 在`MultimodalLanguagePreallocQueue`中添加Resume逻辑

```python
class MultimodalLanguagePreallocQueue:
    def __init__(self, ...):
        # 现有代码...
        self.default_allocate_tokens = int(
            os.getenv("SGLANG_EMBEDDING_DEFAULT_ALLOCATE_BUFFER_SIZE", "8192")
        )
        # 新增：跟踪需要resume的请求
        self.resume_queue: List[MultimodalLanguageRequest] = []
```

#### 2.2 修改`MooncakeEmbeddingReceiver.init()`方法

添加`allocated_tokens`参数：
```python
def init(
    self,
    embedding_indices: Optional[List[int]] = None,
    allocated_tokens: Optional[int] = None,  # 新增
):
    if embedding_indices is not None:
        embedding_indices_str = ",".join(str(idx) for idx in embedding_indices)
    else:
        embedding_indices_str = ""
    
    # 计算allocated_tokens
    if allocated_tokens is None:
        block_size = self.embedding_mgr.data_args.aux_item_lens[1] // 4
        allocated_tokens = len(embedding_indices) * block_size
    
    for bootstrap_info in self.bootstrap_infos:
        # ...发送消息，包含allocated_tokens
        sock.send_multipart([
            str(self.bootstrap_room).encode("ascii"),
            get_local_ip_by_remote().encode("ascii"),
            str(self.embedding_mgr.rank_port).encode("ascii"),
            self.session_id.encode("ascii"),
            embedding_indices_str.encode("ascii"),
            str(self.required_dst_info_num).encode("ascii"),
            str(allocated_tokens).encode("ascii"),  # 新增
        ])
```

#### 2.3 添加Resume传输方法

```python
class MooncakeEmbeddingReceiver(BaseKVReceiver):
    def resume_transfer(
        self,
        embedding_indices: List[int],
        sent_tokens: int,
        allocated_tokens: int,
    ):
        """Resume transfer with new allocation after partial transfer.
        
        Args:
            embedding_indices: New block allocation for remaining data
            sent_tokens: Number of tokens already transferred
            allocated_tokens: Number of tokens in new allocation
        """
        embedding_indices_str = ",".join(str(idx) for idx in embedding_indices)
        
        for bootstrap_info in self.bootstrap_infos:
            self.embedding_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            
            sock, lock = self._connect("tcp://" + self.embedding_server_url)
            with lock:
                sock.send_multipart([
                    str(self.bootstrap_room).encode("ascii"),
                    get_local_ip_by_remote().encode("ascii"),
                    str(self.embedding_mgr.rank_port).encode("ascii"),
                    self.session_id.encode("ascii"),
                    embedding_indices_str.encode("ascii"),
                    str(self.required_dst_info_num).encode("ascii"),
                    str(sent_tokens).encode("ascii"),      # 标识resume
                    str(allocated_tokens).encode("ascii"),  # 新分配的大小
                ])
```

#### 2.4 修改`MultimodalLanguageTransferQueue.pop_transferred()`

检测部分传输，并触发resume：
```python
def pop_transferred(self):
    # ...现有代码...
    
    for i, (language_req, poll) in enumerate(zip(self.queue, polls)):
        if poll == KVPoll.Transferring:  # 新状态：部分传输完成
            # 获取实际需要的总长度
            block_indices = language_req.embedding_indices
            embedding_data, fill_ids, mrope_positions, aux_datas = (
                self.metadata_buffers.get_buf(block_indices=block_indices)
            )
            actual_total_length = aux_datas[0]  # 实际总长度
            sent_tokens = len(fill_ids)          # 已发送的token数
            
            if actual_total_length > sent_tokens:
                # 需要resume传输
                remaining_tokens = actual_total_length - sent_tokens
                
                # 缓存已接收的数据
                language_req.req.partial_input_embeds = embedding_data
                language_req.req.partial_fill_ids = fill_ids.tolist()
                language_req.req.partial_mrope_positions = mrope_positions
                language_req.req.partial_aux_datas = aux_datas
                
                # 释放旧的分配
                self.req_to_metadata_buffer_idx_allocator.free(
                    block_indices=block_indices,
                    req_id=language_req.req.rid,
                    fake=isinstance(language_req.embedding_receiver, FakeKVReceiver),
                )
                
                # 重新分配剩余空间
                new_allocation = self.req_to_metadata_buffer_idx_allocator.alloc(
                    num_tokens=remaining_tokens,
                    req_id=language_req.req.rid,
                    fake=isinstance(language_req.embedding_receiver, FakeKVReceiver),
                )
                
                if new_allocation is None:
                    # 内存不足，稍后重试
                    logger.warning(f"Not enough memory to resume transfer for {language_req.req.rid}")
                    continue
                
                # 更新embedding_indices
                language_req.embedding_indices = new_allocation
                
                # 发送resume请求
                language_req.embedding_receiver.resume_transfer(
                    embedding_indices=new_allocation,
                    sent_tokens=sent_tokens,
                    allocated_tokens=remaining_tokens,
                )
                
                # 继续等待
                continue
        
        elif poll == KVPoll.Success:
            # 完整传输完成
            # 如果有partial数据，需要合并
            if hasattr(language_req.req, 'partial_input_embeds'):
                # 合并数据
                new_embedding_data, new_fill_ids, new_mrope_positions, _ = (
                    self.metadata_buffers.get_buf(block_indices=language_req.embedding_indices)
                )
                
                language_req.req.input_embeds = torch.cat([
                    language_req.req.partial_input_embeds,
                    new_embedding_data
                ])
                language_req.req.origin_input_ids = (
                    language_req.req.partial_fill_ids + new_fill_ids.tolist()
                )
                # 合并mrope_positions...
                
                # 清理partial数据
                del language_req.req.partial_input_embeds
                del language_req.req.partial_fill_ids
                del language_req.req.partial_mrope_positions
                del language_req.req.partial_aux_datas
            else:
                # 正常单次传输完成
                # ...现有代码...
            
            transferred_reqs.append(language_req.req)
            indices_to_remove.add(i)
```

---

### 3. Embedding侧修改 (multimodal_embedding.py)

#### 3.1 修改`send_embedding_chunk()`

添加`sent_tokens`参数支持resume：
```python
def send_embedding_chunk(
    self: Scheduler,
    req: Req,
    last_chunk: bool = False,
    sent_tokens: int = 0,  # 新增：已发送的token数
):
    assert last_chunk == True or sent_tokens > 0  # 要么是最后一块，要么是resume
    
    if last_chunk and sent_tokens == 0:
        # 初次传输
        self.disagg_metadata_buffers.set_buf(req)
    
    # 计算要发送的indices（跳过已发送的部分）
    total_tokens = len(req.fill_ids)
    remaining_tokens = total_tokens - sent_tokens
    
    # Send using block_indices
    req.disagg_embedding_sender.send_embedding(
        embedding_indices=req.embedding_indices,
        last_chunk=last_chunk,
        total_tokens=remaining_tokens,
        sent_tokens=sent_tokens,
        block_size=self.disagg_metadata_buffers.block_size,
    )
```

#### 3.2 修改`MooncakeEmbeddingSender`

添加resume支持：
```python
class MooncakeEmbeddingSender(BaseKVSender):
    def __init__(self, ...):
        # ...现有代码...
        self.sent_tokens = 0  # 新增：跟踪已发送的token数
    
    def send_embedding(
        self,
        embedding_indices: List[int] = None,
        last_chunk: bool = True,
        total_tokens: int = None,
        block_size: int = None,
        sent_tokens: int = 0,  # 新增
    ):
        """Send embedding data to language instances using block-based transfer.
        
        Args:
            embedding_indices: List of source embedding indices
            last_chunk: Whether this is the last chunk
            total_tokens: Total number of tokens to transfer (excluding sent_tokens)
            block_size: Number of tokens per block
            sent_tokens: Number of tokens already sent (for resume)
        """
        self.sent_tokens = sent_tokens
        self.embedding_mgr.add_transfer_request(
            self.bootstrap_room,
            embedding_indices,
            last_chunk,
            total_tokens,
            block_size,
            sent_tokens,
        )
```

---

### 4. Connection层修改 (conn_multimodal.py)

#### 4.1 修改`send_embedding()`方法

支持部分传输：
```python
def send_embedding(
    self,
    mooncake_session_id: str,
    embedding_indices: List[int],
    dst_embedding_ptrs: list[int],
    dst_embedding_indices: List[int],
    total_tokens: int,
    block_size: int,
    sent_tokens: int = 0,  # 新增：已发送的token数
):
    """Send embedding data using block-based transfer.
    
    Args:
        sent_tokens: Number of tokens already sent (for resume transfer)
    """
    # Validate: 移除严格的验证，允许部分传输
    if sent_tokens == 0 and len(embedding_indices) > len(dst_embedding_indices):
        # 初次传输且缓冲区不足：只发送部分
        logger.warning(
            f"Partial transfer: Source blocks ({len(embedding_indices)}) > "
            f"destination blocks ({len(dst_embedding_indices)}). "
            f"Will transfer {len(dst_embedding_indices)} blocks first."
        )
        # 只传输能容纳的部分
        tokens_to_send = len(dst_embedding_indices) * block_size
        total_tokens = min(total_tokens, tokens_to_send)
    
    # 计算要发送的block范围
    start_block = sent_tokens // block_size
    embedding_indices_to_send = embedding_indices[start_block:]
    
    # 限制为dst容量
    if len(embedding_indices_to_send) > len(dst_embedding_indices):
        embedding_indices_to_send = embedding_indices_to_send[:len(dst_embedding_indices)]
    
    src_addrs = []
    dst_addrs = []
    lengths = []
    
    for block_idx, (src_block_idx, dst_block_idx) in enumerate(
        zip(embedding_indices_to_send, dst_embedding_indices)
    ):
        # Calculate tokens in this block
        tokens_in_prev_blocks = start_block * block_size + block_idx * block_size
        start_pos = tokens_in_prev_blocks - sent_tokens
        end_pos = min(start_pos + block_size, total_tokens)
        tokens_in_block = end_pos - start_pos
        
        if tokens_in_block <= 0:
            break
        
        # Transfer each buffer type within the block
        for buffer_type_idx in range(len(self.data_args.aux_item_lens)):
            embedding_item_len = self.data_args.aux_item_lens[buffer_type_idx]
            
            # Calculate chunk size
            if buffer_type_idx == 3:  # aux_datas
                if sent_tokens == 0 and block_idx == 0:  # 只在初次传输的第一个块发送
                    chunk_size = embedding_item_len
                else:
                    continue
            else:
                chunk_size = (embedding_item_len * tokens_in_block) // block_size
            
            embedding_addr = (
                self.data_args.aux_data_ptrs[buffer_type_idx]
                + src_block_idx * embedding_item_len
            )
            dst_embedding_addr = (
                dst_embedding_ptrs[buffer_type_idx]
                + dst_block_idx * embedding_item_len
            )
            
            src_addrs.append(embedding_addr)
            dst_addrs.append(dst_embedding_addr)
            lengths.append(chunk_size)
    
    return self.engine.batch_transfer_sync(
        mooncake_session_id, src_addrs, dst_addrs, lengths
    )
```

#### 4.2 修改`embedding_thread()`处理Resume消息

```python
def embedding_thread():
    """This thread recvs pre-alloc notification from the language engine"""
    while True:
        waiting_req_bytes = self.server_socket.recv_multipart()
        room = waiting_req_bytes[0].decode("ascii")
        mooncake_session_id = waiting_req_bytes[3].decode("ascii")
        
        if room == "None":
            # 注册请求
            self.language_args_table[mooncake_session_id] = (
                EmbeddingArgsRegisterInfo.from_zmq(waiting_req_bytes)
            )
            # ...现有代码...
        else:
            room = int(room)
            
            # 解析消息
            dst_embedding_indices_str = waiting_req_bytes[4].decode("ascii")
            dst_embedding_indices = (
                [int(x) for x in dst_embedding_indices_str.split(",")]
                if dst_embedding_indices_str else []
            )
            required_dst_info_num = int(waiting_req_bytes[5].decode("ascii"))
            
            # 检查是否是resume请求（7个字段）
            is_resume = len(waiting_req_bytes) >= 8
            
            if is_resume:
                # Resume请求
                sent_tokens = int(waiting_req_bytes[6].decode("ascii"))
                allocated_tokens = int(waiting_req_bytes[7].decode("ascii"))
                
                # 更新现有transfer_info
                if room in self.transfer_infos and mooncake_session_id in self.transfer_infos[room]:
                    transfer_info = self.transfer_infos[room][mooncake_session_id]
                    transfer_info.sent_tokens = sent_tokens
                    transfer_info.allocated_tokens = allocated_tokens
                    transfer_info.dst_embedding_indices = dst_embedding_indices
                    
                    # 不重置status，保持当前状态（Transferring）
                    logger.info(
                        f"Resume transfer for room={room}, sent_tokens={sent_tokens}, "
                        f"allocated_tokens={allocated_tokens}"
                    )
                else:
                    logger.error(f"Cannot resume: room={room} not found in transfer_infos")
            else:
                # 初次请求
                allocated_tokens = int(waiting_req_bytes[6].decode("ascii"))
                
                if room not in self.transfer_infos:
                    self.transfer_infos[room] = {}
                
                self.transfer_infos[room][mooncake_session_id] = TransferEmbeddingInfo(
                    room=room,
                    endpoint=waiting_req_bytes[1].decode("ascii"),
                    dst_port=int(waiting_req_bytes[2].decode("ascii")),
                    mooncake_session_id=mooncake_session_id,
                    dst_embedding_indices=dst_embedding_indices,
                    required_dst_info_num=required_dst_info_num,
                    sent_tokens=0,
                    allocated_tokens=allocated_tokens,
                )
                
                # 标记为WaitingForInput
                if len(self.transfer_infos[room]) == required_dst_info_num:
                    self.update_status(room, KVPoll.WaitingForInput)
```

#### 4.3 修改`transfer_worker()`判断是否完整传输

```python
def transfer_worker(self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor):
    while True:
        try:
            embedding_chunk: TransferEmbeddingChunk = queue.get()
            # ...现有代码...
            
            for req in reqs_to_be_processed:
                # 获取allocated_tokens
                allocated_tokens = req.allocated_tokens
                sent_tokens = req.sent_tokens
                
                # 判断是否是最后一次传输
                remaining_tokens = embedding_chunk.total_tokens - sent_tokens
                is_last = (remaining_tokens <= allocated_tokens)
                
                # 计算实际要发送的token数
                tokens_to_send = min(remaining_tokens, allocated_tokens)
                
                block_size = self.data_args.aux_item_lens[1] // 4
                
                ret = self.send_embedding(
                    req.mooncake_session_id,
                    embedding_chunk.embedding_indices,
                    self.language_args_table[req.mooncake_session_id].dst_embedding_ptrs,
                    req.dst_embedding_indices,
                    tokens_to_send,  # 使用计算的token数
                    block_size,
                    sent_tokens,     # 传递已发送的token数
                )
                
                if ret != 0:
                    # ...错误处理...
                    break
                
                # 更新sent_tokens
                req.sent_tokens += tokens_to_send
                
                polls.append(True)
                dst_ranks_infos.append((req.endpoint, req.dst_port, req.room))
                
                # 根据is_last设置状态
                if len(polls) == req.required_dst_info_num:
                    if is_last:
                        # 完整传输完成
                        status = KVPoll.Success if all(polls) else KVPoll.Failed
                    else:
                        # 部分传输完成，等待resume
                        status = KVPoll.Transferring if all(polls) else KVPoll.Failed
                    
                    self.update_status(req.room, status)
                    
                    for endpoint, dst_port, room in dst_ranks_infos:
                        self.sync_status_to_language_endpoint(
                            endpoint, dst_port, room, status
                        )
            
            # 只有完全成功时才清理transfer_infos
            if (
                embedding_chunk.room in self.request_status
                and self.check_status(embedding_chunk.room) == KVPoll.Success
            ):
                if embedding_chunk.room in self.transfer_infos:
                    self.transfer_infos.pop(embedding_chunk.room)
        
        except Exception as e:
            raise RuntimeError(f"Transfer thread failed: {e}")
```

#### 4.4 添加防止重复传输的检查

```python
def add_transfer_request(
    self,
    bootstrap_room: int,
    embedding_indices: List[int],
    is_last: bool,
    total_tokens: int,
    block_size: int,
    sent_tokens: int = 0,  # 新增
):
    """Add block-based transfer request to queue."""
    assert self.disaggregation_mode == DisaggregationMode.ENCODE
    
    if bootstrap_room not in self.request_status or self.check_status(bootstrap_room) == KVPoll.Failed:
        return
    
    if bootstrap_room not in self.transfer_infos:
        return
    
    # 防止重复传输：如果status是Transferring且sent_tokens=0，说明已经在传输中
    current_status = self.check_status(bootstrap_room)
    if current_status == KVPoll.Transferring and sent_tokens == 0:
        logger.debug(f"Skip duplicate transfer for room={bootstrap_room}")
        return
    
    # ...现有代码...
    self.transfer_queues[shard_idx].put(
        TransferEmbeddingChunk(
            room=bootstrap_room,
            embedding_indices=embedding_indices,
            is_last=is_last,
            total_tokens=total_tokens,
            sent_tokens=sent_tokens,  # 新增
        )
    )
```

---

## 📊 完整流程示例

### 场景：实际2000 tokens，Language首次分配1024 tokens

```
时间线:

T0: Language侧
    └─ 分配8 blocks (1024 tokens)
    └─ init(embedding_indices=[0,1,2,3,4,5,6,7], allocated_tokens=1024)
    └─ Status: Bootstrapping -> WaitingForInput

T1: Embedding侧
    └─ 接收到init请求
    └─ actual_length = 2000 tokens
    └─ 分配16 blocks (2000 tokens)
    └─ 判断: is_last = (2000 <= 1024) = False
    └─ send_embedding(tokens=1024, is_last=False, sent_tokens=0)

T2: 第一次传输
    └─ Embedding -> Language: 1024 tokens
    └─ Status: WaitingForInput -> Transferring
    └─ aux_datas[0] = 2000 (实际总长度)

T3: Language侧检测到部分传输
    └─ poll() = Transferring
    └─ 读取aux_datas: actual_total_length = 2000
    └─ 计算: remaining = 2000 - 1024 = 976 tokens
    └─ 缓存已接收的1024 tokens
    └─ 释放旧分配的8 blocks
    └─ 重新分配8 blocks (976 tokens)

T4: Language侧发送Resume请求
    └─ resume_transfer(
          embedding_indices=[8,9,10,11,12,13,14,15],
          sent_tokens=1024,
          allocated_tokens=976
        )

T5: Embedding侧接收Resume请求
    └─ 更新transfer_info:
        - sent_tokens = 1024
        - allocated_tokens = 976
        - dst_embedding_indices = [8,9,10,11,12,13,14,15]
    └─ Status保持: Transferring (不重置)

T6: Embedding侧继续传输
    └─ 判断: is_last = (976 <= 976) = True
    └─ send_embedding(
          tokens=976,
          is_last=True,
          sent_tokens=1024
        )

T7: 第二次传输
    └─ Embedding -> Language: 976 tokens (从offset 1024开始)
    └─ Status: Transferring -> Success

T8: Language侧完成
    └─ poll() = Success
    └─ 合并数据: 1024 + 976 = 2000 tokens ✅
    └─ 处理请求
```

---

## 🔑 关键设计要点

### 1. Status转换规则
```
小数据（一次完成）:
  Bootstrapping -> WaitingForInput -> Success

大数据（需要Resume）:
  Bootstrapping -> WaitingForInput -> Transferring -> Success

失败:
  任意状态 -> Failed
```

### 2. 防止重复传输
```python
# 使用status和sent_tokens双重检查
if current_status == KVPoll.Transferring and sent_tokens == 0:
    return  # 已经在传输中，跳过
```

### 3. 数据一致性
- aux_datas在第一次传输时发送，包含实际总长度
- Language侧根据aux_datas判断是否需要resume
- 使用sent_tokens精确跟踪进度

### 4. 内存管理
- Language侧在resume前释放旧的分配
- Embedding侧保持transfer_infos直到完全成功
- 支持缓存部分数据

---

## 🧪 测试场景

### 1. 小数据（无需Resume）
```
实际: 500 tokens
默认: 1024 tokens
预期: 一次传输完成，Status: WaitingForInput -> Success
```

### 2. 大数据（需要Resume）
```
实际: 2000 tokens
默认: 1024 tokens
预期: 两次传输，Status: WaitingForInput -> Transferring -> Success
```

### 3. 极端情况
```
实际: 10000 tokens
默认: 1024 tokens
预期: 多次Resume? (取决于实现，建议单次Resume)
```

### 4. 失败场景
```
- Resume时内存不足
- 传输中断
- Session失效
预期: Status -> Failed，清理资源
```

---

## 📝 实现建议

### Phase 1: 核心Resume机制
1. ✅ 添加`sent_tokens`和`allocated_tokens`字段
2. ✅ 实现`resume_transfer()`方法
3. ✅ 修改消息格式支持resume
4. ✅ 修改status转换逻辑

### Phase 2: 数据管理
1. ✅ 实现部分数据缓存
2. ✅ 实现数据合并逻辑
3. ✅ 添加防止重复传输检查

### Phase 3: 错误处理
1. ✅ Resume时内存不足处理
2. ✅ 传输失败清理
3. ✅ Session管理

### Phase 4: 优化
1. 🔄 考虑是否支持多次Resume
2. 🔄 优化默认buffer大小策略
3. 🔄 添加监控和日志

---

## ❓ 讨论问题

1. **是否支持多次Resume?**
   - 当前设计支持单次Resume（两阶段传输）
   - 是否需要支持多次Resume？（如10000 tokens的场景）

2. **默认buffer大小策略**
   - 当前使用固定的8192 tokens
   - 是否需要根据历史请求动态调整？

3. **内存不足时的处理**
   - Resume时如果内存不足，是否应该：
     a) 等待释放后重试
     b) 立即失败
     c) 降级到更小的分配

4. **性能优化**
   - 是否需要预分配更大的buffer以减少Resume概率？
   - 是否需要异步处理Resume请求？

---

## 🎯 下一步

请review这个设计方案，特别关注：
1. Resume机制是否符合预期？
2. 数据流程是否清晰？
3. 是否有遗漏的场景？
4. 实现优先级是否合理？

确认后我将开始实现代码。
