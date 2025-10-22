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

#### 2.1 在`MultimodalLanguagePreallocQueue`中配置默认buffer大小

```python
class MultimodalLanguagePreallocQueue:
    def __init__(self, ...):
        # 现有代码...
        # 默认分配的token数（首次分配使用）
        self.default_allocate_tokens = int(
            os.getenv("SGLANG_EMBEDDING_DEFAULT_ALLOCATE_BUFFER_SIZE", "8192")
        )
        # 注意：resume逻辑在MultimodalLanguageTransferQueue中处理，不需要单独的队列
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

#### 3.1 保持`send_embedding_chunk()`不变

**重要说明**：`send_embedding_chunk()`中的`last_chunk`参数是为**chunk-prefill**设计的，不要与当前的**chunk-transfer（resume）**混用。

Resume传输逻辑完全在Connection层处理，`send_embedding_chunk()`在首次调用后不需要再次调用。

```python
def send_embedding_chunk(
    self: Scheduler,
    req: Req,
    last_chunk: bool = False,
):
    # 保持现有实现不变
    # 这个方法只在首次传输时调用一次
    assert last_chunk == True  # For embedding models, always send once
    
    if last_chunk:
        self.disagg_metadata_buffers.set_buf(req)
    
    # Send using block_indices
    req.disagg_embedding_sender.send_embedding(
        embedding_indices=req.embedding_indices,
        last_chunk=last_chunk,
        total_tokens=len(req.fill_ids),
        block_size=self.disagg_metadata_buffers.block_size,
    )
```

#### 3.2 修改`MooncakeEmbeddingSender.send_embedding()`

只在首次调用时触发transfer，resume由Language侧主动发起：
```python
class MooncakeEmbeddingSender(BaseKVSender):
    def send_embedding(
        self,
        embedding_indices: List[int] = None,
        last_chunk: bool = True,
        total_tokens: int = None,
        block_size: int = None,
    ):
        """Send embedding data to language instances using block-based transfer.
        
        Note: 
            - 这个方法只在首次传输时调用一次
            - Resume传输由Language侧通过resume_transfer()消息触发
            - Connection层会根据allocated_tokens自动判断是否需要部分传输
        
        Args:
            embedding_indices: List of source embedding indices
            last_chunk: Whether this is the last chunk (always True for embeddings)
            total_tokens: Total number of tokens to transfer
            block_size: Number of tokens per block
        """
        # 首次传输，sent_tokens=0
        self.embedding_mgr.add_transfer_request(
            self.bootstrap_room,
            embedding_indices,
            is_last=last_chunk,
            total_tokens=total_tokens,
            block_size=block_size,
        )
```

---

### 4. Connection层修改 (conn_multimodal.py)

#### 4.1 修改`send_embedding()`方法

**关键修改**：基于`allocated_tokens`而非block数量来判断buffer是否足够

```python
def send_embedding(
    self,
    mooncake_session_id: str,
    embedding_indices: List[int],
    dst_embedding_ptrs: list[int],
    dst_embedding_indices: List[int],
    total_tokens: int,
    block_size: int,
    sent_tokens: int = 0,         # 新增：已发送的token数
    allocated_tokens: int = None,  # 新增：Language侧分配的token数
):
    """Send embedding data using block-based transfer.
    
    Args:
        sent_tokens: Number of tokens already sent (for resume transfer)
        allocated_tokens: Number of tokens allocated by Language side
    """
    # 校验block_size一致性
    if allocated_tokens is not None:
        expected_block_size = allocated_tokens // len(dst_embedding_indices)
        if expected_block_size != block_size:
            raise ValueError(
                f"Block size mismatch: Embedding side uses {block_size}, "
                f"but Language side allocated {allocated_tokens} tokens "
                f"for {len(dst_embedding_indices)} blocks "
                f"(implies block_size={expected_block_size})"
            )
    else:
        # 向后兼容：如果没有allocated_tokens，用block数量计算
        allocated_tokens = len(dst_embedding_indices) * block_size
    
    # 基于allocated_tokens判断是否需要部分传输
    remaining_tokens = total_tokens - sent_tokens
    
    if remaining_tokens > allocated_tokens:
        # 需要部分传输
        logger.warning(
            f"Partial transfer: remaining={remaining_tokens} > "
            f"allocated={allocated_tokens}. Will transfer {allocated_tokens} tokens."
        )
        tokens_to_send = allocated_tokens
        is_partial = True
    else:
        # 可以完整传输
        tokens_to_send = remaining_tokens
        is_partial = False
    
    # 计算要发送的block范围
    start_block = sent_tokens // block_size
    embedding_indices_to_send = embedding_indices[start_block:]
    
    # 计算需要的dst block数量
    dst_blocks_needed = (tokens_to_send + block_size - 1) // block_size
    
    # 验证dst buffer是否足够
    if dst_blocks_needed > len(dst_embedding_indices):
        raise ValueError(
            f"Insufficient dst blocks: need {dst_blocks_needed} blocks "
            f"for {tokens_to_send} tokens, but only have {len(dst_embedding_indices)} blocks"
        )
    
    # 限制为实际需要的dst blocks
    dst_embedding_indices = dst_embedding_indices[:dst_blocks_needed]
    embedding_indices_to_send = embedding_indices_to_send[:dst_blocks_needed]
    
    src_addrs = []
    dst_addrs = []
    lengths = []
    
    # 记录实际传输的token数（用于返回）
    tokens_transferred = 0
    
    for block_idx, (src_block_idx, dst_block_idx) in enumerate(
        zip(embedding_indices_to_send, dst_embedding_indices)
    ):
        # Calculate tokens in this block
        remaining_in_transfer = tokens_to_send - tokens_transferred
        tokens_in_block = min(block_size, remaining_in_transfer)
        
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
        
        tokens_transferred += tokens_in_block
    
    ret = self.engine.batch_transfer_sync(
        mooncake_session_id, src_addrs, dst_addrs, lengths
    )
    
    # 返回传输结果和是否为部分传输
    return ret, is_partial
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
                # 获取allocated_tokens和sent_tokens
                allocated_tokens = req.allocated_tokens
                sent_tokens = req.sent_tokens
                
                block_size = self.data_args.aux_item_lens[1] // 4
                
                # 调用send_embedding，返回(ret, is_partial)
                ret, is_partial = self.send_embedding(
                    req.mooncake_session_id,
                    embedding_chunk.embedding_indices,
                    self.language_args_table[req.mooncake_session_id].dst_embedding_ptrs,
                    req.dst_embedding_indices,
                    embedding_chunk.total_tokens,  # 传递总token数
                    block_size,
                    sent_tokens,                   # 已发送的token数
                    allocated_tokens,              # Language侧分配的token数
                )
                
                if ret != 0:
                    # ...错误处理...
                    self.record_failure(
                        embedding_chunk.room,
                        f"Failed to send embedding chunk of {embedding_chunk.room} to {req.endpoint}:{req.dst_port}",
                    )
                    self.update_status(embedding_chunk.room, KVPoll.Failed)
                    self.sync_status_to_language_endpoint(
                        req.endpoint, req.dst_port, req.room, KVPoll.Failed
                    )
                    break
                
                # 更新sent_tokens
                tokens_sent = min(embedding_chunk.total_tokens - sent_tokens, allocated_tokens)
                req.sent_tokens += tokens_sent
                
                polls.append(True)
                dst_ranks_infos.append((req.endpoint, req.dst_port, req.room))
                
                # 根据is_partial设置状态
                if len(polls) == req.required_dst_info_num:
                    if is_partial:
                        # 部分传输完成，等待resume
                        status = KVPoll.Transferring if all(polls) else KVPoll.Failed
                        # 保留transfer_infos以支持resume
                    else:
                        # 完整传输完成
                        status = KVPoll.Success if all(polls) else KVPoll.Failed
                    
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
):
    """Add block-based transfer request to queue.
    
    Note:
        - 这个方法只在首次传输时调用（由send_embedding触发）
        - Resume传输由Language侧发送resume消息触发，不经过这个方法
        - sent_tokens信息在transfer_infos中维护
    """
    assert self.disaggregation_mode == DisaggregationMode.ENCODE
    assert is_last  # For embedding data, we only send once at the end
    
    if bootstrap_room not in self.request_status or self.check_status(bootstrap_room) == KVPoll.Failed:
        return
    
    if bootstrap_room not in self.transfer_infos:
        return
    
    # 防止重复传输：检查是否已经开始传输
    current_status = self.check_status(bootstrap_room)
    if current_status in [KVPoll.Transferring, KVPoll.Success]:
        logger.debug(f"Skip duplicate transfer for room={bootstrap_room}, status={current_status}")
        return
    
    # ...现有代码...
    self.transfer_queues[shard_idx].put(
        TransferEmbeddingChunk(
            room=bootstrap_room,
            embedding_indices=embedding_indices,
            is_last=is_last,
            total_tokens=total_tokens,
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

## 📝 实现计划

### Phase 1: 核心数据结构和消息协议
1. ✅ `TransferEmbeddingInfo`添加`sent_tokens`和`allocated_tokens`字段
2. ✅ 修改ZMQ消息格式支持resume（区分init和resume消息）
3. ✅ 修改`TransferEmbeddingInfo.from_zmq()`解析新字段

### Phase 2: Connection层实现
1. ✅ 修改`send_embedding()`方法
   - 添加`allocated_tokens`参数
   - 基于tokens而非blocks判断
   - 校验block_size一致性
   - 返回`(ret, is_partial)`
2. ✅ 修改`embedding_thread()`处理resume消息
3. ✅ 修改`transfer_worker()`根据`is_partial`设置status
4. ✅ 添加防止重复传输检查

### Phase 3: Language侧实现
1. ✅ `MooncakeEmbeddingReceiver.init()`添加`allocated_tokens`参数
2. ✅ 新增`MooncakeEmbeddingReceiver.resume_transfer()`方法
3. ✅ 修改`MultimodalLanguageTransferQueue.pop_transferred()`
   - 检测`KVPoll.Transferring`状态
   - 实现部分数据缓存
   - 触发resume传输
   - 实现数据合并逻辑

### Phase 4: Embedding侧适配（最小修改）
1. ✅ 确认`send_embedding_chunk()`保持不变
2. ✅ 确认`MooncakeEmbeddingSender.send_embedding()`只在首次调用

### Phase 5: 测试和文档
1. 🔄 单元测试：小数据（无Resume）
2. 🔄 单元测试：大数据（单次Resume）
3. 🔄 集成测试：实际模型场景
4. 🔄 错误场景测试：内存不足、传输失败
5. 🔄 更新用户文档和配置说明

---

## ❓ 讨论问题与决策

### ✅ 已确认的设计决策

1. **支持单次Resume，接口预留多次Resume能力**
   - 当前实现：单次Resume（两阶段传输）
   - 接口设计：支持sent_tokens追踪，可扩展为多次Resume
   - 理由：简化实现，满足大部分场景；为buffer不足情况预留扩展性

2. **基于allocated_tokens判断，而非block数量**
   - 使用`allocated_tokens`和`total_tokens`比较
   - 校验block_size一致性（allocated_tokens / block_num == block_size）
   - 支持未来不同block_size的扩展

3. **last_chunk不混用**
   - `last_chunk`仅用于chunk-prefill
   - Resume传输由`is_partial`标志控制
   - `send_embedding_chunk()`只在首次调用，不参与resume流程

4. **移除未使用的resume_queue**
   - Resume逻辑在`MultimodalLanguageTransferQueue.pop_transferred()`中处理
   - 不需要单独的队列

### 🔄 待讨论的问题

1. **默认buffer大小策略**
   - 当前：固定8192 tokens
   - 考虑：是否需要根据模型或历史请求动态调整？

2. **Resume时内存不足的处理**
   - 方案A：等待释放后重试（需要重试队列）
   - 方案B：立即失败（简单但可能影响成功率）
   - 方案C：降级到更小的分配（复杂，支持多次Resume）
   
   **建议**：先实现方案B（立即失败），后续可升级到方案A

---

## 🔧 关键修正总结

根据反馈，已完成以下关键修正：

### 1. ✅ 移除未使用的resume_queue
- Resume逻辑直接在`MultimodalLanguageTransferQueue.pop_transferred()`中处理
- 不需要额外的队列

### 2. ✅ 澄清last_chunk的用途
- `last_chunk`仅用于chunk-prefill，不与chunk-transfer混用
- `send_embedding_chunk()`只在首次调用，不参与resume流程
- Resume由Connection层的`is_partial`标志控制

### 3. ✅ 基于allocated_tokens判断
- 修改buffer验证逻辑：使用`allocated_tokens`而非`len(embedding_indices)`
- 添加block_size一致性校验
- 支持不同block_size的扩展（虽然当前默认一致）

### 4. ✅ 单次Resume + 扩展接口
- 当前实现：单次Resume（两阶段传输）
- 接口设计：支持`sent_tokens`追踪，预留多次Resume能力
- 为buffer不足场景预留扩展性

## 🎯 下一步

设计方案已根据反馈完成修正，请确认：
1. ✅ Resume机制是否符合预期？
2. ✅ last_chunk和is_partial的区分是否清晰？
3. ✅ 基于allocated_tokens的验证逻辑是否正确？
4. ✅ 接口设计是否支持未来扩展？

**确认后即可开始实现代码**。
