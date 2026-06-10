
 ▐▛███▜▌   Claude Code v2.1.70
▝▜█████▛▘  glink/claude-opus-4-6 · API Usage Billing
  ▘▘ ▝▝    ~/Documents/theta_proj/ant_sglang

  Welcome to Opus 4.6

❯ 57cf4790ca2f86543bbd773cee15913a38a026d1                                  
  这个pr合并后，我在运行kimi-k2.5模型(启动命令                              
  SGLANG_USE_CUDA_IPC_TRANSPORT=1 SGLANG_MM_FEATURE_CACHE_MB=2048           
  SGLANG_VLM_CACHE_SIZE_MB=512 python -m sglang.launch_server --model-path  
  /home/admin/Kimi-K2.5/ --host 0.0.0.0 --port 8188 --trust-remote-code     
  --tp-size 8 --enable-cache-report  --log-level info                       
  --max-running-requests 32 --mem-fraction-static 0.9                       
  --chunked-prefill-size 8192  --attention-backend fa3                      
  --allow-auto-output-truncate --reasoning-parser kimi_k2                   
  --tool-call-parser kimi_k2 --mm-attention-backend fa3 --thinking-trigger  
  thinking --default-thinking)，会碰到如下报错：2026-03-30 01:19:47.574     
  ERROR 157662 [ TP2 scheduler.py:3632] Scheduler hit an exception:         
  Traceback (most recent call last):                                        
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]   File "/op 
  t/conda/lib/python3.10/site-packages/sglang/srt/managers/scheduler.py",   
  line 3628, in run_scheduler_process                                       
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]             
  scheduler.run_event_loop()                                                
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]   File "/op 
  t/conda/lib/python3.10/site-packages/sglang/srt/managers/scheduler.py",   
  line 1286, in run_event_loop                                              
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]             
  dispatch_event_loop(self)                                                 
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]   File "/op 
  t/conda/lib/python3.10/site-packages/sglang/srt/managers/scheduler.py",   
  line 3509, in dispatch_event_loop                                         
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]             
  scheduler.event_loop_overlap()                                            
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]   File      
  "/opt/conda/lib/python3.10/site-packages/torch/utils/_contextlib.py",     
  line 120, in decorate_context                                             
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]     return  
  func(*args, **kwargs)                                                     
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]   File "/op 
  t/conda/lib/python3.10/site-packages/sglang/srt/managers/scheduler.py",   
  line 1330, in event_loop_overlap                                          
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]             
  recv_reqs = self.recv_requests()                                          
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]   File "/op 
  t/conda/lib/python3.10/site-packages/sglang/srt/managers/scheduler.py",   
  line 1492, in recv_requests                                               
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]             
  recv_reqs = broadcast_pyobj(                                              
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]   File      
  "/opt/conda/lib/python3.10/site-packages/sglang/srt/utils/common.py",     
  line 1436, in broadcast_pyobj                                             
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]     data =  
  pickle.loads(serialized_data)                                             
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]   File "/op 
  t/conda/lib/python3.10/site-packages/sglang/srt/managers/mm_utils.py",    
  line 1678, in __setstate__                                                
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]             
  self._shm_handle = shared_memory.SharedMemory(name=self.shm_name)         
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]   File      
  "/opt/conda/lib/python3.10/multiprocessing/shared_memory.py", line 104,   
  in __init__                                                               
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]             
  self._fd = _posixshmem.shm_open(                                          
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]             
  FileNotFoundError: [Errno 2] No such file or directory: '/psm_5bef5080'   
  2026-03-30 01:19:47.574 ERROR 157662 [ TP2 scheduler.py:3632]             
  需要注意的是，纯语言输入是不报错的，输入带图的请求，才会报错。请分析下原  
  因和修复办法                                                              

⏺ Bash(git log --oneline 57cf4790ca2f86543bbd773cee15913a38a026d1 -1)
  ⎿  57cf4790ca [VLM] Optimize ShmPointerMMData for multi-pickle safety and
      deferred unwrap (#21465)

⏺ Bash(git show --stat 57cf4790ca2f86543bbd773cee15913a38a026d1)
  ⎿  Waiting…

⏺ Bash(git show 57cf4790ca2f86543bbd773cee15913a38a026d1)
  ⎿  commit 57cf4790ca2f86543bbd773cee15913a38a026d1
     Author: Yuhao Yang <47235274+yhyang201@users.noreply.github.com>
     Date:   Sat Mar 28 23:11:12 2026 +0800
     … +89 lines (ctrl+o to expand)

⏺ 让我看一下 broadcast_pyobj 和相关代码的实现。

⏺ Searched for 1 pattern, read 3 files (ctrl+o to expand)

⏺ Let me also check the tokenizer side and how ShmPointerMMData is created and
  consumed.

⏺ Searched for 2 patterns, read 1 file (ctrl+o to expand)

⏺ 现在我理解了问题的根因。让我确认一下我的分析。

⏺ Read 1 file (ctrl+o to expand)

⏺ 根因分析

  问题是一个竞态条件 (race condition)，出现在 broadcast_pyobj 和
  unwrap_shm_features 之间。

  时序分析

  这个 PR 的核心改动是：把 unwrap_shm_features 从 broadcast 之前 移到了
  broadcast 之后。这样 broadcast 时只序列化 shm_name（元数据），而不是整个
  tensor，让各 TP rank 直接通过共享内存名字打开同一段 shm，避免序列化大 tensor
  的开销。

  但问题在于 dist.broadcast 完成后，各 rank 是异步执行的：

  TP0 (rank 0):                         TP2 (rank 2):
  ─────────────                          ─────────────
  1. dist.broadcast(data) 完成           1. dist.broadcast(data) 完成
  2. return data (原始对象)               2. bytes → numpy 转换
  3. unwrap_shm_features()               3. pickle.loads() → __setstate__()
  4. materialize() → clone + UNLINK shm     ↑ 此时尝试 shm_open("psm_5bef5080")
                                             → FileNotFoundError!

  TP0 在 broadcast_pyobj 返回后直接进入 unwrap_shm_features → materialize() →
  unlink 共享内存。

  TP2 在 broadcast_pyobj 内部还在做 bytes(tensor_data.cpu().numpy()) +
  pickle.loads()，当 __setstate__ 尝试 SharedMemory(name="psm_5bef5080")
  打开这段共享内存时，已经被 TP0 删掉了。

  这就是为什么纯文本输入不报错（没有 ShmPointerMMData），只有带图请求才报错。

  修复方案

  需要在 unwrap_shm_features 之前加一个 dist.barrier()，确保所有 TP rank
  都完成了反序列化（即都已通过 __setstate__ 打开了 shm handle），之后再由任何
  rank 执行 materialize() 去 unlink。

  根据 POSIX 语义，shm_unlink 只移除文件系统条目，已经 open 的 fd
  仍然有效。所以只要所有 rank 在 unlink 之前都已 open，就不会有问题。

⏺ Searched for 2 patterns, read 1 file (ctrl+o to expand)

⏺ Now let me implement the fix:

⏺ Read 1 file (ctrl+o to expand)

⏺ barrier is already imported at line 38. Now let me apply the fix:

⏺ Update(python/sglang/srt/managers/scheduler.py)
  ⎿  Added 7 lines
      1516          # Unwrap shared memory features AFTER all broadcasts 
            complete,
      1517          # so that ShmPointerMMData metadata (not full tensor 
            data) is what
      1518          # gets serialized during broadcast_pyobj.
      1519 +        # A barrier is needed because broadcast_pyobj on the 
           +source rank                                                  
      1520 +        # returns the original objects immediately, while oth
           +er ranks are                                                 
      1521 +        # still deserializing (pickle.loads -> __setstate__ -
           +> shm_open).                                                 
      1522 +        # Without the barrier, the source rank can call mater
           +ialize() which                                               
      1523 +        # unlinks the shared memory before other ranks open i
           +t.                                                           
      1524          if recv_reqs:
      1525 +            if self.tp_size > 1:                             
      1526 +                barrier(group=self.tp_cpu_group)             
      1527              for req in recv_reqs:
      1528                  unwrap_shm_features(req)
      1529  

⏺ 修复已完成。让我验证一下修改后的代码:

⏺ Read 1 file (ctrl+o to expand)

⏺ 总结

  根因

  PR 57cf4790ca 把 unwrap_shm_features 从 broadcast 之前移到了 broadcast
  之后，目的是让 broadcast 时只传输 shm_name 元数据而不是完整 tensor，让各 TP
  rank 直接通过 POSIX 共享内存名打开同一段 shm。

  但这引入了一个竞态条件：

  1. dist.broadcast() 完成后，TP0（source rank）立即返回原始对象，无需反序列化
  2. TP0 迅速进入 unwrap_shm_features() → materialize() → shm_unlink() 
  删除共享内存
  3. 而此时 TP2 还在 broadcast_pyobj 内部做 bytes(tensor_data.cpu().numpy()) +
  pickle.loads() → __setstate__() → 尝试 shm_open("psm_5bef5080") →
  FileNotFoundError

  修复

  在 unwrap_shm_features 之前加一个 barrier(group=self.tp_cpu_group)，确保所有
  TP rank 都完成了 broadcast_pyobj（包括 pickle.loads 中的 __setstate__
  已经打开了 shm handle）之后，才允许任何 rank 执行 materialize() 去 unlink。

  根据 POSIX 语义，shm_unlink 只移除 /dev/shm/ 中的文件系统条目，已经通过
  shm_open 打开的文件描述符仍然有效。所以只要 barrier 保证所有 rank 在 unlink
  之前都已 open，后续读取不受影响。

✻ Cooked for 18m 56s

❯ 确认下是tp_cpu_group还是dp下的attn_tp_cpu_group等不同情况都考虑到了 

⏺ 好的，让我仔细审查 DP attention 下的所有 broadcast 路径。

⏺ Read 1 file (ctrl+o to expand)

⏺ 让我确认一下各 group 之间的关系：

⏺ Searched for 1 pattern (ctrl+o to expand)

⏺ Explore(Explore TP group relationships)
  ⎿  Done (21 tool uses · 51.7k tokens · 2m 21s)
  (ctrl+o to expand)

⏺ 根据分析结果，group 关系如下：

  tp_size = attn_tp_size × attn_cp_size × attn_dp_size

  - tp_group 包含所有 TP rank（全集）
  - attn_tp_group 和 attn_cp_group 是 tp_group 的子组

  例如 tp_size=8, attn_tp_size=4, attn_cp_size=2：
  tp_group:       {0,1,2,3,4,5,6,7}
  attn_tp_group:  {0,1,2,3}, {4,5,6,7}
  attn_cp_group:  {0,4}, {1,5}, {2,6}, {3,7}

  DP attention 路径分析

  DP attention 下的三步顺序 broadcast：

  Step 1: work_reqs via attn_tp_cpu_group  (rank0 → 1,2,3 打开 shm)
  Step 2: work_reqs via attn_cp_cpu_group  (rank0→4, rank1→5, rank2→6, rank3→7
  打开 shm)
  Step 3: control_reqs via tp_cpu_group    (无 ShmPointerMMData)

  关键推理：这三步是顺序执行的。每个 rank 必须完成 step N 才能进入 step N+1。而
  dist.broadcast 是 collective 操作——source rank 会阻塞直到所有组内 rank
  都完成接收。

  所以当 step 3 的 dist.broadcast 在任意 rank 上返回时，所有 rank 都已完成 step 
  1 和 step 2 的 pickle.loads（即所有 shm handle 已打开）。

  之后，barrier(group=self.tp_cpu_group) 再保证所有 rank 完成 step 3 的
  pickle.loads 后才开始 materialize()。

  结论：单个 barrier(group=self.tp_cpu_group) 覆盖了所有场景

  ┌───────────────────────┬─────────────────────────────────────────────────┐
  │         场景          │                   为什么正确                    │
  ├───────────────────────┼─────────────────────────────────────────────────┤
  │ 非 DP attention,      │ 只有一个 broadcast on tp_cpu_group，barrier     │
  │ tp_size>1             │ 直接保护                                        │
  ├───────────────────────┼─────────────────────────────────────────────────┤
  │ DP attention,         │ step 3 的 collective 隐式同步了 step            │
  │ attn_tp>1, attn_cp>1, │ 1/2，barrier 再同步 step 3                      │
  │  tp>1                 │                                                 │
  ├───────────────────────┼─────────────────────────────────────────────────┤
  │ DP attention,         │ step 2 跳过，step 3 的 collective 隐式同步 step │
  │ attn_tp>1,            │  1，barrier 同步 step 3                         │
  │ attn_cp==1, tp>1      │                                                 │
  ├───────────────────────┼─────────────────────────────────────────────────┤
  │                       │ attn_tp_size 和 attn_cp_size 必定也为 1，无     │
  │ tp_size==1            │ broadcast，无 race，if self.tp_size > 1         │
  │                       │ 正确跳过 barrier                                │
  └───────────────────────┴─────────────────────────────────────────────────┘

  当前的修复是正确的，不需要对 attn_tp_cpu_group/attn_cp_cpu_group 分别加
  barrier。

✻ Churned for 8m 27s

❯ 我不是说你要为attn_tp_cpu_group加barrier，而是tp_cpu_group和为attn_tp_cpu 
  _group是不是二选一的，非dp选前者，dp选后者？                              

⏺ 你说得对，这里有个重要问题。让我仔细分析 DP attention 下的情况。

  DP attention 下，不同 DP group 可能有不同的 
  work_reqs（一个有多模态请求，另一个没有），而 control_reqs 才是通过
  tp_cpu_group 统一广播的。这意味着：

  DP group 0 (ranks 0-3): recv_reqs = work_reqs_0 (非空) + control_reqs (空) →
  非空，进 barrier
  DP group 1 (ranks 4-7): recv_reqs = work_reqs_1 (空)   + control_reqs (空) →
  空，  跳 barrier
  → DEADLOCK!

  当前 if recv_reqs: 的守卫条件在 DP attention 场景下不安全。recv_reqs 在非 DP
  场景下各 rank 一定一致（同一个 broadcast_pyobj 的结果），但 DP 场景下不同 DP
  group 可能不一致。

  修复方案：把 barrier 移到 if recv_reqs: 外面，使其无条件执行。因为每轮 event
  loop 已经有 broadcast_pyobj 的 collective 操作，额外一个 gloo barrier
  开销可忽略。

⏺ Read 1 file (ctrl+o to expand)

⏺ Update(python/sglang/srt/managers/scheduler.py)
  ⎿  Added 6 lines, removed 2 lines
      1521          # still deserializing (pickle.loads -> __setstate__ -
            > shm_open).
      1522          # Without the barrier, the source rank can call mater
            ialize() which
      1523          # unlinks the shared memory before other ranks open i
            t.
      1524 +        # The barrier must be unconditional (outside `if recv
           +_reqs`) because                                              
      1525 +        # under DP-attention, different DP groups may have di
           +fferent work_reqs                                            
      1526 +        # (one non-empty, another empty), so recv_reqs can di
           +ffer across ranks.                                           
      1527 +        # A conditional barrier would deadlock in that case. 
      1528 +        if self.tp_size > 1:                                 
      1529 +            barrier(group=self.tp_cpu_group)                 
      1530          if recv_reqs:
      1525 -            if self.tp_size > 1:                             
      1526 -                barrier(group=self.tp_cpu_group)             
      1531              for req in recv_reqs:
      1532                  unwrap_shm_features(req)
      1533  

⏺ 验证最终结果：

⏺ Read 1 file (ctrl+o to expand)

⏺ 总结一下对你问题的回答：

  不需要在 DP 和非 DP 之间二选一 group。tp_cpu_group 在两种场景下都是正确的
  barrier group：

  1. 非 DP attention：broadcast 本身就用 tp_cpu_group，barrier
  也用它，完全对应。
  2. DP attention：虽然 work_reqs 是通过 attn_tp_cpu_group → attn_cp_cpu_group
  逐级 broadcast 的，但这三步 broadcast 是顺序执行的，且 step 3（control_reqs
  via tp_cpu_group）是全员 collective，隐式保证了前两步都已完成。barrier 在
  tp_cpu_group 上再同步一次，确保所有 rank 完成了 step 3 的
  pickle.loads。虽然它比严格必要的范围（per-DP-group）稍大，但 correctness
  没问题。

  真正需要修的是另一个问题：barrier 必须在 if recv_reqs: 外面无条件执行。因为 DP
   attention 下不同 DP group 的 recv_reqs
  可能不一致（一个有请求、另一个没有），放在 if recv_reqs: 里面会导致部分 rank
  进 barrier、部分 skip → 死锁。

✻ Cogitated for 5m 45s

❯ 我感觉还是不对，这样只要tp>1就无条件的barrier太粗放了；明明前面有根据if   
  self.server_args.enable_dp_attention: 和if self.attn_tp_rank == 0 and     
  self.attn_cp_rank == 0:等来区分请求是如何接收到的，为什么你就只判断tp呢   

⏺ 你说得对，我重新审视一下。

  关键在于 DP attention 路径下，step 3（control_reqs 通过 tp_cpu_group
  broadcast）本身就是一个 collective，它隐式提供了同步：所有 rank 必须完成 step
  1/2 的 pickle.loads（即完成 shm_open）后才能参与 step 3 的
  dist.broadcast。所以 DP 路径不需要额外 barrier。

  竞态条件只存在于非 DP 路径——只有一个 broadcast，source rank
  直接返回原始对象后就能跑到 materialize() 去 unlink。

  让我确认下这个推理：

⏺ Read 1 file (ctrl+o to expand)

⏺ 好，确认了 broadcast_pyobj 的行为：

  - Source rank (rank==src)：dist.broadcast 完成后直接 return data（原始对象）
  - Other ranks：dist.broadcast 完成后，还有 bytes() + pickle.loads() 才 return

  现在对两条路径做精确分析：

  非 DP 路径 (elif self.tp_size != 1)：只有一次 broadcast。Source rank
  返回后直接到 unwrap_shm_features，其他 rank 还在 pickle.loads → 有竞态，需要 
  barrier。

  DP attention 路径：三步顺序 broadcast：
  Step 1: work_reqs via attn_tp_cpu_group  (SHM在这里打开)
  Step 2: work_reqs via attn_cp_cpu_group  (SHM在这里打开)
  Step 3: control_reqs via tp_cpu_group    (无SHM，但是全员collective)
  Step 3 的 dist.broadcast(tensor_size) 是全员 collective——rank 0 会阻塞直到所有
   rank 都调用了它。而所有 rank 必须先完成 step 1/2 的 pickle.loads（即
  shm_open）才能进入 step 3。所以当 rank 0 从 step 3 返回时，所有 SHM handle
  已经打开 → POSIX 语义保护，unlink 不影响已有 handle → 不需要额外 barrier。

⏺ Update(python/sglang/srt/managers/scheduler.py)
  ⎿  Added 16 lines, removed 10 lines
      1516          # Unwrap shared memory features AFTER all broadcasts 
            complete,
      1517          # so that ShmPointerMMData metadata (not full tensor 
            data) is what
      1518          # gets serialized during broadcast_pyobj.
      1519 -        # A barrier is needed because broadcast_pyobj on the 
           -source rank                                                  
      1520 -        # returns the original objects immediately, while oth
           -er ranks are                                                 
      1521 -        # still deserializing (pickle.loads -> __setstate__ -
           -> shm_open).                                                 
      1522 -        # Without the barrier, the source rank can call mater
           -ialize() which                                               
      1523 -        # unlinks the shared memory before other ranks open i
           -t.                                                           
      1524 -        # The barrier must be unconditional (outside `if recv
           -_reqs`) because                                              
      1525 -        # under DP-attention, different DP groups may have di
           -fferent work_reqs                                            
      1526 -        # (one non-empty, another empty), so recv_reqs can di
           -ffer across ranks.                                           
      1527 -        # A conditional barrier would deadlock in that case. 
      1528 -        if self.tp_size > 1:                                 
      1519 +        #                                                    
      1520 +        # A barrier is needed for the non-DP-attention path b
           +ecause there is                                              
      1521 +        # only a single broadcast_pyobj on tp_cpu_group: the 
           +source rank                                                  
      1522 +        # returns the original objects immediately while othe
           +r ranks are still                                            
      1523 +        # in pickle.loads (-> __setstate__ -> shm_open).  Wit
           +hout a barrier                                               
      1524 +        # the source rank can call materialize() which unlink
           +s the shared                                                 
      1525 +        # memory before other ranks open it.                 
      1526 +        #                                                    
      1527 +        # Under DP-attention, no extra barrier is required: t
           +he control_reqs                                              
      1528 +        # broadcast on tp_cpu_group (step 3) is a collective 
           +that implicitly                                              
      1529 +        # forces every rank to finish the earlier attn_tp / a
           +ttn_cp work_reqs                                             
      1530 +        # deserializations (steps 1-2, which call shm_open) b
           +efore any rank                                               
      1531 +        # can return from step 3 and reach materialize().  PO
           +SIX guarantees                                               
      1532 +        # that shm_unlink only removes the name; already-open
           + handles stay                                                
      1533 +        # valid.                                             
      1534 +        if not self.server_args.enable_dp_attention and self.
           +tp_size > 1:                                                 
      1535              barrier(group=self.tp_cpu_group)
      1536          if recv_reqs:
      1537              for req in recv_reqs:

⏺ 验证最终代码：

⏺ Read 1 file (ctrl+o to expand)

⏺ 总结修改的逻辑：

  路径: 非 DP, tp>1
  broadcast 结构: 单次 broadcast_pyobj on tp_cpu_group
  是否需要 barrier: 需要
  原因: source rank 直接返回原始对象，其他 rank 还在
    pickle.loads→shm_open；无后续 collective 来隐式同步
  ────────────────────────────────────────
  路径: DP attention
  broadcast 结构: step1 attn_tp → step2 attn_cp → step3 tp
  是否需要 barrier: 不需要
  原因: step3 是全员 collective，所有 rank 进入 step3 前必已完成 step1/2 的
    pickle.loads（shm 已 open）；之后 unlink 不影响已有 handle（POSIX 语义）
  ────────────────────────────────────────
  路径: tp==1
  broadcast 结构: 无 broadcast
  是否需要 barrier: 不需要
  原因: 无并行，无竞态

✻ Cogitated for 6m 37s

❯ 造成这个问题的，是有多模输入的时候吧？另外，control_reqs也是不需要barrier 
  的吧？你再好好检查下你的代码                                              

⏺ 你说得对，让我重新审视。

  1. 只有多模态输入才有 ShmPointerMMData，纯文本请求没有 SHM，不存在竞态
  2. control_reqs 不包含多模态数据，也不需要 barrier
  3. 当前 barrier 在 if recv_reqs: 外面，每轮 event loop 无论有没有请求都会
  barrier，太粗放了

  在非 DP 路径下，recv_reqs 是从同一个 broadcast_pyobj 出来的，所有 rank 
  一致（全空或全非空），所以 if recv_reqs: 作为 guard 不会死锁。把 barrier 放回
  if recv_reqs: 里面即可：

⏺ Update(python/sglang/srt/managers/scheduler.py)
  ⎿  Added 16 lines, removed 17 lines
      1516          # Unwrap shared memory features AFTER all broadcasts 
            complete,
      1517          # so that ShmPointerMMData metadata (not full tensor 
            data) is what
      1518          # gets serialized during broadcast_pyobj.
      1519 -        #                                                    
      1520 -        # A barrier is needed for the non-DP-attention path b
           -ecause there is                                              
      1521 -        # only a single broadcast_pyobj on tp_cpu_group: the 
           -source rank                                                  
      1522 -        # returns the original objects immediately while othe
           -r ranks are still                                            
      1523 -        # in pickle.loads (-> __setstate__ -> shm_open).  Wit
           -hout a barrier                                               
      1524 -        # the source rank can call materialize() which unlink
           -s the shared                                                 
      1525 -        # memory before other ranks open it.                 
      1526 -        #                                                    
      1527 -        # Under DP-attention, no extra barrier is required: t
           -he control_reqs                                              
      1528 -        # broadcast on tp_cpu_group (step 3) is a collective 
           -that implicitly                                              
      1529 -        # forces every rank to finish the earlier attn_tp / a
           -ttn_cp work_reqs                                             
      1530 -        # deserializations (steps 1-2, which call shm_open) b
           -efore any rank                                               
      1531 -        # can return from step 3 and reach materialize().  PO
           -SIX guarantees                                               
      1532 -        # that shm_unlink only removes the name; already-open
           - handles stay                                                
      1533 -        # valid.                                             
      1534 -        if not self.server_args.enable_dp_attention and self.
           -tp_size > 1:                                                 
      1535 -            barrier(group=self.tp_cpu_group)                 
      1519          if recv_reqs:
      1520 +            # Barrier for the non-DP-attention path only: the
           +re is a single                                               
      1521 +            # broadcast_pyobj on tp_cpu_group where the sourc
           +e rank returns                                               
      1522 +            # the original objects immediately while other ra
           +nks are still in                                             
      1523 +            # pickle.loads (-> __setstate__ -> shm_open).  Wi
           +thout a barrier                                              
      1524 +            # the source can call materialize() / shm_unlink 
           +before others                                                
      1525 +            # open the segment.  recv_reqs is consistent acro
           +ss all ranks                                                 
      1526 +            # here (same broadcast), so the guard is deadlock
           +-free.                                                       
      1527 +            #                                                
      1528 +            # Under DP-attention no barrier is needed: the co
           +ntrol_reqs                                                   
      1529 +            # broadcast on tp_cpu_group (step 3) is a collect
           +ive that forces                                              
      1530 +            # every rank to complete the earlier attn_tp / at
           +tn_cp work_reqs                                              
      1531 +            # deserializations (steps 1-2, which call shm_ope
           +n) before any                                                
      1532 +            # rank returns from step 3.  POSIX guarantees shm
           +_unlink only                                                 
      1533 +            # removes the name; already-open handles stay val
           +id.                                                          
      1534 +            if not self.server_args.enable_dp_attention and s
           +elf.tp_size > 1:                                             
      1535 +                barrier(group=self.tp_cpu_group)             
      1536              for req in recv_reqs:
      1537                  unwrap_shm_features(req)
      1538  

⏺ Read 1 file (ctrl+o to expand)

⏺ 现在逻辑是：

  ┌───────────────────┬─────────────────────────────────────────────────────┐
  │       条件        │                    barrier 行为                     │
  ├───────────────────┼─────────────────────────────────────────────────────┤
  │ 无请求 (recv_reqs │ 跳过整个块，不 barrier                              │
  │  空)              │                                                     │
  ├───────────────────┼─────────────────────────────────────────────────────┤
  │ 非 DP, tp>1,      │ barrier on tp_cpu_group（非 DP 路径 recv_reqs 各    │
  │ 有请求            │ rank 一致，不会死锁）                               │
  ├───────────────────┼─────────────────────────────────────────────────────┤
  │ DP attention,     │ 不 barrier（step3 的 collective 已隐式同步了        │
  │ 有请求            │ step1/2 的 shm_open）                               │
  ├───────────────────┼─────────────────────────────────────────────────────┤
  │ tp==1             │ 不 barrier（单 rank 无竞态）                        │
  └───────────────────┴─────────────────────────────────────────────────────┘

  unwrap_shm_features 内部对非多模态请求（包括 control_reqs）是 no-op，不会触发
  materialize()/shm_unlink，所以对它们调用 barrier + unwrap
  无副作用，只是多模态请求实际需要这个保护。

✻ Sautéed for 4m 31s

❯ 生成一个github提交pr的description，英文的，包含motivation、modification等 

⏺ Bash(git diff python/sglang/srt/managers/scheduler.py)
  ⎿  diff --git a/python/sglang/srt/managers/scheduler.py b/python/sglang/s
     rt/managers/scheduler.py
     index 312c4b8c27..e8f0e82f6a 100644
     … +20 lines (ctrl+o to expand)

⏺ ---
  PR Description

  Title: [Bug][VLM] Fix shared memory race condition in ShmPointerMMData broadcast for multi-GPU VLM serving

  ---
  Motivation

  After #21465 (57cf479), running VLM models (e.g. Kimi-K2.5) with tp_size > 1 and multimodal (image) inputs crashes with:

  FileNotFoundError: [Errno 2] No such file or directory: '/psm_5bef5080'

  at ShmPointerMMData.__setstate__ during broadcast_pyobj on non-source TP ranks. Pure text inputs are unaffected.

  Root Cause

  PR #21465 moved unwrap_shm_features() from before broadcast_pyobj to after it, so that only the lightweight shm_name metadata is serialized during the TP broadcast instead of
   the full tensor. Each rank then opens the POSIX shared memory segment independently via __setstate__.

  However, broadcast_pyobj on the source rank returns the original objects immediately after dist.broadcast completes (no deserialization needed), while other ranks still need
  to run bytes() → pickle.loads() → __setstate__ → shm_open(). The source rank races ahead into unwrap_shm_features() → materialize() → shm_unlink(), removing the segment from
  /dev/shm/ before slower ranks can open it.

  Modification

  Add a dist.barrier() on tp_cpu_group after broadcast_pyobj and before unwrap_shm_features, ensuring all ranks have finished deserialization (and thus shm_open) before any
  rank calls materialize() / shm_unlink().

  The barrier is scoped to only fire when needed:

  ┌─────────────────────────────┬──────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │          Condition          │ Barrier? │                                                             Reason                                                             │
  ├─────────────────────────────┼──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │                             │          │ The subsequent control_reqs broadcast on tp_cpu_group (step 3) is a collective that implicitly forces all ranks to complete    │
  │ enable_dp_attention=True    │ No       │ the earlier attn_tp / attn_cp work_reqs deserializations (steps 1-2) before any rank can return from step 3. POSIX guarantees  │
  │                             │          │ shm_unlink only removes the name; already-open file descriptors remain valid.                                                  │
  ├─────────────────────────────┼──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ tp_size == 1                │ No       │ Single rank, no broadcast, no race.                                                                                            │
  ├─────────────────────────────┼──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ recv_reqs is empty          │ No       │ Nothing to unwrap; all ranks see the same empty list from broadcast_pyobj, so skipping is consistent and deadlock-free.        │
  ├─────────────────────────────┼──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Non-DP, tp_size > 1, has    │ Yes      │ Single broadcast_pyobj with no subsequent collective to synchronize deserialization.                                           │
  │ requests                    │          │                                                                                                                                │
  └─────────────────────────────┴──────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Fixes

  Fixes FileNotFoundError: [Errno 2] No such file or directory: '/psm_...' when serving VLM models with tp_size > 1 and shared-memory multimodal transport (non-default
  transport mode).

✻ Churned for 1m 12s

claude --resume a2209091-9cf3-4412-9baa-7f40338d3086                                                                                                                            
