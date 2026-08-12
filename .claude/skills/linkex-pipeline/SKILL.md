---
name: linkex-pipeline
description: LinkEX (ACI) 流水线管理 — 执行流水线、查看/诊断/重试流水线、拉取日志。使用 `aci` CLI 操作 CI/CD 流水线。
---

# LinkEX (ACI) 流水线管理

使用 `aci` CLI 管理 CI/CD 流水线，涵盖执行、查看状态、重试失败 job、拉取日志、诊断问题等场景。

## 前置检查

每次使用本 skill 时必须先执行以下检查：

### 1. 检查 aci CLI 是否存在

```bash
which aci || echo "aci CLI 未安装"
```

如果 aci 不存在，提示用户安装 ACI CLI。安装参考：https://antcli.alipay.com/marketplace/?name=aci-cli&tab=all&from=atc

### 2. 检查 aci 是否已登录

```bash
aci auth status
```

如果输出显示未登录，执行：

```bash
aci auth login
```

按提示完成认证。认证通过后才能进行后续操作。

---

## 场景一：执行流水线

```bash
# 在当前仓库目录下，用仓库中的 .aci YAML 触发
aci pipeline run --yaml .aci/xxx.yaml

# 指定分支
aci pipeline run --yaml .aci/xxx.yaml --branch <branch>

# 传递自定义参数
aci pipeline run --yaml .aci/xxx.yaml --param KEY=VALUE
```

执行后会输出 pipeline ID，用于后续查看/诊断。

---

## 场景二：查看与监听流水线

```bash
# 列出最近的流水线
aci pipeline list

# 查看流水线详情（表格形式）
aci pipeline show <pipeline-id>

# JSON 格式输出
aci pipeline show <pipeline-id> -o json

# 实时监听流水线状态（自动刷新）
aci pipeline watch <pipeline-id>

# 等待流水线结束
aci pipeline wait <pipeline-id>
```

---

## 场景三：重试与取消流水线

```bash
# 重试整个流水线（自动重试所有失败的 job）
aci pipeline retry <pipeline-id>

# 取消运行中的流水线
aci pipeline cancel <pipeline-id>
```

### 重试单个 Job

```bash
# 查看 job 详情找到 job ID
aci pipeline show <pipeline-id> -o json

# 重试特定 job
aci job retry <job-id>

# 重新执行 job（即使成功也可以重跑）
aci job rerun <job-id>
```

### 确认 CONFIRM 状态的 Job

```bash
# 确认执行需要人工确认的 job
aci job confirm <job-id>

# 跳过 job
aci job skip <job-id>
```

---

## 场景四：拉取流水线日志

```bash
# 获取单个 job 的日志
aci job log <job-id>

# 获取 job 调度日志
aci job scheduler-log <job-id>

# 查看 job 详情（含状态、耗时等）
aci job show <job-id>
```

---

## 场景五：诊断流水线

### 5.1 整体状态诊断

```bash
# 获取流水线完整信息（JSON）
aci pipeline show <pipeline-id> -o json

# 列出所有流水线，找最近的失败流水线
aci pipeline list
```

### 5.2 逐 Job 诊断

```bash
# 1. 先获取流水线详细 JSON
aci pipeline show <pipeline-id> -o json > /tmp/pipeline.json

# 2. 用 Python 分析各 stage/job 状态
python3 -c "
import json
with open('/tmp/pipeline.json') as f:
    data = json.load(f)
for stage in data.get('stageExecutions', []):
    sn = stage.get('name', '?')
    for job in stage.get('jobExecutions', []):
        jn = job.get('name', '?')
        js = job.get('jobStatus', '?')
        jid = job.get('id')
        print(f'[{js}] {sn} / {jn}  jobId={jid}')
"
```

### 5.3 查看 Job 详情与产物

```bash
# job 详情
aci job show <job-id> -o json

# 列出流水线产物
aci pipeline artifacts <pipeline-id>
```

### 5.4 拉取失败 Job 日志

```bash
# 查看失败 job 的日志，定位错误
aci job log <job-id>

# 查看调度日志，排查调度层面问题
aci job scheduler-log <job-id>
```

---

## 常用命令速查

| 操作 | 命令 |
|------|------|
| 前置检查 | `which aci && aci auth status` |
| 登录 | `aci auth login` |
| 执行流水线 | `aci pipeline run --yaml .aci/xxx.yaml` |
| 列出流水线 | `aci pipeline list` |
| 查看流水线 | `aci pipeline show <id>` |
| 监听流水线 | `aci pipeline watch <id>` |
| 等待结束 | `aci pipeline wait <id>` |
| 重试流水线 | `aci pipeline retry <id>` |
| 取消流水线 | `aci pipeline cancel <id>` |
| 查看 job | `aci job show <job-id>` |
| 重试 job | `aci job retry <job-id>` |
| 确认 job | `aci job confirm <job-id>` |
| 跳过 job | `aci job skip <job-id>` |
| 拉取日志 | `aci job log <job-id>` |
| 调度日志 | `aci job scheduler-log <job-id>` |
| 查看产物 | `aci pipeline artifacts <id>` |

---

## 常见问题

### aci 命令找不到？

检查 aci CLI 是否安装：`which aci`。如未安装，安装参考：https://atc.alipay.com/atc/cli

### aci 未登录？

```bash
aci auth login
```

### 流水线 CANCELED 了怎么办？

用 `aci pipeline retry <id>` 重试。

### Job 卡在 CONFIRM 状态？

```bash
aci job confirm <job-id>
```

### 想看某个 job 为什么失败？

```bash
aci job log <job-id>
```

### 如何查看流水线的 YAML 定义？

```bash
aci pipeline yaml <pipeline-id>
```

---

## SGLang 镜像构建流水线

本仓库 `docker/ant/` 目录下有三条 ACI 流水线，用于构建 SGLang 引擎镜像。YAML 文件位于 `docker/ant/aci/`。

### 流水线概览

| 流水线 | YAML 路径 | 用途 |
|--------|-----------|------|
| 完整构建 | `docker/ant/aci/sglang_runtime.aci.yml` | 从源码完整编译构建生产镜像 |
| 快速覆盖 | `docker/ant/aci/sglang_fast.aci.yml` | 基于已有镜像快速覆盖安装新 whl 包 |
| 编译镜像 | `docker/ant/aci/sglang_prepare_compile_image.aci.yml` | 创建编译 sglang/sglang-kernel wheel 包的编译镜像 |

### 常用构建流程

1. **完整构建**：通过 `sglang_runtime.aci.yml` 从源码编译 sglang、sglang-kernel，再构建运行时镜像。构建时间较长，一般约 2 小时。

2. **快速迭代**：先用 `sglang_runtime.aci.yml` 只编译 whl 包（`build_whl_only: "true"`），再触发 `sglang_fast.aci.yml` 基于已有基础镜像覆盖安装新的 whl 包。
   > **注意**：`sglang_fast.aci.yml` 使用的基础镜像需与 whl 包的依赖（cuda、torch 等）兼容。

---

### sglang_runtime.aci.yml（完整构建流水线）

从源码完整编译 sglang + sglang-kernel wheel 包，并构建运行时镜像。

**YAML 路径**：`docker/ant/aci/sglang_runtime.aci.yml`

#### 流水线阶段

| 阶段 | 说明 |
|------|------|
| Verify-Parameters | 参数校验和转换 |
| Build-Sglang-Build-Image | 构建 sglang-kernel 编译镜像 |
| Build-Wheels | 编译 sglang_kernel 和 sglang wheel 包 |
| Build-Image | 构建运行时镜像 |
| STC-Scan | 安全扫描 |
| Image-Scan | 镜像扫描 |
| Push-Image | 推送镜像至多集群 |

#### 常用构建场景

##### 场景 A：从源码完整构建生产镜像

```bash
aci pipeline run --yaml docker/ant/aci/sglang_runtime.aci.yml --branch <branch> --param build_whl_only=false
```

> 注意：不能同时设置 `build_sglang_whl_only=true`，否则 sglang-kernel whl 包会是空文件。

##### 场景 B：只编译 whl 包（调试 / 配合快速构建）

默认行为，直接执行即可：

```bash
aci pipeline run --yaml docker/ant/aci/sglang_runtime.aci.yml --branch <branch>
```

该流水线默认 `build_whl_only: "true"`，仅执行编译阶段，输出 sglang_kernel 和 sglang wheel 包，不构建镜像。

##### 场景 C：只编译 sglang whl 包（快速编译）

```bash
aci pipeline run --yaml docker/ant/aci/sglang_runtime.aci.yml --branch <branch> --param build_sglang_whl_only=true
```

跳过 sglang_kernel 编译（mock 空包），只编译 sglang wheel 包。

##### 场景 D：使用预编译 wheel 包构建镜像

```bash
aci pipeline run --yaml docker/ant/aci/sglang_runtime.aci.yml --branch <branch> \
  --param skip_build_stage=true \
  --param build_whl_only=false \
  --param sglang_kernel_whl_url="https://xxx/sglang_kernel-xxx.whl" \
  --param sglang_whl_url="https://xxx/sglang-xxx.whl"
```

跳过编译阶段，直接使用指定 URL 的 wheel 包构建运行时镜像。

#### 参数说明

##### 构建控制参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `build_whl_only` | `true` | 仅编译 wheel 包，不构建镜像 |
| `build_sglang_whl_only` | `false` | 仅编译 sglang wheel 包，mock sglang_kernel |
| `skip_build_stage` | `false` | 跳过编译阶段，使用预编译 wheel 包 |

##### CUDA / Python 版本

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cuda_version_full` | `13.0.1` | 完整 CUDA 版本（`12.9.1` 或 `13.0.1`） |
| `torch_version` | `2.11.0` | PyTorch 版本，改非 2.9.1 版本时需同时配 cu_tag |
| `python_version` | `3.10` | Python 版本（仅影响编译，与 runtime 中 Python 无关） |

##### 编译参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_jobs` | `24` | ninja 编译并发度 |
| `nvcc_threads` | `4` | nvcc 编译并发度 |

##### Wheel 包 URL 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `deepgemm_whl_url` | `""` | DeepGEMM whl（已内置到 sglang-kernel，老版本才需要） |
| `flashmla_whl_url` | `""` | FlashMLA whl（可选） |
| `deepep_whl_url` | `""` | DeepEP whl（可选） |
| `transfer_engine_whl_url` | `""` | Mooncake Transfer Engine whl |
| `kvpool_whl_url` | `""` | 蚂蚁内部 kvpool whl（--no-deps 安装） |

##### 基础镜像与源

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `runtime_base_image` | `reg.docker.alibaba-inc.com/antos/ubuntu-ai-x86_64-ngc` | 运行时基础镜像 |
| `ubuntu_mirror` | `http://mirrors.cloud.aliyuncs.com` | ubuntu 镜像源 |
| `pip_default_index` | `https://pypi.antfin-inc.com/simple` | pip 安装源 |
| `rustup_dist_mirror` | `https://mirrors.aliyun.com/rustup` | rustup 安装源 |
| `github_artifactory` | `github.ednovas.xyz/https://github.com` | GitHub 镜像代理 |

##### 功能开关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `install_flashinfer_jit_cache` | `1` | 是否预下载 FlashInfer JIT 缓存 |
| `enable_below_sm90` | `ON` | 是否支持 SM90 以下 GPU |
| `arch` | `x86_64` | CPU 架构（`x86_64` 或 `aarch64`） |
| `image_build_target` | `framework_final` | 镜像目标（`runtime` 精简 / `framework_final` 完整） |

##### xruntime 版本

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `runtime_version` | `2.0.0` | xruntime 组件版本 |
| `runtime_llm_version` | `2.0.0` | xruntime LLM 组件版本 |

---

### sglang_fast.aci.yml（快速覆盖构建流水线）

基于已有镜像，快速覆盖安装新版本的 sglang 和 sglang-kernel whl 包，减少镜像构建时间。

**YAML 路径**：`docker/ant/aci/sglang_fast.aci.yml`

**模板参考**：https://linkex.alipay.com/project/161500101/template?tenant_path=alipay&current=1&star=false&templateTab=edit&templateId=174800029

#### 流水线阶段

| 阶段 | 说明 |
|------|------|
| Build-Image | 基于已有镜像覆盖安装 whl 包 |
| STC-Scan | 安全扫描 |
| Image-Scan | 镜像扫描 |
| Push-Image | 推送镜像至多集群 |

#### 触发命令

```bash
aci pipeline run --yaml docker/ant/aci/sglang_fast.aci.yml --branch <branch> \
  --param sglang_kernel_whl_url="https://xxx/sglang_kernel-xxx.whl" \
  --param sglang_whl_url="https://xxx/sglang-xxx.whl"
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sglang_whl_url` | `""` | sglang wheel 包下载地址 |
| `sglang_kernel_whl_url` | `""` | sglang_kernel wheel 包下载地址 |
| `runtime_base_image` | `reg.docker.alibaba-inc.com/sglang/deepep-base` | 已制作好的 runtime 基础镜像 |
| `runtime_base_image_tag` | `5f20a4c5-20260719115403` | 基础镜像 tag（已测试 dsv3.1，基于 commit 5f20a4c5） |

> **注意**：
> - 预编译的 whl 包和 base 镜像中的 sglang/sglang-kernel 依赖组件版本必须一致。
> - 默认基础镜像基于 commit `1ef14f05`，使用时确保 whl 包依赖与此 commit 中的 `pyproject.toml` 声明一致。

#### 构建流程（beforeScript 中动态生成 Dockerfile.fast）

1. 基于 `runtime_base_image:runtime_base_image_tag` 启动
2. 安装系统依赖（jq、ffmpeg、ossutil）
3. 安装 Python 依赖（scipy、distro、cryptography、modelscope 等）
4. 覆盖安装 sglang whl 包
5. 覆盖安装 sglang-kernel whl 包
6. 安装蚂蚁内部 decord2 解码包
7. 校验 flashinfer-jit-cache / flashinfer-cubin 版本一致性

#### 输出镜像

- **镜像地址**：`reg.docker.alibaba-inc.com/sglang/deepep-base:<commitSha>-<timestamp>`
- **上海站点**：`acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/sglang/deepep-base:<commitSha>-<timestamp>`
- **北京站点**：`acr-bj-ant-registry-vpc.cn-beijing.cr.aliyuncs.com/sglang/deepep-base:<commitSha>-<timestamp>`

---

### sglang_prepare_compile_image.aci.yml（编译镜像流水线）

用于创建蚂蚁内部编译 sglang 和 sglang-kernel wheel 包的 docker 编译镜像。

**YAML 路径**：`docker/ant/aci/sglang_prepare_compile_image.aci.yml`

#### 流水线阶段

| 阶段 | 说明 |
|------|------|
| Build-Image | 构建编译镜像（基于 Dockerfile.compile） |
| STC-Scan | 安全扫描 |
| Image-Scan | 镜像扫描 |

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cuindex` | `cu130` | torch 对应的 CUDA 版本标识 |
| `cuda_version` | `13.0` | CUDA 版本（`12.9` 或 `13.0`） |
| `python_tag` | `cp312-cp312` | Python 环境标签 |

#### 输出镜像

- **镜像地址**：`reg.docker.alibaba-inc.com/sglang/theta_sglang_build`
- **镜像标签**：`<python_tag>-cuda<cuda_version>`

---

### 相关 Dockerfile

| 文件 | 用途 |
|------|------|
| `docker/Dockerfile` | 多阶段构建，用于 `sglang_runtime.aci.yml` |
| `docker/Dockerfile.compile` | 编译镜像，用于 `sglang_prepare_compile_image.aci.yml` |
| `python/sglang/kernels/aot/Dockerfile` | sglang-kernel 编译依赖镜像 |

### 参考文档

> **⚠️ 路径变更记录**：sglang-kernel 相关文件已从仓库根目录的 `sgl-kernel/` 迁移至 `python/sglang/kernels/aot/`。如果在新路径下找不到 `build.sh` 或 `Dockerfile`，请先检查 `python/sglang/kernels/aot/` 目录。查找规则：
> 1. 优先查找 `python/sglang/kernels/aot/`（新路径）
> 2. 如果不存在，回退查找 `sgl-kernel/`（旧路径，已废弃）
> 3. 涉及的关键文件对应关系：
>    - `sgl-kernel/Dockerfile` → `python/sglang/kernels/aot/Dockerfile`
>    - `sgl-kernel/build.sh` → `python/sglang/kernels/aot/build.sh`
>    - `sgl-kernel/rename_wheels.sh` → `python/sglang/kernels/aot/rename_wheels.sh`

- 语雀文档：[sglang 引擎镜像制作流水线说明](https://yuque.antfin.com/hegpb4/kg7h1z/lvyiqs9tuxcem3hw)
- LinkEX 项目：https://linkex.alipay.com/project/161500101

---

## sglang_runtime.aci.yml 深度分析

### 涉及的文件与依赖关系

```
sglang_runtime.aci.yml
├── Stage 1: Verify-Parameters (shell-with-clone-and-output)
│   └── 读取: python/sglang/kernels/aot/Dockerfile (生成 DEPS_TAG)
│
├── Stage 2: Build-Sglang-Build-Image (docker-image-build)
│   └── 构建: python/sglang/kernels/aot/Dockerfile (target: deps)
│       └── 输出: reg.docker.alibaba-inc.com/sglang/theta_sglang_build:<DEPS_TAG>
│
├── Stage 3: Build-Wheels (ant-cxx-build)
│   ├── 使用镜像: theta_sglang_build:<DEPS_TAG>
│   ├── 执行: python/sglang/kernels/aot/build.sh (编译 sglang-kernel whl)
│   ├── 执行: python/pyproject.toml (编译 sglang whl)
│   ├── 辅助: python/sglang/kernels/aot/rename_wheels.sh (重命名 whl，添加 CUDA 后缀)
│   ├── 辅助: proxy.py (mitmproxy 代理，从 dmsint 下载)
│   ├── 依赖: ~/.ssh ssh_key (从 dmsint 获取，用于 git fetch origin_ssh)
│   └── 输出: sglang_kernel whl + sglang whl
│
├── Stage 4: Build-Image (docker-image-build)
│   └── 构建: docker/Dockerfile (target: framework_final 或 runtime)
│       └── 输出: reg.docker.alibaba-inc.com/sglang/deepep-base:<sha>-<ts>
│
├── Stage 5: STC-Scan (stc)
├── Stage 6: Image-Scan (image-scan)
└── Stage 7: Push-Image × 3
    ├── image-sync-from-alipay-to-ats → 上海 S 集群
    ├── image-sync-from-alipay-to-ats → 北京 B1 集群
    └── push-image-to-keeper → 镜像信息登记
```

### 各阶段详细逻辑

---

#### Stage 1: Verify-Parameters

**组件**: `shell-with-clone-and-output`

**逻辑**:
1. 从 `cuda_version_full` 参数推导 `CUDA_VERSION`（12.9 或 13.0）
2. 根据 `arch` 选择基础编译镜像：
   - `x86_64` → `registry.cn-hangzhou.aliyuncs.com/augusto/manylinux-builder`
   - `aarch64` → `registry.cn-hangzhou.aliyuncs.com/augusto/manylinuxaarch64-builder`
3. 根据 CUDA 版本映射 runtime 基础镜像 tag：
   - CUDA 12.9.1 → `25.06`
   - CUDA 13.0.1 → `25.08`
4. 通过 `git log -n 1 -- ./python/sglang/kernels/aot/Dockerfile` 获取最近修改该文件的 commit 短 hash
5. 生成 `DEPS_TAG`：`cuda<CUDA_VERSION>-<PY_TAG>-<SHORT_HASH>-<ARCH>`
   - 例如：`cuda13.0-cp310-cp310-a1b2c3d-x86_64`
6. 输出 JSON 供后续 stage 使用

**常见错误**:
- `Unsupported CUDA version` → 检查 `cuda_version_full` 参数是否合法（仅支持 12.9.1 / 13.0.1）
- `git log` 失败 → 确保仓库包含 python/sglang/kernels/aot/ 目录（旧路径 sgl-kernel/ 已废弃）

---

#### Stage 2: Build-Sglang-Build-Image

**组件**: `docker-image-build`
**Dockerfile**: `python/sglang/kernels/aot/Dockerfile`
**Target**: `deps`

**构建参数传递链**:
```
参数                    → Dockerfile ARG
BASE_IMG               → BASE_IMG
CUDA_VERSION           → CUDA_VERSION
ARCH                   → ARCH
python_version         → PYTHON_VERSION
py_tag                 → PYTHON_TAG
github_artifactory     → GITHUB_ARTIFACTORY
pip_default_index      → PIP_DEFAULT_INDEX
yum_mirror             → YUM_MIRROR
```

**Dockerfile deps 阶段逻辑** (`python/sglang/kernels/aot/Dockerfile`):
1. `FROM ${BASE_IMG}:cuda${CUDA_VERSION}`
2. 配置 yum 镜像源（如果指定了 `YUM_MIRROR`）
3. 安装系统依赖：gcc, gcc-c++, make, numactl-devel, libibverbs, zstd-devel, xxhash-devel
4. 下载安装 CMake（3.31.x，支持缓存复用）
5. 编译安装 ccache 4.12.1（支持 CUDA 编译缓存）
6. 建立 libcuda.so stub 链接
7. 安装 Python 依赖：torch, ninja, setuptools, wheel, numpy, uv, scikit-build-core

**关键配置**:
- `skipWhenImageExists: true` — 如果 DEPS_TAG 对应的镜像已存在则跳过，加速构建
- `cacheStrategy: always` — 使用 Docker 构建缓存
- `nydusSwitch: true` — 构建 nydus 加速镜像
- `timeoutInSec: 6000` — 超时 100 分钟

**常见错误**:
- CMake 下载失败 → `GITHUB_ARTIFACTORY` 代理不可用，检查 `github.ednovas.xyz` 连通性
- ccache 编译失败 → xxhash-devel 或 libzstd-devel 未正确安装
- pip install torch 失败 → `PIP_DEFAULT_INDEX` 或 `PYTORCH_INDEX_BASE` 不可达

---

#### Stage 3: Build-Wheels（核心编译阶段）

**组件**: `ant-cxx-build`
**镜像**: `reg.docker.alibaba-inc.com/sglang/theta_sglang_build:<DEPS_TAG>`
**超时**: 28800 秒（8 小时）
**跳过条件**: `skip_build_stage=true`

**编译环境准备**:
1. 设置 OSS 环境变量（用于产物上传）：`OSS_URL`, `OSS_AI`, `OSS_AK`
2. 建立 ccache 软链接到 workspace 缓存路径 `/.cache/daily/cuda-<ver>`
3. 下载并启动 mitmproxy 代理，用于拦截 GitHub 请求转发到内部 OSS
4. 配置 SSH key（从 dmsint 获取），添加 `origin_ssh` 远程
5. 配置大量 `git config --global url` 重定向，将 GitHub 依赖替换为 gitee 镜像或蚂蚁内部仓库

**Git 依赖重定向表**（编译阶段关键 — 任一仓库不可达都会导致编译失败）:

| 原始地址（GitHub） | 替换地址 |
|---|---|
| `github.com/NVIDIA/cutlass` | `gitee.com/staugust/cutlass` |
| `github.com/NVIDIA/nccl` | `gitee.com/mirrors/nccl` |
| `github.com/fmtlib/fmt` | `gitee.com/mirrors/fmt` |
| `github.com/intel/mkl-dnn` | `gitee.com/mirrors/mkl-dnn` |
| `github.com/gflags/gflags` | `gitee.com/mirrors/gflags` |
| `github.com/sgl-project/flashinfer` | `code.alipay.com:inference/flashinfer` |
| `github.com/sgl-project/sgl-attn` | `code.alipay.com:inference/sgl-attn` |
| `github.com/sgl-project/DeepGEMM` | `code.alipay.com:inference/DeepGEMM` |
| `github.com/sgl-project/FlashMLA` | `code.alipay.com:inference_thirdparty/FlashMLA` |
| `github.com/deepseek-ai/DeepGEMM` | `code.alipay.com:inference/DeepGEMM` |
| `github.com/fla-org/flash-linear-attention` | `code.alipay.com:Theta/flash-linear-attention` |
| `github.com/triton-lang/triton` | `code.alipay.com:inference_thirdparty/triton` |
| `github.com/Dao-AILab/flash-attention` | `code.alipay.com:inference_thirdparty/flash-attention` |
| `github.com/dmlc/dlpack` | `code.alipay.com:inference_thirdparty/dlpack` |
| `github.com/gabime/spdlog` | `code.alipay.com:inference_thirdparty/spdlog` |
| `github.com/google/googletest` | `code.alipay.com:inference_thirdparty/googletest` |
| `github.com/microsoft/mscclpp` | `code.alipay.com:inference_thirdparty/mscclpp` |
| `github.com/Tessil/robin-map` | `code.alipay.com:inference_thirdparty/robin-map` |
| `github.com/wjakob/nanobind` | `code.alipay.com:inference_thirdparty/nanobind` |
| `github.com/NVIDIA/cccl` | `code.alipay.com:inference_thirdparty/cccl` |
| `github.com/NVIDIA/nvbench` | `code.alipay.com:inference_thirdparty/nvbench` |
| `github.com/ROCm/composable_kernel` | `code.alipay.com:inference/composable_kernel` |
| `github.com/InternLM/turbomind` | `code.alipay.com:inference_thirdparty/turbomind` |
| `github.com/sgl-project/fast-hadamard-transform` | `code.alipay.com:inference_thirdparty/fast-hadamard-transform` |

**编译 sglang-kernel whl**:
```bash
cd ./python/sglang/kernels/aot
bash build.sh ${PYTHON_VERSION} ${CUDA_VERSION}
```
`build.sh` 实际使用 `uv build --wheel` 编译，通过 `rename_wheels.sh` 处理 whl 文件名（添加 `+cuXXX` 后缀和 `manylinux2014` 平台标签）。

如果 `build_sglang_whl_only=true`，则跳过真实编译，创建空的 mock whl 包。

**编译 sglang whl**:
```bash
cd ../python
cp ../README.md ../LICENSE .
pip install build wheel setuptools-scm setuptools-rust
python -m build --wheel --no-isolation
```

**产物输出**:
- `sglang_kernel`: `${ACB_BUILD_DIR}/code-repo/aci/artifacts/sgl_kernel/sgl*_kernel*.whl`
- `sglang`: `${ACB_BUILD_DIR}/code-repo/aci/artifacts/sglang/sglang*.whl`

**常见错误**:

| 错误 | 原因 | 解决 |
|------|------|------|
| `fatal: Could not read from remote repository` | antcode 仓库不存在或无权限 | 检查对应的 `code.alipay.com` 仓库是否存在，是否已同步上游 |
| `mitmdump` 启动失败 | mitmproxy 未安装或端口冲突 | 确认 `pip install mitmproxy` 成功 |
| `build.sh` 中 cmake 错误 | 缺少编译依赖或 CUDA 版本不匹配 | 检查 `python/sglang/kernels/aot/Dockerfile` 构建的镜像是否包含所需依赖 |
| `setuptools-scm` 版本问题 | sglang 版本推导依赖 git tag | 确保 `git fetch origin_ssh --tag` 成功 |
| ccache 缓存路径不存在 | 首次构建或 workspace 清理 | 首次构建会自动创建，检查 `/.cache/daily/` 权限 |
| gitee 镜像仓库不存在 | gitee mirror 未同步最新代码 | 联系管理员更新 gitee 镜像，或临时使用原始 GitHub 地址 |

---

#### Stage 4: Build-Image

**组件**: `docker-image-build`
**Dockerfile**: `docker/Dockerfile`
**Target**: `${{parameters.image_build_target}}`（默认 `framework_final`）
**跳过条件**: `build_whl_only=true`

**构建参数传递链**:
```
参数                        → Dockerfile ARG
SGLANG_BUILD_COMMIT         → SGLANG_BUILD_COMMIT
SGLANG_BUILD_URL            → SGLANG_BUILD_URL
runtime_base_image:tag      → RUNTIME_BASE_IMAGE
cuda_version_full           → CUDA_VERSION
sglang_kernel whl URL       → SGLANG_KERNEL_CURRENT_WHL
sglang whl URL              → SGLANG_CURRENT_WHL
deepgemm_whl_url            → DEEPGEMM_WHL_URL
flashmla_whl_url            → FLASHMLA_WHL_URL
deepep_whl_url              → DEEPEP_WHL_URL
transfer_engine_whl_url     → TRANSFER_ENGINE_WHL_URL
kvpool_whl_url              → KVPOOL_WHL_URL
ubuntu_mirror               → UBUNTU_MIRROR
pip_default_index           → PIP_DEFAULT_INDEX
github_artifactory           → GITHUB_ARTIFACTORY
install_flashinfer_jit_cache → INSTALL_FLASHINFER_JIT_CACHE
runtime_version             → RUNTIME_VERSION
runtime_llm_version         → RUNTIME_LLM_VERSION
rustup_dist_mirror          → RUSTUP_DIST_MIRROR
```

**docker/Dockerfile 多阶段构建结构**:

```
${RUNTIME_BASE_IMAGE}
├── base                    # 基础系统 + Python 3.12 + RDMA/IB + GDRCopy
│   ├── torch_deps          # Rust + sglang-kernel + sglang 依赖 (torch, flashinfer...)
│   │   ├── deepep_builder   # 编译 DeepEP 通信库 whl
│   │   ├── hpc_ops_builder  # 编译 HPC-Ops (sm90a 专用)
│   │   └── flashinfer_cache # 预下载 FlashInfer JIT 缓存
│   ├── devtools_builder    # 开发工具 (vim, tmux, zsh, nsight...)
│   ├── gateway_builder     # sgl-model-gateway (当前未启用)
│   └── local_src           # 复制本地源码
│
├── framework_final         # 完整开发镜像 (target: framework_final)
│   └── 合并所有 parallel builder 产物 + sglang editable install
│
└── runtime                 # 精简生产镜像 (target: runtime)
    └── 从 framework_final COPY site-packages + 二进制
```

**framework_final 阶段关键操作**:
1. 安装 DeepEP / HPC-Ops whl（从 builder 阶段 COPY）
2. 复制 FlashInfer cubin/jit-cache
3. 复制开发工具（gdb, vim, tmux, zsh, nsight）
4. 安装 Mooncake transfer-engine
5. 编译安装 MSCCL++
6. 安装 Python 开发工具包（pytest, black, pre-commit 等）
7. 可选安装 ai-dynamo
8. sglang 源码 editable install（`pip install -e "python[all]"`）
9. 下载 sgl-kernel cubins（sgl-kernel 源码位于 python/sglang/kernels/aot/）（`kernels download`）
10. 安装 x-runtime（serving_runtime + serving_runtime_llm）
11. 安装蚂蚁内部组件：
    - `decord2`（替代社区 decord2，避免解码卡死）
    - `model-manager`（异步模型挂载）
    - `modctl` + `aistudio-modelhub`
    - `sglang-router`
    - `ossutil`
12. DeepGEMM JIT 缓存（DeepSeek V4 专用）
13. CVE 修复（pip/apt 升级安全补丁）

**runtime 阶段与 framework_final 的区别**:
- runtime 不含开发工具（vim, tmux, zsh, nsight, clang-format 等）
- runtime 不含 pytest, black, pre-commit 等 Python 工具
- runtime 镜像比 framework_final 约小 1GB
- runtime 构建时间更长（因为多了一次 FROM 和 COPY）

**Build-Image 阶段配置**:
- `compression: gzip` + `forceCompression: true` — 强制 gzip 压缩所有层
- `nydusSwitch: true` — 构建 nydus 加速镜像
- `cacheStrategy: always` — 使用构建缓存
- `timeoutInSec: 14400` — 超时 4 小时
- `strictLabel: largeResource` — 使用大规格机器

**常见错误**:

| 错误 | 原因 | 解决 |
|------|------|------|
| sglang whl 安装失败 | whl 包 URL 不可达或与 CUDA 版本不匹配 | 检查 Build-Wheels 阶段产物是否正确上传，确认 whl URL |
| `kernels download` 失败 | huggingface 不可达或 sgl-flash-attn3 无匹配 cubin | x86_64 会自动重试 3 次并 fallback 到 JIT；aarch64 直接走 JIT |
| DeepEP 编译失败 | NVCC 或 CUDA 版本不匹配 | 检查 `deepep_builder` 日志 |
| MSCCL++ 编译失败 | 需要 cmake/ccache/nvcc，且依赖 dlpack, nlohmann-json, nanobind | 检查 `git config` 重定向是否有效 |
| x-runtime 下载失败 | `artifacts.antgroup-inc.cn` 不可达 | 检查网络/artifacts 服务状态，确认版本号存在 |
| decord2 安装失败 | whl URL 不存在或版本不兼容 | 检查 `DECORD2_WHL_URL` |
| DeepGEMM JIT cache 下载失败 | `dmsint` 服务不可达 | 检查 dmsint 连通性，失败可重试 |
| CVE 修复失败 | apt 源不可达 | 检查 `ubuntu_mirror` 连通性 |
| `pip install -e` 失败 | 依赖冲突或 pyproject.toml 版本要求 | 检查 constraints.txt 是否正确生成 |

---

#### Stage 5-7: 扫描与推送

**STC-Scan**:
- 组件: `stc`
- 扫描镜像: `reg.docker.alibaba-inc.com/sglang/deepep-base:<sha>-<ts>`
- 项目类型: AIDC

**Image-Scan**:
- 组件: `image-scan`
- 租户: AIDC

**Push-Image (3 个并行 job)**:
1. **推送至上海 S 集群**: `acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com`
2. **推送至北京 B1 集群**: `acr-bj-ant-registry-vpc.cn-beijing.cr.aliyuncs.com`
3. **推送镜像信息至 keeper**: 登记 `mainSiteImage` + `aidcSiteImage`

> 所有后置阶段在 `build_whl_only=true` 时自动跳过。

---

### 关键外部依赖与风险点

#### 网络依赖

| 依赖 | 地址 | 用途 | 失败影响 |
|------|------|------|----------|
| GitHub 代理 | `github.ednovas.xyz` | CMake/ccache 等源码下载 | Stage 2 失败 |
| PyPI 蚂蚁源 | `pypi.antfin-inc.com` | Python 包安装 | Stage 2/3/4 失败 |
| PyTorch 官方源 | `download.pytorch.org` | torch 下载 | Stage 2 失败 |
| gitee 镜像 | `gitee.com` | Git 依赖重定向 | Stage 3 CMake FetchContent 失败 |
| antcode | `code.alipay.com` | Git SSH 依赖重定向 | Stage 3 Git clone 失败 |
| dmsint | `dmsint.cn-hangzhou.alipay.aliyun-inc.com` | SSH key 下载、decord2 whl、proxy.py | Stage 3 启动失败 |
| OSS | `cmps-model.cn-hangzhou.alipay.aliyun-inc.com` | mitmproxy 代理目标 (nlohmann/json) | Stage 3 代理失败 |
| artifacts | `artifacts.antgroup-inc.cn` | x-runtime / model-manager / sglang-router | Stage 4 部分组件缺失 |
| OSS 公共 | `gosspublic.alicdn.com` | ossutil 下载 | Stage 4 ossutil 缺失 |
| 阿里云镜像 | `mirrors.aliyun.com` / `mirrors.cloud.aliyuncs.com` | apt/rustup 镜像 | Stage 4 安装失败 |

#### Secrets 依赖

| Secret | 用途 | 缺失影响 |
|--------|------|----------|
| `OSS_URL` | whl 产物上传 | 产物无法上传，后续 stage 无法获取 whl |
| `OSS_AI` | OSS 认证 ID | 同上 |
| `OSS_AK` | OSS 认证 Key | 同上 |

#### Workspace 缓存

| 缓存路径 | 用途 | 清理影响 |
|----------|------|----------|
| `/.cache/daily/cuda-<ver>` | ccache 编译缓存 | 首次构建变慢（~2 小时 → ~3-4 小时） |
| `/root/.cache/pip` | pip 下载缓存 | 每次重新下载所有 Python 包 |
| `/root/.cache/huggingface` | HF 模型缓存 | `kernels download` 变慢 |
| `/var/cache/apt` | apt 包缓存 | 每次重新下载系统包 |

---

### 故障排查指南

#### 构建失败诊断流程

```bash
# 1. 获取流水线 ID
aci pipeline list

# 2. 查看整体状态
aci pipeline show <pipeline-id> -o json > /tmp/pipeline.json

# 3. 找出失败的 job
python3 -c "
import json
with open('/tmp/pipeline.json') as f:
    data = json.load(f)
for stage in data.get('stageExecutions', []):
    for job in stage.get('jobExecutions', []):
        if job.get('jobStatus') not in ('SUCCEED', 'RUNNING'):
            print(f\"FAIL: {stage['name']}/{job['name']} jobId={job['id']}\")
"

# 4. 查看失败 job 日志
aci job log <job-id>

# 5. 查看调度日志（排查调度层面问题）
aci job scheduler-log <job-id>
```

#### 常见失败场景速查

| 失败阶段 | 典型日志关键词 | 根因 | 修复方式 |
|----------|---------------|------|----------|
| Verify-Parameters | `Unsupported CUDA version` | 不支持的 CUDA 版本 | 只支持 12.9.1 / 13.0.1 |
| Build-Sglang-Build-Image | `wget: unable to resolve host address` | GitHub 代理不可达 | 检查 `github_artifactory` DNS |
| Build-Sglang-Build-Image | `ccache: command not found` | ccache 编译失败 | 检查 xxhash-devel / zstd-devel |
| Build-Wheels | `fatal: Could not read from remote repository` | antcode 仓库不存在 | 检查对应仓库是否已 fork 到 antcode |
| Build-Wheels | `No such file: dist/*.whl` | sglang-kernel 编译失败 | 检查 `build.sh` 日志，确认 CUDA/Python 版本 |
| Build-Wheels | `setuptools-scm: version ... not found` | git tag 未 fetch | 检查 `git fetch origin_ssh --tag` |
| Build-Wheels | `mitmdump: command not found` | mitmproxy 安装失败 | 检查 `pip install mitmproxy` |
| Build-Image | `403 Forbidden` (whl URL) | whl 产物未正确上传 OSS | 检查 OSS secrets 配置 |
| Build-Image | `kernels download` 超时 | huggingface 不可达 | 重试或使用 JIT fallback |
| Build-Image | `wget: ... artifacts.antgroup-inc.cn` 超时 | artifacts 服务不可达 | 确认版本号正确，检查服务状态 |
| Build-Image | `E: Unable to locate package` | apt 镜像源不可达 | 更换 `ubuntu_mirror` 参数 |
| Image-Scan / STC | 超时 | 扫描服务繁忙 | 重试 job |

#### 重试策略

- **Build-Sglang-Build-Image 失败**: 直接重试 job（`skipWhenImageExists=true`，已成功的层会复用）
- **Build-Wheels 失败**: 先清理 workspace 缓存 `/.cache/daily/cuda-<ver>`，再重试
- **Build-Image 失败**: 优先检查是否是外部依赖不可达，网络问题直接重试
- **扫描/推送失败**: 直接重试对应 job

---

### 与社区版的关键差异

| 方面 | 社区版 (sgl-project/sglang) | 蚂蚁内部版 |
|------|---------------------------|-----------|
| 基础镜像 | `nvidia/cuda:xxx-runtime-ubuntu24.04` | `reg.docker.alibaba-inc.com/antos/ubuntu-ai-x86_64-ngc` |
| Python 源 | `pypi.org` | `pypi.antfin-inc.com` |
| apt 源 | `archive.ubuntu.com` | `mirrors.cloud.aliyuncs.com` |
| GitHub 访问 | 直连 | 通过 `github_artifactory` 代理 |
| sglang-kernel whl | 从 PyPI 安装社区版 | 从源码编译 |
| DeepEP | 从 GitHub 编译 | 从 antcode fork 编译 |
| Git 依赖 | GitHub 直连 | Gitee 镜像 + antcode fork |
| x-runtime | 无 | 安装蚂蚁内部 serving_runtime |
| decord2 | 社区版 | 蚂蚁内部修复版 |
| Mooncake | 可选编译 | 从蚂蚁内部源安装预编译包 |
| MSCCL++ | 从 GitHub 编译 | 从 antcode fork 编译 |
| DeepGEMM JIT | 运行时下载 | 预置到镜像 |
| CVE 修复 | 无 | apt/pip 升级修复已知 CVE |

---

### 获取社区文件对比

当需要对比分析本地（蚂蚁内部版）与社区（sgl-project/sglang）构建逻辑差异时，可以直接通过 GitHub raw URL 获取社区版对应文件。

#### 核心文件获取

| 本地文件 | 社区 raw URL 模板 |
|----------|-------------------|
| `python/sglang/kernels/aot/build.sh` | `https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/{ref}/python/sglang/kernels/aot/build.sh` |
| `python/sglang/kernels/aot/Dockerfile` | `https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/{ref}/python/sglang/kernels/aot/Dockerfile` |
| `python/sglang/kernels/aot/rename_wheels.sh` | `https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/{ref}/python/sglang/kernels/aot/rename_wheels.sh` |
| `docker/Dockerfile` | `https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/{ref}/docker/Dockerfile` |
| `docker/Dockerfile.compile` | `https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/{ref}/docker/Dockerfile.compile` |
| `python/pyproject.toml` | `https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/{ref}/python/pyproject.toml` |

其中 `{ref}` 可以是：
- **分支名**：如 `main`、`dev/something`
- **commit SHA**：如 `e1bc001872985a23af65c367b802ff8fb44edafc`
- **tag**：如 `v0.5.12.post1`

#### 获取方式

```bash
# 方式一：curl（推荐，更通用）
REF="main"  # 或具体的 commit SHA
curl -sL "https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/${REF}/python/sglang/kernels/aot/build.sh" -o /tmp/community_build.sh

# 方式二：wget
wget -q "https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/${REF}/python/sglang/kernels/aot/build.sh" -O /tmp/community_build.sh
```

#### 对比分析工作流

```bash
# 1. 确定要对比的社区版本（分支或 commit）
COMMUNITY_REF="main"  # 或 e1bc001872985a23af65c367b802ff8fb44edafc

# 2. 依次拉取社区版本的关键文件
mkdir -p /tmp/sglang_community /tmp/sglang_local_diff

# 获取 build.sh
curl -sL "https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/${COMMUNITY_REF}/python/sglang/kernels/aot/build.sh" \
  -o /tmp/sglang_community/build.sh

# 获取 Dockerfile（kernel 编译镜像）
curl -sL "https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/${COMMUNITY_REF}/python/sglang/kernels/aot/Dockerfile" \
  -o /tmp/sglang_community/kernel_Dockerfile

# 获取主 Dockerfile
curl -sL "https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/${COMMUNITY_REF}/docker/Dockerfile" \
  -o /tmp/sglang_community/runtime_Dockerfile

# 3. 逐文件 diff 对比
echo "=== diff build.sh ==="
diff -u /tmp/sglang_community/build.sh python/sglang/kernels/aot/build.sh

echo "=== diff kernel Dockerfile ==="
diff -u /tmp/sglang_community/kernel_Dockerfile python/sglang/kernels/aot/Dockerfile

echo "=== diff runtime Dockerfile ==="
diff -u /tmp/sglang_community/runtime_Dockerfile docker/Dockerfile
```

#### 常见对比关注点

| 关注项 | 文件 | 典型差异 |
|--------|------|----------|
| 基础镜像源 | `Dockerfile`, `kernel Dockerfile` | 社区用 `nvidia/cuda`/`pytorch/manylinux`，蚂蚁用 `reg.docker.alibaba-inc.com` |
| pip/apt 镜像 | `Dockerfile`, `build.sh` | 社区用 `pypi.org`/`archive.ubuntu.com`，蚂蚁用内部源 |
| Git 依赖重定向 | `build.sh`（编译脚本中 git config） | 社区直连 GitHub，蚂蚁替换为 gitee/antcode |
| 额外组件安装 | `Dockerfile` | 社区无 x-runtime/decord2/model-manager/modctl/sglang-router |
| CUDA 兼容层 | `Dockerfile` | 蚂蚁有 `update_cuda_compat.sh` |
| CVE 修复 | `Dockerfile` | 蚂蚁有额外的 apt/pip CVE 修复步骤 |
| DeepGEMM JIT cache | `Dockerfile` | 蚂蚁预置 DeepSeek V4 cache |
| 编译并发度 | `build.sh` | 蚂蚁可能调整 `BUILD_JOBS`/`NVCC_THREADS` |

#### 批量对比脚本

将以下脚本保存执行，一次性拉取并对比所有关键文件：

```bash
#!/bin/bash
# 用法: bash compare_with_community.sh <ref>
# 示例: bash compare_with_community.sh main
#       bash compare_with_community.sh e1bc001872

set -e
COMMUNITY_REF="${1:-main}"
COMM_DIR="/tmp/sglang_community_${COMMUNITY_REF}"
LOCAL_ROOT="$(git rev-parse --show-toplevel)"
DIFF_FOUND=0

mkdir -p "${COMM_DIR}"

# 定义要对比的文件列表：社区URL路径|本地相对路径|显示名称
FILES=(
  "python/sglang/kernels/aot/build.sh|python/sglang/kernels/aot/build.sh|build.sh"
  "python/sglang/kernels/aot/Dockerfile|python/sglang/kernels/aot/Dockerfile|kernel Dockerfile"
  "python/sglang/kernels/aot/rename_wheels.sh|python/sglang/kernels/aot/rename_wheels.sh|rename_wheels.sh"
  "docker/Dockerfile|docker/Dockerfile|runtime Dockerfile"
  "docker/Dockerfile.compile|docker/Dockerfile.compile|compile Dockerfile"
  "python/pyproject.toml|python/pyproject.toml|pyproject.toml"
)

for entry in "${FILES[@]}"; do
  IFS='|' read -r url_path local_path display_name <<< "$entry"
  community_file="${COMM_DIR}/${display_name// /_}"
  local_file="${LOCAL_ROOT}/${local_path}"
  
  echo "=== Fetching: ${display_name} ==="
  if curl -fsSL --connect-timeout 10 \
    "https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/${COMMUNITY_REF}/${url_path}" \
    -o "${community_file}"; then
    echo "  Downloaded: $(wc -l < "${community_file}") lines"
    
    if [ -f "${local_file}" ]; then
      if ! diff -q "${community_file}" "${local_file}" > /dev/null 2>&1; then
        echo "  *** DIFF FOUND ***"
        diff -u "${community_file}" "${local_file}" > "${COMM_DIR}/${display_name// /_}.diff" || true
        echo "  Diff saved to: ${COMM_DIR}/${display_name// /_}.diff"
        DIFF_FOUND=1
      else
        echo "  No differences (identical)"
      fi
    else
      echo "  WARNING: local file not found: ${local_file}"
    fi
  else
    echo "  ERROR: failed to download (ref=${COMMUNITY_REF}, path=${url_path})"
  fi
  echo ""
done

if [ "${DIFF_FOUND}" -eq 1 ]; then
  echo "Differences found. See ${COMM_DIR}/ for .diff files."
else
  echo "All files are identical to community version (ref: ${COMMUNITY_REF})."
fi
```

> **注意**：GitHub raw URL 在国内可能访问不稳定，如果下载失败可通过 `github_artifactory` 代理：
> ```bash
> curl -sL "https://github.ednovas.xyz/https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/${REF}/python/sglang/kernels/aot/build.sh"
> ```
