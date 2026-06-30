# 使用说明

当前目录存放蚂蚁内部的sglang包构建和运行时镜像流水线和Dockerfile，以及构建镜像所需的一些配置文件，详情参考语雀文档[sglang引擎镜像制作流水线说明](https://yuque.antfin.com/hegpb4/kg7h1z/lvyiqs9tuxcem3hw)。
常用的制作流程是： 
1. 通过`sglang_runtime.aci.yml` 完整编译构建出可供线上使用的生产镜像，保证镜像中安装`sglang` 和`sglang-kernel` 依赖的必要包。由于是从编译sglang，sglang-kernel，然后编译安装deepep，flash-mla等sglang依赖的组件，此流水线构建时间比较费时，一般要两个小时。

2. 在有了完整构建的镜像之后，如果仅仅是修改`sglang`或者`sglang-kernel`代码，可以通过触发`sglang_runtime.aci.yml`只构建`sglang`， `sglang-kernel`两个包，然后触发`sglang_fast`流水线，基于第一步构建的镜像，覆盖安装新制作的`sglang`和`sglang-kernel`包。
**注意：** `sglang_fast.aci.yml`中默认使用的基础镜像基于commit [1ef14f05](https://code.alipay.com/Theta/SGLang/commit/1ef14f05291acb0b187bf98db6c3d52c6f917a86), 如果使用此版本镜像作为基础镜像，请确保提交流水线时设置的`sglang`和`sglang-kernel`的依赖和此commit中声明的一致，比如 [pyproject.toml](https://code.alipay.com/Theta/SGLang/blob/1ef14f05291acb0b187bf98db6c3d52c6f917a86/python/pyproject.toml)。


## 常见构建场景

### 1. pre-compile 流水线
用来构建编译`sglang` 和 `sglang-kernel`使用的镜像的流水线，参见 `sglang_prepare_compile_image.aci.yml`介绍。

### 2. 从源码完整构建线上使用的sglang引擎镜像
配置参数 `build_whl_only`: "false", 流水线会在编译`sglang`和`sglang-kernel`包之后，继续构建镜像。
注意：这种情况下，不能配置`build_sglang_whl_only` == `true`, 否则构建出来的`sglang-kernel`whl包是个空文件。

### 3. 使用预编译 wheel 包构建镜像
也是执行流水线，选择`docker/ant/aci/sglang_runtime.aci.yml`，然后配置如下三个流水线变量：
```yaml
skip_build_stage: "true"
build_whl_only: "false"
sglang_kernel_whl_url: "https://xxx/sglang_kernel-xxx.whl"
sglang_whl_url: "https://xxx/sglang-xxx.whl"
```
跳过编译阶段，直接使用指定 URL 的 wheel 包构建运行时镜像。

### 4. 只编译 `sglang_kernel`, `sglang` wheel 包（调试使用, 或者配合3快速做镜像）
https://code.alipay.com/Theta/SGLang/pipelines页面选择执行流水线，从仓库yml中选择代码分支，配置使用`docker/ant/aci/sglang_runtime.aci.yml`, 这条流水线默认仅执行编译阶段，输出 `sglang_kernel` 和 `sglang` wheel 包，不构建镜像.

### 5. 只编译 sglang wheel 包（快速编译）
同上，需要在流水线选项里添加如下变量：
```yaml
build_sglang_whl_only: "true"
```
跳过 sglang_kernel 编译（mock 空包），只编译 sglang wheel 包。


# 流水线详细介绍

此目录下有三条流水线：
1. `sglang_prepare_compile_image.aci.yml`: 用于创建蚂蚁内部编译`sglang`和`sglang_kernel` wheel包的docker镜像。
2. `sglang_runtime.aci.yml`: 用于创建线上生产使用或者集成开发、测试使用的sglang运行时镜像。
3. `sglang_fast.aci.yml`: 用于快速替换镜像中sglang-kernel 和 sglang包，制作生产使用的镜像。

## `sglang_fast`
这个流水线模版是用来快速覆盖已经编译好的镜像中的`sglang` 和 `sglang_kernel`包，减少镜像构建时间的，注意预编译好的`sglang` 和 `sglang_kernel`包和base镜像中`sglang` 和 `sglang_kernel`版本一致, 依赖组件版本也必须一致。

| 参数名 | 默认值 | 说明 
|--------|--------|------|
| `sglang_whl_url` | `` | sglang wheel 包地址 |
| `sglang_kernel_whl_url` | `` | sglang_kernel wheel 包地址 |
| `runtime_base_image_tag` | `d834d95c-20260511192941` | 运行时镜像 tag, 这个tag对应的是之前通过`sglang_runtime.aci.yml`制作好的镜像|

## `sglang_runtime.aci.yml`

这条流水线用于创建 sglang 运行时镜像，基于 `docker/Dockerfile` 多阶段构建，支持从源码编译 wheel 包或使用预编译的 wheel 包。

### 流水线阶段

1. **Verify-Parameters**: 参数校验和转换
2. **Build-Wheels**: 编译 sglang_kernel 和 sglang wheel 包
3. **Build-Image**: 构建运行时镜像
4. **STC-Scan**: 安全扫描
5. **Image-Scan**: 镜像扫描
6. **Push-Image**: 推送镜像至多集群

### 完整参数说明

#### 构建控制参数

| 参数名 | 默认值 | 说明 | 可选项 |
|--------|--------|------|--------|
| `build_whl_only` | `true` | 仅编译 wheel 包，不构建镜像 | `true`, `false` |
| `build_sglang_whl_only` | `false` | 仅编译 `sglang` wheel 包，mock `sglang_kernel`包。 | `true`, `false` |
| `skip_build_stage` | `false` | 跳过编译阶段，使用预编译 wheel 包 | `true`, `false` |

#### CUDA 和 Python 版本参数

| 参数名 | 默认值 | 说明 | 可选项 |
|--------|--------|------|--------|
| `cuda_version_full` | `13.0.1` | 完整 CUDA 版本，需与 `torch_version` 兼容 | `12.9.1`, `13.0.1` |
| `torch_version` | `2.11.0` | PyTorch 版本，需与 `cuda_version_full` 兼容，改为 2.9.1 之外的版本时需要同时配置 cu_tag | - |
| `python_version` | `3.12` | Python 版本，3.12 默认安装了 torch、uv 等组件，构建比较快，且社区 runtime 也使用 Python 3.12 | `3.8`, `3.9`, `3.10`, `3.11`, `3.12`, `3.13`, `3.14` |
| `enable_below_sm90` | `ON` | 是否支持 SM90 以下 GPU（Hopper 以下架构），默认支持 | `ON`, `OFF` |

#### 编译参数

| 参数名 | 默认值 | 说明 | 可选项 |
|--------|--------|------|--------|
| `max_jobs` | `24` | ninja 编译并发度，一般不用修改，调大可能导致 OOM | - |
| `nvcc_threads` | `4` | nvcc 编译并发度，一般不用修改，调大可能导致 OOM | - |

#### Wheel 包 URL 参数

| 参数名 | 默认值 | 说明 | 可选项 |
|--------|--------|------|--------|
| `sglang_kernel_whl_url` | `""` | 预编译 sglang_kernel wheel 包 URL | - |
| `sglang_whl_url` | `""` | 预编译 sglang wheel 包 URL | - |
| `deepgemm_whl_url` | `""` | DeepGEMM wheel 包 URL，DeepGEMM 已编译到 sglang-kernel 包，不需要安装了，如果安装老版本的 sglang 才需要 | - |
| `flashmla_whl_url` | `""` | FlashMLA wheel 包 URL（可选项） | - |
| `deepep_whl_url` | `""` | DeepEP wheel 包 URL（可选项） | - |
| `transfer_engine_whl_url` | `""` | Mooncake Transfer Engine wheel 包 URL，如有开源或内部预编译的包可指定，否则会在 Dockerfile 里编译安装。如指定安装，需要确认 kvpool 版本和依赖与 sglang 一致，在 Dockerfile 中 pip install --no-deps 安装，避免破坏 sglang 依赖的包版本 | - |
| `kvpool_whl_url` | `""` | 蚂蚁内部的 kvpool whl 包，默认不安装，同上，--no-deps 模式，避免破坏 sglang 依赖的包版本 | - |

#### 基础镜像参数

| 参数名 | 默认值 | 说明 | 可选项 |
|--------|--------|------|--------|
| `runtime_base_image` | `reg.docker.alibaba-inc.com/antos/ubuntu-ai-x86_64-ngc` | 运行时基础镜像，蚂蚁内部镜像为ubuntu-ai, 社区为cuda | `reg.docker.alibaba-inc.com/antos/ubuntu-ai-x86_64-ngc`, `registry.cn-hangzhou.aliyuncs.com/augusto/cuda` |
#### 镜像源和代理参数

| 参数名 | 默认值 | 说明 | 可选项 |
|--------|--------|------|--------|
| `ubuntu_mirror` | `https://mirrors.aliyun.com` | 构建过程中使用的 ubuntu 镜像源，默认使用阿里云镜像源 | `https://mirrors.aliyun.com`, `https://mirrors.tuna.tsinghua.edu.cn`, `https://mirrors.ustc.edu.cn`, `https://mirrors.cloud.tencent.com` |
| `pip_default_index` | `https://pypi.antfin-inc.com/simple` | 构建过程中使用的 pip 安装源，默认使用蚂蚁内部镜像源 | `https://pypi.antfin-inc.com/simple`, `https://pypi.tuna.tsinghua.edu.cn/simple`, `https://mirrors.aliyun.com/pypi/simple` |
| `github_artifactory` | `github.ednovas.xyz/https://github.com` | 由于 github 在国内访问不稳定，构建过程中访问 github 的地址可以替换成这个镜像地址，默认不替换，如果需要替换，在 dockerfile 里访问 github 的地方需要替换成这个地址 | - |

#### 功能开关参数

| 参数名 | 默认值 | 说明 | 可选项 |
|--------|--------|------|--------|
| `install_flashinfer_jit_cache` | `0` | 是否预下载 FlashInfer JIT 缓存 | `0`, `1` |
| `image_build_target` | `runtime` | 镜像构建目标，`runtime` 为生产镜像，`framework_final` 为开发调试镜像 | `runtime`, `framework_final` |

#### xruntime 版本参数

| 参数名 | 默认值 | 说明 | 可选项 |
|--------|--------|------|--------|
| `runtime_version` | `1.1.15` | xruntime 组件版本，如需自定义请咨询 **天缺** | - |
| `runtime_llm_version` | `1.1.9` | xruntime LLM 组件版本，如需自定义请咨询 **天缺** | - |


## `sglang_prepare_compile_image.aci.yml`

此流水线用于创建编译镜像，基于 `Dockerfile.compile` 构建，输出镜像用于编译 sglang wheel 包。

### 流水线参数

| 参数名 | 默认值 | 说明 | 可选项 |
|--------|--------|------|--------|
| `cuindex` | `cu130` | 安装 torch 对应的 CUDA 版本标识，需与 `cuda_version` 匹配 | - |
| `cuda_version` | `13.0` | CUDA 版本 | `12.9`, `13.0` |
| `python_tag` | `cp312-cp312` | Python 环境标签，3.12 默认安装了 torch、uv 等组件，构建比较快 | `cp310-cp310`, `cp311-cp311`, `cp312-cp312`, `cp313-cp313`, `cp313-cp313t`, `cp314-cp314`, `cp314-cp314t`, `cp38-cp38`, `cp39-cp39` |

### 输出镜像

- **镜像地址**: `reg.docker.alibaba-inc.com/sglang/theta_sglang_build`
- **镜像标签**: `${python_tag}-cuda${cuda_version}`

### 流水线阶段

1. **Build-Image**: 构建编译镜像
2. **STC-Scan**: 安全扫描
3. **Image-Scan**: 镜像扫描


# Dockerfile介绍

## Dockerfile.compile
`Dockerfile.compile`被`aci/sglang_prepare_compile_image.aci.yml`使用，用来构建蚂蚁内部的编译`sglang-kernel` 和 `sglang` wheel包的编译镜像。该镜像基于 `manylinux-builder` 基础镜像，主要特点：

- 预装 **torch 2.9.1**（与 CUDA 版本匹配）
- 安装 CMake 3.31+、ccache 4.12+ 等编译工具
- 配置 Rust 工具链（用于 `sglang-grpc` 扩展）
- 支持多 Python 版本（cp310-cp310 到 cp314-cp314t 等）

## Dockerfile
`Dockerfile` 参考社区 `docker/Dockerfile`，采用多阶段并行构建策略，分别用于构建：

- **生产镜像 (`runtime` target)**：精简的运行时环境
- **开发调试镜像 (`framework_final` target)**：包含完整开发工具

主要构建阶段：
1. `base`：基础系统依赖和 CUDA 环境
2. `torch_deps`：PyTorch 依赖和 sglang  wheel 包安装
3. `deepep_builder`：DeepEP 通信库编译
4. `flashinfer_cache`：FlashInfer JIT 缓存（可选）
5. `devtools_builder`：开发工具
6. `gateway_builder`：网关组件
7. `framework`：整合所有组件的最终镜像


---

和社区版本对比记录：
```版本记录
since Theta/SGLang branch `sglang_public_tracker`: ed80ee79504d06736e2a896c6f7e13c8e716fd96
since sgl-project/sglang branch `main`: e1bc001872985a23af65c367b802ff8fb44edafc
```
