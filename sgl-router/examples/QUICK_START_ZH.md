# Mini Load Balancer 快速开始指南

## 概述

Mini Load Balancer (Mini LB) 是一个轻量级的 Rust 实现的负载均衡器，专门为 SGLang 的 Prefill-Decode 分离架构设计。

## 快速开始

### 1. 启动 Prefill 服务器

```bash
# 启动第一个 Prefill 服务器
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 30000 \
  --host 0.0.0.0 \
  --disagg-prefill-only \
  --bootstrap-port 30001

# （可选）启动第二个 Prefill 服务器
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 30002 \
  --host 0.0.0.0 \
  --disagg-prefill-only \
  --bootstrap-port 30003
```

### 2. 启动 Decode 服务器

```bash
# 启动第一个 Decode 服务器
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 31000 \
  --host 0.0.0.0 \
  --disagg-decode-only

# （可选）启动第二个 Decode 服务器
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 31001 \
  --host 0.0.0.0 \
  --disagg-decode-only
```

### 3. 启动 Mini Load Balancer

```bash
# 方式 1: 运行基础示例
cd sgl-router
cargo run --example mini_lb_basic

# 方式 2: 运行自定义配置示例
cargo run --example mini_lb_custom_config
```

### 4. 发送测试请求

#### 使用 cURL

```bash
# 健康检查
curl http://localhost:8080/health

# 聊天补全
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "What is Rust?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'

# 流式聊天补全
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Count from 1 to 5"}
    ],
    "stream": true,
    "max_tokens": 50
  }'

# 文本生成
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Once upon a time",
    "sampling_params": {
      "temperature": 0.8,
      "max_new_tokens": 50
    }
  }'
```

#### 使用 Python

```bash
# 运行 Python 测试客户端
cd sgl-router/examples
python mini_lb_python_example.py
```

#### 使用 Rust 客户端

```bash
cd sgl-router
cargo run --example mini_lb_client
```

## 配置说明

### 基础配置

```rust
let config = MiniLbConfig::new(
    "0.0.0.0".to_string(),  // 监听地址
    8080,                   // 监听端口
    vec![                   // Prefill 服务器列表
        ("http://localhost:30000".to_string(), Some(30001)),
    ],
    vec![                   // Decode 服务器列表
        "http://localhost:31000".to_string(),
    ],
    1800,                   // 超时时间（秒）
);
```

### 多服务器配置

```rust
let config = MiniLbConfig::new(
    "0.0.0.0".to_string(),
    8080,
    vec![
        ("http://prefill-1:30000".to_string(), Some(30001)),
        ("http://prefill-2:30000".to_string(), Some(30001)),
        ("http://prefill-3:30000".to_string(), Some(30001)),
    ],
    vec![
        "http://decode-1:31000".to_string(),
        "http://decode-2:31000".to_string(),
        "http://decode-3:31000".to_string(),
    ],
    3600,
);
```

## 支持的端点

### 健康检查
- `GET /health` - 基础健康检查
- `GET /health_generate` - 后端服务器健康检查

### 管理端点
- `POST /flush_cache` - 刷新缓存
- `GET /get_server_info` - 获取服务器信息
- `GET /get_model_info` - 获取模型信息

### OpenAI API
- `POST /v1/chat/completions` - 聊天补全
- `POST /v1/completions` - 文本补全
- `GET /v1/models` - 列出模型

### SGLang API
- `POST /generate` - 文本生成

## 常见问题

### 1. 负载均衡器无法启动
- 检查端口 8080 是否被占用
- 验证配置中的主机和端口设置
- 查看日志中的错误信息

### 2. 请求超时
- 增加配置中的 `timeout_secs`
- 确认后端服务器正在运行且可访问
- 检查网络连接

### 3. "没有可用服务器"错误
- 确保 `prefill_urls` 和 `decode_urls` 不为空
- 验证服务器 URL 格式正确
- 确认后端服务器正在运行

## 性能提示

1. **服务器数量**: 根据负载调整 Prefill 和 Decode 服务器数量
2. **超时设置**: 根据模型大小和请求复杂度调整超时时间
3. **并发连接**: Mini LB 自动处理并发请求
4. **监控**: 使用 `/get_server_info` 端点监控服务器状态

## 下一步

- 阅读[完整文档](./mini_lb_README.md)
- 查看[实现总结](./MINI_LB_IMPLEMENTATION_SUMMARY.md)
- 参考示例代码学习更多用法
- 根据需求调整配置

## 获取帮助

如有问题，请：
1. 查看日志输出
2. 检查后端服务器状态
3. 验证网络连接
4. 提交 GitHub issue

---

Happy coding! 🚀
