# Mini Load Balancer - 文档索引

## 📖 文档导航

### 🚀 快速开始
如果您是第一次使用 Mini Load Balancer，从这里开始：
- [**快速开始指南 (中文)**](examples/QUICK_START_ZH.md) - 最快上手的方式

### 📚 完整文档
需要深入了解所有功能和配置：
- [**完整使用文档 (英文)**](examples/mini_lb_README.md) - 详细的功能说明和 API 参考

### 📝 实现说明
了解实现细节和设计决策：
- [**实现总结**](examples/MINI_LB_IMPLEMENTATION_SUMMARY.md) - 实现概述和功能对比
- [**功能总结**](../MINI_LB_FEATURE_SUMMARY.md) - 完整的功能清单和代码统计

## 💻 代码示例

### Rust 示例
1. [**基础示例**](examples/mini_lb_basic.rs) - 简单配置和启动
2. [**自定义配置示例**](examples/mini_lb_custom_config.rs) - 高级配置
3. [**客户端示例**](examples/mini_lb_client.rs) - 如何调用 API

### Python 示例
1. [**Python 客户端**](examples/mini_lb_python_example.py) - 从 Python 使用 Mini LB

## 🧪 测试
- [**单元测试**](tests/mini_lb_test.rs) - 测试用例和验证

## 📂 源代码
```
sgl-router/src/mini_lb/
├── mod.rs      # 模块定义
├── types.rs    # 数据类型
└── router.rs   # 核心实现
```

## 🎯 使用场景

| 场景 | 推荐文档 |
|-----|---------|
| 我想快速试用 | [快速开始指南](examples/QUICK_START_ZH.md) |
| 我需要配置多个服务器 | [自定义配置示例](examples/mini_lb_custom_config.rs) |
| 我想了解所有 API | [完整使用文档](examples/mini_lb_README.md) |
| 我想从 Python 使用 | [Python 客户端示例](examples/mini_lb_python_example.py) |
| 我想了解实现细节 | [实现总结](examples/MINI_LB_IMPLEMENTATION_SUMMARY.md) |
| 我想查看所有功能 | [功能总结](../MINI_LB_FEATURE_SUMMARY.md) |

## 🛠️ 快速命令

```bash
# 运行基础示例
cargo run --example mini_lb_basic

# 运行客户端测试
cargo run --example mini_lb_client

# 运行 Python 测试
python examples/mini_lb_python_example.py

# 运行单元测试
cargo test --test mini_lb_test
```

## 🔗 相关链接

- [SGLang 官方文档](https://sglang.readthedocs.io/)
- [Prefill-Decode 分离](https://sglang.readthedocs.io/en/latest/advanced_features/disaggregation.html)
- [完整 Router 文档](../README.md)

## 📞 获取帮助

遇到问题？
1. 查看[快速开始指南](examples/QUICK_START_ZH.md)的常见问题部分
2. 阅读[完整文档](examples/mini_lb_README.md)的故障排除章节
3. 查看示例代码
4. 提交 GitHub Issue

---

**提示**: 所有文档都包含详细的代码示例和配置说明，建议根据您的需求选择合适的文档阅读。
