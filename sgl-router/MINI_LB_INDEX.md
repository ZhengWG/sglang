# Mini Load Balancer - Documentation Index

## 📖 Documentation Navigation

### 🚀 Quick Start
If you're using Mini Load Balancer for the first time, start here:
- [**Quick Start Guide**](examples/QUICK_START.md) - Fastest way to get started

### 📚 Complete Documentation
For in-depth understanding of all features and configuration:
- [**Complete Usage Documentation**](examples/mini_lb_README.md) - Detailed feature descriptions and API reference

### 📝 Implementation Details
Learn about implementation details and design decisions:
- [**Implementation Summary**](examples/MINI_LB_IMPLEMENTATION_SUMMARY.md) - Implementation overview and feature comparison
- [**Feature Summary**](../MINI_LB_FEATURE_SUMMARY.md) - Complete feature list and code statistics

## 💻 Code Examples

### Rust Examples
1. [**Basic Example**](examples/mini_lb_basic.rs) - Simple configuration and startup
2. [**Custom Config Example**](examples/mini_lb_custom_config.rs) - Advanced configuration
3. [**Client Example**](examples/mini_lb_client.rs) - How to call the API

### Python Examples
1. [**Python Client**](examples/mini_lb_python_example.py) - Using Mini LB from Python

## 🧪 Tests
- [**Unit Tests**](tests/mini_lb_test.rs) - Test cases and validation

## 📂 Source Code
```
sgl-router/src/mini_lb/
├── mod.rs      # Module definition
├── types.rs    # Data types
└── router.rs   # Core implementation
```

## 🎯 Use Cases

| Scenario | Recommended Documentation |
|----------|--------------------------|
| I want to try it quickly | [Quick Start Guide](examples/QUICK_START.md) |
| I need to configure multiple servers | [Custom Config Example](examples/mini_lb_custom_config.rs) |
| I want to understand all APIs | [Complete Documentation](examples/mini_lb_README.md) |
| I want to use from Python | [Python Client Example](examples/mini_lb_python_example.py) |
| I want implementation details | [Implementation Summary](examples/MINI_LB_IMPLEMENTATION_SUMMARY.md) |
| I want to see all features | [Feature Summary](../MINI_LB_FEATURE_SUMMARY.md) |

## 🛠️ Quick Commands

```bash
# Run basic example
cargo run --example mini_lb_basic

# Run client test
cargo run --example mini_lb_client

# Run Python test
python examples/mini_lb_python_example.py

# Run unit tests
cargo test --test mini_lb_test
```

## 🔗 Related Links

- [SGLang Official Documentation](https://sglang.readthedocs.io/)
- [Prefill-Decode Disaggregation](https://sglang.readthedocs.io/en/latest/advanced_features/disaggregation.html)
- [Full Router Documentation](../README.md)

## 📞 Getting Help

Having issues?
1. Check the FAQ section in the [Quick Start Guide](examples/QUICK_START.md)
2. Read the Troubleshooting chapter in the [Complete Documentation](examples/mini_lb_README.md)
3. Review the example code
4. Submit a GitHub Issue

---

**Tip**: All documentation includes detailed code examples and configuration instructions. Choose the appropriate documentation based on your needs.
