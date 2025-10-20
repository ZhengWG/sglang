# Mini Load Balancer Quick Start Guide

## Overview

Mini Load Balancer (Mini LB) is a lightweight Rust-based load balancer specifically designed for SGLang's Prefill-Decode disaggregation architecture.

## Quick Start

### 1. Start Prefill Servers

```bash
# Start the first Prefill server
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 30000 \
  --host 0.0.0.0 \
  --disagg-prefill-only \
  --bootstrap-port 30001

# (Optional) Start the second Prefill server
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 30002 \
  --host 0.0.0.0 \
  --disagg-prefill-only \
  --bootstrap-port 30003
```

### 2. Start Decode Servers

```bash
# Start the first Decode server
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 31000 \
  --host 0.0.0.0 \
  --disagg-decode-only

# (Optional) Start the second Decode server
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 31001 \
  --host 0.0.0.0 \
  --disagg-decode-only
```

### 3. Start Mini Load Balancer

```bash
# Method 1: Run basic example
cd sgl-router
cargo run --example mini_lb_basic

# Method 2: Run custom config example
cargo run --example mini_lb_custom_config
```

### 4. Send Test Requests

#### Using cURL

```bash
# Health check
curl http://localhost:8080/health

# Chat completion
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

# Streaming chat completion
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

# Text generation
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

#### Using Python

```bash
# Run Python test client
cd sgl-router/examples
python mini_lb_python_example.py
```

#### Using Rust Client

```bash
cd sgl-router
cargo run --example mini_lb_client
```

## Configuration Guide

### Basic Configuration

```rust
let config = MiniLbConfig::new(
    "0.0.0.0".to_string(),  // Listen address
    8080,                   // Listen port
    vec![                   // Prefill server list
        ("http://localhost:30000".to_string(), Some(30001)),
    ],
    vec![                   // Decode server list
        "http://localhost:31000".to_string(),
    ],
    1800,                   // Timeout in seconds
);
```

### Multi-Server Configuration

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

## Supported Endpoints

### Health Checks
- `GET /health` - Basic health check
- `GET /health_generate` - Backend server health check

### Management Endpoints
- `POST /flush_cache` - Flush cache
- `GET /get_server_info` - Get server information
- `GET /get_model_info` - Get model information

### OpenAI API
- `POST /v1/chat/completions` - Chat completion
- `POST /v1/completions` - Text completion
- `GET /v1/models` - List models

### SGLang API
- `POST /generate` - Text generation

## Troubleshooting

### 1. Load Balancer Won't Start
- Check if port 8080 is already in use
- Verify host and port settings in configuration
- Check error messages in logs

### 2. Request Timeouts
- Increase `timeout_secs` in configuration
- Verify backend servers are running and accessible
- Check network connectivity

### 3. "No Servers Available" Error
- Ensure `prefill_urls` and `decode_urls` are not empty
- Verify server URL format is correct
- Confirm backend servers are running

## Performance Tips

1. **Server Count**: Adjust Prefill and Decode server count based on load
2. **Timeout Settings**: Adjust timeout based on model size and request complexity
3. **Concurrent Connections**: Mini LB handles concurrent requests automatically
4. **Monitoring**: Use `/get_server_info` endpoint to monitor server status

## Next Steps

- Read the [complete documentation](./mini_lb_README.md)
- Check the [implementation summary](./MINI_LB_IMPLEMENTATION_SUMMARY.md)
- Study example code for more usage patterns
- Adjust configuration based on your needs

## Getting Help

If you encounter issues:
1. Check log output
2. Verify backend server status
3. Check network connectivity
4. Submit a GitHub issue

---

Happy coding! 🚀
