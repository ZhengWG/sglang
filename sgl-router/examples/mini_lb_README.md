# Mini Load Balancer (Rust Implementation)

A minimal HTTP load balancer for Prefill-Decode (P/D) disaggregation in SGLang, implemented in Rust for high performance and reliability.

## Overview

The Mini Load Balancer (`mini_lb`) distributes incoming requests across multiple prefill and decode servers using a random selection strategy. This is particularly useful for:

- **Testing P/D disaggregation setups**: Quickly test your prefill-decode architecture
- **Development environments**: Lightweight load balancing during development
- **Simple deployments**: When you need basic load balancing without complex policies

> **Note**: This is designed for debugging and testing purposes. For production deployments, use the full SGLang router with advanced features like cache-aware routing, health checks, and circuit breakers.

## Features

- ✅ Random server selection for optimal load distribution
- ✅ Support for both streaming and non-streaming requests
- ✅ Automatic bootstrap configuration for P/D communication
- ✅ Health check endpoints for monitoring
- ✅ Compatible with OpenAI API format
- ✅ Built-in request/response merging for logprobs
- ✅ Low latency with async I/O
- ✅ Written in Rust for memory safety and performance

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Mini Load Balancer │  (Random Selection)
│   (Rust/Axum)       │
└─────────┬───────────┘
          │
          ├──────────────┬──────────────┐
          ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Prefill  │   │ Prefill  │   │ Prefill  │
    │ Server 1 │   │ Server 2 │   │ Server 3 │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
         └──────────────┼──────────────┘
                        │ Bootstrap
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  Decode  │   │  Decode  │   │  Decode  │
    │ Server 1 │   │ Server 2 │   │ Server 3 │
    └──────────┘   └──────────┘   └──────────┘
```

## API Endpoints

The Mini Load Balancer exposes the following endpoints:

### Health & Management

- `GET /health` - Basic health check
- `GET /health_generate` - Health check for all backend servers
- `POST /flush_cache` - Flush cache on all backend servers
- `GET /get_server_info` - Get information about backend servers
- `GET /get_model_info` - Get model information

### OpenAI-Compatible API

- `POST /v1/chat/completions` - Chat completion requests (streaming & non-streaming)
- `POST /v1/completions` - Text completion requests
- `GET /v1/models` - List available models

### SGLang-Specific API

- `POST /generate` - Text generation with SGLang format

## Usage Examples

### 1. Basic Rust Example

```rust
use sglang_router_rs::mini_lb::{MiniLbConfig, MiniLoadBalancer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure the load balancer
    let config = MiniLbConfig::new(
        "0.0.0.0".to_string(),
        8080,
        vec![
            ("http://localhost:30000".to_string(), Some(30001)),
            ("http://localhost:30002".to_string(), Some(30003)),
        ],
        vec![
            "http://localhost:31000".to_string(),
            "http://localhost:31001".to_string(),
        ],
        1800,
    );

    // Start the load balancer
    let lb = MiniLoadBalancer::new(config)?;
    lb.start().await?;

    Ok(())
}
```

Run the example:
```bash
cargo run --example mini_lb_basic
```

### 2. Python Client Example

```python
import requests

# Chat completion
response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
)
print(response.json())
```

Run the full Python example:
```bash
python examples/mini_lb_python_example.py
```

### 3. cURL Examples

**Health Check:**
```bash
curl http://localhost:8080/health
```

**Chat Completion:**
```bash
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
```

**Streaming Chat Completion:**
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Count from 1 to 10"}
    ],
    "stream": true,
    "max_tokens": 50
  }'
```

**Text Generation:**
```bash
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

## Configuration

### MiniLbConfig Structure

```rust
pub struct MiniLbConfig {
    /// Host to bind the server (e.g., "0.0.0.0" or "127.0.0.1")
    pub host: String,
    
    /// Port to bind the server (e.g., 8080)
    pub port: u16,
    
    /// List of (prefill_url, bootstrap_port) tuples
    /// Example: [("http://localhost:30000", Some(30001))]
    pub prefill_urls: Vec<(String, Option<u16>)>,
    
    /// List of decode server URLs
    /// Example: ["http://localhost:31000", "http://localhost:31001"]
    pub decode_urls: Vec<String>,
    
    /// Request timeout in seconds
    pub timeout_secs: u64,
}
```

### Example Configurations

**Simple 1:1 Setup:**
```rust
let config = MiniLbConfig::new(
    "0.0.0.0".to_string(),
    8080,
    vec![("http://prefill:30000".to_string(), Some(30001))],
    vec!["http://decode:31000".to_string()],
    1800,
);
```

**Multi-Server Setup:**
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
        "http://decode-4:31000".to_string(),
    ],
    3600,
);
```

## How It Works

1. **Request Reception**: Client sends a request to the load balancer
2. **Server Selection**: Load balancer randomly selects a prefill and decode server pair
3. **Bootstrap Injection**: Adds bootstrap configuration (host, port, room) to the request
4. **Parallel Dispatch**: Sends the request to both prefill and decode servers concurrently
5. **Response Handling**:
   - For non-streaming: Waits for both responses, merges logprobs if needed, returns decode response
   - For streaming: Returns decode server's streaming response
6. **Client Response**: Sends the final response back to the client

### Bootstrap Configuration

The load balancer automatically adds these fields to requests:
- `bootstrap_host`: Hostname of the prefill server
- `bootstrap_port`: Bootstrap port for P/D communication
- `bootstrap_room`: Random room ID for request tracking

## Comparison with Python Implementation

| Feature | Rust Implementation | Python Implementation |
|---------|-------------------|----------------------|
| Performance | Higher throughput, lower latency | Moderate |
| Memory | Lower memory footprint | Higher (Python runtime) |
| Async I/O | Native async with Tokio | aiohttp |
| Type Safety | Compile-time type checking | Runtime type checking |
| Binary Size | ~5-10MB | Requires Python interpreter |
| Startup Time | Milliseconds | Slower (Python startup) |
| Multimodal | Not yet implemented | Supported |

## Running the Examples

### Build the examples:
```bash
cd sgl-router
cargo build --examples
```

### Run basic example:
```bash
cargo run --example mini_lb_basic
```

### Run custom config example:
```bash
cargo run --example mini_lb_custom_config
```

### Run client test:
```bash
# First start the load balancer in one terminal:
cargo run --example mini_lb_basic

# Then in another terminal:
cargo run --example mini_lb_client
```

### Run Python client:
```bash
# Make sure the load balancer is running first
python examples/mini_lb_python_example.py
```

## Testing with Real Servers

To test with actual SGLang servers:

1. **Start Prefill Server:**
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 30000 \
  --host 0.0.0.0 \
  --disagg-prefill-only \
  --bootstrap-port 30001
```

2. **Start Decode Server:**
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 31000 \
  --host 0.0.0.0 \
  --disagg-decode-only
```

3. **Start Mini Load Balancer:**
```bash
cargo run --example mini_lb_basic
```

4. **Send Requests:**
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Limitations

- **Random Policy Only**: Currently only supports random server selection
- **No Health Checks**: Doesn't actively monitor server health
- **No Circuit Breakers**: No automatic failover on server errors
- **Basic Metrics**: Limited observability compared to full router
- **No Multimodal Support**: Vision-language models not yet supported
- **Testing Only**: Not recommended for production use

## Future Improvements

Potential enhancements (contributions welcome!):

- [ ] Add round-robin selection strategy
- [ ] Implement health check monitoring
- [ ] Add basic metrics/observability
- [ ] Support multimodal disaggregation
- [ ] Add retry logic for failed requests
- [ ] Implement connection pooling
- [ ] Add request rate limiting
- [ ] Support custom load balancing strategies

## Troubleshooting

**Load balancer won't start:**
- Check if port 8080 is already in use
- Verify host/port configuration
- Check logs for error messages

**Requests timing out:**
- Increase `timeout_secs` in configuration
- Verify backend servers are running and accessible
- Check network connectivity

**"No servers available" error:**
- Ensure prefill_urls and decode_urls are not empty
- Verify server URLs are correct
- Check that backend servers are running

## License

Same as SGLang project.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## See Also

- [SGLang Documentation](https://sglang.readthedocs.io/)
- [Prefill-Decode Disaggregation Guide](https://sglang.readthedocs.io/en/latest/advanced_features/disaggregation.html)
- [Full Router Documentation](../README.md)
