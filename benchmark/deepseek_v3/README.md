# DeepSeek V3 Support

The SGLang and DeepSeek teams collaborated to get DeepSeek V3 FP8 running on NVIDIA and AMD GPUs **from day one**. SGLang also supports [MLA optimization](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#deepseek-multi-head-latent-attention-mla-throughput-optimizations) and [DP attention](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models), making SGLang one of the best open-source LLM engines for running DeepSeek models. SGLang is the inference engine recommended by the official [DeepSeek team](https://github.com/deepseek-ai/DeepSeek-V3/tree/main?tab=readme-ov-file#62-inference-with-sglang-recommended).

Special thanks to Meituan's Search & Recommend Platform Team and Baseten's Model Performance Team for implementing the model, and DataCrunch for providing GPU resources.

## Hardware Recommendation
- 8 x NVIDIA H200 GPUs

If you do not have GPUs with large enough memory, please try multi-node tensor parallelism. There is an example serving with [2 H20 nodes](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-2-h208) below.

## Installation & Launch

If you encounter errors when starting the server, ensure the weights have finished downloading. It's recommended to download them beforehand or restart multiple times until all weights are downloaded.

### Using Docker (Recommended)
```bash
# Pull latest image
# https://hub.docker.com/r/lmsysorg/sglang/tags
docker pull lmsysorg/sglang:latest

# Launch
docker run --gpus all --shm-size 32g -p 30000:30000 -v ~/.cache/huggingface:/root/.cache/huggingface --ipc=host lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --port 30000
```

For high QPS scenarios, add the `--enable-dp-attention` argument to boost throughput.

### Using pip
```bash
# Installation
pip install "sglang[all]>=0.4.1.post3" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer

# Launch
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code
```

For high QPS scenarios, add the `--enable-dp-attention` argument to boost throughput.

### Example with OpenAI API

```python3
import openai
client = openai.Client(
    base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

# Chat completion
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)
print(response)
```
### Example serving with 2 H20*8
For example, there are two H20 nodes, each with 8 GPUs. The first node's IP is `10.0.0.1`, and the second node's IP is `10.0.0.2`.

```bash
# node 1
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --nccl-init 10.0.0.1:5000 --nnodes 2 --node-rank 0 --trust-remote-code

# node 2
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --nccl-init 10.0.0.1:5000 --nnodes 2 --node-rank 1 --trust-remote-code
```

If you have two H100 nodes, the usage is similar to the aforementioned H20.

## DeepSeek V3 Optimization Plan

https://github.com/sgl-project/sglang/issues/2591
