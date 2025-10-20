#!/usr/bin/env python3
"""
Python example for using the Mini Load Balancer

This script demonstrates how to interact with the Rust-based Mini Load Balancer
from Python using the requests library.

Usage:
    python mini_lb_python_example.py
"""

import requests
import json
from typing import Dict, Any

# Configuration
LB_URL = "http://localhost:8080"


def health_check() -> bool:
    """Check if the load balancer is healthy"""
    try:
        response = requests.get(f"{LB_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def get_models() -> Dict[str, Any]:
    """Get available models from the load balancer"""
    try:
        response = requests.get(f"{LB_URL}/v1/models", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to get models: {e}")
        return {}


def chat_completion(
    messages: list,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    temperature: float = 0.7,
    max_tokens: int = 100,
    stream: bool = False,
) -> Dict[str, Any]:
    """Send a chat completion request"""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    try:
        response = requests.post(
            f"{LB_URL}/v1/chat/completions",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        if stream:
            # For streaming, return the raw response
            return {"stream": response.iter_lines()}
        else:
            return response.json()
    except Exception as e:
        print(f"Chat completion failed: {e}")
        return {"error": str(e)}


def generate_text(
    text: str,
    temperature: float = 0.8,
    max_new_tokens: int = 50,
    stream: bool = False,
) -> Dict[str, Any]:
    """Send a text generation request"""
    payload = {
        "text": text,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        },
        "stream": stream,
    }

    try:
        response = requests.post(
            f"{LB_URL}/generate",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        if stream:
            return {"stream": response.iter_lines()}
        else:
            return response.json()
    except Exception as e:
        print(f"Text generation failed: {e}")
        return {"error": str(e)}


def get_server_info() -> Dict[str, Any]:
    """Get information about backend servers"""
    try:
        response = requests.get(f"{LB_URL}/get_server_info", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to get server info: {e}")
        return {}


def main():
    print("=" * 60)
    print("Mini Load Balancer - Python Example")
    print("=" * 60)

    # Test 1: Health check
    print("\n1. Testing health check...")
    if health_check():
        print("   ✓ Load balancer is healthy")
    else:
        print("   ✗ Load balancer is not responding")
        return

    # Test 2: Get models
    print("\n2. Getting available models...")
    models = get_models()
    if models:
        print(f"   Available models: {json.dumps(models, indent=2)}")
    else:
        print("   No models available or request failed")

    # Test 3: Chat completion (non-streaming)
    print("\n3. Testing chat completion (non-streaming)...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    result = chat_completion(messages, stream=False)
    if "error" not in result:
        print(f"   Response: {json.dumps(result, indent=2)}")
    else:
        print(f"   Error: {result['error']}")

    # Test 4: Text generation
    print("\n4. Testing text generation...")
    result = generate_text("Once upon a time", max_new_tokens=30)
    if "error" not in result:
        print(f"   Generated text: {json.dumps(result, indent=2)}")
    else:
        print(f"   Error: {result['error']}")

    # Test 5: Get server info
    print("\n5. Getting server information...")
    server_info = get_server_info()
    if server_info:
        print(f"   Server info: {json.dumps(server_info, indent=2)}")
    else:
        print("   Failed to get server information")

    # Test 6: Streaming chat completion
    print("\n6. Testing streaming chat completion...")
    messages = [
        {"role": "user", "content": "Count from 1 to 5."},
    ]
    result = chat_completion(messages, stream=True, max_tokens=50)
    if "stream" in result:
        print("   Streaming response:")
        for i, line in enumerate(result["stream"]):
            if i >= 5:  # Just show first 5 chunks
                print("   ...")
                break
            if line:
                print(f"   {line.decode('utf-8')}")
    else:
        print(f"   Error: {result.get('error', 'Unknown error')}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
