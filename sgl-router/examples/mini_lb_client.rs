/// Example client for testing the Mini Load Balancer
///
/// This example demonstrates how to send requests to the mini load balancer.
///
/// Usage:
///   cargo run --example mini_lb_client

use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let lb_url = "http://localhost:8080";

    println!("Testing Mini Load Balancer at {}", lb_url);

    // Test 1: Health check
    println!("\n1. Testing /health endpoint...");
    let health_response = client
        .get(&format!("{}/health", lb_url))
        .send()
        .await?;
    println!("   Status: {}", health_response.status());

    // Test 2: Get models
    println!("\n2. Testing /v1/models endpoint...");
    match client
        .get(&format!("{}/v1/models", lb_url))
        .send()
        .await
    {
        Ok(models_response) => {
            println!("   Status: {}", models_response.status());
            if models_response.status().is_success() {
                let models: serde_json::Value = models_response.json().await?;
                println!("   Response: {}", serde_json::to_string_pretty(&models)?);
            }
        }
        Err(e) => {
            println!("   Error: {}", e);
        }
    }

    // Test 3: Send a non-streaming chat completion request
    println!("\n3. Testing /v1/chat/completions (non-streaming)...");
    let chat_request = json!({
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": false
    });

    match client
        .post(&format!("{}/v1/chat/completions", lb_url))
        .json(&chat_request)
        .send()
        .await
    {
        Ok(chat_response) => {
            println!("   Status: {}", chat_response.status());
            if chat_response.status().is_success() {
                let response: serde_json::Value = chat_response.json().await?;
                println!("   Response: {}", serde_json::to_string_pretty(&response)?);
            }
        }
        Err(e) => {
            println!("   Error: {}", e);
        }
    }

    // Test 4: Send a streaming chat completion request
    println!("\n4. Testing /v1/chat/completions (streaming)...");
    let streaming_request = json!({
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": "Count from 1 to 5."
            }
        ],
        "temperature": 0.7,
        "max_tokens": 50,
        "stream": true
    });

    match client
        .post(&format!("{}/v1/chat/completions", lb_url))
        .json(&streaming_request)
        .send()
        .await
    {
        Ok(streaming_response) => {
            println!("   Status: {}", streaming_response.status());
            if streaming_response.status().is_success() {
                println!("   Streaming response (first 500 bytes):");
                let bytes = streaming_response.bytes().await?;
                let preview = String::from_utf8_lossy(&bytes[..bytes.len().min(500)]);
                println!("   {}", preview);
            }
        }
        Err(e) => {
            println!("   Error: {}", e);
        }
    }

    // Test 5: Send a generate request
    println!("\n5. Testing /generate endpoint...");
    let generate_request = json!({
        "text": "Once upon a time",
        "sampling_params": {
            "temperature": 0.8,
            "max_new_tokens": 50
        },
        "stream": false
    });

    match client
        .post(&format!("{}/generate", lb_url))
        .json(&generate_request)
        .send()
        .await
    {
        Ok(generate_response) => {
            println!("   Status: {}", generate_response.status());
            if generate_response.status().is_success() {
                let response: serde_json::Value = generate_response.json().await?;
                println!("   Response: {}", serde_json::to_string_pretty(&response)?);
            }
        }
        Err(e) => {
            println!("   Error: {}", e);
        }
    }

    println!("\nAll tests completed!");

    Ok(())
}
