/// Basic example of using the Mini Load Balancer
///
/// This example demonstrates how to create and start a mini load balancer
/// that distributes requests between prefill and decode servers.
///
/// Usage:
///   cargo run --example mini_lb_basic

use sglang_router_rs::mini_lb::{MiniLbConfig, MiniLoadBalancer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // Configure the mini load balancer
    let config = MiniLbConfig::new(
        "0.0.0.0".to_string(),      // Host
        8080,                         // Port
        vec![
            // Prefill servers with their bootstrap ports
            ("http://localhost:30000".to_string(), Some(30001)),
            ("http://localhost:30002".to_string(), Some(30003)),
        ],
        vec![
            // Decode servers
            "http://localhost:31000".to_string(),
            "http://localhost:31001".to_string(),
        ],
        1800, // Timeout in seconds (30 minutes)
    );

    // Create and start the load balancer
    let lb = MiniLoadBalancer::new(config)?;
    
    println!("Starting Mini Load Balancer on {}:{}", "0.0.0.0", 8080);
    println!("Available endpoints:");
    println!("  - GET  /health");
    println!("  - GET  /health_generate");
    println!("  - POST /flush_cache");
    println!("  - GET  /get_server_info");
    println!("  - GET  /get_model_info");
    println!("  - POST /generate");
    println!("  - POST /v1/chat/completions");
    println!("  - POST /v1/completions");
    println!("  - GET  /v1/models");

    lb.start().await?;

    Ok(())
}
