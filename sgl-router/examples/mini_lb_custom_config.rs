/// Advanced example with custom configuration
///
/// This example shows how to create a mini load balancer with more
/// complex configuration including multiple servers and custom timeouts.
///
/// Usage:
///   cargo run --example mini_lb_custom_config

use sglang_router_rs::mini_lb::{MiniLbConfig, MiniLoadBalancer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging with DEBUG level for more details
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    // Create a more complex configuration
    let config = MiniLbConfig {
        host: "127.0.0.1".to_string(),
        port: 9090,
        prefill_urls: vec![
            ("http://prefill-1.example.com:30000".to_string(), Some(30001)),
            ("http://prefill-2.example.com:30000".to_string(), Some(30001)),
            ("http://prefill-3.example.com:30000".to_string(), Some(30001)),
        ],
        decode_urls: vec![
            "http://decode-1.example.com:31000".to_string(),
            "http://decode-2.example.com:31000".to_string(),
            "http://decode-3.example.com:31000".to_string(),
            "http://decode-4.example.com:31000".to_string(),
        ],
        timeout_secs: 3600, // 1 hour timeout for long-running requests
    };

    // Validate the configuration
    config.validate()?;

    // Create the load balancer
    let lb = MiniLoadBalancer::new(config)?;

    println!("Starting Mini Load Balancer with custom configuration");
    println!("Configuration:");
    println!("  - Host: 127.0.0.1");
    println!("  - Port: 9090");
    println!("  - Prefill servers: 3");
    println!("  - Decode servers: 4");
    println!("  - Timeout: 3600 seconds (1 hour)");
    println!("\nThe load balancer will randomly distribute requests across:");
    println!("  - 3 prefill servers × 4 decode servers = 12 possible combinations");

    // Start the load balancer
    lb.start().await?;

    Ok(())
}
