use serde::{Deserialize, Serialize};

/// Configuration for the Mini Load Balancer
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MiniLbConfig {
    /// Host to bind the load balancer server
    pub host: String,
    /// Port to bind the load balancer server
    pub port: u16,
    /// List of prefill server URLs with their bootstrap ports
    pub prefill_urls: Vec<(String, Option<u16>)>,
    /// List of decode server URLs
    pub decode_urls: Vec<String>,
    /// Request timeout in seconds
    pub timeout_secs: u64,
}

impl MiniLbConfig {
    pub fn new(
        host: String,
        port: u16,
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        timeout_secs: u64,
    ) -> Self {
        Self {
            host,
            port,
            prefill_urls,
            decode_urls,
            timeout_secs,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.prefill_urls.is_empty() {
            return Err("At least one prefill server is required".to_string());
        }
        if self.decode_urls.is_empty() {
            return Err("At least one decode server is required".to_string());
        }
        Ok(())
    }
}

/// A pair of prefill and decode servers selected for a request
#[derive(Debug, Clone)]
pub struct ServerPair {
    pub prefill_url: String,
    pub prefill_bootstrap_port: Option<u16>,
    pub decode_url: String,
}

impl ServerPair {
    pub fn new(prefill_url: String, prefill_bootstrap_port: Option<u16>, decode_url: String) -> Self {
        Self {
            prefill_url,
            prefill_bootstrap_port,
            decode_url,
        }
    }
}
