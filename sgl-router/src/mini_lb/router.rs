use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use parking_lot::RwLock;
use rand::Rng;
use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tracing::{error, info, warn};

use super::types::{MiniLbConfig, ServerPair};

/// Mini Load Balancer for Prefill-Decode Disaggregation
///
/// This implements a minimal load balancer that distributes requests between
/// prefill and decode servers using a random selection strategy.
pub struct MiniLoadBalancer {
    config: MiniLbConfig,
    client: Client,
    /// Current index for round-robin selection of prefill servers (not used in random mode)
    prefill_index: Arc<RwLock<usize>>,
    /// Current index for round-robin selection of decode servers (not used in random mode)
    decode_index: Arc<RwLock<usize>>,
}

impl MiniLoadBalancer {
    pub fn new(config: MiniLbConfig) -> Result<Self, String> {
        config.validate()?;

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        Ok(Self {
            config,
            client,
            prefill_index: Arc::new(RwLock::new(0)),
            decode_index: Arc::new(RwLock::new(0)),
        })
    }

    /// Select a random pair of prefill and decode servers
    pub fn select_pair(&self) -> ServerPair {
        let mut rng = rand::thread_rng();
        
        let prefill_idx = rng.gen_range(0..self.config.prefill_urls.len());
        let decode_idx = rng.gen_range(0..self.config.decode_urls.len());

        let (prefill_url, prefill_bootstrap_port) = self.config.prefill_urls[prefill_idx].clone();
        let decode_url = self.config.decode_urls[decode_idx].clone();

        ServerPair::new(prefill_url, prefill_bootstrap_port, decode_url)
    }

    /// Select a pair using round-robin strategy (alternative to random)
    #[allow(dead_code)]
    pub fn select_pair_round_robin(&self) -> ServerPair {
        let mut prefill_idx = self.prefill_index.write();
        let mut decode_idx = self.decode_index.write();

        let (prefill_url, prefill_bootstrap_port) = 
            self.config.prefill_urls[*prefill_idx % self.config.prefill_urls.len()].clone();
        let decode_url = 
            self.config.decode_urls[*decode_idx % self.config.decode_urls.len()].clone();

        *prefill_idx = (*prefill_idx + 1) % self.config.prefill_urls.len();
        *decode_idx = (*decode_idx + 1) % self.config.decode_urls.len();

        ServerPair::new(prefill_url, prefill_bootstrap_port, decode_url)
    }

    /// Create the Axum router with all endpoints
    pub fn create_router(self) -> Router {
        let shared_state = Arc::new(self);

        Router::new()
            .route("/health", get(health_check))
            .route("/health_generate", get(health_generate))
            .route("/flush_cache", post(flush_cache))
            .route("/get_server_info", get(get_server_info))
            .route("/get_model_info", get(get_model_info))
            .route("/generate", post(handle_generate))
            .route("/v1/chat/completions", post(handle_chat_completions))
            .route("/v1/completions", post(handle_completions))
            .route("/v1/models", get(get_models))
            .with_state(shared_state)
    }

    /// Start the load balancer server
    pub async fn start(self) -> Result<(), Box<dyn std::error::Error>> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        info!("Starting Mini Load Balancer on {}", addr);
        
        warn!("MiniLB is only for debugging purposes, it only supports random policy!");

        let app = self.create_router();
        let listener = TcpListener::bind(&addr).await?;
        
        info!("Mini Load Balancer listening on {}", addr);
        axum::serve(listener, app).await?;

        Ok(())
    }
}

// ===== HTTP Endpoint Handlers =====

async fn health_check() -> impl IntoResponse {
    StatusCode::OK
}

async fn health_generate(State(lb): State<Arc<MiniLoadBalancer>>) -> impl IntoResponse {
    // Check health of all prefill and decode servers
    let mut tasks = Vec::new();
    
    for (prefill_url, _) in &lb.config.prefill_urls {
        let client = lb.client.clone();
        let url = format!("{}/health_generate", prefill_url);
        tasks.push(tokio::spawn(async move {
            client.get(&url).send().await
        }));
    }

    for decode_url in &lb.config.decode_urls {
        let client = lb.client.clone();
        let url = format!("{}/health_generate", decode_url);
        tasks.push(tokio::spawn(async move {
            client.get(&url).send().await
        }));
    }

    // Wait for all health checks
    for task in tasks {
        if let Ok(Err(e)) = task.await {
            error!("Health check failed: {}", e);
            return StatusCode::SERVICE_UNAVAILABLE;
        }
    }

    StatusCode::OK
}

async fn flush_cache(State(lb): State<Arc<MiniLoadBalancer>>) -> impl IntoResponse {
    let mut tasks = Vec::new();
    
    for (prefill_url, _) in &lb.config.prefill_urls {
        let client = lb.client.clone();
        let url = format!("{}/flush_cache", prefill_url);
        tasks.push(tokio::spawn(async move {
            client.post(&url).send().await
        }));
    }

    for decode_url in &lb.config.decode_urls {
        let client = lb.client.clone();
        let url = format!("{}/flush_cache", decode_url);
        tasks.push(tokio::spawn(async move {
            client.post(&url).send().await
        }));
    }

    for task in tasks {
        if let Ok(Err(e)) = task.await {
            error!("Flush cache failed: {}", e);
        }
    }

    StatusCode::OK
}

async fn get_server_info(State(lb): State<Arc<MiniLoadBalancer>>) -> impl IntoResponse {
    let mut prefill_infos = Vec::new();
    let mut decode_infos = Vec::new();
    let mut all_internal_states = Vec::new();

    // Get info from prefill servers
    for (prefill_url, _) in &lb.config.prefill_urls {
        let url = format!("{}/get_server_info", prefill_url);
        match lb.client.get(&url).send().await {
            Ok(resp) => {
                if let Ok(info) = resp.json::<Value>().await {
                    prefill_infos.push(info);
                }
            }
            Err(e) => error!("Failed to get prefill server info: {}", e),
        }
    }

    // Get info from decode servers
    for decode_url in &lb.config.decode_urls {
        let url = format!("{}/get_server_info", decode_url);
        match lb.client.get(&url).send().await {
            Ok(resp) => {
                if let Ok(info) = resp.json::<Value>().await {
                    if let Some(states) = info.get("internal_states") {
                        if let Some(arr) = states.as_array() {
                            all_internal_states.extend(arr.clone());
                        }
                    }
                    decode_infos.push(info);
                }
            }
            Err(e) => error!("Failed to get decode server info: {}", e),
        }
    }

    let response = if !all_internal_states.is_empty() {
        json!({
            "internal_states": all_internal_states,
            "prefill": prefill_infos,
            "decode": decode_infos,
        })
    } else {
        json!({
            "internal_states": [{
                "last_gen_throughput": 0.0,
                "avg_spec_accept_length": null,
            }],
            "prefill": prefill_infos,
            "decode": decode_infos,
        })
    };

    Json(response)
}

async fn get_model_info(State(lb): State<Arc<MiniLoadBalancer>>) -> Result<Json<Value>, StatusCode> {
    if lb.config.prefill_urls.is_empty() {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    let prefill_url = &lb.config.prefill_urls[0].0;
    let url = format!("{}/get_model_info", prefill_url);

    match lb.client.get(&url).send().await {
        Ok(resp) => {
            if resp.status().is_success() {
                match resp.json::<Value>().await {
                    Ok(model_info) => Ok(Json(model_info)),
                    Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
                }
            } else {
                Err(StatusCode::BAD_GATEWAY)
            }
        }
        Err(_) => Err(StatusCode::SERVICE_UNAVAILABLE),
    }
}

async fn get_models(State(lb): State<Arc<MiniLoadBalancer>>) -> Result<Json<Value>, StatusCode> {
    if lb.config.prefill_urls.is_empty() {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    let prefill_url = &lb.config.prefill_urls[0].0;
    let url = format!("{}/v1/models", prefill_url);

    match lb.client.get(&url).send().await {
        Ok(resp) => {
            if resp.status().is_success() {
                match resp.json::<Value>().await {
                    Ok(models) => Ok(Json(models)),
                    Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
                }
            } else {
                Err(StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY))
            }
        }
        Err(_) => Err(StatusCode::SERVICE_UNAVAILABLE),
    }
}

async fn handle_generate(
    State(lb): State<Arc<MiniLoadBalancer>>,
    Json(mut request): Json<Value>,
) -> Response {
    let server_pair = lb.select_pair();
    modify_request_with_bootstrap(&mut request, &server_pair);

    let is_stream = request.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    if is_stream {
        handle_streaming_request(lb, request, server_pair, "generate").await
    } else {
        handle_non_streaming_request(lb, request, server_pair, "generate").await
    }
}

async fn handle_chat_completions(
    State(lb): State<Arc<MiniLoadBalancer>>,
    Json(mut request): Json<Value>,
) -> Response {
    let server_pair = lb.select_pair();
    modify_request_with_bootstrap(&mut request, &server_pair);

    let is_stream = request.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    if is_stream {
        handle_streaming_request(lb, request, server_pair, "v1/chat/completions").await
    } else {
        handle_non_streaming_request(lb, request, server_pair, "v1/chat/completions").await
    }
}

async fn handle_completions(
    State(lb): State<Arc<MiniLoadBalancer>>,
    Json(mut request): Json<Value>,
) -> Response {
    let server_pair = lb.select_pair();
    modify_request_with_bootstrap(&mut request, &server_pair);

    let is_stream = request.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    if is_stream {
        handle_streaming_request(lb, request, server_pair, "v1/completions").await
    } else {
        handle_non_streaming_request(lb, request, server_pair, "v1/completions").await
    }
}

// ===== Helper Functions =====

fn modify_request_with_bootstrap(request: &mut Value, server_pair: &ServerPair) {
    let hostname = extract_hostname(&server_pair.prefill_url);
    let bootstrap_port = server_pair.prefill_bootstrap_port.unwrap_or(30000);
    let bootstrap_room = generate_bootstrap_room();

    if let Some(obj) = request.as_object_mut() {
        obj.insert("bootstrap_host".to_string(), json!(hostname));
        obj.insert("bootstrap_port".to_string(), json!(bootstrap_port));
        obj.insert("bootstrap_room".to_string(), json!(bootstrap_room));
    }
}

fn extract_hostname(url: &str) -> String {
    if let Ok(parsed) = url::Url::parse(url) {
        if let Some(host) = parsed.host_str() {
            // Wrap IPv6 addresses in brackets
            if host.contains(':') && !host.starts_with('[') {
                return format!("[{}]", host);
            }
            return host.to_string();
        }
    }
    url.to_string()
}

fn generate_bootstrap_room() -> i64 {
    rand::thread_rng().gen_range(0..i64::MAX)
}

async fn handle_non_streaming_request(
    lb: Arc<MiniLoadBalancer>,
    request: Value,
    server_pair: ServerPair,
    endpoint: &str,
) -> Response {
    let prefill_url = format!("{}/{}", server_pair.prefill_url, endpoint);
    let decode_url = format!("{}/{}", server_pair.decode_url, endpoint);

    // Send requests to both prefill and decode servers concurrently
    let prefill_task = lb.client.post(&prefill_url).json(&request).send();
    let decode_task = lb.client.post(&decode_url).json(&request).send();

    match tokio::try_join!(prefill_task, decode_task) {
        Ok((prefill_resp, decode_resp)) => {
            // Check if we need to merge logprobs
            let has_logprob = request.get("return_logprob").and_then(|v| v.as_bool()).unwrap_or(false);

            if has_logprob {
                // Merge input_token_logprobs from prefill into decode response
                if let (Ok(mut decode_json), Ok(prefill_json)) = 
                    (decode_resp.json::<Value>().await, prefill_resp.json::<Value>().await) {
                    if let Some(decode_meta) = decode_json.get_mut("meta_info") {
                        if let Some(prefill_meta) = prefill_json.get("meta_info") {
                            if let Some(prefill_logprobs) = prefill_meta.get("input_token_logprobs") {
                                if let Some(decode_logprobs) = decode_meta.get("input_token_logprobs") {
                                    // Merge the logprobs
                                    if let (Some(p_arr), Some(d_arr)) = 
                                        (prefill_logprobs.as_array(), decode_logprobs.as_array()) {
                                        let mut merged = p_arr.clone();
                                        merged.extend(d_arr.clone());
                                        if let Some(meta_obj) = decode_meta.as_object_mut() {
                                            meta_obj.insert("input_token_logprobs".to_string(), json!(merged));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    return Json(decode_json).into_response();
                }
            }

            // No logprob merging needed, just return decode response
            match decode_resp.json::<Value>().await {
                Ok(json) => Json(json).into_response(),
                Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
            }
        }
        Err(e) => {
            error!("Request failed: {}", e);
            StatusCode::BAD_GATEWAY.into_response()
        }
    }
}

async fn handle_streaming_request(
    lb: Arc<MiniLoadBalancer>,
    request: Value,
    server_pair: ServerPair,
    endpoint: &str,
) -> Response {
    let prefill_url = format!("{}/{}", server_pair.prefill_url, endpoint);
    let decode_url = format!("{}/{}", server_pair.decode_url, endpoint);

    // For streaming, we only need the decode server's stream
    // The prefill server will complete first
    let prefill_task = lb.client.post(&prefill_url).json(&request).send();
    let decode_task = lb.client.post(&decode_url).json(&request).send();

    match tokio::try_join!(prefill_task, decode_task) {
        Ok((_prefill_resp, decode_resp)) => {
            // Stream the decode response
            let stream = decode_resp.bytes_stream();
            let body = axum::body::Body::from_stream(stream);
            
            Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "text/event-stream")
                .body(body)
                .unwrap()
        }
        Err(e) => {
            error!("Streaming request failed: {}", e);
            StatusCode::BAD_GATEWAY.into_response()
        }
    }
}
