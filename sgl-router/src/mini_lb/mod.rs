/// Mini Load Balancer for Prefill-Decode Disaggregation
///
/// This module provides a minimal HTTP load balancer for testing prefill and decode
/// servers in a disaggregated setup. It implements round-robin and random selection
/// strategies for distributing requests across prefill and decode server pairs.

pub mod router;
pub mod types;

pub use router::MiniLoadBalancer;
pub use types::{MiniLbConfig, ServerPair};
