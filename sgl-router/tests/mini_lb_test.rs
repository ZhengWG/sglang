/// Integration tests for Mini Load Balancer

use sglang_router_rs::mini_lb::{MiniLbConfig, MiniLoadBalancer};

#[test]
fn test_config_creation() {
    let config = MiniLbConfig::new(
        "127.0.0.1".to_string(),
        8080,
        vec![("http://localhost:30000".to_string(), Some(30001))],
        vec!["http://localhost:31000".to_string()],
        1800,
    );

    assert_eq!(config.host, "127.0.0.1");
    assert_eq!(config.port, 8080);
    assert_eq!(config.prefill_urls.len(), 1);
    assert_eq!(config.decode_urls.len(), 1);
    assert_eq!(config.timeout_secs, 1800);
}

#[test]
fn test_config_validation_success() {
    let config = MiniLbConfig::new(
        "0.0.0.0".to_string(),
        8080,
        vec![("http://localhost:30000".to_string(), Some(30001))],
        vec!["http://localhost:31000".to_string()],
        1800,
    );

    assert!(config.validate().is_ok());
}

#[test]
fn test_config_validation_no_prefill() {
    let config = MiniLbConfig::new(
        "0.0.0.0".to_string(),
        8080,
        vec![],
        vec!["http://localhost:31000".to_string()],
        1800,
    );

    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_no_decode() {
    let config = MiniLbConfig::new(
        "0.0.0.0".to_string(),
        8080,
        vec![("http://localhost:30000".to_string(), Some(30001))],
        vec![],
        1800,
    );

    assert!(config.validate().is_err());
}

#[test]
fn test_load_balancer_creation() {
    let config = MiniLbConfig::new(
        "127.0.0.1".to_string(),
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

    let lb = MiniLoadBalancer::new(config);
    assert!(lb.is_ok());
}

#[test]
fn test_load_balancer_invalid_config() {
    let config = MiniLbConfig::new(
        "127.0.0.1".to_string(),
        8080,
        vec![],
        vec!["http://localhost:31000".to_string()],
        1800,
    );

    let lb = MiniLoadBalancer::new(config);
    assert!(lb.is_err());
}

#[test]
fn test_select_pair() {
    let config = MiniLbConfig::new(
        "127.0.0.1".to_string(),
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

    let lb = MiniLoadBalancer::new(config).unwrap();

    // Test that select_pair returns valid pairs
    for _ in 0..10 {
        let pair = lb.select_pair();
        assert!(
            pair.prefill_url == "http://localhost:30000"
                || pair.prefill_url == "http://localhost:30002"
        );
        assert!(
            pair.decode_url == "http://localhost:31000"
                || pair.decode_url == "http://localhost:31001"
        );
        assert!(
            pair.prefill_bootstrap_port == Some(30001)
                || pair.prefill_bootstrap_port == Some(30003)
        );
    }
}

#[test]
fn test_select_pair_round_robin() {
    let config = MiniLbConfig::new(
        "127.0.0.1".to_string(),
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

    let lb = MiniLoadBalancer::new(config).unwrap();

    // Test round-robin selection
    let pair1 = lb.select_pair_round_robin();
    assert_eq!(pair1.prefill_url, "http://localhost:30000");
    assert_eq!(pair1.decode_url, "http://localhost:31000");

    let pair2 = lb.select_pair_round_robin();
    assert_eq!(pair2.prefill_url, "http://localhost:30002");
    assert_eq!(pair2.decode_url, "http://localhost:31001");

    let pair3 = lb.select_pair_round_robin();
    assert_eq!(pair3.prefill_url, "http://localhost:30000");
    assert_eq!(pair3.decode_url, "http://localhost:31000");
}
