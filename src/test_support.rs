//! Shared deterministic test fixtures, compiled only under `cfg(test)`.
//!
//! Declared as `#[cfg(test)] mod test_support;` in `src/main.rs`, so this
//! module never ships in the binary. `test_config()` is independent of the
//! host environment (no `MODEL_CONFIG_PATH` / `/etc/cc-proxy/config.toml`
//! reads) so encoders can be exercised deterministically — the golden
//! harness and the `responses/request.rs` tests both use it.

use crate::config::{Config, ModelProfile, ProviderConfig, WireApi};
use std::collections::HashMap;

/// Deterministic `Config` for tests: deepseek/moonshot(kimi)/glm/gpt
/// providers plus the model profiles the golden fixtures reference.
pub(crate) fn test_config() -> Config {
    let mut providers = HashMap::new();

    providers.insert(
        "deepseek".to_string(),
        ProviderConfig {
            reasoning_field: "reasoning_content".to_string(),
            reasoning_field_alt: vec![],
            thinking_param: Some("thinking".to_string()),
            thinking_type_enabled: Some("enabled".to_string()),
            thinking_type_disabled: Some("disabled".to_string()),
            disable_thinking: false,
            effort_param: "reasoning_effort".to_string(),
            effort_map: {
                let mut m = HashMap::new();
                m.insert("low".to_string(), "high".to_string());
                m.insert("medium".to_string(), "high".to_string());
                m.insert("high".to_string(), "high".to_string());
                m.insert("max".to_string(), "max".to_string());
                m.insert("xhigh".to_string(), "max".to_string());
                m
            },
            responses_reasoning_summary: None,
            cache_policy: None,
        },
    );

    providers.insert(
        "moonshot".to_string(),
        ProviderConfig {
            reasoning_field: "reasoning".to_string(),
            reasoning_field_alt: vec![],
            thinking_param: None,
            thinking_type_enabled: None,
            thinking_type_disabled: None,
            disable_thinking: true,
            effort_param: "reasoning_effort".to_string(),
            effort_map: {
                let mut m = HashMap::new();
                m.insert("low".to_string(), "low".to_string());
                m.insert("medium".to_string(), "medium".to_string());
                m.insert("high".to_string(), "high".to_string());
                m.insert("max".to_string(), "max".to_string());
                m.insert("xhigh".to_string(), "max".to_string());
                m
            },
            responses_reasoning_summary: None,
            cache_policy: None,
        },
    );

    providers.insert(
        "glm".to_string(),
        ProviderConfig {
            reasoning_field: "reasoning_content".to_string(),
            reasoning_field_alt: vec![],
            thinking_param: Some("thinking".to_string()),
            thinking_type_enabled: Some("enabled".to_string()),
            thinking_type_disabled: Some("disabled".to_string()),
            disable_thinking: false,
            effort_param: "reasoning_effort".to_string(),
            effort_map: {
                let mut m = HashMap::new();
                m.insert("none".to_string(), "none".to_string());
                m.insert("minimal".to_string(), "minimal".to_string());
                m.insert("low".to_string(), "low".to_string());
                m.insert("medium".to_string(), "medium".to_string());
                m.insert("high".to_string(), "high".to_string());
                m.insert("xhigh".to_string(), "xhigh".to_string());
                m.insert("max".to_string(), "max".to_string());
                m
            },
            responses_reasoning_summary: None,
            cache_policy: None,
        },
    );

    providers.insert(
        "gpt".to_string(),
        ProviderConfig {
            reasoning_field: String::new(),
            reasoning_field_alt: Vec::new(),
            thinking_param: None,
            thinking_type_enabled: None,
            thinking_type_disabled: None,
            disable_thinking: false,
            effort_param: "reasoning_effort".to_string(),
            effort_map: {
                let mut m = HashMap::new();
                m.insert("max".to_string(), "max".to_string());
                m
            },
            responses_reasoning_summary: None,
            cache_policy: None,
        },
    );

    let model_profiles = vec![
        ModelProfile {
            name: "deepseek-v4-pro".to_string(),
            provider: "deepseek".to_string(),
            reasoning_enabled: true,
            reasoning_replay: true,
            toolcall_requires_reasoning: true,
            aliases: vec!["deepseek-chat".to_string(), "deepseek-reasoner".to_string()],
            wire_api: WireApi::ChatCompletions,
        },
        ModelProfile {
            name: "deepseek-v4-flash".to_string(),
            provider: "deepseek".to_string(),
            reasoning_enabled: true,
            reasoning_replay: true,
            toolcall_requires_reasoning: true,
            aliases: vec![],
            wire_api: WireApi::ChatCompletions,
        },
        ModelProfile {
            name: "kimi-k3".to_string(),
            provider: "moonshot".to_string(),
            reasoning_enabled: true,
            reasoning_replay: true,
            toolcall_requires_reasoning: false,
            aliases: vec![],
            wire_api: WireApi::ChatCompletions,
        },
        ModelProfile {
            name: "glm-5.2".to_string(),
            provider: "glm".to_string(),
            reasoning_enabled: true,
            reasoning_replay: false,
            toolcall_requires_reasoning: false,
            aliases: vec![],
            wire_api: WireApi::ChatCompletions,
        },
        ModelProfile {
            name: "gpt-5.6-luna".to_string(),
            provider: "gpt".to_string(),
            reasoning_enabled: true,
            reasoning_replay: false,
            toolcall_requires_reasoning: false,
            aliases: vec![],
            wire_api: WireApi::Responses,
        },
    ];

    let mut profile_by_name = HashMap::new();
    for (i, profile) in model_profiles.iter().enumerate() {
        profile_by_name.insert(profile.name.clone(), i);
        for alias in &profile.aliases {
            profile_by_name.insert(alias.clone(), i);
        }
    }

    let mut model_mapping = HashMap::new();
    model_mapping.insert("claude-opus-4".to_string(), "deepseek-v4-pro".to_string());
    model_mapping.insert(
        "claude-sonnet-4".to_string(),
        "deepseek-v4-flash".to_string(),
    );

    Config {
        listen_addr: "0.0.0.0:11435".to_string(),
        eswitch_url: "http://127.0.0.1:11434".to_string(),
        moonshot_official_url: String::new(),
        moonshot_official_api_key: String::new(),
        api_key: "test-key".to_string(),
        log_level: "info".to_string(),
        model_mapping,
        default_model: "deepseek-v4-pro".to_string(),
        model_profiles,
        providers,
        profile_by_name,
    }
}
