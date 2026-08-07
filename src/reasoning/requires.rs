use crate::config::Config;

/// Now config-driven: looks up the model in ModelProfile.reasoning_enabled.
pub fn requires_reasoning_content(model: &str, config: &Config) -> bool {
    config
        .model_profile(model)
        .map(|p| p.reasoning_enabled)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::collections::HashMap;

    fn test_config() -> Config {
        let mut providers = HashMap::new();
        providers.insert(
            "deepseek".to_string(),
            crate::config::ProviderConfig {
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
                    m.insert("high".to_string(), "high".to_string());
                    m.insert("max".to_string(), "max".to_string());
                    m
                },
                responses_reasoning_summary: None,
            },
        );

        let model_profiles = vec![
            crate::config::ModelProfile {
                name: "deepseek-v4-pro".to_string(),
                provider: "deepseek".to_string(),
                reasoning_enabled: true,
                reasoning_replay: true,
                toolcall_requires_reasoning: true,
                aliases: vec!["deepseek-chat".to_string(), "deepseek-reasoner".to_string()],
                wire_api: crate::config::WireApi::ChatCompletions,
            },
            crate::config::ModelProfile {
                name: "deepseek-v4-flash".to_string(),
                provider: "deepseek".to_string(),
                reasoning_enabled: true,
                reasoning_replay: true,
                toolcall_requires_reasoning: true,
                aliases: vec![],
                wire_api: crate::config::WireApi::ChatCompletions,
            },
            crate::config::ModelProfile {
                name: "glm-5.2".to_string(),
                provider: "glm".to_string(),
                reasoning_enabled: true,
                reasoning_replay: false,
                toolcall_requires_reasoning: false,
                aliases: vec![],
                wire_api: crate::config::WireApi::ChatCompletions,
            },
            crate::config::ModelProfile {
                name: "kimi-k3".to_string(),
                provider: "fireworks".to_string(),
                reasoning_enabled: true,
                reasoning_replay: true,
                toolcall_requires_reasoning: false,
                aliases: vec![],
                wire_api: crate::config::WireApi::ChatCompletions,
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

        Config {
            listen_addr: "0.0.0.0:11435".to_string(),
            eswitch_url: "http://127.0.0.1:11434".to_string(),
            api_key: "test-key".to_string(),
            log_level: "info".to_string(),
            model_mapping,
            default_model: "deepseek-v4-pro".to_string(),
            model_profiles,
            providers,
            profile_by_name,
        }
    }

    #[test]
    fn test_deepseek_v4_models() {
        let config = test_config();
        assert!(requires_reasoning_content("deepseek-v4-pro", &config));
        assert!(requires_reasoning_content("deepseek-v4-flash", &config));
    }

    #[test]
    fn test_deepseek_aliases() {
        let config = test_config();
        assert!(requires_reasoning_content("deepseek-chat", &config));
        assert!(requires_reasoning_content("deepseek-reasoner", &config));
    }

    #[test]
    fn test_non_reasoning_models() {
        let config = test_config();
        assert!(!requires_reasoning_content("gpt-4", &config));
        assert!(!requires_reasoning_content("claude-sonnet-4", &config));
        assert!(!requires_reasoning_content("llama-3", &config));
    }

    #[test]
    fn test_kimi_models() {
        let config = test_config();
        assert!(requires_reasoning_content("kimi-k3", &config));
    }

    #[test]
    fn test_glm_model() {
        let config = test_config();
        assert!(requires_reasoning_content("glm-5.2", &config));
    }
}
