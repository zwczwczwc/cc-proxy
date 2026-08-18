use crate::config::Config;

/// Determines whether reasoning content should be replayed to the API.
/// Now config-driven: checks ModelProfile.reasoning_replay instead of delegating
/// to the old hardcoded requires_reasoning_content.
pub fn should_replay_reasoning_content(model: &str, effort: Option<&str>, config: &Config) -> bool {
    // If effort is explicitly "off" / "disabled" / "none" / "false", don't replay
    if let Some(eff) = effort {
        let eff_lower = eff.to_lowercase();
        if eff_lower == "off"
            || eff_lower == "disabled"
            || eff_lower == "none"
            || eff_lower == "false"
        {
            return false;
        }
    }
    config
        .model_profile(model)
        .map(|p| p.reasoning_replay)
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

    #[test]
    fn test_replay_for_deepseek() {
        let config = test_config();
        assert!(should_replay_reasoning_content(
            "deepseek-v4-pro",
            None,
            &config
        ));
        assert!(should_replay_reasoning_content(
            "deepseek-v4-pro",
            Some("high"),
            &config
        ));
    }

    #[test]
    fn test_no_replay_when_off() {
        let config = test_config();
        assert!(!should_replay_reasoning_content(
            "deepseek-v4-pro",
            Some("off"),
            &config
        ));
        assert!(!should_replay_reasoning_content(
            "deepseek-v4-pro",
            Some("disabled"),
            &config
        ));
        assert!(!should_replay_reasoning_content(
            "deepseek-v4-pro",
            Some("none"),
            &config
        ));
        assert!(!should_replay_reasoning_content(
            "deepseek-v4-pro",
            Some("false"),
            &config
        ));
    }

    #[test]
    fn test_no_replay_for_non_ds() {
        let config = test_config();
        assert!(!should_replay_reasoning_content("gpt-4", None, &config));
    }

    #[test]
    fn test_no_replay_for_glm() {
        let config = test_config();
        // glm-5.2 has reasoning_replay=false (from实测)
        assert!(!should_replay_reasoning_content("glm-5.2", None, &config));
    }
}
