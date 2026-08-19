use crate::config::ProviderConfig;
use serde_json::Value;

/// Applies reasoning_effort to the OpenAI request body.
/// Now config-driven: uses ProviderConfig (effort_map, thinking_param, disable_thinking)
/// instead of hardcoded provider match branches.
#[allow(dead_code)]
pub fn apply_reasoning_effort(body: &mut Value, effort: Option<&str>, provider: &ProviderConfig) {
    let effort = match effort {
        Some(e) => e.to_lowercase(),
        None => return,
    };

    let effort = effort.trim();

    match effort {
        "off" | "disabled" | "none" | "false" => {
            if provider.disable_thinking {
                // Cannot turn off thinking; set to lowest effort
                let lowest = provider
                    .effort_map
                    .get("low")
                    .cloned()
                    .unwrap_or_else(|| "low".to_string());
                body[&provider.effort_param] = serde_json::json!(lowest);
                if let Some(obj) = body.as_object_mut() {
                    if let Some(ref tp) = provider.thinking_param {
                        obj.remove(tp);
                    }
                }
            } else {
                // Set thinking.type = disabled
                if let Some(ref tp) = provider.thinking_param {
                    body[tp] = serde_json::json!({
                        "type": provider.thinking_type_disabled.as_deref().unwrap_or("disabled")
                    });
                }
                // Remove reasoning_effort
                if let Some(obj) = body.as_object_mut() {
                    obj.remove(&provider.effort_param);
                }
            }
        }
        _ => {
            // Map effort through effort_map, default to "high" if unknown
            let mapped = provider
                .effort_map
                .get(effort)
                .cloned()
                .unwrap_or_else(|| "high".to_string());
            body[&provider.effort_param] = serde_json::json!(mapped);

            // Set thinking.type = enabled if provider supports it
            if let Some(ref tp) = provider.thinking_param {
                body[tp] = serde_json::json!({
                    "type": provider.thinking_type_enabled.as_deref().unwrap_or("enabled")
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn deepseek_provider() -> ProviderConfig {
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
        }
    }

    fn kimi_provider() -> ProviderConfig {
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
        }
    }

    #[test]
    fn test_deepseek_max() {
        let mut body = serde_json::json!({});
        apply_reasoning_effort(&mut body, Some("max"), &deepseek_provider());
        assert_eq!(body["reasoning_effort"], "max");
        assert_eq!(body["thinking"]["type"], "enabled");
    }

    #[test]
    fn test_deepseek_off() {
        let mut body = serde_json::json!({});
        apply_reasoning_effort(&mut body, Some("off"), &deepseek_provider());
        assert_eq!(body["thinking"]["type"], "disabled");
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn test_deepseek_low() {
        let mut body = serde_json::json!({});
        apply_reasoning_effort(&mut body, Some("low"), &deepseek_provider());
        assert_eq!(body["reasoning_effort"], "high");
        assert_eq!(body["thinking"]["type"], "enabled");
    }

    #[test]
    fn test_no_effort() {
        let mut body = serde_json::json!({});
        apply_reasoning_effort(&mut body, None, &deepseek_provider());
        assert!(body.as_object().unwrap().is_empty());
    }

    #[test]
    fn test_kimi_max() {
        let mut body = serde_json::json!({});
        apply_reasoning_effort(&mut body, Some("max"), &kimi_provider());
        assert_eq!(body["reasoning_effort"], "max");
        // Kimi does NOT support thinking.type
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn test_kimi_xhigh() {
        let mut body = serde_json::json!({});
        apply_reasoning_effort(&mut body, Some("xhigh"), &kimi_provider());
        assert_eq!(body["reasoning_effort"], "max");
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn test_kimi_off() {
        let mut body = serde_json::json!({});
        apply_reasoning_effort(&mut body, Some("off"), &kimi_provider());
        // Kimi can't turn off thinking, falls back to lowest effort
        assert_eq!(body["reasoning_effort"], "low");
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn test_kimi_low() {
        let mut body = serde_json::json!({});
        apply_reasoning_effort(&mut body, Some("low"), &kimi_provider());
        assert_eq!(body["reasoning_effort"], "low");
        assert!(body.get("thinking").is_none());
    }
}
