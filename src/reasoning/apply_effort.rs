use serde_json::Value;

/// Applies reasoning_effort to the OpenAI request body.
/// Simplified from CodeWhale client.rs L1103-L1261 — only DeepSeek/OpenRouter branches.
pub fn apply_reasoning_effort(body: &mut Value, effort: Option<&str>, provider: &str) {
    let effort = match effort {
        Some(e) => e.to_lowercase(),
        None => return,
    };

    let effort = effort.trim();

    match effort {
        "off" | "disabled" | "none" | "false" => {
            // Disable thinking
            if provider == "deepseek" || provider == "openrouter" {
                body["thinking"] = serde_json::json!({"type": "disabled"});
            }
            // Remove any existing reasoning_effort
            if let Some(obj) = body.as_object_mut() {
                obj.remove("reasoning_effort");
            }
        }
        "low" | "medium" | "high" => {
            if provider == "deepseek" {
                body["reasoning_effort"] = serde_json::json!("high");
                body["thinking"] = serde_json::json!({"type": "enabled"});
            } else if provider == "openrouter" {
                body["reasoning_effort"] = serde_json::json!(effort);
                body["thinking"] = serde_json::json!({"type": "enabled"});
            }
        }
        "max" | "xhigh" => {
            if provider == "deepseek" {
                body["reasoning_effort"] = serde_json::json!("max");
                body["thinking"] = serde_json::json!({"type": "enabled"});
            } else if provider == "openrouter" {
                body["reasoning_effort"] = serde_json::json!("xhigh");
                body["thinking"] = serde_json::json!({"type": "enabled"});
            }
        }
        _ => {
            // Unknown effort level, treat as enabled with high
            if provider == "deepseek" {
                body["reasoning_effort"] = serde_json::json!("high");
                body["thinking"] = serde_json::json!({"type": "enabled"});
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deepseek_max() {
        let mut body = serde_json::json!({});
        apply_reasoning_effort(&mut body, Some("max"), "deepseek");
        assert_eq!(body["reasoning_effort"], "max");
        assert_eq!(body["thinking"]["type"], "enabled");
    }

    #[test]
    fn test_deepseek_off() {
        let mut body = serde_json::json!({});
        apply_reasoning_effort(&mut body, Some("off"), "deepseek");
        assert_eq!(body["thinking"]["type"], "disabled");
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn test_deepseek_low() {
        let mut body = serde_json::json!({});
        apply_reasoning_effort(&mut body, Some("low"), "deepseek");
        assert_eq!(body["reasoning_effort"], "high");
        assert_eq!(body["thinking"]["type"], "enabled");
    }

    #[test]
    fn test_no_effort() {
        let mut body = serde_json::json!({});
        apply_reasoning_effort(&mut body, None, "deepseek");
        assert!(body.as_object().unwrap().is_empty());
    }
}