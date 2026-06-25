/// Determines if a model requires reasoning_content in API messages.
/// Translated from CodeWhale chat.rs L1897-L1913.
pub fn requires_reasoning_content(model: &str) -> bool {
    let model_lower = model.to_lowercase();

    // DeepSeek V4 series
    if model_lower.starts_with("deepseek-v4") {
        return true;
    }

    // GLM-5 series (glm-5.1, glm-5.2) — reasoning models
    if model_lower.starts_with("glm-5") {
        return true;
    }

    // DeepSeek chat/reasoner aliases
    if model_lower == "deepseek-chat" || model_lower == "deepseek-reasoner" {
        return true;
    }

    // Generic markers
    if model_lower.contains("reasoner")
        || model_lower.contains("-reasoning")
        || model_lower.contains("-thinking")
    {
        return true;
    }

    // DeepSeek R-series markers (deepseek-r1, deepseek-r2, etc.)
    if has_deepseek_r_series_marker(&model_lower) {
        return true;
    }

    false
}

fn has_deepseek_r_series_marker(model: &str) -> bool {
    let model_lower = model.to_lowercase();
    if !model_lower.starts_with("deepseek") {
        return false;
    }
    // Check for "deepseek-r" followed by a digit
    if let Some(rest) = model_lower.strip_prefix("deepseek") {
        let rest = rest.trim_start_matches('-');
        if rest.starts_with('r') {
            let after_r = &rest[1..];
            if after_r.starts_with(|c: char| c.is_ascii_digit()) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deepseek_v4_models() {
        assert!(requires_reasoning_content("deepseek-v4-pro"));
        assert!(requires_reasoning_content("deepseek-v4-flash"));
        assert!(requires_reasoning_content("deepseek-v4"));
    }

    #[test]
    fn test_deepseek_r_series() {
        assert!(requires_reasoning_content("deepseek-r1"));
        assert!(requires_reasoning_content("deepseek-r2"));
    }

    #[test]
    fn test_non_reasoning_models() {
        assert!(!requires_reasoning_content("gpt-4"));
        assert!(!requires_reasoning_content("claude-sonnet-4"));
        assert!(!requires_reasoning_content("llama-3"));
    }

    #[test]
    fn test_reasoner_markers() {
        assert!(requires_reasoning_content("some-reasoner-model"));
        assert!(requires_reasoning_content("model-with-reasoning"));
        assert!(requires_reasoning_content("model-with-thinking"));
    }
}