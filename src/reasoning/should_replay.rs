use super::requires::requires_reasoning_content;

/// Determines whether reasoning content should be replayed to the API.
/// Translated from CodeWhale chat.rs L1915-L1929.
pub fn should_replay_reasoning_content(model: &str, effort: Option<&str>) -> bool {
    // If effort is explicitly "off" / "disabled" / "none" / "false", don't replay
    if let Some(eff) = effort {
        let eff_lower = eff.to_lowercase();
        if eff_lower == "off" || eff_lower == "disabled" || eff_lower == "none" || eff_lower == "false" {
            return false;
        }
    }
    requires_reasoning_content(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_for_deepseek() {
        assert!(should_replay_reasoning_content("deepseek-v4-pro", None));
        assert!(should_replay_reasoning_content("deepseek-v4-pro", Some("high")));
    }

    #[test]
    fn test_no_replay_when_off() {
        assert!(!should_replay_reasoning_content("deepseek-v4-pro", Some("off")));
        assert!(!should_replay_reasoning_content("deepseek-v4-pro", Some("disabled")));
        assert!(!should_replay_reasoning_content("deepseek-v4-pro", Some("none")));
        assert!(!should_replay_reasoning_content("deepseek-v4-pro", Some("false")));
    }

    #[test]
    fn test_no_replay_for_non_ds() {
        assert!(!should_replay_reasoning_content("gpt-4", None));
    }
}