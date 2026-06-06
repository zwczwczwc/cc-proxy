use serde_json::Value;

/// Sanitizes thinking mode messages: ensures that assistant messages with tool_calls
/// always have a reasoning_content field (even if placeholder).
/// Translated from CodeWhale chat.rs L1768-L1820.
pub fn sanitize_thinking_mode_messages(body: &mut Value) {
    let messages = match body.get_mut("messages") {
        Some(Value::Array(arr)) => arr,
        _ => return,
    };

    for msg in messages.iter_mut() {
        let role = match msg.get("role").and_then(|v| v.as_str()) {
            Some(r) => r,
            None => continue,
        };

        if role != "assistant" {
            continue;
        }

        let has_tool_calls = msg.get("tool_calls").map_or(false, |v| !v.is_null());

        if !has_tool_calls {
            continue;
        }

        let has_reasoning = msg
            .get("reasoning_content")
            .map_or(false, |v| {
                if v.is_null() {
                    return false;
                }
                if let Some(s) = v.as_str() {
                    return !s.trim().is_empty();
                }
                // If it's some other non-null value, treat as having reasoning
                true
            });

        if !has_reasoning {
            msg["reasoning_content"] = serde_json::Value::String("(reasoning omitted)".to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_placeholder() {
        let mut body = serde_json::json!({
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]}
            ]
        });
        sanitize_thinking_mode_messages(&mut body);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[1]["reasoning_content"], "(reasoning omitted)");
    }

    #[test]
    fn test_sanitize_empty_string() {
        let mut body = serde_json::json!({
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}], "reasoning_content": ""}
            ]
        });
        sanitize_thinking_mode_messages(&mut body);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["reasoning_content"], "(reasoning omitted)");
    }

    #[test]
    fn test_sanitize_no_change() {
        let mut body = serde_json::json!({
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}], "reasoning_content": "some reasoning"}
            ]
        });
        sanitize_thinking_mode_messages(&mut body);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["reasoning_content"], "some reasoning");
    }

    #[test]
    fn test_sanitize_null_reasoning() {
        let mut body = serde_json::json!({
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}], "reasoning_content": null}
            ]
        });
        sanitize_thinking_mode_messages(&mut body);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["reasoning_content"], "(reasoning omitted)");
    }
}