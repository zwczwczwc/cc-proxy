use std::collections::HashMap;
use serde_json::Value;
use crate::anthropic::types::{MessagesRequest, Tool, ToolChoice};
use crate::config::Config;
use crate::openai::types::{ChatCompletionRequest, OpenAiTool, OpenAiFunction, StreamOptions};
use crate::reasoning::sanitize::sanitize_thinking_mode_messages;
use crate::reasoning::build_messages::build_chat_messages_with_reasoning;
use crate::reasoning::requires::requires_reasoning_content;
use crate::reasoning::prefix::compute_prefix_fingerprint;

/// Map Claude model names to upstream DeepSeek model names using the config's model_mapping.
/// The `[1m]` suffix is stripped before lookup. Unknown models fall back to config's default_model.
fn map_model_to_upstream(model: &str, mapping: &HashMap<String, String>, default: &str) -> String {
    let clean = model.trim_end_matches("[1m]").trim();
    // Check if it's already a DeepSeek model — pass through
    if clean.starts_with("deepseek") {
        return clean.to_string();
    }
    mapping.get(clean).cloned().unwrap_or_else(|| default.to_string())
}

/// Convert Anthropic MessagesRequest to OpenAI ChatCompletionRequest.
pub fn convert_request(req: &MessagesRequest, config: &Config) -> anyhow::Result<ChatCompletionRequest> {
    let model = req.model.clone();
    // Map Claude model names to DeepSeek model names for eswitch
    let upstream_model = map_model_to_upstream(&model, &config.model_mapping, &config.default_model);
    let is_reasoning_model = requires_reasoning_content(&upstream_model);

    // Build OpenAI messages
    let messages = build_chat_messages_with_reasoning(
        req.system.as_ref(),
        &req.messages,
        is_reasoning_model,
    );

    let stream = req.stream.unwrap_or(false);

    let mut openai_req = ChatCompletionRequest {
        model: upstream_model,
        messages,
        max_tokens: None,
        max_completion_tokens: Some(req.max_tokens),
        stream: Some(stream),
        stream_options: if stream {
            Some(StreamOptions { include_usage: true })
        } else {
            None
        },
        temperature: None,
        top_p: None,
        reasoning_effort: None,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop: None,
    };

    // Convert tools (sorted by name for KV cache prefix stability)
    if let Some(tools) = &req.tools {
        let mut openai_tools: Vec<OpenAiTool> = tools
            .iter()
            .map(|t| convert_tool(t))
            .collect();
        openai_tools.sort_by(|a, b| a.function.name.cmp(&b.function.name));
        openai_req.tools = Some(openai_tools);
    }

    // Convert tool_choice
    if let Some(tc) = &req.tool_choice {
        openai_req.tool_choice = Some(convert_tool_choice(tc));
    }

    // Apply reasoning effort from thinking config
    if let Some(thinking) = &req.thinking {
        if thinking.is_enabled() {
            let budget = thinking.budget_tokens().unwrap_or(0);
            let effort = if budget >= 4096 { "max" } else { "high" };
            apply_effort_direct(&mut openai_req, effort);
        } else {
            apply_effort_direct(&mut openai_req, "off");
        }
    } else if is_reasoning_model {
        apply_effort_direct(&mut openai_req, "high");
    }

    // Sanitize messages
    let mut body = serde_json::to_value(&openai_req)?;
    sanitize_thinking_mode_messages(&mut body);
    openai_req = serde_json::from_value(body)?;

    // F6 (simplified): Per-request prefix fingerprint for KV cache observability.
    // No cross-request comparison — external monitoring aggregates and analyses.
    let sys_prompt = openai_req.messages.first()
        .and_then(|m| m.get("content").and_then(|v| v.as_str()))
        .unwrap_or("");
    let fingerprint = compute_prefix_fingerprint(sys_prompt, openai_req.tools.as_deref());
    tracing::info!(
        prefix_fingerprint = %fingerprint,
        model = %openai_req.model,
        msg_count = openai_req.messages.len(),
        reasoning_effort = ?openai_req.reasoning_effort,
        "OpenAI request built"
    );

    Ok(openai_req)
}

fn apply_effort_direct(req: &mut ChatCompletionRequest, effort: &str) {
    match effort {
        "off" | "disabled" | "none" | "false" => {
            req.thinking = Some(crate::openai::types::DeepSeekThinking {
                thinking_type: "disabled".to_string(),
            });
            req.reasoning_effort = None;
        }
        "low" | "medium" | "high" => {
            req.reasoning_effort = Some("high".to_string());
            req.thinking = Some(crate::openai::types::DeepSeekThinking {
                thinking_type: "enabled".to_string(),
            });
        }
        "max" | "xhigh" => {
            req.reasoning_effort = Some("max".to_string());
            req.thinking = Some(crate::openai::types::DeepSeekThinking {
                thinking_type: "enabled".to_string(),
            });
        }
        _ => {
            req.reasoning_effort = Some("high".to_string());
            req.thinking = Some(crate::openai::types::DeepSeekThinking {
                thinking_type: "enabled".to_string(),
            });
        }
    }
}

fn convert_tool(tool: &Tool) -> OpenAiTool {
    OpenAiTool {
        tool_type: "function".to_string(),
        function: OpenAiFunction {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.input_schema.clone(),
        },
    }
}

fn convert_tool_choice(tc: &ToolChoice) -> Value {
    match tc {
        ToolChoice::Auto { .. } => serde_json::json!("auto"),
        ToolChoice::Any { .. } => serde_json::json!("required"),
        ToolChoice::Tool { name, .. } => {
            serde_json::json!({"type": "function", "function": {"name": name}})
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::types::{ContentValue, Message, SystemPrompt, ThinkingConfig};

    /// Helper to create a minimal Config for tests.
    fn test_config() -> Config {
        let mut mapping = HashMap::new();
        mapping.insert("claude-opus-4".to_string(), "deepseek-v4-pro".to_string());
        mapping.insert("claude-sonnet-4".to_string(), "deepseek-v4-pro".to_string());
        mapping.insert("claude-haiku-4".to_string(), "deepseek-v4-pro".to_string());
        Config {
            listen_addr: "0.0.0.0:11435".to_string(),
            eswitch_url: "http://127.0.0.1:11434".to_string(),
            api_key: "test-key".to_string(),
            log_level: "info".to_string(),
            model_mapping: mapping,
            default_model: "deepseek-v4-pro".to_string(),
        }
    }

    #[test]
    fn test_basic_conversion() {
        let config = test_config();
        let req = MessagesRequest {
            model: "deepseek-v4-pro".to_string(),
            system: Some(SystemPrompt::Text("You are helpful".to_string())),
            messages: vec![Message {
                role: "user".to_string(),
                content: ContentValue::Text("hello".to_string()),
            }],
            max_tokens: 4096,
            stream: Some(false),
            thinking: None,
            tools: None,
            tool_choice: None,
            metadata: None,
            stop_sequences: None,
            temperature: None,
            top_p: None,
            top_k: None,
        };

        let result = convert_request(&req, &config).unwrap();
        assert_eq!(result.model, "deepseek-v4-pro");
        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.messages[0]["role"], "system");
        assert_eq!(result.messages[1]["role"], "user");
    }

    #[test]
    fn test_thinking_enabled() {
        let config = test_config();
        let req = MessagesRequest {
            model: "deepseek-v4-pro".to_string(),
            system: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: ContentValue::Text("hello".to_string()),
            }],
            max_tokens: 32768,
            stream: Some(false),
            thinking: Some(ThinkingConfig::Enabled {
                config_type: "enabled".to_string(),
                budget_tokens: Some(16000),
                display: None,
            }),
            tools: None,
            tool_choice: None,
            metadata: None,
            stop_sequences: None,
            temperature: None,
            top_p: None,
            top_k: None,
        };

        let result = convert_request(&req, &config).unwrap();
        assert_eq!(result.reasoning_effort, Some("max".to_string()));
        assert!(result.thinking.is_some());
    }

    #[test]
    fn test_thinking_adaptive() {
        let config = test_config();
        let req = MessagesRequest {
            model: "deepseek-v4-pro".to_string(),
            system: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: ContentValue::Text("hello".to_string()),
            }],
            max_tokens: 32768,
            stream: Some(false),
            thinking: Some(ThinkingConfig::Adaptive {
                config_type: "enabled".to_string(),
                display: Some("omitted".to_string()),
            }),
            tools: None,
            tool_choice: None,
            metadata: None,
            stop_sequences: None,
            temperature: None,
            top_p: None,
            top_k: None,
        };

        let result = convert_request(&req, &config).unwrap();
        assert_eq!(result.reasoning_effort, Some("high".to_string()));
        assert!(result.thinking.is_some());
    }

    #[test]
    fn test_model_mapping_claude_to_deepseek() {
        let config = test_config();
        let req = MessagesRequest {
            model: "claude-sonnet-4".to_string(),
            system: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: ContentValue::Text("hello".to_string()),
            }],
            max_tokens: 4096,
            stream: Some(false),
            thinking: None,
            tools: None,
            tool_choice: None,
            metadata: None,
            stop_sequences: None,
            temperature: None,
            top_p: None,
            top_k: None,
        };

        let result = convert_request(&req, &config).unwrap();
        assert_eq!(result.model, "deepseek-v4-pro");
    }

    #[test]
    fn test_model_mapping_unknown_falls_back_to_default() {
        let config = test_config();
        let req = MessagesRequest {
            model: "some-unknown-model".to_string(),
            system: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: ContentValue::Text("hello".to_string()),
            }],
            max_tokens: 4096,
            stream: Some(false),
            thinking: None,
            tools: None,
            tool_choice: None,
            metadata: None,
            stop_sequences: None,
            temperature: None,
            top_p: None,
            top_k: None,
        };

        let result = convert_request(&req, &config).unwrap();
        assert_eq!(result.model, "deepseek-v4-pro");
    }

    #[test]
    fn test_model_mapping_deepseek_passthrough() {
        let config = test_config();
        let req = MessagesRequest {
            model: "deepseek-v3".to_string(),
            system: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: ContentValue::Text("hello".to_string()),
            }],
            max_tokens: 4096,
            stream: Some(false),
            thinking: None,
            tools: None,
            tool_choice: None,
            metadata: None,
            stop_sequences: None,
            temperature: None,
            top_p: None,
            top_k: None,
        };

        let result = convert_request(&req, &config).unwrap();
        assert_eq!(result.model, "deepseek-v3");
    }

    #[test]
    fn test_map_model_strips_1m_suffix() {
        let config = test_config();
        let req = MessagesRequest {
            model: "deepseek-v4-pro[1m]".to_string(),
            system: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: ContentValue::Text("hello".to_string()),
            }],
            max_tokens: 4096,
            stream: Some(false),
            thinking: None,
            tools: None,
            tool_choice: None,
            metadata: None,
            stop_sequences: None,
            temperature: None,
            top_p: None,
            top_k: None,
        };

        let result = convert_request(&req, &config).unwrap();
        assert_eq!(result.model, "deepseek-v4-pro");
    }
}
