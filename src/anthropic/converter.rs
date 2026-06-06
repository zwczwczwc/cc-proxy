use serde_json::Value;
use crate::anthropic::types::{MessagesRequest, Tool, ToolChoice};
use crate::openai::types::{ChatCompletionRequest, OpenAiTool, OpenAiFunction, StreamOptions};
use crate::reasoning::sanitize::sanitize_thinking_mode_messages;
use crate::reasoning::build_messages::build_chat_messages_with_reasoning;
use crate::reasoning::requires::requires_reasoning_content;

/// Map Claude model names to upstream DeepSeek model names.
/// The `[1m]` suffix is stripped before lookup. Unknown models default to `deepseek-v4-pro`.
fn map_model_to_upstream(model: &str) -> String {
    // Strip [1m] suffix if present
    let clean = model.trim_end_matches("[1m]").trim();
    match clean {
        "claude-opus-4-7" | "claude-opus-4-6" | "claude-opus-4-5" | "claude-opus-4" => {
            "deepseek-v4-pro".to_string()
        }
        "claude-sonnet-4-6" | "claude-sonnet-4-5" | "claude-sonnet-4" | "claude-3-5-sonnet" => {
            "deepseek-v4-flash".to_string()
        }
        "claude-haiku-4-5" | "claude-3-haiku" | "claude-haiku-4" => {
            "qwen3.6-inner-free".to_string()
        }
        // Already a DeepSeek model, pass through
        m if m.starts_with("deepseek") => m.to_string(),
        // Unknown, default to v4-pro
        _ => "deepseek-v4-pro".to_string(),
    }
}

/// Convert Anthropic MessagesRequest to OpenAI ChatCompletionRequest.
pub fn convert_request(req: &MessagesRequest) -> anyhow::Result<ChatCompletionRequest> {
    let model = req.model.clone();
    let is_reasoning_model = requires_reasoning_content(&model);

    // Map Claude model names to DeepSeek model names for eswitch
    let upstream_model = map_model_to_upstream(&model);

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
        temperature: req.temperature,
        top_p: req.top_p,
        reasoning_effort: None,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop: req.stop_sequences.clone(),
    };

    // Convert tools
    if let Some(tools) = &req.tools {
        let openai_tools: Vec<OpenAiTool> = tools
            .iter()
            .map(|t| convert_tool(t))
            .collect();
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

    #[test]
    fn test_basic_conversion() {
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

        let result = convert_request(&req).unwrap();
        assert_eq!(result.model, "deepseek-v4-pro");
        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.messages[0]["role"], "system");
        assert_eq!(result.messages[1]["role"], "user");
    }

    #[test]
    fn test_thinking_enabled() {
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

        let result = convert_request(&req).unwrap();
        assert_eq!(result.reasoning_effort, Some("max".to_string()));
        assert!(result.thinking.is_some());
    }

    #[test]
    fn test_thinking_adaptive() {
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

        let result = convert_request(&req).unwrap();
        assert_eq!(result.reasoning_effort, Some("high".to_string()));
        assert!(result.thinking.is_some());
    }
}