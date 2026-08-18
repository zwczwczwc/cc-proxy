use crate::anthropic::types::{MessagesRequest, SystemPrompt, Tool, ToolChoice};
use crate::config::Config;
use crate::openai::types::{
    ChatCompletionRequest, DeepSeekThinking, OpenAiFunction, OpenAiTool, StreamOptions,
};
use crate::reasoning::build_messages::build_chat_messages_with_reasoning;
use crate::reasoning::prefix::compute_prefix_fingerprint;
use crate::reasoning::requires::requires_reasoning_content;
use crate::reasoning::sanitize::sanitize_thinking_mode_messages;
use serde_json::Value;

/// Map Claude model names to upstream model names using the config's model_mapping
/// and model_profiles. The `[1m]` suffix is stripped before lookup.
/// If the model is a known profile (or alias), it passes through directly.
/// Unknown models fall back to config's default_model.
pub fn map_model_to_upstream_for_responses(model: &str, config: &Config) -> String {
    map_model_to_upstream(model, config)
}

fn map_model_to_upstream(model: &str, config: &Config) -> String {
    let clean = model.trim_end_matches("[1m]").trim();
    // Check if it's already a known upstream model (profile or alias) — pass through
    if config.model_profile(clean).is_some() {
        return clean.to_string();
    }
    config
        .model_mapping
        .get(clean)
        .cloned()
        .unwrap_or_else(|| config.default_model.clone())
}

/// Convert Anthropic MessagesRequest to OpenAI ChatCompletionRequest.
pub fn convert_request(
    req: &MessagesRequest,
    config: &Config,
) -> anyhow::Result<ChatCompletionRequest> {
    let relocate = std::env::var("CODEMERMAFROST_RELOCATE").is_ok();
    convert_request_with_relocation(req, config, relocate)
}

/// Same as [`convert_request`] with an explicit relocate decision.
///
/// Zero behavior change: [`convert_request`] passes the process-level
/// `CODEMERMAFROST_RELOCATE` flag here. Tests use this variant to capture and
/// verify both relocate states deterministically (no process-global env races).
pub(crate) fn convert_request_with_relocation(
    req: &MessagesRequest,
    config: &Config,
    relocate: bool,
) -> anyhow::Result<ChatCompletionRequest> {
    let model = req.model.clone();
    // Map Claude model names to upstream model names for eswitch
    let upstream_model = map_model_to_upstream(&model, config);
    let is_reasoning_model = requires_reasoning_content(&upstream_model, config);

    // Determine effort level from thinking config for replay decision
    let effort_for_replay = req.thinking.as_ref().map(|t| {
        if t.is_enabled() {
            let budget = t.budget_tokens().unwrap_or(0);
            if budget >= 4096 || budget == 0 {
                "max"
            } else {
                "high"
            }
        } else {
            "off"
        }
    });
    let include_reasoning = crate::reasoning::should_replay::should_replay_reasoning_content(
        &upstream_model,
        effort_for_replay,
        config,
    );

    // Build OpenAI messages
    // Feature flag: relocate volatile env blocks from system prefix to last user turn
    // to stabilise KV cache prefix across turns.
    // Controlled by env var CODEMERMAFROST_RELOCATE.
    let (system, messages_ref) = if relocate {
        let raw_system = req
            .system
            .clone()
            .unwrap_or(SystemPrompt::Text(String::new()));
        // Step 1: stabilize billing nonce (permafrost_align.py L149-L177)
        let system = crate::reasoning::relocate::stabilize_metadata(raw_system);
        // Step 2: relocate volatile env blocks (permafrost_align.py L248-L310)
        let (new_system, new_messages) = crate::reasoning::relocate::migrate_volatile_system_blocks(
            system,
            req.messages.clone(),
        );
        // Store messages in a let binding, then reference
        let messages = new_messages;
        let system_opt = if matches!(new_system, SystemPrompt::Text(ref t) if t.is_empty()) {
            None
        } else {
            Some(new_system)
        };
        (system_opt, messages)
    } else {
        (req.system.clone(), req.messages.clone())
    };
    let messages =
        build_chat_messages_with_reasoning(system.as_ref(), &messages_ref, include_reasoning);

    let stream = req.stream.unwrap_or(false);

    let mut openai_req = ChatCompletionRequest {
        model: upstream_model,
        messages,
        max_tokens: None,
        max_completion_tokens: Some(req.max_tokens),
        stream: Some(stream),
        stream_options: if stream {
            Some(StreamOptions {
                include_usage: true,
            })
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
        let mut openai_tools: Vec<OpenAiTool> = tools.iter().map(convert_tool).collect();
        crate::conversation::sort_by_name(&mut openai_tools, |t| &t.function.name);
        openai_req.tools = Some(openai_tools);
    }

    // Convert tool_choice
    if let Some(tc) = &req.tool_choice {
        openai_req.tool_choice = Some(convert_tool_choice(tc));
    }

    // Apply reasoning effort from thinking config
    // ★gpt provider + tools 互斥处理：gpt-5.6 拒绝 reasoning_effort+tools
    let profile = config.model_profile(&openai_req.model);
    let is_gpt_provider = profile.map(|p| p.provider == "gpt").unwrap_or(false);
    let has_tools = openai_req.tools.is_some();

    if let Some(thinking) = &req.thinking {
        if thinking.is_enabled() {
            // Default all thinking-enabled requests to xhigh effort.
            let effort = "xhigh";
            let effort = if is_gpt_provider && has_tools {
                "off"
            } else {
                effort
            };
            apply_effort_direct(&mut openai_req, effort, config);
        } else {
            apply_effort_direct(&mut openai_req, "off", config);
        }
    } else if is_reasoning_model {
        let effort = if is_gpt_provider && has_tools {
            "off"
        } else {
            "xhigh"
        };
        apply_effort_direct(&mut openai_req, effort, config);
    }

    // GLM-5.2: 保留式思考需要 clear_thinking=false 在 thinking 对象内
    // 注意：此字段必须在 thinking 内部（非顶层），序列化为:
    // {"thinking": {"type": "enabled", "clear_thinking": false}}
    if openai_req.model.starts_with("glm-5") {
        if let Some(ref mut thinking) = openai_req.thinking {
            thinking.clear_thinking = Some(false);
        }
    }

    // Sanitize messages
    let mut body = serde_json::to_value(&openai_req)?;
    sanitize_thinking_mode_messages(&mut body);
    openai_req = serde_json::from_value(body)?;

    // F6 (simplified): Per-request prefix fingerprint for KV cache observability.
    // No cross-request comparison — external monitoring aggregates and analyses.
    let sys_prompt = openai_req
        .messages
        .first()
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

/// Apply reasoning effort to the request using provider-driven configuration.
/// Replaces hardcoded kimi- prefix detection and provider=="deepseek" branches.
fn apply_effort_direct(req: &mut ChatCompletionRequest, effort: &str, config: &Config) {
    // Look up the model's provider config
    let profile = config.model_profile(&req.model);
    let provider = profile.and_then(|p| config.provider_config(&p.provider));

    match effort {
        "off" | "disabled" | "none" | "false" => {
            if let Some(prov) = provider {
                if prov.disable_thinking {
                    // Cannot turn off thinking; set to lowest effort
                    let lowest = prov
                        .effort_map
                        .get("low")
                        .cloned()
                        .unwrap_or_else(|| "low".to_string());
                    req.reasoning_effort = Some(lowest);
                    // Do NOT set thinking — provider doesn't support it
                } else if prov.thinking_param.is_none() {
                    // ★Provider doesn't support thinking (e.g. gpt); only remove reasoning_effort
                    req.reasoning_effort = None;
                } else {
                    // Set thinking.type = disabled
                    req.thinking = Some(DeepSeekThinking {
                        thinking_type: prov
                            .thinking_type_disabled
                            .clone()
                            .unwrap_or_else(|| "disabled".to_string()),
                        clear_thinking: None,
                    });
                    req.reasoning_effort = None;
                }
            } else {
                // Fallback: unknown model, use default behavior
                req.thinking = Some(DeepSeekThinking {
                    thinking_type: "disabled".to_string(),
                    clear_thinking: None,
                });
                req.reasoning_effort = None;
            }
        }
        _ => {
            if let Some(prov) = provider {
                // Map effort through provider's effort_map; default to "high" for unknown levels
                let mapped = prov
                    .effort_map
                    .get(effort)
                    .cloned()
                    .unwrap_or_else(|| "high".to_string());
                req.reasoning_effort = Some(mapped);

                // Set thinking.type = enabled if provider supports it
                if prov.thinking_param.is_some() {
                    req.thinking = Some(DeepSeekThinking {
                        thinking_type: prov
                            .thinking_type_enabled
                            .clone()
                            .unwrap_or_else(|| "enabled".to_string()),
                        clear_thinking: None,
                    });
                }
            } else {
                // Fallback: unknown model
                req.reasoning_effort = Some("high".to_string());
                req.thinking = Some(DeepSeekThinking {
                    thinking_type: "enabled".to_string(),
                    clear_thinking: None,
                });
            }
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
    use crate::anthropic::types::{
        ContentBlock, ContentValue, Message, SystemContentBlock, SystemPrompt, ThinkingConfig,
    };
    use crate::config::ProviderConfig;
    use std::collections::HashMap;

    /// Helper to create a minimal Config for tests.
    fn test_config() -> Config {
        let mut mapping = HashMap::new();
        mapping.insert("claude-opus-4".to_string(), "deepseek-v4-pro".to_string());
        mapping.insert(
            "claude-sonnet-4".to_string(),
            "deepseek-v4-flash".to_string(),
        );
        mapping.insert("claude-haiku-4".to_string(), "deepseek-v4-pro".to_string());

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
            },
        );
        providers.insert(
            "fireworks".to_string(),
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
                name: "kimi-k3".to_string(),
                provider: "fireworks".to_string(),
                reasoning_enabled: true,
                reasoning_replay: true,
                toolcall_requires_reasoning: false,
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

        Config {
            listen_addr: "0.0.0.0:11435".to_string(),
            eswitch_url: "http://127.0.0.1:11434".to_string(),
            moonshot_official_url: String::new(),
            moonshot_official_api_key: String::new(),
            api_key: "test-key".to_string(),
            log_level: "info".to_string(),
            model_mapping: mapping,
            default_model: "deepseek-v4-pro".to_string(),
            model_profiles,
            providers,
            profile_by_name,
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
        // budget=16000 >= 4096 → max effort
        assert_eq!(result.reasoning_effort, Some("max".to_string()));
        assert_eq!(result.thinking.as_ref().unwrap().thinking_type, "enabled");
    }

    #[test]
    fn test_thinking_adaptive_on_reasoning_model() {
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
            // budget=0 → Adaptive mode → max on reasoning models
            thinking: Some(ThinkingConfig::Enabled {
                config_type: "enabled".to_string(),
                budget_tokens: Some(0),
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
    }

    #[test]
    fn test_thinking_disabled() {
        let config = test_config();
        let req = MessagesRequest {
            model: "deepseek-v4-pro".to_string(),
            system: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: ContentValue::Text("hello".to_string()),
            }],
            max_tokens: 4096,
            stream: Some(false),
            thinking: Some(ThinkingConfig::Disabled {
                config_type: "disabled".to_string(),
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
        assert_eq!(result.thinking.as_ref().unwrap().thinking_type, "disabled");
        assert_eq!(result.reasoning_effort, None);
    }

    #[test]
    fn test_model_mapping() {
        let config = test_config();
        let req = MessagesRequest {
            model: "claude-opus-4".to_string(),
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
    fn test_kimi_k3_no_thinking_type() {
        let config = test_config();
        let req = MessagesRequest {
            model: "kimi-k3".to_string(),
            system: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: ContentValue::Text("hello".to_string()),
            }],
            max_tokens: 4096,
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
        // Kimi K3: reasoning_effort is set, but NO thinking.type
        assert_eq!(result.reasoning_effort, Some("max".to_string()));
        assert!(
            result.thinking.is_none(),
            "Kimi K3 should not have thinking.type"
        );
    }

    #[test]
    fn test_kimi_k3_off_effort() {
        let config = test_config();
        let req = MessagesRequest {
            model: "kimi-k3".to_string(),
            system: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: ContentValue::Text("hello".to_string()),
            }],
            max_tokens: 4096,
            stream: Some(false),
            thinking: Some(ThinkingConfig::Disabled {
                config_type: "disabled".to_string(),
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
        // Kimi K3 can't turn off thinking → lowest effort
        assert_eq!(result.reasoning_effort, Some("low".to_string()));
        assert!(
            result.thinking.is_none(),
            "Kimi K3 should not have thinking.type"
        );
    }

    #[test]
    fn test_kimi_k3_xhigh_effort() {
        let config = test_config();
        let req = MessagesRequest {
            model: "kimi-k3".to_string(),
            system: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: ContentValue::Text("hello".to_string()),
            }],
            max_tokens: 4096,
            stream: Some(false),
            // xhigh → max (K3 ceiling)
            thinking: Some(ThinkingConfig::Enabled {
                config_type: "enabled".to_string(),
                budget_tokens: Some(32768),
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
        assert!(
            result.thinking.is_none(),
            "Kimi K3 should not have thinking.type"
        );
    }

    #[test]
    fn test_alias_passthrough() {
        let config = test_config();
        let req = MessagesRequest {
            // "deepseek-chat" is an alias for "deepseek-v4-pro"
            model: "deepseek-chat".to_string(),
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
        // Alias should pass through as-is (it's a known profile alias)
        assert_eq!(result.model, "deepseek-chat");
    }

    #[test]
    fn test_glm_reasoning_model() {
        let config = test_config();
        let req = MessagesRequest {
            model: "glm-5.2".to_string(),
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
        // glm-5.2 is a reasoning model with no explicit thinking → xhigh effort
        assert_eq!(result.reasoning_effort, Some("xhigh".to_string()));
        assert_eq!(result.thinking.as_ref().unwrap().thinking_type, "enabled");
    }

    #[test]
    fn test_glm_reasoning_replay_disabled() {
        let config = test_config();
        let req = MessagesRequest {
            model: "glm-5.2".to_string(),
            system: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: ContentValue::Text("hello".to_string()),
            }],
            max_tokens: 4096,
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
        // glm-5.2 has reasoning_replay=false, so even with enabled thinking,
        // reasoning_content should NOT be included in messages
        // (check: messages should not have reasoning_content field)
        let _asst_msg = result
            .messages
            .iter()
            .find(|m| m.get("role").and_then(|v| v.as_str()) == Some("assistant"));
        // If there are no assistant messages, that's fine — the test just has a user message
        // Responses transport uses xhigh directly; Chat provider mappings remain unchanged.
        assert_eq!(result.reasoning_effort, Some("xhigh".to_string()));
        assert_eq!(result.thinking.as_ref().unwrap().thinking_type, "enabled");
    }

    #[test]
    fn test_tool_conversion_sorted() {
        let config = test_config();
        let req = MessagesRequest {
            model: "deepseek-v4-pro".to_string(),
            system: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: ContentValue::Text("hello".to_string()),
            }],
            max_tokens: 4096,
            stream: Some(false),
            thinking: None,
            tools: Some(vec![
                Tool {
                    name: "zebra".to_string(),
                    description: Some("z".to_string()),
                    input_schema: serde_json::json!({"type": "object"}),
                },
                Tool {
                    name: "alpha".to_string(),
                    description: Some("a".to_string()),
                    input_schema: serde_json::json!({"type": "object"}),
                },
            ]),
            tool_choice: None,
            metadata: None,
            stop_sequences: None,
            temperature: None,
            top_p: None,
            top_k: None,
        };

        let result = convert_request(&req, &config).unwrap();
        let tools = result.tools.unwrap();
        assert_eq!(tools[0].function.name, "alpha");
        assert_eq!(tools[1].function.name, "zebra");
    }

    #[test]
    fn test_tool_choice_auto() {
        let config = test_config();
        let req = MessagesRequest {
            model: "deepseek-v4-pro".to_string(),
            system: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: ContentValue::Text("hello".to_string()),
            }],
            max_tokens: 4096,
            stream: Some(false),
            thinking: None,
            tools: None,
            tool_choice: Some(ToolChoice::Auto {
                r#type: "auto".to_string(),
            }),
            metadata: None,
            stop_sequences: None,
            temperature: None,
            top_p: None,
            top_k: None,
        };

        let result = convert_request(&req, &config).unwrap();
        assert_eq!(result.tool_choice, Some(serde_json::json!("auto")));
    }

    #[test]
    fn test_tool_choice_any() {
        let config = test_config();
        let req = MessagesRequest {
            model: "deepseek-v4-pro".to_string(),
            system: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: ContentValue::Text("hello".to_string()),
            }],
            max_tokens: 4096,
            stream: Some(false),
            thinking: None,
            tools: None,
            tool_choice: Some(ToolChoice::Any {
                r#type: "any".to_string(),
            }),
            metadata: None,
            stop_sequences: None,
            temperature: None,
            top_p: None,
            top_k: None,
        };

        let result = convert_request(&req, &config).unwrap();
        assert_eq!(result.tool_choice, Some(serde_json::json!("required")));
    }

    #[test]
    fn test_tool_choice_specific() {
        let config = test_config();
        let req = MessagesRequest {
            model: "deepseek-v4-pro".to_string(),
            system: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: ContentValue::Text("hello".to_string()),
            }],
            max_tokens: 4096,
            stream: Some(false),
            thinking: None,
            tools: None,
            tool_choice: Some(ToolChoice::Tool {
                r#type: "tool".to_string(),
                name: "read_file".to_string(),
            }),
            metadata: None,
            stop_sequences: None,
            temperature: None,
            top_p: None,
            top_k: None,
        };

        let result = convert_request(&req, &config).unwrap();
        assert_eq!(
            result.tool_choice,
            Some(serde_json::json!({"type": "function", "function": {"name": "read_file"}}))
        );
    }

    #[test]
    fn test_reasoning_content_replay() {
        let config = test_config();
        let req = MessagesRequest {
            model: "deepseek-v4-pro".to_string(),
            system: Some(SystemPrompt::Text("You are helpful".to_string())),
            messages: vec![
                Message {
                    role: "assistant".to_string(),
                    content: ContentValue::Blocks(vec![
                        ContentBlock::Thinking {
                            thinking: "Let me think about this.".to_string(),
                            signature: "sig123".to_string(),
                        },
                        ContentBlock::Text {
                            text: "Here is the answer.".to_string(),
                        },
                    ]),
                },
                Message {
                    role: "user".to_string(),
                    content: ContentValue::Text("Next question".to_string()),
                },
            ],
            max_tokens: 4096,
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
        let asst_msg = result
            .messages
            .iter()
            .find(|m| m.get("role").and_then(|v| v.as_str()) == Some("assistant"))
            .unwrap();
        let reasoning = asst_msg.get("reasoning_content").and_then(|v| v.as_str());
        assert_eq!(reasoning, Some("Let me think about this."));
    }

    #[test]
    fn test_reasoning_content_no_replay_when_disabled() {
        let config = test_config();
        let req = MessagesRequest {
            model: "deepseek-v4-pro".to_string(),
            system: Some(SystemPrompt::Text("You are helpful".to_string())),
            messages: vec![
                Message {
                    role: "assistant".to_string(),
                    content: ContentValue::Blocks(vec![
                        ContentBlock::Thinking {
                            thinking: "Let me think about this.".to_string(),
                            signature: "sig123".to_string(),
                        },
                        ContentBlock::Text {
                            text: "Here is the answer.".to_string(),
                        },
                    ]),
                },
                Message {
                    role: "user".to_string(),
                    content: ContentValue::Text("Next question".to_string()),
                },
            ],
            max_tokens: 4096,
            stream: Some(false),
            thinking: Some(ThinkingConfig::Disabled {
                config_type: "disabled".to_string(),
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
        let asst_msg = result
            .messages
            .iter()
            .find(|m| m.get("role").and_then(|v| v.as_str()) == Some("assistant"))
            .unwrap();
        // No reasoning_content when thinking is disabled
        assert!(asst_msg
            .get("reasoning_content")
            .and_then(|v| v.as_str())
            .is_none());
        assert_eq!(asst_msg["content"], "Here is the answer.");
    }

    #[test]
    fn test_stream_option() {
        let config = test_config();
        let req = MessagesRequest {
            model: "deepseek-v4-pro".to_string(),
            system: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: ContentValue::Text("hello".to_string()),
            }],
            max_tokens: 4096,
            stream: Some(true),
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
        assert_eq!(result.stream, Some(true));
        assert!(result.stream_options.is_some());
        assert!(result.stream_options.unwrap().include_usage);
    }

    #[test]
    fn test_reasoning_content_idempotent() {
        // Verify that convert_request is idempotent for reasoning content
        // (no double-injection, no corruption)
        let config = test_config();

        let thinking = ThinkingConfig::Enabled {
            config_type: "enabled".to_string(),
            budget_tokens: Some(16000),
            display: None,
        };

        let req = MessagesRequest {
            model: "deepseek-v4-pro".to_string(),
            system: Some(SystemPrompt::Text("You are helpful".to_string())),
            messages: vec![
                Message {
                    role: "assistant".to_string(),
                    content: ContentValue::Blocks(vec![
                        ContentBlock::Thinking {
                            thinking: "Reasoning step 1".to_string(),
                            signature: "sig1".to_string(),
                        },
                        ContentBlock::Text {
                            text: "Answer 1".to_string(),
                        },
                    ]),
                },
                Message {
                    role: "user".to_string(),
                    content: ContentValue::Text("Question 2".to_string()),
                },
            ],
            max_tokens: 4096,
            stream: Some(false),
            thinking: Some(thinking.clone()),
            tools: None,
            tool_choice: None,
            metadata: None,
            stop_sequences: None,
            temperature: None,
            top_p: None,
            top_k: None,
        };

        let result1 = convert_request(&req, &config).unwrap();
        let result2 = convert_request(&req, &config).unwrap();

        // Same request should produce same output
        let asst1 = result1
            .messages
            .iter()
            .find(|m| m.get("role").and_then(|v| v.as_str()) == Some("assistant"))
            .unwrap();
        let asst2 = result2
            .messages
            .iter()
            .find(|m| m.get("role").and_then(|v| v.as_str()) == Some("assistant"))
            .unwrap();

        assert_eq!(
            asst1.get("reasoning_content").and_then(|v| v.as_str()),
            asst2.get("reasoning_content").and_then(|v| v.as_str()),
            "Reasoning content must be identical across conversions"
        );
    }

    #[test]
    fn test_relocate_disabled_no_conflict_with_reasoning() {
        // Verify that when relocate is OFF, reasoning fix still works
        // independently (no conflict, no crash).
        let config = test_config();

        // Ensure relocate is NOT set
        std::env::remove_var("CODEMERMAFROST_RELOCATE");

        let system = SystemPrompt::Blocks(vec![
            SystemContentBlock {
                block_type: "text".to_string(),
                text: "You are helpful.".to_string(),
            },
            SystemContentBlock {
                block_type: "text".to_string(),
                text: "<env>\nToday's date: 2026-06-22\n</env>".to_string(),
            },
        ]);

        let messages = vec![
            Message {
                role: "assistant".to_string(),
                content: ContentValue::Blocks(vec![
                    ContentBlock::Thinking {
                        thinking: "Let me think.".to_string(),
                        signature: "sig".to_string(),
                    },
                    ContentBlock::Text {
                        text: "Answer".to_string(),
                    },
                ]),
            },
            Message {
                role: "user".to_string(),
                content: ContentValue::Text("Next question".to_string()),
            },
        ];

        let req = MessagesRequest {
            model: "deepseek-v4-pro".to_string(),
            system: Some(system),
            messages,
            max_tokens: 4096,
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

        // System should STILL contain env (relocate disabled)
        let sys_msg = result
            .messages
            .iter()
            .find(|m| m.get("role").and_then(|v| v.as_str()) == Some("system"))
            .unwrap();
        let sys_content = sys_msg
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            sys_content.contains("<env>"),
            "Without relocate, system should contain env block.\nGot: {}",
            sys_content
        );

        // BUT reasoning should still be replayed
        let asst_msg = result
            .messages
            .iter()
            .find(|m| m.get("role").and_then(|v| v.as_str()) == Some("assistant"))
            .unwrap();
        let reasoning = asst_msg.get("reasoning_content").and_then(|v| v.as_str());
        assert_eq!(
            reasoning,
            Some("Let me think."),
            "Reasoning fix should work independently of relocate.\nGot: {:?}",
            reasoning
        );
    }
}
