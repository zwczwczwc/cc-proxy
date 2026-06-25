use std::collections::HashMap;
use serde_json::Value;
use crate::anthropic::types::{MessagesRequest, SystemPrompt, Tool, ToolChoice};
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
    // Check if it's a GLM model — pass through directly
    if clean.starts_with("glm") {
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

    // Determine effort level from thinking config for replay decision
    let effort_for_replay = req.thinking.as_ref().and_then(|t| {
        if t.is_enabled() {
            let budget = t.budget_tokens().unwrap_or(0);
            if budget >= 4096 || budget == 0 {
                Some("max")
            } else {
                Some("high")
            }
        } else {
            Some("off")
        }
    });
    let include_reasoning = crate::reasoning::should_replay::should_replay_reasoning_content(
        &upstream_model,
        effort_for_replay.as_deref(),
    );

    // Build OpenAI messages
    // Feature flag: relocate volatile env blocks from system prefix to last user turn
    // to stabilise KV cache prefix across turns.
    // Controlled by env var CODEMERMAFROST_RELOCATE.
    let (system, messages_ref) = if std::env::var("CODEMERMAFROST_RELOCATE").is_ok() {
        let raw_system = req.system.clone().unwrap_or(SystemPrompt::Text(String::new()));
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
    let messages = build_chat_messages_with_reasoning(
        system.as_ref(),
        &messages_ref,
        include_reasoning,
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
            // For reasoning models, budget=0 (Adaptive mode) defaults to max.
            // Otherwise, budget >= 4096 → max, budget < 4096 → high.
            let effort = if budget >= 4096 || (is_reasoning_model && budget == 0) {
                "max"
            } else {
                "high"
            };
            apply_effort_direct(&mut openai_req, effort);
        } else {
            apply_effort_direct(&mut openai_req, "off");
        }
    } else if is_reasoning_model {
        apply_effort_direct(&mut openai_req, "xhigh");
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
                clear_thinking: None,
            });
            req.reasoning_effort = None;
        }
        "low" | "medium" | "high" => {
            req.reasoning_effort = Some("high".to_string());
            req.thinking = Some(crate::openai::types::DeepSeekThinking {
                thinking_type: "enabled".to_string(),
                clear_thinking: None,
            });
        }
        "max" | "xhigh" => {
            req.reasoning_effort = Some("max".to_string());
            req.thinking = Some(crate::openai::types::DeepSeekThinking {
                thinking_type: "enabled".to_string(),
                clear_thinking: None,
            });
        }
        _ => {
            req.reasoning_effort = Some("high".to_string());
            req.thinking = Some(crate::openai::types::DeepSeekThinking {
                thinking_type: "enabled".to_string(),
                clear_thinking: None,
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
    use crate::anthropic::types::{ContentBlock, ContentValue, Message, SystemContentBlock, SystemPrompt, ThinkingConfig, ToolResultContent};

    /// Helper to create a minimal Config for tests.
    fn test_config() -> Config {
        let mut mapping = HashMap::new();
        mapping.insert("claude-opus-4".to_string(), "deepseek-v4-pro".to_string());
        mapping.insert("claude-sonnet-4".to_string(), "deepseek-v4-flash".to_string());
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
        // Adaptive thinking on reasoning models now defaults to max (not high).
        assert_eq!(result.reasoning_effort, Some("max".to_string()));
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
        assert_eq!(result.model, "deepseek-v4-flash");
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

    #[test]
    fn test_converter_include_reasoning_effort_max() {
        // thinking=enabled, budget>=4096 → include_reasoning=true
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
        // Verify max thinking is still applied
        assert_eq!(result.reasoning_effort, Some("max".to_string()));
        assert!(result.thinking.is_some());
        // No assistant messages in single-turn, so reasoning_content replay not visible
        // But the code path is exercised (include_reasoning should be true)
    }

    #[test]
    fn test_converter_include_reasoning_effort_off() {
        // thinking=disabled → include_reasoning=false
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
        // Verify thinking is disabled
        assert_eq!(result.reasoning_effort, None);
        assert!(result.thinking.is_some());
        // Effort=off → include_reasoning=false (no replay), code path exercised
    }

    // =========================================================================
    // Integration tests: both fixes working together
    // =========================================================================

    #[test]
    fn test_relocate_and_reasoning_together_multiturn() {
        // Verify both fixes work together:
        // 1. Relocate moves volatile system blocks to last user turn
        // 2. Reasoning content is deterministically replayed for all assistant messages
        let config = test_config();

        // Set the relocate flag for this test
        std::env::set_var("CODEMERMAFROST_RELOCATE", "1");
        // Ensure cleanup
        let _guard = scopeguard::guard((), |_| {
            std::env::remove_var("CODEMERMAFROST_RELOCATE");
        });

        let system = SystemPrompt::Blocks(vec![
            SystemContentBlock {
                block_type: "text".to_string(),
                text: "You are a helpful assistant.".to_string(),
            },
            SystemContentBlock {
                block_type: "text".to_string(),
                text: "<env>\nWorking directory: /tmp\nToday's date: 2026-06-22\nPlatform: linux\n</env>"
                    .to_string(),
            },
        ]);

        let messages = vec![
            // Historical assistant with thinking → should get reasoning_content replayed
            Message {
                role: "assistant".to_string(),
                content: ContentValue::Blocks(vec![
                    ContentBlock::Thinking {
                        thinking: "Let me check the file.".to_string(),
                        signature: "sig001".to_string(),
                    },
                    ContentBlock::ToolUse {
                        id: "toolu_001".to_string(),
                        name: "read_file".to_string(),
                        input: serde_json::json!({"path": "/tmp/test"}),
                    },
                ]),
            },
            // Tool result
            Message {
                role: "user".to_string(),
                content: ContentValue::Blocks(vec![
                    ContentBlock::ToolResult {
                        tool_use_id: "toolu_001".to_string(),
                        content: ToolResultContent::Text("file contents".to_string()),
                        is_error: Some(false),
                    },
                ]),
            },
            // Current user turn
            Message {
                role: "user".to_string(),
                content: ContentValue::Text("What does the file say?".to_string()),
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

        // Verify the request was built successfully
        assert!(!result.messages.is_empty(), "Messages should not be empty");

        // Find the system message — it should NOT contain the env block
        let sys_msg = result.messages.iter().find(|m| {
            m.get("role").and_then(|v| v.as_str()) == Some("system")
        });
        assert!(sys_msg.is_some(), "System message should exist");
        let sys_content = sys_msg.unwrap().get("content").and_then(|v| v.as_str()).unwrap_or("");
        // System should be stable (no volatile env block)
        assert!(
            !sys_content.contains("<env>") && !sys_content.contains("Working directory"),
            "System prompt should NOT contain volatile env block after relocate.\nSystem content: {}",
            sys_content
        );
        assert!(
            sys_content.contains("You are a helpful assistant"),
            "System should retain stable instructions.\nSystem content: {}",
            sys_content
        );

        // Find the last user message — it should contain the relocated env block
        let user_msgs: Vec<_> = result.messages.iter()
            .filter(|m| m.get("role").and_then(|v| v.as_str()) == Some("user"))
            .collect();
        assert!(!user_msgs.is_empty(), "Should have user messages");
        let last_user_content = user_msgs.last().unwrap()
            .get("content").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            last_user_content.contains("permafrost:relocated-context"),
            "Last user message should contain relocated env block.\nContent: {}",
            last_user_content
        );
        assert!(
            last_user_content.contains("Working directory"),
            "Relocated content should contain env details.\nContent: {}",
            last_user_content
        );

        // Find the assistant message — it should have reasoning_content
        let asst_msg = result.messages.iter().find(|m| {
            m.get("role").and_then(|v| v.as_str()) == Some("assistant")
        });
        assert!(asst_msg.is_some(), "Assistant message should exist");
        let reasoning = asst_msg.unwrap()
            .get("reasoning_content")
            .and_then(|v| v.as_str());
        assert_eq!(
            reasoning,
            Some("Let me check the file."),
            "Reasoning content should be deterministically replayed.\nGot: {:?}",
            reasoning
        );
    }

    #[test]
        fn test_relocate_and_reasoning_deterministic_byte_stability() {
            // Verify that the same input always produces the same byte-level
            // message structure (KV cache prefix stability).
            // NOTE: This test does NOT set CODEMERMAFROST_RELOCATE to avoid
            // env-var race conditions in parallel test execution.
            // The relocate behavior is covered by other dedicated tests.
            let config = test_config();

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
                            thinking: "I will help.".to_string(),
                            signature: "sig".to_string(),
                        },
                        ContentBlock::Text {
                            text: "Hello!".to_string(),
                        },
                    ]),
                },
                Message {
                    role: "user".to_string(),
                    content: ContentValue::Text("Thanks!".to_string()),
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

            // Convert twice — byte representation must be identical
            let result1 = convert_request(&req, &config).unwrap();
            let result2 = convert_request(&req, &config).unwrap();

            let json1 = serde_json::to_string(&result1).unwrap();
            let json2 = serde_json::to_string(&result2).unwrap();

            assert_eq!(
                json1, json2,
                "Same input must produce identical byte-level output for KV cache stability.\nDiff length: {} vs {}",
                json1.len(), json2.len()
            );

            // Also verify reasoning_content is present and deterministic
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
        let sys_msg = result.messages.iter().find(|m| {
            m.get("role").and_then(|v| v.as_str()) == Some("system")
        }).unwrap();
        let sys_content = sys_msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            sys_content.contains("<env>"),
            "Without relocate, system should contain env block.\nGot: {}",
            sys_content
        );

        // BUT reasoning should still be replayed
        let asst_msg = result.messages.iter().find(|m| {
            m.get("role").and_then(|v| v.as_str()) == Some("assistant")
        }).unwrap();
        let reasoning = asst_msg.get("reasoning_content").and_then(|v| v.as_str());
        assert_eq!(
            reasoning,
            Some("Let me think."),
            "Reasoning fix should work independently of relocate.\nGot: {:?}",
            reasoning
        );
    }
}
