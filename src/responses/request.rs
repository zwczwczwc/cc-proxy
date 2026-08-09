use super::types::{ReasoningConfig, ResponsesRequest, ResponsesTool};
use crate::anthropic::types::{
    ContentBlock, ContentValue, Message, MessagesRequest, SystemPrompt, Tool, ToolChoice,
};
use crate::config::Config;
use serde_json::Value;
use sha2::{Digest, Sha256};
use uuid::Uuid;

pub fn convert_request(req: &MessagesRequest, config: &Config) -> anyhow::Result<ResponsesRequest> {
    convert_request_with_relocation(
        req,
        config,
        std::env::var("CODEMERMAFROST_RELOCATE").is_ok(),
    )
}

fn convert_request_with_relocation(
    req: &MessagesRequest,
    config: &Config,
    relocate: bool,
) -> anyhow::Result<ResponsesRequest> {
    let model =
        crate::anthropic::converter::map_model_to_upstream_for_responses(&req.model, config);
    if req.max_tokens < 16 {
        anyhow::bail!("max_tokens must be at least 16 for Responses API");
    }
    let (system, messages, volatile_texts) = if relocate {
        let system = crate::reasoning::relocate::stabilize_metadata(
            req.system
                .clone()
                .unwrap_or(SystemPrompt::Text(String::new())),
        );
        let (system, volatile_texts) =
            crate::reasoning::relocate::split_volatile_system_blocks(system);
        (system, req.messages.clone(), volatile_texts)
    } else {
        (
            req.system
                .clone()
                .unwrap_or(SystemPrompt::Text(String::new())),
            req.messages.clone(),
            Vec::new(),
        )
    };
    let instructions = system_text(Some(&system));
    let mut input = Vec::new();
    for message in &messages {
        append_message(&mut input, message)?;
    }
    append_synthetic_context_tail(&mut input, &volatile_texts);
    let tools = req.tools.as_ref().map(|items| {
        let mut converted: Vec<_> = items.iter().map(convert_tool).collect();
        converted.sort_by(|a, b| a.name.cmp(&b.name));
        converted
    });
    let tool_choice = req.tool_choice.as_ref().map(convert_tool_choice);
    let reasoning = reasoning_config(req, config, &model);
    let static_prefix_hash = canonical_hash(&serde_json::json!({
        "model": model,
        "instructions": instructions,
        "tools": tools,
        "tool_choice": tool_choice,
        "reasoning": reasoning.as_ref().map(|value| {
            serde_json::json!({"effort": value.effort})
        }),
    }));
    let history_item_count = input.len() - usize::from(!volatile_texts.is_empty());
    let history_prefix_hash = canonical_hash(&Value::Array(input[..history_item_count].to_vec()));
    let wire_input_hash = canonical_hash(&Value::Array(input.clone()));
    let input_item_types = input
        .iter()
        .map(|item| {
            item.get("type")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| {
                    item.get("role")
                        .and_then(Value::as_str)
                        .map(|role| format!("role:{role}"))
                        .unwrap_or_else(|| "unknown".to_string())
                })
        })
        .collect::<Vec<_>>();
    tracing::info!(
        static_prefix_hash = %static_prefix_hash,
        history_prefix_hash = %history_prefix_hash,
        wire_input_hash = %wire_input_hash,
        input_item_count = input.len(),
        history_item_count,
        synthetic_tail_present = !volatile_texts.is_empty(),
        input_item_types = ?input_item_types,
        model = %model,
        "Responses request built"
    );
    Ok(ResponsesRequest {
        model,
        request_id: Uuid::new_v4().simple().to_string(),
        instructions,
        input,
        tools,
        tool_choice,
        parallel_tool_calls: None,
        reasoning,
        max_output_tokens: req.max_tokens,
        stream: req.stream.unwrap_or(false),
        static_prefix_hash,
        history_prefix_hash,
        wire_input_hash,
        input_item_types,
        synthetic_tail_present: !volatile_texts.is_empty(),
    })
}

fn canonical_hash(value: &Value) -> String {
    let mut hasher = Sha256::new();
    hasher.update(canonical_json(value));
    hex::encode(&hasher.finalize()[..8])
}

fn canonical_json(value: &Value) -> Vec<u8> {
    serde_json::to_vec(&canonicalize(value)).expect("JSON serialization cannot fail")
}

fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            let sorted = object
                .iter()
                .map(|(key, value)| (key.clone(), canonicalize(value)))
                .collect::<std::collections::BTreeMap<_, _>>();
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(canonicalize).collect()),
        _ => value.clone(),
    }
}

fn append_synthetic_context_tail(input: &mut Vec<Value>, volatile_texts: &[String]) {
    if volatile_texts.is_empty() {
        return;
    }
    let mut text = String::from(
        "\n\n<permafrost:relocated-context>\nMoved out of the cache prefix so it can change without resetting the cache. Same meaning, later position.\n</permafrost:relocated-context>\n\n",
    );
    for value in volatile_texts {
        text.push_str(value);
        text.push('\n');
    }
    input.push(serde_json::json!({
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }));
}

fn reasoning_config(
    req: &MessagesRequest,
    config: &Config,
    model: &str,
) -> Option<ReasoningConfig> {
    let enabled = req
        .thinking
        .as_ref()
        .is_some_and(|thinking| thinking.is_enabled());
    let supported = config
        .model_profile(model)
        .is_some_and(|profile| profile.reasoning_enabled);
    let effort = config
        .model_profile(model)
        .and_then(|profile| config.provider_config(&profile.provider))
        .and_then(|provider| provider.effort_map.get("max"))
        .cloned()
        .unwrap_or_else(|| "max".to_string());
    let summary = config
        .model_profile(model)
        .and_then(|profile| config.provider_config(&profile.provider))
        .and_then(|provider| provider.responses_reasoning_summary.as_deref())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| "auto".to_string());
    let summary = match summary.as_str() {
        "off" => None,
        "detailed" => Some("detailed".to_string()),
        _ => Some("auto".to_string()),
    };
    (enabled && supported).then_some(ReasoningConfig { effort, summary })
}

fn system_text(system: Option<&SystemPrompt>) -> Option<String> {
    let text = match system {
        Some(SystemPrompt::Text(text)) => text.clone(),
        Some(SystemPrompt::Blocks(blocks)) => blocks
            .iter()
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        None => String::new(),
    };
    (!text.is_empty()).then_some(text)
}

fn append_message(input: &mut Vec<Value>, message: &Message) -> anyhow::Result<()> {
    match &message.content {
        ContentValue::Text(text) => input.push(serde_json::json!({
            "role": message.role,
            "content": [{"type": text_content_type(&message.role), "text": text}],
        })),
        ContentValue::Null => {}
        ContentValue::Blocks(blocks) => {
            let mut content = Vec::new();
            for block in blocks {
                match block {
                    ContentBlock::Text { text } => content.push(serde_json::json!({
                        "type": text_content_type(&message.role),
                        "text": text,
                    })),
                    ContentBlock::Image { source } => content.push(serde_json::json!({"type":"input_image", "image_url": format!("data:{};base64,{}", source.media_type, source.data)})),
                    ContentBlock::ToolUse {
                        id,
                        name,
                        input: arguments,
                    } => {
                        if !content.is_empty() {
                            input.push(serde_json::json!({"role": message.role, "content": std::mem::take(&mut content)}));
                        }
                        input.push(serde_json::json!({"type":"function_call", "call_id": id, "name": name, "arguments": serde_json::to_string(arguments)?}));
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content: result,
                        ..
                    } => {
                        if !content.is_empty() {
                            input.push(serde_json::json!({"role": message.role, "content": std::mem::take(&mut content)}));
                        }
                        input.push(serde_json::json!({"type":"function_call_output", "call_id": tool_use_id, "output": tool_result_text(result)}));
                    }
                    ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } | ContentBlock::Unknown => {}
                }
            }
            if !content.is_empty() {
                input.push(serde_json::json!({"role": message.role, "content": content}));
            }
        }
    }
    Ok(())
}

fn text_content_type(role: &str) -> &'static str {
    if role == "assistant" {
        "output_text"
    } else {
        "input_text"
    }
}

fn tool_result_text(content: &crate::anthropic::types::ToolResultContent) -> String {
    match content {
        crate::anthropic::types::ToolResultContent::Text(text) => text.clone(),
        crate::anthropic::types::ToolResultContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| match b {
                crate::anthropic::types::ToolResultContentBlock::Text { text } => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn convert_tool(tool: &Tool) -> ResponsesTool {
    ResponsesTool {
        tool_type: "function".to_string(),
        name: tool.name.clone(),
        description: tool.description.clone(),
        parameters: tool.input_schema.clone(),
    }
}

fn convert_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto { .. } => serde_json::json!("auto"),
        ToolChoice::Any { .. } => serde_json::json!("required"),
        ToolChoice::Tool { name, .. } => serde_json::json!({"type":"function", "name":name}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_result_is_function_call_output_and_arguments_are_json() {
        let result = serde_json::from_str::<MessagesRequest>(r#"{
            "model":"gpt-5.6-luna","max_tokens":128,
            "messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"call-1","content":"ok"}]}]
        }"#).unwrap();
        let value = serde_json::to_value(
            convert_request(&result, &crate::config::Config::from_env()).unwrap(),
        )
        .unwrap();
        assert_eq!(value["input"][0]["type"], "function_call_output");
        assert_eq!(value["input"][0]["call_id"], "call-1");
    }

    #[test]
    fn responses_uses_provider_max_effort_for_gpt() {
        use crate::config::{Config, ModelProfile, ProviderConfig, WireApi};
        use std::collections::HashMap;

        let mut effort_map = HashMap::new();
        effort_map.insert("max".to_string(), "max".to_string());
        let mut providers = HashMap::new();
        providers.insert(
            "gpt".to_string(),
            ProviderConfig {
                reasoning_field: String::new(),
                reasoning_field_alt: Vec::new(),
                thinking_param: None,
                thinking_type_enabled: None,
                thinking_type_disabled: None,
                disable_thinking: false,
                effort_param: "reasoning_effort".to_string(),
                effort_map,
                responses_reasoning_summary: None,
            },
        );
        let mut profile_by_name = HashMap::new();
        profile_by_name.insert("gpt-5.6-luna".to_string(), 0);
        let config = Config {
            listen_addr: String::new(),
            eswitch_url: String::new(),
            api_key: String::new(),
            log_level: String::new(),
            model_mapping: HashMap::new(),
            default_model: "gpt-5.6-luna".to_string(),
            model_profiles: vec![ModelProfile {
                name: "gpt-5.6-luna".to_string(),
                provider: "gpt".to_string(),
                reasoning_enabled: true,
                reasoning_replay: false,
                toolcall_requires_reasoning: false,
                aliases: Vec::new(),
                wire_api: WireApi::Responses,
            }],
            providers,
            profile_by_name,
        };
        let request = serde_json::from_value::<MessagesRequest>(serde_json::json!({
            "model": "gpt-5.6-luna",
            "max_tokens": 128,
            "thinking": {"type": "enabled", "budget_tokens": 32000},
            "messages": [{"role": "user", "content": "question"}]
        }))
        .unwrap();

        let wire = convert_request_with_relocation(&request, &config, false).unwrap();

        let wire_value = serde_json::to_value(&wire).unwrap();
        assert_eq!(wire_value["reasoning"]["effort"], "max");
        // Summary is a response-control field and must be requested separately
        // from the cache-relevant input/history/tools wire.
        assert_eq!(wire_value["reasoning"]["summary"], "auto");
        assert_eq!(
            wire.static_prefix_hash,
            canonical_hash(&serde_json::json!({
                "model": "gpt-5.6-luna",
                "instructions": serde_json::Value::Null,
                "tools": serde_json::Value::Null,
                "tool_choice": serde_json::Value::Null,
                "reasoning": {"effort": "max"},
            }))
        );
    }

    #[test]
    fn summary_control_does_not_change_cache_relevant_input_hashes() {
        let config = crate::config::Config::from_env();
        let request = serde_json::from_value::<MessagesRequest>(serde_json::json!({
            "model": "gpt-5.6-luna",
            "max_tokens": 128,
            "thinking": {"type": "enabled", "budget_tokens": 32000},
            "messages": [{"role": "user", "content": "question"}]
        }))
        .unwrap();
        let wire = convert_request_with_relocation(&request, &config, false).unwrap();
        let value = serde_json::to_value(&wire).unwrap();
        let input_hash = canonical_hash(&Value::Array(wire.input.clone()));
        let mut wire_without_summary = value.clone();
        wire_without_summary["reasoning"]
            .as_object_mut()
            .unwrap()
            .remove("summary");
        assert_eq!(input_hash, canonical_hash(&wire_without_summary["input"]));
        assert_eq!(
            wire.static_prefix_hash,
            canonical_hash(&serde_json::json!({
                "model": wire.model,
                "instructions": wire.instructions,
                "tools": wire.tools,
                "tool_choice": wire.tool_choice,
                "reasoning": {"effort": "max"},
            }))
        );
    }

    #[test]
    fn assistant_text_uses_output_text_while_user_text_uses_input_text() {
        let result = serde_json::from_value::<MessagesRequest>(serde_json::json!({
            "model": "gpt-5.6-luna",
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"}
            ]
        }))
        .unwrap();

        let value = serde_json::to_value(
            convert_request_with_relocation(&result, &crate::config::Config::from_env(), false)
                .unwrap(),
        )
        .unwrap();

        assert_eq!(value["input"][0]["content"][0]["type"], "input_text");
        assert_eq!(value["input"][1]["content"][0]["type"], "output_text");
    }

    fn request(round: &str) -> MessagesRequest {
        serde_json::from_value(serde_json::json!({
            "model":"gpt-5.6-luna", "max_tokens":128,
            "system":[
                {"type":"text", "text":"stable policy"},
                {"type":"text", "text":format!("<env>date: {round}</env>")}
            ],
            "messages":[
                {"role":"user", "content":"hello"},
                {"role":"assistant", "content":[{"type":"tool_use","id":"call-1","name":"lookup","input":{"q":"Paris"}}]},
                {"role":"user", "content":[{"type":"tool_result","tool_use_id":"call-1","content":"ok"}]}
            ]
        })).unwrap()
    }

    #[test]
    fn relocated_responses_wire_keeps_three_round_history_stable() {
        let config = crate::config::Config::from_env();
        let first = convert_request_with_relocation(&request("2026-08-05"), &config, true).unwrap();
        let second =
            convert_request_with_relocation(&request("2026-08-06"), &config, true).unwrap();
        let third = convert_request_with_relocation(&request("2026-08-07"), &config, true).unwrap();
        let history_len = first.input.len() - 1;
        assert_eq!(first.input[..history_len], second.input[..history_len]);
        assert_eq!(second.input[..history_len], third.input[..history_len]);
        assert_eq!(first.input[history_len]["role"], "user");
        assert_eq!(first.input[history_len]["content"][0]["type"], "input_text");
        assert!(first.synthetic_tail_present);
        assert_eq!(first.history_prefix_hash, second.history_prefix_hash);
        assert_ne!(first.wire_input_hash, second.wire_input_hash);
    }

    #[test]
    fn three_hashes_have_independent_canonical_semantics() {
        let config = crate::config::Config::from_env();
        let base = convert_request_with_relocation(&request("2026-08-05"), &config, true).unwrap();
        let tail_changed =
            convert_request_with_relocation(&request("2026-08-06"), &config, true).unwrap();
        let no_tail =
            convert_request_with_relocation(&request("2026-08-05"), &config, false).unwrap();
        assert_eq!(base.history_prefix_hash, tail_changed.history_prefix_hash);
        assert_ne!(base.wire_input_hash, tail_changed.wire_input_hash);
        assert_ne!(base.history_prefix_hash, base.wire_input_hash);
        assert_eq!(no_tail.history_prefix_hash, no_tail.wire_input_hash);
        assert_eq!(
            canonical_hash(&serde_json::json!({"b": 2, "a": 1})),
            canonical_hash(&serde_json::json!({"a": 1, "b": 2}))
        );
    }
}
