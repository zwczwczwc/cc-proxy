use crate::anthropic::types::{ContentBlock, ContentValue, Message, SystemPrompt};
use serde_json::{json, Value};

/// Compute a hash key for tool_result deduplication (P1-3).
fn compute_result_key(content: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// Compress tool_result: HEAD 4000 + TAIL 4000 (P1-3).
fn compact_tool_result(content: &str) -> String {
    const HEAD: usize = 4000;
    const TAIL: usize = 4000;
    if content.len() <= HEAD + TAIL {
        content.to_string()
    } else {
        // Safe UTF-8 char boundary slicing (P5-2: fix panic on multi-byte chars)
        let head_end = content
            .char_indices()
            .nth(HEAD)
            .map(|(i, _)| i)
            .unwrap_or(content.len());
        let tail_start = content
            .char_indices()
            .rev()
            .nth(TAIL)
            .map(|(i, _)| i)
            .unwrap_or(0);
        format!(
            "{}...\n[{} bytes truncated]\n...{}",
            &content[..head_end],
            content.len() - head_end - (content.len() - tail_start),
            &content[tail_start..]
        )
    }
}

/// Builds OpenAI-format chat messages from Anthropic Messages API format.
/// Adapted from CodeWhale chat.rs L1357-L1696.
/// Returns a Vec<Value> of OpenAI-format messages.
pub fn build_chat_messages_with_reasoning(
    system: Option<&SystemPrompt>,
    messages: &[Message],
    include_reasoning: bool,
) -> Vec<Value> {
    let mut result: Vec<Value> = Vec::new();

    // P1-3: tool_result dedup tracking
    use std::collections::HashMap;
    let mut seen_results: HashMap<String, String> = HashMap::new(); // tool_call_id -> content_hash

    // 1. Handle system prompt
    if let Some(sys) = system {
        let sys_text = system_prompt_to_text(sys);
        if !sys_text.is_empty() {
            result.push(json!({
                "role": "system",
                "content": sys_text,
            }));
        }
    }

    // 2. Convert each message
    for msg in messages {
        let role = &msg.role;
        match role.as_str() {
            "user" => {
                // Check if this is actually a tool_result message
                if let ContentValue::Blocks(blocks) = &msg.content {
                    let has_tool_results = blocks
                        .iter()
                        .any(|b| matches!(b, ContentBlock::ToolResult { .. }));
                    if has_tool_results {
                        // Convert to tool role messages with dedup + compression (P1-3)
                        for block in blocks {
                            if let ContentBlock::ToolResult {
                                tool_use_id,
                                content: tool_content,
                                is_error,
                            } = block
                            {
                                let text = tool_result_to_text(tool_content);
                                let result_key = compute_result_key(&text);
                                let compacted = compact_tool_result(&text);

                                let tool_id = tool_use_id.clone();
                                if let Some(existing_hash) = seen_results.get(&tool_id) {
                                    if *existing_hash == result_key {
                                        // Duplicate result, skip
                                        continue;
                                    }
                                }
                                seen_results.insert(tool_id, result_key);

                                let content = if let Some(true) = is_error {
                                    format!("[ERROR] {}", compacted)
                                } else {
                                    compacted
                                };
                                result.push(json!({
                                    "role": "tool",
                                    "tool_call_id": tool_use_id,
                                    "content": content,
                                }));
                            }
                        }
                        continue;
                    }
                }
                let openai_msg = convert_user_message(&msg.content);
                result.push(json!({
                    "role": "user",
                    "content": openai_msg,
                }));
            }
            "assistant" => {
                let openai_msg = convert_assistant_message(&msg.content, include_reasoning);
                result.push(openai_msg);
            }
            _ => {
                // Unknown role, keep as-is
                let content = content_value_to_json(&msg.content);
                result.push(json!({
                    "role": role,
                    "content": content,
                }));
            }
        }
    }

    // 3. Clean up orphan tool_calls (safety net from L1587-L1693)
    cleanup_orphan_tool_calls(&mut result);

    result
}

fn system_prompt_to_text(sys: &SystemPrompt) -> String {
    match sys {
        SystemPrompt::Text(s) => s.clone(),
        SystemPrompt::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| {
                if b.block_type == "text" {
                    Some(b.text.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn content_value_to_json(content: &ContentValue) -> Value {
    match content {
        ContentValue::Text(s) => json!(s),
        ContentValue::Null => Value::Null,
        ContentValue::Blocks(blocks) => {
            let items: Vec<Value> = blocks
                .iter()
                .map(|block| match block {
                    ContentBlock::Text { text } => json!({"type": "text", "text": text}),
                    ContentBlock::ToolUse { id, name, input } => {
                        json!({"type": "tool_use", "id": id, "name": name, "input": input})
                    }
                    _ => json!({}),
                })
                .collect();
            json!(items)
        }
    }
}

fn convert_user_message(content: &ContentValue) -> Value {
    match content {
        ContentValue::Text(s) => json!(s),
        ContentValue::Null => Value::Null,
        ContentValue::Blocks(blocks) => {
            // Check if it's a simple text block array
            let all_text = blocks
                .iter()
                .all(|b| matches!(b, ContentBlock::Text { .. }));
            if all_text {
                let text: String = blocks
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::Text { text } = b {
                            Some(text.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                return json!(text);
            }

            // Mixed content: convert to OpenAI multi-part format
            let parts: Vec<Value> = blocks
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => Some(json!({"type": "text", "text": text})),
                    ContentBlock::Image { source } => {
                        // Convert Anthropic base64 image to OpenAI data URL
                        let data_url = format!("data:{};base64,{}", source.media_type, source.data);
                        Some(json!({
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        }))
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content: tool_content,
                        is_error,
                    } => {
                        let tool_text = tool_result_to_text(tool_content);
                        let mut val = json!({
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": tool_text,
                        });
                        if let Some(true) = is_error {
                            // Attach error info in content
                            val["content"] = json!(format!("[ERROR] {}", tool_text));
                        }
                        Some(val)
                    }
                    _ => None,
                })
                .collect();

            json!(parts)
        }
    }
}

fn tool_result_to_text(content: &crate::anthropic::types::ToolResultContent) -> String {
    match content {
        crate::anthropic::types::ToolResultContent::Text(s) => s.clone(),
        crate::anthropic::types::ToolResultContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| match b {
                crate::anthropic::types::ToolResultContentBlock::Text { text } => {
                    Some(text.clone())
                }
                crate::anthropic::types::ToolResultContentBlock::Image { source: _ } => {
                    Some("[image]".to_string())
                }
                crate::anthropic::types::ToolResultContentBlock::Unknown => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn convert_assistant_message(content: &ContentValue, include_reasoning: bool) -> Value {
    let blocks = match content {
        ContentValue::Text(text) => {
            return json!({
                "role": "assistant",
                "content": text,
            });
        }
        ContentValue::Null => {
            return json!({
                "role": "assistant",
                "content": null,
            });
        }
        ContentValue::Blocks(b) => b,
    };

    let mut thinking_parts: Vec<String> = Vec::new();
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();

    for block in blocks {
        match block {
            ContentBlock::Thinking {
                thinking,
                signature: _,
            } => {
                thinking_parts.push(thinking.clone());
            }
            ContentBlock::RedactedThinking { data: _ } => {
                thinking_parts.push("(redacted thinking)".to_string());
            }
            ContentBlock::Text { text } => {
                text_parts.push(text.clone());
            }
            ContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": serde_json::to_string(input).unwrap_or_default(),
                    }
                }));
            }
            _ => {}
        }
    }

    let has_tool_calls = !tool_calls.is_empty();
    let text_content = text_parts.join("\n");
    let mut reasoning_content_str = thinking_parts.join("\n");

    // Reasoning replay must be a function of the stored message ONLY,
    // never of later history. DeepSeek's prefix cache hashes the raw
    // bytes of every message; flipping reasoning_content on/off depending
    // on whether a follow-up user turn exists rewrites a historical message
    // between turns and busts the cache from that point onwards.
    // Always emit reasoning_content when the model requires replay AND
    // the stored message carries thinking text.
    // Tool-call messages with empty thinking still need a placeholder
    // (DeepSeek 400s without it), but text-only assistant messages
    // simply omit the field when there's nothing to replay.
    let mut has_reasoning = include_reasoning && !reasoning_content_str.trim().is_empty();
    if include_reasoning && has_tool_calls && !has_reasoning {
        tracing::warn!(
            "Substituting placeholder reasoning_content for tool-call assistant message"
        );
        reasoning_content_str = "(reasoning omitted)".to_string();
        has_reasoning = true;
    }

    let mut msg = json!({
        "role": "assistant",
    });

    if !text_content.is_empty() {
        msg["content"] = json!(text_content);
    } else if has_reasoning {
        msg["content"] = json!(""); // DeepSeek rejects null content with reasoning
    } else {
        msg["content"] = json!(null);
    }

    if has_reasoning {
        msg["reasoning_content"] = json!(reasoning_content_str);
    }

    if has_tool_calls {
        msg["tool_calls"] = json!(tool_calls);
    }

    msg
}

/// Safety net: removes orphaned tool_calls from messages that shouldn't have them.
/// Adapted from CodeWhale chat.rs L1587-L1693 with full ID set matching (P1-2).
/// Only removes tool_calls from non-final assistant messages (last message's tool_calls are always valid).
fn cleanup_orphan_tool_calls(messages: &mut Vec<Value>) {
    use std::collections::HashSet;

    let mut orphan_indices: Vec<usize> = Vec::new();

    // Find the index of the last assistant message
    let last_assistant_idx = messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, msg)| msg.get("role").and_then(|v| v.as_str()) == Some("assistant"))
        .map(|(i, _)| i);

    for (i, msg) in messages.iter().enumerate() {
        // Skip the last assistant message (its tool_calls are always valid)
        if Some(i) == last_assistant_idx {
            continue;
        }

        let is_assistant = msg.get("role").and_then(|v| v.as_str()) == Some("assistant");

        if !is_assistant {
            continue;
        }

        let tool_calls = msg.get("tool_calls").and_then(|v| v.as_array());
        if tool_calls.is_none() {
            continue;
        }

        let tool_calls = tool_calls.unwrap();

        // Collect tool call IDs from this assistant message
        let mut ids: HashSet<String> = HashSet::new();
        for tc in tool_calls {
            if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                ids.insert(id.to_string());
            }
        }

        // Scan subsequent messages for matching tool results
        let mut found_ids: HashSet<String> = HashSet::new();
        for message in messages.iter().skip(i + 1) {
            if let Some(tool_id) = message.get("tool_call_id").and_then(|v| v.as_str()) {
                found_ids.insert(tool_id.to_string());
            }
        }

        // If not all tool call IDs have matching tool results, mark as orphan
        if !ids.is_subset(&found_ids) {
            orphan_indices.push(i);
        }
    }

    // Remove orphan tool_calls (iterate in reverse to preserve indices)
    for &idx in orphan_indices.iter().rev() {
        if let Some(obj) = messages[idx].as_object_mut() {
            // Collect orphan IDs before removing tool_calls
            let orphan_ids: Vec<String> = obj
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|tc| {
                            tc.get("id").and_then(|v| v.as_str()).map(|s| s.to_string())
                        })
                        .collect()
                })
                .unwrap_or_default();

            // Remove tool_calls from this message
            obj.remove("tool_calls");

            // If assistant had no content, remove the entire message
            let has_content = obj
                .get("content")
                .is_some_and(|v| !v.is_null() && v.as_str() != Some(""));

            if !has_content {
                messages.remove(idx);
            }

            // Remove associated orphan tool result messages
            messages.retain(|msg| {
                msg.get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .is_none_or(|tid| !orphan_ids.contains(&tid.to_string()))
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_user_message() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: ContentValue::Text("hello".to_string()),
        }];
        let result = build_chat_messages_with_reasoning(None, &messages, false);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "user");
        assert_eq!(result[0]["content"], "hello");
    }

    #[test]
    fn test_assistant_with_thinking() {
        // reasoning_content should contain actual thinking text when include_reasoning=true.
        // Aligned with CodeWhale chat.rs L1549-L1591.
        let messages = vec![Message {
            role: "assistant".to_string(),
            content: ContentValue::Blocks(vec![
                ContentBlock::Thinking {
                    thinking: "let me think...".to_string(),
                    signature: "sig1".to_string(),
                },
                ContentBlock::Text {
                    text: "here is the answer".to_string(),
                },
            ]),
        }];
        let result = build_chat_messages_with_reasoning(None, &messages, true);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "assistant");
        // Fix: reasoning_content should contain actual thinking text
        assert_eq!(result[0]["reasoning_content"], "let me think...");
        assert_eq!(result[0]["content"], "here is the answer");
    }

    #[test]
    fn test_tool_calls_placeholder() {
        let messages = vec![Message {
            role: "assistant".to_string(),
            content: ContentValue::Blocks(vec![ContentBlock::ToolUse {
                id: "toolu_1".to_string(),
                name: "read_file".to_string(),
                input: serde_json::json!({"path": "/tmp/test"}),
            }]),
        }];
        let result = build_chat_messages_with_reasoning(None, &messages, true);
        assert_eq!(result[0]["reasoning_content"], "(reasoning omitted)");
        assert!(result[0]["tool_calls"].is_array());
    }

    #[test]
    fn test_system_prompt() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: ContentValue::Text("hi".to_string()),
        }];
        let result = build_chat_messages_with_reasoning(
            Some(&SystemPrompt::Text("You are helpful".to_string())),
            &messages,
            false,
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["role"], "system");
        assert_eq!(result[0]["content"], "You are helpful");
    }

    #[test]
    fn test_assistant_with_thinking_and_tool_calls() {
        // Thinking + ToolUse combination: reasoning_content should be actual thinking text
        let messages = vec![Message {
            role: "assistant".to_string(),
            content: ContentValue::Blocks(vec![
                ContentBlock::Thinking {
                    thinking: "I should read the file".to_string(),
                    signature: "sig1".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test"}),
                },
            ]),
        }];
        let result = build_chat_messages_with_reasoning(None, &messages, true);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "assistant");
        assert_eq!(result[0]["reasoning_content"], "I should read the file");
        assert!(result[0]["tool_calls"].is_array());
    }

    #[test]
    fn test_assistant_include_reasoning_false() {
        // include_reasoning=false: reasoning_content should be omitted even with thinking
        let messages = vec![Message {
            role: "assistant".to_string(),
            content: ContentValue::Blocks(vec![
                ContentBlock::Thinking {
                    thinking: "let me think...".to_string(),
                    signature: "sig1".to_string(),
                },
                ContentBlock::Text {
                    text: "here is the answer".to_string(),
                },
            ]),
        }];
        let result = build_chat_messages_with_reasoning(None, &messages, false);
        assert_eq!(result.len(), 1);
        assert!(
            result[0].get("reasoning_content").is_none()
                || result[0]["reasoning_content"].is_null()
        );
        assert_eq!(result[0]["content"], "here is the answer");
    }

    #[test]
    fn test_assistant_redacted_thinking() {
        // RedactedThinking block should be included as "(redacted thinking)" text
        let messages = vec![Message {
            role: "assistant".to_string(),
            content: ContentValue::Blocks(vec![
                ContentBlock::RedactedThinking {
                    data: "encrypted_data".to_string(),
                },
                ContentBlock::Text {
                    text: "redacted response".to_string(),
                },
            ]),
        }];
        let result = build_chat_messages_with_reasoning(None, &messages, true);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["reasoning_content"], "(redacted thinking)");
        assert_eq!(result[0]["content"], "redacted response");
    }
}
