use crate::anthropic::types::{ContentBlock, ContentValue, Message, SystemPrompt};
use crate::conversation::AssistantPart;
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
///
/// `append_only` (Phase 4c) disables the chat encoder's history rewriting:
/// orphan `tool_calls` are not cleaned up, `tool_result` bodies are not
/// compacted, and repeated results are not deduplicated — stored history is
/// preserved byte-for-byte. `false` (the default) keeps the legacy
/// cleanup/compaction/dedup behavior.
pub fn build_chat_messages_with_reasoning(
    system: Option<&SystemPrompt>,
    messages: &[Message],
    include_reasoning: bool,
    append_only: bool,
) -> Vec<Value> {
    let conversation = crate::conversation::build_conversation(system, messages, Vec::new());
    render_chat_messages(&conversation, include_reasoning, append_only)
}

/// Render the lean ConversationIR to OpenAI chat wire messages.
/// Chat-specific safety nets (tool_result dedup/compaction, orphan tool_call
/// cleanup, reasoning placeholder) stay here on the wire form. In
/// `append_only` mode (Phase 4c) the cleanup/compaction/dedup safety nets are
/// disabled so already-provided history is never rewritten; only the
/// reasoning placeholder (a function of the stored message itself, Phase 4b)
/// still applies.
fn render_chat_messages(
    conversation: &crate::conversation::Conversation,
    include_reasoning: bool,
    append_only: bool,
) -> Vec<Value> {
    let mut result: Vec<Value> = Vec::new();

    // P1-3: tool_result dedup tracking (append-only mode never consults it)
    use std::collections::HashMap;
    let mut seen_results: HashMap<String, String> = HashMap::new(); // tool_call_id -> content_hash

    // 1. Handle system prompt
    if let Some(sys) = &conversation.system {
        let sys_text = system_prompt_to_text(sys);
        if !sys_text.is_empty() {
            result.push(json!({
                "role": "system",
                "content": sys_text,
            }));
        }
    }

    // 2. Convert each turn
    use crate::conversation::Turn;
    for turn in &conversation.turns {
        match turn {
            Turn::User { content } => {
                // Check if this is actually a tool_result message
                if let ContentValue::Blocks(blocks) = content {
                    let has_tool_results = blocks
                        .iter()
                        .any(|b| matches!(b, ContentBlock::ToolResult { .. }));
                    if has_tool_results {
                        // Convert to tool role messages. Default: dedup +
                        // compression (P1-3). Append-only (Phase 4c): stored
                        // history is preserved byte-for-byte — no dedup, no
                        // compaction, order/tool IDs/content bytes intact.
                        for block in blocks {
                            if let ContentBlock::ToolResult {
                                tool_use_id,
                                content: tool_content,
                                is_error,
                            } = block
                            {
                                let text = tool_result_to_text(tool_content);

                                let wire_content = if append_only {
                                    text
                                } else {
                                    let result_key = compute_result_key(&text);
                                    let tool_id = tool_use_id.clone();
                                    if let Some(existing_hash) = seen_results.get(&tool_id) {
                                        if *existing_hash == result_key {
                                            // Duplicate result, skip
                                            continue;
                                        }
                                    }
                                    seen_results.insert(tool_id, result_key);
                                    compact_tool_result(&text)
                                };

                                let content = if let Some(true) = is_error {
                                    format!("[ERROR] {}", wire_content)
                                } else {
                                    wire_content
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
                let openai_msg = convert_user_message(content);
                result.push(json!({
                    "role": "user",
                    "content": openai_msg,
                }));
            }
            Turn::Assistant { parts } => {
                let openai_msg = convert_assistant_parts(parts, include_reasoning);
                result.push(openai_msg);
            }
            Turn::Unknown { role, content } => {
                // Unknown role, keep as-is
                let content = content_value_to_json(content);
                result.push(json!({
                    "role": role,
                    "content": content,
                }));
            }
        }
    }

    // 3. Clean up orphan tool_calls (safety net from L1587-L1693).
    //    Phase 4c: disabled under append-only — stored history (including
    //    orphaned tool_calls and their results) is preserved byte-for-byte.
    if !append_only {
        cleanup_orphan_tool_calls(&mut result);
    }

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

fn convert_assistant_parts(parts: &[AssistantPart], include_reasoning: bool) -> Value {
    let mut thinking_parts: Vec<String> = Vec::new();
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();

    for part in parts {
        match part {
            AssistantPart::Reasoning(text) => {
                thinking_parts.push(text.clone());
            }
            AssistantPart::Text(text) => {
                text_parts.push(text.clone());
            }
            AssistantPart::ToolCall { id, name, input } => {
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": serde_json::to_string(input).unwrap_or_default(),
                    }
                }));
            }
            // ToolResult / Image inside an assistant message are dropped by the
            // Chat encoder (matches the pre-IR `_ => {}` behavior).
            AssistantPart::ToolResult { .. } | AssistantPart::Image { .. } => {}
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
        let result = build_chat_messages_with_reasoning(None, &messages, false, false);
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
        let result = build_chat_messages_with_reasoning(None, &messages, true, false);
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
        let result = build_chat_messages_with_reasoning(None, &messages, true, false);
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
        let result = build_chat_messages_with_reasoning(None, &messages, true, false);
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
        let result = build_chat_messages_with_reasoning(None, &messages, false, false);
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
        let result = build_chat_messages_with_reasoning(None, &messages, true, false);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["reasoning_content"], "(redacted thinking)");
        assert_eq!(result[0]["content"], "redacted response");
    }

    // --- Phase 4c: append-only stored-history preservation ---
    //
    // `append_only=true` must never rewrite already-provided history on the
    // wire: orphan tool_calls are not cleaned up, oversized tool_result bodies
    // are not compacted, repeated results are not deduplicated — order, tool
    // IDs and content bytes are preserved exactly as stored. `false` (the
    // default) keeps the legacy cleanup/compaction/dedup byte-for-byte.

    fn tool_result_block(id: &str, text: &str) -> ContentBlock {
        ContentBlock::ToolResult {
            tool_use_id: id.to_string(),
            content: crate::anthropic::types::ToolResultContent::Text(text.to_string()),
            is_error: None,
        }
    }

    #[test]
    fn append_only_preserves_orphan_tool_calls() {
        // Stored assistant tool_calls with NO matching tool result are
        // "orphans" (cleanup_orphan_tool_calls only strips them from
        // NON-final assistant messages). Legacy drops the orphan tool-call
        // assistant message entirely (empty content); append-only must keep
        // the tool_calls intact.
        let messages = vec![
            Message {
                role: "assistant".to_string(),
                content: ContentValue::Blocks(vec![ContentBlock::ToolUse {
                    id: "toolu_orphan".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/x"}),
                }]),
            },
            Message {
                role: "user".to_string(),
                content: ContentValue::Text("interlude".to_string()),
            },
            Message {
                role: "assistant".to_string(),
                content: ContentValue::Text("final answer".to_string()),
            },
        ];

        let legacy = build_chat_messages_with_reasoning(None, &messages, false, false);
        let legacy_assts: Vec<&Value> =
            legacy.iter().filter(|m| m["role"] == "assistant").collect();
        assert_eq!(
            legacy_assts.len(),
            1,
            "legacy: orphan tool-call assistant is cleaned up, only the final remains"
        );
        assert_eq!(legacy_assts[0]["content"], "final answer");

        let preserved = build_chat_messages_with_reasoning(None, &messages, false, true);
        let asst = preserved
            .iter()
            .find(|m| m["role"] == "assistant")
            .expect("append-only: orphan assistant message must be preserved");
        let calls = asst["tool_calls"].as_array().expect("tool_calls array");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "toolu_orphan");
        // The final assistant message survives too.
        assert!(preserved
            .iter()
            .any(|m| m["role"] == "assistant" && m["content"] == "final answer"));
    }

    #[test]
    fn append_only_preserves_oversized_tool_result() {
        // A tool_result body larger than the 4000+4000 compaction window is
        // truncated by the legacy encoder; append-only must keep the full
        // bytes (order, tool IDs and content bytes preserved).
        let big = "x".repeat(12000);
        let messages = vec![Message {
            role: "user".to_string(),
            content: ContentValue::Blocks(vec![tool_result_block("toolu_big", &big)]),
        }];

        let legacy = build_chat_messages_with_reasoning(None, &messages, false, false);
        let tool = legacy
            .iter()
            .find(|m| m["role"] == "tool")
            .expect("legacy: tool message present");
        let legacy_content = tool["content"].as_str().unwrap();
        assert!(
            legacy_content.len() < big.len() && legacy_content.contains("bytes truncated"),
            "legacy must compact the oversized result"
        );

        let preserved = build_chat_messages_with_reasoning(None, &messages, false, true);
        let tool = preserved
            .iter()
            .find(|m| m["role"] == "tool")
            .expect("append-only: tool message present");
        assert_eq!(
            tool["content"].as_str().unwrap(),
            big,
            "append-only must preserve the full result bytes"
        );
        assert_eq!(tool["tool_call_id"], "toolu_big");
    }

    #[test]
    fn append_only_preserves_repeated_tool_results() {
        // The P1-3 dedup drops a repeated result for the same tool_call_id.
        // Append-only must keep every occurrence, in order, with IDs and
        // content bytes intact.
        let messages = vec![Message {
            role: "user".to_string(),
            content: ContentValue::Blocks(vec![
                tool_result_block("toolu_dup", "result A"),
                tool_result_block("toolu_dup", "result A"),
            ]),
        }];

        let legacy = build_chat_messages_with_reasoning(None, &messages, false, false);
        let legacy_tools: Vec<&Value> = legacy.iter().filter(|m| m["role"] == "tool").collect();
        assert_eq!(
            legacy_tools.len(),
            1,
            "legacy: duplicate result for the same id is deduplicated"
        );

        let preserved = build_chat_messages_with_reasoning(None, &messages, false, true);
        let tools: Vec<&Value> = preserved.iter().filter(|m| m["role"] == "tool").collect();
        assert_eq!(
            tools.len(),
            2,
            "append-only: repeated results are preserved"
        );
        assert_eq!(tools[0]["tool_call_id"], "toolu_dup");
        assert_eq!(tools[0]["content"], "result A");
        assert_eq!(tools[1]["tool_call_id"], "toolu_dup");
        assert_eq!(tools[1]["content"], "result A");
    }

    #[test]
    fn append_only_preserves_order_and_ids() {
        // Order, tool IDs and content bytes are preserved exactly as stored.
        let messages = vec![Message {
            role: "user".to_string(),
            content: ContentValue::Blocks(vec![
                tool_result_block("toolu_b", "beta"),
                tool_result_block("toolu_a", "alpha"),
                tool_result_block("toolu_c", "gamma"),
            ]),
        }];

        let preserved = build_chat_messages_with_reasoning(None, &messages, false, true);
        let ids: Vec<&str> = preserved
            .iter()
            .filter(|m| m["role"] == "tool")
            .map(|m| m["tool_call_id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["toolu_b", "toolu_a", "toolu_c"]);
        let contents: Vec<&str> = preserved
            .iter()
            .filter(|m| m["role"] == "tool")
            .map(|m| m["content"].as_str().unwrap())
            .collect();
        assert_eq!(contents, vec!["beta", "alpha", "gamma"]);
    }

    #[test]
    fn off_retains_legacy_cleanup_compaction_dedup() {
        // Golden policy-off baseline: with append_only=false every legacy
        // safety net still fires in one combined history — orphan cleanup,
        // oversized-result compaction and duplicate-result dedup.
        let big = "z".repeat(12000);
        let messages = vec![
            Message {
                role: "assistant".to_string(),
                content: ContentValue::Blocks(vec![ContentBlock::ToolUse {
                    id: "toolu_orphan".to_string(),
                    name: "grep".to_string(),
                    input: serde_json::json!({"q": "x"}),
                }]),
            },
            Message {
                role: "user".to_string(),
                content: ContentValue::Blocks(vec![
                    tool_result_block("toolu_big", &big),
                    tool_result_block("toolu_big", &big),
                ]),
            },
            Message {
                role: "user".to_string(),
                content: ContentValue::Text("done".to_string()),
            },
            Message {
                role: "assistant".to_string(),
                content: ContentValue::Text("final".to_string()),
            },
        ];

        let legacy = build_chat_messages_with_reasoning(None, &messages, false, false);
        // Orphan assistant (non-final, no matching result) is removed.
        let assts: Vec<&Value> = legacy.iter().filter(|m| m["role"] == "assistant").collect();
        assert_eq!(assts.len(), 1, "orphan assistant cleaned up");
        assert_eq!(assts[0]["content"], "final");
        // Duplicate result for toolu_big is deduplicated to a single tool msg.
        let tools: Vec<&Value> = legacy.iter().filter(|m| m["role"] == "tool").collect();
        assert_eq!(tools.len(), 1, "duplicate result deduplicated");
        // Oversized result is compacted.
        let content = tools[0]["content"].as_str().unwrap();
        assert!(
            content.len() < big.len() && content.contains("bytes truncated"),
            "oversized result compacted"
        );
    }
}
