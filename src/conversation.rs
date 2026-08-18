//! Lean Conversation IR — the normalized backbone shared by the Chat and
//! Responses wire encoders (Phase 1: zero behavior change, gate G5).
//!
//! Per KIMI-K3-CACHE-OPTIMIZATION-FINAL-PLAN §5.1: this IR carries only the
//! normalization decisions both encoders share — system (post-relocation),
//! role dispatch, thinking→reasoning, ordered assistant parts, an explicit
//! `synthetic_tail` slot — and explicitly does **not** model wire vocabulary
//! (Chat `reasoning_content`/`role:"tool"` vs Responses
//! `output_text`/`function_call` stay in their respective encoders).
//!
//! Design note (byte-exactness requirement): assistant turns are normalized to
//! *ordered* parts rather than the plan's `{reasoning, text, tool_calls}`
//! buckets, because Responses flushes accumulated content before every
//! `function_call`/`function_call_output` — bucketting loses the interleave
//! order and would change wire bytes (G5 → NO-GO). The encoders bucket the
//! ordered parts themselves.
//!
//! The Chat relocate path in Phase 1 is unchanged (`migrate_volatile_system_blocks`
//! appends into the last user turn *before* the IR is built); the
//! `synthetic_tail` slot is populated by the Responses split-tail path only.
//! Switching Chat relocate to split-tail belongs to Phase 4d, not here.

use crate::anthropic::types::{
    ContentBlock, ContentValue, ImageSource, Message, SystemPrompt, ToolResultContent,
};
use serde_json::Value;

/// A normalized conversation: system + ordered turns + an explicit
/// synthetic-tail slot.
#[derive(Debug, Clone)]
pub struct Conversation {
    /// Post-relocation system prompt (stable part after migrate/split).
    pub system: Option<SystemPrompt>,
    /// Ordered, role-normalized turns.
    pub turns: Vec<Turn>,
    /// Volatile system texts relocated to the tail (Responses split-tail).
    /// Chat in Phase 1 does not populate this (Phase 4d switches it).
    pub synthetic_tail: Vec<String>,
}

/// A normalized conversation turn. Wire vocabulary lives in the encoders, not
/// here.
#[derive(Debug, Clone)]
pub enum Turn {
    /// User turn. Blocks are preserved as-is so wire-specific rendering keeps
    /// exact behavior (e.g. Chat's "any tool_result ⇒ tool-role path" rule and
    /// Responses' per-block input_text / function_call_output emission).
    User { content: ContentValue },
    /// Assistant turn: ordered parts (thinking→reasoning, text, tool calls,
    /// tool results, images). Both encoders bucket/filter these parts.
    Assistant { parts: Vec<AssistantPart> },
    /// Unknown/other-role passthrough (keeps role + original content). Chat
    /// renders via `content_value_to_json`, Responses via its content walk.
    Unknown { role: String, content: ContentValue },
}

/// Ordered assistant content parts. Every block kind the Anthropic request
/// model can carry is preserved so each encoder reproduces its own wire
/// behavior byte-for-byte:
///   * Chat:  Reasoning → `reasoning_content`, Text → `content`, ToolCall →
///     `tool_calls`, ToolResult/Image → dropped.
///   * Responses: Reasoning → dropped, Text → `output_text`, Image →
///     `input_image`, ToolCall → `function_call`, ToolResult →
///     `function_call_output`.
#[derive(Debug, Clone)]
pub enum AssistantPart {
    /// thinking or redacted-thinking text, normalized (thinking → reasoning).
    Reasoning(String),
    Text(String),
    ToolCall {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: ToolResultContent,
    },
    Image {
        source: ImageSource,
    },
}

/// Build the normalized conversation from the (possibly relocated) system and
/// messages. This is the single role-dispatch + reasoning-normalization point.
pub fn build_conversation(
    system: Option<&SystemPrompt>,
    messages: &[Message],
    synthetic_tail: Vec<String>,
) -> Conversation {
    let turns = messages
        .iter()
        .map(|msg| match msg.role.as_str() {
            "user" => Turn::User {
                content: msg.content.clone(),
            },
            "assistant" => Turn::Assistant {
                parts: assistant_parts(&msg.content),
            },
            role => Turn::Unknown {
                role: role.to_string(),
                content: msg.content.clone(),
            },
        })
        .collect();
    Conversation {
        system: system.cloned(),
        turns,
        synthetic_tail,
    }
}

/// Normalize an assistant message into ordered parts.
/// Thinking/RedactedThinking → Reasoning, Text → Text, ToolUse → ToolCall,
/// ToolResult → ToolResult, Image → Image. `Unknown` blocks are dropped —
/// both encoders drop them for assistant messages.
fn assistant_parts(content: &ContentValue) -> Vec<AssistantPart> {
    match content {
        ContentValue::Text(text) => vec![AssistantPart::Text(text.clone())],
        ContentValue::Null => Vec::new(),
        ContentValue::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Thinking { thinking, .. } => {
                    Some(AssistantPart::Reasoning(thinking.clone()))
                }
                ContentBlock::RedactedThinking { .. } => {
                    Some(AssistantPart::Reasoning("(redacted thinking)".to_string()))
                }
                ContentBlock::Text { text } => Some(AssistantPart::Text(text.clone())),
                ContentBlock::ToolUse { id, name, input } => Some(AssistantPart::ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                }),
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error: _,
                } => Some(AssistantPart::ToolResult {
                    tool_use_id: tool_use_id.clone(),
                    content: content.clone(),
                }),
                ContentBlock::Image { source } => Some(AssistantPart::Image {
                    source: source.clone(),
                }),
                ContentBlock::Unknown => None,
            })
            .collect(),
    }
}

/// Minimal shared helper: sort a slice of wire tools by name for KV-cache
/// prefix stability. Chat sorts by `function.name`, Responses by `name`.
pub(crate) fn sort_by_name<T>(items: &mut [T], name: impl Fn(&T) -> &str) {
    items.sort_by(|a, b| name(a).cmp(name(b)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::types::ContentBlock;

    fn msg(role: &str, content: ContentValue) -> Message {
        Message {
            role: role.to_string(),
            content,
        }
    }

    #[test]
    fn assistant_blocks_normalize_to_ordered_parts() {
        let parts = assistant_parts(&ContentValue::Blocks(vec![
            ContentBlock::Thinking {
                thinking: "t1".to_string(),
                signature: "s".to_string(),
            },
            ContentBlock::Text {
                text: "a1".to_string(),
            },
            ContentBlock::ToolUse {
                id: "call-1".to_string(),
                name: "read_file".to_string(),
                input: serde_json::json!({"path": "/tmp/x"}),
            },
            ContentBlock::Image {
                source: crate::anthropic::types::ImageSource {
                    source_type: "base64".to_string(),
                    media_type: "image/png".to_string(),
                    data: "aGVsbG8=".to_string(),
                },
            },
            ContentBlock::ToolResult {
                tool_use_id: "call-1".to_string(),
                content: ToolResultContent::Text("ok".to_string()),
                is_error: None,
            },
            ContentBlock::Unknown,
        ]));
        assert_eq!(parts.len(), 5, "Unknown block must be dropped");
        assert!(matches!(parts[0], AssistantPart::Reasoning(ref r) if r == "t1"));
        assert!(matches!(parts[1], AssistantPart::Text(ref t) if t == "a1"));
        assert!(matches!(
            parts[2],
            AssistantPart::ToolCall { ref id, ref name, .. } if id == "call-1" && name == "read_file"
        ));
        assert!(matches!(parts[3], AssistantPart::Image { .. }));
        assert!(matches!(
            parts[4],
            AssistantPart::ToolResult { ref tool_use_id, ref content, .. }
                if tool_use_id == "call-1"
                    && matches!(content, ToolResultContent::Text(ref t) if t == "ok")
        ));
    }

    #[test]
    fn redacted_thinking_becomes_reasoning_marker() {
        let parts = assistant_parts(&ContentValue::Blocks(vec![
            ContentBlock::RedactedThinking {
                data: "enc".to_string(),
            },
        ]));
        assert!(matches!(parts[0], AssistantPart::Reasoning(ref r) if r == "(redacted thinking)"));
    }

    #[test]
    fn text_and_null_content_map_to_parts() {
        assert!(matches!(
            assistant_parts(&ContentValue::Text("hi".to_string()))[0],
            AssistantPart::Text(ref t) if t == "hi"
        ));
        assert!(assistant_parts(&ContentValue::Null).is_empty());
    }

    #[test]
    fn role_dispatch_and_synthetic_tail() {
        let messages = vec![
            msg("user", ContentValue::Text("u".to_string())),
            msg("assistant", ContentValue::Text("a".to_string())),
            msg("developer", ContentValue::Text("d".to_string())),
        ];
        let conv = build_conversation(
            Some(&SystemPrompt::Text("sys".to_string())),
            &messages,
            vec!["<env>volatile</env>".to_string()],
        );
        assert!(matches!(conv.system, Some(SystemPrompt::Text(ref t)) if t == "sys"));
        assert_eq!(conv.turns.len(), 3);
        assert!(matches!(conv.turns[0], Turn::User { .. }));
        assert!(matches!(conv.turns[1], Turn::Assistant { .. }));
        assert!(matches!(conv.turns[2], Turn::Unknown { ref role, .. } if role == "developer"));
        assert_eq!(conv.synthetic_tail, vec!["<env>volatile</env>".to_string()]);
    }

    #[test]
    fn sort_by_name_orders_ascending() {
        let mut items = vec!["zeta", "alpha", "mid"];
        sort_by_name(&mut items, |s| s);
        assert_eq!(items, vec!["alpha", "mid", "zeta"]);
    }
}
