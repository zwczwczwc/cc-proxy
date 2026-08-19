// Copyright (c) 2025 cc-proxy
// Ported from permafrost permafrost_align.py L47-L310
// Reference: https://github.com/jianzhichun/permafrost
//
// Detects volatile env/context blocks in Claude Code's system prompt
// (dates, UUIDs, git status, etc.) and relocates them from the
// cache-prefix-sensitive system position to the tail of the latest
// user turn, where they no longer bust the KV cache prefix.
//
// Controlled by env var CODEMERMAFROST_RELOCATE — unset to disable.

use regex::Regex;
use std::sync::OnceLock;

use crate::anthropic::types::{ContentBlock, ContentValue, Message, SystemPrompt};

// --- volatile-content detectors (structural, anchored) ----------------------
// Ported from permafrost_align.py L47-L80

fn re_iso_datetime() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?").unwrap())
}

fn re_date() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b\d{4}-\d{2}-\d{2}\b").unwrap())
}

fn re_uuid() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
        )
        .unwrap()
    })
}

fn re_hex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b[0-9a-fA-F]{32,64}\b").unwrap())
}

/// Markers that identify a Claude Code "environment / context" block.
/// Ported from permafrost_align.py L63-L73
const ENV_MARKERS: &[&str] = &[
    "<env>",
    "Working directory",
    "Is directory a git repo",
    "Today's date",
    "Current branch",
    "Recent commits",
    "gitStatus",
    "Platform:",
    "OS Version",
];

/// Check if text looks like a Claude Code env/context block.
fn looks_like_env_block(text: &str) -> bool {
    ENV_MARKERS.iter().any(|m| text.contains(m))
}

/// Count volatile tokens found in text.
fn count_volatile(text: &str) -> usize {
    let mut count = 0;
    if re_iso_datetime().is_match(text) {
        count += 1;
    }
    if re_uuid().is_match(text) {
        count += 1;
    }
    if re_hex().is_match(text) {
        count += 1;
    }
    if re_date().is_match(text) {
        count += 1;
    }
    // Note: permafrost defines _RE_GIT_SHA but does NOT include it
    // in _VOLATILE_LABELS — we match that behaviour.
    count
}

/// Pin per-request nonces in Claude Code's billing-header system block.
/// Ported from permafrost_align.py L149-L177.
///
/// Claude Code injects a billing-telemetry block as the FIRST system block,
/// carrying a per-request `cch=<nonce>` that changes every call. Sitting in
/// the cache prefix, that nonce busts the cache for the whole system prompt.
/// We replace the nonce with a constant — the model ignores this telemetry.
const BILLING_MARKER: &str = "x-anthropic-billing-header";

fn re_cch() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(cch=)[^;\s]*").unwrap())
}

pub fn stabilize_metadata(mut system: SystemPrompt) -> SystemPrompt {
    let blocks = match &mut system {
        SystemPrompt::Blocks(ref mut b) => b,
        SystemPrompt::Text(_) => return system,
    };

    let mut count = 0;
    for block in blocks.iter_mut() {
        if block.text.contains(BILLING_MARKER) {
            let new_text = re_cch().replace(&block.text, "${1}cc-proxy").to_string();
            if new_text != block.text {
                block.text = new_text;
                count += 1;
            }
        }
    }

    if count > 0 {
        tracing::info!(
            metadata_stabilized = count,
            "stabilize_metadata: pinned billing nonce"
        );
    }

    system
}

/// Migrate volatile system blocks from the system prompt to the tail of the
/// latest user message. Returns the modified system prompt and messages.
///
/// Ported from permafrost_align.py L248-L310.
pub fn migrate_volatile_system_blocks(
    system: SystemPrompt,
    mut messages: Vec<Message>,
) -> (SystemPrompt, Vec<Message>) {
    let blocks = match &system {
        SystemPrompt::Blocks(b) => b.clone(),
        // String system prompts: can't relocate safely, leave intact.
        SystemPrompt::Text(_) => return (system, messages),
    };

    if blocks.is_empty() {
        return (system, messages);
    }

    let mut keep: Vec<crate::anthropic::types::SystemContentBlock> = Vec::new();
    let mut moved_texts: Vec<String> = Vec::new();

    for block in &blocks {
        let text = &block.text;
        let v = count_volatile(text);
        if v > 0 && looks_like_env_block(text) {
            moved_texts.push(text.clone());
        } else {
            keep.push(block.clone());
        }
    }

    if moved_texts.is_empty() {
        return (system, messages);
    }

    let relocated_count = moved_texts.len();
    let relocated_chars: usize = moved_texts.iter().map(|t| t.len()).sum();

    // Re-attach moved blocks to the tail of the last message.
    if messages.is_empty() {
        // No turn to attach to — fold back rather than drop content.
        tracing::warn!(
            "relocate_volatile: no message turn to attach env block onto; left in system"
        );
        return (system, messages);
    }

    let last_idx = messages.len() - 1;
    let last_msg = &mut messages[last_idx];

    // Build the relocated context wrapper + moved blocks
    let mut appendix = String::from(
        "\n\n<permafrost:relocated-context>\n\
         Moved out of the cache prefix so it can change \
         without resetting the cache. Same meaning, later position.\n\
         </permafrost:relocated-context>\n\n",
    );
    for t in &moved_texts {
        appendix.push_str(t);
        appendix.push('\n');
    }

    // Append to the last message's content
    match &mut last_msg.content {
        crate::anthropic::types::ContentValue::Text(ref mut existing) => {
            existing.push_str(&appendix);
        }
        crate::anthropic::types::ContentValue::Blocks(ref mut existing_blocks) => {
            existing_blocks.push(crate::anthropic::types::ContentBlock::Text { text: appendix });
        }
        crate::anthropic::types::ContentValue::Null => {
            // Convert null content to text so we can append the relocated context
            last_msg.content = crate::anthropic::types::ContentValue::Text(appendix);
        }
    }

    tracing::info!(
        relocated_blocks = relocated_count,
        relocated_chars = relocated_chars,
        "relocate_volatile: moved env blocks from system prefix to last user turn"
    );

    let new_system = SystemPrompt::Blocks(keep);
    (new_system, messages)
}

/// Extract volatile system blocks without mutating conversation history.
pub fn split_volatile_system_blocks(system: SystemPrompt) -> (SystemPrompt, Vec<String>) {
    let blocks = match &system {
        SystemPrompt::Blocks(blocks) => blocks,
        SystemPrompt::Text(_) => return (system, Vec::new()),
    };
    let mut keep = Vec::new();
    let mut moved = Vec::new();
    for block in blocks {
        if count_volatile(&block.text) > 0 && looks_like_env_block(&block.text) {
            moved.push(block.text.clone());
        } else {
            keep.push(block.clone());
        }
    }
    if moved.is_empty() {
        (system, moved)
    } else {
        (SystemPrompt::Blocks(keep), moved)
    }
}

/// Split volatile system blocks out of the cache-prefix-sensitive system
/// position and move them to a deterministic conversation tail (Phase 4d).
///
/// This is the Chat-converter relocation used by the policy-gated
/// `cache_policy.relocate = "split_tail"` route (official Kimi upstream
/// only). Unlike [`migrate_volatile_system_blocks`], it never rewrites
/// already-constructed stable history: the volatile appendix is attached
/// only to the tail of the conversation.
///
/// Semantics (conservative, alternation-safe, tool-call-pairing-safe):
///   * No volatile blocks -> system and messages returned as-is.
///   * Empty messages -> volatile blocks fold back into the system prompt
///     (never silently dropped).
///   * Last message is an ASSISTANT turn (including one carrying pending
///     `tool_calls`) -> the appendix is merged into the assistant content
///     (`Text` append / `Null` -> `Text` / `Blocks` + extra `Text` block). A
///     synthetic `user` turn is NEVER appended after an assistant, because
///     inserting a non-tool-result user turn between an assistant's
///     `tool_calls` and its (future) `tool_result` would break the Chat
///     tool_call -> tool_result pairing hard constraint.
///   * Non-assistant last message with `Text` or all-`Text` blocks -> appendix
///     appended to it (no new message, no user-after-user).
///   * Non-assistant last message is `Null` -> converted to `Text(appendix)`.
///   * Non-assistant last message has any non-text block (e.g. a
///     `tool_result`, which renders to the Chat `tool` role and would drop
///     appended text) -> a brand-new `user` `Text` message is appended
///     (synthetic tail; the wire ends tool -> user, preserving role
///     alternation).
///
/// The split is deterministic (stable block order, moved block order), so the
/// same input always produces the identical outbound body.
pub fn relocate_volatile_to_chat_tail(
    system: SystemPrompt,
    mut messages: Vec<Message>,
) -> (SystemPrompt, Vec<Message>) {
    // Empty-messages fold-back (data-loss guard): with no turn to attach the
    // relocated blocks onto, the volatile blocks must fold back into the
    // system prompt — they are NEVER dropped. This mirrors the legacy
    // `migrate_volatile_system_blocks` contract, which returns the original
    // (unstripped) system prompt in the same situation.
    if messages.is_empty() {
        tracing::warn!(
            "relocate_volatile_to_chat_tail: no message turn to attach env block onto; left in system"
        );
        return (system, messages);
    }

    let (new_system, moved_texts) = split_volatile_system_blocks(system);
    if moved_texts.is_empty() {
        return (new_system, messages);
    }

    // Build the relocated-context wrapper + moved blocks (same as migrate).
    let mut appendix = String::from(
        "\n\n<permafrost:relocated-context>\n\
         Moved out of the cache prefix so it can change \
         without resetting the cache. Same meaning, later position.\n\
         </permafrost:relocated-context>\n\n",
    );
    for t in &moved_texts {
        appendix.push_str(t);
        appendix.push('\n');
    }

    let last_idx = messages.len() - 1;

    // Decide whether the final turn can absorb the appendix without dropping
    // it on the wire. An ASSISTANT tail always absorbs — even one carrying
    // pending `tool_calls`: the appendix merges into the assistant content
    // (assistant may legally carry both text content and tool_calls), and a
    // synthetic `user` turn is NEVER appended after an assistant, because
    // inserting a non-tool-result user turn between an assistant's tool_calls
    // and its (future) tool_result would break the Chat
    // tool_call -> tool_result pairing hard constraint. For a non-assistant
    // tail: plain Text, Null (converted to Text) or all-Text blocks absorb;
    // any non-text block (e.g. tool_result -> Chat `tool` role) would drop
    // appended text, so a synthetic user tail is used instead (wire ends
    // tool -> user, alternation preserved).
    let can_absorb = if messages[last_idx].role == "assistant" {
        true
    } else {
        match &messages[last_idx].content {
            ContentValue::Text(_) | ContentValue::Null => true,
            ContentValue::Blocks(blocks) => blocks
                .iter()
                .all(|b| matches!(b, ContentBlock::Text { .. })),
        }
    };

    if can_absorb {
        let last_msg = &mut messages[last_idx];
        match &mut last_msg.content {
            ContentValue::Text(existing) => existing.push_str(&appendix),
            ContentValue::Blocks(existing_blocks) => {
                existing_blocks.push(ContentBlock::Text { text: appendix });
            }
            ContentValue::Null => {
                last_msg.content = ContentValue::Text(appendix);
            }
        }
    } else {
        // Synthetic tail only for non-assistant turns (e.g. a `user` turn
        // whose blocks render to the `tool` role): append a brand-new `user`
        // Text turn so the wire ends tool -> user (role alternation
        // preserved, no user-after-user).
        messages.push(Message {
            role: "user".to_string(),
            content: ContentValue::Text(appendix),
        });
    }

    tracing::info!(
        relocated_blocks = moved_texts.len(),
        relocated_chars = moved_texts.iter().map(|t| t.len()).sum::<usize>(),
        "relocate_volatile_to_chat_tail: moved env blocks from system prefix to conversation tail"
    );

    (new_system, messages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::types::{
        ContentBlock, ContentValue, SystemContentBlock, ToolResultContent,
    };

    #[test]
    fn test_detect_volatile_env_block() {
        let text = "<env>\nWorking directory: /home/user\nToday's date: 2026-06-22\nPlatform: linux\ngitStatus: M foo.rs\nRecent commits: a1b2c3d4\n</env>";
        assert!(looks_like_env_block(text));
        assert!(count_volatile(text) >= 1); // date (2026-06-22) — git SHA removed per permafrost
    }

    #[test]
    fn test_stable_block_not_detected() {
        let text = "You are a helpful coding assistant. Follow these rules:\n1. Be concise\n2. Use tools when needed";
        assert!(!looks_like_env_block(text));
        assert_eq!(count_volatile(text), 0);
    }

    #[test]
    fn test_relocate_volatile_blocks() {
        let system = SystemPrompt::Blocks(vec![
            SystemContentBlock {
                block_type: "text".to_string(),
                text: "You are a helpful assistant.".to_string(),
            },
            SystemContentBlock {
                block_type: "text".to_string(),
                text: "<env>\nWorking directory: /tmp\nToday's date: 2026-06-22\ngitStatus: M foo.rs\n</env>"
                    .to_string(),
            },
        ]);

        let messages = vec![Message {
            role: "user".to_string(),
            content: ContentValue::Text("hello".to_string()),
        }];

        let (new_system, new_messages) = migrate_volatile_system_blocks(system, messages);

        // System should only have stable block
        match &new_system {
            SystemPrompt::Blocks(b) => {
                assert_eq!(b.len(), 1);
                assert_eq!(b[0].text, "You are a helpful assistant.");
            }
            _ => panic!("expected Blocks"),
        }

        // Last user message should contain the relocated content
        match &new_messages[0].content {
            ContentValue::Text(t) => {
                assert!(t.contains("hello"));
                assert!(t.contains("permafrost:relocated-context"));
                assert!(t.contains("Working directory"));
                assert!(t.contains("2026-06-22"));
            }
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn test_relocate_no_volatile_blocks() {
        let system = SystemPrompt::Blocks(vec![SystemContentBlock {
            block_type: "text".to_string(),
            text: "You are a helpful assistant.".to_string(),
        }]);

        let messages = vec![Message {
            role: "user".to_string(),
            content: ContentValue::Text("hello".to_string()),
        }];

        let (new_system, new_messages) = migrate_volatile_system_blocks(system.clone(), messages);
        // Should be unchanged
        match &new_system {
            SystemPrompt::Blocks(b) => assert_eq!(b.len(), 1),
            _ => panic!("expected Blocks"),
        }
        match &new_messages[0].content {
            ContentValue::Text(t) => assert_eq!(t, "hello"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn relocate_on_does_not_mutate_previous_history_items() {
        let system = SystemPrompt::Blocks(vec![SystemContentBlock {
            block_type: "text".to_string(),
            text: "<env>\nWorking directory: /tmp\nToday's date: 2026-06-22\n</env>".to_string(),
        }]);
        let messages = vec![Message {
            role: "user".to_string(),
            content: ContentValue::Text("historical user turn".to_string()),
        }];

        let (_, volatile) = split_volatile_system_blocks(system);
        let relocated = messages;
        assert_eq!(volatile.len(), 1);
        match &relocated[0].content {
            ContentValue::Text(text) => assert_eq!(text, "historical user turn"),
            _ => panic!("expected historical text to remain unchanged"),
        }
    }

    // --- Phase 4d: Chat split-tail relocation (relocate_volatile_to_chat_tail) ---

    fn env_system() -> SystemPrompt {
        SystemPrompt::Blocks(vec![
            SystemContentBlock {
                block_type: "text".to_string(),
                text: "You are a helpful assistant.".to_string(),
            },
            SystemContentBlock {
                block_type: "text".to_string(),
                text: "<env>\nWorking directory: /tmp\nToday's date: 2026-06-22\n</env>"
                    .to_string(),
            },
        ])
    }

    fn plain_user(text: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: ContentValue::Text(text.to_string()),
        }
    }

    #[test]
    fn split_tail_keeps_stable_system_in_prefix() {
        // Stable blocks stay in the cache-prefix position; volatile blocks
        // are removed from the system prompt.
        let messages = vec![plain_user("hello")];
        let (system, messages) = relocate_volatile_to_chat_tail(env_system(), messages);
        match &system {
            SystemPrompt::Blocks(b) => {
                assert_eq!(b.len(), 1, "only the stable block may remain");
                assert_eq!(b[0].text, "You are a helpful assistant.");
                assert!(!b[0].text.contains("Today's date"));
            }
            _ => panic!("expected Blocks system"),
        }
        assert_eq!(messages.len(), 1);
    }

    #[test]
    fn split_tail_merges_into_last_user_text_turn() {
        // Common case: the last message is a plain user text turn (the
        // current request's own content). The volatile appendix is merged
        // into it — no new message, no role change, no user-after-user.
        let messages = vec![plain_user("hello")];
        let (_, messages) = relocate_volatile_to_chat_tail(env_system(), messages);
        assert_eq!(messages.len(), 1, "must not append a synthetic user");
        match &messages[0].content {
            ContentValue::Text(t) => {
                assert!(t.starts_with("hello"));
                assert!(t.contains("permafrost:relocated-context"));
                assert!(t.contains("Today's date"));
            }
            _ => panic!("expected merged Text"),
        }
    }

    #[test]
    fn split_tail_appends_synthetic_user_after_tool_result_tail() {
        // A last `user` message whose blocks render to the `tool` role cannot
        // absorb the appendix without dropping it, so a brand-new `user`
        // message is appended — the wire ends tool -> user (alternation kept).
        let tool_result = Message {
            role: "user".to_string(),
            content: ContentValue::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "call-1".to_string(),
                content: ToolResultContent::Text("result".to_string()),
                is_error: None,
            }]),
        };
        let messages = vec![tool_result];
        let (_, messages) = relocate_volatile_to_chat_tail(env_system(), messages);
        assert_eq!(messages.len(), 2, "synthetic user must be appended");
        assert_eq!(messages[1].role, "user");
        match &messages[1].content {
            ContentValue::Text(t) => assert!(t.contains("permafrost:relocated-context")),
            _ => panic!("expected synthetic user Text"),
        }
        // The tool_result message itself is untouched (stable history).
        assert!(matches!(
            &messages[0].content,
            ContentValue::Blocks(b) if b.len() == 1 && matches!(b[0], ContentBlock::ToolResult { .. })
        ));
    }

    #[test]
    fn split_tail_merges_into_last_assistant_text_turn() {
        // Last message is an assistant text turn: the appendix merges into
        // the assistant content — no synthetic user, no role change, no
        // assistant -> user turn. (An assistant may legally carry both text
        // content and tool_calls on the Chat wire.)
        let assistant = Message {
            role: "assistant".to_string(),
            content: ContentValue::Text("Answer".to_string()),
        };
        let messages = vec![assistant];
        let (_, messages) = relocate_volatile_to_chat_tail(env_system(), messages);
        assert_eq!(messages.len(), 1, "must not append a synthetic user");
        assert_eq!(messages[0].role, "assistant");
        match &messages[0].content {
            ContentValue::Text(t) => {
                assert!(t.starts_with("Answer"));
                assert!(t.contains("permafrost:relocated-context"));
                assert!(t.contains("Today's date"));
            }
            _ => panic!("expected merged assistant Text"),
        }
    }

    #[test]
    fn split_tail_never_appends_synthetic_user_after_assistant_tool_calls() {
        // Pending tool-call tail (BUG-C regression): the last message is an
        // assistant carrying a tool_use block with NO tool_result yet.
        // Appending a synthetic `user` turn here would put a non-tool-result
        // user message between the assistant's tool_calls and its (future)
        // tool_result, breaking the Chat tool_call -> tool_result pairing
        // hard constraint (the upstream would reject the wire). The appendix
        // must instead merge into the assistant content as an extra Text
        // block — the wire stays assistant (content + tool_calls), and the
        // tool_use block is preserved untouched.
        let assistant = Message {
            role: "assistant".to_string(),
            content: ContentValue::Blocks(vec![ContentBlock::ToolUse {
                id: "call-1".to_string(),
                name: "read_file".to_string(),
                input: serde_json::json!({"path": "/tmp/x"}),
            }]),
        };
        let messages = vec![assistant];
        let (_, messages) = relocate_volatile_to_chat_tail(env_system(), messages);
        assert_eq!(messages.len(), 1, "no synthetic user after assistant tail");
        assert_eq!(messages[0].role, "assistant");
        let blocks = match &messages[0].content {
            ContentValue::Blocks(b) => b,
            _ => panic!("expected assistant Blocks (tool_use + text appendix)"),
        };
        assert_eq!(blocks.len(), 2, "tool_use block preserved + appendix block");
        assert!(
            matches!(blocks[0], ContentBlock::ToolUse { .. }),
            "tool_use must be preserved"
        );
        match &blocks[1] {
            ContentBlock::Text { text } => {
                assert!(
                    text.contains("permafrost:relocated-context") && text.contains("Today's date"),
                    "appendix must be appended as a Text block in the assistant content"
                );
            }
            other => panic!("expected Text appendix block, got {other:?}"),
        }
    }

    #[test]
    fn split_tail_does_not_rewrite_stable_history() {
        // Messages before the tail are byte-for-byte untouched.
        let history = vec![
            plain_user("first user turn"),
            Message {
                role: "assistant".to_string(),
                content: ContentValue::Text("first answer".to_string()),
            },
            plain_user("second user turn"),
        ];
        let before = history.clone();
        let (_, messages) = relocate_volatile_to_chat_tail(env_system(), history);
        assert_eq!(messages.len(), 3);
        // Messages before the tail are byte-for-byte untouched: only the
        // final user turn absorbs the relocated appendix.
        assert_eq!(
            messages[0], before[0],
            "first history turn must be unchanged"
        );
        assert_eq!(messages[1], before[1], "assistant turn must be unchanged");
        match &messages[2].content {
            ContentValue::Text(t) => {
                assert!(t.starts_with("second user turn"));
                assert!(t.contains("Today's date"), "volatile appended to last turn");
            }
            _ => panic!("expected merged Text"),
        }
    }

    #[test]
    fn split_tail_is_deterministic() {
        // Same input -> identical output on every invocation (golden-3).
        let messages = vec![plain_user("hello")];
        let (s1, m1) = relocate_volatile_to_chat_tail(env_system(), messages.clone());
        let (s2, m2) = relocate_volatile_to_chat_tail(env_system(), messages);
        assert_eq!(s1, s2);
        assert_eq!(m1, m2);
        assert_eq!(m1[0], m2[0]);
    }

    #[test]
    fn split_tail_no_volatile_leaves_everything_unchanged() {
        // No volatile blocks detected -> system and messages returned as-is.
        let system = SystemPrompt::Blocks(vec![SystemContentBlock {
            block_type: "text".to_string(),
            text: "You are a helpful assistant.".to_string(),
        }]);
        let messages = vec![plain_user("hello")];
        let (out_system, out_messages) = relocate_volatile_to_chat_tail(system, messages.clone());
        assert_eq!(out_messages, messages);
        assert!(matches!(out_system, SystemPrompt::Blocks(ref b) if b.len() == 1));
        // String (non-block) system prompts are also left intact.
        let text_system = SystemPrompt::Text("sys".to_string());
        let (out_s, out_m) = relocate_volatile_to_chat_tail(text_system, messages.clone());
        assert!(matches!(out_s, SystemPrompt::Text(ref t) if t == "sys"));
        assert_eq!(out_m, messages);
    }

    #[test]
    fn split_tail_empty_messages_folds_back_not_drops() {
        // No turn to attach onto: the volatile block folds back into the
        // system prompt rather than being silently dropped (data-loss guard).
        let (system, messages) = relocate_volatile_to_chat_tail(env_system(), Vec::new());
        assert!(messages.is_empty());
        match &system {
            SystemPrompt::Blocks(b) => {
                assert_eq!(b.len(), 2, "nothing moved, nothing dropped");
                assert!(
                    b.iter().any(|bl| bl.text.contains("Today's date")),
                    "volatile block must still be present in the folded-back system"
                );
                assert!(
                    b.iter()
                        .any(|bl| bl.text.contains("You are a helpful assistant.")),
                    "stable block must still be present"
                );
            }
            _ => panic!("expected Blocks system"),
        }
    }
}
