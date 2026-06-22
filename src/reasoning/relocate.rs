// Copyright (c) 2025 codewhale-proxy
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

use crate::anthropic::types::{Message, SystemPrompt};

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
        Regex::new(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
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
            let new_text = re_cch().replace(&block.text, "${1}codewhale-proxy").to_string();
            if new_text != block.text {
                block.text = new_text;
                count += 1;
            }
        }
    }

    if count > 0 {
        tracing::info!(metadata_stabilized = count, "stabilize_metadata: pinned billing nonce");
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
            existing_blocks.push(crate::anthropic::types::ContentBlock::Text {
                text: appendix,
            });
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::types::{ContentValue, SystemContentBlock};

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
}