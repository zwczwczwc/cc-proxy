// Copyright (c) 2025 cc-proxy
// Ported from CodeWhale prefix_cache.rs (534 lines)
// Reference: https://github.com/Hmbown/CodeWhale
//
// F6 (simplified): Per-request prefix fingerprint computation
// for KV cache observability. No cross-request state — external
// monitoring systems can aggregate and analyse fingerprints.

use crate::openai::types::OpenAiTool;
use sha2::{Digest, Sha256};

/// Computes a SHA-256 double-hash fingerprint of the system prompt and tools.
///
/// Reference: CodeWhale prefix_cache.rs — compute_prefix_fingerprint.
/// SHA-256 is used for collision resistance; double-hash for extra stability.
/// Tools are explicitly sorted by name to eliminate ordering differences
/// (Reference: CodeWhale prefix_cache.rs:64-66).
pub fn compute_prefix_fingerprint(system_prompt: &str, tools: Option<&[OpenAiTool]>) -> String {
    let mut hasher = Sha256::new();

    // Hash the system prompt
    hasher.update(system_prompt.as_bytes());

    // Hash tools (sorted by name for deterministic output)
    if let Some(tools) = tools {
        // Collect tool names and sort them (reference: prefix_cache.rs:64-66)
        let mut tool_names: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
        tool_names.sort();

        for name in &tool_names {
            hasher.update(name.as_bytes());
        }
    }

    // Double hash for extra stability
    let first = hasher.finalize();
    let mut second = Sha256::new();
    second.update(first);
    let result = second.finalize();

    // Truncate to 16 hex chars for readability
    let result_bytes = result.as_slice();
    hex::encode(&result_bytes[..8])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_prefix_fingerprint_deterministic() {
        let fp1 = compute_prefix_fingerprint("You are helpful", None);
        let fp2 = compute_prefix_fingerprint("You are helpful", None);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_compute_prefix_fingerprint_different_system() {
        let fp1 = compute_prefix_fingerprint("You are helpful", None);
        let fp2 = compute_prefix_fingerprint("You are an expert", None);
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn test_compute_prefix_fingerprint_with_tools() {
        let tools = vec![
            OpenAiTool {
                tool_type: "function".to_string(),
                function: crate::openai::types::OpenAiFunction {
                    name: "read_file".to_string(),
                    description: None,
                    parameters: serde_json::json!({}),
                },
            },
            OpenAiTool {
                tool_type: "function".to_string(),
                function: crate::openai::types::OpenAiFunction {
                    name: "write_file".to_string(),
                    description: None,
                    parameters: serde_json::json!({}),
                },
            },
        ];

        let fp1 = compute_prefix_fingerprint("sys", Some(&tools));
        let fp2 = compute_prefix_fingerprint("sys", Some(&tools));
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_compute_prefix_fingerprint_tool_order_independent() {
        let tools_a = vec![
            OpenAiTool {
                tool_type: "function".to_string(),
                function: crate::openai::types::OpenAiFunction {
                    name: "write_file".to_string(),
                    description: None,
                    parameters: serde_json::json!({}),
                },
            },
            OpenAiTool {
                tool_type: "function".to_string(),
                function: crate::openai::types::OpenAiFunction {
                    name: "read_file".to_string(),
                    description: None,
                    parameters: serde_json::json!({}),
                },
            },
        ];

        let tools_b = vec![
            OpenAiTool {
                tool_type: "function".to_string(),
                function: crate::openai::types::OpenAiFunction {
                    name: "read_file".to_string(),
                    description: None,
                    parameters: serde_json::json!({}),
                },
            },
            OpenAiTool {
                tool_type: "function".to_string(),
                function: crate::openai::types::OpenAiFunction {
                    name: "write_file".to_string(),
                    description: None,
                    parameters: serde_json::json!({}),
                },
            },
        ];

        let fp1 = compute_prefix_fingerprint("sys", Some(&tools_a));
        let fp2 = compute_prefix_fingerprint("sys", Some(&tools_b));
        assert_eq!(fp1, fp2, "Tool order should not affect fingerprint");
    }
}
