// Copyright (c) 2025 cc-proxy
// Ported from CodeWhale prefix_cache.rs (534 lines)
// Reference: https://github.com/Hmbown/CodeWhale
//
// F6 (simplified): Per-request prefix fingerprint computation
// for KV cache observability. No cross-request state — external
// monitoring systems can aggregate and analyse fingerprints.

use crate::openai::types::OpenAiTool;
use crate::schema::canonical_hash;
use serde_json::Value;
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

/// v2 fingerprint: full canonical prefix content (system + messages + full tool
/// schemas including description and parameters).
///
/// Observational metric only. v1 (`compute_prefix_fingerprint`) and its callers
/// stay byte-for-byte compatible; the `"version": "v2"` marker is hashed into
/// the output so v2 values can never be confused with v1 fingerprints or
/// silently reinterpret historical metrics. Object keys are canonicalized;
/// array order (messages, tools) is semantic and is preserved, never sorted.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "observational full-prefix fingerprint; wired in a later phase"
    )
)]
pub fn compute_prefix_fingerprint_v2(
    system_prompt: Option<&str>,
    messages: Option<&[Value]>,
    tools: Option<&[OpenAiTool]>,
) -> String {
    let tools_json = tools.map(|items| items.iter().map(tool_to_json).collect::<Vec<_>>());
    let content = serde_json::json!({
        "version": "v2",
        "system": system_prompt.unwrap_or(""),
        "messages": messages.unwrap_or(&[]),
        "tools": tools_json,
    });
    canonical_hash(&content)
}

/// Full tool schema as JSON (name, description, parameters) so schema-content
/// drift is observable, unlike v1 which hashes only sorted tool names.
fn tool_to_json(tool: &OpenAiTool) -> Value {
    serde_json::json!({
        "type": tool.tool_type,
        "function": {
            "name": tool.function.name,
            "description": tool.function.description,
            "parameters": tool.function.parameters,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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

    fn tool(name: &str, description: Option<&str>, parameters: Value) -> OpenAiTool {
        OpenAiTool {
            tool_type: "function".to_string(),
            function: crate::openai::types::OpenAiFunction {
                name: name.to_string(),
                description: description.map(str::to_string),
                parameters,
            },
        }
    }

    #[test]
    fn v2_is_deterministic() {
        let tools = [tool(
            "read_file",
            Some("reads a file"),
            json!({"type": "object"}),
        )];
        let messages = [json!({"role": "user", "content": "hi"})];
        let a = compute_prefix_fingerprint_v2(Some("sys"), Some(&messages), Some(&tools));
        let b = compute_prefix_fingerprint_v2(Some("sys"), Some(&messages), Some(&tools));
        assert_eq!(a, b);
    }

    #[test]
    fn v2_detects_tool_schema_content_drift_that_v1_misses() {
        // v1 hashes only sorted tool NAMES, so two tools sharing a name but with
        // different `parameters` schemas produce the same v1 fingerprint. v2 hashes
        // the full schema and must tell them apart.
        let tools_a = [tool(
            "lookup",
            None,
            json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        )];
        let tools_b = [tool(
            "lookup",
            None,
            json!({"type": "object", "properties": {"limit": {"type": "integer"}}}),
        )];
        assert_eq!(
            compute_prefix_fingerprint("sys", Some(&tools_a)),
            compute_prefix_fingerprint("sys", Some(&tools_b)),
            "v1 cannot see schema content"
        );
        assert_ne!(
            compute_prefix_fingerprint_v2(Some("sys"), None, Some(&tools_a)),
            compute_prefix_fingerprint_v2(Some("sys"), None, Some(&tools_b)),
            "v2 must detect schema content drift"
        );
    }

    #[test]
    fn v2_detects_tool_description_change() {
        let tools_a = [tool("lookup", Some("fetch a record"), json!({}))];
        let tools_b = [tool("lookup", Some("fetch TWO records"), json!({}))];
        assert_ne!(
            compute_prefix_fingerprint_v2(Some("sys"), None, Some(&tools_a)),
            compute_prefix_fingerprint_v2(Some("sys"), None, Some(&tools_b))
        );
    }

    #[test]
    fn v2_includes_messages_content() {
        let messages_a = [json!({"role": "user", "content": "hello"})];
        let messages_b = [json!({"role": "user", "content": "hello world"})];
        assert_ne!(
            compute_prefix_fingerprint_v2(Some("sys"), Some(&messages_a), None),
            compute_prefix_fingerprint_v2(Some("sys"), Some(&messages_b), None)
        );
    }

    #[test]
    fn v2_preserves_semantic_tool_order() {
        // Array order is semantic and preserved by v2 (never sorted).
        let a = [
            tool("read", None, json!({})),
            tool("write", None, json!({})),
        ];
        let b = [
            tool("write", None, json!({})),
            tool("read", None, json!({})),
        ];
        assert_ne!(
            compute_prefix_fingerprint_v2(Some("sys"), None, Some(&a)),
            compute_prefix_fingerprint_v2(Some("sys"), None, Some(&b)),
            "reordered tools are a different prefix"
        );
    }

    #[test]
    fn v2_values_are_never_confusable_with_v1() {
        // The version marker guarantees v2 output can't silently reinterpret
        // historical v1 metrics.
        let tools = [tool("read", None, json!({}))];
        let v1 = compute_prefix_fingerprint("sys", Some(&tools));
        let v2 = compute_prefix_fingerprint_v2(Some("sys"), None, Some(&tools));
        assert_ne!(v1, v2);
    }
}
