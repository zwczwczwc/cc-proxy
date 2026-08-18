use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Value>,
    pub max_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<DeepSeekThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Optional session cache key (Kimi `prompt_cache_key`).
    ///
    /// Phase 2b.2 scope: field + serde contract only — the value is always
    /// `None` (the only literal in the repo sets it to `None`), so the field
    /// is omitted from the wire and every per-wire golden stays byte-identical.
    /// Phase 3 wires `cache::session_key_from_source` here; the Responses
    /// encoder is an explicit non-goal and never carries this field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekThinking {
    #[serde(rename = "type")]
    pub thinking_type: String,
    /// GLM-5.2: preserved thinking mode (clear_thinking=false)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clear_thinking: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAiFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiFunction {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
}

// --- OpenAI Response types ---

#[derive(Debug, Clone, serde::Deserialize)]
#[expect(
    dead_code,
    reason = "response fields preserve upstream protocol compatibility"
)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[expect(
    dead_code,
    reason = "choice fields preserve upstream protocol compatibility"
)]
pub struct Choice {
    pub index: u32,
    pub message: Option<ChatMessage>,
    pub delta: Option<ChatDelta>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[expect(
    dead_code,
    reason = "message fields preserve upstream protocol compatibility"
)]
pub struct ChatMessage {
    pub role: Option<String>,
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    /// kimi-k3 uses "reasoning" instead of "reasoning_content"
    #[serde(default)]
    pub reasoning: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ChatMessage {
    /// Extract reasoning content using the configured field name.
    /// Tries primary field first, then alt fields, then returns None.
    pub fn get_reasoning(&self, field: &str, alt_fields: &[String]) -> Option<&str> {
        // Try primary field
        if let Some(val) = self.field_value(field) {
            if !val.trim().is_empty() {
                return Some(val);
            }
        }
        // Try alt fields
        for alt in alt_fields {
            if let Some(val) = self.field_value(alt) {
                if !val.trim().is_empty() {
                    return Some(val);
                }
            }
        }
        None
    }

    fn field_value(&self, field: &str) -> Option<&str> {
        match field {
            "reasoning_content" => self.reasoning_content.as_deref(),
            "reasoning" => self.reasoning.as_deref(),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ChatDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    /// kimi-k3 uses "reasoning" instead of "reasoning_content"
    #[serde(default)]
    pub reasoning: Option<String>,
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

impl ChatDelta {
    /// Extract reasoning content using the configured field name.
    /// Tries primary field first, then alt fields, then returns None.
    pub fn get_reasoning(&self, field: &str, alt_fields: &[String]) -> Option<&str> {
        // Try primary field
        if let Some(val) = self.field_value(field) {
            if !val.trim().is_empty() {
                return Some(val);
            }
        }
        // Try alt fields
        for alt in alt_fields {
            if let Some(val) = self.field_value(alt) {
                if !val.trim().is_empty() {
                    return Some(val);
                }
            }
        }
        None
    }

    fn field_value(&self, field: &str) -> Option<&str> {
        match field {
            "reasoning_content" => self.reasoning_content.as_deref(),
            "reasoning" => self.reasoning.as_deref(),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[expect(
    dead_code,
    reason = "tool call type preserves upstream protocol compatibility"
)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: Option<String>,
    pub function: Option<FunctionCall>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[expect(
    dead_code,
    reason = "tool call delta fields preserve upstream protocol compatibility"
)]
pub struct ToolCallDelta {
    pub index: Option<u32>,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub call_type: Option<String>,
    pub function: Option<FunctionCallDelta>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct FunctionCallDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[expect(
    dead_code,
    reason = "usage fields preserve upstream protocol compatibility"
)]
pub struct Usage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
    /// DeepSeek: tokens served from KV cache (~98% cheaper)
    #[serde(default)]
    pub prompt_cache_hit_tokens: Option<u32>,
    /// DeepSeek: tokens NOT served from cache (billed at full price)
    #[serde(default)]
    pub prompt_cache_miss_tokens: Option<u32>,
    /// Kimi (GAP-A): optional top-level cached_tokens. Observed, never assumed
    /// to be present upstream; falls back to nested details when absent.
    #[serde(default)]
    pub cached_tokens: Option<u32>,
    /// Standard OpenAI: cached tokens detail
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

// --- Model list response ---

#[derive(Debug, Clone, Serialize)]
#[expect(
    dead_code,
    reason = "model listing types preserve endpoint compatibility"
)]
pub struct ModelListResponse {
    pub data: Vec<ModelInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_reasoning_primary_field() {
        // kimi uses "reasoning" as primary field
        let msg = ChatMessage {
            role: Some("assistant".to_string()),
            content: Some("answer".to_string()),
            reasoning_content: None,
            reasoning: Some("kimi thinking".to_string()),
            tool_calls: None,
        };
        let result = msg.get_reasoning("reasoning", &[]);
        assert_eq!(result, Some("kimi thinking"));
    }

    #[test]
    fn test_get_reasoning_deepseek() {
        // DeepSeek uses "reasoning_content" as primary field
        let msg = ChatMessage {
            role: Some("assistant".to_string()),
            content: Some("answer".to_string()),
            reasoning_content: Some("deepseek thinking".to_string()),
            reasoning: None,
            tool_calls: None,
        };
        let result = msg.get_reasoning("reasoning_content", &[]);
        assert_eq!(result, Some("deepseek thinking"));
    }

    #[test]
    fn test_get_reasoning_empty_primary_uses_alt() {
        // Primary field is empty string, should fall back to alt
        let msg = ChatMessage {
            role: Some("assistant".to_string()),
            content: Some("answer".to_string()),
            reasoning_content: Some("  ".to_string()), // whitespace only = empty
            reasoning: Some("alt thinking".to_string()),
            tool_calls: None,
        };
        let result = msg.get_reasoning("reasoning_content", &["reasoning".to_string()]);
        assert_eq!(result, Some("alt thinking"));
    }

    #[test]
    fn test_get_reasoning_none() {
        let msg = ChatMessage {
            role: Some("assistant".to_string()),
            content: Some("answer".to_string()),
            reasoning_content: None,
            reasoning: None,
            tool_calls: None,
        };
        let result = msg.get_reasoning("reasoning_content", &["reasoning".to_string()]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_chat_delta_get_reasoning() {
        let delta = ChatDelta {
            role: None,
            content: None,
            reasoning_content: None,
            reasoning: Some("stream thinking".to_string()),
            tool_calls: None,
        };
        let result = delta.get_reasoning("reasoning", &[]);
        assert_eq!(result, Some("stream thinking"));
    }

    #[test]
    fn test_usage_parses_top_level_cached_tokens() {
        // Kimi GAP-A: top-level `cached_tokens` is optional but must be read
        // when present without breaking deserialization of the nested fields.
        let usage: Usage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cached_tokens": 70,
            "prompt_tokens_details": {"cached_tokens": 70},
        }))
        .unwrap();
        assert_eq!(usage.cached_tokens, Some(70));
        assert_eq!(
            usage.prompt_tokens_details.as_ref().unwrap().cached_tokens,
            Some(70)
        );
    }

    #[test]
    fn test_usage_absent_top_level_cached_tokens_is_none() {
        // Upstream is never assumed to return top-level cached_tokens.
        let usage: Usage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }))
        .unwrap();
        assert_eq!(usage.cached_tokens, None);
        assert!(usage.prompt_tokens_details.is_none());
    }

    #[test]
    fn test_usage_parses_deepseek_hit_and_miss_fields() {
        let usage: Usage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_cache_hit_tokens": 60,
            "prompt_cache_miss_tokens": 40,
        }))
        .unwrap();
        assert_eq!(usage.prompt_cache_hit_tokens, Some(60));
        assert_eq!(usage.prompt_cache_miss_tokens, Some(40));
        assert_eq!(usage.cached_tokens, None);
    }

    // --- Phase 2b.2: Chat `prompt_cache_key` serde contract (T16) ---

    fn chat_request(prompt_cache_key: Option<String>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "kimi-k3-turbo".to_string(),
            messages: vec![serde_json::json!({"role": "user", "content": "hi"})],
            max_tokens: None,
            max_completion_tokens: None,
            stream: None,
            stream_options: None,
            temperature: None,
            top_p: None,
            reasoning_effort: None,
            thinking: None,
            tools: None,
            tool_choice: None,
            stop: None,
            prompt_cache_key,
        }
    }

    #[test]
    fn chat_request_none_prompt_cache_key_is_omitted_from_wire() {
        // T16 (MUST, serde): with `prompt_cache_key: None` the field must not
        // appear on the serialized wire at all (fail-closed; nothing injected).
        let value = serde_json::to_value(chat_request(None)).unwrap();
        assert!(
            !value.as_object().unwrap().contains_key("prompt_cache_key"),
            "None must omit the field from the wire"
        );
        // A missing field on the wire round-trips back to None.
        let back: ChatCompletionRequest = serde_json::from_value(value).unwrap();
        assert_eq!(back.prompt_cache_key, None);
    }

    #[test]
    fn chat_request_some_prompt_cache_key_serializes_field() {
        // Isolated unit contract: when a later phase (Phase 3) injects a key,
        // the field serializes as a plain string. Not wired into any
        // production path — the only repo literal sets it to None.
        let key = "0123456789abcdef0123456789abcdef".to_string();
        let value = serde_json::to_value(chat_request(Some(key.clone()))).unwrap();
        assert_eq!(
            value.get("prompt_cache_key").and_then(|v| v.as_str()),
            Some(key.as_str())
        );
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}
