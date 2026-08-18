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

/// Tolerant single-field delta value (`content`, `reasoning_content`,
/// `reasoning`) used by [`ChatDelta`].
///
/// OpenAI's standard streaming delta carries a plain string. kimi-k3-class
/// upstreams (eswitch → clawbot) send an array of content parts
/// (`[{"type":"text","text":...}]`) or reasoning parts
/// (`[{"type":"reasoning_summary_text","summary_text":...}]`), and some send a
/// bare object wrapper. All of these MUST deserialize WITHOUT error so the
/// `serde_json::from_value::<ChatDelta>(...).ok()` gate in `src/sse/stream.rs`
/// never drops a content-bearing chunk (commit 156d3ad documented the same
/// failure class: "invalid type: sequence, expected a string").
///
/// Anything that carries no extractable text is retained verbatim in `raw` so
/// callers can surface it (structured warning/error) instead of silently
/// losing content — a stream must never synthesize a clean `message_stop` over
/// dropped content.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DeltaContentValue {
    /// Concatenated logical text (string form, or the joined `text`/
    /// `summary_text` of array parts).
    pub text: Option<String>,
    /// Wire value(s) with no extractable text, retained verbatim for
    /// observability. `Some` means this field carried a shape the converter
    /// could not map to text.
    pub raw: Option<Value>,
}

impl DeltaContentValue {
    fn from_value(value: Value) -> Self {
        match value {
            Value::String(s) => Self {
                text: Some(s),
                raw: None,
            },
            Value::Null => Self::default(),
            Value::Array(parts) => {
                let mut text = String::new();
                let mut raw: Option<Value> = None;
                for part in parts {
                    match part {
                        Value::String(s) => text.push_str(&s),
                        Value::Object(map) => {
                            // content part: {"type":"text","text":...}
                            // reasoning part: {"type":"reasoning_summary_text","summary_text":...}
                            let extract = map
                                .get("text")
                                .and_then(Value::as_str)
                                .or_else(|| map.get("summary_text").and_then(Value::as_str));
                            match extract {
                                Some(s) => text.push_str(s),
                                None => {
                                    raw.get_or_insert(Value::Object(map));
                                }
                            }
                        }
                        other => {
                            raw.get_or_insert(other);
                        }
                    }
                }
                let text = if text.is_empty() { None } else { Some(text) };
                Self { text, raw }
            }
            // Bare object wrapper: {"text":...} or {"summary_text":...}.
            Value::Object(map) => {
                let extract = map
                    .get("text")
                    .and_then(Value::as_str)
                    .or_else(|| map.get("summary_text").and_then(Value::as_str));
                match extract {
                    Some(s) => Self {
                        text: Some(s.to_string()),
                        raw: None,
                    },
                    None => Self {
                        text: None,
                        raw: Some(Value::Object(map)),
                    },
                }
            }
            other => Self {
                text: None,
                raw: Some(other),
            },
        }
    }

    /// True when this field carried neither text nor unparseable content
    /// (equivalent to a missing/empty field).
    pub fn is_empty(&self) -> bool {
        self.text.is_none() && self.raw.is_none()
    }
}

impl From<String> for DeltaContentValue {
    fn from(s: String) -> Self {
        Self {
            text: Some(s),
            raw: None,
        }
    }
}

impl From<Option<String>> for DeltaContentValue {
    fn from(s: Option<String>) -> Self {
        match s {
            Some(s) => Self::from(s),
            None => Self::default(),
        }
    }
}

impl<'de> serde::Deserialize<'de> for DeltaContentValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self::from_value(Value::deserialize(deserializer)?))
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ChatDelta {
    pub role: Option<String>,
    #[serde(default)]
    pub content: DeltaContentValue,
    #[serde(default)]
    pub reasoning_content: DeltaContentValue,
    /// kimi-k3 uses "reasoning" instead of "reasoning_content"
    #[serde(default)]
    pub reasoning: DeltaContentValue,
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
            "reasoning_content" => self.reasoning_content.text.as_deref(),
            "reasoning" => self.reasoning.text.as_deref(),
            _ => None,
        }
    }

    /// True when any content field carried data that could not be mapped to
    /// text. Callers must surface this rather than silently continue — a clean
    /// `message_stop` over dropped content is the exact defect this type
    /// prevents.
    pub fn has_unparseable_content(&self) -> bool {
        self.content.raw.is_some()
            || self.reasoning_content.raw.is_some()
            || self.reasoning.raw.is_some()
    }

    /// True when this delta yields no text from any content field.
    pub fn has_no_text(&self) -> bool {
        self.content.text.is_none()
            && self.reasoning_content.text.is_none()
            && self.reasoning.text.is_none()
    }

    /// A short, safe preview of any unparseable content (for logging/errors),
    /// capped to avoid dumping large payloads.
    pub fn unparseable_preview(&self) -> String {
        let joined = [&self.content, &self.reasoning_content, &self.reasoning]
            .iter()
            .filter_map(|v| v.raw.as_ref())
            .map(|v| serde_json::to_string(v).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("; ");
        if joined.len() > 500 {
            // Char-boundary-safe truncation: slicing `&joined[..500]` at a raw
            // byte index panics when the upstream-controlled raw content holds
            // multi-byte UTF-8 ("end byte index 500 is not a char boundary").
            // That panic fires on the fail-closed error path and drops the SSE
            // channel, silently truncating the stream. Cap by characters, never
            // by raw bytes.
            let truncated: String = joined.chars().take(500).collect();
            format!("{truncated}...[truncated]")
        } else {
            joined
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
            content: Default::default(),
            reasoning_content: Default::default(),
            reasoning: Some("stream thinking".to_string()).into(),
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

    // ================= STREAM DELTA SILENT-LOSS REGRESSION (RED) =============
    // Root cause (report 84): `src/sse/stream.rs` gates every upstream delta
    // through `serde_json::from_value::<ChatDelta>(delta).ok()`. kimi-k3-class
    // upstreams (eswitch → clawbot) emit `content`/`reasoning` as an array of
    // content parts (commit 156d3ad documents the same failure class:
    // "invalid type: sequence, expected a string"). Because `ChatDelta`'s
    // content fields are `Option<String>`, those deltas fail deserialization,
    // `.ok()` turns the error into `None`, `process_delta` is never called, and
    // the chunk contributes ZERO content frames with no log/warn.

    #[test]
    fn chat_delta_array_content_deserializes() {
        // kimi-k3-class upstream: `content` arrives as an array of parts.
        // Pre-fix this fails `ChatDelta` deserialize → `.ok()` → None → silent
        // content loss (Phase5B symptom: 0 content blocks, only terminal).
        let d: ChatDelta = serde_json::from_value(serde_json::json!({
            "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}]
        }))
        .expect("array-valued content MUST deserialize (regression: silent drop)");
        let _ = d;
    }

    #[test]
    fn chat_delta_reasoning_array_deserializes() {
        // kimi thinking deltas arrive as `reasoning` part arrays; these must
        // also deserialize so thinking is not silently dropped either.
        let d: ChatDelta = serde_json::from_value(serde_json::json!({
            "reasoning": [{"type": "reasoning_summary_text", "summary_text": "think"}]
        }))
        .expect("array-valued reasoning MUST deserialize (regression: silent drop)");
        let _ = d;
    }

    // ----- GREEN: tolerant shapes + unknown-shape observability -----

    #[test]
    fn chat_delta_array_content_text_is_concatenated() {
        // Array parts join into a single logical text value.
        let d: ChatDelta = serde_json::from_value(serde_json::json!({
            "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}]
        }))
        .unwrap();
        assert_eq!(d.content.text.as_deref(), Some("Hello world"));
        assert!(d.content.raw.is_none(), "fully decoded, no raw retained");
        assert!(!d.has_unparseable_content());
        assert!(!d.has_no_text());
    }

    #[test]
    fn chat_delta_parses_object_content_wrapper() {
        // Some upstreams wrap a single part in an object.
        let d: ChatDelta = serde_json::from_value(serde_json::json!({
            "content": {"type": "text", "text": "Hello"}
        }))
        .unwrap();
        assert_eq!(d.content.text.as_deref(), Some("Hello"));
        assert!(d.content.raw.is_none());
    }

    #[test]
    fn chat_delta_parses_null_content() {
        // null content is equivalent to an absent/empty field (tool-only chunks).
        let d: ChatDelta = serde_json::from_value(serde_json::json!({
            "content": null, "role": "assistant"
        }))
        .unwrap();
        assert!(d.content.is_empty());
        assert_eq!(d.role.as_deref(), Some("assistant"));
    }

    #[test]
    fn chat_delta_string_form_unchanged() {
        // The standard OpenAI string form still deserializes and reads as text.
        let d: ChatDelta = serde_json::from_value(serde_json::json!({
            "content": "Hello"
        }))
        .unwrap();
        assert_eq!(d.content.text.as_deref(), Some("Hello"));
        assert!(!d.has_unparseable_content());
    }

    #[test]
    fn chat_delta_reasoning_array_text_and_field_routing() {
        // Reasoning part arrays decode into the reasoning field only (never the
        // content field), so thinking is never misread as plain text.
        let d: ChatDelta = serde_json::from_value(serde_json::json!({
            "reasoning": [{"type": "reasoning_summary_text", "summary_text": "let me think"}],
            "content": [{"type": "text", "text": "answer"}]
        }))
        .unwrap();
        assert!(d.content.text.is_none() || d.content.text.as_deref() != Some("let me think"));
        assert_eq!(d.reasoning.text.as_deref(), Some("let me think"));
        assert_eq!(d.get_reasoning("reasoning", &[]), Some("let me think"));
        assert_eq!(d.content.text.as_deref(), Some("answer"));
    }

    #[test]
    fn chat_delta_unknown_scalar_content_is_observable_not_lost() {
        // An unrecognized shape must not fail deserialization (which would drop
        // the whole chunk via `.ok()`); it is retained so callers can surface it.
        let d: ChatDelta = serde_json::from_value(serde_json::json!({
            "content": 12345
        }))
        .expect("unknown content shape still deserializes (observable, not dropped)");
        assert!(d.content.text.is_none());
        assert!(d.has_unparseable_content());
        assert!(d.has_no_text());
        assert!(
            d.unparseable_preview().contains("12345"),
            "preview exposes the unknown value: {}",
            d.unparseable_preview()
        );
    }

    #[test]
    fn chat_delta_unknown_object_content_is_observable() {
        let d: ChatDelta = serde_json::from_value(serde_json::json!({
            "content": {"weird": "shape"}
        }))
        .expect("unknown object content still deserializes (observable, not dropped)");
        assert!(d.content.text.is_none());
        assert!(d.has_unparseable_content());
        assert!(d.has_no_text());
    }

    #[test]
    fn chat_delta_unparseable_preview_multibyte_does_not_panic() {
        // F2 regression: `unparseable_preview()` sliced `&joined[..500]` at a
        // raw byte index. When the upstream-controlled raw content contains
        // multi-byte UTF-8 (here: 300 × 3-byte CJK chars = 900 bytes), byte 500
        // lands mid-character and the slice PANICS ("end byte index 500 is not
        // a char boundary") on the exact fail-closed error path — turning the
        // intended "surface it loudly" into a silently truncated stream.
        // Char-safe truncation must not panic, must keep the length cap and
        // still mark the preview as truncated.
        let long = "中".repeat(300); // 900 bytes, far past the 500 cap
        let d: ChatDelta = serde_json::from_value(serde_json::json!({
            "content": {"weird": long}
        }))
        .unwrap();
        let preview = d.unparseable_preview();
        assert!(
            preview.ends_with("...[truncated]"),
            "preview must be marked truncated, got {preview:?}"
        );
        assert!(
            preview.chars().count() <= 500 + "[truncated]".len(),
            "preview stays bounded: {} chars",
            preview.chars().count()
        );
        assert!(
            preview.contains("中"),
            "multi-byte content survives truncation intact: {preview:?}"
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
