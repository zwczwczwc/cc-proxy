use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Anthropic Messages API request type
#[derive(Debug, Clone, Deserialize)]
#[expect(
    dead_code,
    reason = "request fields are deserialized for protocol compatibility"
)]
pub struct MessagesRequest {
    pub model: String,
    pub system: Option<SystemPrompt>,
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    #[serde(default)]
    pub stream: Option<bool>,
    pub thinking: Option<ThinkingConfig>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub metadata: Option<Metadata>,
    pub stop_sequences: Option<Vec<String>>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<SystemContentBlock>),
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SystemContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: ContentValue,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ContentValueInner {
    Text(String),
    Blocks(Vec<ContentBlock>),
    /// Catch-all: captures any value that Text and Blocks cannot handle.
    /// Enables logging the actual value instead of failing with 422.
    Raw(serde_json::Value),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentValue {
    Text(String),
    Blocks(Vec<ContentBlock>),
    /// Anthropic API allows `null` content for assistant messages with only tool_calls.
    Null,
}

impl<'de> Deserialize<'de> for ContentValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let inner = ContentValueInner::deserialize(deserializer)?;
        Ok(match inner {
            ContentValueInner::Text(s) => ContentValue::Text(s),
            ContentValueInner::Blocks(b) => ContentValue::Blocks(b),
            ContentValueInner::Raw(v) => {
                // Log the unexpected value so we can see the actual root cause
                let preview = format!("{}", v);
                let preview = if preview.len() > 500 {
                    &preview[..500]
                } else {
                    &preview
                };
                tracing::warn!(
                    raw_type = if v.is_null() { "null" } else if v.is_string() { "string" } else if v.is_array() { "array" } else if v.is_object() { "object" } else if v.is_number() { "number" } else if v.is_boolean() { "boolean" } else { "unknown" },
                    raw_preview = %preview,
                    "ContentValue::Raw: unexpected content format, treating as Null"
                );
                ContentValue::Null
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: ToolResultContent,
        #[serde(default)]
        is_error: Option<bool>,
    },
    #[serde(rename = "image")]
    Image { source: ImageSource },
    /// Catch-all for unknown content block types (e.g., server_tool_use, search_result)
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ToolResultContentBlock>),
}

/// A single content block inside a tool_result.
/// Anthropic API supports text and image blocks in tool results.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "type")]
pub enum ToolResultContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: ImageSource },
    /// Catch-all for unknown block types in tool results
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Deserialize)]
#[expect(
    dead_code,
    reason = "thinking configuration fields are deserialized for protocol compatibility"
)]
#[serde(untagged)]
pub enum ThinkingConfig {
    Enabled {
        #[serde(rename = "type")]
        config_type: String,
        budget_tokens: Option<u32>,
        display: Option<String>,
    },
    Disabled {
        #[serde(rename = "type")]
        config_type: String,
    },
    Adaptive {
        #[serde(rename = "type")]
        config_type: String,
        display: Option<String>,
    },
}

impl ThinkingConfig {
    /// Semantic check driven by the wire `type` string, NOT the untagged variant.
    ///
    /// `ThinkingConfig` is `#[serde(untagged)]` with `Enabled` declared first and
    /// every field optional, so serde deserializes ANY thinking object — including
    /// `{"type":"disabled"}` — into the `Enabled` variant. The variant is therefore
    /// not a reliable signal: `{"type":"disabled"}` used to be treated as enabled,
    /// which sent `reasoning_effort=high` upstream instead of turning thinking off.
    /// The `config_type` string is the source of truth.
    pub fn is_enabled(&self) -> bool {
        matches!(self.type_str(), "enabled" | "adaptive")
    }

    /// Wire type string (identical across all untagged variants).
    pub fn type_str(&self) -> &str {
        match self {
            ThinkingConfig::Enabled { config_type, .. } => config_type,
            ThinkingConfig::Disabled { config_type } => config_type,
            ThinkingConfig::Adaptive { config_type, .. } => config_type,
        }
    }

    pub fn budget_tokens(&self) -> Option<u32> {
        match self {
            ThinkingConfig::Enabled {
                config_type,
                budget_tokens,
                ..
            } if config_type == "enabled" => *budget_tokens,
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Deserialize)]
#[expect(
    dead_code,
    reason = "tool choice type tags are deserialized for protocol compatibility"
)]
#[serde(untagged)]
pub enum ToolChoice {
    Auto { r#type: String },
    Any { r#type: String },
    Tool { r#type: String, name: String },
}

#[derive(Debug, Clone, Deserialize)]
pub struct Metadata {
    /// Stable per-session identifier. Phase 3 P3-C reads this as the
    /// inbound source for the deterministic Chat `prompt_cache_key` when the
    /// provider's cache policy opts in (fail-closed: None/empty ⇒ no key).
    pub user_id: Option<String>,
}

// --- Anthropic Response types ---

#[derive(Debug, Clone, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<ResponseContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ResponseContentBlock {
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
}

// --- SSE Event types ---

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum SseEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStartData },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: u32,
        content_block: ContentBlockStartData,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: u32,
        delta: ContentBlockDeltaData,
    },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: u32 },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaData,
        usage: Option<StreamUsage>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "error")]
    Error { error: ErrorData },
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorData {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageStartData {
    pub id: String,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub role: String,
    pub content: Vec<Value>,
    pub model: String,
    pub usage: Option<StreamUsage>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ContentBlockStartData {
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Clone, Serialize)]
#[expect(
    clippy::enum_variant_names,
    reason = "Anthropic SSE wire names require Delta suffixes"
)]
#[serde(tag = "type")]
pub enum ContentBlockDeltaData {
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageDeltaData {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamUsage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: `{"type":"disabled"}` must NOT be treated as enabled.
    ///
    /// `ThinkingConfig` is `#[serde(untagged)]` with `Enabled` declared first and all
    /// fields optional, so serde deserializes the JSON into the `Enabled` variant even
    /// for `disabled`/`adaptive` type strings. `is_enabled()` must therefore read the
    /// wire `type` string, not the untagged variant (b01860f bug: disabled requests
    /// were sent upstream with `reasoning_effort=high` instead of `low`).
    #[test]
    fn disabled_type_string_is_not_enabled() {
        let parsed: ThinkingConfig =
            serde_json::from_str(r#"{"type":"disabled"}"#).expect("deserialize disabled");
        assert!(!parsed.is_enabled());
        assert_eq!(parsed.type_str(), "disabled");
    }

    #[test]
    fn enabled_type_string_is_enabled() {
        let parsed: ThinkingConfig =
            serde_json::from_str(r#"{"type":"enabled","budget_tokens":4096}"#)
                .expect("deserialize enabled");
        assert!(parsed.is_enabled());
        assert_eq!(parsed.budget_tokens(), Some(4096));
    }

    #[test]
    fn adaptive_type_string_is_enabled() {
        let parsed: ThinkingConfig =
            serde_json::from_str(r#"{"type":"adaptive"}"#).expect("deserialize adaptive");
        assert!(parsed.is_enabled());
        assert_eq!(parsed.budget_tokens(), None);
    }
}
