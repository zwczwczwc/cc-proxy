use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Anthropic Messages API request type
#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<SystemContentBlock>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct SystemContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
}

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone)]
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
                let preview = if preview.len() > 500 { &preview[..500] } else { &preview };
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

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        signature: String,
    },
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
    Image {
        source: ImageSource,
    },
    /// Catch-all for unknown content block types (e.g., server_tool_use, search_result)
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ToolResultContentBlock>),
}

/// A single content block inside a tool_result.
/// Anthropic API supports text and image blocks in tool results.
#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Deserialize)]
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
    pub fn is_enabled(&self) -> bool {
        matches!(self, ThinkingConfig::Enabled { .. } | ThinkingConfig::Adaptive { .. })
    }

    pub fn budget_tokens(&self) -> Option<u32> {
        match self {
            ThinkingConfig::Enabled { budget_tokens, .. } => *budget_tokens,
            ThinkingConfig::Adaptive { .. } => None,
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
#[serde(untagged)]
pub enum ToolChoice {
    Auto { r#type: String },
    Any { r#type: String },
    Tool { r#type: String, name: String },
}

#[derive(Debug, Clone, Deserialize)]
pub struct Metadata {
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
    Thinking {
        thinking: String,
        signature: String,
    },
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
}

// --- SSE Event types ---

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum SseEvent {
    #[serde(rename = "message_start")]
    MessageStart {
        message: MessageStartData,
    },
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
    ContentBlockStop {
        index: u32,
    },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaData,
        usage: Option<StreamUsage>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "error")]
    Error {
        error: ErrorData,
    },
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
    ToolUse { id: String, name: String, input: Value },
}

#[derive(Debug, Clone, Serialize)]
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
}