use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize)]
pub struct ResponsesRequest {
    pub model: String,
    #[serde(skip)]
    pub request_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    pub input: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponsesTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    pub max_output_tokens: u32,
    pub stream: bool,
    #[serde(skip)]
    pub static_prefix_hash: String,
    #[serde(skip)]
    pub history_prefix_hash: String,
    #[serde(skip)]
    pub wire_input_hash: String,
    #[serde(skip)]
    pub input_item_types: Vec<String>,
    #[serde(skip)]
    pub synthetic_tail_present: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponsesTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReasoningConfig {
    pub effort: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[expect(
    dead_code,
    reason = "response identifiers preserve upstream protocol compatibility"
)]
pub struct ResponsesResponse {
    pub id: Option<String>,
    pub model: Option<String>,
    pub status: Option<String>,
    #[serde(default)]
    pub output: Vec<Value>,
    pub usage: Option<ResponsesUsage>,
    pub error: Option<Value>,
    pub incomplete_details: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesUsage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub input_tokens_details: Option<InputTokenDetails>,
    #[serde(default)]
    pub cache_write_tokens: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InputTokenDetails {
    pub cached_tokens: Option<u32>,
    #[serde(default)]
    pub cache_write_tokens: Option<u32>,
}
