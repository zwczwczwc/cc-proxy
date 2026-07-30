use serde::{Serialize, Deserialize};
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
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: Option<ChatMessage>,
    pub delta: Option<ChatDelta>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
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
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: Option<String>,
    pub function: Option<FunctionCall>,
}

#[derive(Debug, Clone, serde::Deserialize)]
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
/// Standard OpenAI: cached tokens detail
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

// --- Model list response ---

#[derive(Debug, Clone, Serialize)]
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
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}