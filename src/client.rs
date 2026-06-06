use reqwest::Client;
use serde_json::Value;
use crate::openai::types::ChatCompletionRequest;

/// Sanitize error response body: truncate + redact sensitive fields.
fn sanitize_error_body(body: &str) -> String {
    let truncated = if body.len() > 1024 {
        format!("{}...[truncated]", &body[..1024])
    } else {
        body.to_string()
    };
    // Simple sanitization: replace common sensitive patterns
    truncated
        .replace("Authorization", "Authorization:[REDACTED]")
        .replace("api_key", "api_key:[REDACTED]")
        .replace("Bearer ", "Bearer [REDACTED]")
}

/// DeepSeek HTTP client for sending requests to eswitch.
pub struct DeepSeekClient {
    client: Client,
    base_url: String,
    api_key: String,
}

impl DeepSeekClient {
    pub fn new(base_url: String, api_key: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
            api_key,
        }
    }

    /// Send a non-streaming chat completion request.
    pub async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> anyhow::Result<Value> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            let sanitized = sanitize_error_body(&body_text);
            anyhow::bail!("DeepSeek API error ({}): {}", status, sanitized);
        }

        let body: Value = response.json().await?;
        Ok(body)
    }

    /// Send a streaming chat completion request.
    /// Returns a stream of bytes from the SSE response.
    pub async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> anyhow::Result<impl futures_util::Stream<Item = reqwest::Result<bytes::Bytes>>> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            let sanitized = sanitize_error_body(&body_text);
            anyhow::bail!("DeepSeek API error ({}): {}", status, sanitized);
        }

        Ok(response.bytes_stream())
    }

    /// Health check the upstream API.
    pub async fn health_check(&self) -> anyhow::Result<bool> {
        let url = format!("{}/v1/models", self.base_url);
        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        Ok(response.status().is_success())
    }
}