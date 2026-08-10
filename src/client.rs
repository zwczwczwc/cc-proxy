use crate::openai::types::ChatCompletionRequest;
use crate::responses::types::ResponsesRequest;
use reqwest::header::{CONTENT_LENGTH, CONTENT_TYPE, TRANSFER_ENCODING};
use reqwest::Client;
use serde_json::Value;
use std::time::{Duration, Instant};

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
            client: Client::builder()
                .read_timeout(Duration::from_secs(600))
                .connect_timeout(Duration::from_secs(10))
                .pool_idle_timeout(Duration::from_secs(60))
                .tcp_keepalive(Duration::from_secs(60))
                .pool_max_idle_per_host(4)
                .build()
                .expect("Failed to create HTTP client"),
            base_url,
            api_key,
        }
    }

    /// Send a non-streaming chat completion request.
    pub async fn chat_completion(&self, request: &ChatCompletionRequest) -> anyhow::Result<Value> {
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

    pub async fn responses_completion(&self, request: &ResponsesRequest) -> anyhow::Result<Value> {
        let started_at = Instant::now();
        tracing::info!(
            request_id = %request.request_id,
            static_prefix_hash = %request.static_prefix_hash,
            history_prefix_hash = %request.history_prefix_hash,
            wire_input_hash = %request.wire_input_hash,
            input_item_types = ?request.input_item_types,
            synthetic_tail_present = request.synthetic_tail_present,
            "Responses request telemetry"
        );
        let response = self
            .client
            .post(format!("{}/v1/responses", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await?;
        let headers_at = started_at.elapsed();
        let status = response.status();
        tracing::info!(
            request_id = %request.request_id,
            upstream_http_status = status.as_u16(),
            headers_elapsed_ms = headers_at.as_millis() as u64,
            response_content_type = ?response.headers().get(CONTENT_TYPE).and_then(|v| v.to_str().ok()),
            response_content_length = ?response.headers().get(CONTENT_LENGTH).and_then(|v| v.to_str().ok()),
            response_transfer_encoding = ?response.headers().get(TRANSFER_ENCODING).and_then(|v| v.to_str().ok()),
            "Responses response headers"
        );
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            tracing::warn!(
                request_id = %request.request_id,
                upstream_http_status = status.as_u16(),
                elapsed_ms = started_at.elapsed().as_millis() as u64,
                error_body_len = body.len(),
                "Responses upstream request failed"
            );
            anyhow::bail!(
                "Responses API error ({}): {}",
                status,
                sanitize_error_body(&body)
            );
        }
        let body: Value = response.json().await?;
        tracing::info!(
            request_id = %request.request_id,
            elapsed_ms = started_at.elapsed().as_millis() as u64,
            terminal_status = ?body
                .get("status")
                .and_then(|value| value.as_str()),
            "Responses non-stream completed"
        );
        Ok(body)
    }

    /// Send a streaming Responses API request and return the upstream SSE body.
    pub async fn responses_completion_stream(
        &self,
        request: &ResponsesRequest,
    ) -> anyhow::Result<impl futures_util::Stream<Item = reqwest::Result<bytes::Bytes>>> {
        let started_at = Instant::now();
        tracing::info!(
            request_id = %request.request_id,
            static_prefix_hash = %request.static_prefix_hash,
            history_prefix_hash = %request.history_prefix_hash,
            wire_input_hash = %request.wire_input_hash,
            input_item_types = ?request.input_item_types,
            synthetic_tail_present = request.synthetic_tail_present,
            "Responses stream request telemetry"
        );
        let response = self
            .client
            .post(format!("{}/v1/responses", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await?;
        let headers_at = started_at.elapsed();
        let status = response.status();
        tracing::info!(
            request_id = %request.request_id,
            upstream_http_status = status.as_u16(),
            headers_elapsed_ms = headers_at.as_millis() as u64,
            response_content_type = ?response.headers().get(CONTENT_TYPE).and_then(|v| v.to_str().ok()),
            response_content_length = ?response.headers().get(CONTENT_LENGTH).and_then(|v| v.to_str().ok()),
            response_transfer_encoding = ?response.headers().get(TRANSFER_ENCODING).and_then(|v| v.to_str().ok()),
            "Responses stream response headers"
        );
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            tracing::warn!(
                request_id = %request.request_id,
                upstream_http_status = status.as_u16(),
                elapsed_ms = started_at.elapsed().as_millis() as u64,
                error_body_len = body.len(),
                "Responses stream upstream request failed"
            );
            anyhow::bail!(
                "Responses API error ({}): {}",
                status,
                sanitize_error_body(&body)
            );
        }
        let request_id = request.request_id.clone();
        let first_byte = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let first_byte_at = started_at;
        let bytes = response.bytes_stream();
        Ok(Box::pin(futures_util::stream::unfold(
            (bytes, first_byte, first_byte_at),
            move |(mut bytes, first_byte, started_at)| {
                let request_id = request_id.clone();
                async move {
                    use futures_util::StreamExt;
                    match bytes.next().await {
                        Some(Ok(chunk)) => {
                            if !first_byte.swap(true, std::sync::atomic::Ordering::Relaxed) {
                                tracing::info!(
                                    request_id = %request_id,
                                    first_byte_elapsed_ms = started_at.elapsed().as_millis() as u64,
                                    first_byte_len = chunk.len(),
                                    "Responses stream first byte"
                                );
                            }
                            Some((Ok(chunk), (bytes, first_byte, started_at)))
                        }
                        Some(Err(error)) => {
                            tracing::warn!(
                                request_id = %request_id,
                                elapsed_ms = started_at.elapsed().as_millis() as u64,
                                error = %error,
                                "Responses stream upstream read error"
                            );
                            Some((Err(error), (bytes, first_byte, started_at)))
                        }
                        None => {
                            tracing::info!(
                                request_id = %request_id,
                                elapsed_ms = started_at.elapsed().as_millis() as u64,
                                "Responses stream upstream EOF"
                            );
                            None
                        }
                    }
                }
            },
        )))
    }
}
