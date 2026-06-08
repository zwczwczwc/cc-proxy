use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    Router,
    routing::post,
};
use std::sync::Arc;
use uuid::Uuid;
use crate::anthropic::types::MessagesRequest;
use crate::anthropic::converter::convert_request;
use crate::config::Config;
use crate::openai::converter::convert_non_stream_response;
use crate::client::DeepSeekClient;
use crate::reasoning::requires::requires_reasoning_content;
use crate::sse::stream::process_stream;

pub fn routes(client: Arc<DeepSeekClient>, config: Arc<Config>) -> Router {
    Router::new()
        .route("/v1/messages", post(handle_messages))
        .with_state((client, config))
}

async fn handle_messages(
    State((client, config)): State<(Arc<DeepSeekClient>, Arc<Config>)>,
    Json(req): Json<MessagesRequest>,
) -> Response {
    let model = req.model.clone();
    let msg_id = format!("msg_{}", Uuid::new_v4().to_string().replace("-", "")[..20].to_string());
    let stream = req.stream.unwrap_or(false);

    // Convert request
    let openai_req = match convert_request(&req, &config) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Request conversion error: {}", e);
            return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": format!("Failed to convert request: {}", e),
                }
            }))).into_response();
        }
    };

    if stream {
        // Streaming response
        match client.chat_completion_stream(&openai_req).await {
            Ok(byte_stream) => {
                let upstream_model = &openai_req.model;
                let is_reasoning_model = requires_reasoning_content(upstream_model);
                let sse_response = process_stream(model, is_reasoning_model, msg_id, byte_stream);
                sse_response.into_response()
            }
            Err(e) => {
                tracing::error!("Stream request error: {}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": format!("Upstream error: {}", e),
                    }
                }))).into_response()
            }
        }
    } else {
        // Non-streaming response
        match client.chat_completion(&openai_req).await {
            Ok(openai_resp) => {
                match serde_json::from_value::<crate::openai::types::ChatCompletionResponse>(openai_resp) {
                    Ok(parsed) => {
                        let anthropic_resp = convert_non_stream_response(&parsed, &model, &msg_id);
                        (StatusCode::OK, Json(anthropic_resp)).into_response()
                    }
                    Err(e) => {
                        tracing::error!("Response parsing error: {}", e);
                        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": format!("Failed to parse response: {}", e),
                            }
                        }))).into_response()
                    }
                }
            }
            Err(e) => {
                tracing::error!("Upstream request error: {}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": format!("Upstream error: {}", e),
                    }
                }))).into_response()
            }
        }
    }
}
