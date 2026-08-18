use crate::anthropic::converter::convert_request;
use crate::anthropic::types::MessagesRequest;
use crate::client::DeepSeekClient;
use crate::config::Config;
use crate::config::WireApi;
use crate::openai::converter::convert_non_stream_response;
use crate::reasoning::requires::requires_reasoning_content;
use crate::sse::stream::process_stream;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::post,
    Router,
};
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

const DEFAULT_MAX_RETRIES: u32 = 2;

fn max_retries() -> u32 {
    std::env::var("PROXY_MAX_RETRIES")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(DEFAULT_MAX_RETRIES)
}

/// Look up the reasoning field names for a model from the config.
/// Returns (reasoning_field, reasoning_field_alt).
fn get_reasoning_fields(model: &str, config: &Config) -> (String, Vec<String>) {
    if let Some(profile) = config.model_profile(model) {
        if let Some(provider) = config.provider_config(&profile.provider) {
            return (
                provider.reasoning_field.clone(),
                provider.reasoning_field_alt.clone(),
            );
        }
    }
    // Fallback: try common field names (Phase 1 compatibility)
    (
        "reasoning_content".to_string(),
        vec!["reasoning".to_string()],
    )
}

/// Look up the provider's declarative cache policy for a model — one lookup,
/// passed down to the Responses handlers. `None`/off ⇒ every cache-usage path
/// takes its Legacy branch. Phase 2b declares no policy in config.toml, so
/// this always returns `None` here; the gate is the policy, never a
/// provider-name string.
fn cache_policy_for(config: &Config, model: &str) -> Option<crate::cache::CachePolicy> {
    config
        .model_profile(model)
        .and_then(|profile| config.provider_config(&profile.provider))
        .and_then(|provider| provider.cache_policy.clone())
}

pub fn routes(
    client: Arc<DeepSeekClient>,
    official_client: Arc<DeepSeekClient>,
    config: Arc<Config>,
) -> Router {
    Router::new()
        .route("/v1/messages", post(handle_messages))
        .with_state((client, official_client, config))
}

/// Select the upstream client for a model: profiles bound to provider
/// "moonshot-official" go to the official Moonshot (Kimi For Coding) upstream;
/// everything else goes to the default eswitch upstream.
fn select_client<'a>(
    model: &str,
    config: &Config,
    client: &'a Arc<DeepSeekClient>,
    official_client: &'a Arc<DeepSeekClient>,
) -> &'a Arc<DeepSeekClient> {
    match config.model_profile(model) {
        Some(profile) if profile.provider == "moonshot-official" => official_client,
        _ => client,
    }
}

async fn handle_messages(
    State((client, official_client, config)): State<(
        Arc<DeepSeekClient>,
        Arc<DeepSeekClient>,
        Arc<Config>,
    )>,
    Json(req): Json<MessagesRequest>,
) -> Response {
    let model = req.model.clone();
    let msg_id = format!("msg_{:.20}", Uuid::new_v4().simple());
    let stream = req.stream.unwrap_or(false);

    let upstream_model =
        crate::anthropic::converter::map_model_to_upstream_for_responses(&model, &config);
    let upstream_client = select_client(&upstream_model, &config, &client, &official_client);
    if config.wire_api_for_model(&upstream_model) == WireApi::Responses {
        let cache_policy = cache_policy_for(&config, &upstream_model);
        let responses_req = match crate::responses::convert_request(&req, &config) {
            Ok(request) => request,
            Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"type":"error","error":{"type":"invalid_request_error","message":e.to_string()}}))).into_response(),
        };
        if stream {
            let byte_stream = match upstream_client.responses_completion_stream(&responses_req).await {
                Ok(stream) => stream,
                Err(e) => return (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"type":"error","error":{"type":"api_error","message":e.to_string()}}))).into_response(),
            };
            return crate::responses::stream::process_stream(
                upstream_model,
                msg_id,
                responses_req.request_id.clone(),
                byte_stream,
                cache_policy,
            )
            .into_response();
        }
        return match upstream_client.responses_completion(&responses_req).await {
            Ok(value) => match serde_json::from_value::<crate::responses::types::ResponsesResponse>(value) {
                Ok(response) => match crate::responses::convert_response(&response, &upstream_model, &msg_id, cache_policy.as_ref()) { Ok(result) => (StatusCode::OK, Json(result)).into_response(), Err(e) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"type":"error","error":{"type":"api_error","message":e.to_string()}}))).into_response() },
                Err(e) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"type":"error","error":{"type":"api_error","message":e.to_string()}}))).into_response(),
            },
            Err(e) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"type":"error","error":{"type":"api_error","message":e.to_string()}}))).into_response(),
        };
    }

    // Convert request
    let openai_req = match convert_request(&req, &config) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Request conversion error: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": format!("Failed to convert request: {}", e),
                    }
                })),
            )
                .into_response();
        }
    };

    let max_retries = max_retries();
    if stream {
        // Streaming response — retry on connection failure
        let mut retries = 0;
        let byte_stream = loop {
            match upstream_client.chat_completion_stream(&openai_req).await {
                Ok(s) => break s,
                Err(e) if retries < max_retries => {
                    retries += 1;
                    let delay = Duration::from_secs(2u64.pow(retries));
                    tracing::warn!(
                        "Stream request failed: {}, retrying in {:?} ({}/{})",
                        e,
                        delay,
                        retries,
                        max_retries
                    );
                    tokio::time::sleep(delay).await;
                    continue;
                }
                Err(e) => {
                    tracing::error!("Stream request failed after {} retries: {}", max_retries, e);
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": format!("Upstream error after {} retries: {}", max_retries, e),
                        }
                    }))).into_response();
                }
            }
        };
        let upstream_model = &openai_req.model;
        let is_reasoning_model = requires_reasoning_content(upstream_model, &config);
        let (reasoning_field, reasoning_field_alt) = get_reasoning_fields(upstream_model, &config);
        let sse_response = process_stream(
            model,
            is_reasoning_model,
            reasoning_field,
            reasoning_field_alt,
            msg_id,
            byte_stream,
        );
        sse_response.into_response()
    } else {
        // Non-streaming response — retry on connection failure
        let mut retries = 0;
        let openai_resp = loop {
            match upstream_client.chat_completion(&openai_req).await {
                Ok(r) => break r,
                Err(e) if retries < max_retries => {
                    retries += 1;
                    let delay = Duration::from_secs(2u64.pow(retries));
                    tracing::warn!(
                        "Non-stream request failed: {}, retrying in {:?} ({}/{})",
                        e,
                        delay,
                        retries,
                        max_retries
                    );
                    tokio::time::sleep(delay).await;
                    continue;
                }
                Err(e) => {
                    tracing::error!(
                        "Non-stream request failed after {} retries: {}",
                        max_retries,
                        e
                    );
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": format!("Upstream error after {} retries: {}", max_retries, e),
                        }
                    }))).into_response();
                }
            }
        };
        match serde_json::from_value::<crate::openai::types::ChatCompletionResponse>(openai_resp) {
            Ok(parsed) => {
                let upstream_model = &openai_req.model;
                let (reasoning_field, reasoning_field_alt) =
                    get_reasoning_fields(upstream_model, &config);
                let anthropic_resp = convert_non_stream_response(
                    &parsed,
                    &model,
                    &msg_id,
                    &reasoning_field,
                    &reasoning_field_alt,
                );
                (StatusCode::OK, Json(anthropic_resp)).into_response()
            }
            Err(e) => {
                tracing::error!("Response parsing error: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": format!("Failed to parse response: {}", e),
                        }
                    })),
                )
                    .into_response()
            }
        }
    }
}
