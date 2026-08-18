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

/// Select the upstream client for a model.
///
/// Routing is declarative (Phase 3 P3-A): a model whose provider binds the
/// `"official"` upstream — via `cache_policy.upstream` or the legacy default
/// binding carried by the `moonshot-official` provider name — goes to the
/// official Moonshot (Kimi For Coding) client; everything else goes to the
/// default eswitch upstream. `Config::effective_upstream_binding` resolves
/// the effective binding (explicit policy > legacy default binding > eswitch)
/// and `Config::provider_config` resolves the legacy `moonshot-official`
/// provider alias to the canonical `moonshot` config, so pre-policy configs
/// keep resolving cleanly. There is no provider-name string match here (G7):
/// the binding resolution is the only gate.
fn select_client<'a>(
    model: &str,
    config: &Config,
    client: &'a Arc<DeepSeekClient>,
    official_client: &'a Arc<DeepSeekClient>,
) -> &'a Arc<DeepSeekClient> {
    let bound_official = config
        .model_profile(model)
        .and_then(|p| config.effective_upstream_binding(&p.provider))
        == Some("official");
    if bound_official {
        official_client
    } else {
        client
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
        let cache_policy = cache_policy_for(&config, upstream_model);
        let sse_response = process_stream(
            model,
            is_reasoning_model,
            reasoning_field,
            reasoning_field_alt,
            msg_id,
            cache_policy,
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
                let cache_policy = cache_policy_for(&config, upstream_model);
                let anthropic_resp = convert_non_stream_response(
                    &parsed,
                    &model,
                    &msg_id,
                    &reasoning_field,
                    &reasoning_field_alt,
                    cache_policy.as_ref(),
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

#[cfg(test)]
mod tests {
    use super::select_client;
    use crate::cache::{CachePolicy, UsagePolicy};
    use crate::client::DeepSeekClient;
    use crate::config::Config;
    use crate::test_support::test_config;
    use std::sync::Arc;

    fn clients() -> (Arc<DeepSeekClient>, Arc<DeepSeekClient>) {
        (
            Arc::new(DeepSeekClient::new(
                "http://default.test".to_string(),
                "key".to_string(),
            )),
            Arc::new(DeepSeekClient::new(
                "http://official.test".to_string(),
                "key".to_string(),
            )),
        )
    }

    fn assert_routes_to(model: &str, config: &Config, expected_official: bool) {
        let (client, official_client) = clients();
        let selected = select_client(model, config, &client, &official_client);
        let is_official = Arc::ptr_eq(selected, &official_client);
        assert_eq!(
            is_official,
            expected_official,
            "model '{model}' should route to {}",
            if expected_official {
                "official"
            } else {
                "default"
            }
        );
    }

    #[test]
    fn legacy_config_without_policy_keeps_default_routing_for_all_models() {
        // P3-A "old config default behavior": with no cache_policy declared,
        // every model (including kimi-k3 on the canonical moonshot provider)
        // keeps the default eswitch routing — identical to pre-P3-A.
        let config = test_config();
        assert_routes_to("kimi-k3", &config, false);
        assert_routes_to("deepseek-v4-pro", &config, false);
        assert_routes_to("deepseek-v4-flash", &config, false);
        assert_routes_to("glm-5.2", &config, false);
        assert_routes_to("gpt-5.6-luna", &config, false);
    }

    #[test]
    fn official_client_is_selected_by_policy_upstream_binding() {
        // P3-A "official client selected by policy": when the canonical
        // moonshot provider declares cache_policy.upstream = "official", the
        // kimi-k3 profile routes to the official Moonshot client — the policy
        // binding replaces the merged `profile.provider == "moonshot-official"`
        // string match. Non-moonshot providers are unaffected.
        let mut config = test_config();
        let moonshot = config
            .providers
            .get_mut("moonshot")
            .expect("canonical moonshot provider exists in test_config");
        moonshot.cache_policy = Some(CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: Some("official".to_string()),
            effort_enum: None,
        });
        assert_routes_to("kimi-k3", &config, true);
        assert_routes_to("deepseek-v4-pro", &config, false);
        assert_routes_to("glm-5.2", &config, false);
        assert_routes_to("gpt-5.6-luna", &config, false);
    }

    #[test]
    fn legacy_moonshot_official_provider_name_still_resolves_to_official_upstream() {
        // P3-A "existing merged moonshot-official routing preserved": a
        // profile that still names its provider "moonshot-official" resolves
        // via the declared alias to the canonical moonshot provider, so once
        // that provider binds "official" the legacy name routes to the
        // official client exactly as the pre-policy string match did.
        let mut config = test_config();
        let kimi = config
            .model_profiles
            .iter_mut()
            .find(|p| p.name == "kimi-k3")
            .expect("kimi-k3 profile exists in test_config");
        kimi.provider = "moonshot-official".to_string();
        let moonshot = config
            .providers
            .get_mut("moonshot")
            .expect("canonical moonshot provider exists in test_config");
        moonshot.cache_policy = Some(CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: Some("official".to_string()),
            effort_enum: None,
        });
        assert_routes_to("kimi-k3", &config, true);
    }

    #[test]
    fn upstream_binding_other_than_official_keeps_default_routing() {
        // An upstream binding that is not "official" is not a known binding
        // and must never route to the official client; unknown bindings are
        // rejected by validate(), so this is defensive: only the exact
        // "official" name selects the official client.
        let mut config = test_config();
        let moonshot = config
            .providers
            .get_mut("moonshot")
            .expect("canonical moonshot provider exists in test_config");
        moonshot.cache_policy = Some(CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: Some("eswitch".to_string()),
            effort_enum: None,
        });
        assert_routes_to("kimi-k3", &config, false);
    }

    /// Build a config that faithfully reproduces the **live production**
    /// `[providers.moonshot-official]` shape (report 58, M1): an explicit
    /// provider block with `reasoning_field = "reasoning_content"` and
    /// **no** `cache_policy`, plus the `kimi-k3` profile bound to
    /// `moonshot-official`. In this shape `Config::provider_config` hits the
    /// explicit block directly (the alias fallback never fires), which is
    /// exactly why the pre-policy `provider == "moonshot-official"` string
    /// match was the only thing keeping these models on the official
    /// upstream — and why P3-A's policy-only gate regresses them to eswitch.
    fn live_shape_config() -> Config {
        let mut config = test_config();
        let mut legacy = config.providers["moonshot"].clone();
        legacy.reasoning_field = "reasoning_content".to_string();
        legacy.reasoning_field_alt = vec!["reasoning".to_string()];
        legacy.cache_policy = None; // live block declares no cache_policy
        config
            .providers
            .insert("moonshot-official".to_string(), legacy);
        let kimi = config
            .model_profiles
            .iter_mut()
            .find(|p| p.name == "kimi-k3")
            .expect("kimi-k3 profile exists in test_config");
        kimi.provider = "moonshot-official".to_string();
        config
    }

    #[test]
    fn live_shape_explicit_moonshot_official_block_without_policy_routes_to_official() {
        // MUST_FIX regression (report 58 M1): the live production config binds
        // kimi-k3 (L1 Opus + L4 Fable) to `moonshot-official` with an explicit
        // `[providers.moonshot-official]` block and NO cache_policy. The
        // origin/master `profile.provider == "moonshot-official"` string match
        // routed this to the official Moonshot (Kimi For Coding) client; P3-A
        // must preserve that routing via the legacy default upstream binding
        // (data, not a provider-name string match) instead of silently
        // falling back to the default eswitch client.
        let config = live_shape_config();
        assert_routes_to("kimi-k3", &config, true);
        // Non-moonshot models are unaffected and still default to eswitch.
        assert_routes_to("deepseek-v4-pro", &config, false);
        assert_routes_to("glm-5.2", &config, false);
        assert_routes_to("gpt-5.6-luna", &config, false);
    }

    #[test]
    fn live_shape_canonical_moonshot_without_policy_keeps_eswitch_routing() {
        // The canonical `moonshot` provider with no policy must keep default
        // (eswitch) routing — P3-A's confirmed behavior is unchanged; only
        // the legacy `moonshot-official` name carries the official default.
        let config = live_shape_config();
        let mut canonical = config;
        let kimi = canonical
            .model_profiles
            .iter_mut()
            .find(|p| p.name == "kimi-k3")
            .expect("kimi-k3 profile exists");
        kimi.provider = "moonshot".to_string();
        assert_routes_to("kimi-k3", &canonical, false);
    }

    #[test]
    fn legacy_moonshot_official_explicit_cache_policy_upstream_routes_official() {
        // Explicit policy precedence: when the legacy `moonshot-official`
        // provider declares `cache_policy.upstream = "official"`, the explicit
        // policy wins (priority 1) and the model routes to the official
        // client — the same result the legacy default binding produces, via
        // the explicit declarative gate.
        let mut config = live_shape_config();
        let legacy = config
            .providers
            .get_mut("moonshot-official")
            .expect("legacy block exists in live_shape_config");
        legacy.cache_policy = Some(CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: Some("official".to_string()),
            effort_enum: None,
        });
        assert_routes_to("kimi-k3", &config, true);
        assert_routes_to("deepseek-v4-pro", &config, false);
    }

    #[test]
    fn declared_policy_without_upstream_keeps_legacy_default_official_binding() {
        // Precedence rule: only an explicit `cache_policy.upstream` value
        // overrides the legacy default binding for `moonshot-official`. A
        // declared policy that names no upstream (upstream = None) does NOT
        // flip the legacy name to eswitch — the legacy default applies unless
        // an upstream value is explicitly declared. This keeps the live
        // production routing stable even if an empty cache_policy block is
        // ever added to the explicit provider.
        let mut config = live_shape_config();
        let legacy = config
            .providers
            .get_mut("moonshot-official")
            .expect("legacy block exists in live_shape_config");
        legacy.cache_policy = Some(CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: None,
            effort_enum: None,
        });
        assert_routes_to("kimi-k3", &config, true);
    }
}
