use axum::{Json, Router, extract::State, routing::get};
use std::sync::Arc;
use crate::config::Config;

pub fn routes(config: Arc<Config>) -> Router {
    Router::new()
        .route("/v1/models", get(handle_models))
        .with_state(config)
}

async fn handle_models(State(config): State<Arc<Config>>) -> Json<serde_json::Value> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let data: Vec<serde_json::Value> = config
        .model_profiles
        .iter()
        .map(|p| {
            serde_json::json!({
                "id": p.name,
                "object": "model",
                "created": now,
                "owned_by": p.provider,
            })
        })
        .collect();

    Json(serde_json::json!({ "data": data }))
}