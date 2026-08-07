use axum::{routing::get, Json, Router};

pub fn routes() -> Router {
    Router::new().route("/health", get(handle_health))
}

async fn handle_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "cc-proxy"
    }))
}
