use axum::{Json, Router, routing::get};

pub fn routes() -> Router {
    Router::new().route("/v1/models", get(handle_models))
}

async fn handle_models() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "data": [
            {
                "id": "deepseek-v4-pro",
                "object": "model",
                "created": 1725148800,
                "owned_by": "deepseek"
            },
            {
                "id": "deepseek-v4-flash",
                "object": "model",
                "created": 1725148800,
                "owned_by": "deepseek"
            },
            {
                "id": "deepseek-v4",
                "object": "model",
                "created": 1725148800,
                "owned_by": "deepseek"
            },
            {
                "id": "deepseek-chat",
                "object": "model",
                "created": 1725148800,
                "owned_by": "deepseek"
            },
            {
                "id": "deepseek-reasoner",
                "object": "model",
                "created": 1725148800,
                "owned_by": "deepseek"
            }
        ]
    }))
}