pub mod messages;
pub mod models;
pub mod health;

use axum::Router;
use crate::client::DeepSeekClient;
use std::sync::Arc;

pub fn create_router(client: Arc<DeepSeekClient>) -> Router {
    Router::new()
        .merge(messages::routes(client.clone()))
        .merge(models::routes())
        .merge(health::routes())
}