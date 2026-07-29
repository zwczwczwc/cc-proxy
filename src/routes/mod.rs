use axum::Router;
use crate::client::DeepSeekClient;
use crate::config::Config;
use std::sync::Arc;

pub fn create_router(client: Arc<DeepSeekClient>, config: Arc<Config>) -> Router {
    Router::new()
        .merge(messages::routes(client.clone(), config.clone()))
        .merge(models::routes(config.clone()))
        .merge(health::routes())
}

pub mod messages;
pub mod models;
pub mod health;
