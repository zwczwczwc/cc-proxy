use crate::client::DeepSeekClient;
use crate::config::Config;
use axum::Router;
use std::sync::Arc;

pub fn create_router(
    client: Arc<DeepSeekClient>,
    official_client: Arc<DeepSeekClient>,
    config: Arc<Config>,
) -> Router {
    Router::new()
        .merge(messages::routes(client, official_client, config.clone()))
        .merge(models::routes(config.clone()))
        .merge(health::routes())
}

pub mod health;
pub mod messages;
pub mod models;
