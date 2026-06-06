mod config;
mod client;
mod anthropic;
mod openai;
mod reasoning;
mod sse;
mod routes;

use std::sync::Arc;
use tower_http::cors::{CorsLayer, Any};
use tracing_subscriber::{fmt, EnvFilter};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));
    fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .init();

    let config = config::Config::from_env();

    tracing::info!("codewhale-proxy v0.1.0 starting");
    tracing::info!("Listening on: {}", config.listen_addr);
    tracing::info!("eswitch URL: {}", config.eswitch_url);

    let client = Arc::new(client::DeepSeekClient::new(
        config.eswitch_url.clone(),
        config.api_key.clone(),
    ));

    // Health check upstream
    match client.health_check().await {
        Ok(true) => tracing::info!("eswitch health check: OK"),
        Ok(false) => tracing::warn!("eswitch health check: FAILED"),
        Err(e) => tracing::warn!("eswitch health check error: {}", e),
    }

    let app = routes::create_router(client)
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    let listener = tokio::net::TcpListener::bind(&config.listen_addr).await?;
    tracing::info!("Server ready on {}", config.listen_addr);

    axum::serve(listener, app).await?;

    Ok(())
}