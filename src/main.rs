mod config;
mod client;
mod anthropic;
mod openai;
mod reasoning;
mod sse;
mod routes;

use std::sync::Arc;
use tokio::signal;
use tower_http::cors::{CorsLayer, Any};
use axum::extract::DefaultBodyLimit;
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

    let config = Arc::new(config::Config::from_env());

    tracing::info!("codewhale-proxy v0.1.0 starting");
    tracing::info!("Listening on: {}", config.listen_addr);
    tracing::info!("eswitch URL: {}", config.eswitch_url);
    tracing::info!("Default model: {}", config.default_model);

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

    let app = routes::create_router(client, config.clone())
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    let listener = tokio::net::TcpListener::bind(&config.listen_addr).await?;
    tracing::info!("Server ready on {}", config.listen_addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            // Handle SIGTERM (from kill/systemd stop) and SIGINT (Ctrl+C)
            let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())
                .expect("Failed to create SIGTERM handler");
            tokio::select! {
                _ = sigterm.recv() => {
                    tracing::info!("Received SIGTERM, shutting down gracefully...");
                }
                _ = signal::ctrl_c() => {
                    tracing::info!("Received SIGINT, shutting down gracefully...");
                }
            }
        })
        .await?;

    Ok(())
}
