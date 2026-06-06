use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub listen_addr: String,
    pub eswitch_url: String,
    pub api_key: String,
    pub log_level: String,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            listen_addr: env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:11435".to_string()),
            eswitch_url: env::var("ESWITCH_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".to_string()),
            api_key: env::var("DEEPSEEK_API_KEY").unwrap_or_else(|_| "not-needed".to_string()),
            log_level: env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
        }
    }
}