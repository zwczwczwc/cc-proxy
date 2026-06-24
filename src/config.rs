use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use serde::Deserialize;

/// TOML config file structure.
#[derive(Debug, Clone, Deserialize)]
struct ConfigFile {
    models: Option<ModelsSection>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelsSection {
    default: Option<String>,
    mapping: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub listen_addr: String,
    pub eswitch_url: String,
    pub api_key: String,
    pub log_level: String,
    pub model_mapping: HashMap<String, String>,
    pub default_model: String,
}

impl Config {
    pub fn from_env() -> Self {
        let (model_mapping, default_model) = Self::load_model_config();

        Self {
            listen_addr: env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:11435".to_string()),
            eswitch_url: env::var("ESWITCH_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".to_string()),
            api_key: env::var("DEEPSEEK_API_KEY").unwrap_or_else(|_| "not-needed".to_string()),
            log_level: env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
            model_mapping,
            default_model,
        }
    }

    /// Load model mapping from config file with priority:
    /// 1. Env var MODEL_CONFIG_PATH
    /// 2. /etc/codewhale-proxy/config.toml
    /// 3. Built-in hardcoded defaults
    fn load_model_config() -> (HashMap<String, String>, String) {
        let config_path = env::var("MODEL_CONFIG_PATH")
            .ok()
            .map(PathBuf::from)
            .or_else(|| {
                let etc = PathBuf::from("/etc/codewhale-proxy/config.toml");
                if etc.exists() {
                    Some(etc)
                } else {
                    None
                }
            });

        if let Some(path) = config_path {
            if let Ok(contents) = fs::read_to_string(&path) {
                if let Ok(cf) = toml::from_str::<ConfigFile>(&contents) {
                    if let Some(models) = cf.models {
                        let mapping = models.mapping.unwrap_or_default();
                        let default = models.default.unwrap_or_else(|| "deepseek-v4-pro".to_string());
                        tracing::info!("Loaded model config from {}", path.display());
                        return (mapping, default);
                    }
                }
                tracing::warn!("Failed to parse model config from {}, using defaults", path.display());
            }
        }

        // Built-in hardcoded defaults
        let defaults = Self::builtin_defaults();
        tracing::info!("Using built-in model mapping defaults");
        defaults
    }

    fn builtin_defaults() -> (HashMap<String, String>, String) {
        let mut mapping = HashMap::new();
        // Claude Opus variants
        mapping.insert("claude-opus-4-7".to_string(), "deepseek-v4-pro".to_string());
        mapping.insert("claude-opus-4-6".to_string(), "deepseek-v4-pro".to_string());
        mapping.insert("claude-opus-4-5".to_string(), "deepseek-v4-pro".to_string());
        mapping.insert("claude-opus-4".to_string(), "deepseek-v4-pro".to_string());
        // Claude Sonnet variants → v4-flash (cost-effective for most tasks)
        mapping.insert("claude-sonnet-4-6".to_string(), "deepseek-v4-flash".to_string());
        mapping.insert("claude-sonnet-4-5".to_string(), "deepseek-v4-flash".to_string());
        mapping.insert("claude-sonnet-4".to_string(), "deepseek-v4-flash".to_string());
        mapping.insert("claude-3-5-sonnet".to_string(), "deepseek-v4-flash".to_string());
        // Claude Haiku variants → v4-flash (lighter, cost-effective)
        mapping.insert("claude-haiku-4-5".to_string(), "deepseek-v4-flash".to_string());
        mapping.insert("claude-haiku-4".to_string(), "deepseek-v4-flash".to_string());
        mapping.insert("claude-3-haiku".to_string(), "deepseek-v4-flash".to_string());
        (mapping, "deepseek-v4-pro".to_string())
    }
}
