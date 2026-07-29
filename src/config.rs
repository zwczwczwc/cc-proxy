use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use serde::Deserialize;

/// TOML config file structure.
#[derive(Debug, Clone, Deserialize)]
struct ConfigFile {
    models: Option<ModelsSection>,
    #[serde(default)]
    providers: Option<HashMap<String, ProviderConfig>>,
    #[serde(default)]
    model_profiles: Option<Vec<ModelProfile>>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelsSection {
    default: Option<String>,
    mapping: Option<HashMap<String, String>>,
}

/// Per-provider reasoning/thinking protocol configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ProviderConfig {
    /// Primary field name for reasoning content in API responses (e.g. "reasoning_content", "reasoning").
    pub reasoning_field: String,
    /// Alternative field names to try if primary is empty/missing.
    #[serde(default)]
    pub reasoning_field_alt: Vec<String>,
    /// API parameter name for thinking configuration (e.g. "thinking"). None if unsupported.
    pub thinking_param: Option<String>,
    /// Value for thinking.type when enabling thinking (e.g. "enabled").
    pub thinking_type_enabled: Option<String>,
    /// Value for thinking.type when disabling thinking (e.g. "disabled").
    pub thinking_type_disabled: Option<String>,
    /// When true, this provider cannot turn off thinking; "off" effort sets lowest level instead.
    #[serde(default)]
    pub disable_thinking: bool,
    /// API parameter name for reasoning effort (e.g. "reasoning_effort").
    pub effort_param: String,
    /// Maps Anthropic effort levels to provider-specific values.
    pub effort_map: HashMap<String, String>,
}

/// Per-model behavioral profile.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelProfile {
    /// Canonical model name (as known to the upstream API).
    pub name: String,
    /// Provider key (must match a key in [providers]).
    pub provider: String,
    /// Whether this model supports reasoning/thinking.
    #[serde(default)]
    pub reasoning_enabled: bool,
    /// Whether reasoning content from previous turns must be replayed to maintain cache.
    #[serde(default)]
    pub reasoning_replay: bool,
    /// Whether tool-call requests require reasoning_content to be present.
    #[serde(default)]
    pub toolcall_requires_reasoning: bool,
    /// Alternative names for this model (e.g. "deepseek-chat" → "deepseek-v4-pro").
    #[serde(default)]
    pub aliases: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub listen_addr: String,
    pub eswitch_url: String,
    pub api_key: String,
    pub log_level: String,
    pub model_mapping: HashMap<String, String>,
    pub default_model: String,
    pub model_profiles: Vec<ModelProfile>,
    pub providers: HashMap<String, ProviderConfig>,
    /// Index: model name (or alias) → index into model_profiles
    pub(crate) profile_by_name: HashMap<String, usize>,
}

impl Config {
    pub fn from_env() -> Self {
        let (model_mapping, default_model, model_profiles, providers) = Self::load_model_config();

        let mut config = Self {
            listen_addr: env::var("LISTEN_ADDR")
                .unwrap_or_else(|_| "0.0.0.0:11435".to_string()),
            eswitch_url: env::var("ESWITCH_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string()),
            api_key: env::var("DEEPSEEK_API_KEY")
                .unwrap_or_else(|_| "not-needed".to_string()),
            log_level: env::var("RUST_LOG")
                .unwrap_or_else(|_| "info".to_string()),
            model_mapping,
            default_model,
            model_profiles,
            providers,
            profile_by_name: HashMap::new(),
        };

        config.build_profile_index();
        config.validate();

        tracing::info!(
            "Loaded {} model profiles, {} providers",
            config.model_profiles.len(),
            config.providers.len()
        );

        config
    }

    /// Look up a ModelProfile by model name or alias.
    pub fn model_profile(&self, model: &str) -> Option<&ModelProfile> {
        self.profile_by_name
            .get(model)
            .map(|&i| &self.model_profiles[i])
    }

    /// Look up a ProviderConfig by provider name.
    pub fn provider_config(&self, provider: &str) -> Option<&ProviderConfig> {
        self.providers.get(provider)
    }

    /// Build the profile_by_name index from model_profiles (names + aliases).
    fn build_profile_index(&mut self) {
        for (i, profile) in self.model_profiles.iter().enumerate() {
            self.profile_by_name
                .insert(profile.name.clone(), i);
            for alias in &profile.aliases {
                self.profile_by_name
                    .insert(alias.clone(), i);
            }
        }
    }

    /// Startup validation: panic on misconfiguration (no silent failures).
    fn validate(&self) {
        // 1. Each model_profile.provider must exist in [providers]
        for profile in &self.model_profiles {
            if !self.providers.contains_key(&profile.provider) {
                panic!(
                    "Model profile '{}' references unknown provider '{}'. \
                     Available providers: {:?}",
                    profile.name,
                    profile.provider,
                    self.providers.keys().collect::<Vec<_>>()
                );
            }
        }

        // 2. Default model must be in model_profiles
        if self.profile_by_name.get(&self.default_model).is_none() {
            panic!(
                "Default model '{}' not found in model_profiles. \
                 Available: {:?}",
                self.default_model,
                self.model_profiles
                    .iter()
                    .map(|p| &p.name)
                    .collect::<Vec<_>>()
            );
        }

        // 3. reasoning_enabled=true with disable_thinking=false:
        //    thinking_param and thinking_type_enabled/disabled must be non-empty
        for profile in &self.model_profiles {
            if profile.reasoning_enabled {
                if let Some(provider) = self.providers.get(&profile.provider) {
                    if !provider.disable_thinking {
                        if provider.thinking_param.is_none() {
                            panic!(
                                "Provider '{}' (used by '{}') has reasoning_enabled=true \
                                 but thinking_param is not set",
                                profile.provider, profile.name
                            );
                        }
                        if provider.thinking_type_enabled.is_none() {
                            panic!(
                                "Provider '{}' (used by '{}') has reasoning_enabled=true \
                                 but thinking_type_enabled is not set",
                                profile.provider, profile.name
                            );
                        }
                        if provider.thinking_type_disabled.is_none() {
                            panic!(
                                "Provider '{}' (used by '{}') has reasoning_enabled=true \
                                 but thinking_type_disabled is not set",
                                profile.provider, profile.name
                            );
                        }
                    }
                }
            }
        }

        // 4. reasoning_replay=true: reasoning_field must be non-empty
        for profile in &self.model_profiles {
            if profile.reasoning_replay {
                if let Some(provider) = self.providers.get(&profile.provider) {
                    if provider.reasoning_field.is_empty() {
                        panic!(
                            "Provider '{}' (used by '{}') has reasoning_replay=true \
                             but reasoning_field is empty",
                            profile.provider, profile.name
                        );
                    }
                }
            }
        }

        // 5. effort_map must contain "high" and "max"
        for (name, provider) in &self.providers {
            if !provider.effort_map.contains_key("high") {
                panic!(
                    "Provider '{}' effort_map missing required key 'high'. \
                     effort_map keys: {:?}",
                    name,
                    provider.effort_map.keys().collect::<Vec<_>>()
                );
            }
            if !provider.effort_map.contains_key("max") {
                panic!(
                    "Provider '{}' effort_map missing required key 'max'. \
                     effort_map keys: {:?}",
                    name,
                    provider.effort_map.keys().collect::<Vec<_>>()
                );
            }
        }

        // 6. model_profiles.name must be unique (including aliases)
        let mut seen: HashMap<&str, &str> = HashMap::new(); // name → origin
        for profile in &self.model_profiles {
            if let Some(existing) = seen.insert(&profile.name, &profile.name) {
                panic!(
                    "Duplicate model name '{}' in model_profiles (from '{}' and '{}')",
                    profile.name, existing, profile.name
                );
            }
            for alias in &profile.aliases {
                if let Some(existing) = seen.insert(alias, &profile.name) {
                    panic!(
                        "Duplicate model alias '{}' in model_profiles \
                         (from '{}' and '{}')",
                        alias, existing, profile.name
                    );
                }
            }
        }
    }

    /// Load model config from file with priority:
    /// 1. Env var MODEL_CONFIG_PATH
    /// 2. /etc/codewhale-proxy/config.toml
    /// 3. Built-in hardcoded defaults
    fn load_model_config() -> (
        HashMap<String, String>,
        String,
        Vec<ModelProfile>,
        HashMap<String, ProviderConfig>,
    ) {
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
            match fs::read_to_string(&path) {
                Ok(contents) => {
                    match toml::from_str::<ConfigFile>(&contents) {
                        Ok(cf) => {
                            let mapping = cf
                                .models
                                .as_ref()
                                .and_then(|m| m.mapping.clone())
                                .unwrap_or_default();
                            let default = cf
                                .models
                                .as_ref()
                                .and_then(|m| m.default.clone())
                                .unwrap_or_else(|| "deepseek-v4-pro".to_string());
                            let providers =
                                cf.providers.unwrap_or_else(Self::builtin_default_providers);
                            let model_profiles = cf
                                .model_profiles
                                .unwrap_or_else(Self::builtin_default_profiles);
                            tracing::info!("Loaded model config from {}", path.display());
                            return (mapping, default, model_profiles, providers);
                        }
                        Err(e) => {
                            panic!(
                                "Failed to parse model config from {}: {}. \
                                 Config file must be valid TOML. Fix the syntax error or remove the file.",
                                path.display(),
                                e
                            );
                        }
                    }
                }
                Err(e) => {
                    panic!(
                        "Failed to read config file {}: {}",
                        path.display(),
                        e
                    );
                }
            }
        }

        // Built-in hardcoded defaults
        let (mapping, default) = Self::builtin_defaults();
        tracing::info!("Using built-in model config defaults");
        (
            mapping,
            default,
            Self::builtin_default_profiles(),
            Self::builtin_default_providers(),
        )
    }

    fn builtin_defaults() -> (HashMap<String, String>, String) {
        let mut mapping = HashMap::new();
        // Claude Opus variants
        mapping.insert("claude-opus-4-7".to_string(), "deepseek-v4-pro".to_string());
        mapping.insert("claude-opus-4-6".to_string(), "deepseek-v4-pro".to_string());
        mapping.insert("claude-opus-4-5".to_string(), "deepseek-v4-pro".to_string());
        mapping.insert("claude-opus-4".to_string(), "deepseek-v4-pro".to_string());
        // Claude Sonnet variants → v4-flash (cost-effective for most tasks)
        mapping.insert("claude-sonnet-4-7".to_string(), "kimi-k3".to_string());
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

    fn builtin_default_providers() -> HashMap<String, ProviderConfig> {
        let mut providers = HashMap::new();

        // DeepSeek: reasoning_content field, thinking.type supported, 5-level effort
        providers.insert(
            "deepseek".to_string(),
            ProviderConfig {
                reasoning_field: "reasoning_content".to_string(),
                reasoning_field_alt: vec![],
                thinking_param: Some("thinking".to_string()),
                thinking_type_enabled: Some("enabled".to_string()),
                thinking_type_disabled: Some("disabled".to_string()),
                disable_thinking: false,
                effort_param: "reasoning_effort".to_string(),
                effort_map: {
                    let mut m = HashMap::new();
                    m.insert("low".to_string(), "high".to_string());
                    m.insert("medium".to_string(), "high".to_string());
                    m.insert("high".to_string(), "high".to_string());
                    m.insert("max".to_string(), "max".to_string());
                    m.insert("xhigh".to_string(), "max".to_string());
                    m
                },
            },
        );

        // GLM: reasoning_content field, thinking.type supported, 7-level effort
        providers.insert(
            "glm".to_string(),
            ProviderConfig {
                reasoning_field: "reasoning_content".to_string(),
                reasoning_field_alt: vec![],
                thinking_param: Some("thinking".to_string()),
                thinking_type_enabled: Some("enabled".to_string()),
                thinking_type_disabled: Some("disabled".to_string()),
                disable_thinking: false,
                effort_param: "reasoning_effort".to_string(),
                effort_map: {
                    let mut m = HashMap::new();
                    m.insert("none".to_string(), "none".to_string());
                    m.insert("minimal".to_string(), "minimal".to_string());
                    m.insert("low".to_string(), "low".to_string());
                    m.insert("medium".to_string(), "medium".to_string());
                    m.insert("high".to_string(), "high".to_string());
                    m.insert("xhigh".to_string(), "xhigh".to_string());
                    m.insert("max".to_string(), "max".to_string());
                    m
                },
            },
        );

        // Fireworks (kimi-k3): reasoning field, no thinking.type, effort passthrough
        providers.insert(
            "fireworks".to_string(),
            ProviderConfig {
                reasoning_field: "reasoning".to_string(),
                reasoning_field_alt: vec![],
                thinking_param: None,
                thinking_type_enabled: None,
                thinking_type_disabled: None,
                disable_thinking: true,
                effort_param: "reasoning_effort".to_string(),
                effort_map: {
                    let mut m = HashMap::new();
                    m.insert("low".to_string(), "low".to_string());
                    m.insert("medium".to_string(), "medium".to_string());
                    m.insert("high".to_string(), "high".to_string());
                    m.insert("max".to_string(), "max".to_string());
                    m.insert("xhigh".to_string(), "max".to_string());
                    m
                },
            },
        );

        providers
    }

    fn builtin_default_profiles() -> Vec<ModelProfile> {
        vec![
            ModelProfile {
                name: "deepseek-v4-pro".to_string(),
                provider: "deepseek".to_string(),
                reasoning_enabled: true,
                reasoning_replay: true,
                toolcall_requires_reasoning: true,
                aliases: vec![
                    "deepseek-chat".to_string(),
                    "deepseek-reasoner".to_string(),
                ],
            },
            ModelProfile {
                name: "deepseek-v4-flash".to_string(),
                provider: "deepseek".to_string(),
                reasoning_enabled: true,
                reasoning_replay: true,
                toolcall_requires_reasoning: true,
                aliases: vec![],
            },
            ModelProfile {
                name: "glm-5.2".to_string(),
                provider: "glm".to_string(),
                reasoning_enabled: true,
                reasoning_replay: false,
                toolcall_requires_reasoning: false,
                aliases: vec![],
            },
            ModelProfile {
                name: "kimi-k3".to_string(),
                provider: "fireworks".to_string(),
                reasoning_enabled: true,
                reasoning_replay: true,
                toolcall_requires_reasoning: false,
                aliases: vec![],
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_file_parse_with_providers_and_profiles() {
        let toml_str = r#"
[models]
default = "deepseek-v4-pro"

[providers.deepseek]
reasoning_field = "reasoning_content"
reasoning_field_alt = []
thinking_param = "thinking"
thinking_type_enabled = "enabled"
thinking_type_disabled = "disabled"
disable_thinking = false
effort_param = "reasoning_effort"

[providers.deepseek.effort_map]
low = "high"
high = "high"
max = "max"

[providers.fireworks]
reasoning_field = "reasoning"
disable_thinking = true
effort_param = "reasoning_effort"

[providers.fireworks.effort_map]
low = "low"
high = "high"
max = "max"

[[model_profiles]]
name = "deepseek-v4-pro"
provider = "deepseek"
reasoning_enabled = true
reasoning_replay = true
toolcall_requires_reasoning = true
aliases = ["deepseek-chat"]

[[model_profiles]]
name = "kimi-k3"
provider = "fireworks"
reasoning_enabled = true
reasoning_replay = true
aliases = []
"#;

        let cf: ConfigFile = toml::from_str(toml_str).expect("TOML should parse");

        // Verify providers are parsed
        let providers = cf.providers.expect("providers should be Some");
        assert_eq!(providers.len(), 2);
        assert!(providers.contains_key("deepseek"));
        assert!(providers.contains_key("fireworks"));

        // Verify provider fields
        let ds = &providers["deepseek"];
        assert_eq!(ds.reasoning_field, "reasoning_content");
        assert_eq!(ds.thinking_param.as_deref(), Some("thinking"));
        assert!(!ds.disable_thinking);

        let fw = &providers["fireworks"];
        assert_eq!(fw.reasoning_field, "reasoning");
        assert_eq!(fw.reasoning_field_alt, Vec::<String>::new());
        assert!(fw.disable_thinking);
        assert!(fw.thinking_param.is_none());

        // Verify model_profiles are parsed
        let profiles = cf.model_profiles.expect("model_profiles should be Some");
        assert_eq!(profiles.len(), 2);
        assert_eq!(profiles[0].name, "deepseek-v4-pro");
        assert_eq!(profiles[0].provider, "deepseek");
        assert_eq!(profiles[1].name, "kimi-k3");
        assert_eq!(profiles[1].provider, "fireworks");
    }

    #[test]
    #[should_panic(expected = "unknown provider")]
    fn test_validate_panics_on_unknown_provider() {
        let mut providers = HashMap::new();
        providers.insert(
            "deepseek".to_string(),
            ProviderConfig {
                reasoning_field: "reasoning_content".to_string(),
                reasoning_field_alt: vec![],
                thinking_param: Some("thinking".to_string()),
                thinking_type_enabled: Some("enabled".to_string()),
                thinking_type_disabled: Some("disabled".to_string()),
                disable_thinking: false,
                effort_param: "reasoning_effort".to_string(),
                effort_map: {
                    let mut m = HashMap::new();
                    m.insert("high".to_string(), "high".to_string());
                    m.insert("max".to_string(), "max".to_string());
                    m
                },
            },
        );

        let model_profiles = vec![ModelProfile {
            name: "bad-model".to_string(),
            provider: "nonexistent".to_string(), // does not exist in providers
            reasoning_enabled: false,
            reasoning_replay: false,
            toolcall_requires_reasoning: false,
            aliases: vec![],
        }];

        let mut config = Config {
            listen_addr: "0.0.0.0:11435".to_string(),
            eswitch_url: "http://127.0.0.1:11434".to_string(),
            api_key: "test".to_string(),
            log_level: "info".to_string(),
            model_mapping: HashMap::new(),
            default_model: "bad-model".to_string(),
            model_profiles,
            providers,
            profile_by_name: HashMap::new(),
        };
        config.build_profile_index();
        config.validate(); // should panic
    }

    #[test]
    fn test_fireworks_provider_optional_fields() {
        // Verify that fireworks (no thinking_param) parses correctly
        let toml_str = r#"
[providers.fireworks]
reasoning_field = "reasoning"
disable_thinking = true
effort_param = "reasoning_effort"

[providers.fireworks.effort_map]
low = "low"
high = "high"
max = "max"

[[model_profiles]]
name = "kimi-k3"
provider = "fireworks"
reasoning_enabled = true
"#;

        let cf: ConfigFile = toml::from_str(toml_str).expect("TOML should parse");
        let providers = cf.providers.expect("providers should be Some");
        let fw = &providers["fireworks"];
        assert_eq!(fw.reasoning_field, "reasoning");
        assert!(fw.thinking_param.is_none(), "thinking_param should be None for fireworks");
        assert!(fw.thinking_type_enabled.is_none());
        assert!(fw.thinking_type_disabled.is_none());
    }
}