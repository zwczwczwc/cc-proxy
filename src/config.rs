use crate::cache::CachePolicy;
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

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
    #[allow(dead_code)]
    pub effort_param: String,
    /// Maps Anthropic effort levels to provider-specific values.
    pub effort_map: HashMap<String, String>,
    /// Responses-only visible reasoning summary mode (off/auto/detailed).
    /// This is response control and is excluded from cache identity hashes.
    #[serde(default)]
    pub responses_reasoning_summary: Option<String>,
    /// Declarative cache policy. `None` (default) = all cache behavior off.
    /// Old configs that omit this field parse identically (`Option` defaults
    /// to `None`); serde ignores unknown fields, so existing config.toml
    /// files need no change.
    #[serde(default)]
    pub cache_policy: Option<CachePolicy>,
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
    #[expect(
        dead_code,
        reason = "model behavior setting is retained for compatibility"
    )]
    pub toolcall_requires_reasoning: bool,
    /// Alternative names for this model (e.g. "deepseek-chat" → "deepseek-v4-pro").
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Upstream wire protocol. Legacy profiles default to Chat Completions.
    #[serde(default)]
    pub wire_api: WireApi,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum WireApi {
    #[serde(rename = "chat_completions")]
    #[default]
    ChatCompletions,
    Responses,
}

#[derive(Debug, Clone)]
#[expect(
    dead_code,
    reason = "configuration fields are retained for runtime integrations"
)]
pub struct Config {
    pub listen_addr: String,
    pub eswitch_url: String,
    pub api_key: String,
    pub log_level: String,
    pub model_mapping: HashMap<String, String>,
    pub default_model: String,
    pub model_profiles: Vec<ModelProfile>,
    pub providers: HashMap<String, ProviderConfig>,
    /// Official Moonshot (Kimi For Coding) upstream base URL, no path suffix
    /// (client appends /v1/chat_completions etc.). Env: MOONSHOT_OFFICIAL_URL.
    pub moonshot_official_url: String,
    /// Official Moonshot API key. Env: MOONSHOT_OFFICIAL_API_KEY.
    pub moonshot_official_api_key: String,
    /// Index: model name (or alias) → index into model_profiles
    pub(crate) profile_by_name: HashMap<String, usize>,
}

/// Recognized upstream binding names for `CachePolicy.upstream`.
/// `None` (default) keeps eswitch routing.
const KNOWN_UPSTREAMS: &[&str] = &["official"];

/// A declared legacy provider-name alias.
///
/// Maps a historical provider name to the canonical `[providers]` key, and
/// carries the *pre-policy default upstream binding* for that name — the
/// routing the merged PR #4 `select_client` string match produced before
/// Phase 3. This is data (config-resolution metadata), not request-path
/// routing logic.
struct ProviderAlias {
    /// Legacy provider name as referenced by `model_profiles[].provider`.
    alias: &'static str,
    /// Canonical provider key in `[providers]`.
    canonical: &'static str,
    /// Default upstream binding for this alias name, applied only when the
    /// resolved provider declares no explicit `cache_policy.upstream`.
    /// `Some("official")` preserves the pre-policy official Moonshot (Kimi
    /// For Coding) routing for configs that still use the legacy name — with
    /// or without an explicit `[providers.moonshot-official]` block. `None`
    /// means "no default binding" (default eswitch routing).
    default_upstream: Option<&'static str>,
}

/// Provider-name aliases that resolve to a canonical provider.
///
/// P3-A canonicalizes the three historically inconsistent names
/// (`fireworks` / `moonshot` / `moonshot-official`) to the single canonical
/// `moonshot` provider. `moonshot-official` was the pre-policy provider name
/// routed to the official Moonshot (Kimi For Coding) upstream by the merged
/// PR #4 `select_client` string match; keeping it here as a *declared alias*
/// (data, not routing logic) means configs that still name their provider
/// that way resolve to the canonical config — reasoning fields, effort map
/// and cache policy — instead of failing lookup, and keep their pre-policy
/// `official` upstream binding unless they declare an explicit
/// `cache_policy.upstream`. An explicitly-defined
/// `[providers.moonshot-official]` block takes precedence over the alias for
/// config lookup; for routing, an explicit `cache_policy.upstream` takes
/// precedence over the alias's default binding.
const PROVIDER_ALIASES: &[ProviderAlias] = &[ProviderAlias {
    alias: "moonshot-official",
    canonical: "moonshot",
    default_upstream: Some("official"),
}];

impl Config {
    pub fn from_env() -> Self {
        let (model_mapping, default_model, model_profiles, providers) = Self::load_model_config();

        let mut config = Self {
            listen_addr: env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:11435".to_string()),
            eswitch_url: env::var("ESWITCH_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string()),
            api_key: env::var("DEEPSEEK_API_KEY").unwrap_or_else(|_| "not-needed".to_string()),
            log_level: env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
            moonshot_official_url: env::var("MOONSHOT_OFFICIAL_URL")
                .unwrap_or_else(|_| "https://api.kimi.com/coding".to_string()),
            moonshot_official_api_key: env::var("MOONSHOT_OFFICIAL_API_KEY").unwrap_or_default(),
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
    ///
    /// Resolves the declared [`PROVIDER_ALIASES`] (e.g. the legacy
    /// `moonshot-official`) to the canonical provider config. An explicitly
    /// configured provider block always takes precedence over an alias.
    pub fn provider_config(&self, provider: &str) -> Option<&ProviderConfig> {
        self.providers.get(provider).or_else(|| {
            PROVIDER_ALIASES
                .iter()
                .find(|alias| alias.alias == provider)
                .and_then(|alias| self.providers.get(alias.canonical))
        })
    }

    /// Resolve the effective upstream binding for a provider as referenced by
    /// a profile.
    ///
    /// Declarative and data-driven — there is no provider-name string match in
    /// request routing (G7). Priority:
    /// 1. an explicit `cache_policy.upstream` on the resolved provider config
    ///    always wins;
    /// 2. otherwise a *declared* default upstream binding for the provider
    ///    name (e.g. the legacy `moonshot-official` → `official`) applies —
    ///    this preserves the pre-policy official Moonshot (Kimi For Coding)
    ///    routing for configs that still use the legacy name, even when an
    ///    explicit `[providers.moonshot-official]` block shadows the alias
    ///    (report 58 M1);
    /// 3. otherwise `None` = default eswitch routing.
    pub fn effective_upstream_binding(&self, provider: &str) -> Option<&str> {
        if let Some(upstream) = self
            .provider_config(provider)
            .and_then(|pc| pc.cache_policy.as_ref())
            .and_then(|cp| cp.upstream.as_deref())
        {
            return Some(upstream);
        }
        PROVIDER_ALIASES
            .iter()
            .find(|alias| alias.alias == provider)
            .and_then(|alias| alias.default_upstream)
    }

    /// Return the configured wire API for a canonical model name or alias.
    pub fn wire_api_for_model(&self, model: &str) -> WireApi {
        self.model_profile(model)
            .map(|profile| profile.wire_api.clone())
            .unwrap_or_default()
    }

    /// Build the profile_by_name index from model_profiles (names + aliases).
    fn build_profile_index(&mut self) {
        for (i, profile) in self.model_profiles.iter().enumerate() {
            self.profile_by_name.insert(profile.name.clone(), i);
            for alias in &profile.aliases {
                self.profile_by_name.insert(alias.clone(), i);
            }
        }
    }

    /// Startup validation: panic on misconfiguration (no silent failures).
    fn validate(&self) {
        // 1. Each model_profile.provider must resolve to a [providers] entry
        //    (directly or via a declared provider alias).
        for profile in &self.model_profiles {
            if self.provider_config(&profile.provider).is_none() {
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
        if !self.profile_by_name.contains_key(&self.default_model) {
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
        //    Exception: thinking_param can be None for providers that don't support thinking (e.g. gpt)
        //    (resolved alias-aware so legacy provider names get the same safety checks)
        for profile in &self.model_profiles {
            if profile.reasoning_enabled {
                if let Some(provider) = self.provider_config(&profile.provider) {
                    if !provider.disable_thinking {
                        if provider.thinking_param.is_none() {
                            // OK: provider intentionally doesn't support thinking (e.g. gpt-5.6)
                            // Skip thinking_type checks since they're not applicable
                            continue;
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
        //    (resolved alias-aware so legacy provider names get the same safety checks)
        for profile in &self.model_profiles {
            if profile.reasoning_replay {
                if let Some(provider) = self.provider_config(&profile.provider) {
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

        // 7. cache_policy.upstream, when present, must name a known upstream
        //    binding. `None` (default, cache behavior fully off) is always
        //    valid and keeps the legacy eswitch routing. This is deliberately
        //    NOT an effort fail-fast: Kimi effort enum validation is a Phase 3
        //    startup behavior and must not be enabled in this phase.
        for (name, provider) in &self.providers {
            if let Some(policy) = &provider.cache_policy {
                if let Some(upstream) = &policy.upstream {
                    if !KNOWN_UPSTREAMS.contains(&upstream.as_str()) {
                        panic!(
                            "Provider '{}' cache_policy.upstream '{}' is not a \
                             known upstream binding. Known upstreams: {:?}",
                            name, upstream, KNOWN_UPSTREAMS
                        );
                    }
                }
            }
        }

        // 8. cache_policy.effort_enum, when declared, must accept every
        //    effort_map output value (P3-B fail-fast, no silent normalization).
        //    Only opt-in providers are validated: `None` — the default and the
        //    state of every current config — keeps legacy maps with
        //    `medium`/`xhigh`/`none` outputs valid unchanged, so non-opt-in
        //    and default-off profiles stay byte-for-byte backward compatible.
        for (name, provider) in &self.providers {
            if let Some(policy) = &provider.cache_policy {
                policy.validate_effort_enum(name, &provider.effort_map);
            }
        }
    }

    /// Load model config from file with priority:
    /// 1. Env var MODEL_CONFIG_PATH
    /// 2. /etc/cc-proxy/config.toml
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
                let etc = PathBuf::from("/etc/cc-proxy/config.toml");
                if etc.exists() {
                    Some(etc)
                } else {
                    None
                }
            });

        if let Some(path) = config_path {
            match fs::read_to_string(&path) {
                Ok(contents) => match toml::from_str::<ConfigFile>(&contents) {
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
                },
                Err(e) => {
                    panic!("Failed to read config file {}: {}", path.display(), e);
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
        mapping.insert(
            "claude-sonnet-4-6".to_string(),
            "deepseek-v4-flash".to_string(),
        );
        mapping.insert(
            "claude-sonnet-4-5".to_string(),
            "deepseek-v4-flash".to_string(),
        );
        mapping.insert(
            "claude-sonnet-4".to_string(),
            "deepseek-v4-flash".to_string(),
        );
        mapping.insert(
            "claude-3-5-sonnet".to_string(),
            "deepseek-v4-flash".to_string(),
        );
        // Claude Haiku variants → v4-flash (lighter, cost-effective)
        mapping.insert(
            "claude-haiku-4-5".to_string(),
            "deepseek-v4-flash".to_string(),
        );
        mapping.insert(
            "claude-haiku-4".to_string(),
            "deepseek-v4-flash".to_string(),
        );
        mapping.insert(
            "claude-3-haiku".to_string(),
            "deepseek-v4-flash".to_string(),
        );
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
                responses_reasoning_summary: None,
                cache_policy: None,
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
                responses_reasoning_summary: None,
                cache_policy: None,
            },
        );

        // Moonshot (kimi-k3): reasoning field, no thinking.type, effort passthrough.
        // Phase 3 (P3-A) canonicalized the provider name from the legacy
        // "fireworks" to "moonshot" — the single canonical name shared with
        // config.toml and the routing policy. `moonshot-official` remains a
        // declared alias (see PROVIDER_ALIASES) so pre-policy configs keep
        // resolving cleanly.
        providers.insert(
            "moonshot".to_string(),
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
                responses_reasoning_summary: None,
                cache_policy: None,
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
                aliases: vec!["deepseek-chat".to_string(), "deepseek-reasoner".to_string()],
                wire_api: WireApi::ChatCompletions,
            },
            ModelProfile {
                name: "deepseek-v4-flash".to_string(),
                provider: "deepseek".to_string(),
                reasoning_enabled: true,
                reasoning_replay: true,
                toolcall_requires_reasoning: true,
                aliases: vec![],
                wire_api: WireApi::ChatCompletions,
            },
            ModelProfile {
                name: "glm-5.2".to_string(),
                provider: "glm".to_string(),
                reasoning_enabled: true,
                reasoning_replay: false,
                toolcall_requires_reasoning: false,
                aliases: vec![],
                wire_api: WireApi::ChatCompletions,
            },
            ModelProfile {
                name: "kimi-k3".to_string(),
                provider: "moonshot".to_string(),
                reasoning_enabled: true,
                reasoning_replay: true,
                toolcall_requires_reasoning: false,
                aliases: vec![],
                wire_api: WireApi::ChatCompletions,
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::UsagePolicy;

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

[providers.moonshot]
reasoning_field = "reasoning"
disable_thinking = true
effort_param = "reasoning_effort"

[providers.moonshot.effort_map]
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
provider = "moonshot"
reasoning_enabled = true
reasoning_replay = true
aliases = []
"#;

        let cf: ConfigFile = toml::from_str(toml_str).expect("TOML should parse");

        // Verify providers are parsed
        let providers = cf.providers.expect("providers should be Some");
        assert_eq!(providers.len(), 2);
        assert!(providers.contains_key("deepseek"));
        assert!(providers.contains_key("moonshot"));

        // Verify provider fields
        let ds = &providers["deepseek"];
        assert_eq!(ds.reasoning_field, "reasoning_content");
        assert_eq!(ds.thinking_param.as_deref(), Some("thinking"));
        assert!(!ds.disable_thinking);
        // Old config without a cache_policy block stays fully off (None).
        assert!(
            ds.cache_policy.is_none(),
            "legacy provider without cache_policy must deserialize to None (cache off)"
        );

        let fw = &providers["moonshot"];
        assert_eq!(fw.reasoning_field, "reasoning");
        assert_eq!(fw.reasoning_field_alt, Vec::<String>::new());
        assert!(fw.disable_thinking);
        assert!(fw.thinking_param.is_none());
        assert!(
            fw.cache_policy.is_none(),
            "legacy provider without cache_policy must deserialize to None (cache off)"
        );

        // Verify model_profiles are parsed
        let profiles = cf.model_profiles.expect("model_profiles should be Some");
        assert_eq!(profiles.len(), 2);
        assert_eq!(profiles[0].name, "deepseek-v4-pro");
        assert_eq!(profiles[0].provider, "deepseek");
        assert_eq!(profiles[1].name, "kimi-k3");
        assert_eq!(profiles[1].provider, "moonshot");
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
                responses_reasoning_summary: None,
                cache_policy: None,
            },
        );

        let model_profiles = vec![ModelProfile {
            name: "bad-model".to_string(),
            provider: "nonexistent".to_string(), // does not exist in providers
            reasoning_enabled: false,
            reasoning_replay: false,
            toolcall_requires_reasoning: false,
            aliases: vec![],
            wire_api: WireApi::ChatCompletions,
        }];

        let mut config = Config {
            listen_addr: "0.0.0.0:11435".to_string(),
            eswitch_url: "http://127.0.0.1:11434".to_string(),
            moonshot_official_url: String::new(),
            moonshot_official_api_key: String::new(),
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
        // Verify that moonshot (no thinking_param) parses correctly
        let toml_str = r#"
[providers.moonshot]
reasoning_field = "reasoning"
disable_thinking = true
effort_param = "reasoning_effort"

[providers.moonshot.effort_map]
low = "low"
high = "high"
max = "max"

[[model_profiles]]
name = "kimi-k3"
provider = "moonshot"
reasoning_enabled = true
"#;

        let cf: ConfigFile = toml::from_str(toml_str).expect("TOML should parse");
        let providers = cf.providers.expect("providers should be Some");
        let fw = &providers["moonshot"];
        assert_eq!(fw.reasoning_field, "reasoning");
        assert!(
            fw.thinking_param.is_none(),
            "thinking_param should be None for moonshot"
        );
        assert!(fw.thinking_type_enabled.is_none());
        assert!(fw.thinking_type_disabled.is_none());
    }

    #[test]
    fn wire_api_is_responses_only_for_gpt_profile_and_chat_by_default() {
        let profiles = vec![
            ModelProfile {
                name: "gpt-5.6-luna".to_string(),
                provider: "gpt".to_string(),
                reasoning_enabled: true,
                reasoning_replay: false,
                toolcall_requires_reasoning: false,
                aliases: vec!["claude-sonnet-4-6".to_string()],
                wire_api: WireApi::Responses,
            },
            ModelProfile {
                name: "deepseek-v4-pro".to_string(),
                provider: "deepseek".to_string(),
                reasoning_enabled: true,
                reasoning_replay: true,
                toolcall_requires_reasoning: true,
                aliases: vec!["deepseek-chat".to_string()],
                wire_api: WireApi::ChatCompletions,
            },
            ModelProfile {
                name: "glm-5.2".to_string(),
                provider: "glm".to_string(),
                reasoning_enabled: true,
                reasoning_replay: false,
                toolcall_requires_reasoning: false,
                aliases: vec![],
                wire_api: WireApi::ChatCompletions,
            },
            ModelProfile {
                name: "kimi-k3".to_string(),
                provider: "moonshot".to_string(),
                reasoning_enabled: true,
                reasoning_replay: true,
                toolcall_requires_reasoning: false,
                aliases: vec![],
                wire_api: WireApi::ChatCompletions,
            },
        ];
        let mut config = Config {
            listen_addr: String::new(),
            eswitch_url: String::new(),
            moonshot_official_url: String::new(),
            moonshot_official_api_key: String::new(),
            api_key: String::new(),
            log_level: String::new(),
            model_mapping: HashMap::new(),
            default_model: "deepseek-v4-pro".to_string(),
            model_profiles: profiles,
            providers: HashMap::new(),
            profile_by_name: HashMap::new(),
        };
        config.build_profile_index();

        assert_eq!(
            config.wire_api_for_model("gpt-5.6-luna"),
            WireApi::Responses
        );
        assert_eq!(
            config.wire_api_for_model("claude-sonnet-4-6"),
            WireApi::Responses
        );
        assert_eq!(
            config.wire_api_for_model("deepseek-chat"),
            WireApi::ChatCompletions
        );
        assert_eq!(
            config.wire_api_for_model("glm-5.2"),
            WireApi::ChatCompletions
        );
        assert_eq!(
            config.wire_api_for_model("kimi-k3"),
            WireApi::ChatCompletions
        );
        assert_eq!(
            config.wire_api_for_model("unlisted-model"),
            WireApi::ChatCompletions
        );
    }

    #[test]
    fn cache_policy_parses_declared_and_defaults_off() {
        // A provider that declares a cache_policy block parses it; a provider
        // with an (empty) policy block stays off. No config.toml in this
        // phase declares one — this is the opt-in path's parse contract.
        let toml_str = r#"
[providers.deepseek]
reasoning_field = "reasoning_content"
disable_thinking = false
effort_param = "reasoning_effort"

[providers.deepseek.effort_map]
low = "high"
high = "high"
max = "max"

[providers.deepseek.cache_policy]
usage = "top_level_cached_tokens"

[providers.glm]
reasoning_field = "reasoning_content"
disable_thinking = false
effort_param = "reasoning_effort"

[providers.glm.effort_map]
high = "high"
max = "max"

[providers.glm.cache_policy]
"#;

        let cf: ConfigFile = toml::from_str(toml_str).expect("TOML should parse");
        let providers = cf.providers.expect("providers should be Some");

        let ds = &providers["deepseek"];
        let policy = ds
            .cache_policy
            .as_ref()
            .expect("declared cache_policy should deserialize");
        assert_eq!(policy.usage, UsagePolicy::TopLevelCachedTokens);
        assert_eq!(
            policy.upstream, None,
            "upstream defaults to None (eswitch routing)"
        );
        assert!(
            policy.cache_usage_enabled(),
            "naming a usage source opts into cache-usage telemetry"
        );

        // An empty policy block is still default-off.
        let glm = &providers["glm"];
        let policy = glm
            .cache_policy
            .as_ref()
            .expect("empty cache_policy parses");
        assert_eq!(policy.usage, UsagePolicy::Off);
        assert_eq!(policy.upstream, None);
        assert!(
            !policy.cache_usage_enabled(),
            "empty policy stays cache-off"
        );
    }

    #[test]
    #[should_panic(expected = "cache_policy.upstream")]
    fn validate_rejects_unknown_cache_policy_upstream() {
        let mut providers = HashMap::new();
        providers.insert(
            "moonshot".to_string(),
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
                    m.insert("high".to_string(), "high".to_string());
                    m.insert("max".to_string(), "max".to_string());
                    m
                },
                responses_reasoning_summary: None,
                cache_policy: Some(CachePolicy {
                    usage: UsagePolicy::TopLevelCachedTokens,
                    prompt_cache_key_enabled: false,
                    upstream: Some("not-a-real-upstream".to_string()),
                    effort_enum: None,
                    replay: crate::cache::ReplayPolicy::Off,
                    history: crate::cache::HistoryPolicy::Off,
                    relocate: crate::cache::RelocatePolicy::Off,
                    pinned_effort: None,
                }),
            },
        );
        let model_profiles = vec![ModelProfile {
            name: "kimi-k3".to_string(),
            provider: "moonshot".to_string(),
            reasoning_enabled: true,
            reasoning_replay: true,
            toolcall_requires_reasoning: false,
            aliases: vec![],
            wire_api: WireApi::ChatCompletions,
        }];

        let mut config = Config {
            listen_addr: "0.0.0.0:11435".to_string(),
            eswitch_url: "http://127.0.0.1:11434".to_string(),
            moonshot_official_url: String::new(),
            moonshot_official_api_key: String::new(),
            api_key: "test".to_string(),
            log_level: "info".to_string(),
            model_mapping: HashMap::new(),
            default_model: "kimi-k3".to_string(),
            model_profiles,
            providers,
            profile_by_name: HashMap::new(),
        };
        config.build_profile_index();
        config.validate(); // should panic on unknown upstream binding
    }

    #[test]
    fn validate_accepts_known_cache_policy_upstream_and_none() {
        // upstream = None (the only state any built-in config uses) and the
        // known "official" binding both validate cleanly.
        let mut providers = HashMap::new();

        // Provider with no cache_policy at all (legacy state).
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
                responses_reasoning_summary: None,
                cache_policy: None,
            },
        );

        // Provider with a policy whose upstream binds the official upstream.
        providers.insert(
            "moonshot".to_string(),
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
                    m.insert("high".to_string(), "high".to_string());
                    m.insert("max".to_string(), "max".to_string());
                    m
                },
                responses_reasoning_summary: None,
                cache_policy: Some(CachePolicy {
                    usage: UsagePolicy::Off,
                    prompt_cache_key_enabled: false,
                    upstream: Some("official".to_string()),
                    effort_enum: None,
                    replay: crate::cache::ReplayPolicy::Off,
                    history: crate::cache::HistoryPolicy::Off,
                    relocate: crate::cache::RelocatePolicy::Off,
                    pinned_effort: None,
                }),
            },
        );

        let model_profiles = vec![
            ModelProfile {
                name: "deepseek-v4-pro".to_string(),
                provider: "deepseek".to_string(),
                reasoning_enabled: true,
                reasoning_replay: true,
                toolcall_requires_reasoning: true,
                aliases: vec![],
                wire_api: WireApi::ChatCompletions,
            },
            ModelProfile {
                name: "kimi-k3".to_string(),
                provider: "moonshot".to_string(),
                reasoning_enabled: true,
                reasoning_replay: true,
                toolcall_requires_reasoning: false,
                aliases: vec![],
                wire_api: WireApi::ChatCompletions,
            },
        ];

        let mut config = Config {
            listen_addr: "0.0.0.0:11435".to_string(),
            eswitch_url: "http://127.0.0.1:11434".to_string(),
            moonshot_official_url: String::new(),
            moonshot_official_api_key: String::new(),
            api_key: "test".to_string(),
            log_level: "info".to_string(),
            model_mapping: HashMap::new(),
            default_model: "deepseek-v4-pro".to_string(),
            model_profiles,
            providers,
            profile_by_name: HashMap::new(),
        };
        config.build_profile_index();
        config.validate(); // no panic expected
    }

    // --- Phase 3 (P3-A): provider-name canonicalization (C11) ---

    #[test]
    fn builtin_defaults_use_single_canonical_moonshot_provider() {
        // The three inconsistent names (fireworks / moonshot /
        // moonshot-official) must be unified to the single canonical
        // `moonshot` provider. No duplicate provider names may remain.
        let providers = Config::builtin_default_providers();
        assert!(
            providers.contains_key("moonshot"),
            "canonical provider 'moonshot' must exist in builtin defaults"
        );
        assert!(
            !providers.contains_key("fireworks"),
            "legacy 'fireworks' provider name must be canonicalized away"
        );
        let profiles = Config::builtin_default_profiles();
        let kimi = profiles
            .iter()
            .find(|p| p.name == "kimi-k3")
            .expect("kimi-k3 profile exists in builtin defaults");
        assert_eq!(
            kimi.provider, "moonshot",
            "kimi-k3 must reference the canonical 'moonshot' provider"
        );
        // No profile may reference the legacy provider names.
        for profile in &profiles {
            assert_ne!(
                profile.provider, "fireworks",
                "no profile may reference legacy 'fireworks'"
            );
        }
    }

    #[test]
    fn provider_alias_resolves_moonshot_official_to_canonical_and_validates() {
        // `moonshot-official` was the pre-policy provider name routed to the
        // official Moonshot upstream in the merged PR #4 `select_client`
        // string match. P3-A keeps it valid as a *declared alias* of the
        // canonical `moonshot` provider (data, not routing logic): a profile
        // that still names it resolves to the canonical provider config and
        // validates cleanly.
        let mut providers = HashMap::new();
        providers.insert(
            "moonshot".to_string(),
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
                    m.insert("high".to_string(), "high".to_string());
                    m.insert("max".to_string(), "max".to_string());
                    m
                },
                responses_reasoning_summary: None,
                cache_policy: None,
            },
        );
        let model_profiles = vec![
            ModelProfile {
                name: "kimi-k3".to_string(),
                provider: "moonshot".to_string(),
                reasoning_enabled: true,
                reasoning_replay: true,
                toolcall_requires_reasoning: false,
                aliases: vec![],
                wire_api: WireApi::ChatCompletions,
            },
            // A legacy profile that still names the provider "moonshot-official".
            ModelProfile {
                name: "kimi-k3-legacy".to_string(),
                provider: "moonshot-official".to_string(),
                reasoning_enabled: true,
                reasoning_replay: true,
                toolcall_requires_reasoning: false,
                aliases: vec![],
                wire_api: WireApi::ChatCompletions,
            },
        ];

        let mut config = Config {
            listen_addr: "0.0.0.0:11435".to_string(),
            eswitch_url: "http://127.0.0.1:11434".to_string(),
            moonshot_official_url: String::new(),
            moonshot_official_api_key: String::new(),
            api_key: "test".to_string(),
            log_level: "info".to_string(),
            model_mapping: HashMap::new(),
            default_model: "kimi-k3".to_string(),
            model_profiles,
            providers,
            profile_by_name: HashMap::new(),
        };
        config.build_profile_index();
        config.validate(); // must not panic on the aliased profile provider

        // The alias resolves to the canonical provider config.
        let canonical = config
            .provider_config("moonshot")
            .expect("canonical provider resolves");
        let via_alias = config
            .provider_config("moonshot-official")
            .expect("legacy provider alias resolves to canonical config");
        assert_eq!(
            canonical.reasoning_field, via_alias.reasoning_field,
            "alias must resolve to the canonical provider's reasoning_field"
        );
        assert_eq!(
            canonical.effort_map, via_alias.effort_map,
            "alias must resolve to the canonical provider's effort_map"
        );
    }

    #[test]
    #[should_panic(expected = "reasoning_field is empty")]
    fn validate_applies_safety_checks_to_alias_resolved_provider() {
        // report 58 S1: validate() steps 3/4 must resolve the provider
        // alias-aware (via `provider_config`), not via a direct
        // `providers.get(profile.provider)` lookup. A legacy profile that
        // names `moonshot-official` — with no explicit
        // `[providers.moonshot-official]` block — resolves to the canonical
        // `moonshot` provider config, so the reasoning_field safety check
        // must still fire there. A direct lookup would silently miss the
        // aliased name and let the invalid config through.
        let mut providers = HashMap::new();
        providers.insert(
            "moonshot".to_string(),
            ProviderConfig {
                reasoning_field: String::new(), // empty → must fail step 4
                reasoning_field_alt: vec![],
                thinking_param: None,
                thinking_type_enabled: None,
                thinking_type_disabled: None,
                disable_thinking: true,
                effort_param: "reasoning_effort".to_string(),
                effort_map: {
                    let mut m = HashMap::new();
                    m.insert("low".to_string(), "low".to_string());
                    m.insert("high".to_string(), "high".to_string());
                    m.insert("max".to_string(), "max".to_string());
                    m
                },
                responses_reasoning_summary: None,
                cache_policy: None,
            },
        );
        let model_profiles = vec![ModelProfile {
            name: "kimi-k3-legacy".to_string(),
            provider: "moonshot-official".to_string(), // alias, no explicit block
            reasoning_enabled: true,
            reasoning_replay: true, // forces the step-4 reasoning_field check
            toolcall_requires_reasoning: false,
            aliases: vec![],
            wire_api: WireApi::ChatCompletions,
        }];
        let mut config = Config {
            listen_addr: "0.0.0.0:11435".to_string(),
            eswitch_url: "http://127.0.0.1:11434".to_string(),
            moonshot_official_url: String::new(),
            moonshot_official_api_key: String::new(),
            api_key: "test".to_string(),
            log_level: "info".to_string(),
            model_mapping: HashMap::new(),
            default_model: "kimi-k3-legacy".to_string(),
            model_profiles,
            providers,
            profile_by_name: HashMap::new(),
        };
        config.build_profile_index();
        config.validate(); // must panic on the aliased provider's empty reasoning_field
    }

    // --- Phase 3 (P3-B): cache_policy.effort_enum validation (C12) ---

    /// Build a Config with a single canonical `moonshot` provider whose
    /// `cache_policy` and `effort_map` are supplied by the caller, wired to a
    /// `kimi-k3` profile, and fully validated.
    fn moonshot_config_with_policy(
        policy: CachePolicy,
        effort_map: HashMap<String, String>,
    ) -> Config {
        let mut providers = HashMap::new();
        providers.insert(
            "moonshot".to_string(),
            ProviderConfig {
                reasoning_field: "reasoning".to_string(),
                reasoning_field_alt: vec![],
                thinking_param: None,
                thinking_type_enabled: None,
                thinking_type_disabled: None,
                disable_thinking: true,
                effort_param: "reasoning_effort".to_string(),
                effort_map,
                responses_reasoning_summary: None,
                cache_policy: Some(policy),
            },
        );
        let model_profiles = vec![ModelProfile {
            name: "kimi-k3".to_string(),
            provider: "moonshot".to_string(),
            reasoning_enabled: true,
            reasoning_replay: true,
            toolcall_requires_reasoning: false,
            aliases: vec![],
            wire_api: WireApi::ChatCompletions,
        }];
        let mut config = Config {
            listen_addr: "0.0.0.0:11435".to_string(),
            eswitch_url: "http://127.0.0.1:11434".to_string(),
            moonshot_official_url: String::new(),
            moonshot_official_api_key: String::new(),
            api_key: "test".to_string(),
            log_level: "info".to_string(),
            model_mapping: HashMap::new(),
            default_model: "kimi-k3".to_string(),
            model_profiles,
            providers,
            profile_by_name: HashMap::new(),
        };
        config.build_profile_index();
        config
    }

    fn effort_map_with(extra: &[(&str, &str)]) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("low".to_string(), "low".to_string());
        m.insert("high".to_string(), "high".to_string());
        m.insert("max".to_string(), "max".to_string());
        for (k, v) in extra {
            m.insert((*k).to_string(), (*v).to_string());
        }
        m
    }

    fn kimi_effort_enum() -> Vec<String> {
        vec!["low".to_string(), "high".to_string(), "max".to_string()]
    }

    #[test]
    fn validate_accepts_kimi_effort_enum_with_legal_outputs() {
        // An explicit Kimi enum {low,high,max} whose effort_map outputs all
        // stay inside the set validates cleanly at startup.
        let policy = CachePolicy {
            usage: UsagePolicy::TopLevelCachedTokens,
            prompt_cache_key_enabled: false,
            upstream: Some("official".to_string()),
            effort_enum: Some(kimi_effort_enum()),
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: crate::cache::RelocatePolicy::Off,
            pinned_effort: None,
        };
        let config = moonshot_config_with_policy(policy, effort_map_with(&[]));
        config.validate(); // must not panic
    }

    #[test]
    #[should_panic(expected = "effort_enum")]
    fn validate_rejects_effort_map_output_outside_declared_enum() {
        // A `medium` output under the explicit Kimi enum is an illegal wire
        // effort: validate() must fail fast at startup, never silently
        // normalize. This is the P3-B opt-in gate (and the exact risk the
        // current config.toml would hit if it ever declared effort_enum).
        let policy = CachePolicy {
            usage: UsagePolicy::TopLevelCachedTokens,
            prompt_cache_key_enabled: false,
            upstream: Some("official".to_string()),
            effort_enum: Some(kimi_effort_enum()),
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: crate::cache::RelocatePolicy::Off,
            pinned_effort: None,
        };
        let config = moonshot_config_with_policy(policy, effort_map_with(&[("medium", "medium")]));
        config.validate(); // should panic on the illegal medium output
    }

    #[test]
    #[should_panic(expected = "effort_enum")]
    fn validate_rejects_xhigh_output_under_explicit_enum() {
        // The OSS `xhigh` acceptance is deliberately NOT copied: an xhigh
        // output under the Kimi enum is illegal and fails fast.
        let policy = CachePolicy {
            usage: UsagePolicy::TopLevelCachedTokens,
            prompt_cache_key_enabled: false,
            upstream: Some("official".to_string()),
            effort_enum: Some(kimi_effort_enum()),
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: crate::cache::RelocatePolicy::Off,
            pinned_effort: None,
        };
        let config = moonshot_config_with_policy(policy, effort_map_with(&[("xhigh", "xhigh")]));
        config.validate(); // should panic on the illegal xhigh output
    }

    #[test]
    fn validate_keeps_legacy_effort_maps_valid_without_enum() {
        // No explicit enum (None — the state of every current config): a
        // policy-bound provider whose effort_map still emits medium/xhigh/none
        // stays valid. Non-opt-in and default-off profiles are unchanged.
        let policy = CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: None,
            effort_enum: None,
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: crate::cache::RelocatePolicy::Off,
            pinned_effort: None,
        };
        let config = moonshot_config_with_policy(
            policy,
            effort_map_with(&[("medium", "medium"), ("xhigh", "max"), ("none", "none")]),
        );
        config.validate(); // must not panic
    }

    // --- Phase 4a: pinned_effort config validation (T12/T13) ---

    #[test]
    fn validate_accepts_legal_pinned_effort() {
        // A pin inside the declared Kimi enum {low,high,max} with a matching
        // effort_map validates cleanly at startup (opt-in only).
        for pin in ["low", "high", "max"] {
            let policy = CachePolicy {
                usage: UsagePolicy::Off,
                prompt_cache_key_enabled: false,
                upstream: Some("official".to_string()),
                effort_enum: Some(kimi_effort_enum()),
                replay: crate::cache::ReplayPolicy::Off,
                history: crate::cache::HistoryPolicy::Off,
                relocate: crate::cache::RelocatePolicy::Off,
                pinned_effort: Some(pin.to_string()),
            };
            let config = moonshot_config_with_policy(policy, effort_map_with(&[]));
            config.validate(); // must not panic
        }
    }

    #[test]
    #[should_panic(expected = "effort_enum")]
    fn validate_rejects_pinned_effort_without_declared_enum() {
        // Fail-closed: a pin is a wire-effort promise and must be validated
        // against an explicit legal set — it can never be declared without
        // one (T12: config validation is never silent).
        let policy = CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: Some("official".to_string()),
            effort_enum: None,
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: crate::cache::RelocatePolicy::Off,
            pinned_effort: Some("high".to_string()),
        };
        let config = moonshot_config_with_policy(policy, effort_map_with(&[]));
        config.validate(); // should panic on the un-validatable pin
    }

    #[test]
    #[should_panic(expected = "effort_enum")]
    fn validate_rejects_pinned_effort_outside_declared_enum() {
        // `medium` is not in the official Kimi set {low,high,max}: an invalid
        // pin must fail fast at startup, never be silently normalized or
        // coerced to a different effort on the wire.
        let policy = CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: Some("official".to_string()),
            effort_enum: Some(kimi_effort_enum()),
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: crate::cache::RelocatePolicy::Off,
            pinned_effort: Some("medium".to_string()),
        };
        let config = moonshot_config_with_policy(policy, effort_map_with(&[]));
        config.validate(); // should panic on the illegal pinned effort
    }

    #[test]
    fn validate_keeps_legacy_configs_valid_without_pin() {
        // No pinned_effort (the state of every current config and the
        // default-off `test_config()` fixtures): validation is unchanged and
        // legacy effort maps with medium/xhigh/none outputs stay valid.
        let policy = CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: None,
            effort_enum: None,
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: crate::cache::RelocatePolicy::Off,
            pinned_effort: None,
        };
        let config = moonshot_config_with_policy(
            policy,
            effort_map_with(&[("medium", "medium"), ("xhigh", "max"), ("none", "none")]),
        );
        config.validate(); // must not panic
    }

    // --- Phase 4a remediation (S1): pin must be an actual effort_map key ---

    #[test]
    #[should_panic(expected = "effort_map")]
    fn validate_rejects_pinned_effort_missing_from_effort_map() {
        // S1: the pin is a member of the declared effort_enum BUT not a key
        // of the provider's effort_map. Without this gate,
        // `apply_effort_direct`'s `effort_map.get(pin).unwrap_or("high")`
        // (converter.rs `_ =>` branch) would SILENTLY coerce the pin to
        // "high" on the wire, contradicting the "never silently coerced"
        // promise. A missing key must fail fast at startup.
        let policy = CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: Some("official".to_string()),
            effort_enum: Some(kimi_effort_enum()),
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: crate::cache::RelocatePolicy::Off,
            pinned_effort: Some("low".to_string()),
        };
        // effort_map deliberately missing the "low" key (only high/max).
        let mut map = HashMap::new();
        map.insert("high".to_string(), "high".to_string());
        map.insert("max".to_string(), "max".to_string());
        let config = moonshot_config_with_policy(policy, map);
        config.validate(); // should panic: pin "low" is not an effort_map key
    }

    #[test]
    fn validate_accepts_pinned_effort_that_is_an_effort_map_key() {
        // S1 remediation guard: a pin that IS both in the declared enum and a
        // key of the provider's effort_map validates cleanly (the shape of
        // every built-in moonshot effort_map).
        let policy = CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: Some("official".to_string()),
            effort_enum: Some(kimi_effort_enum()),
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: crate::cache::RelocatePolicy::Off,
            pinned_effort: Some("low".to_string()),
        };
        let config = moonshot_config_with_policy(policy, effort_map_with(&[]));
        config.validate(); // must not panic
    }
}
