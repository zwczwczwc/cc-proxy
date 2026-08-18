// Copyright (c) 2025 cc-proxy
//
// Provider-neutral raw cache telemetry.
//
// Phase 2 scope: pure mapping functions + a policy-gated selector. As of
// Phase 2b.3 the Responses path (`responses/response.rs`, `responses/stream.rs`)
// reads its cache-usage view through `responses_usage_view`, which selects
// exactly one of two mutually-exclusive modes per request:
//   * `CacheStatsMode::Legacy` (default — policy `None`/off): reproduces the
//     pre-existing Responses three-bucket arithmetic byte-for-byte, so
//     non-opt-in providers see zero log/wire change.
//   * `CacheStatsMode::Raw` (explicit `cache_policy.usage`): the canonical
//     `CacheStats` projection (`from_responses_usage`) with a guarded miss and
//     a raw hit rate.
// The Chat adapters (`from_chat_usage` / `from_optional_*`) remain unwired;
// Phase 2b.4 threads them behind the same gate.
//
// Semantics:
// - `input_tokens`          : prompt / input tokens as reported upstream.
// - `cache_read_tokens`     : tokens served from a cache hit (explicit read).
// - `cache_write_tokens`    : tokens newly written to cache. ONLY ever set from
//                             an explicit upstream write field; never derived
//                             from `prompt - cached` (that remainder is
//                             uncached/miss, not a write).
// - `cache_miss_tokens`     : tokens not served from cache. Explicit when the
//                             provider reports it (DeepSeek), otherwise derived
//                             as input - read - write. Unknown / inconsistent
//                             inputs (e.g. cached > prompt) stay `None` — a
//                             negative or fabricated value is never produced.
//
// Every bucket is an `Option` so "not reported by upstream" stays distinct from
// a real zero. The raw upstream fields live on the input `Usage`/`ResponsesUsage`
// structs; `CacheStats` is a normalized projection that keeps the source marker.

use crate::openai::types::Usage as ChatUsage;
use crate::responses::types::ResponsesUsage;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Which upstream usage shape produced a `CacheStats`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum CacheSource {
    /// OpenAI-compatible chat completions usage (Kimi / OpenAI / DeepSeek / GLM shapes).
    #[default]
    Chat,
    /// Responses API usage.
    Responses,
}

/// Raw, provider-neutral cache telemetry buckets.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct CacheStats {
    pub(crate) input_tokens: Option<u32>,
    pub(crate) cache_read_tokens: Option<u32>,
    pub(crate) cache_write_tokens: Option<u32>,
    pub(crate) cache_miss_tokens: Option<u32>,
    pub(crate) source: CacheSource,
}

// ---------------------------------------------------------------------------
// Declarative cache policy (default-off).
//
// Phase 2b.1 scope: define the type surface and default-off serde semantics
// only. Nothing here is wired into production logs or outbound wire behavior —
// `ProviderConfig.cache_policy` defaults to `None` for every built-in provider
// and no config.toml declares a policy, so every legacy path is preserved.
// Opt-in happens in later phases (2b.2+, Phase 3).
// ---------------------------------------------------------------------------

/// Where cache-usage telemetry should be sourced from for a provider.
///
/// `Off` (the default) means no cache-usage telemetry. Every other variant is
/// a named upstream field that a later phase's selector maps into
/// [`CacheStats`]. Variants are additive; a provider that has not opted in
/// stays `Off`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UsagePolicy {
    /// No cache-usage telemetry (default).
    #[default]
    Off,
    /// Kimi top-level `usage.cached_tokens`.
    TopLevelCachedTokens,
}

/// Declarative cache policy attached to a provider via
/// [`crate::config::ProviderConfig::cache_policy`].
///
/// Every field is `#[serde(default)]` and the whole policy is an `Option` on
/// the provider, so a missing policy — or a policy missing fields — is
/// equivalent to "cache behavior fully off".
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CachePolicy {
    /// Where cache-usage telemetry comes from. `Off` (default) = disabled.
    pub usage: UsagePolicy,
    /// Optional upstream binding name. `Some("official")` binds this provider
    /// to the official Moonshot (Kimi For Coding) upstream; `None` keeps the
    /// default (eswitch) routing. Phase 3 canonicalizes provider names and
    /// replaces the `select_client` string match with this binding.
    pub upstream: Option<String>,
}

impl CachePolicy {
    /// Whether this policy opts into cache-usage telemetry.
    ///
    /// `cache_usage_enabled() == false` is the default; a provider only reports
    /// cache usage once its policy names a concrete usage source.
    pub fn cache_usage_enabled(&self) -> bool {
        !matches!(self.usage, UsagePolicy::Off)
    }
}

// ---------------------------------------------------------------------------
// Deterministic, fail-closed session cache key (Phase 2b.2).
//
// Phase 2b.2 scope: pure helper + contract tests only. Nothing injects the
// key into outbound requests yet — Chat encoder injection is Phase 3
// behavior, and the Responses encoder is an explicit non-goal that never
// carries `prompt_cache_key` (Kimi rides the Chat wire).
// ---------------------------------------------------------------------------

/// Derive a deterministic, fail-closed session cache key.
///
/// Contract (KIMI-K3-CACHE-OPTIMIZATION-FINAL-PLAN §3.3, report 45/46):
///
/// ```text
/// session_key := sha256( upstream_provider | model | source_name | source_value )[..16]
/// ```
///
/// where `upstream_provider` combines `provider` with the optional policy
/// `upstream` binding (`"provider:upstream"` when bound, `"provider"` when
/// unbound), and this entry point names the `metadata.user_id` source.
///
/// - **Fail-closed**: no stable source ⇒ `None`. The key is NEVER derived
///   from a UUID / random / clock — a per-request nonce would guarantee a
///   cache miss, so the plan forbids it outright.
/// - **Deterministic & stateless**: identical inputs yield an identical key
///   across calls and across process restarts (pure function of inbound
///   signal; no shared state).
/// - **No plaintext**: the returned value is a hex digest of the first 16
///   bytes of the hash; user/token-like source text never appears in it, and
///   this function performs no I/O and writes no logs — only the outbound key
///   is returned.
///
/// Future inbound sources (e.g. a session-id header set by an ingress) should
/// extend this module with their own source label so keys stay namespaced by
/// source channel.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "fail-closed session key; wired in Phase 3 injection"
    )
)]
pub(crate) fn session_key_from_source(
    source: Option<&str>,
    provider: &str,
    model: &str,
    upstream: Option<&str>,
) -> Option<String> {
    let source_value = source?;
    let upstream_provider = match upstream {
        Some(binding) => format!("{provider}:{binding}"),
        None => provider.to_string(),
    };
    // Canonical framing of the plan's `upstream_provider | model |
    // source_name | source_value`. `source_name` is fixed to the metadata
    // source for this entry point.
    let canonical = format!("{upstream_provider}|{model}|metadata.user_id|{source_value}");
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    let digest = hasher.finalize();
    Some(hex::encode(&digest[..16]))
}

/// Map an OpenAI-compatible chat completions `Usage` into `CacheStats`.
///
/// Handles Kimi top-level `cached_tokens` (GAP-A, optional), Kimi/OpenAI nested
/// `prompt_tokens_details.cached_tokens`, and DeepSeek `prompt_cache_hit_tokens`
/// / `prompt_cache_miss_tokens`. Read priority: top-level → nested → hit.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "usage adapter; wired in a later phase")
)]
pub(crate) fn from_chat_usage(usage: &ChatUsage) -> CacheStats {
    let input = usage.prompt_tokens;
    let read = usage
        .cached_tokens
        .or_else(|| {
            usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens)
        })
        .or(usage.prompt_cache_hit_tokens);
    // Chat completions never report a cache write. The old `prompt - cached`
    // remainder was labeled "creation"; per the raw-semantics rule it is the
    // uncached (miss) remainder, NOT a write — so write stays None.
    let write = None;
    let miss = usage
        .prompt_cache_miss_tokens
        .or_else(|| derive_miss(input, read, write));
    CacheStats {
        input_tokens: input,
        cache_read_tokens: read,
        cache_write_tokens: write,
        cache_miss_tokens: miss,
        source: CacheSource::Chat,
    }
}

/// Map a Responses API `ResponsesUsage` into `CacheStats`.
///
/// `cache_write_tokens` only ever comes from an explicit upstream write field
/// (nested `input_tokens_details.cache_write_tokens`, falling back to the
/// top-level `cache_write_tokens`); the miss bucket is derived as the remainder.
pub(crate) fn from_responses_usage(usage: &ResponsesUsage) -> CacheStats {
    let details = usage.input_tokens_details.as_ref();
    let read = details.and_then(|details| details.cached_tokens);
    let write = details
        .and_then(|details| details.cache_write_tokens)
        .or(usage.cache_write_tokens);
    let miss = derive_miss(usage.input_tokens, read, write);
    CacheStats {
        input_tokens: usage.input_tokens,
        cache_read_tokens: read,
        cache_write_tokens: write,
        cache_miss_tokens: miss,
        source: CacheSource::Responses,
    }
}

/// Optional entry points: HTTP errors / timeouts carry no usage object, so they
/// must never surface as a cache miss. Absent usage ⇒ absent stats (never a
/// fabricated zero or miss bucket).
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "usage adapter; wired in a later phase")
)]
pub(crate) fn from_optional_chat_usage(usage: Option<&ChatUsage>) -> Option<CacheStats> {
    usage.map(from_chat_usage)
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "usage adapter; wired in a later phase")
)]
pub(crate) fn from_optional_responses_usage(usage: Option<&ResponsesUsage>) -> Option<CacheStats> {
    usage.map(from_responses_usage)
}

/// Derive the miss bucket as `input - read - write` when the data is consistent.
/// Returns `None` (unknown) on inconsistent input (e.g. cached > prompt) instead
/// of fabricating a negative or clamped value.
fn derive_miss(input: Option<u32>, read: Option<u32>, write: Option<u32>) -> Option<u32> {
    let (Some(input), Some(read)) = (input, read) else {
        return None;
    };
    let consumed = read.saturating_add(write.unwrap_or(0));
    // `then_some` would evaluate the subtraction eagerly; use an explicit guard
    // so inconsistent input (cached > prompt) never overflows.
    if input >= consumed {
        Some(input - consumed)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Responses usage view selector (Phase 2b.3).
//
// Exactly ONE mode is computed per request — `Legacy` or `Raw` — decided by
// `CacheStatsMode::from_policy`. The view is the single source for the log
// buckets (and, via its read/creation fields, the wire), so log and wire can
// never disagree and a request never reports cache usage twice.
// ---------------------------------------------------------------------------

/// How a request's cache-usage telemetry is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheStatsMode {
    /// Pre-existing Responses three-bucket arithmetic (clamped miss, hit rate
    /// `0.0` on zero input). Default: policy `None` or `usage: off`.
    Legacy,
    /// Canonical `CacheStats` projection (guarded miss, raw hit rate).
    Raw,
}

impl CacheStatsMode {
    /// The mode selected by an optional cache policy. `None`/off ⇒ `Legacy`;
    /// an explicit usage source ⇒ `Raw`. This is the only gate — there is no
    /// provider-string check anywhere (G7).
    pub(crate) fn from_policy(policy: Option<&CachePolicy>) -> Self {
        match policy {
            Some(policy) if policy.cache_usage_enabled() => CacheStatsMode::Raw,
            _ => CacheStatsMode::Legacy,
        }
    }
}

/// Normalized Responses cache-usage view (input / read / creation / miss /
/// hit rate).
///
/// `read` comes from `input_tokens_details.cached_tokens`; `creation` only from
/// an explicit `cache_write_tokens` (nested, then top-level fallback); `miss`
/// is derived and never negative; absent usage ⇒ an empty view (never a
/// fabricated miss — an HTTP error or a missing usage object is not a miss).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ResponsesUsageView {
    pub(crate) input: Option<u32>,
    pub(crate) read: Option<u32>,
    pub(crate) creation: Option<u32>,
    pub(crate) miss: Option<u32>,
    pub(crate) hit_rate: Option<f64>,
}

impl ResponsesUsageView {
    fn empty() -> Self {
        ResponsesUsageView {
            input: None,
            read: None,
            creation: None,
            miss: None,
            hit_rate: None,
        }
    }
}

/// Non-stream selector: map a typed `ResponsesUsage` (plus policy) into the
/// view. `None` usage (HTTP error / timeout / missing usage) yields an empty
/// view — never a miss bucket.
pub(crate) fn responses_usage_view(
    usage: Option<&ResponsesUsage>,
    policy: Option<&CachePolicy>,
) -> ResponsesUsageView {
    let Some(usage) = usage else {
        return ResponsesUsageView::empty();
    };
    match CacheStatsMode::from_policy(policy) {
        CacheStatsMode::Raw => {
            // Canonical raw projection (the `CacheStats` adapter), plus a raw
            // hit rate. Only reachable with an explicit policy usage source;
            // config.toml does not declare one in Phase 2b.
            let stats = from_responses_usage(usage);
            ResponsesUsageView {
                input: stats.input_tokens,
                read: stats.cache_read_tokens,
                creation: stats.cache_write_tokens,
                miss: stats.cache_miss_tokens,
                hit_rate: raw_hit_rate(stats.input_tokens, stats.cache_read_tokens),
            }
        }
        CacheStatsMode::Legacy => {
            let details = usage.input_tokens_details.as_ref();
            responses_usage_view_from_buckets(
                usage.input_tokens,
                details.and_then(|details| details.cached_tokens),
                details
                    .and_then(|details| details.cache_write_tokens)
                    .or(usage.cache_write_tokens),
                None,
            )
        }
    }
}

/// Stream selector: build the view from buckets already extracted by the SSE
/// terminal handler. Shares the same mode selection with
/// [`responses_usage_view`], so streamed and non-streamed responses report
/// cache usage identically for a given policy.
pub(crate) fn responses_usage_view_from_buckets(
    input: Option<u32>,
    read: Option<u32>,
    creation: Option<u32>,
    policy: Option<&CachePolicy>,
) -> ResponsesUsageView {
    match CacheStatsMode::from_policy(policy) {
        CacheStatsMode::Raw => ResponsesUsageView {
            input,
            read,
            creation,
            miss: derive_miss(input, read, creation),
            hit_rate: raw_hit_rate(input, read),
        },
        CacheStatsMode::Legacy => {
            let miss = input.map(|input| {
                input
                    .saturating_sub(read.unwrap_or(0))
                    .saturating_sub(creation.unwrap_or(0))
            });
            let hit_rate = input.map(|input| {
                if input == 0 {
                    0.0
                } else {
                    read.unwrap_or(0) as f64 / input as f64 * 100.0
                }
            });
            ResponsesUsageView {
                input,
                read,
                creation,
                miss,
                hit_rate,
            }
        }
    }
}

/// Raw hit rate: `read / input * 100`, `None` when input is zero/unknown or
/// read is unknown (unlike the legacy branch, which clamps zero input to `0.0`).
fn raw_hit_rate(input: Option<u32>, read: Option<u32>) -> Option<f64> {
    match (input, read) {
        (Some(input), Some(read)) if input > 0 => Some(read as f64 / input as f64 * 100.0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chat_usage(value: serde_json::Value) -> ChatUsage {
        serde_json::from_value(value).unwrap()
    }

    fn responses_usage(value: serde_json::Value) -> ResponsesUsage {
        serde_json::from_value(value).unwrap()
    }

    // --- Kimi shapes ---

    #[test]
    fn kimi_top_level_cached_tokens_maps_to_read() {
        let usage = chat_usage(json!({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cached_tokens": 70,
        }));
        let stats = from_chat_usage(&usage);
        assert_eq!(stats.source, CacheSource::Chat);
        assert_eq!(stats.input_tokens, Some(100));
        assert_eq!(stats.cache_read_tokens, Some(70));
        assert_eq!(
            stats.cache_write_tokens, None,
            "write must never be fabricated"
        );
        assert_eq!(stats.cache_miss_tokens, Some(30));
    }

    #[test]
    fn kimi_nested_cached_tokens_used_when_top_level_absent() {
        let usage = chat_usage(json!({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 70},
        }));
        let stats = from_chat_usage(&usage);
        assert_eq!(stats.cache_read_tokens, Some(70));
        assert_eq!(stats.cache_miss_tokens, Some(30));
        assert_eq!(stats.cache_write_tokens, None);
    }

    #[test]
    fn top_level_cached_tokens_wins_over_nested_without_double_count() {
        let usage = chat_usage(json!({
            "prompt_tokens": 100,
            "cached_tokens": 70,
            "prompt_tokens_details": {"cached_tokens": 60},
        }));
        let stats = from_chat_usage(&usage);
        assert_eq!(
            stats.cache_read_tokens,
            Some(70),
            "top-level wins, no double count"
        );
        assert_eq!(stats.cache_miss_tokens, Some(30));
    }

    #[test]
    fn top_level_cached_tokens_missing_means_unknown_not_zero() {
        let usage = chat_usage(json!({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }));
        let stats = from_chat_usage(&usage);
        assert_eq!(stats.cache_read_tokens, None, "unknown read, not 0");
        assert_eq!(stats.cache_miss_tokens, None, "no read => no derived miss");
        assert_eq!(stats.cache_write_tokens, None);
    }

    // --- DeepSeek shapes ---

    #[test]
    fn deepseek_explicit_hit_and_miss_are_preserved() {
        let usage = chat_usage(json!({
            "prompt_tokens": 100,
            "prompt_cache_hit_tokens": 60,
            "prompt_cache_miss_tokens": 40,
        }));
        let stats = from_chat_usage(&usage);
        assert_eq!(stats.cache_read_tokens, Some(60));
        assert_eq!(
            stats.cache_miss_tokens,
            Some(40),
            "explicit miss is preserved, not re-derived"
        );
        assert_eq!(stats.cache_write_tokens, None);
    }

    #[test]
    fn deepseek_hit_only_derives_miss() {
        let usage = chat_usage(json!({
            "prompt_tokens": 100,
            "prompt_cache_hit_tokens": 60,
        }));
        let stats = from_chat_usage(&usage);
        assert_eq!(stats.cache_read_tokens, Some(60));
        assert_eq!(stats.cache_miss_tokens, Some(40));
    }

    // --- Responses shapes ---

    #[test]
    fn responses_three_buckets_are_separated() {
        let usage = responses_usage(json!({
            "input_tokens": 100,
            "output_tokens": 20,
            "input_tokens_details": {"cached_tokens": 70, "cache_write_tokens": 5},
        }));
        let stats = from_responses_usage(&usage);
        assert_eq!(stats.source, CacheSource::Responses);
        assert_eq!(stats.input_tokens, Some(100));
        assert_eq!(stats.cache_read_tokens, Some(70));
        assert_eq!(stats.cache_write_tokens, Some(5), "explicit write only");
        assert_eq!(stats.cache_miss_tokens, Some(25));
    }

    #[test]
    fn responses_top_level_write_is_explicit_fallback() {
        let usage = responses_usage(json!({
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 70},
            "cache_write_tokens": 5,
        }));
        let stats = from_responses_usage(&usage);
        assert_eq!(stats.cache_read_tokens, Some(70));
        assert_eq!(stats.cache_write_tokens, Some(5));
        assert_eq!(stats.cache_miss_tokens, Some(25));
    }

    #[test]
    fn responses_write_absent_means_no_write_never_fabricated() {
        let usage = responses_usage(json!({
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 70},
        }));
        let stats = from_responses_usage(&usage);
        assert_eq!(stats.cache_read_tokens, Some(70));
        assert_eq!(stats.cache_write_tokens, None);
        assert_eq!(stats.cache_miss_tokens, Some(30));
    }

    // --- Boundaries & error separation ---

    #[test]
    fn cached_greater_than_prompt_yields_unknown_miss_no_panic_no_negative() {
        let usage = chat_usage(json!({
            "prompt_tokens": 50,
            "cached_tokens": 70,
        }));
        let stats = from_chat_usage(&usage);
        assert_eq!(stats.cache_read_tokens, Some(70));
        assert_eq!(
            stats.cache_miss_tokens, None,
            "inconsistent upstream data must stay unknown, never negative"
        );
    }

    #[test]
    fn prompt_zero_boundary_does_not_panic() {
        let usage = chat_usage(json!({
            "prompt_tokens": 0,
            "cached_tokens": 0,
        }));
        let stats = from_chat_usage(&usage);
        assert_eq!(stats.input_tokens, Some(0));
        assert_eq!(stats.cache_read_tokens, Some(0));
        assert_eq!(stats.cache_miss_tokens, Some(0));
    }

    #[test]
    fn chat_write_is_never_fabricated_from_prompt_minus_cached() {
        // The historical Chat `prompt - cached` remainder must map to miss, and
        // must never be reported as a write/creation.
        let usage = chat_usage(json!({
            "prompt_tokens": 200,
            "cached_tokens": 120,
        }));
        let stats = from_chat_usage(&usage);
        assert_eq!(stats.cache_read_tokens, Some(120));
        assert_eq!(stats.cache_write_tokens, None);
        assert_eq!(stats.cache_miss_tokens, Some(80));
    }

    #[test]
    fn http_error_or_timeout_without_usage_never_counts_as_miss() {
        // Error/timeout responses carry no usage object; the optional adapters
        // must yield None (absent stats), not a fabricated zero or miss bucket.
        assert_eq!(from_optional_chat_usage(None), None);
        assert_eq!(from_optional_responses_usage(None), None);
    }

    #[test]
    fn http_error_or_timeout_with_usage_still_records() {
        // A successful response that happens to include usage maps normally.
        let usage = chat_usage(json!({
            "prompt_tokens": 100,
            "cached_tokens": 70,
        }));
        let stats = from_optional_chat_usage(Some(&usage)).expect("usage present");
        assert_eq!(stats.cache_read_tokens, Some(70));
        assert_eq!(stats.cache_miss_tokens, Some(30));
    }

    // --- four-shape matrix summary ---

    #[test]
    fn four_usage_shapes_map_to_a_consistent_telemetry_shape() {
        let kimi = from_chat_usage(&chat_usage(json!({
            "prompt_tokens": 100, "cached_tokens": 70,
        })));
        let deepseek = from_chat_usage(&chat_usage(json!({
            "prompt_tokens": 100, "prompt_cache_hit_tokens": 60, "prompt_cache_miss_tokens": 40,
        })));
        let openai_ptd = from_chat_usage(&chat_usage(json!({
            "prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 70},
        })));
        let responses = from_responses_usage(&responses_usage(json!({
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 70, "cache_write_tokens": 5},
        })));

        // Every shape produces the same four-bucket telemetry shape: input + read
        // present, write only where upstream explicitly reported it, miss derived
        // from the remainder. No panic, no negative, no fabricated write.
        assert_eq!(kimi.cache_read_tokens, Some(70));
        assert_eq!(kimi.cache_miss_tokens, Some(30));
        assert_eq!(kimi.cache_write_tokens, None);

        assert_eq!(deepseek.cache_read_tokens, Some(60));
        assert_eq!(deepseek.cache_miss_tokens, Some(40));
        assert_eq!(deepseek.cache_write_tokens, None);

        assert_eq!(openai_ptd.cache_read_tokens, Some(70));
        assert_eq!(openai_ptd.cache_miss_tokens, Some(30));
        assert_eq!(openai_ptd.cache_write_tokens, None);

        assert_eq!(responses.cache_read_tokens, Some(70));
        assert_eq!(responses.cache_write_tokens, Some(5));
        assert_eq!(responses.cache_miss_tokens, Some(25));

        // Exactly one shape (Responses) carries an explicit write bucket.
        let all = [kimi, deepseek, openai_ptd, responses];
        assert_eq!(
            all.iter()
                .filter(|stats| stats.cache_write_tokens.is_some())
                .count(),
            1
        );
    }

    // --- default-off cache policy contract ---

    #[test]
    fn cache_policy_default_is_fully_off() {
        let policy = CachePolicy::default();
        assert_eq!(policy.usage, UsagePolicy::Off);
        assert_eq!(policy.upstream, None);
        assert!(
            !policy.cache_usage_enabled(),
            "default policy must not opt into cache-usage telemetry"
        );
        // UsagePolicy default is Off (the serde default source).
        assert_eq!(UsagePolicy::default(), UsagePolicy::Off);
    }

    #[test]
    fn cache_usage_enabled_follows_usage_source() {
        let enabled = CachePolicy {
            usage: UsagePolicy::TopLevelCachedTokens,
            upstream: None,
        };
        assert!(enabled.cache_usage_enabled());

        let binding_only = CachePolicy {
            usage: UsagePolicy::Off,
            upstream: Some("official".to_string()),
        };
        assert!(
            !binding_only.cache_usage_enabled(),
            "an upstream binding without a usage source is still cache-off"
        );
    }

    #[test]
    fn cache_policy_serde_missing_fields_default_off() {
        // Empty JSON object -> every field defaults off.
        let policy: CachePolicy = serde_json::from_str("{}").unwrap();
        assert_eq!(policy.usage, UsagePolicy::Off);
        assert_eq!(policy.upstream, None);

        // Unknown fields are ignored (forward-compat with later phases).
        let policy: CachePolicy =
            serde_json::from_str(r#"{"usage":"top_level_cached_tokens","unknown_future":"x"}"#)
                .unwrap();
        assert_eq!(policy.usage, UsagePolicy::TopLevelCachedTokens);
    }

    #[test]
    fn cache_policy_serde_roundtrip() {
        let policy = CachePolicy {
            usage: UsagePolicy::TopLevelCachedTokens,
            upstream: Some("official".to_string()),
        };
        let json = serde_json::to_string(&policy).unwrap();
        let back: CachePolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back, policy);
        assert_eq!(
            json,
            r#"{"usage":"top_level_cached_tokens","upstream":"official"}"#
        );
    }

    // --- Phase 2b.2: fail-closed session cache key contract (T16-T19) ---

    #[test]
    fn no_stable_source_returns_none_fail_closed() {
        // T16 (MUST): no session context => no key (fail-closed). A None
        // source must never be replaced by a random / UUID / time fallback.
        assert_eq!(
            session_key_from_source(None, "moonshot-official", "kimi-k3-turbo", None),
            None
        );
        assert_eq!(
            session_key_from_source(None, "moonshot-official", "kimi-k3-turbo", Some("official")),
            None
        );
    }

    #[test]
    fn same_session_derives_byte_equal_key_across_calls() {
        // T17 (MUST): the same session identity over multiple turns yields a
        // byte-equal key on every call.
        let first =
            session_key_from_source(Some("user_123"), "moonshot-official", "kimi-k3-turbo", None);
        let second =
            session_key_from_source(Some("user_123"), "moonshot-official", "kimi-k3-turbo", None);
        assert_eq!(first, second);
        assert!(first.is_some());
    }

    #[test]
    fn reconnect_and_restart_preserve_key_deterministically() {
        // T18 (MUST): reconnect / process restart must not change the key.
        // The derivation is a stateless deterministic hash of inbound signal
        // (no UUID / random / clock), so re-deriving it must yield the same
        // bytes. Exercise it across many repeated calls to mirror restarting.
        let key =
            session_key_from_source(Some("user_456"), "moonshot-official", "kimi-k3-turbo", None)
                .expect("stable source present");
        for _ in 0..100 {
            assert_eq!(
                session_key_from_source(
                    Some("user_456"),
                    "moonshot-official",
                    "kimi-k3-turbo",
                    None
                )
                .as_deref(),
                Some(key.as_str())
            );
        }
    }

    #[test]
    fn different_session_yields_different_key() {
        // T19 (MUST, part 1): a different session identity => a different key.
        let a = session_key_from_source(
            Some("session-A"),
            "moonshot-official",
            "kimi-k3-turbo",
            None,
        );
        let b = session_key_from_source(
            Some("session-B"),
            "moonshot-official",
            "kimi-k3-turbo",
            None,
        );
        assert_ne!(a, b);
    }

    #[test]
    fn session_key_is_hashed_hex_and_hides_plaintext() {
        // T19 (MUST, part 2): user/token-like source text is hashed — the
        // outbound key never contains the plaintext and has a fixed
        // length/format (first 16 bytes of sha256 => 32 lowercase hex chars).
        let secret = "sk-ant-0123456789abcdef_secret-user-token";
        let key = session_key_from_source(
            Some(secret),
            "moonshot-official",
            "kimi-k3-turbo",
            Some("official"),
        )
        .expect("stable source present");
        assert!(
            !key.contains(secret),
            "plaintext must never leak into the key"
        );
        assert_eq!(key.len(), 32, "16 hash bytes => 32 hex chars");
        assert!(
            key.chars().all(|c| c.is_ascii_hexdigit()),
            "key must be hex"
        );
        assert_eq!(key, key.to_lowercase(), "hex digest is lowercase");
        // Deterministic across repeated calls (T18).
        assert_eq!(
            session_key_from_source(
                Some(secret),
                "moonshot-official",
                "kimi-k3-turbo",
                Some("official")
            )
            .as_deref(),
            Some(key.as_str())
        );
    }

    #[test]
    fn key_is_namespaced_by_provider_model_and_upstream() {
        // The plan pins the key to (user_id/session, model, provider); changing
        // any of those inputs must change the key, and an upstream binding
        // must namespace the key too.
        let base = session_key_from_source(Some("u_1"), "moonshot-official", "kimi-k3-turbo", None);
        let other_provider = session_key_from_source(Some("u_1"), "eswitch", "kimi-k3-turbo", None);
        let other_model =
            session_key_from_source(Some("u_1"), "moonshot-official", "kimi-k3-turbo-next", None);
        let bound = session_key_from_source(
            Some("u_1"),
            "moonshot-official",
            "kimi-k3-turbo",
            Some("official"),
        );
        assert_ne!(base, other_provider, "provider change must change key");
        assert_ne!(base, other_model, "model change must change key");
        assert_ne!(base, bound, "upstream binding must namespace the key");
    }

    // --- Phase 2b.3: Responses usage view selector (legacy vs raw) ---

    fn raw_policy() -> CachePolicy {
        CachePolicy {
            usage: UsagePolicy::TopLevelCachedTokens,
            upstream: None,
        }
    }

    #[test]
    fn cache_stats_mode_follows_policy_only() {
        assert_eq!(CacheStatsMode::from_policy(None), CacheStatsMode::Legacy);
        assert_eq!(
            CacheStatsMode::from_policy(Some(&CachePolicy {
                usage: UsagePolicy::Off,
                upstream: None,
            })),
            CacheStatsMode::Legacy,
            "explicit usage:off is still legacy (never activates)"
        );
        assert_eq!(
            CacheStatsMode::from_policy(Some(&raw_policy())),
            CacheStatsMode::Raw
        );
    }

    #[test]
    fn responses_usage_view_default_off_matches_legacy_three_bucket_baseline() {
        let usage = responses_usage(json!({
            "input_tokens": 100,
            "output_tokens": 20,
            "input_tokens_details": {"cached_tokens": 70, "cache_write_tokens": 5},
        }));
        // No policy ⇒ Legacy mode with the exact old numbers.
        let view = responses_usage_view(Some(&usage), None);
        assert_eq!(view.input, Some(100));
        assert_eq!(
            view.read,
            Some(70),
            "read = input_tokens_details.cached_tokens"
        );
        assert_eq!(
            view.creation,
            Some(5),
            "write only from explicit cache_write_tokens"
        );
        assert_eq!(view.miss, Some(25), "legacy miss = input - read - creation");
        assert_eq!(view.hit_rate, Some(70.0));

        // An explicitly off policy must behave byte-identically to None.
        let off = CachePolicy {
            usage: UsagePolicy::Off,
            upstream: None,
        };
        assert_eq!(
            responses_usage_view(Some(&usage), Some(&off)),
            view,
            "usage:off must equal policy None"
        );
    }

    #[test]
    fn responses_usage_view_raw_under_opt_in_uses_cache_stats_projection() {
        let usage = responses_usage(json!({
            "input_tokens": 100,
            "output_tokens": 20,
            "input_tokens_details": {"cached_tokens": 70, "cache_write_tokens": 5},
        }));
        let view = responses_usage_view(Some(&usage), Some(&raw_policy()));
        assert_eq!(view.input, Some(100));
        assert_eq!(view.read, Some(70));
        assert_eq!(view.creation, Some(5));
        assert_eq!(view.miss, Some(25));
        assert_eq!(view.hit_rate, Some(70.0));
        // The raw projection must be exactly `from_responses_usage` (the
        // canonical `CacheStats` adapter).
        let stats = from_responses_usage(&usage);
        assert_eq!(view.read, stats.cache_read_tokens);
        assert_eq!(view.creation, stats.cache_write_tokens);
        assert_eq!(view.miss, stats.cache_miss_tokens);
    }

    #[test]
    fn responses_usage_view_legacy_clamps_where_raw_stays_unknown() {
        // cached > prompt: legacy saturates to 0; raw refuses to fabricate a
        // negative and reports unknown miss.
        let usage = responses_usage(json!({
            "input_tokens": 50,
            "input_tokens_details": {"cached_tokens": 70, "cache_write_tokens": 0},
        }));
        let legacy = responses_usage_view(Some(&usage), None);
        assert_eq!(
            legacy.miss,
            Some(0),
            "legacy preserves the clamped-to-zero behavior"
        );
        assert_eq!(legacy.hit_rate, Some(140.0), "legacy hit rate = read/input");

        let raw = responses_usage_view(Some(&usage), Some(&raw_policy()));
        assert_eq!(
            raw.miss, None,
            "raw must never fabricate a negative/clamped miss"
        );
        assert_eq!(raw.hit_rate, Some(140.0));
    }

    #[test]
    fn responses_usage_view_none_usage_is_empty_never_a_miss() {
        // HTTP error / timeout / missing usage object ⇒ empty view in BOTH
        // modes; it must never surface as a cache miss or fabricated zero.
        let legacy = responses_usage_view(None, None);
        let raw = responses_usage_view(None, Some(&raw_policy()));
        let empty = ResponsesUsageView {
            input: None,
            read: None,
            creation: None,
            miss: None,
            hit_rate: None,
        };
        assert_eq!(legacy, empty);
        assert_eq!(raw, empty);
    }

    #[test]
    fn responses_usage_view_creation_only_from_explicit_write() {
        // No write field anywhere ⇒ creation None, never fabricated from
        // `input - read`.
        let usage = responses_usage(json!({
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 70},
        }));
        let legacy = responses_usage_view(Some(&usage), None);
        assert_eq!(legacy.creation, None);
        assert_eq!(legacy.miss, Some(30));
        let raw = responses_usage_view(Some(&usage), Some(&raw_policy()));
        assert_eq!(raw.creation, None);
        assert_eq!(raw.miss, Some(30));

        // Top-level `cache_write_tokens` is the explicit fallback.
        let top = responses_usage(json!({
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 70},
            "cache_write_tokens": 5,
        }));
        assert_eq!(
            responses_usage_view(Some(&top), Some(&raw_policy())).creation,
            Some(5)
        );
    }

    #[test]
    fn responses_usage_view_zero_input_hit_rate_differs_legacy_vs_raw() {
        let usage = responses_usage(json!({
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
        }));
        let legacy = responses_usage_view(Some(&usage), None);
        assert_eq!(
            legacy.hit_rate,
            Some(0.0),
            "legacy clamps zero input to 0.0"
        );
        assert_eq!(legacy.miss, Some(0));
        let raw = responses_usage_view(Some(&usage), Some(&raw_policy()));
        assert_eq!(
            raw.hit_rate, None,
            "raw hit rate is unknown when input is zero"
        );
        assert_eq!(raw.miss, Some(0), "input=0, read=0 ⇒ miss 0 (consistent)");
    }

    #[test]
    fn responses_usage_view_from_buckets_matches_non_stream_selector() {
        // Stream path uses the same selector over already-extracted buckets, so
        // streamed and non-streamed responses report identically.
        let usage = responses_usage(json!({
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 70, "cache_write_tokens": 5},
        }));
        let via_usage = responses_usage_view(Some(&usage), Some(&raw_policy()));
        let via_buckets =
            responses_usage_view_from_buckets(Some(100), Some(70), Some(5), Some(&raw_policy()));
        assert_eq!(via_usage, via_buckets);

        let legacy_usage = responses_usage_view(Some(&usage), None);
        let legacy_buckets = responses_usage_view_from_buckets(Some(100), Some(70), Some(5), None);
        assert_eq!(legacy_usage, legacy_buckets);
    }
}
