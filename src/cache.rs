// Copyright (c) 2025 cc-proxy
//
// Provider-neutral raw cache telemetry.
//
// Phase 2 scope: pure mapping functions only. Nothing in this module is wired
// into production logs or outbound wire behavior — every bucket is opt-in by a
// later phase. Existing per-provider adapters (e.g. `responses::response::
// CacheStats`) are intentionally left untouched so providers that have not
// opted in see zero output change.
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

/// Which upstream usage shape produced a `CacheStats`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "telemetry source marker; wired in a later phase")
)]
pub(crate) enum CacheSource {
    /// OpenAI-compatible chat completions usage (Kimi / OpenAI / DeepSeek / GLM shapes).
    #[default]
    Chat,
    /// Responses API usage.
    Responses,
}

/// Raw, provider-neutral cache telemetry buckets.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "provider-neutral cache telemetry; wired in a later phase"
    )
)]
pub(crate) struct CacheStats {
    pub(crate) input_tokens: Option<u32>,
    pub(crate) cache_read_tokens: Option<u32>,
    pub(crate) cache_write_tokens: Option<u32>,
    pub(crate) cache_miss_tokens: Option<u32>,
    pub(crate) source: CacheSource,
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
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "usage adapter; wired in a later phase")
)]
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
}
