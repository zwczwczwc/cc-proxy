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
// As of Phase 2b.4 the Chat path (`openai/converter.rs`, `sse/stream.rs`)
// reads its cache-usage view through `chat_usage_view` /
// `chat_usage_view_from_buckets`, behind the same policy gate — `from_chat_usage`
// now has production callers; the `from_optional_*` wrappers remain test-only.
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
use std::collections::HashMap;

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

/// Assistant-history replay policy (Phase 4b).
///
/// `Off` (the default) is the legacy gate: `reasoning_content` replay for
/// stored assistant messages is decided by the **current** request's
/// thinking/effort, so a historical message can be rewritten between turns
/// when the client varies its thinking budget — busting the upstream prefix
/// cache from that message onwards (report 34 K2 / OSS F5).
///
/// `FullAssistant` opts into replaying every stored assistant message in
/// full (reasoning + text + tool_calls) independent of the current request's
/// thinking budget or effort — the direct precedent is
/// `raine/claude-code-proxy::push_assistant_message`, which unconditionally
/// replays `thinking -> reasoning_content` and `tool_use -> tool_calls`
/// without consulting the current request's effort. Activation additionally
/// requires the canonical `"official"` upstream binding (see
/// [`CachePolicy::full_assistant_replay_for_upstream`]); it never depends on
/// a provider-name string (G7). The policy governs **history replay only** —
/// the current request's own output control (`thinking.type` /
/// `reasoning_effort`) is untouched and never resurrected by it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayPolicy {
    /// Legacy gate: replay follows the current request's thinking/effort.
    #[default]
    Off,
    /// Full assistant-history replay, independent of current request effort.
    FullAssistant,
}

/// Whether a [`ReplayPolicy`] is the default (`Off`) — used to omit the
/// field from serialized config so the default-off surface stays minimal.
fn is_default_replay(value: &ReplayPolicy) -> bool {
    *value == ReplayPolicy::default()
}

/// Append-only history policy (Phase 4c).
///
/// `Off` (the default) keeps the legacy chat encoder byte-for-byte: it may
/// compact oversized `tool_result` bodies (HEAD/TAIL 4000), drop repeated
/// tool results for the same `tool_call_id`, and strip orphaned `tool_calls`
/// from non-final assistant messages. Each of those rewrites a stored
/// message and busts the upstream prefix cache from that message onwards
/// (report 34 K2 / OSS F5).
///
/// `AppendOnly` opts into **never rewriting already-provided history** on
/// the wire: orphan `tool_calls` are not deleted, `tool_result` bodies are
/// not compacted, repeated results are not deduplicated — order, tool IDs
/// and content bytes are preserved exactly as stored. Activation
/// additionally requires the canonical `"official"` upstream binding (see
/// [`CachePolicy::append_only_history_for_upstream`]); it never depends on
/// a provider-name string (G7). The policy governs **stored-history
/// preservation only** — the current request's own output control
/// (`thinking.type` / `reasoning_effort`) and assistant reasoning replay
/// (`ReplayPolicy`) are untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryPolicy {
    /// Legacy: cleanup/compaction/dedup stay enabled (default).
    #[default]
    Off,
    /// Preserve stored history append-only: no orphan cleanup, no tool
    /// result compaction, no repeated-result dedup.
    AppendOnly,
}

/// Whether a [`HistoryPolicy`] is the default (`Off`) — used to omit the
/// field from serialized config so the default-off surface stays minimal.
fn is_default_history(value: &HistoryPolicy) -> bool {
    *value == HistoryPolicy::default()
}

/// Volatile-context relocation policy (Phase 4d).
///
/// `Off` (the default) keeps the legacy relocation path byte-for-byte: the
/// Chat encoder relocates volatile env blocks only when the process-level
/// `CODEMERMAFROST_RELOCATE` env var is present, and the relocation appends
/// into the last message (`migrate_volatile_system_blocks`). The Responses
/// encoder is unaffected.
///
/// `SplitTail` opts into the data-driven Kimi split-tail relocation: volatile
/// system blocks are split out of the cache-prefix-sensitive system position
/// and moved to the conversation tail without rewriting already-constructed
/// stable history. Activation additionally requires the canonical `"official"`
/// upstream binding (see [`CachePolicy::split_tail_relocate_for_upstream`]);
/// it never depends on a provider-name string (G7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelocatePolicy {
    /// Legacy: relocation gated by the `CODEMERMAFROST_RELOCATE` env var.
    #[default]
    Off,
    /// Kimi official: split volatile system blocks to the conversation tail.
    SplitTail,
}

/// Whether a [`RelocatePolicy`] is the default (`Off`) — used to omit the
/// field from serialized config so the default-off surface stays minimal.
fn is_default_relocate(value: &RelocatePolicy) -> bool {
    *value == RelocatePolicy::default()
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
    /// Optional strict legal-effort allowlist (Phase 3 P3-B).
    ///
    /// When declared, `validate_effort_enum()` fail-fasts at startup on any
    /// `effort_map` output outside this set — a provider that opts in to a strict enum can
    /// never silently emit an illegal effort value. The Kimi (Moonshot)
    /// official set is strictly `{low, high, max}`. `None` (default) keeps
    /// every legacy effort_map valid: DeepSeek/GLM/GPT maps (which include
    /// `medium`/`xhigh`/`none` outputs) are untouched unless a provider
    /// explicitly declares this capability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort_enum: Option<Vec<String>>,
    /// Explicit, stateless reasoning-effort pin (Phase 4a, T13).
    ///
    /// When set to a legal wire effort (the official Kimi set is exactly
    /// `{low, high, max}` — validated against [`Self::effort_enum`]), the
    /// converter's effort path uses this value instead of recomputing effort
    /// from the per-request `thinking.budget_tokens`, so a session's
    /// `reasoning_effort` never flips as the client varies its thinking
    /// budget between requests.
    ///
    /// The pin must name an actual key of the provider's `effort_map`
    /// (validated at startup by [`Self::validate_effort_enum`]): the effort
    /// is resolved through that map on the wire, so a pin absent from the map
    /// would otherwise be silently coerced to `high` — a hard config error,
    /// never a silent fallback.
    ///
    /// This is a **static, stateless pin** declared in config: it does NOT
    /// "capture the first request's effort and remember it per session".
    /// True per-session derivation would require implicit server-side session
    /// state, which this codebase deliberately avoids; the pin is explicit in
    /// config instead, so it is deterministic across calls, restarts and
    /// sessions by construction. Default `None` (no pin) keeps the existing
    /// dynamic per-request effort derivation byte-for-byte.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pinned_effort: Option<String>,
    /// Whether this policy opts into injecting a deterministic session
    /// `prompt_cache_key` on the outbound Chat wire (Phase 3 P3-C).
    ///
    /// Default `false`. This is a pure capability flag: activation is gated on
    /// this explicit opt-in PLUS a stable inbound `metadata.user_id` source
    /// PLUS an official Moonshot upstream binding (resolved data-driven via
    /// [`crate::config::Config::effective_upstream_binding`]) — never on a
    /// provider-name string, and never on `upstream: "official"` alone: a
    /// legacy `moonshot-official` route that declares no policy keeps sending
    /// no key. The Responses encoder is an explicit non-goal and never carries
    /// a key (Kimi rides the Chat wire).
    #[serde(default, skip_serializing_if = "is_false")]
    pub prompt_cache_key_enabled: bool,
    /// Assistant-history replay policy (Phase 4b).
    ///
    /// `Off` (default) keeps the legacy gate byte-for-byte: `reasoning_content`
    /// replay is decided by the current request's thinking/effort, so a
    /// historical assistant message can be rewritten between turns when the
    /// client varies its budget. `FullAssistant` (opt-in, official Kimi
    /// upstream only — see
    /// [`Self::full_assistant_replay_for_upstream`]) replays stored assistant
    /// reasoning/text/tool_calls independent of the current request,
    /// matching `raine/claude-code-proxy::push_assistant_message`'s
    /// unconditional replay. It governs history replay only: current-request
    /// output control (`thinking.type` / `reasoning_effort`) is never
    /// changed or resurrected by it.
    #[serde(default, skip_serializing_if = "is_default_replay")]
    pub replay: ReplayPolicy,
    /// Append-only history policy (Phase 4c).
    ///
    /// `Off` (default) keeps the legacy chat encoder byte-for-byte: oversized
    /// `tool_result` bodies are compacted, repeated results are deduplicated,
    /// and orphaned `tool_calls` are stripped from non-final assistant
    /// messages. `AppendOnly` (opt-in, official Kimi upstream only — see
    /// [`Self::append_only_history_for_upstream`]) never rewrites
    /// already-provided history: orphan `tool_calls` are kept, `tool_result`
    /// bodies are not compacted, repeated results are not deduplicated —
    /// order, tool IDs and content bytes are preserved exactly as stored.
    /// It governs stored-history preservation only: assistant reasoning
    /// replay (`Self::replay`) and current-request output control are
    /// untouched.
    #[serde(default, skip_serializing_if = "is_default_history")]
    pub history: HistoryPolicy,
    /// Volatile-context relocation policy (Phase 4d).
    ///
    /// `Off` (default) keeps the legacy env-driven `CODEMERMAFROST_RELOCATE`
    /// relocation path byte-for-byte. `SplitTail` (opt-in, official Kimi
    /// upstream only — see
    /// [`Self::split_tail_relocate_for_upstream`]) splits volatile system
    /// blocks out of the cache-prefix-sensitive system position and moves
    /// them to the conversation tail without rewriting already-constructed
    /// stable history. It governs system-prompt relocation only: assistant
    /// reasoning replay (`Self::replay`), stored-history preservation
    /// (`Self::history`) and current-request output control are untouched.
    #[serde(default, skip_serializing_if = "is_default_relocate")]
    pub relocate: RelocatePolicy,
}

#[inline]
fn is_false(value: &bool) -> bool {
    !*value
}

impl CachePolicy {
    /// Whether this policy opts into cache-usage telemetry.
    ///
    /// `cache_usage_enabled() == false` is the default; a provider only reports
    /// cache usage once its policy names a concrete usage source.
    pub fn cache_usage_enabled(&self) -> bool {
        !matches!(self.usage, UsagePolicy::Off)
    }

    /// The reasoning effort pinned by this policy, gated on the canonical
    /// official upstream (Phase 4a, T13).
    ///
    /// Returns `Some(effort)` only when BOTH hold: this policy declares an
    /// explicit `pinned_effort` AND the provider's effective upstream binding
    /// resolves to `"official"` (Kimi For Coding). Any other route — no pin,
    /// or a provider routed through the default eswitch upstream — returns
    /// `None` and keeps the dynamic per-request effort derivation. The
    /// binding must be resolved data-driven (never a provider-name string,
    /// G7); the converter calls this with
    /// [`crate::config::Config::effective_upstream_binding`].
    pub fn pinned_effort_for_upstream(&self, effective_upstream: Option<&str>) -> Option<&str> {
        if effective_upstream == Some("official") {
            self.pinned_effort.as_deref()
        } else {
            None
        }
    }

    /// Whether this policy opts into full assistant-history replay (Phase 4b).
    ///
    /// Returns `true` only when BOTH hold: this policy declares
    /// `replay = FullAssistant` AND the provider's effective upstream binding
    /// resolves to `"official"` (Kimi For Coding). Any other route — replay
    /// off, or a provider routed through the default eswitch upstream —
    /// returns `false` and keeps the legacy current-request-gated replay
    /// byte-for-byte (fail-closed). The binding must be resolved data-driven
    /// (never a provider-name string, G7); the converter calls this with
    /// [`crate::config::Config::effective_upstream_binding`]. The policy
    /// governs assistant-history replay only — it never touches the current
    /// request's `thinking.type` / `reasoning_effort` output control.
    pub fn full_assistant_replay_for_upstream(&self, effective_upstream: Option<&str>) -> bool {
        self.replay == ReplayPolicy::FullAssistant && effective_upstream == Some("official")
    }

    /// Whether this policy opts into append-only stored-history preservation
    /// (Phase 4c).
    ///
    /// Returns `true` only when BOTH hold: this policy declares
    /// `history = AppendOnly` AND the provider's effective upstream binding
    /// resolves to `"official"` (Kimi For Coding). Any other route — history
    /// off, or a provider routed through the default eswitch upstream —
    /// returns `false` and keeps the legacy cleanup/compaction/dedup
    /// byte-for-byte (fail-closed). The binding must be resolved data-driven
    /// (never a provider-name string, G7); the converter calls this with
    /// [`crate::config::Config::effective_upstream_binding`]. The policy
    /// governs stored-history preservation only — it never touches the
    /// current request's `thinking.type` / `reasoning_effort` output control
    /// nor the assistant reasoning replay policy (`Self::replay`).
    pub fn append_only_history_for_upstream(&self, effective_upstream: Option<&str>) -> bool {
        self.history == HistoryPolicy::AppendOnly && effective_upstream == Some("official")
    }

    /// Whether this policy opts into split-tail volatile-context relocation
    /// (Phase 4d).
    ///
    /// Returns `true` only when BOTH hold: this policy declares
    /// `relocate = SplitTail` AND the provider's effective upstream binding
    /// resolves to `"official"` (Kimi For Coding). Any other route — relocate
    /// off, or a provider routed through the default eswitch upstream —
    /// returns `false` and keeps the legacy env-driven relocation path
    /// byte-for-byte (fail-closed). The binding must be resolved data-driven
    /// (never a provider-name string, G7); the converter calls this with
    /// [`crate::config::Config::effective_upstream_binding`]. The policy
    /// governs system-prompt relocation only — it never touches assistant
    /// reasoning replay (`Self::replay`), stored-history preservation
    /// (`Self::history`) nor current-request output control.
    pub fn split_tail_relocate_for_upstream(&self, effective_upstream: Option<&str>) -> bool {
        self.relocate == RelocatePolicy::SplitTail && effective_upstream == Some("official")
    }

    /// Fail-fast startup validation of a provider's `effort_map` outputs
    /// against this policy's declared `effort_enum` (Phase 3 P3-B).
    ///
    /// Only providers that explicitly opt in by declaring `effort_enum` are
    /// checked: when `effort_enum == None` (the default, and the state of every
    /// current config) all legacy `effort_map` outputs — including the
    /// DeepSeek/GLM/GPT `medium`/`xhigh`/`none` entries — stay valid, so
    /// default-off and non-opt-in profiles are byte-for-byte backward
    /// compatible. When a strict enum *is* declared, any `effort_map` output
    /// value outside the allowlist is a hard configuration error: this panics
    /// at startup instead of silently normalizing or coercing an illegal wire
    /// effort value.
    ///
    /// Phase 4a additionally requires a declared `pinned_effort` to (1) be a
    /// member of the enum and (2) be an actual key of the provider's
    /// `effort_map` — the effort is resolved through that map on the wire, so
    /// a pin absent from the map would otherwise be silently coerced to
    /// `high` by `apply_effort_direct`. Both failures panic at startup.
    pub fn validate_effort_enum(&self, provider: &str, effort_map: &HashMap<String, String>) {
        let Some(allowlist) = &self.effort_enum else {
            // Fail-closed: a pinned effort is a wire-effort promise and must
            // be validated against an explicit legal set — it can never be
            // declared without one (Phase 4a, T12/T13). This is a startup
            // error, never a silent skip.
            if self.pinned_effort.is_some() {
                panic!(
                    "Provider '{}' declares cache_policy.pinned_effort but no \
                     cache_policy.effort_enum; a pinned effort must be \
                     validated against an explicit legal wire set \
                     (Kimi official: low/high/max).",
                    provider
                );
            }
            return; // not opted in — legacy effort maps stay valid
        };
        for (input, output) in effort_map {
            if !allowlist.contains(output) {
                panic!(
                    "Provider '{}' cache_policy.effort_enum declares {:?}, but \
                     effort_map maps '{}' -> '{}', which is outside the legal \
                     wire effort set. Fix the effort_map output or drop the \
                     strict enum.",
                    provider, allowlist, input, output
                );
            }
        }
        // Phase 4a: the pinned effort itself must be a member of the declared
        // legal set. An invalid pin is a hard config error — never silently
        // normalized, never coerced to a different effort on the wire.
        if let Some(pinned) = &self.pinned_effort {
            if !allowlist.contains(pinned) {
                panic!(
                    "Provider '{}' cache_policy.pinned_effort '{}' is not in \
                     the declared effort_enum {:?}. Legal wire efforts for \
                     this provider are exactly the declared set.",
                    provider, pinned, allowlist
                );
            }
            // Phase 4a remediation (S1): the pin must ALSO be an actual key of
            // the provider's effort_map. `apply_effort_direct` resolves effort
            // through the map (`effort_map.get(effort).unwrap_or("high")`), so
            // a pin that is a legal enum member but absent from the map would
            // be SILENTLY coerced to "high" on the wire — the exact
            // "never silently coerced" promise this validator makes. A missing
            // key is a hard config error and must fail fast at startup.
            if !effort_map.contains_key(pinned) {
                panic!(
                    "Provider '{}' cache_policy.pinned_effort '{}' is not a \
                     key of the provider effort_map {:?}. A pinned effort must \
                     name an actual effort_map input so it is applied verbatim \
                     on the wire — never silently coerced.",
                    provider,
                    pinned,
                    effort_map.keys().collect::<Vec<_>>()
                );
            }
        }
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
/// Production callers: `chat_usage_view` (raw branch) and the SSE stream
/// terminal handler's raw-read capture.
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

// ---------------------------------------------------------------------------
// Chat usage view selector (Phase 2b.4).
//
// Mirrors `responses_usage_view` (Phase 2b.3) for the Chat wire. Exactly ONE
// mode is computed per request — `Legacy` or `Raw` — decided by the same
// `CacheStatsMode::from_policy` gate:
//   * `Legacy` (default — policy `None`/off): reproduces the pre-existing Chat
//     arithmetic byte-for-byte — read ONLY from `prompt_tokens_details.cached_
//     tokens`, creation labeled as `prompt - cached` (the historical remainder
//     label), clamped miss and `0.0` hit rate on zero input.
//   * `Raw` (explicit `cache_policy.usage`): the canonical `CacheStats`
//     projection (`from_chat_usage`) — read priority top-level `cached_tokens`
//     → nested `prompt_tokens_details.cached_tokens` → DeepSeek hit; creation
//     ALWAYS `None` for Chat (the `prompt - cached` remainder is miss, never a
//     write); guarded miss (unknown on inconsistent input, never negative).
// The view is the single source for the wire `read`/`creation` fields and the
// KV-cache log buckets, so wire and log can never disagree and a request never
// reports cache usage twice.
// ---------------------------------------------------------------------------

/// Normalized Chat cache-usage view (input / read / creation / miss / hit rate).
///
/// `read`/`creation` feed the Anthropic wire; `miss`/`hit_rate` feed the
/// KV-cache log. Absent usage ⇒ an empty view (never a fabricated miss — an
/// HTTP error or a missing usage object is not a miss).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ChatUsageView {
    pub(crate) input: Option<u32>,
    pub(crate) read: Option<u32>,
    pub(crate) creation: Option<u32>,
    pub(crate) miss: Option<u32>,
    pub(crate) hit_rate: Option<f64>,
}

impl ChatUsageView {
    fn empty() -> Self {
        ChatUsageView {
            input: None,
            read: None,
            creation: None,
            miss: None,
            hit_rate: None,
        }
    }
}

/// Non-stream selector: map a typed `ChatUsage` (plus policy) into the view.
/// `None` usage (HTTP error / timeout / missing usage) yields an empty view —
/// never a miss bucket.
pub(crate) fn chat_usage_view(
    usage: Option<&ChatUsage>,
    policy: Option<&CachePolicy>,
) -> ChatUsageView {
    let Some(usage) = usage else {
        return ChatUsageView::empty();
    };
    match CacheStatsMode::from_policy(policy) {
        CacheStatsMode::Raw => {
            // Canonical raw projection (the `CacheStats` adapter), plus a raw
            // hit rate. Creation stays `None`: Chat completions never report a
            // write, and the `prompt - cached` remainder is miss, not creation.
            let stats = from_chat_usage(usage);
            ChatUsageView {
                input: stats.input_tokens,
                read: stats.cache_read_tokens,
                creation: None,
                miss: stats.cache_miss_tokens,
                hit_rate: raw_hit_rate(stats.input_tokens, stats.cache_read_tokens),
            }
        }
        CacheStatsMode::Legacy => {
            // Legacy read is ptd-only (top-level cached_tokens and DeepSeek
            // hit/miss are invisible to non-opt-in providers); legacy creation
            // is the historical `prompt - cached` remainder label.
            let read = usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens);
            chat_usage_view_from_buckets(usage.prompt_tokens, read, None, None)
        }
    }
}

/// Stream selector: build the view from the buckets already merged by the SSE
/// terminal handler.
///
/// `read` is the legacy read (nested `prompt_tokens_details.cached_tokens`,
/// byte-preserving for non-opt-in providers); `raw_read` is the canonical read
/// (top-level → nested → DeepSeek hit) used only under opt-in. `None`/off
/// policy ignores `raw_read` and reproduces the historical wire/log exactly.
pub(crate) fn chat_usage_view_from_buckets(
    input: Option<u32>,
    read: Option<u32>,
    raw_read: Option<u32>,
    policy: Option<&CachePolicy>,
) -> ChatUsageView {
    match CacheStatsMode::from_policy(policy) {
        CacheStatsMode::Raw => ChatUsageView {
            input,
            read: raw_read,
            creation: None,
            miss: derive_miss(input, raw_read, None),
            hit_rate: raw_hit_rate(input, raw_read),
        },
        CacheStatsMode::Legacy => {
            let creation = match (input, read) {
                (Some(p), Some(c)) if p > c => Some(p - c),
                _ => None,
            };
            let miss = input.map(|input| input.saturating_sub(read.unwrap_or(0)));
            let hit_rate = input.map(|input| {
                if input == 0 {
                    0.0
                } else {
                    read.unwrap_or(0) as f64 / input as f64 * 100.0
                }
            });
            ChatUsageView {
                input,
                read,
                creation,
                miss,
                hit_rate,
            }
        }
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
            prompt_cache_key_enabled: false,
            upstream: None,
            effort_enum: None,
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            pinned_effort: None,
        };
        assert!(enabled.cache_usage_enabled());

        let binding_only = CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: Some("official".to_string()),
            effort_enum: None,
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            pinned_effort: None,
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
        assert!(
            !policy.prompt_cache_key_enabled,
            "prompt_cache_key capability must default off (P3-C fail-closed)"
        );

        // Unknown fields are ignored (forward-compat with later phases).
        let policy: CachePolicy =
            serde_json::from_str(r#"{"usage":"top_level_cached_tokens","unknown_future":"x"}"#)
                .unwrap();
        assert_eq!(policy.usage, UsagePolicy::TopLevelCachedTokens);
    }

    #[test]
    fn prompt_cache_key_flag_serde_off_by_default_and_roundtrips_when_on() {
        // P3-C: the opt-in capability flag serializes only when true (off is
        // omitted from the wire config), defaults false on missing input, and
        // round-trips.
        let on = CachePolicy {
            usage: UsagePolicy::Off,
            upstream: Some("official".to_string()),
            effort_enum: None,
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            pinned_effort: None,
            prompt_cache_key_enabled: true,
        };
        let json = serde_json::to_string(&on).unwrap();
        assert_eq!(
            json,
            r#"{"usage":"off","upstream":"official","prompt_cache_key_enabled":true}"#
        );
        let back: CachePolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back, on);
        assert!(back.prompt_cache_key_enabled);
    }

    #[test]
    fn cache_policy_serde_roundtrip() {
        let policy = CachePolicy {
            usage: UsagePolicy::TopLevelCachedTokens,
            prompt_cache_key_enabled: false,
            upstream: Some("official".to_string()),
            effort_enum: None,
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            pinned_effort: None,
        };
        let json = serde_json::to_string(&policy).unwrap();
        let back: CachePolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back, policy);
        assert_eq!(
            json,
            r#"{"usage":"top_level_cached_tokens","upstream":"official"}"#
        );
    }

    // --- Phase 4d: RelocatePolicy (volatile-context split-tail relocation) ---

    #[test]
    fn relocate_policy_defaults_off_and_omitted_from_serialized_config() {
        // Default-off surface stays minimal: a policy with relocate Off
        // serializes without the field (skip_serializing_if), and missing
        // input deserializes to Off (fail-closed).
        let policy = CachePolicy {
            usage: UsagePolicy::Off,
            upstream: Some("official".to_string()),
            effort_enum: None,
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            pinned_effort: None,
            prompt_cache_key_enabled: false,
        };
        let json = serde_json::to_string(&policy).unwrap();
        assert_eq!(json, r#"{"usage":"off","upstream":"official"}"#);
        assert!(
            !json.contains("relocate"),
            "default relocate=off must be omitted: {json}"
        );
        let back: CachePolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back.relocate, RelocatePolicy::Off);
        let empty: CachePolicy = serde_json::from_str("{}").unwrap();
        assert_eq!(empty.relocate, RelocatePolicy::Off);
    }

    #[test]
    fn relocate_policy_serde_roundtrip_split_tail() {
        // Explicit split_tail round-trips through the serialized config.
        let policy = CachePolicy {
            usage: UsagePolicy::Off,
            upstream: Some("official".to_string()),
            effort_enum: None,
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            pinned_effort: None,
            prompt_cache_key_enabled: false,
            relocate: RelocatePolicy::SplitTail,
        };
        let json = serde_json::to_string(&policy).unwrap();
        assert!(
            json.contains(r#""relocate":"split_tail""#),
            "split_tail must serialize: {json}"
        );
        let back: CachePolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back.relocate, RelocatePolicy::SplitTail);
        assert_eq!(back, policy);
    }

    #[test]
    fn split_tail_gate_requires_policy_and_official_binding() {
        // Data-driven activation: relocate=split_tail alone is not enough —
        // the effective upstream binding must resolve to the canonical
        // "official" upstream (fail-closed otherwise).
        let policy = CachePolicy {
            usage: UsagePolicy::Off,
            upstream: Some("official".to_string()),
            effort_enum: None,
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            pinned_effort: None,
            prompt_cache_key_enabled: false,
            relocate: RelocatePolicy::SplitTail,
        };
        assert!(policy.split_tail_relocate_for_upstream(Some("official")));
        assert!(!policy.split_tail_relocate_for_upstream(None));
        assert!(!policy.split_tail_relocate_for_upstream(Some("eswitch")));

        let off = CachePolicy {
            usage: UsagePolicy::Off,
            upstream: Some("official".to_string()),
            effort_enum: None,
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            pinned_effort: None,
            prompt_cache_key_enabled: false,
        };
        assert!(
            !off.split_tail_relocate_for_upstream(Some("official")),
            "relocate=off (default) must never activate split-tail"
        );
    }

    // --- Phase 3 P3-B: effort_enum validation (opt-in fail-fast) ---

    fn kimi_effort_map() -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("low".to_string(), "low".to_string());
        m.insert("high".to_string(), "high".to_string());
        m.insert("max".to_string(), "max".to_string());
        m
    }

    #[test]
    fn effort_enum_none_keeps_legacy_effort_maps_valid() {
        // No opt-in (the default and every current config's state): legacy
        // maps with medium/xhigh/none outputs must stay valid — validate() is
        // a no-op and must never panic.
        let policy = CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: None,
            effort_enum: None,
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            pinned_effort: None,
        };
        let mut legacy = HashMap::new();
        legacy.insert("medium".to_string(), "medium".to_string());
        legacy.insert("xhigh".to_string(), "max".to_string());
        legacy.insert("none".to_string(), "none".to_string());
        policy.validate_effort_enum("deepseek", &legacy); // must not panic
    }

    #[test]
    fn effort_enum_accepts_all_outputs_in_declared_set() {
        // Kimi official legal wire effort is exactly {low, high, max}: every
        // output in the declared set validates cleanly.
        let policy = CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: None,
            effort_enum: Some(vec![
                "low".to_string(),
                "high".to_string(),
                "max".to_string(),
            ]),
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            pinned_effort: None,
        };
        policy.validate_effort_enum("moonshot", &kimi_effort_map()); // must not panic
    }

    #[test]
    #[should_panic(expected = "effort_enum")]
    fn effort_enum_rejects_medium_output() {
        // Explicit Kimi enum {low,high,max}: a `medium` output is an illegal
        // wire effort and must fail fast at startup — never silently
        // normalized, never coerced.
        let policy = CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: None,
            effort_enum: Some(vec![
                "low".to_string(),
                "high".to_string(),
                "max".to_string(),
            ]),
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            pinned_effort: None,
        };
        let mut map = kimi_effort_map();
        map.insert("medium".to_string(), "medium".to_string());
        policy.validate_effort_enum("moonshot", &map);
    }

    #[test]
    #[should_panic(expected = "effort_enum")]
    fn effort_enum_rejects_xhigh_output() {
        // The OSS `xhigh` acceptance must NOT be copied: under the Kimi
        // enum an xhigh output (even when the enum itself is not accepted) is
        // illegal and fails fast.
        let policy = CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: None,
            effort_enum: Some(vec![
                "low".to_string(),
                "high".to_string(),
                "max".to_string(),
            ]),
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            pinned_effort: None,
        };
        let mut map = kimi_effort_map();
        map.insert("xhigh".to_string(), "xhigh".to_string());
        policy.validate_effort_enum("moonshot", &map);
    }

    // --- Phase 4a: deterministic Kimi reasoning-effort pin (T13) ---

    #[test]
    fn pinned_effort_for_upstream_gates_on_official_binding() {
        // A pin only applies when BOTH an explicit pinned_effort is declared
        // AND the effective upstream binding is the canonical "official"
        // (Kimi For Coding) upstream. Everything else is `None` — the dynamic
        // per-request derivation is preserved.
        let policy = CachePolicy {
            pinned_effort: Some("high".to_string()),
            ..Default::default()
        };
        assert_eq!(
            policy.pinned_effort_for_upstream(Some("official")),
            Some("high")
        );
        assert_eq!(policy.pinned_effort_for_upstream(Some("eswitch")), None);
        assert_eq!(policy.pinned_effort_for_upstream(None), None);

        // No pin declared => never returns one, even on the official upstream.
        let no_pin = CachePolicy::default();
        assert_eq!(no_pin.pinned_effort_for_upstream(Some("official")), None);
    }

    #[test]
    fn pinned_effort_serde_off_by_default_and_roundtrips_when_set() {
        // T13 config surface: absent -> None (no pin, legacy dynamic effort).
        let absent: CachePolicy = serde_json::from_str("{}").unwrap();
        assert_eq!(absent.pinned_effort, None);

        let pinned = CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: Some("official".to_string()),
            effort_enum: Some(vec![
                "low".to_string(),
                "high".to_string(),
                "max".to_string(),
            ]),
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            pinned_effort: Some("high".to_string()),
        };
        let json = serde_json::to_string(&pinned).unwrap();
        assert!(
            json.contains(r#""pinned_effort":"high""#),
            "a set pin must serialize: {json}"
        );
        let back: CachePolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back, pinned);

        // None is omitted from the wire config (fail-closed default).
        let none = CachePolicy {
            pinned_effort: None,
            ..pinned.clone()
        };
        let json_none = serde_json::to_string(&none).unwrap();
        assert!(!json_none.contains("pinned_effort"));
    }

    #[test]
    #[should_panic(expected = "effort_enum")]
    fn pinned_effort_without_declared_enum_fails_fast() {
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
            relocate: RelocatePolicy::Off,
            pinned_effort: Some("high".to_string()),
        };
        policy.validate_effort_enum("moonshot", &kimi_effort_map());
    }

    #[test]
    #[should_panic(expected = "effort_enum")]
    fn pinned_effort_rejects_value_outside_declared_enum() {
        // `medium` is not in the official Kimi set {low,high,max}: an invalid
        // pin must fail fast at startup, never be silently normalized or
        // coerced to a different effort on the wire.
        let policy = CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: Some("official".to_string()),
            effort_enum: Some(vec![
                "low".to_string(),
                "high".to_string(),
                "max".to_string(),
            ]),
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            pinned_effort: Some("medium".to_string()),
        };
        policy.validate_effort_enum("moonshot", &kimi_effort_map());
    }

    #[test]
    fn pinned_effort_legal_value_passes_validation() {
        // Every legal Kimi wire effort validates cleanly when the enum is
        // declared alongside the pin.
        for pin in ["low", "high", "max"] {
            let policy = CachePolicy {
                usage: UsagePolicy::Off,
                prompt_cache_key_enabled: false,
                upstream: Some("official".to_string()),
                effort_enum: Some(vec![
                    "low".to_string(),
                    "high".to_string(),
                    "max".to_string(),
                ]),
                replay: crate::cache::ReplayPolicy::Off,
                history: crate::cache::HistoryPolicy::Off,
                relocate: RelocatePolicy::Off,
                pinned_effort: Some(pin.to_string()),
            };
            policy.validate_effort_enum("moonshot", &kimi_effort_map()); // must not panic
        }
    }

    // --- Phase 4a remediation (S1): pin must be an actual effort_map key ---

    /// A map whose keys are a strict subset of the Kimi enum — deliberately
    /// MISSING the "low" key so the S1 fail-fast path can be exercised.
    fn kimi_effort_map_without_low() -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("high".to_string(), "high".to_string());
        m.insert("max".to_string(), "max".to_string());
        m
    }

    #[test]
    fn pinned_effort_present_as_effort_map_key_passes_validation() {
        // S1 remediation guard: a pin that IS a key of the provider's
        // effort_map (and in the declared enum) stays valid. This is the
        // shape of every built-in moonshot effort_map, so no opt-in provider
        // in production is affected.
        let policy = CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: Some("official".to_string()),
            effort_enum: Some(vec![
                "low".to_string(),
                "high".to_string(),
                "max".to_string(),
            ]),
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            pinned_effort: Some("low".to_string()),
        };
        policy.validate_effort_enum("moonshot", &kimi_effort_map()); // must not panic
    }

    #[test]
    #[should_panic(expected = "effort_map")]
    fn pinned_effort_missing_from_effort_map_fails_fast() {
        // S1: the pin is a legal wire effort (member of effort_enum) but the
        // provider's effort_map does NOT contain it as a key. Without this
        // check, `apply_effort_direct`'s `effort_map.get(pin).unwrap_or("high")`
        // (converter.rs `_ =>` branch) would SILENTLY coerce the pin to "high"
        // on the wire — contradicting the "never silently coerced" promise.
        // A missing key must fail fast at startup with a clear error.
        let policy = CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: Some("official".to_string()),
            effort_enum: Some(vec![
                "low".to_string(),
                "high".to_string(),
                "max".to_string(),
            ]),
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            pinned_effort: Some("low".to_string()),
        };
        policy.validate_effort_enum("moonshot", &kimi_effort_map_without_low());
    }

    // --- Phase 4b: policy-gated full assistant-history replay (T06/T07) ---

    #[test]
    fn replay_policy_defaults_to_off() {
        // Fail-closed default: absent config field -> Off (legacy gate). No
        // provider is opted into full replay without an explicit declaration.
        let policy = CachePolicy::default();
        assert_eq!(policy.replay, ReplayPolicy::Off);
        assert!(!policy.full_assistant_replay_for_upstream(Some("official")));
        assert!(!policy.full_assistant_replay_for_upstream(None));

        let absent: CachePolicy = serde_json::from_str("{}").unwrap();
        assert_eq!(absent.replay, ReplayPolicy::Off);
        // Round-trip: default policy serializes with no replay field.
        let json = serde_json::to_string(&CachePolicy::default()).unwrap();
        assert!(
            !json.contains("replay"),
            "default-off replay must be omitted: {json}"
        );
    }

    #[test]
    fn replay_serde_accepts_full_assistant_and_roundtrips() {
        // serde surface: `replay = "full_assistant"` parses, round-trips, and
        // serializes under the snake_case name.
        let policy: CachePolicy = serde_json::from_str(r#"{"replay":"full_assistant"}"#).unwrap();
        assert_eq!(policy.replay, ReplayPolicy::FullAssistant);
        let json = serde_json::to_string(&policy).unwrap();
        assert!(
            json.contains(r#""replay":"full_assistant""#),
            "FullAssistant must serialize: {json}"
        );
        let back: CachePolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back.replay, ReplayPolicy::FullAssistant);
        // Unknown replay values fail rather than silently defaulting.
        assert!(serde_json::from_str::<CachePolicy>(r#"{"replay":"bogus"}"#).is_err());
    }

    #[test]
    fn full_assistant_replay_gates_on_official_binding() {
        // The gate is data-driven (G7, never a provider-name string):
        // FullAssistant applies ONLY on the canonical "official" upstream.
        let policy = CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: Some("official".to_string()),
            effort_enum: None,
            pinned_effort: None,
            replay: ReplayPolicy::FullAssistant,
            history: crate::cache::HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
        };
        assert!(policy.full_assistant_replay_for_upstream(Some("official")));
        assert!(!policy.full_assistant_replay_for_upstream(Some("eswitch")));
        assert!(!policy.full_assistant_replay_for_upstream(None));

        // Replay Off never opts in, even on the official upstream.
        let off = CachePolicy {
            replay: ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            ..policy.clone()
        };
        assert!(!off.full_assistant_replay_for_upstream(Some("official")));
    }

    // --- Phase 4c: append-only stored-history policy ---

    #[test]
    fn history_policy_default_is_off() {
        // Fail-closed default: history is Off unless explicitly opted in, so
        // the legacy chat encoder (cleanup/compaction/dedup) stays
        // byte-for-byte for every current config.
        assert_eq!(HistoryPolicy::default(), HistoryPolicy::Off);
    }

    #[test]
    fn history_policy_serde_snake_case_and_default_omission() {
        // "append_only" round-trips as snake_case; the default-off field is
        // omitted from serialized config so the opt-in surface stays minimal.
        let on = CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: Some("official".to_string()),
            effort_enum: None,
            pinned_effort: None,
            replay: ReplayPolicy::Off,
            history: HistoryPolicy::AppendOnly,
            relocate: RelocatePolicy::Off,
        };
        let json = serde_json::to_string(&on).unwrap();
        assert!(
            json.contains(r#""history":"append_only""#),
            "AppendOnly must serialize as snake_case append_only, got: {json}"
        );
        let back: CachePolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back.history, HistoryPolicy::AppendOnly);

        // Off (the default) is omitted from serialized config.
        let off = CachePolicy {
            history: HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            ..on.clone()
        };
        let json = serde_json::to_string(&off).unwrap();
        assert!(
            !json.contains("history"),
            "Off must be omitted, got: {json}"
        );
        // Unknown history values fail rather than silently defaulting.
        assert!(serde_json::from_str::<CachePolicy>(r#"{"history":"bogus"}"#).is_err());
    }

    #[test]
    fn append_only_history_gates_on_official_binding() {
        // Fail-closed, data-driven (G7, never a provider-name string):
        // AppendOnly applies ONLY on the canonical "official" upstream
        // binding — never on eswitch/None, and never for the Off policy.
        let policy = CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: Some("official".to_string()),
            effort_enum: None,
            pinned_effort: None,
            replay: ReplayPolicy::Off,
            history: HistoryPolicy::AppendOnly,
            relocate: RelocatePolicy::Off,
        };
        assert!(policy.append_only_history_for_upstream(Some("official")));
        assert!(!policy.append_only_history_for_upstream(Some("eswitch")));
        assert!(!policy.append_only_history_for_upstream(None));

        // Off never opts in, even on the official upstream.
        let off = CachePolicy {
            history: HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            ..policy.clone()
        };
        assert!(!off.append_only_history_for_upstream(Some("official")));
        assert!(!off.append_only_history_for_upstream(None));
    }

    // --- Phase 2b.2: fail-closed session cache key contract (T16-T19) ---

    #[test]
    fn no_stable_source_returns_none_fail_closed() {
        // T16 (MUST): no session context => no key (fail-closed). A None
        // source must never be replaced by a random / UUID / time fallback.
        assert_eq!(
            session_key_from_source(None, "moonshot", "kimi-k3-turbo", None),
            None
        );
        assert_eq!(
            session_key_from_source(None, "moonshot", "kimi-k3-turbo", Some("official")),
            None
        );
    }

    #[test]
    fn same_session_derives_byte_equal_key_across_calls() {
        // T17 (MUST): the same session identity over multiple turns yields a
        // byte-equal key on every call.
        let first = session_key_from_source(Some("user_123"), "moonshot", "kimi-k3-turbo", None);
        let second = session_key_from_source(Some("user_123"), "moonshot", "kimi-k3-turbo", None);
        assert_eq!(first, second);
        assert!(first.is_some());
    }

    #[test]
    fn reconnect_and_restart_preserve_key_deterministically() {
        // T18 (MUST): reconnect / process restart must not change the key.
        // The derivation is a stateless deterministic hash of inbound signal
        // (no UUID / random / clock), so re-deriving it must yield the same
        // bytes. Exercise it across many repeated calls to mirror restarting.
        let key = session_key_from_source(Some("user_456"), "moonshot", "kimi-k3-turbo", None)
            .expect("stable source present");
        for _ in 0..100 {
            assert_eq!(
                session_key_from_source(Some("user_456"), "moonshot", "kimi-k3-turbo", None)
                    .as_deref(),
                Some(key.as_str())
            );
        }
    }

    #[test]
    fn different_session_yields_different_key() {
        // T19 (MUST, part 1): a different session identity => a different key.
        let a = session_key_from_source(Some("session-A"), "moonshot", "kimi-k3-turbo", None);
        let b = session_key_from_source(Some("session-B"), "moonshot", "kimi-k3-turbo", None);
        assert_ne!(a, b);
    }

    #[test]
    fn session_key_is_hashed_hex_and_hides_plaintext() {
        // T19 (MUST, part 2): user/token-like source text is hashed — the
        // outbound key never contains the plaintext and has a fixed
        // length/format (first 16 bytes of sha256 => 32 lowercase hex chars).
        let secret = "«redacted:sk-…»";
        let key =
            session_key_from_source(Some(secret), "moonshot", "kimi-k3-turbo", Some("official"))
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
            session_key_from_source(Some(secret), "moonshot", "kimi-k3-turbo", Some("official"))
                .as_deref(),
            Some(key.as_str())
        );
    }

    #[test]
    fn key_is_namespaced_by_provider_model_and_upstream() {
        // The plan pins the key to (user_id/session, model, provider); changing
        // any of those inputs must change the key, and an upstream binding
        // must namespace the key too.
        let base = session_key_from_source(Some("u_1"), "moonshot", "kimi-k3-turbo", None);
        let other_provider = session_key_from_source(Some("u_1"), "eswitch", "kimi-k3-turbo", None);
        let other_model =
            session_key_from_source(Some("u_1"), "moonshot", "kimi-k3-turbo-next", None);
        let bound =
            session_key_from_source(Some("u_1"), "moonshot", "kimi-k3-turbo", Some("official"));
        assert_ne!(base, other_provider, "provider change must change key");
        assert_ne!(base, other_model, "model change must change key");
        assert_ne!(base, bound, "upstream binding must namespace the key");
    }

    // --- Phase 2b.3: Responses usage view selector (legacy vs raw) ---

    fn raw_policy() -> CachePolicy {
        CachePolicy {
            usage: UsagePolicy::TopLevelCachedTokens,
            prompt_cache_key_enabled: false,
            upstream: None,
            effort_enum: None,
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            pinned_effort: None,
        }
    }

    #[test]
    fn cache_stats_mode_follows_policy_only() {
        assert_eq!(CacheStatsMode::from_policy(None), CacheStatsMode::Legacy);
        assert_eq!(
            CacheStatsMode::from_policy(Some(&CachePolicy {
                usage: UsagePolicy::Off,
                prompt_cache_key_enabled: false,
                upstream: None,
                effort_enum: None,
                replay: crate::cache::ReplayPolicy::Off,
                history: crate::cache::HistoryPolicy::Off,
                relocate: RelocatePolicy::Off,
                pinned_effort: None,
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
            prompt_cache_key_enabled: false,
            upstream: None,
            effort_enum: None,
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            pinned_effort: None,
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

    // --- Phase 2b.4: Chat usage view selector (legacy vs raw) ---

    #[test]
    fn chat_usage_view_default_off_matches_legacy_wire_baseline() {
        // Nested ptd under no policy: legacy read = ptd.cached_tokens only,
        // creation = prompt - cached (the historical remainder label), clamped
        // miss, percentage hit rate.
        let usage = chat_usage(json!({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 70},
        }));
        let view = chat_usage_view(Some(&usage), None);
        assert_eq!(view.input, Some(100));
        assert_eq!(view.read, Some(70), "legacy read = ptd.cached_tokens only");
        assert_eq!(
            view.creation,
            Some(30),
            "legacy creation = prompt - cached (historical remainder label)"
        );
        assert_eq!(view.miss, Some(30), "legacy miss = input - read");
        assert_eq!(view.hit_rate, Some(70.0));

        // An explicitly off policy must behave byte-identically to None.
        let off = CachePolicy {
            usage: UsagePolicy::Off,
            prompt_cache_key_enabled: false,
            upstream: None,
            effort_enum: None,
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: RelocatePolicy::Off,
            pinned_effort: None,
        };
        assert_eq!(chat_usage_view(Some(&usage), Some(&off)), view);
    }

    #[test]
    fn chat_usage_view_raw_under_opt_in_reads_top_level_first() {
        // Kimi top-level cached_tokens wins over nested; creation is NEVER
        // fabricated from `prompt - cached`.
        let usage = chat_usage(json!({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cached_tokens": 70,
            "prompt_tokens_details": {"cached_tokens": 60},
        }));
        let view = chat_usage_view(Some(&usage), Some(&raw_policy()));
        assert_eq!(view.read, Some(70), "raw read priority: top-level → nested");
        assert_eq!(
            view.creation, None,
            "Chat never fabricates a write/creation"
        );
        assert_eq!(view.miss, Some(30));
        assert_eq!(view.hit_rate, Some(70.0));
        // The raw projection must be exactly `from_chat_usage`.
        let stats = from_chat_usage(&usage);
        assert_eq!(view.read, stats.cache_read_tokens);
        assert_eq!(view.miss, stats.cache_miss_tokens);
    }

    #[test]
    fn chat_usage_view_raw_deepseek_hit_and_miss_are_preserved() {
        let usage = chat_usage(json!({
            "prompt_tokens": 100,
            "prompt_cache_hit_tokens": 60,
            "prompt_cache_miss_tokens": 40,
        }));
        let view = chat_usage_view(Some(&usage), Some(&raw_policy()));
        assert_eq!(view.read, Some(60));
        assert_eq!(view.creation, None);
        assert_eq!(
            view.miss,
            Some(40),
            "explicit miss preserved, not re-derived"
        );
    }

    #[test]
    fn chat_usage_view_legacy_ignores_top_level_and_deepseek() {
        // Legacy wire/log reads ONLY ptd.cached_tokens: top-level cached_tokens
        // and DeepSeek hit/miss are invisible to non-opt-in providers.
        let top = chat_usage(json!({
            "prompt_tokens": 100,
            "cached_tokens": 70,
        }));
        let legacy_top = chat_usage_view(Some(&top), None);
        assert_eq!(
            legacy_top.read, None,
            "legacy ignores top-level cached_tokens"
        );
        assert_eq!(legacy_top.creation, None);
        assert_eq!(
            legacy_top.miss,
            Some(100),
            "legacy clamps miss to input when read is unknown"
        );

        let ds = chat_usage(json!({
            "prompt_tokens": 100,
            "prompt_cache_hit_tokens": 60,
        }));
        assert_eq!(
            chat_usage_view(Some(&ds), None).read,
            None,
            "legacy ignores DeepSeek hit"
        );
    }

    #[test]
    fn chat_usage_view_raw_cached_greater_than_prompt_yields_unknown_miss() {
        let usage = chat_usage(json!({
            "prompt_tokens": 50,
            "cached_tokens": 70,
        }));
        let raw = chat_usage_view(Some(&usage), Some(&raw_policy()));
        assert_eq!(raw.read, Some(70));
        assert_eq!(
            raw.miss, None,
            "raw never fabricates a negative/clamped miss"
        );
        // Legacy ignores top-level: read unknown ⇒ miss clamps to input.
        let legacy = chat_usage_view(Some(&usage), None);
        assert_eq!(legacy.read, None);
        assert_eq!(legacy.miss, Some(50));
    }

    #[test]
    fn chat_usage_view_none_usage_is_empty_never_a_miss() {
        // HTTP error / timeout / missing usage object ⇒ empty view in BOTH
        // modes; never a fabricated miss or zero.
        let legacy = chat_usage_view(None, None);
        let raw = chat_usage_view(None, Some(&raw_policy()));
        let empty = ChatUsageView {
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
    fn chat_usage_view_zero_input_hit_rate_differs_legacy_vs_raw() {
        let usage = chat_usage(json!({
            "prompt_tokens": 0,
            "cached_tokens": 0,
        }));
        let legacy = chat_usage_view(Some(&usage), None);
        assert_eq!(
            legacy.hit_rate,
            Some(0.0),
            "legacy clamps zero input to 0.0"
        );
        assert_eq!(legacy.read, None, "legacy ignores top-level even at zero");
        assert_eq!(legacy.miss, Some(0));
        let raw = chat_usage_view(Some(&usage), Some(&raw_policy()));
        assert_eq!(
            raw.hit_rate, None,
            "raw hit rate unknown when input is zero"
        );
        assert_eq!(raw.read, Some(0));
        assert_eq!(raw.miss, Some(0), "input=0, read=0 ⇒ miss 0 (consistent)");
    }

    #[test]
    fn chat_usage_view_from_buckets_matches_non_stream_selector() {
        // Stream path uses the same selector over already-extracted buckets, so
        // streamed and non-streamed chats report identically for a policy.
        let usage = chat_usage(json!({
            "prompt_tokens": 100,
            "cached_tokens": 70,
            "prompt_tokens_details": {"cached_tokens": 60},
        }));
        let raw_via_usage = chat_usage_view(Some(&usage), Some(&raw_policy()));
        let raw_via_buckets =
            chat_usage_view_from_buckets(Some(100), Some(60), Some(70), Some(&raw_policy()));
        assert_eq!(raw_via_usage, raw_via_buckets);

        let legacy_usage = chat_usage_view(Some(&usage), None);
        let legacy_buckets = chat_usage_view_from_buckets(Some(100), Some(60), None, None);
        assert_eq!(legacy_usage, legacy_buckets);
    }

    #[test]
    fn chat_usage_view_creation_never_fabricated_under_raw_across_shapes() {
        // Every Chat shape under opt-in reports creation None — the historical
        // `prompt - cached` remainder is miss, never a write.
        for value in [
            json!({"prompt_tokens": 100, "cached_tokens": 70}),
            json!({"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 70}}),
            json!({"prompt_tokens": 100, "prompt_cache_hit_tokens": 60}),
        ] {
            let usage = chat_usage(value);
            let view = chat_usage_view(Some(&usage), Some(&raw_policy()));
            assert_eq!(view.creation, None);
        }
    }
}
