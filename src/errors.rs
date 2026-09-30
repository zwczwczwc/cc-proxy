//! Error-class mapping for the Anthropic-compatible surface (Package A).
//!
//! WHY THIS EXISTS
//! ---------------
//! Claude Code recovers from upstream capability rejections by matching the
//! upstream's own error *wording*, or a stable token, against a fixed class
//! table. When this proxy replaces the upstream error body it hides that
//! wording, so the client stops recognising recoverable failures and the
//! session is stranded. The client-side predicates are message-based and do
//! NOT look at the HTTP status (verified in claude.exe: `M2(e) =
//! ILe(e.message) || wS(e.message, "prompt_too_long")`), and the token matcher
//! `wS` only requires a non-identifier character after the token — therefore a
//! message may carry the official literal, the token, and the upstream text.
//!
//! SCOPE (Package A-lite)
//! ----------------------
//! Status codes are left untouched here. Only the *message* is rewritten, and
//! only when a class matches with confidence; every other error keeps the
//! exact envelope it had before. Classification is deliberately conservative:
//! per the gateway contract, "a wrong token triggers the wrong client
//! recovery". In particular the Chinese upstream message
//! `该模型始终思考，不支持关闭思考；请使用 low、high 或 max。` MUST NOT match
//! anything — it is fixed on the proxy side (Package B), not by a token.

/// Official class literals, taken from the gateway contract embedded in the
/// client. A class maps to the wording the client itself matches on.
const LIT_PROMPT_TOO_LONG: &str = "Prompt is too long";
const LIT_MAX_TOKENS_OVERFLOW: &str = "input length and `max_tokens` exceed context limit";
const LIT_THINKING_SIGNATURE: &str = "Invalid signature in thinking block";
const LIT_EFFORT_UNSUPPORTED: &str = "This model does not support the effort parameter";
const LIT_MEDIA_BUDGET: &str = "Too much media";
const LIT_IMAGE_BLOCK: &str = "Could not process image";
const LIT_DOCUMENT_BLOCK: &str = "Could not process PDF";

const CLASS_PROMPT_TOO_LONG: &str = "prompt_too_long";
const CLASS_MAX_TOKENS_OVERFLOW: &str = "max_tokens_context_overflow";
const CLASS_THINKING_SIGNATURE: &str = "thinking_signature";
const CLASS_EFFORT_UNSUPPORTED: &str = "effort_unsupported";
const CLASS_MEDIA_BUDGET: &str = "media_budget";
const CLASS_IMAGE_BLOCK: &str = "image_block";
const CLASS_DOCUMENT_BLOCK: &str = "document_block";

/// Truncate on a char boundary (byte slicing `&s[..n]` panics on multi-byte
/// input — the upstream errors on this deployment are Chinese).
pub fn truncate_chars(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = 0usize;
    for (i, c) in s.char_indices() {
        let next = i + c.len_utf8();
        if next > max_bytes {
            break;
        }
        end = next;
    }
    &s[..end]
}

/// Classify an upstream error message. Wording-only (no status input) because
/// the client's own recovery predicates are wording-only as well; callers that
/// hold the HTTP status may additionally gate on it.
///
/// Order and conditions mirror the official class table. Only classes whose
/// wording is unambiguous are implemented.
pub fn classify(msg: &str) -> Option<&'static str> {
    let m = msg.to_lowercase();
    let has = |needle: &str| m.contains(needle);

    // 1. context overflow — the deployment's actual upstream wording included.
    if has("prompt exceeds max length")
        || has("prompt is too long")
        || has("input is too long for requested model")
    {
        return Some(CLASS_PROMPT_TOO_LONG);
    }
    if has("input length and `max_tokens` exceed context limit") {
        return Some(CLASS_MAX_TOKENS_OVERFLOW);
    }

    // 2. thinking-block signature rejected.
    if has("signature in thinking block")
        || (has("thinking.signature") && has("field required"))
        || ((has("thinking block") || has("redacted_thinking"))
            && (has("cannot be modified") || has("invalid signature")))
    {
        return Some(CLASS_THINKING_SIGNATURE);
    }

    // 3. effort / output_config rejected. NOTE: the bare phrase
    //    "requires a model that supports" also appears in unrelated tool
    //    failures, so it only counts together with an effort/output_config cue.
    if has("does not support the effort parameter")
        || (has("extra inputs are not permitted") && has("output_config"))
        || (has("requires a model that supports") && (has("effort") || has("output_config")))
    {
        return Some(CLASS_EFFORT_UNSUPPORTED);
    }

    // 4. media / image / document blocks.
    if has("too much media") {
        return Some(CLASS_MEDIA_BUDGET);
    }
    if has("could not process image")
        || has("image exceeds")
        || has("image dimensions exceed")
        || has("image cannot be empty")
    {
        return Some(CLASS_IMAGE_BLOCK);
    }
    if has("could not process pdf") || has("pdf pages") || has("pdf cannot be empty") {
        return Some(CLASS_DOCUMENT_BLOCK);
    }

    // Deliberately NOT classified (conservative; wrong token = wrong recovery):
    //   - "该模型始终思考，不支持关闭思考…"  (fixed proxy-side, Package B)
    //   - "thinking.type `disabled` is not supported" (class enum is
    //     <enabled|adaptive> only, so no valid class exists for `disabled`)
    //   - mid_conv_system / cache_control_field / beta_header (need request
    //     context this layer does not have)
    None
}

/// The human-readable literal for a class (what the client matches on).
pub fn literal_for(class: &str) -> &'static str {
    match class {
        CLASS_PROMPT_TOO_LONG => LIT_PROMPT_TOO_LONG,
        CLASS_MAX_TOKENS_OVERFLOW => LIT_MAX_TOKENS_OVERFLOW,
        CLASS_THINKING_SIGNATURE => LIT_THINKING_SIGNATURE,
        CLASS_EFFORT_UNSUPPORTED => LIT_EFFORT_UNSUPPORTED,
        CLASS_MEDIA_BUDGET => LIT_MEDIA_BUDGET,
        CLASS_IMAGE_BLOCK => LIT_IMAGE_BLOCK,
        CLASS_DOCUMENT_BLOCK => LIT_DOCUMENT_BLOCK,
        _ => "upstream rejected the request",
    }
}

/// Build the client-facing message for a matched class.
///
/// Shape (deliberately satisfies every known client predicate at once):
///   `<official literal> (upstream: <sanitized upstream text>) [capability_rejected: <class>]`
/// * the literal satisfies the wording matchers (`ILe`/`Fft`/`hWo`/…),
/// * the token satisfies `wS` (it is followed by `]`, a non-identifier char),
/// * the parenthetical keeps the upstream text and any request id for support.
pub fn capability_message(class: &str, upstream: &str) -> String {
    let short = truncate_chars(upstream.trim(), 600);
    format!(
        "{} (upstream: {}) [capability_rejected: {}]",
        literal_for(class),
        short,
        class
    )
}

/// Rewrite an upstream error into a capability-token message when a class
/// matches, otherwise return `fallback` unchanged (zero behaviour change for
/// every non-classified error).
pub fn maybe_capability_message(upstream: &str, fallback: String) -> String {
    match classify(upstream) {
        Some(class) => capability_message(class, upstream),
        None => fallback,
    }
}

/// Message-only helper that still honours `CC_PROXY_ERROR_CONTRACT=off`
/// (used by the Responses-branch sites, which keep their own status).
pub fn message_only(err_text: &str, fallback: String) -> String {
    match contract_mode() {
        ContractMode::Off => fallback,
        _ => maybe_capability_message(err_text, fallback),
    }
}

// ───────────────────────── Package A-full ─────────────────────────

/// Contract mode, runtime-switchable so a bad rollout is one env change away
/// from the previous behaviour (no rebuild, no redeploy):
///   `off`  → legacy: 500/502 + original message text (pre-A behaviour)
///   `lite` → default: messages rewritten, statuses untouched
///   `full` → messages rewritten + upstream status preserved
pub fn contract_mode() -> ContractMode {
    match std::env::var("CC_PROXY_ERROR_CONTRACT")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "off" | "0" | "false" => ContractMode::Off,
        "full" | "on" | "1" | "true" => ContractMode::Full,
        _ => ContractMode::Lite,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractMode {
    Off,
    Lite,
    Full,
}

/// Extract the upstream HTTP status the proxy embedded in its own error text
/// (`DeepSeek API error (400 Bad Request): …`, `… (429 Too Many Requests)`).
///
/// Parsing the text avoids restructuring the retry loop's `anyhow` error type;
/// the wording is produced by this crate, so it is a stable contract.
pub fn status_from_error_text(s: &str) -> Option<u16> {
    let idx = s.find("API error (")?;
    let rest = &s[idx + "API error (".len()..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse::<u16>().ok()
}

/// A status we are willing to relay verbatim.
///
/// Deliberate exclusions (review R2/R4): 401/403 are held at the proxy's own
/// 5xx because the upstream's 401 means "the *gateway's* token is exhausted",
/// and relaying it makes Claude Code believe its own credential is bad; 429 is
/// excluded so the existing retry semantics do not change in this rollout.
pub fn relayable_status(status: Option<u16>) -> Option<u16> {
    match status {
        Some(s @ (400 | 404 | 408 | 413 | 422 | 500 | 501 | 502 | 503 | 529)) => Some(s),
        _ => None,
    }
}

/// The client-visible error type for a status (official mapping table).
pub fn err_type_for(status: u16) -> &'static str {
    match status {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        501 => "not_supported",
        529 => "overloaded_error",
        _ => "api_error",
    }
}

/// Should the upstream call be retried?
///
/// Deterministic 4xx rejections (400/401/403/404/413/422 …) do not change on a
/// second attempt: over 14 days 1,559 of 1,991 retries (78%) were spent
/// re-sending 400s that could never succeed, and each retry also slept 2–4 s.
/// Connection-level failures (no status) and 408/429/5xx stay retryable.
pub fn is_retryable(err_text: &str) -> bool {
    match status_from_error_text(err_text) {
        None => true,
        Some(s) if s == 408 || s == 429 || s >= 500 => true,
        Some(_) => false,
    }
}

/// Resolve (status, message) for an upstream failure according to the mode.
/// Returns the proxy's default status when the mode or the status forbids
/// relaying.
pub fn resolve(err_text: &str, default_status: u16, fallback_message: String) -> (u16, String) {
    let mode = contract_mode();
    let message = match mode {
        ContractMode::Off => fallback_message,
        _ => maybe_capability_message(err_text, fallback_message),
    };
    let status = match mode {
        ContractMode::Full => relayable_status(status_from_error_text(err_text)).unwrap_or(default_status),
        _ => default_status,
    };
    (status, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_this_deployments_context_overflow() {
        // exact upstream wording observed 34x in the 14-day log
        let m = "Prompt exceeds max length";
        assert_eq!(classify(m), Some("prompt_too_long"));
        let out = capability_message("prompt_too_long", m);
        assert!(out.starts_with("Prompt is too long"), "{out}");
        assert!(out.contains("capability_rejected: prompt_too_long"), "{out}");
        // token must be followed by a non-identifier char (wS contract)
        let idx = out.find("capability_rejected: prompt_too_long").unwrap();
        let after = out[idx + "capability_rejected: prompt_too_long".len()..].chars().next();
        assert!(matches!(after, Some(c) if !c.is_ascii_alphanumeric() && !"_:.-".contains(c)));
    }

    #[test]
    fn official_wordings_are_recognised() {
        assert_eq!(classify("prompt is too long: 210000 tokens > 200000"), Some("prompt_too_long"));
        assert_eq!(classify("Input is too long for requested model"), Some("prompt_too_long"));
        assert_eq!(
            classify("input length and `max_tokens` exceed context limit: 100 + 5 > 100"),
            Some("max_tokens_context_overflow")
        );
        assert_eq!(classify("Invalid signature in thinking block"), Some("thinking_signature"));
        assert_eq!(
            classify("This model does not support the effort parameter"),
            Some("effort_unsupported")
        );
        assert_eq!(classify("Too much media: 20 document pages + 0 images > 100"), Some("media_budget"));
        assert_eq!(classify("Could not process image"), Some("image_block"));
        assert_eq!(classify("Could not process PDF"), Some("document_block"));
    }

    /// The most important negative test: the upstream message this deployment
    /// actually sees must never be tokenised.
    #[test]
    fn chinese_always_thinking_message_is_never_classified() {
        let m = "该模型始终思考，不支持关闭思考；请使用 low、high 或 max。 (request id: 2026091320520340995041970621804)";
        assert_eq!(classify(m), None);
        assert_eq!(
            classify("thinking.type `disabled` is not supported by this model"),
            None
        );
    }

    #[test]
    fn unrelated_errors_keep_their_fallback_text() {
        let upstream = "该令牌额度已用尽 (request id: 123)";
        let fallback = "Upstream error after 2 retries: xyz".to_string();
        assert_eq!(maybe_capability_message(upstream, fallback.clone()), fallback);
    }

    #[test]
    fn bare_requires_a_model_phrase_is_not_effort_unsupported() {
        // must not fire without an effort/output_config cue
        assert_eq!(classify("tool X requires a model that supports schemas"), None);
        assert_eq!(
            classify("output_config requires a model that supports effort"),
            Some("effort_unsupported")
        );
    }

    #[test]
    fn retry_policy_skips_deterministic_4xx() {
        let e = |s: u16| format!("DeepSeek API error ({s} Bad Request): boom");
        assert!(!is_retryable(&e(400)));
        assert!(!is_retryable(&e(401)));
        assert!(!is_retryable(&e(403)));
        assert!(!is_retryable(&e(404)));
        assert!(!is_retryable(&e(413)));
        assert!(is_retryable(&e(408)));
        assert!(is_retryable(&e(429)));
        assert!(is_retryable(&e(500)));
        assert!(is_retryable(&e(502)));
        assert!(is_retryable(&e(503)));
        assert!(is_retryable(&e(529)));
        // 连接层错误（无状态码）仍应重试
        assert!(is_retryable("error sending request for url (http://clawbot:11434/v1/chat/completions)"));
    }

    #[test]
    fn extracts_upstream_status_from_proxy_wording() {
        assert_eq!(
            status_from_error_text("DeepSeek API error (400 Bad Request): {\"error\":{}}"),
            Some(400)
        );
        assert_eq!(
            status_from_error_text("DeepSeek API error (401 Unauthorized): quota"),
            Some(401)
        );
        assert_eq!(status_from_error_text("error sending request for url (...)"), None);
    }

    /// 401/403/429 must NOT be relayed (client would blame its own credential /
    /// retry semantics would silently change).
    #[test]
    fn relayable_status_excludes_auth_and_rate_limit() {
        assert_eq!(relayable_status(Some(400)), Some(400));
        assert_eq!(relayable_status(Some(413)), Some(413));
        assert_eq!(relayable_status(Some(401)), None);
        assert_eq!(relayable_status(Some(403)), None);
        assert_eq!(relayable_status(Some(429)), None);
        assert_eq!(relayable_status(Some(302)), None, "3xx must never be relayed");
        assert_eq!(relayable_status(None), None);
    }

    #[test]
    fn error_types_follow_the_official_table() {
        assert_eq!(err_type_for(400), "invalid_request_error");
        assert_eq!(err_type_for(401), "authentication_error");
        assert_eq!(err_type_for(413), "request_too_large");
        assert_eq!(err_type_for(429), "rate_limit_error");
        assert_eq!(err_type_for(529), "overloaded_error");
        assert_eq!(err_type_for(502), "api_error");
    }

    #[test]
    fn truncation_never_splits_multibyte_chars() {
        let s = "该模型始终思考，不支持关闭思考；请使用 low、high 或 max。".repeat(40);
        for n in [1usize, 2, 3, 4, 1023, 1024, 1025] {
            let t = truncate_chars(&s, n);
            assert!(s.starts_with(t));
            assert!(t.len() <= n || n < 3);
        }
        // 1024 bytes of this string is inside a character; must not panic
        let _ = truncate_chars(&s, 1024);
    }
}
