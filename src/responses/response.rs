use super::types::ResponsesResponse;
use crate::anthropic::types::{MessagesResponse, ResponseContentBlock, Usage};
use crate::cache::{responses_usage_view, CachePolicy};
use serde_json::Value;

/// Legacy (non-opt-in) Responses cache bucket adapter.
///
/// This is the byte-preserving baseline: the exact arithmetic the proxy has
/// shipped since the Responses transport landed (`legacy_cache_stats_from_values`
/// clamps the miss and returns `0.0` hit rate on zero input). Providers that
/// have not opted into a `cache_policy` keep logging these numbers. The
/// canonical raw projection lives in `crate::cache::CacheStats`, selected by
/// `responses_usage_view` (default off) — the two are mutually exclusive per
/// request and never both computed.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "legacy Responses baseline adapter; asserted by unit tests, raw path lives in cache::responses_usage_view"
    )
)]
pub(crate) struct LegacyCacheStats {
    pub(crate) input_tokens: Option<u32>,
    pub(crate) cache_read_input_tokens: Option<u32>,
    pub(crate) cache_creation_input_tokens: Option<u32>,
    pub(crate) cache_miss_input_tokens: Option<u32>,
    pub(crate) hit_rate_percent: Option<f64>,
}

/// Legacy arithmetic only: clamped miss, `0.0` hit rate on zero input.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "legacy Responses baseline adapter; asserted by unit tests, raw path lives in cache::responses_usage_view"
    )
)]
pub(crate) fn legacy_cache_stats_from_values(
    input_tokens: Option<u32>,
    cache_read_input_tokens: Option<u32>,
    cache_creation_input_tokens: Option<u32>,
) -> LegacyCacheStats {
    let cache_miss_input_tokens = input_tokens.map(|input| {
        input
            .saturating_sub(cache_read_input_tokens.unwrap_or(0))
            .saturating_sub(cache_creation_input_tokens.unwrap_or(0))
    });
    let hit_rate_percent = input_tokens.map(|input| {
        if input == 0 {
            0.0
        } else {
            cache_read_input_tokens.unwrap_or(0) as f64 / input as f64 * 100.0
        }
    });
    LegacyCacheStats {
        input_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
        cache_miss_input_tokens,
        hit_rate_percent,
    }
}

pub fn convert_response(
    response: &ResponsesResponse,
    model: &str,
    id: &str,
    cache_policy: Option<&CachePolicy>,
) -> anyhow::Result<MessagesResponse> {
    if let Some(error) = &response.error {
        anyhow::bail!("Responses API error: {error}");
    }
    if response.status.as_deref() == Some("failed") {
        let message = response
            .error
            .as_ref()
            .and_then(|e| e.get("message"))
            .and_then(Value::as_str)
            .or_else(|| {
                response
                    .incomplete_details
                    .as_ref()
                    .and_then(|e| e.get("message"))
                    .and_then(Value::as_str)
            })
            .unwrap_or("Responses API failed");
        anyhow::bail!("Responses API failed: {message}");
    }
    match response.status.as_deref() {
        Some("completed") | Some("incomplete") => {}
        Some(status) => anyhow::bail!("Responses API non-terminal status: {status}"),
        None => anyhow::bail!("Responses API response missing status"),
    }
    let mut content = Vec::new();
    let mut has_tool = false;
    for item in &response.output {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                if let Some(parts) = item.get("content").and_then(Value::as_array) {
                    for part in parts {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            content.push(ResponseContentBlock::Text {
                                text: text.to_string(),
                            });
                        }
                    }
                }
            }
            Some("function_call") => {
                let input: Value = serde_json::from_str(
                    item.get("arguments").and_then(Value::as_str).unwrap_or(""),
                )?;
                content.push(ResponseContentBlock::ToolUse {
                    id: item
                        .get("call_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    name: item
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    input,
                });
                has_tool = true;
            }
            Some("reasoning") => {
                if let Some(summary) = item.get("summary").and_then(Value::as_array) {
                    for part in summary {
                        if let Some(text) = part
                            .get("text")
                            .and_then(Value::as_str)
                            .filter(|text| !text.is_empty())
                        {
                            content.push(ResponseContentBlock::Thinking {
                                thinking: text.to_string(),
                                // Responses exposes summary text, not Anthropic's encrypted
                                // reasoning signature. An empty signature is intentional:
                                // never fabricate a value or replay unverifiable ciphertext.
                                signature: String::new(),
                            });
                        }
                    }
                }
            }
            Some("refusal") => {
                if let Some(text) = item.get("refusal").and_then(Value::as_str) {
                    content.push(ResponseContentBlock::Text {
                        text: text.to_string(),
                    });
                }
            }
            _ => {}
        }
    }
    let usage = response.usage.as_ref();
    let details = usage.and_then(|u| u.input_tokens_details.as_ref());
    if let Some(usage) = usage {
        // The policy-gated selector picks exactly one mode per request: Legacy
        // (default — policy None/off) reproduces the historical three-bucket
        // numbers byte-for-byte; Raw (explicit usage source) reports the
        // canonical `CacheStats` buckets. Both modes feed this single log, so
        // log and wire below can never disagree.
        let view = responses_usage_view(Some(usage), cache_policy);
        tracing::info!(
            upstream_http_status = 200u16,
            status = response.status.as_deref().unwrap_or("unknown"),
            input_tokens = ?view.input,
            output_tokens = ?usage.output_tokens,
            cache_read_input_tokens = ?view.read,
            cache_creation_input_tokens = ?view.creation,
            cache_miss_input_tokens = ?view.miss,
            hit_rate_percent = ?view.hit_rate,
            "Responses cache stats"
        );
    }
    Ok(MessagesResponse {
        id: id.to_string(),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: response.model.as_deref().unwrap_or(model).to_string(),
        stop_reason: Some(
            if has_tool {
                "tool_use"
            } else if response.status.as_deref() == Some("incomplete") {
                match response
                    .incomplete_details
                    .as_ref()
                    .and_then(|v| v.get("reason"))
                    .and_then(Value::as_str)
                {
                    Some("max_output_tokens") | Some("max_tokens") => "max_tokens",
                    _ => "end_turn",
                }
            } else {
                "end_turn"
            }
            .to_string(),
        ),
        stop_sequence: None,
        usage: Usage {
            input_tokens: usage.and_then(|u| u.input_tokens).unwrap_or(0),
            output_tokens: usage.and_then(|u| u.output_tokens).unwrap_or(0),
            cache_read_input_tokens: details.and_then(|d| d.cached_tokens),
            cache_creation_input_tokens: details
                .and_then(|d| d.cache_write_tokens)
                .or_else(|| usage.and_then(|u| u.cache_write_tokens)),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses::types::{InputTokenDetails, ResponsesUsage};

    fn response(status: &str, output: Vec<Value>) -> ResponsesResponse {
        ResponsesResponse {
            id: Some("resp-1".into()),
            model: Some("gpt-5.6-luna".into()),
            status: Some(status.into()),
            output,
            usage: None,
            error: None,
            incomplete_details: None,
        }
    }

    #[test]
    fn converts_function_call_and_usage_cache_fields() {
        let mut response = response(
            "completed",
            vec![
                serde_json::json!({"type":"function_call","call_id":"call-1","name":"lookup","arguments":"{\"city\":\"Paris\"}"}),
            ],
        );
        response.usage = Some(ResponsesUsage {
            input_tokens: Some(100),
            output_tokens: Some(12),
            input_tokens_details: Some(InputTokenDetails {
                cached_tokens: Some(80),
                cache_write_tokens: Some(3),
            }),
            cache_write_tokens: Some(4),
        });
        let converted = convert_response(&response, "fallback", "msg-1", None).unwrap();
        assert_eq!(converted.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(converted.usage.cache_read_input_tokens, Some(80));
        assert_eq!(converted.usage.cache_creation_input_tokens, Some(3));
    }

    #[test]
    fn maps_incomplete_and_rejects_failed_or_malformed() {
        let mut incomplete = response("incomplete", Vec::new());
        incomplete.incomplete_details = Some(serde_json::json!({"reason":"max_output_tokens"}));
        assert_eq!(
            convert_response(&incomplete, "fallback", "msg-1", None)
                .unwrap()
                .stop_reason
                .as_deref(),
            Some("max_tokens")
        );
        assert!(
            convert_response(&response("failed", Vec::new()), "fallback", "msg-1", None).is_err()
        );
        assert!(convert_response(
            &response(
                "completed",
                vec![serde_json::json!({"type":"function_call","arguments":"not-json"})]
            ),
            "fallback",
            "msg-1",
            None
        )
        .is_err());
    }

    #[test]
    fn cache_stats_distinguish_read_creation_and_uncached_tokens() {
        let usage = ResponsesUsage {
            input_tokens: Some(100),
            output_tokens: Some(12),
            input_tokens_details: Some(InputTokenDetails {
                cached_tokens: Some(70),
                cache_write_tokens: Some(20),
            }),
            cache_write_tokens: None,
        };
        // Default policy (None) ⇒ Legacy mode, the exact historical numbers.
        let view = crate::cache::responses_usage_view(Some(&usage), None);
        assert_eq!(view.read, Some(70));
        assert_eq!(view.creation, Some(20));
        assert_eq!(view.miss, Some(10));
        assert_eq!(view.hit_rate, Some(70.0));
        // The dedicated legacy baseline adapter pins the same values.
        let stats = legacy_cache_stats_from_values(
            usage.input_tokens,
            usage
                .input_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens),
            usage
                .input_tokens_details
                .as_ref()
                .and_then(|details| details.cache_write_tokens)
                .or(usage.cache_write_tokens),
        );
        assert_eq!(stats.cache_read_input_tokens, Some(70));
        assert_eq!(stats.cache_creation_input_tokens, Some(20));
        assert_eq!(stats.cache_miss_input_tokens, Some(10));
        assert_eq!(stats.hit_rate_percent, Some(70.0));
    }

    #[test]
    fn legacy_baseline_matches_view_legacy_branch_across_buckets() {
        // Drift guard: the view's Legacy branch and `legacy_cache_stats_from_values`
        // must agree on every bucket, so the wired log never diverges from the
        // documented baseline adapter.
        let usage = ResponsesUsage {
            input_tokens: Some(100),
            output_tokens: Some(12),
            input_tokens_details: Some(InputTokenDetails {
                cached_tokens: Some(70),
                cache_write_tokens: Some(5),
            }),
            cache_write_tokens: None,
        };
        let view = crate::cache::responses_usage_view(Some(&usage), None);
        let legacy = legacy_cache_stats_from_values(
            usage.input_tokens,
            usage
                .input_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens),
            usage
                .input_tokens_details
                .as_ref()
                .and_then(|details| details.cache_write_tokens)
                .or(usage.cache_write_tokens),
        );
        assert_eq!(view.input, legacy.input_tokens);
        assert_eq!(view.read, legacy.cache_read_input_tokens);
        assert_eq!(view.creation, legacy.cache_creation_input_tokens);
        assert_eq!(view.miss, legacy.cache_miss_input_tokens);
        assert_eq!(view.hit_rate, legacy.hit_rate_percent);
    }

    #[test]
    fn convert_response_completed_usage_preserves_wire_and_log_legacy() {
        // Non-stream completed usage under the default (None) policy: the wire
        // fields and the view's legacy buckets must match the pre-2b.3 baseline
        // exactly (read = nested cached_tokens, creation = nested then top-level
        // write fallback, guarded-by-consistency legacy miss).
        let mut response = response("completed", Vec::new());
        response.usage = Some(ResponsesUsage {
            input_tokens: Some(100),
            output_tokens: Some(12),
            input_tokens_details: Some(InputTokenDetails {
                cached_tokens: Some(80),
                cache_write_tokens: Some(3),
            }),
            cache_write_tokens: Some(4),
        });
        let converted = convert_response(&response, "fallback", "msg-1", None).unwrap();
        assert_eq!(converted.usage.input_tokens, 100);
        assert_eq!(converted.usage.output_tokens, 12);
        assert_eq!(
            converted.usage.cache_read_input_tokens,
            Some(80),
            "wire read unchanged under legacy"
        );
        assert_eq!(
            converted.usage.cache_creation_input_tokens,
            Some(3),
            "wire creation = nested cache_write_tokens (top-level is only a fallback)"
        );
        let view = crate::cache::responses_usage_view(response.usage.as_ref(), None);
        assert_eq!(view.read, Some(80));
        assert_eq!(view.creation, Some(3));
        assert_eq!(view.miss, Some(17));
        assert_eq!(view.hit_rate, Some(80.0));
    }

    #[test]
    fn convert_response_raw_view_under_explicit_policy_keeps_wire_unchanged() {
        // Non-stream completed usage under an explicit policy: the log view
        // switches to the raw `CacheStats` projection while the outbound wire
        // stays byte-identical to legacy (read/creation sources are the same).
        let mut response = response("completed", Vec::new());
        response.usage = Some(ResponsesUsage {
            input_tokens: Some(100),
            output_tokens: Some(12),
            input_tokens_details: Some(InputTokenDetails {
                cached_tokens: Some(80),
                cache_write_tokens: Some(3),
            }),
            cache_write_tokens: None,
        });
        let policy = crate::cache::CachePolicy {
            usage: crate::cache::UsagePolicy::TopLevelCachedTokens,
            prompt_cache_key_enabled: false,
            upstream: None,
            effort_enum: None,
            replay: crate::cache::ReplayPolicy::Off,
            history: crate::cache::HistoryPolicy::Off,
            relocate: crate::cache::RelocatePolicy::Off,
            pinned_effort: None,
        };
        let converted = convert_response(&response, "fallback", "msg-1", Some(&policy)).unwrap();
        assert_eq!(converted.usage.cache_read_input_tokens, Some(80));
        assert_eq!(converted.usage.cache_creation_input_tokens, Some(3));
        // The raw view matches the canonical CacheStats projection.
        let view = crate::cache::responses_usage_view(response.usage.as_ref(), Some(&policy));
        let stats = crate::cache::from_responses_usage(response.usage.as_ref().unwrap());
        assert_eq!(view.read, stats.cache_read_tokens);
        assert_eq!(view.creation, stats.cache_write_tokens);
        assert_eq!(view.miss, stats.cache_miss_tokens);
        assert_eq!(view.miss, Some(17));
        assert_eq!(view.hit_rate, Some(80.0));
    }

    #[test]
    fn convert_response_without_usage_emits_zero_wire_and_no_log_side_effects() {
        // No usage object: wire usage is all-zero (unchanged legacy behavior)
        // and no log branch runs — an absent usage object is never a cache miss.
        let response = response("completed", Vec::new());
        let converted = convert_response(&response, "fallback", "msg-1", None).unwrap();
        assert_eq!(converted.usage.input_tokens, 0);
        assert_eq!(converted.usage.cache_read_input_tokens, None);
        assert_eq!(converted.usage.cache_creation_input_tokens, None);
    }

    #[test]
    fn readable_reasoning_summary_is_unsigned_and_encrypted_payload_is_not_thinking() {
        let response = response(
            "completed",
            vec![
                serde_json::json!({
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "short summary"}],
                    "encrypted_content": "ciphertext"
                }),
                serde_json::json!({
                    "type": "reasoning",
                    "encrypted_content": "ciphertext-only"
                }),
            ],
        );
        let converted = convert_response(&response, "fallback", "msg-1", None).unwrap();
        assert_eq!(converted.content.len(), 1);
        match &converted.content[0] {
            ResponseContentBlock::Thinking {
                thinking,
                signature,
            } => {
                assert_eq!(thinking, "short summary");
                assert!(signature.is_empty());
            }
            other => panic!("expected thinking block, got {other:?}"),
        }
    }

    #[test]
    fn rejects_non_terminal_response_statuses() {
        for status in ["cancelled", "in_progress", "queued"] {
            assert!(
                convert_response(&response(status, Vec::new()), "fallback", "msg-1", None).is_err()
            );
        }
    }

    #[test]
    fn empty_reasoning_summary_does_not_emit_thinking_block() {
        let response = response(
            "completed",
            vec![serde_json::json!({
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": ""}]
            })],
        );
        let converted = convert_response(&response, "fallback", "msg-1", None).unwrap();
        assert!(converted.content.is_empty());
    }
}
