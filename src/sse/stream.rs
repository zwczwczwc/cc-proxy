use axum::response::sse::{Event, Sse};
use futures_util::stream::Stream;
use serde_json::Value;
use std::convert::Infallible;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use tokio::time::timeout;

use crate::anthropic::types::SseEvent;
use crate::openai::converter::SseStateMachine;

/// Convert an SseEvent to an axum SSE Event.
pub fn sse_event_to_axum(event: &SseEvent) -> Event {
    let json = serde_json::to_string(event).unwrap_or_default();
    match event {
        SseEvent::MessageStart { .. } => Event::default().event("message_start").data(json),
        SseEvent::ContentBlockStart { .. } => {
            Event::default().event("content_block_start").data(json)
        }
        SseEvent::ContentBlockDelta { .. } => {
            Event::default().event("content_block_delta").data(json)
        }
        SseEvent::ContentBlockStop { .. } => {
            Event::default().event("content_block_stop").data(json)
        }
        SseEvent::MessageDelta { .. } => Event::default().event("message_delta").data(json),
        SseEvent::MessageStop => Event::default().event("message_stop").data(json),
        SseEvent::Error { .. } => Event::default().event("error").data(json),
    }
}

/// A stream that wraps the SSE parsing logic and emits axum SSE Events.
pub struct SseEventStream {
    receiver: mpsc::Receiver<Event>,
}

impl Stream for SseEventStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx).map(|v| v.map(Ok))
    }
}

/// Process a stream of OpenAI SSE chunks into Anthropic SSE events.
/// Returns an axum Sse response.
pub fn process_stream(
    model: String,
    is_reasoning_model: bool,
    reasoning_field: String,
    reasoning_field_alt: Vec<String>,
    msg_id: String,
    cache_policy: Option<crate::cache::CachePolicy>,
    body_stream: impl Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + Unpin + 'static,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<Event>(256);

    const MAX_SSE_BUF: usize = 4 * 1024 * 1024; // 4MB
    let idle_timeout = tokio::time::Duration::from_secs(
        std::env::var("PROXY_STREAM_IDLE_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(600),
    );

    tokio::spawn(async move {
        let mut state_machine = SseStateMachine::new(
            is_reasoning_model,
            reasoning_field,
            reasoning_field_alt,
            cache_policy.clone(),
        );

        // Send message_start first (audit defect 3.1)
        let msg_start = state_machine.message_start(&model, &msg_id);
        let _ = tx.send(sse_event_to_axum(&msg_start)).await;

        use futures_util::StreamExt;
        let mut stream = Box::pin(body_stream);

        let mut buffer = String::new();
        let mut done = false;
        let mut completed = false; // dsv4-cc-proxy pattern: prevent duplicate finalize
        let mut last_usage: Option<crate::openai::types::Usage> = None;
        let mut pending_finish_reason: Option<String> = None;
        let mut pending_output_tokens: Option<u32> = None;

        loop {
            let chunk_result = match timeout(idle_timeout, stream.next()).await {
                Ok(Some(Ok(chunk))) => Ok(chunk),
                Ok(Some(Err(e))) => Err(e),
                Ok(None) => {
                    // stream ended — finalize pending if any (eswitch may not send [DONE])
                    if pending_finish_reason.is_some() {
                        let output_tokens = last_usage
                            .as_ref()
                            .and_then(|u| u.completion_tokens)
                            .or(pending_output_tokens);
                        let final_events = state_machine.finalize(
                            pending_finish_reason.as_deref(),
                            output_tokens,
                            last_usage.as_ref(),
                        );
                        for event in &final_events {
                            let _ = tx.send(sse_event_to_axum(event)).await;
                        }
                    }
                    break;
                }
                Err(_elapsed) => {
                    tracing::warn!("SSE stream idle timeout after {:?}", idle_timeout);
                    if pending_finish_reason.is_some() {
                        let output_tokens = last_usage
                            .as_ref()
                            .and_then(|u| u.completion_tokens)
                            .or(pending_output_tokens);
                        let final_events = state_machine.finalize(
                            pending_finish_reason.as_deref(),
                            output_tokens,
                            last_usage.as_ref(),
                        );
                        for event in &final_events {
                            let _ = tx.send(sse_event_to_axum(event)).await;
                        }
                    } else {
                        let final_events = state_machine.finalize(None, None, None);
                        for event in &final_events {
                            let _ = tx.send(sse_event_to_axum(event)).await;
                        }
                    }
                    return;
                }
            };

            match chunk_result {
                Ok(chunk) => {
                    let chunk_str = String::from_utf8_lossy(&chunk);
                    buffer.push_str(&chunk_str);

                    // P0-2: buffer size guard
                    if buffer.len() > MAX_SSE_BUF {
                        tracing::error!(
                            "SSE buffer exceeded {} bytes, aborting stream",
                            MAX_SSE_BUF
                        );
                        break;
                    }

                    // Process complete SSE lines
                    while let Some(line_end) = buffer.find('\n') {
                        let line = buffer[..line_end].trim().to_string();
                        buffer = buffer[line_end + 1..].to_string();

                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }

                        // Handle "data: " prefix
                        let data = if let Some(data) = line.strip_prefix("data: ") {
                            data
                        } else if let Some(data) = line.strip_prefix("data:") {
                            data
                        } else {
                            continue;
                        };

                        if data == "[DONE]" {
                            if completed && pending_finish_reason.is_some() {
                                let output_tokens = last_usage
                                    .as_ref()
                                    .and_then(|u| u.completion_tokens)
                                    .or(pending_output_tokens);
                                let final_events = state_machine.finalize(
                                    pending_finish_reason.as_deref(),
                                    output_tokens,
                                    last_usage.as_ref(),
                                );
                                for event in &final_events {
                                    let _ = tx.send(sse_event_to_axum(event)).await;
                                }
                                // F3: finalize already ran for this finish_reason —
                                // clear the pending marker so the post-loop block
                                // below (also keyed on pending_finish_reason) does
                                // NOT emit a second message_delta/message_stop.
                                pending_finish_reason = None;
                            }
                            done = true;
                            break;
                        }

                        // Parse JSON
                        let chunk_value: Value = match serde_json::from_str(data) {
                            Ok(v) => v,
                            Err(e) => {
                                tracing::warn!("Failed to parse SSE chunk: {} — data: {}", e, data);
                                continue;
                            }
                        };

                        // Extract usage
                        let usage: Option<crate::openai::types::Usage> = chunk_value
                            .get("usage")
                            .and_then(|u| serde_json::from_value(u.clone()).ok());
                        if let Some(ref u) = usage {
                            last_usage = Some(u.clone());
                        }

                        // Extract choices
                        if let Some(choices) = chunk_value.get("choices").and_then(|v| v.as_array())
                        {
                            for choice in choices {
                                let finish_reason = choice
                                    .get("finish_reason")
                                    .and_then(|v| v.as_str())
                                    .filter(|s| !s.is_empty());

                                let delta = choice.get("delta");

                                if let Some(delta) = delta {
                                    let output_tokens =
                                        usage.as_ref().and_then(|u| u.completion_tokens);

                                    // Tolerant decode: known OpenAI-compatible content
                                    // shapes (string, content-part array, object wrapper,
                                    // null) never fail. Anything that cannot be mapped to
                                    // text is retained in `raw` and surfaced here — never
                                    // silently dropped (root cause report 84: `from_value
                                    // .ok()` -> None -> process_delta never runs -> 0 content
                                    // blocks while the terminal still succeeds).
                                    let chat_delta: Option<crate::openai::types::ChatDelta> =
                                        match serde_json::from_value(delta.clone()) {
                                            Ok(cd) => Some(cd),
                                            Err(e) => {
                                                // Fail-closed: a delta we cannot decode at
                                                // all becomes an observable error frame and
                                                // the stream terminates on the error instead
                                                // of silently dropping the chunk.
                                                tracing::error!(
                                                    serde_error = %e,
                                                    raw = %delta,
                                                    "undecodable chat streaming delta"
                                                );
                                                let error_event =
                                                    sse_event_to_axum(&SseEvent::Error {
                                                        error: crate::anthropic::types::ErrorData {
                                                            error_type: "stream_error".to_string(),
                                                            message: format!(
                                                                "undecodable streaming delta: {e}"
                                                            ),
                                                        },
                                                    });
                                                if tx.send(error_event).await.is_err() {
                                                    return;
                                                }
                                                return;
                                            }
                                        };

                                    if let Some(ref cd) = chat_delta {
                                        // Surface content the tolerant decoder could not map
                                        // to text instead of losing it silently.
                                        if cd.has_unparseable_content() {
                                            let preview = cd.unparseable_preview();
                                            if cd.has_no_text() {
                                                // Total loss for this chunk: fail closed with
                                                // an observable error frame + error terminal,
                                                // never a clean message_stop over dropped
                                                // content (the Phase5B stream bug).
                                                tracing::error!(
                                                    raw_preview = %preview,
                                                    "chat delta carried only unparseable content — aborting stream"
                                                );
                                                let error_event = sse_event_to_axum(
                                                    &SseEvent::Error {
                                                        error: crate::anthropic::types::ErrorData {
                                                            error_type: "stream_error".to_string(),
                                                            message: format!(
                                                                "streaming delta content could not be decoded: {preview}"
                                                            ),
                                                        },
                                                    },
                                                );
                                                if tx.send(error_event).await.is_err() {
                                                    return;
                                                }
                                                return;
                                            }
                                            // Partial loss: keep the decodable text, but make
                                            // the loss observable.
                                            tracing::warn!(
                                                raw_preview = %preview,
                                                "chat delta partially unparseable — text delivered, some content dropped"
                                            );
                                        }

                                        let events =
                                            state_machine.process_delta(cd, usage.as_ref());
                                        for event in &events {
                                            if tx.send(sse_event_to_axum(event)).await.is_err() {
                                                return; // Client disconnected
                                            }
                                        }

                                        // Handle finish_reason (dsv4-cc-proxy pattern: idempotent)
                                        if let Some(fr) = finish_reason {
                                            if completed {
                                                continue;
                                            }
                                            completed = true;
                                            pending_finish_reason = Some(fr.to_string());
                                            pending_output_tokens = output_tokens;
                                        }
                                    } else if finish_reason.is_some() {
                                        // Delta was None but finish_reason is set
                                        if completed {
                                            continue;
                                        }
                                        completed = true;
                                        pending_finish_reason =
                                            finish_reason.map(|s| s.to_string());
                                        pending_output_tokens = output_tokens;
                                    }
                                } else if let Some(fr) = finish_reason {
                                    if completed {
                                        continue;
                                    }
                                    completed = true;
                                    pending_finish_reason = Some(fr.to_string());
                                    pending_output_tokens =
                                        usage.as_ref().and_then(|u| u.completion_tokens);
                                }
                            }
                        }

                        if done || (completed && last_usage.is_some()) {
                            if pending_finish_reason.is_some() {
                                let output_tokens = last_usage
                                    .as_ref()
                                    .and_then(|u| u.completion_tokens)
                                    .or(pending_output_tokens);
                                let final_events = state_machine.finalize(
                                    pending_finish_reason.as_deref(),
                                    output_tokens,
                                    last_usage.as_ref(),
                                );
                                for event in &final_events {
                                    let _ = tx.send(sse_event_to_axum(event)).await;
                                }
                                pending_finish_reason = None;
                            }
                            break;
                        }
                    }

                    if done || (completed && last_usage.is_some()) {
                        if pending_finish_reason.is_some() {
                            let output_tokens = last_usage
                                .as_ref()
                                .and_then(|u| u.completion_tokens)
                                .or(pending_output_tokens);
                            let final_events = state_machine.finalize(
                                pending_finish_reason.as_deref(),
                                output_tokens,
                                last_usage.as_ref(),
                            );
                            for event in &final_events {
                                let _ = tx.send(sse_event_to_axum(event)).await;
                            }
                        }
                        break;
                    }
                }
                Err(e) => {
                    let error_event = sse_event_to_axum(&SseEvent::Error {
                        error: crate::anthropic::types::ErrorData {
                            error_type: "stream_error".to_string(),
                            message: format!("{}", e),
                        },
                    });
                    let _ = tx.send(error_event).await;
                    return;
                }
            }
        }

        // Drain remaining chunks after [DONE] to prevent stale events in cc-connect
        // (10s timeout protection against upstream hanging)
        if timeout(tokio::time::Duration::from_secs(10), async {
            use futures_util::StreamExt;
            while stream.next().await.is_some() {}
        })
        .await
        .is_err()
        {
            tracing::warn!("Drain timeout after 10s, dropping stream");
        }

        // Handle [DONE] without finish_reason (the empty response bug)
        // When [DONE] is received, done=true but finalize() was never called.
        if done && !completed {
            tracing::warn!("[DONE] received without finish_reason — finalizing stream");
            let final_events = state_machine.finalize(None, None, last_usage.as_ref());
            for event in &final_events {
                let _ = tx.send(sse_event_to_axum(event)).await;
            }
        }

        // If stream ended naturally without finish_reason or [DONE], finalize
        if !done && !completed {
            tracing::warn!("Stream ended without finish_reason or [DONE] — sending empty finalize");
            let final_events = state_machine.finalize(None, None, last_usage.as_ref());
            for event in &final_events {
                let _ = tx.send(sse_event_to_axum(event)).await;
            }
        }

        // Log KV cache statistics if available. A single policy-gated view
        // computes the buckets: Legacy (default — policy None/off) reproduces
        // the historical numbers exactly (hit = ptd.cached_tokens, clamped
        // miss, formatted hit rate); Raw (explicit usage source) reads the
        // canonical read (top-level → nested → DeepSeek hit) with a guarded
        // miss. Field names and formatting are unchanged.
        if let Some(ref u) = last_usage {
            let view = crate::cache::chat_usage_view(Some(u), cache_policy.as_ref());
            tracing::info!(
                cache_hit = view.read.unwrap_or(0),
                cache_miss = view.miss.unwrap_or(0),
                prompt_tokens = view.input.unwrap_or(0),
                hit_rate = format!("{:.1}%", view.hit_rate.unwrap_or(0.0)),
                "KV cache stats"
            );
        }

        // Explicitly drop the sender to close the SSE channel
        drop(tx);
    });

    Sse::new(SseEventStream { receiver: rx })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;

    /// Run `process_stream` over a synthetic SSE body and collect the serialized
    /// wire frames as `(event_name, data_json)` pairs.
    async fn run_stream(
        is_reasoning_model: bool,
        reasoning_field: &str,
        body: &str,
    ) -> Vec<(String, serde_json::Value)> {
        let bytes = bytes::Bytes::from(body.to_string());
        let stream = futures_util::stream::iter(vec![Ok::<_, reqwest::Error>(bytes)]);
        let sse = process_stream(
            "kimi-k3".to_string(),
            is_reasoning_model,
            reasoning_field.to_string(),
            vec![],
            "msg_test".to_string(),
            None,
            stream,
        );
        let response = sse.into_response();
        let body_bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
            .await
            .expect("read response body");
        parse_sse(&String::from_utf8_lossy(&body_bytes))
    }

    /// Parse an SSE payload into `(event, data)` pairs. Each frame is
    /// `event: <name>\ndata: <json>\n\n`.
    fn parse_sse(text: &str) -> Vec<(String, serde_json::Value)> {
        let mut frames = Vec::new();
        for block in text.split("\n\n") {
            let block = block.trim();
            if block.is_empty() {
                continue;
            }
            let mut event = String::new();
            let mut data = String::new();
            for line in block.lines() {
                if let Some(v) = line.strip_prefix("event: ") {
                    event = v.to_string();
                } else if let Some(v) = line.strip_prefix("data: ") {
                    data.push_str(v);
                }
            }
            if !data.is_empty() {
                let json: serde_json::Value = serde_json::from_str(&data)
                    .unwrap_or_else(|e| panic!("bad SSE data frame: {e}: {data}"));
                frames.push((event, json));
            }
        }
        frames
    }

    /// Concatenate all `content_block_delta` text deltas into one string.
    fn concat_text_deltas(frames: &[(String, serde_json::Value)]) -> String {
        let mut s = String::new();
        for (event, json) in frames {
            if event == "content_block_delta" {
                if let Some(text) = json.pointer("/delta/text") {
                    if let Some(text) = text.as_str() {
                        s.push_str(text);
                    }
                }
            }
        }
        s
    }

    fn frame_types(frames: &[(String, serde_json::Value)]) -> Vec<&str> {
        frames
            .iter()
            .map(|(_, j)| j.get("type").and_then(|v| v.as_str()).unwrap_or(""))
            .collect()
    }

    #[tokio::test]
    async fn stream_preserves_array_content_deltas() {
        // RED (pre-fix): array-valued `content` deltas fail the `ChatDelta`
        // gate, so the stream emits NO content_block_* frames — only
        // message_start / message_delta / message_stop (Phase5B symptom).
        let body = concat!(
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"type\":\"text\",\"text\":\"Hello\"}]}}]}\n\n",
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"type\":\"text\",\"text\":\" world\"}]}}]}\n\n",
            "data: {\"id\":\"c1\",\"usage\":{\"prompt_tokens\":120,\"completion_tokens\":512,\"total_tokens\":632,\"prompt_tokens_details\":{\"cached_tokens\":0}},\"choices\":[]}\n\n",
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":120,\"completion_tokens\":512,\"total_tokens\":632,\"prompt_tokens_details\":{\"cached_tokens\":0}}}\n\n",
            "data: [DONE]\n\n",
        );
        let frames = run_stream(false, "reasoning", body).await;

        assert!(
            frames.iter().any(|(e, _)| e == "content_block_start"),
            "content_block_start missing (array content dropped at gate): {frames:?}"
        );
        assert!(
            frames.iter().any(|(e, _)| e == "content_block_delta"),
            "content_block_delta missing (array content dropped at gate): {frames:?}"
        );
        assert_eq!(concat_text_deltas(&frames), "Hello world");
        // Complete Anthropic terminal: message_delta (usage) then message_stop.
        assert!(
            frames.iter().any(|(e, _)| e == "message_delta"),
            "message_delta missing: {frames:?}"
        );
        assert!(
            frames.iter().any(|(e, _)| e == "message_stop"),
            "message_stop missing: {frames:?}"
        );
        // message_start always carries a usage object.
        let ms = frames
            .iter()
            .find(|(e, _)| e == "message_start")
            .expect("message_start present");
        assert!(
            ms.1.get("message").and_then(|m| m.get("usage")).is_some(),
            "message_start.usage object present: {frames:?}"
        );
        // No error frame on a healthy stream.
        assert!(
            !frames.iter().any(|(e, _)| e == "error"),
            "no error expected on healthy stream: {frames:?}"
        );
        // Terminal ordering: message_delta before message_stop.
        let types = frame_types(&frames);
        let md = types.iter().position(|t| *t == "message_delta");
        let ms_pos = types.iter().position(|t| *t == "message_stop");
        assert!(
            md.is_some() && ms_pos.is_some() && md < ms_pos,
            "terminal order: {types:?}"
        );
    }

    #[tokio::test]
    async fn stream_string_content_delta_still_works() {
        // Control: the standard OpenAI string form must keep working unchanged.
        let body = concat!(
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}\n\n",
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let frames = run_stream(false, "reasoning", body).await;
        assert_eq!(concat_text_deltas(&frames), "Hello");
        assert!(
            frames.iter().any(|(e, _)| e == "content_block_stop"),
            "content_block_stop missing: {frames:?}"
        );
        // F3 regression: this is the no-usage body (finish_reason then [DONE],
        // no usage object). The terminal must be emitted EXACTLY once — a
        // duplicated message_delta/message_stop is a broken SSE stream.
        let message_deltas = frames.iter().filter(|(e, _)| e == "message_delta").count();
        let message_stops = frames.iter().filter(|(e, _)| e == "message_stop").count();
        assert_eq!(
            message_deltas, 1,
            "no-usage stream must emit exactly ONE message_delta, got {message_deltas}: {frames:?}"
        );
        assert_eq!(
            message_stops, 1,
            "no-usage stream must emit exactly ONE message_stop, got {message_stops}: {frames:?}"
        );
        assert!(
            !frames.iter().any(|(e, _)| e == "error"),
            "no error expected on a healthy no-usage stream: {frames:?}"
        );
    }

    #[tokio::test]
    async fn stream_malformed_delta_emits_error_terminal_not_silent_success() {
        // A delta that cannot be decoded at all (here: `tool_calls` with a
        // non-array value) must fail CLOSED: an observable `error` frame and an
        // aborted stream — never a clean `message_stop` over dropped content.
        let body = concat!(
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}\n\n",
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":\"not-an-array\"}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let frames = run_stream(false, "reasoning", body).await;
        assert!(
            frames.iter().any(|(e, _)| e == "error"),
            "undecodable delta must produce an error frame: {frames:?}"
        );
        assert!(
            !frames.iter().any(|(e, _)| e == "message_stop"),
            "malformed delta must NOT terminate with a clean message_stop: {frames:?}"
        );
    }

    #[tokio::test]
    async fn stream_unknown_content_shape_aborts_not_silent_success() {
        // A delta whose content decodes but carries no extractable text (unknown
        // shape) must not yield a successful empty response: error frame +
        // abort, never a clean `message_stop` over dropped content.
        let body = concat!(
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":12345}}]}\n\n",
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let frames = run_stream(false, "reasoning", body).await;
        assert!(
            frames.iter().any(|(e, _)| e == "error"),
            "unknown content shape must produce an error frame: {frames:?}"
        );
        assert!(
            !frames.iter().any(|(e, _)| e == "message_stop"),
            "unknown content shape must NOT yield a clean message_stop over dropped content: {frames:?}"
        );
    }

    #[tokio::test]
    async fn stream_multibyte_unparseable_content_aborts_not_panic() {
        // F2 regression at stream level: an unknown content shape containing
        // multi-byte UTF-8 previously panicked inside `unparseable_preview()`
        // on the fail-closed path (`&joined[..500]` landed mid-character),
        // which dropped the SSE channel → silent EOF with no error frame and no
        // terminal. It must instead produce an observable error frame and abort
        // — never a clean `message_stop` over dropped content, and never a
        // panic. 300 × 3-byte CJK chars = 900 bytes, so byte 500 is mid-char.
        let long = "中".repeat(300);
        let body = format!(
            "data: {{\"id\":\"c1\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":{{\"weird\":\"{long}\"}}}}}}]}}\n\ndata: [DONE]\n\n"
        );
        let frames = run_stream(false, "reasoning", &body).await;
        assert!(
            frames.iter().any(|(e, _)| e == "error"),
            "multi-byte unknown content must produce an observable error frame (not a panic/EOF): {frames:?}"
        );
        assert!(
            !frames.iter().any(|(e, _)| e == "message_stop"),
            "multi-byte unknown content must NOT yield a clean message_stop: {frames:?}"
        );
    }
}
