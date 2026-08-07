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
        let mut state_machine =
            SseStateMachine::new(is_reasoning_model, reasoning_field, reasoning_field_alt);

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
                                    let chat_delta: Option<crate::openai::types::ChatDelta> =
                                        serde_json::from_value(delta.clone()).ok();

                                    let output_tokens =
                                        usage.as_ref().and_then(|u| u.completion_tokens);

                                    if let Some(ref cd) = chat_delta {
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

        // Log KV cache statistics if available
        if let Some(ref u) = last_usage {
            let hit = u
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens)
                .unwrap_or(0);
            let prompt_total = u.prompt_tokens.unwrap_or(0);
            let miss = prompt_total.saturating_sub(hit);
            let rate = if prompt_total > 0 {
                (hit as f64 / prompt_total as f64) * 100.0
            } else {
                0.0
            };
            tracing::info!(
                cache_hit = hit,
                cache_miss = miss,
                prompt_tokens = prompt_total,
                hit_rate = format!("{:.1}%", rate),
                "KV cache stats"
            );
        }

        // Explicitly drop the sender to close the SSE channel
        drop(tx);
    });

    Sse::new(SseEventStream { receiver: rx })
}
