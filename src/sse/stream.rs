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
        SseEvent::ContentBlockStart { .. } => Event::default().event("content_block_start").data(json),
        SseEvent::ContentBlockDelta { .. } => Event::default().event("content_block_delta").data(json),
        SseEvent::ContentBlockStop { .. } => Event::default().event("content_block_stop").data(json),
        SseEvent::MessageDelta { .. } => Event::default().event("message_delta").data(json),
        SseEvent::MessageStop => Event::default().event("message_stop").data(json),
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
    msg_id: String,
    mut body_stream: impl Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + Unpin + 'static,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<Event>(256);

    const MAX_SSE_BUF: usize = 4 * 1024 * 1024; // 4MB
    let idle_timeout = tokio::time::Duration::from_secs(300);

    tokio::spawn(async move {
        let mut state_machine = SseStateMachine::new(is_reasoning_model);

        // Send message_start first (audit defect 3.1)
        let msg_start = state_machine.message_start(&model, &msg_id);
        let _ = tx.send(sse_event_to_axum(&msg_start)).await;

        use futures_util::StreamExt;
        let mut stream = Box::pin(body_stream);

        let mut buffer = String::new();
        let mut done = false;
        let mut last_usage: Option<crate::openai::types::Usage> = None;

        loop {
            let chunk_result = match timeout(idle_timeout, stream.next()).await {
                Ok(Some(Ok(chunk))) => Ok(chunk),
                Ok(Some(Err(e))) => Err(e),
                Ok(None) => break, // stream ended
                Err(_elapsed) => {
                    tracing::warn!("SSE stream idle timeout after 300s");
                    break;
                }
            };

            match chunk_result {
                Ok(chunk) => {
                    let chunk_str = String::from_utf8_lossy(&chunk);
                    buffer.push_str(&chunk_str);

                    // P0-2: buffer size guard
                    if buffer.len() > MAX_SSE_BUF {
                        tracing::error!("SSE buffer exceeded {} bytes, aborting stream", MAX_SSE_BUF);
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
                        let usage: Option<crate::openai::types::Usage> =
                            chunk_value.get("usage").and_then(|u| {
                                serde_json::from_value(u.clone()).ok()
                            });
                        if let Some(ref u) = usage {
                            last_usage = Some(u.clone());
                        }

                        // Extract choices
                        if let Some(choices) = chunk_value.get("choices").and_then(|v| v.as_array()) {
                            for choice in choices {
                                let finish_reason = choice
                                    .get("finish_reason")
                                    .and_then(|v| v.as_str())
                                    .filter(|s| !s.is_empty());

                                let delta = choice.get("delta");

                                if let Some(delta) = delta {
                                    let chat_delta: Option<crate::openai::types::ChatDelta> =
                                        serde_json::from_value(delta.clone()).ok();

                                    let output_tokens = usage
                                        .as_ref()
                                        .and_then(|u| u.completion_tokens);

                                    if let Some(ref cd) = chat_delta {
                                        let events = state_machine.process_delta(cd, usage.as_ref());
                                        for event in &events {
                                            if tx.send(sse_event_to_axum(event)).await.is_err() {
                                                return; // Client disconnected
                                            }
                                        }

                                        // Handle finish_reason
                                        if let Some(fr) = finish_reason {
                                            let final_events = state_machine.finalize(Some(fr), output_tokens);
                                            for event in &final_events {
                                                if tx.send(sse_event_to_axum(event)).await.is_err() {
                                                    return;
                                                }
                                            }
                                            done = true;
                                            break;
                                        }
                                    } else if finish_reason.is_some() {
                                        // Delta was None but finish_reason is set
                                        let final_events = state_machine.finalize(finish_reason, output_tokens);
                                        for event in &final_events {
                                            if tx.send(sse_event_to_axum(event)).await.is_err() {
                                                return;
                                            }
                                        }
                                        done = true;
                                        break;
                                    }
                                } else if let Some(fr) = finish_reason {
                                    let output_tokens = usage
                                        .as_ref()
                                        .and_then(|u| u.completion_tokens);
                                    let final_events = state_machine.finalize(Some(fr), output_tokens);
                                    for event in &final_events {
                                        if tx.send(sse_event_to_axum(event)).await.is_err() {
                                            return;
                                        }
                                    }
                                    done = true;
                                    break;
                                }
                            }
                        }

                        if done {
                            break;
                        }
                    }

                    if done {
                        break;
                    }
                }
                Err(e) => {
                    tracing::error!("Stream error: {}", e);
                    break;
                }
            }
        }

        // If stream ended without finish_reason, finalize
        if !done {
            let final_events = state_machine.finalize(None, None);
            for event in &final_events {
                let _ = tx.send(sse_event_to_axum(event)).await;
            }
        }

        // Log KV cache statistics if available
        if let Some(ref u) = last_usage {
            let hit = u.prompt_cache_hit_tokens.unwrap_or(0);
            let miss = u.prompt_cache_miss_tokens.unwrap_or(0);
            let total = hit + miss;
            if total > 0 {
                let rate = (hit as f64 / total as f64) * 100.0;
                tracing::info!(
                    cache_hit = hit,
                    cache_miss = miss,
                    hit_rate = format!("{:.1}%", rate),
                    "KV cache stats"
                );
            }
        }
    });

    Sse::new(SseEventStream { receiver: rx })
}