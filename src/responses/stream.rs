use crate::anthropic::types::{
    ContentBlockDeltaData, ContentBlockStartData, ErrorData, MessageDeltaData, MessageStartData,
    SseEvent, StreamUsage,
};
use crate::cache::{responses_usage_view_from_buckets, CachePolicy};
use axum::response::sse::{Event, Sse};
use futures_util::Stream;
use std::{
    collections::{HashMap, HashSet},
    convert::Infallible,
    pin::Pin,
    task::{Context, Poll},
};
use tokio::sync::mpsc;

fn log_cache_stats(
    model: &str,
    usage: &StreamUsage,
    status: &str,
    cache_policy: Option<&CachePolicy>,
) {
    // Single policy-gated selector over the already-extracted buckets: Legacy
    // (default — policy None/off) reproduces the historical numbers exactly;
    // Raw (explicit usage source) reports the canonical `CacheStats` buckets.
    // The wire (`MessageDelta.usage`) is built from the same read/creation
    // buckets, so log and wire can never disagree.
    let view = responses_usage_view_from_buckets(
        usage.input_tokens,
        usage.cache_read_input_tokens,
        usage.cache_creation_input_tokens,
        cache_policy,
    );
    tracing::info!(
        model = %model,
        status,
        upstream_http_status = 200u16,
        input_tokens = ?view.input,
        output_tokens = ?usage.output_tokens,
        cache_read_input_tokens = ?view.read,
        cache_creation_input_tokens = ?view.creation,
        cache_miss_input_tokens = ?view.miss,
        hit_rate_percent = ?view.hit_rate,
        "Responses cache stats"
    );
}

fn terminal_usage(value: &serde_json::Value) -> Option<StreamUsage> {
    let usage = value
        .get("response")
        .and_then(|response| response.get("usage"))
        .or_else(|| value.get("usage"))?;
    Some(StreamUsage {
        input_tokens: usage
            .get("input_tokens")
            .and_then(|value| value.as_u64())
            .map(|value| value as u32),
        output_tokens: usage
            .get("output_tokens")
            .and_then(|value| value.as_u64())
            .map(|value| value as u32),
        cache_read_input_tokens: usage
            .get("input_tokens_details")
            .and_then(|details| details.get("cached_tokens"))
            .and_then(|value| value.as_u64())
            .map(|value| value as u32),
        cache_creation_input_tokens: usage
            .get("input_tokens_details")
            .and_then(|details| details.get("cache_write_tokens"))
            .or_else(|| usage.get("cache_write_tokens"))
            .and_then(|value| value.as_u64())
            .map(|value| value as u32),
    })
}

struct EventStream(mpsc::Receiver<Event>);

impl Stream for EventStream {
    type Item = Result<Event, Infallible>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.0.poll_recv(cx).map(|event| event.map(Ok))
    }
}

fn axum_event(event: &SseEvent) -> Event {
    let name = match event {
        SseEvent::MessageStart { .. } => "message_start",
        SseEvent::ContentBlockStart { .. } => "content_block_start",
        SseEvent::ContentBlockDelta { .. } => "content_block_delta",
        SseEvent::ContentBlockStop { .. } => "content_block_stop",
        SseEvent::MessageDelta { .. } => "message_delta",
        SseEvent::MessageStop => "message_stop",
        SseEvent::Error { .. } => "error",
    };
    Event::default()
        .event(name)
        .data(serde_json::to_string(event).unwrap_or_default())
}

struct State {
    text_started: bool,
    reasoning_started: bool,
    active_reasoning: Option<String>,
    tool_indices: HashMap<String, u32>,
    tool_arguments: HashMap<u32, String>,
    closed_tools: HashSet<u32>,
    reasoning_accumulators: HashMap<String, String>,
    pending_tool_arguments: HashMap<String, String>,
    tool_order: Vec<u32>,
    next_index: u32,
    tool_stop_reason: bool,
    terminal: bool,
}

impl State {
    fn new() -> Self {
        Self {
            text_started: false,
            reasoning_started: false,
            active_reasoning: None,
            tool_indices: HashMap::new(),
            tool_arguments: HashMap::new(),
            closed_tools: HashSet::new(),
            reasoning_accumulators: HashMap::new(),
            pending_tool_arguments: HashMap::new(),
            tool_order: Vec::new(),
            next_index: 0,
            tool_stop_reason: false,
            terminal: false,
        }
    }

    fn start_text(&mut self, out: &mut Vec<SseEvent>) {
        if !self.text_started {
            self.close_reasoning(out);
            self.text_started = true;
            out.push(SseEvent::ContentBlockStart {
                index: self.next_index,
                content_block: ContentBlockStartData::Text {
                    text: String::new(),
                },
            });
            self.next_index += 1;
        }
    }

    fn close_reasoning(&mut self, out: &mut Vec<SseEvent>) {
        if self.reasoning_started {
            out.push(SseEvent::ContentBlockStop {
                index: self.next_index - 1,
            });
            self.reasoning_started = false;
            self.active_reasoning = None;
        }
    }

    fn close_text(&mut self, out: &mut Vec<SseEvent>) {
        if self.text_started {
            out.push(SseEvent::ContentBlockStop {
                index: self.next_index - 1,
            });
            self.text_started = false;
        }
    }

    fn finish(&mut self) -> Vec<SseEvent> {
        if self.terminal {
            return Vec::new();
        }
        self.terminal = true;
        let mut out = Vec::new();
        if self.text_started {
            out.push(SseEvent::ContentBlockStop {
                index: self.next_index - 1,
            });
            self.text_started = false;
        }
        self.close_reasoning(&mut out);
        for index in &self.tool_order {
            if self.closed_tools.insert(*index) {
                out.push(SseEvent::ContentBlockStop { index: *index });
            }
        }
        out
    }

    fn reasoning_delta(&mut self, key: &str, text: &str, out: &mut Vec<SseEvent>) {
        if text.is_empty() {
            return;
        }
        self.reasoning_accumulators
            .entry(key.to_string())
            .or_default()
            .push_str(text);
        out.push(SseEvent::ContentBlockDelta {
            index: self.next_index - 1,
            delta: ContentBlockDeltaData::ThinkingDelta {
                thinking: text.to_string(),
            },
        });
    }

    fn reasoning_done(&mut self, key: &str, text: &str, out: &mut Vec<SseEvent>) {
        let already = self
            .reasoning_accumulators
            .get(key)
            .map(String::as_str)
            .unwrap_or_default();
        let suffix = text.strip_prefix(already).unwrap_or("");
        self.reasoning_delta(key, suffix, out);
    }

    fn event(&mut self, value: &serde_json::Value) -> Vec<SseEvent> {
        let mut out = Vec::new();
        if self.terminal {
            return out;
        }
        let kind = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match kind {
            "response.output_text.delta" => {
                let text = value
                    .get("delta")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                self.start_text(&mut out);
                if !text.is_empty() {
                    out.push(SseEvent::ContentBlockDelta {
                        index: self.next_index - 1,
                        delta: ContentBlockDeltaData::TextDelta {
                            text: text.to_string(),
                        },
                    });
                }
            }
            "response.reasoning_summary_text.delta" => {
                let key = format!(
                    "{}:{}",
                    value
                        .get("item_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("default"),
                    value
                        .get("summary_index")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0)
                );
                if self.active_reasoning.as_deref() != Some(key.as_str()) {
                    self.close_reasoning(&mut out);
                    self.close_text(&mut out);
                    self.reasoning_started = true;
                    self.active_reasoning = Some(key.clone());
                    out.push(SseEvent::ContentBlockStart {
                        index: self.next_index,
                        content_block: ContentBlockStartData::Thinking {
                            thinking: String::new(),
                            signature: String::new(),
                        },
                    });
                    self.next_index += 1;
                }
                if let Some(text) = value.get("delta").and_then(|v| v.as_str()) {
                    self.reasoning_delta(&key, text, &mut out);
                }
            }
            "response.reasoning_summary_text.done" => {
                let key = format!(
                    "{}:{}",
                    value
                        .get("item_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("default"),
                    value
                        .get("summary_index")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0)
                );
                if self.active_reasoning.as_deref() != Some(key.as_str()) {
                    self.close_reasoning(&mut out);
                    self.close_text(&mut out);
                    self.reasoning_started = true;
                    self.active_reasoning = Some(key.clone());
                    out.push(SseEvent::ContentBlockStart {
                        index: self.next_index,
                        content_block: ContentBlockStartData::Thinking {
                            thinking: String::new(),
                            signature: String::new(),
                        },
                    });
                    self.next_index += 1;
                }
                if let Some(text) = value.get("text").and_then(|v| v.as_str()) {
                    self.reasoning_done(&key, text, &mut out);
                }
            }
            "response.reasoning_text.delta" => {
                let key = format!(
                    "raw:{}:{}",
                    value
                        .get("item_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("default"),
                    value
                        .get("content_index")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0)
                );
                if self.active_reasoning.as_deref() != Some(key.as_str()) {
                    self.close_reasoning(&mut out);
                    self.close_text(&mut out);
                    self.reasoning_started = true;
                    self.active_reasoning = Some(key.clone());
                    out.push(SseEvent::ContentBlockStart {
                        index: self.next_index,
                        content_block: ContentBlockStartData::Thinking {
                            thinking: String::new(),
                            signature: String::new(),
                        },
                    });
                    self.next_index += 1;
                }
                if let Some(text) = value.get("delta").and_then(|v| v.as_str()) {
                    self.reasoning_delta(&key, text, &mut out);
                }
            }
            "response.reasoning_text.done" => {
                let key = format!(
                    "raw:{}:{}",
                    value
                        .get("item_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("default"),
                    value
                        .get("content_index")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0)
                );
                if let Some(text) = value.get("text").and_then(|v| v.as_str()) {
                    self.reasoning_done(&key, text, &mut out);
                }
            }
            "response.output_item.added" => {
                let item = value.get("item").unwrap_or(value);
                if item.get("type").and_then(|v| v.as_str()) != Some("function_call") {
                    return out;
                }
                let item_id = item
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let call_id = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let existing_index = self
                    .tool_indices
                    .get(&item_id)
                    .copied()
                    .or_else(|| self.tool_indices.get(&call_id).copied());
                if let Some(index) = existing_index {
                    if !call_id.is_empty() {
                        self.tool_indices.insert(call_id, index);
                    }
                    if !item_id.is_empty() {
                        self.tool_indices.insert(item_id, index);
                    }
                    return out;
                }
                let id = if !call_id.is_empty() {
                    call_id.clone()
                } else {
                    item_id.clone()
                };
                let index = self.next_index;
                self.next_index += 1;
                if !call_id.is_empty() {
                    self.tool_indices.insert(call_id, index);
                }
                if !item_id.is_empty() {
                    self.tool_indices.insert(item_id.clone(), index);
                }
                self.tool_order.push(index);
                let pending = self.pending_tool_arguments.remove(&item_id);
                self.tool_arguments
                    .insert(index, pending.clone().unwrap_or_default());
                out.push(SseEvent::ContentBlockStart {
                    index,
                    content_block: ContentBlockStartData::ToolUse {
                        id,
                        name,
                        input: serde_json::json!({}),
                    },
                });
                if let Some(arguments) = pending.filter(|value| !value.is_empty()) {
                    out.push(SseEvent::ContentBlockDelta {
                        index,
                        delta: ContentBlockDeltaData::InputJsonDelta {
                            partial_json: arguments,
                        },
                    });
                }
            }
            "response.function_call_arguments.delta" => {
                let id = value
                    .get("item_id")
                    .or_else(|| value.get("call_id"))
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let key = if id.is_empty() {
                    value
                        .get("output_index")
                        .or_else(|| value.get("index"))
                        .map(|v| format!("index:{v}"))
                        .unwrap_or_else(|| "index:unknown".to_string())
                } else {
                    id.to_string()
                };
                let Some(index) = self.tool_indices.get(&key).copied() else {
                    self.pending_tool_arguments
                        .entry(key)
                        .or_default()
                        .push_str(
                            value
                                .get("delta")
                                .and_then(|v| v.as_str())
                                .unwrap_or_default(),
                        );
                    return out;
                };
                if self.closed_tools.contains(&index) {
                    return out;
                }
                if let Some(delta) = value.get("delta").and_then(|v| v.as_str()) {
                    self.tool_arguments
                        .entry(index)
                        .or_default()
                        .push_str(delta);
                    out.push(SseEvent::ContentBlockDelta {
                        index,
                        delta: ContentBlockDeltaData::InputJsonDelta {
                            partial_json: delta.to_string(),
                        },
                    });
                }
            }
            "response.function_call_arguments.done" => {
                self.tool_stop_reason = true;
                let id = value
                    .get("item_id")
                    .or_else(|| value.get("call_id"))
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let key = id.to_string();
                if let Some(index) = self.tool_indices.get(&key).copied() {
                    if self.closed_tools.contains(&index) {
                        return out;
                    }
                    let already = self.tool_arguments.entry(index).or_default();
                    if let Some(arguments) = value.get("arguments").and_then(|v| v.as_str()) {
                        let suffix = arguments.strip_prefix(already.as_str()).unwrap_or("");
                        if !suffix.is_empty() {
                            out.push(SseEvent::ContentBlockDelta {
                                index,
                                delta: ContentBlockDeltaData::InputJsonDelta {
                                    partial_json: suffix.to_string(),
                                },
                            });
                        }
                        if arguments.starts_with(already.as_str()) {
                            *already = arguments.to_string();
                        }
                    }
                } else if let Some(arguments) = value.get("arguments").and_then(|v| v.as_str()) {
                    self.pending_tool_arguments
                        .insert(key, arguments.to_string());
                }
            }
            "response.output_item.done"
                if value
                    .get("item")
                    .and_then(|v| v.get("type"))
                    .and_then(|v| v.as_str())
                    == Some("function_call") =>
            {
                self.tool_stop_reason = true;
                let item = value.get("item").unwrap_or(value);
                let key = item
                    .get("call_id")
                    .or_else(|| item.get("id"))
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                if let Some(index) = self.tool_indices.get(key).copied() {
                    if self.closed_tools.insert(index) {
                        out.push(SseEvent::ContentBlockStop { index });
                    }
                }
            }
            "response.failed" | "response.error" | "error" => {
                let mut close_events = self.finish();
                out.append(&mut close_events);
                let message = value
                    .get("response")
                    .and_then(|v| v.get("error"))
                    .and_then(|v| v.get("message"))
                    .and_then(|v| v.as_str())
                    .or_else(|| {
                        value
                            .get("error")
                            .and_then(|v| v.get("message"))
                            .and_then(|v| v.as_str())
                    })
                    .or_else(|| value.get("error").and_then(|v| v.as_str()))
                    .or_else(|| value.get("message").and_then(|v| v.as_str()))
                    .unwrap_or("Responses stream failed");
                out.push(SseEvent::Error {
                    error: ErrorData {
                        error_type: "api_error".to_string(),
                        message: message.to_string(),
                    },
                });
            }
            "response.cancelled" => {
                let mut close_events = self.finish();
                out.append(&mut close_events);
                out.push(SseEvent::Error {
                    error: ErrorData {
                        error_type: "api_error".to_string(),
                        message: "Responses request cancelled".to_string(),
                    },
                });
            }
            _ => {}
        }
        out
    }
}

pub fn process_stream(
    model: String,
    msg_id: String,
    request_id: String,
    body_stream: impl Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + Unpin + 'static,
    cache_policy: Option<CachePolicy>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel(256);
    let stats_model = model.clone();
    let telemetry_request_id = request_id.clone();
    tokio::spawn(async move {
        use futures_util::StreamExt;
        let mut state = State::new();
        let start = SseEvent::MessageStart {
            message: MessageStartData {
                id: msg_id,
                msg_type: "message".to_string(),
                role: "assistant".to_string(),
                content: Vec::new(),
                model,
                // usage 必须为对象而非 null：Claude Code 的 Agent 工具会读取
                // message_start.message.usage.input_tokens，null 会直接崩溃
                //（"null is not an object (evaluating 'o.input_tokens')"）。
                usage: Some(StreamUsage {
                    input_tokens: Some(0),
                    output_tokens: Some(0),
                    cache_read_input_tokens: Some(0),
                    cache_creation_input_tokens: Some(0),
                }),
            },
        };
        if tx.send(axum_event(&start)).await.is_err() {
            return;
        }
        let mut stream = body_stream;
        let mut buffer = String::new();
        let mut utf8_pending = Vec::new();
        while let Some(result) = stream.next().await {
            let bytes = match result {
                Ok(bytes) => bytes,
                Err(error) => {
                    let _ = tx
                        .send(axum_event(&SseEvent::Error {
                            error: ErrorData {
                                error_type: "stream_error".to_string(),
                                message: error.to_string(),
                            },
                        }))
                        .await;
                    tracing::warn!(
                        request_id = %telemetry_request_id,
                        elapsed_ms = 0u64,
                        error = %error,
                        "Responses stream downstream conversion error"
                    );
                    return;
                }
            };
            utf8_pending.extend_from_slice(&bytes);
            match std::str::from_utf8(&utf8_pending) {
                Ok(text) => {
                    buffer.push_str(text);
                    utf8_pending.clear();
                }
                Err(error) if error.error_len().is_none() => continue,
                Err(_) => {
                    buffer.push_str(&String::from_utf8_lossy(&utf8_pending));
                    utf8_pending.clear();
                }
            }
            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer.drain(..=pos);
                let Some(data) = line.strip_prefix("data:").map(str::trim) else {
                    continue;
                };
                if data == "[DONE]" {
                    continue;
                }
                let value: serde_json::Value = match serde_json::from_str(data) {
                    Ok(value) => value,
                    Err(_) => continue,
                };
                for event in state.event(&value) {
                    if tx.send(axum_event(&event)).await.is_err() {
                        return;
                    }
                }
                let event_type = value.get("type").and_then(|v| v.as_str());
                if matches!(
                    event_type,
                    Some("response.completed") | Some("response.incomplete")
                ) {
                    let terminal_usage = terminal_usage(&value);
                    if let Some(ref u) = terminal_usage {
                        log_cache_stats(
                            &stats_model,
                            u,
                            event_type
                                .unwrap_or("incomplete")
                                .strip_prefix("response.")
                                .unwrap_or("incomplete"),
                            cache_policy.as_ref(),
                        );
                    }
                    tracing::info!(
                        request_id = %telemetry_request_id,
                        terminal_event = %event_type.unwrap_or("unknown"),
                        "Responses stream terminal event"
                    );
                    for event in state.finish() {
                        let _ = tx.send(axum_event(&event)).await;
                    }
                    let stop_reason = if event_type == Some("response.incomplete") {
                        match value
                            .get("response")
                            .and_then(|v| v.get("incomplete_details"))
                            .or_else(|| value.get("incomplete_details"))
                            .and_then(|v| v.get("reason"))
                            .and_then(|v| v.as_str())
                        {
                            Some("max_output_tokens") | Some("max_tokens") => "max_tokens",
                            _ => "end_turn",
                        }
                    } else if state.tool_stop_reason {
                        "tool_use"
                    } else {
                        "end_turn"
                    };
                    let _ = tx
                        .send(axum_event(&SseEvent::MessageDelta {
                            delta: MessageDeltaData {
                                stop_reason: Some(stop_reason.to_string()),
                                stop_sequence: None,
                            },
                            usage: terminal_usage,
                        }))
                        .await;
                    let _ = tx.send(axum_event(&SseEvent::MessageStop)).await;
                    return;
                }
                if matches!(
                    event_type,
                    Some("response.failed")
                        | Some("response.error")
                        | Some("response.cancelled")
                        | Some("error")
                ) {
                    // State::event already emitted the closed blocks and the
                    // Anthropic error event. Do not append a second error or a
                    // synthetic successful terminal message.
                    return;
                }
            }
        }
        let _ = tx
            .send(axum_event(&SseEvent::Error {
                error: ErrorData {
                    error_type: "stream_error".to_string(),
                    message: "Responses stream ended without a terminal event".to_string(),
                },
            }))
            .await;
        tracing::warn!(
            request_id = %telemetry_request_id,
            "Responses stream EOF without terminal event"
        );
    });
    Sse::new(EventStream(rx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_text_delta_and_completed() {
        let mut state = State::new();
        let events = state.event(&serde_json::json!({
            "type": "response.output_text.delta",
            "delta": "hello"
        }));
        assert!(matches!(events[0], SseEvent::ContentBlockStart { .. }));
        assert!(matches!(events[1], SseEvent::ContentBlockDelta { .. }));
        let finished = state.finish();
        assert!(matches!(finished[0], SseEvent::ContentBlockStop { .. }));
    }

    #[test]
    fn converts_function_call_arguments_delta() {
        let mut state = State::new();
        let started = state.event(&serde_json::json!({
            "type": "response.output_item.added",
            "item": {"type": "function_call", "call_id": "call-1", "name": "lookup"}
        }));
        assert!(matches!(started[0], SseEvent::ContentBlockStart { .. }));
        let delta = state.event(&serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "call-1",
            "delta": "{\"q\":"
        }));
        assert!(matches!(
            delta[0],
            SseEvent::ContentBlockDelta {
                delta: ContentBlockDeltaData::InputJsonDelta { .. },
                ..
            }
        ));
    }

    #[test]
    fn function_call_delta_without_item_added_is_buffered() {
        let mut state = State::new();
        let events = state.event(&serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "call-late",
            "delta": "{\"q\":1}"
        }));
        assert!(events.is_empty());
        assert_eq!(
            state
                .pending_tool_arguments
                .get("call-late")
                .map(String::as_str),
            Some("{\"q\":1}")
        );
    }

    #[test]
    fn arguments_before_added_are_buffered_and_real_call_id_is_emitted() {
        let mut state = State::new();
        let delta = state.event(&serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc-item-late",
            "delta": "{\"q\":1}"
        }));
        assert!(delta.is_empty());
        let added = state.event(&serde_json::json!({
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "fc-item-late",
                "call_id": "call-real",
                "name": "lookup"
            }
        }));
        assert!(matches!(
            added.first(),
            Some(SseEvent::ContentBlockStart {
                content_block: ContentBlockStartData::ToolUse { id, name, .. },
                ..
            }) if id == "call-real" && name == "lookup"
        ));
        assert!(added.iter().any(|event| matches!(
            event,
            SseEvent::ContentBlockDelta {
                delta: ContentBlockDeltaData::InputJsonDelta { partial_json },
                ..
            } if partial_json == "{\"q\":1}"
        )));
    }

    #[test]
    fn finishes_multiple_tools_in_start_order() {
        let mut state = State::new();
        state.event(&serde_json::json!({
            "type": "response.output_item.added",
            "item": {"type": "function_call", "call_id": "call-a", "name": "a"}
        }));
        state.event(&serde_json::json!({
            "type": "response.output_item.added",
            "item": {"type": "function_call", "call_id": "call-b", "name": "b"}
        }));
        let indices: Vec<_> = state
            .finish()
            .into_iter()
            .map(|event| match event {
                SseEvent::ContentBlockStop { index } => index,
                _ => panic!("expected only content block stops"),
            })
            .collect();
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn converts_reasoning_summary_delta() {
        let mut state = State::new();
        let events = state.event(&serde_json::json!({
            "type": "response.reasoning_summary_text.delta",
            "delta": "thinking"
        }));
        assert!(matches!(
            events[0],
            SseEvent::ContentBlockStart {
                content_block: ContentBlockStartData::Thinking { .. },
                ..
            }
        ));
        assert!(matches!(
            events[1],
            SseEvent::ContentBlockDelta {
                delta: ContentBlockDeltaData::ThinkingDelta { .. },
                ..
            }
        ));
    }

    #[test]
    fn reasoning_summary_done_emits_full_text_when_no_delta_arrived() {
        let mut state = State::new();
        let events = state.event(&serde_json::json!({
            "type": "response.reasoning_summary_text.done",
            "text": "complete summary"
        }));

        assert!(events.iter().any(|event| matches!(
            event,
            SseEvent::ContentBlockDelta {
                delta: ContentBlockDeltaData::ThinkingDelta { thinking },
                ..
            } if thinking == "complete summary"
        )));
    }

    #[test]
    fn function_call_arguments_done_emits_only_unseen_suffix() {
        let mut state = State::new();
        state.event(&serde_json::json!({
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "fc-item-1",
                "call_id": "call-1",
                "name": "lookup"
            }
        }));
        state.event(&serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc-item-1",
            "delta": "{\"city\":"
        }));

        let events = state.event(&serde_json::json!({
            "type": "response.function_call_arguments.done",
            "item_id": "fc-item-1",
            "arguments": "{\"city\":\"Paris\"}"
        }));
        let suffixes: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                SseEvent::ContentBlockDelta {
                    delta: ContentBlockDeltaData::InputJsonDelta { partial_json },
                    ..
                } => Some(partial_json.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(suffixes, vec!["\"Paris\"}"]);
    }

    #[test]
    fn duplicate_function_call_item_added_does_not_start_a_second_tool_block() {
        let mut state = State::new();
        let item = serde_json::json!({
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "fc-item-1",
                "call_id": "call-1",
                "name": "lookup"
            }
        });
        let first = state.event(&item);
        let duplicate = state.event(&item);
        assert_eq!(
            first
                .iter()
                .filter(|event| matches!(event, SseEvent::ContentBlockStart { .. }))
                .count(),
            1
        );
        assert!(duplicate.is_empty());
    }

    #[test]
    fn responses_error_event_becomes_anthropic_error_and_stops_processing() {
        let mut state = State::new();
        let events = state.event(&serde_json::json!({
            "type": "error",
            "code": "rate_limit",
            "message": "upstream rejected request"
        }));

        assert!(events.iter().any(|event| matches!(
            event,
            SseEvent::Error { error } if error.error_type == "api_error"
                && error.message == "upstream rejected request"
        )));
        assert!(state
            .event(&serde_json::json!({
                "type": "response.output_text.delta",
                "delta": "must not leak after error"
            }))
            .is_empty());
    }

    #[test]
    fn closes_text_block_before_reasoning_block() {
        let mut state = State::new();
        state.event(&serde_json::json!({
            "type": "response.output_text.delta",
            "delta": "answer"
        }));

        let events = state.event(&serde_json::json!({
            "type": "response.reasoning_summary_text.delta",
            "delta": "thinking"
        }));

        assert!(matches!(events[0], SseEvent::ContentBlockStop { index: 0 }));
        assert!(matches!(
            events[1],
            SseEvent::ContentBlockStart {
                index: 1,
                content_block: ContentBlockStartData::Thinking { .. }
            }
        ));
    }

    #[test]
    fn closes_reasoning_block_before_resuming_text() {
        let mut state = State::new();
        state.event(&serde_json::json!({
            "type": "response.output_text.delta",
            "delta": "answer"
        }));
        state.event(&serde_json::json!({
            "type": "response.reasoning_summary_text.delta",
            "delta": "thinking"
        }));

        let events = state.event(&serde_json::json!({
            "type": "response.output_text.delta",
            "delta": " more"
        }));

        assert!(matches!(events[0], SseEvent::ContentBlockStop { index: 1 }));
        assert!(matches!(
            events[1],
            SseEvent::ContentBlockStart {
                index: 2,
                content_block: ContentBlockStartData::Text { .. }
            }
        ));
    }

    #[test]
    fn completed_closes_each_block_once_after_text_then_reasoning() {
        let mut state = State::new();
        state.event(&serde_json::json!({
            "type": "response.output_text.delta",
            "delta": "answer"
        }));
        let transition = state.event(&serde_json::json!({
            "type": "response.reasoning_summary_text.delta",
            "delta": "thinking"
        }));
        let completed = state.finish();

        let transition_stops: Vec<_> = transition
            .iter()
            .filter_map(|event| match event {
                SseEvent::ContentBlockStop { index } => Some(*index),
                _ => None,
            })
            .collect();
        let completed_stops: Vec<_> = completed
            .iter()
            .filter_map(|event| match event {
                SseEvent::ContentBlockStop { index } => Some(*index),
                _ => None,
            })
            .collect();

        assert_eq!(transition_stops, vec![0]);
        assert_eq!(completed_stops, vec![1]);
        assert!(state.finish().is_empty());
    }

    #[test]
    fn completed_closes_each_block_once_after_reasoning_then_text() {
        let mut state = State::new();
        state.event(&serde_json::json!({
            "type": "response.reasoning_summary_text.delta",
            "delta": "thinking"
        }));
        let transition = state.event(&serde_json::json!({
            "type": "response.output_text.delta",
            "delta": "answer"
        }));
        let completed = state.finish();

        let transition_stops: Vec<_> = transition
            .iter()
            .filter_map(|event| match event {
                SseEvent::ContentBlockStop { index } => Some(*index),
                _ => None,
            })
            .collect();
        let completed_stops: Vec<_> = completed
            .iter()
            .filter_map(|event| match event {
                SseEvent::ContentBlockStop { index } => Some(*index),
                _ => None,
            })
            .collect();

        assert_eq!(transition_stops, vec![0]);
        assert_eq!(completed_stops, vec![1]);
        assert!(state.finish().is_empty());
    }

    #[test]
    fn terminal_usage_requires_usage_on_the_current_terminal_event() {
        let previous = serde_json::json!({
            "type": "response.output_text.delta",
            "usage": {"input_tokens": 10}
        });
        let completed_without_usage = serde_json::json!({
            "type": "response.completed",
            "response": {}
        });

        assert!(terminal_usage(&previous).is_some());
        assert!(terminal_usage(&completed_without_usage).is_none());
    }

    #[test]
    fn terminal_usage_reads_completed_usage_once_and_supports_incomplete() {
        let completed = serde_json::json!({
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 12, "output_tokens": 3}}
        });
        let incomplete = serde_json::json!({
            "type": "response.incomplete",
            "response": {"usage": {"input_tokens": 20, "output_tokens": 4}}
        });

        assert_eq!(terminal_usage(&completed).unwrap().input_tokens, Some(12));
        assert_eq!(terminal_usage(&incomplete).unwrap().input_tokens, Some(20));
    }

    #[test]
    fn failed_event_uses_nested_response_error_message() {
        let mut state = State::new();
        let events = state.event(&serde_json::json!({
            "type": "response.failed",
            "response": {"error": {"message": "nested failure"}}
        }));
        assert!(events.iter().any(|event| matches!(
            event,
            SseEvent::Error { error } if error.error_type == "api_error"
                && error.message == "nested failure"
        )));
        assert!(state
            .event(&serde_json::json!({
                "type": "response.output_text.delta",
                "delta": "must not emit"
            }))
            .is_empty());
    }

    #[test]
    fn cancelled_event_is_an_error_not_a_successful_end_turn() {
        let mut state = State::new();
        let events = state.event(&serde_json::json!({
            "type": "response.cancelled",
            "response": {"status": "cancelled"}
        }));
        assert!(events.iter().any(|event| matches!(
            event,
            SseEvent::Error { error } if error.error_type == "api_error"
        )));
        assert!(state.terminal);
    }

    #[test]
    fn abnormal_eof_is_not_successful_end_turn() {
        let mut state = State::new();
        let events = state.event(&serde_json::json!({
            "type": "response.output_text.delta",
            "delta": "partial"
        }));
        assert!(!events.is_empty());
        let eof = state.finish();
        assert!(eof
            .iter()
            .any(|event| matches!(event, SseEvent::ContentBlockStop { .. })));
    }

    #[test]
    fn summary_parts_keep_independent_accumulators() {
        let mut state = State::new();
        let first = state.event(&serde_json::json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs-1",
            "summary_index": 0,
            "delta": "first"
        }));
        let second = state.event(&serde_json::json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs-1",
            "summary_index": 1,
            "delta": "second"
        }));
        let starts = first
            .iter()
            .chain(second.iter())
            .filter(|event| {
                matches!(
                    event,
                    SseEvent::ContentBlockStart {
                        content_block: ContentBlockStartData::Thinking { .. },
                        ..
                    }
                )
            })
            .count();
        let thinking: Vec<_> = first
            .iter()
            .chain(second.iter())
            .filter_map(|event| match event {
                SseEvent::ContentBlockDelta {
                    delta: ContentBlockDeltaData::ThinkingDelta { thinking },
                    ..
                } => Some(thinking.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(starts, 2);
        assert_eq!(thinking, vec!["first", "second"]);
    }

    #[test]
    fn raw_reasoning_text_delta_is_exposed_as_readable_thinking() {
        let mut state = State::new();
        let events = state.event(&serde_json::json!({
            "type": "response.reasoning_text.delta",
            "item_id": "rs-raw",
            "content_index": 0,
            "delta": "visible reasoning"
        }));
        assert!(events.iter().any(|event| matches!(
            event,
            SseEvent::ContentBlockDelta {
                delta: ContentBlockDeltaData::ThinkingDelta { thinking },
                ..
            } if thinking == "visible reasoning"
        )));
    }

    #[test]
    fn late_tool_argument_delta_after_stop_is_ignored() {
        let mut state = State::new();
        state.event(&serde_json::json!({
            "type": "response.output_item.added",
            "item": {"type": "function_call", "call_id": "call-closed", "name": "lookup"}
        }));
        state.event(&serde_json::json!({
            "type": "response.output_item.done",
            "item": {"type": "function_call", "call_id": "call-closed"}
        }));
        let events = state.event(&serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "call-closed",
            "delta": "late"
        }));
        assert!(events.is_empty());
    }

    // --- Phase 2b.3: stream cache-usage coverage ---

    #[test]
    fn terminal_usage_reads_cache_buckets_from_completed() {
        // Feed a `response.completed` with nested cached_tokens + cache_write_tokens
        // (and a top-level cache_write_tokens fallback) through the terminal
        // extractor; assert the StreamUsage read/creation buckets that feed both
        // the log view and the wire `MessageDelta.usage`.
        let completed = serde_json::json!({
            "type": "response.completed",
            "response": {
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 12,
                    "input_tokens_details": {
                        "cached_tokens": 80,
                        "cache_write_tokens": 3
                    },
                    "cache_write_tokens": 9
                }
            }
        });
        let usage = terminal_usage(&completed).expect("completed usage present");
        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.output_tokens, Some(12));
        assert_eq!(
            usage.cache_read_input_tokens,
            Some(80),
            "read = input_tokens_details.cached_tokens"
        );
        assert_eq!(
            usage.cache_creation_input_tokens,
            Some(3),
            "creation = nested cache_write_tokens (top-level is only a fallback)"
        );

        // Top-level-only write is the explicit fallback when nested is absent.
        let top_level = serde_json::json!({
            "type": "response.completed",
            "response": {
                "usage": {
                    "input_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 80},
                    "cache_write_tokens": 9
                }
            }
        });
        let usage = terminal_usage(&top_level).expect("completed usage present");
        assert_eq!(usage.cache_creation_input_tokens, Some(9));
    }

    #[test]
    fn stream_cache_view_legacy_matches_non_stream_and_log_buckets() {
        // The stream selector (over extracted buckets) under the default None
        // policy must reproduce the exact historical log numbers: read, clamped
        // miss, and hit rate.
        let legacy =
            crate::cache::responses_usage_view_from_buckets(Some(100), Some(80), Some(3), None);
        assert_eq!(legacy.read, Some(80));
        assert_eq!(legacy.creation, Some(3));
        assert_eq!(legacy.miss, Some(17));
        assert_eq!(legacy.hit_rate, Some(80.0));

        // Same buckets through the non-stream selector must agree.
        let usage = crate::responses::types::ResponsesUsage {
            input_tokens: Some(100),
            output_tokens: Some(12),
            input_tokens_details: Some(crate::responses::types::InputTokenDetails {
                cached_tokens: Some(80),
                cache_write_tokens: Some(3),
            }),
            cache_write_tokens: None,
        };
        let non_stream = crate::cache::responses_usage_view(Some(&usage), None);
        assert_eq!(legacy, non_stream);
    }

    #[test]
    fn stream_cache_view_raw_under_opt_in_uses_guarded_miss() {
        // Explicit policy on the stream path: raw projection with a guarded miss
        // (never negative), while legacy clamps.
        let policy = crate::cache::CachePolicy {
            usage: crate::cache::UsagePolicy::TopLevelCachedTokens,
            prompt_cache_key_enabled: false,
            upstream: None,
            effort_enum: None,
        };
        // Consistent data: miss = input - read - creation.
        let raw = crate::cache::responses_usage_view_from_buckets(
            Some(100),
            Some(80),
            Some(3),
            Some(&policy),
        );
        assert_eq!(raw.read, Some(80));
        assert_eq!(raw.creation, Some(3));
        assert_eq!(raw.miss, Some(17));
        assert_eq!(raw.hit_rate, Some(80.0));

        // Inconsistent data (cached > prompt): raw keeps miss unknown; the
        // legacy branch clamps it to zero — never a negative.
        let raw_inconsistent = crate::cache::responses_usage_view_from_buckets(
            Some(50),
            Some(70),
            Some(0),
            Some(&policy),
        );
        assert_eq!(raw_inconsistent.miss, None);
        let legacy_inconsistent =
            crate::cache::responses_usage_view_from_buckets(Some(50), Some(70), Some(0), None);
        assert_eq!(legacy_inconsistent.miss, Some(0));
    }

    #[test]
    fn terminal_usage_missing_usage_is_none_not_a_miss() {
        // A terminal event without a usage object yields None (the log is
        // skipped, and the wire MessageDelta.usage is None) — never a miss.
        let completed_without_usage = serde_json::json!({
            "type": "response.completed",
            "response": {}
        });
        assert!(terminal_usage(&completed_without_usage).is_none());
    }
}
