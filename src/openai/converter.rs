use serde_json::Value;
use crate::anthropic::types::{
    MessagesResponse, ResponseContentBlock, Usage, SseEvent,
    MessageStartData, ContentBlockStartData, ContentBlockDeltaData,
    MessageDeltaData, StreamUsage,
};
use crate::openai::types::{ChatCompletionResponse, ChatDelta};

/// Convert OpenAI non-streaming response to Anthropic format.
pub fn convert_non_stream_response(
    openai_resp: &ChatCompletionResponse,
    model: &str,
    msg_id: &str,
) -> MessagesResponse {
    let choice = openai_resp.choices.first();
    let message = choice.and_then(|c| c.message.as_ref());

    let mut content: Vec<ResponseContentBlock> = Vec::new();

    if let Some(msg) = message {
        // 1. Thinking block (from reasoning_content or reasoning)
        // kimi-k3 uses "reasoning" field, DeepSeek uses "reasoning_content"
        let reasoning = msg.reasoning_content.as_ref().or(msg.reasoning.as_ref());
        if let Some(ref rc) = reasoning {
            if !rc.trim().is_empty() {
                content.push(ResponseContentBlock::Thinking {
                    thinking: rc.to_string(),
                    signature: "sig_proxy_placeholder".to_string(),
                });
            }
        }

        // 2. Text block
        if let Some(ref text) = msg.content {
            if !text.trim().is_empty() {
                content.push(ResponseContentBlock::Text {
                    text: text.clone(),
                });
            }
        }

        // 3. Tool use blocks
        if let Some(tool_calls) = &msg.tool_calls {
            for tc in tool_calls {
                let input: Value = tc
                    .function
                    .as_ref()
                    .and_then(|f| serde_json::from_str(&f.arguments).ok())
                    .unwrap_or(Value::Null);
                content.push(ResponseContentBlock::ToolUse {
                    id: tc.id.clone(),
                    name: tc.function.as_ref().map_or("unknown".to_string(), |f| f.name.clone()),
                    input,
                });
            }
        }
    }

    let stop_reason = choice
        .and_then(|c| c.finish_reason.as_ref())
        .map(|fr| map_finish_reason(fr));

    let usage = openai_resp.usage.as_ref().map_or(
        Usage {
            input_tokens: 0,
            output_tokens: 0,
        },
        |u| Usage {
            input_tokens: u.prompt_tokens.unwrap_or(0),
            output_tokens: u.completion_tokens.unwrap_or(0),
        },
    );

    MessagesResponse {
        id: msg_id.to_string(),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: model.to_string(),
        stop_reason,
        stop_sequence: None,
        usage,
    }
}

fn map_finish_reason(fr: &str) -> String {
    match fr {
        "stop" => "end_turn".to_string(),
        "tool_calls" => "tool_use".to_string(),
        "length" => "max_tokens".to_string(),
        other => other.to_string(),
    }
}

/// SSE stream state machine — tracks block transitions during streaming.
pub struct SseStateMachine {
    content_index: u32,
    text_started: bool,
    thinking_started: bool,
    tool_indices: std::collections::HashMap<u32, u32>,
    is_reasoning_model: bool,
    /// Accumulated text for current text block
    current_text: String,
    /// Accumulated thinking for current thinking block
    current_thinking: String,
    /// Track tool call names for content_block_start
    tool_names: std::collections::HashMap<u32, String>,
    /// Track if signature_delta has been sent for current thinking block
    thinking_signature_sent: bool,
    /// Track if message_start has been sent
    message_start_sent: bool,
    /// Input tokens from usage
    input_tokens: Option<u32>,
}

impl SseStateMachine {
    pub fn new(is_reasoning_model: bool) -> Self {
        Self {
            content_index: 0,
            text_started: false,
            thinking_started: false,
            tool_indices: std::collections::HashMap::new(),
            is_reasoning_model,
            current_text: String::new(),
            current_thinking: String::new(),
            tool_names: std::collections::HashMap::new(),
            thinking_signature_sent: false,
            message_start_sent: false,
            input_tokens: None,
        }
    }

    /// Process a single SSE delta chunk. Returns Vec of SSE events to emit.
    pub fn process_delta(
        &mut self,
        delta: &ChatDelta,
        usage: Option<&crate::openai::types::Usage>,
    ) -> Vec<SseEvent> {
        let mut events = Vec::new();

        // Handle usage-only chunks (no choices/delta)
        // kimi-k3 uses "reasoning" field, DeepSeek uses "reasoning_content"
        let has_reasoning = delta.reasoning_content.is_some() || delta.reasoning.is_some();
        if !has_reasoning
            && delta.content.is_none()
            && delta.tool_calls.is_none()
            && delta.role.is_none()
        {
            if let Some(usage) = usage {
                self.input_tokens = usage.prompt_tokens;
            }
            return events;
        }

        // 1. Process reasoning_content (DeepSeek) or reasoning (kimi)
        let reasoning_delta = delta.reasoning_content.as_ref().or(delta.reasoning.as_ref());
        if let Some(ref rc) = reasoning_delta {
            if !rc.is_empty() && self.is_reasoning_model {
                if !self.thinking_started {
                    // Close any open text block first
                    if self.text_started {
                        events.push(SseEvent::ContentBlockStop {
                            index: self.content_index - 1,
                        });
                        self.text_started = false;
                    }
                    // Start thinking block
                    events.push(SseEvent::ContentBlockStart {
                        index: self.content_index,
                        content_block: ContentBlockStartData::Thinking {
                            thinking: String::new(),
                            signature: String::new(),
                        },
                    });
                    self.thinking_started = true;
                    self.thinking_signature_sent = false;
                    self.current_thinking = String::new();
                }
                self.current_thinking.push_str(rc);
                events.push(SseEvent::ContentBlockDelta {
                    index: self.content_index,
                    delta: ContentBlockDeltaData::ThinkingDelta {
                        thinking: rc.to_string(),
                    },
                });
            }
        }

        // 2. Process content
        if let Some(ref text) = delta.content {
            if !text.is_empty() {
                // Close thinking block if open
                if self.thinking_started {
                    // Send signature_delta before closing thinking block
                    if !self.thinking_signature_sent {
                        events.push(SseEvent::ContentBlockDelta {
                            index: self.content_index,
                            delta: ContentBlockDeltaData::SignatureDelta {
                                signature: "sig_proxy_placeholder".to_string(),
                            },
                        });
                        self.thinking_signature_sent = true;
                    }
                    events.push(SseEvent::ContentBlockStop {
                        index: self.content_index,
                    });
                    self.content_index += 1;
                    self.thinking_started = false;
                }
                if !self.text_started {
                    events.push(SseEvent::ContentBlockStart {
                        index: self.content_index,
                        content_block: ContentBlockStartData::Text {
                            text: String::new(),
                        },
                    });
                    self.text_started = true;
                    self.current_text = String::new();
                }
                self.current_text.push_str(text);
                events.push(SseEvent::ContentBlockDelta {
                    index: self.content_index,
                    delta: ContentBlockDeltaData::TextDelta {
                        text: text.clone(),
                    },
                });
            }
        }

        // 3. Process tool_calls
        if let Some(tool_calls) = &delta.tool_calls {
            for tc in tool_calls {
                let tc_index = tc.index.unwrap_or(0);
                let sse_index = if let Some(&idx) = self.tool_indices.get(&tc_index) {
                    idx
                } else {
                    // New tool call
                    let new_idx = if self.thinking_started || self.text_started {
                        // Close current block
                        if self.thinking_started {
                            if !self.thinking_signature_sent {
                                events.push(SseEvent::ContentBlockDelta {
                                    index: self.content_index,
                                    delta: ContentBlockDeltaData::SignatureDelta {
                                        signature: "sig_proxy_placeholder".to_string(),
                                    },
                                });
                                self.thinking_signature_sent = true;
                            }
                            events.push(SseEvent::ContentBlockStop {
                                index: self.content_index,
                            });
                            self.content_index += 1;
                            self.thinking_started = false;
                        }
                        if self.text_started {
                            events.push(SseEvent::ContentBlockStop {
                                index: self.content_index,
                            });
                            self.content_index += 1;
                            self.text_started = false;
                        }
                        self.content_index
                    } else {
                        self.content_index
                    };
                    self.tool_indices.insert(tc_index, new_idx);

                    // Get tool call id and name correctly (P1-1: fix id/name swap)
                    // id: from tc.id (call ID), name: from tc.function.name (tool name)
                    let call_id = tc.id.clone().unwrap_or_else(|| "unknown_call".to_string());
                    let tool_name = if let Some(ref func) = tc.function {
                        func.name.clone().unwrap_or_else(|| "unknown_tool".to_string())
                    } else {
                        "unknown_tool".to_string()
                    };

                    self.tool_names.insert(new_idx, tool_name.clone());

                    events.push(SseEvent::ContentBlockStart {
                        index: new_idx,
                        content_block: ContentBlockStartData::ToolUse {
                            id: call_id,
                            name: tool_name,
                            input: Value::Object(serde_json::Map::new()),
                        },
                    });
                    self.content_index += 1;

                    new_idx
                };

                // Handle tool call delta
                if let Some(ref func) = tc.function {
                    if let Some(ref name) = func.name {
                        // Update name in tool_names
                        self.tool_names.insert(sse_index, name.clone());
                    }
                    if let Some(ref args) = func.arguments {
                        events.push(SseEvent::ContentBlockDelta {
                            index: sse_index,
                            delta: ContentBlockDeltaData::InputJsonDelta {
                                partial_json: args.clone(),
                            },
                        });
                    }
                }
                // Handle tool call id
                if let Some(ref id) = tc.id {
                    self.tool_names.insert(sse_index, id.clone());
                }
            }
        }

        events
    }

    /// Close all open blocks and return final events (message_delta + message_stop).
    pub fn finalize(
        &mut self,
        stop_reason: Option<&str>,
        output_tokens: Option<u32>,
    ) -> Vec<SseEvent> {
        let mut events = Vec::new();

        // Close thinking block
        if self.thinking_started {
            if !self.thinking_signature_sent {
                events.push(SseEvent::ContentBlockDelta {
                    index: self.content_index,
                    delta: ContentBlockDeltaData::SignatureDelta {
                        signature: "sig_proxy_placeholder".to_string(),
                    },
                });
            }
            events.push(SseEvent::ContentBlockStop {
                index: self.content_index,
            });
            self.thinking_started = false;
        }

        // Close text block
        if self.text_started {
            events.push(SseEvent::ContentBlockStop {
                index: self.content_index,
            });
            self.text_started = false;
        }

        // Close all tool blocks
        for (&_tc_idx, &sse_idx) in &self.tool_indices {
            events.push(SseEvent::ContentBlockStop { index: sse_idx });
        }

        // Message delta
        let mapped_reason = stop_reason.map(|fr| map_finish_reason(fr));
        events.push(SseEvent::MessageDelta {
            delta: MessageDeltaData {
                stop_reason: mapped_reason,
                stop_sequence: None,
            },
            usage: Some(StreamUsage {
                input_tokens: self.input_tokens,
                output_tokens,
            }),
        });

        // Message stop
        events.push(SseEvent::MessageStop);

        events
    }

    /// Generate the message_start event (should be sent first in the stream).
    pub fn message_start(&mut self, model: &str, msg_id: &str) -> SseEvent {
        self.message_start_sent = true;
        SseEvent::MessageStart {
            message: MessageStartData {
                id: msg_id.to_string(),
                msg_type: "message".to_string(),
                role: "assistant".to_string(),
                content: vec![],
                model: model.to_string(),
                usage: Some(StreamUsage {
                    input_tokens: None,
                    output_tokens: None,
                }),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_non_stream_with_thinking() {
        let openai_resp = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            choices: vec![crate::openai::types::Choice {
                index: 0,
                message: Some(crate::openai::types::ChatMessage {
                    role: Some("assistant".to_string()),
                    content: Some("answer".to_string()),
                    reasoning_content: Some("let me think".to_string()),
                    reasoning: None,
                    tool_calls: None,
                }),
                delta: None,
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(crate::openai::types::Usage {
                prompt_tokens: Some(10),
                completion_tokens: Some(5),
                total_tokens: Some(15),
                prompt_cache_hit_tokens: None,
                prompt_cache_miss_tokens: None,
                prompt_tokens_details: None,
            }),
        };

        let result = convert_non_stream_response(&openai_resp, "deepseek-v4", "msg_123");
        assert_eq!(result.content.len(), 2);
        assert!(matches!(result.content[0], ResponseContentBlock::Thinking { .. }));
        assert!(matches!(result.content[1], ResponseContentBlock::Text { .. }));
        assert_eq!(result.stop_reason, Some("end_turn".to_string()));
    }
}