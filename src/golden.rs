//! Per-wire golden snapshot capture + verification (Phase 1, gate G5).
//!
//! Captures the *actual upstream serialized request body* produced by the
//! current encoders — Chat via `anthropic::converter::convert_request`
//! (post-`sanitize_thinking_mode_messages` body), Responses via
//! `responses::request::convert_request` (all `#[serde(skip)]` fields such as
//! `request_id` / hash telemetry excluded automatically) — and stores it under
//! `tests/golden/*.json` as **pure data**.
//!
//! Harness constraints (KIMI-K3-CACHE-OPTIMIZATION-FINAL-PLAN §6, 27 SF-1):
//! this crate is a binary-only crate with no `src/lib.rs` and no Rust
//! integration-test crate; capture logic lives in `#[cfg(test)]` modules
//! (this file is declared `#[cfg(test)] mod golden;` in `src/main.rs`) and the
//! JSON files are read via `CARGO_MANIFEST_DIR`. Cargo never treats `.json`
//! under `tests/` as a Rust test target.
//!
//! Modes:
//!   * default — verify: re-encode every fixture and compare the
//!     resulting wire bytes (sha256) + body against the
//!     committed snapshots. Byte mismatch ⇒ NO-GO.
//!   * GOLDEN_CAPTURE=1 — (re)capture: write the snapshots from the *current*
//!     encoder. Must be run *before* the ConversationIR
//!     refactor so the committed snapshots reflect the old
//!     encoder (zero-behavior baseline).

use crate::anthropic::converter::convert_request_with_relocation as chat_convert;
use crate::anthropic::types::MessagesRequest;
use crate::config::Config;
use crate::responses::request::convert_request_with_relocation as responses_convert;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const GOLDEN_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden");
const GOLDEN_SCHEMA: &str = "cc-proxy-golden/v1";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Wire {
    Chat,
    Responses,
}

impl Wire {
    fn as_str(self) -> &'static str {
        match self {
            Wire::Chat => "chat",
            Wire::Responses => "responses",
        }
    }
}

struct Fixture {
    name: &'static str,
    chat: Value,
    responses: Value,
}

/// The fixture set. Content dimensions covered: text / thinking(+redacted) /
/// tool call + tool result / multi-turn history / null content / image /
/// unknown role / volatile env system blocks (relocate-relevant).
/// Stream is a per-fixture attribute (`text_stream`, `tool_round_stream`).
/// "refusal" is not representable in the inbound `Message`/`ContentBlock`
/// types (no Anthropic refusal block exists in the request model) — recorded
/// as a fixture gap, not fabricated.
fn fixtures() -> Vec<Fixture> {
    vec![
        Fixture {
            name: "text",
            chat: json!({
                "model": "kimi-k3", "max_tokens": 4096,
                "system": "You are a helpful coding assistant.",
                "messages": [{"role":"user","content":"hello"}]
            }),
            responses: json!({
                "model": "gpt-5.6-luna", "max_tokens": 4096,
                "system": "You are a helpful coding assistant.",
                "messages": [{"role":"user","content":"hello"}]
            }),
        },
        Fixture {
            name: "text_stream",
            chat: json!({
                "model": "kimi-k3", "max_tokens": 4096, "stream": true,
                "system": "You are a helpful coding assistant.",
                "messages": [{"role":"user","content":"hello"}]
            }),
            responses: json!({
                "model": "gpt-5.6-luna", "max_tokens": 4096, "stream": true,
                "system": "You are a helpful coding assistant.",
                "messages": [{"role":"user","content":"hello"}]
            }),
        },
        Fixture {
            name: "thinking",
            chat: json!({
                "model": "kimi-k3", "max_tokens": 32768,
                "thinking": {"type":"enabled","budget_tokens":16000},
                "system": "You are a helpful coding assistant.",
                "messages": [
                    {"role":"user","content":"q1"},
                    {"role":"assistant","content":[
                        {"type":"thinking","thinking":"let me reason","signature":"sig1"},
                        {"type":"redacted_thinking","data":"encrypted_data"},
                        {"type":"text","text":"a1"}
                    ]}
                ]
            }),
            responses: json!({
                "model": "gpt-5.6-luna", "max_tokens": 32768,
                "thinking": {"type":"enabled","budget_tokens":32000},
                "system": "You are a helpful coding assistant.",
                "messages": [
                    {"role":"user","content":"q1"},
                    {"role":"assistant","content":[
                        {"type":"thinking","thinking":"let me reason","signature":"sig1"},
                        {"type":"redacted_thinking","data":"encrypted_data"},
                        {"type":"text","text":"a1"}
                    ]}
                ]
            }),
        },
        Fixture {
            name: "tool_round",
            chat: json!({
                "model": "kimi-k3", "max_tokens": 4096,
                "messages": [
                    {"role":"user","content":"call lookup"},
                    {"role":"assistant","content":[
                        {"type":"tool_use","id":"call-1","name":"lookup","input":{"q":"Paris"}}
                    ]},
                    {"role":"user","content":[
                        {"type":"tool_result","tool_use_id":"call-1","content":"ok"}
                    ]}
                ]
            }),
            responses: json!({
                "model": "gpt-5.6-luna", "max_tokens": 4096,
                "messages": [
                    {"role":"user","content":"call lookup"},
                    {"role":"assistant","content":[
                        {"type":"tool_use","id":"call-1","name":"lookup","input":{"q":"Paris"}}
                    ]},
                    {"role":"user","content":[
                        {"type":"tool_result","tool_use_id":"call-1","content":"ok"}
                    ]}
                ]
            }),
        },
        Fixture {
            name: "tool_round_stream",
            chat: json!({
                "model": "kimi-k3", "max_tokens": 4096, "stream": true,
                "messages": [
                    {"role":"user","content":"call lookup"},
                    {"role":"assistant","content":[
                        {"type":"tool_use","id":"call-1","name":"lookup","input":{"q":"Paris"}}
                    ]},
                    {"role":"user","content":[
                        {"type":"tool_result","tool_use_id":"call-1","content":"ok"}
                    ]}
                ]
            }),
            responses: json!({
                "model": "gpt-5.6-luna", "max_tokens": 4096, "stream": true,
                "messages": [
                    {"role":"user","content":"call lookup"},
                    {"role":"assistant","content":[
                        {"type":"tool_use","id":"call-1","name":"lookup","input":{"q":"Paris"}}
                    ]},
                    {"role":"user","content":[
                        {"type":"tool_result","tool_use_id":"call-1","content":"ok"}
                    ]}
                ]
            }),
        },
        Fixture {
            name: "multi_turn",
            chat: json!({
                "model": "kimi-k3", "max_tokens": 4096,
                "system": [
                    {"type":"text","text":"You are a helpful coding assistant."},
                    {"type":"text","text":"<env>\nWorking directory: /home/user\nToday's date: 2026-06-22\nPlatform: linux\ngitStatus: M foo.rs\n</env>"}
                ],
                "messages": [
                    {"role":"user","content":"u1"},
                    {"role":"assistant","content":[
                        {"type":"thinking","thinking":"think 1","signature":"s1"},
                        {"type":"text","text":"a1"}
                    ]},
                    {"role":"user","content":"u2"},
                    {"role":"assistant","content":[
                        {"type":"tool_use","id":"call-1","name":"read_file","input":{"path":"/tmp/x"}}
                    ]},
                    {"role":"user","content":[
                        {"type":"tool_result","tool_use_id":"call-1","content":"file contents"}
                    ]},
                    {"role":"user","content":"u3"}
                ]
            }),
            responses: json!({
                "model": "gpt-5.6-luna", "max_tokens": 4096,
                "system": [
                    {"type":"text","text":"You are a helpful coding assistant."},
                    {"type":"text","text":"<env>\nWorking directory: /home/user\nToday's date: 2026-06-22\nPlatform: linux\ngitStatus: M foo.rs\n</env>"}
                ],
                "messages": [
                    {"role":"user","content":"u1"},
                    {"role":"assistant","content":[
                        {"type":"thinking","thinking":"think 1","signature":"s1"},
                        {"type":"text","text":"a1"}
                    ]},
                    {"role":"user","content":"u2"},
                    {"role":"assistant","content":[
                        {"type":"tool_use","id":"call-1","name":"read_file","input":{"path":"/tmp/x"}}
                    ]},
                    {"role":"user","content":[
                        {"type":"tool_result","tool_use_id":"call-1","content":"file contents"}
                    ]},
                    {"role":"user","content":"u3"}
                ]
            }),
        },
        Fixture {
            name: "null_content",
            chat: json!({
                "model": "kimi-k3", "max_tokens": 4096,
                "messages": [
                    {"role":"user","content":"hello"},
                    {"role":"assistant","content":null}
                ]
            }),
            responses: json!({
                "model": "gpt-5.6-luna", "max_tokens": 4096,
                "messages": [
                    {"role":"user","content":"hello"},
                    {"role":"assistant","content":null}
                ]
            }),
        },
        Fixture {
            name: "image",
            chat: json!({
                "model": "kimi-k3", "max_tokens": 4096,
                "messages": [
                    {"role":"user","content":[
                        {"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}}
                    ]}
                ]
            }),
            responses: json!({
                "model": "gpt-5.6-luna", "max_tokens": 4096,
                "messages": [
                    {"role":"user","content":[
                        {"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}}
                    ]}
                ]
            }),
        },
        Fixture {
            name: "env_system",
            chat: json!({
                "model": "kimi-k3", "max_tokens": 4096,
                "system": [
                    {"type":"text","text":"You are a helpful coding assistant."},
                    {"type":"text","text":"<env>\nToday's date: 2026-06-22\nWorking directory: /tmp\n</env>"}
                ],
                "messages": [
                    {"role":"user","content":"hello"},
                    {"role":"assistant","content":"hi"},
                    {"role":"user","content":"what now"}
                ]
            }),
            responses: json!({
                "model": "gpt-5.6-luna", "max_tokens": 4096,
                "system": [
                    {"type":"text","text":"You are a helpful coding assistant."},
                    {"type":"text","text":"<env>\nToday's date: 2026-06-22\nWorking directory: /tmp\n</env>"}
                ],
                "messages": [
                    {"role":"user","content":"hello"},
                    {"role":"assistant","content":"hi"},
                    {"role":"user","content":"what now"}
                ]
            }),
        },
        Fixture {
            name: "unknown_role",
            chat: json!({
                "model": "kimi-k3", "max_tokens": 4096,
                "messages": [
                    {"role":"developer","content":"set temperature to 0"},
                    {"role":"user","content":"hello"}
                ]
            }),
            responses: json!({
                "model": "gpt-5.6-luna", "max_tokens": 4096,
                "messages": [
                    {"role":"developer","content":"set temperature to 0"},
                    {"role":"user","content":"hello"}
                ]
            }),
        },
    ]
}

/// Encode a fixture through the requested wire encoder and return the actual
/// upstream serialized body as a `Value` (exactly what `.json(request)` on the
/// HTTP client produces).
fn encode(
    wire: Wire,
    req: &MessagesRequest,
    config: &Config,
    relocate: bool,
) -> anyhow::Result<Value> {
    match wire {
        Wire::Chat => Ok(serde_json::to_value(chat_convert(req, config, relocate)?)?),
        Wire::Responses => Ok(serde_json::to_value(responses_convert(
            req, config, relocate,
        )?)?),
    }
}

fn wire_sha256(body: &Value) -> String {
    let bytes = serde_json::to_vec(body).expect("serialize body cannot fail");
    hex::encode(&Sha256::digest(&bytes)[..])
}

fn golden_path(wire: Wire, fixture: &str) -> PathBuf {
    PathBuf::from(GOLDEN_DIR).join(format!("{}_{}.json", wire.as_str(), fixture))
}

fn write_golden(path: &Path, wire: Wire, fixture: &str, relocate: bool, sha: &str, body: &Value) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create golden dir");
    }
    let doc = json!({
        "schema": GOLDEN_SCHEMA,
        "wire": wire.as_str(),
        "fixture": fixture,
        "relocate": relocate,
        "sha256": sha,
        "body": body,
    });
    let contents = format!(
        "{}\n",
        serde_json::to_string_pretty(&doc).expect("golden doc serialization cannot fail")
    );
    std::fs::write(path, contents).expect("write golden file");
}

fn verify_golden(path: &Path, wire: Wire, fixture: &str, relocate: bool, sha: &str, body: &Value) {
    assert!(
        path.exists(),
        "missing golden snapshot {path:?} — run with GOLDEN_CAPTURE=1 to (re)capture from the current encoder"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let doc: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path:?}: {e}"));
    let expected_sha = doc["sha256"]
        .as_str()
        .unwrap_or_else(|| panic!("{path:?} missing sha256 field"));
    assert_eq!(
        expected_sha, sha,
        "GOLDEN BYTE MISMATCH (G5 → NO-GO): wire={wire:?} fixture={fixture} relocate={relocate} — {path:?}"
    );
    assert_eq!(
        &doc["body"], body,
        "GOLDEN BODY MISMATCH: wire={wire:?} fixture={fixture} relocate={relocate} — {path:?}"
    );
}

fn capture_or_verify(wire: Wire, fixture: &Fixture, config: &Config) {
    let req: MessagesRequest = match wire {
        Wire::Chat => serde_json::from_value(fixture.chat.clone()).expect("chat fixture parses"),
        Wire::Responses => {
            serde_json::from_value(fixture.responses.clone()).expect("responses fixture parses")
        }
    };
    for relocate in [false, true] {
        let body = encode(wire, &req, config, relocate).unwrap_or_else(|e| {
            panic!(
                "encode {wire:?} {} relocate={relocate}: {e:#}",
                fixture.name
            )
        });
        let sha = wire_sha256(&body);
        let path = golden_path(wire, &format!("{}_relocate_{relocate}", fixture.name));
        if std::env::var_os("GOLDEN_CAPTURE").is_some() {
            write_golden(&path, wire, fixture.name, relocate, &sha, &body);
        } else {
            verify_golden(&path, wire, fixture.name, relocate, &sha, &body);
        }
    }
}

#[test]
fn golden_per_wire_bytes_match_committed_snapshots() {
    let config = crate::test_support::test_config();
    let fixtures = fixtures();
    assert!(fixtures.len() >= 8, "fixture set must be non-trivial");
    for fixture in &fixtures {
        capture_or_verify(Wire::Chat, fixture, &config);
        capture_or_verify(Wire::Responses, fixture, &config);
    }
}

#[test]
fn golden_repeated_encoding_is_deterministic() {
    // golden-3 (T03 MUST): same input, repeated encoding → byte-identical.
    let config = crate::test_support::test_config();
    let fixtures = fixtures();
    for fixture in &fixtures {
        for wire in [Wire::Chat, Wire::Responses] {
            let req: MessagesRequest = match wire {
                Wire::Chat => serde_json::from_value(fixture.chat.clone()).unwrap(),
                Wire::Responses => serde_json::from_value(fixture.responses.clone()).unwrap(),
            };
            for relocate in [false, true] {
                let a =
                    serde_json::to_vec(&encode(wire, &req, &config, relocate).unwrap()).unwrap();
                let b =
                    serde_json::to_vec(&encode(wire, &req, &config, relocate).unwrap()).unwrap();
                assert_eq!(
                    a, b,
                    "non-deterministic encode: wire={wire:?} fixture={} relocate={relocate}",
                    fixture.name
                );
            }
        }
    }
}

#[test]
fn cross_wire_semantic_parity_carries_conversation_content() {
    // T04 (SHOULD): both wires carry the same conversation content
    // (user/assistant text, tool name, tool result) while the wire-specific
    // divergence is preserved (Responses drops thinking).
    let config = crate::test_support::test_config();
    let fixture = fixtures()
        .into_iter()
        .find(|f| f.name == "multi_turn")
        .expect("multi_turn fixture present");
    let chat_req: MessagesRequest = serde_json::from_value(fixture.chat).unwrap();
    let resp_req: MessagesRequest = serde_json::from_value(fixture.responses).unwrap();
    let chat_str = encode(Wire::Chat, &chat_req, &config, false)
        .unwrap()
        .to_string();
    let resp_str = encode(Wire::Responses, &resp_req, &config, false)
        .unwrap()
        .to_string();
    for needle in ["u1", "a1", "u2", "u3", "read_file", "file contents"] {
        assert!(chat_str.contains(needle), "Chat must carry {needle:?}");
        assert!(resp_str.contains(needle), "Responses must carry {needle:?}");
    }
    // Chat replays thinking as reasoning_content; Responses drops it (semantic
    // difference is preserved by the encoders, not flattened by the IR).
    assert!(chat_str.contains("think 1"), "Chat replays thinking");
    assert!(!resp_str.contains("think 1"), "Responses drops thinking");
}
