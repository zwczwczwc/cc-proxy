// Copyright (c) 2025 cc-proxy
//
// Canonical JSON hashing / normalization, shared by cache telemetry and tests.
//
// Phase 2 scope: this module ONLY extracts the canonical hash helper that was
// previously private to `responses/request.rs` and reuses it unchanged. It does
// NOT change any outbound wire bytes, and canonicalization is NEVER claimed to
// be a cause of cache hits — it is an observation/normalization tool for
// telemetry and tests.
//
// Canonicalization contract:
// - object keys are sorted recursively (deterministic hashing regardless of key
//   insertion order), and
// - arrays keep their exact order — array order is semantic and is never sorted.
//   This matters for tool order, message order, and `required`/`enum` lists.

use serde_json::Value;
use sha2::Digest;

/// Recursively canonicalize a JSON value: sort object keys, preserve array order.
pub(crate) fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            let sorted = object
                .iter()
                .map(|(key, value)| (key.clone(), canonicalize(value)))
                .collect::<std::collections::BTreeMap<_, _>>();
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(canonicalize).collect()),
        _ => value.clone(),
    }
}

/// Serialize the canonical form of a value to bytes.
pub(crate) fn canonical_json(value: &Value) -> Vec<u8> {
    serde_json::to_vec(&canonicalize(value)).expect("JSON serialization cannot fail")
}

/// SHA-256 (truncated to 8 bytes, hex) over the canonical JSON of `value`.
///
/// Identical semantics to the private helper that used to live in
/// `responses/request.rs`; the extraction is byte-for-byte behavior preserving.
pub(crate) fn canonical_hash(value: &Value) -> String {
    let mut hasher = sha2::Sha256::new();
    hasher.update(canonical_json(value));
    hex::encode(&hasher.finalize()[..8])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_key_order_does_not_change_hash() {
        let a = canonical_hash(&serde_json::json!({"b": 2, "a": 1}));
        let b = canonical_hash(&serde_json::json!({"a": 1, "b": 2}));
        assert_eq!(a, b);
    }

    #[test]
    fn nested_object_key_order_is_canonicalized() {
        let a = canonical_hash(&serde_json::json!({
            "tools": [{"z": 1, "y": {"inner": true, "outer": "x"}}],
            "model": "m",
        }));
        let b = canonical_hash(&serde_json::json!({
            "model": "m",
            "tools": [{"y": {"outer": "x", "inner": true}, "z": 1}],
        }));
        assert_eq!(a, b);
    }

    #[test]
    fn semantic_arrays_keep_order_and_differ() {
        // Array order is semantic and must NOT be sorted by canonicalization.
        let a = canonical_hash(&serde_json::json!({"arr": [1, 2, 3]}));
        let b = canonical_hash(&serde_json::json!({"arr": [3, 2, 1]}));
        assert_ne!(a, b, "reordered semantic arrays must hash differently");
    }

    #[test]
    fn arrays_inside_objects_keep_order() {
        let a = canonical_hash(&serde_json::json!({"tool": {"required": ["a", "b"]}}));
        let b = canonical_hash(&serde_json::json!({"tool": {"required": ["b", "a"]}}));
        assert_ne!(
            a, b,
            "required/enum list order is semantic and must be preserved"
        );
    }

    #[test]
    fn scalar_and_null_are_stable() {
        let a = canonical_hash(&serde_json::json!({"model": null, "n": 3}));
        let b = canonical_hash(&serde_json::json!({"n": 3, "model": null}));
        assert_eq!(a, b);
    }
}
