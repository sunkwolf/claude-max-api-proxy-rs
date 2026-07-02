#![allow(dead_code)]

use serde::Deserialize;
use std::collections::HashMap;

/// Represents the different message types from the Claude CLI's stream-json output.
/// Each line of stdout is one of these JSON objects.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ClaudeCliMessage {
    #[serde(rename = "system")]
    System(SystemMessage),

    #[serde(rename = "assistant")]
    Assistant(AssistantMessage),

    #[serde(rename = "result")]
    Result(ResultMessage),

    /// Stream events wrapped in a top-level envelope.
    /// The CLI emits these as `{"type":"stream_event","event":{...}}` when
    /// using `--include-partial-messages`.
    #[serde(rename = "stream_event")]
    StreamEventWrapper(StreamEventEnvelope),
}

/// Envelope for stream events emitted as top-level NDJSON lines.
#[derive(Debug, Deserialize)]
pub struct StreamEventEnvelope {
    pub event: StreamEvent,
}

/// A streaming event within the assistant message flow.
/// These appear as nested JSON within assistant messages when using --include-partial-messages.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: Option<u64>,
        content_block: Option<ContentBlock>,
    },

    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: Option<u64>, delta: Delta },

    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: Option<u64> },

    #[serde(rename = "message_start")]
    MessageStart {},

    #[serde(rename = "message_delta")]
    MessageDelta {},

    #[serde(rename = "message_stop")]
    MessageStop {},
}

#[derive(Debug, Deserialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub block_type: Option<String>,
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Delta {
    #[serde(rename = "type")]
    pub delta_type: Option<String>,
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SystemMessage {
    pub subtype: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AssistantMessage {
    pub message: Option<AssistantInner>,
}

#[derive(Debug, Deserialize)]
pub struct AssistantInner {
    pub model: Option<String>,
    pub content: Option<Vec<ContentBlock>>,
}

#[derive(Debug, Deserialize)]
pub struct ResultMessage {
    pub result: Option<String>,
    #[serde(rename = "exitCode")]
    pub exit_code: Option<i32>,
    pub duration_ms: Option<u64>,
    pub duration_api_ms: Option<u64>,
    pub num_turns: Option<u64>,
    /// Real stop reason from the CLI envelope (e.g. "end_turn", "max_tokens").
    /// Absent on older CLIs → falls back to "end_turn" downstream.
    pub stop_reason: Option<String>,
    #[serde(rename = "modelUsage")]
    pub model_usage: Option<HashMap<String, ModelUsage>>,
}

/// Per-model usage as emitted inside the CLI's `modelUsage` map.
///
/// The CLI serializes these keys in **camelCase** (`inputTokens`, `outputTokens`,
/// `cacheReadInputTokens`, `cacheCreationInputTokens`, `costUSD`). Without the
/// explicit `#[serde(rename)]` below every field deserialized to `None`, which is
/// why usage reached callers as 0.
#[derive(Debug, Deserialize, PartialEq)]
pub struct ModelUsage {
    #[serde(rename = "inputTokens")]
    pub input_tokens: Option<u64>,
    #[serde(rename = "outputTokens")]
    pub output_tokens: Option<u64>,
    #[serde(rename = "cacheReadInputTokens")]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(rename = "cacheCreationInputTokens")]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(rename = "costUSD")]
    pub cost_usd: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_system_message() {
        let json = r#"{"type":"system","subtype":"init"}"#;
        let msg: ClaudeCliMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClaudeCliMessage::System(s) => assert_eq!(s.subtype, Some("init".to_string())),
            other => panic!("Expected System, got {:?}", other),
        }
    }

    #[test]
    fn deserialize_assistant_with_model() {
        let json = r#"{"type":"assistant","message":{"model":"claude-opus-4-20250514","content":[{"type":"text","text":"Hi"}]}}"#;
        let msg: ClaudeCliMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClaudeCliMessage::Assistant(a) => {
                let inner = a.message.unwrap();
                assert_eq!(inner.model, Some("claude-opus-4-20250514".to_string()));
                let content = inner.content.unwrap();
                assert_eq!(content.len(), 1);
                assert_eq!(content[0].text, Some("Hi".to_string()));
            }
            other => panic!("Expected Assistant, got {:?}", other),
        }
    }

    #[test]
    fn deserialize_assistant_no_message() {
        let json = r#"{"type":"assistant"}"#;
        let msg: ClaudeCliMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClaudeCliMessage::Assistant(a) => assert!(a.message.is_none()),
            other => panic!("Expected Assistant, got {:?}", other),
        }
    }

    #[test]
    fn deserialize_result_full() {
        let json = r#"{
            "type": "result",
            "result": "Done!",
            "exitCode": 0,
            "duration_ms": 5000,
            "duration_api_ms": 4500,
            "num_turns": 3,
            "stop_reason": "end_turn",
            "modelUsage": {
                "claude-opus-4": {
                    "inputTokens": 1000,
                    "outputTokens": 500,
                    "cacheReadInputTokens": 100,
                    "cacheCreationInputTokens": 50
                }
            }
        }"#;
        let msg: ClaudeCliMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClaudeCliMessage::Result(r) => {
                assert_eq!(r.result, Some("Done!".to_string()));
                assert_eq!(r.exit_code, Some(0));
                assert_eq!(r.duration_ms, Some(5000));
                assert_eq!(r.duration_api_ms, Some(4500));
                assert_eq!(r.num_turns, Some(3));
                assert_eq!(r.stop_reason, Some("end_turn".to_string()));
                let usage = r.model_usage.as_ref().unwrap();
                let u = &usage["claude-opus-4"];
                assert_eq!(u.input_tokens, Some(1000));
                assert_eq!(u.output_tokens, Some(500));
                assert_eq!(u.cache_read_input_tokens, Some(100));
                assert_eq!(u.cache_creation_input_tokens, Some(50));
            }
            other => panic!("Expected Result, got {:?}", other),
        }
    }

    #[test]
    fn deserialize_model_usage_camelcase_real() {
        // Exact camelCase shape emitted by the claude CLI envelope (verified
        // against a live run). Before the serde renames every field here
        // deserialized to None, which is why usage reached callers as 0.
        // Unknown extras (contextWindow, maxOutputTokens) are ignored.
        let json = r#"{
            "claude-haiku-4-5-20251001": {
                "inputTokens": 452,
                "outputTokens": 109,
                "cacheReadInputTokens": 17283,
                "cacheCreationInputTokens": 7922,
                "costUSD": 0.0126278,
                "contextWindow": 200000,
                "maxOutputTokens": 32000
            }
        }"#;
        let mu: HashMap<String, ModelUsage> = serde_json::from_str(json).unwrap();
        let u = &mu["claude-haiku-4-5-20251001"];
        assert_eq!(u.input_tokens, Some(452));
        assert_eq!(u.output_tokens, Some(109));
        assert_eq!(u.cache_read_input_tokens, Some(17283));
        assert_eq!(u.cache_creation_input_tokens, Some(7922));
        assert_eq!(u.cost_usd, Some(0.0126278));
    }

    #[test]
    fn deserialize_result_minimal() {
        let json = r#"{"type":"result"}"#;
        let msg: ClaudeCliMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClaudeCliMessage::Result(r) => {
                assert_eq!(r.result, None);
                assert_eq!(r.exit_code, None);
                assert_eq!(r.model_usage, None);
            }
            other => panic!("Expected Result, got {:?}", other),
        }
    }

    #[test]
    fn deserialize_result_multi_model_usage() {
        let json = r#"{
            "type": "result",
            "result": "ok",
            "modelUsage": {
                "claude-opus-4": {"inputTokens": 100, "outputTokens": 50},
                "claude-sonnet-4": {"inputTokens": 200, "outputTokens": 100}
            }
        }"#;
        let msg: ClaudeCliMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClaudeCliMessage::Result(r) => {
                let usage = r.model_usage.as_ref().unwrap();
                assert_eq!(usage.len(), 2);
                assert_eq!(usage["claude-opus-4"].input_tokens, Some(100));
                assert_eq!(usage["claude-sonnet-4"].output_tokens, Some(100));
            }
            other => panic!("Expected Result, got {:?}", other),
        }
    }

    // ── StreamEvent ──────────────────────────────────────────

    #[test]
    fn deserialize_content_block_delta() {
        let json = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, Some(0));
                assert_eq!(delta.delta_type, Some("text_delta".to_string()));
                assert_eq!(delta.text, Some("Hello".to_string()));
            }
            other => panic!("Expected ContentBlockDelta, got {:?}", other),
        }
    }

    #[test]
    fn deserialize_content_block_start() {
        let json = r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                assert_eq!(index, Some(0));
                let block = content_block.unwrap();
                assert_eq!(block.block_type, Some("text".to_string()));
            }
            other => panic!("Expected ContentBlockStart, got {:?}", other),
        }
    }

    #[test]
    fn deserialize_content_block_stop() {
        let json = r#"{"type":"content_block_stop","index":0}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, StreamEvent::ContentBlockStop { .. }));
    }

    #[test]
    fn deserialize_message_start() {
        let json = r#"{"type":"message_start"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, StreamEvent::MessageStart {}));
    }

    #[test]
    fn deserialize_message_delta() {
        let json = r#"{"type":"message_delta"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, StreamEvent::MessageDelta {}));
    }

    #[test]
    fn deserialize_message_stop() {
        let json = r#"{"type":"message_stop"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, StreamEvent::MessageStop {}));
    }
}
