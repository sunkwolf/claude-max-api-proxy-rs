use crate::types::openai::{ChatCompletionRequest, Message, MessageContent};
use std::collections::HashMap;

/// Maps OpenAI model names to Claude CLI model aliases
fn model_map() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        ("claude-opus-4", "opus"),
        ("claude-sonnet-4", "sonnet"),
        ("claude-haiku-4", "haiku"),
        ("claude-code-cli/claude-opus-4", "opus"),
        ("claude-code-cli/claude-sonnet-4", "sonnet"),
        ("claude-code-cli/claude-haiku-4", "haiku"),
        ("opus", "opus"),
        ("sonnet", "sonnet"),
        ("haiku", "haiku"),
    ])
}

/// Resolve the model string to pass to the `--model` flag of the CLI.
///
/// Known short aliases and the `claude-code-cli/` prefix are mapped to the CLI's
/// short names. Anything else is passed through **verbatim**: the CLI accepts
/// full ids such as `claude-sonnet-4-6-20250929` and `claude-opus-4-8[1m]`, and
/// substring-degrading them to `opus`/`sonnet`/`haiku` (the old behavior) dropped
/// the exact snapshot and the `[1m]` context-window suffix.
pub fn extract_model(model: &str) -> String {
    let map = model_map();

    if let Some(&alias) = map.get(model) {
        return alias.to_string();
    }

    // Try stripping "claude-code-cli/" prefix
    if let Some(stripped) = model.strip_prefix("claude-code-cli/") {
        if let Some(&alias) = map.get(stripped) {
            return alias.to_string();
        }
    }

    // Unknown / fully-qualified id → pass through unchanged.
    model.to_string()
}

/// Extract text from MessageContent
fn extract_text(content: &Option<MessageContent>) -> String {
    match content {
        Some(MessageContent::Text(s)) => s.clone(),
        Some(MessageContent::Parts(parts)) => parts
            .iter()
            .filter(|p| p.part_type == "text")
            .filter_map(|p| p.text.as_deref())
            .collect::<Vec<_>>()
            .join(""),
        None => String::new(),
    }
}

/// Convert OpenAI messages to a CLI prompt string.
///
/// - System messages are wrapped in `<system>` tags
/// - User messages are included as bare text
/// - Assistant messages are wrapped in `<previous_response>` tags
pub fn messages_to_prompt(messages: &[Message]) -> String {
    let mut parts: Vec<String> = Vec::new();

    for msg in messages {
        let text = extract_text(&msg.content);
        match msg.role.as_str() {
            "system" => {
                parts.push(format!("<system>\n{}\n</system>\n", text));
            }
            "user" => {
                parts.push(text);
            }
            "assistant" => {
                parts.push(format!("<previous_response>\n{}\n</previous_response>\n", text));
            }
            _ => {
                // Treat unknown roles as user messages
                parts.push(text);
            }
        }
    }

    parts.join("\n").trim().to_string()
}

/// Convert an OpenAI request to CLI arguments and prompt.
/// Returns (model_alias, prompt, optional_session_id).
pub fn openai_to_cli(request: &ChatCompletionRequest) -> (String, String, Option<String>) {
    let model = request
        .model
        .as_deref()
        .map(extract_model)
        .unwrap_or_else(|| "opus".to_string());

    let prompt = request
        .messages
        .as_ref()
        .map(|msgs| messages_to_prompt(msgs))
        .unwrap_or_default();

    let session_id = request.user.clone();

    (model, prompt, session_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::openai::ContentPart;

    // ── extract_model ─────────────────────────────────────────

    #[test]
    fn exact_model_names() {
        assert_eq!(extract_model("claude-opus-4"), "opus");
        assert_eq!(extract_model("claude-sonnet-4"), "sonnet");
        assert_eq!(extract_model("claude-haiku-4"), "haiku");
    }

    #[test]
    fn short_aliases() {
        assert_eq!(extract_model("opus"), "opus");
        assert_eq!(extract_model("sonnet"), "sonnet");
        assert_eq!(extract_model("haiku"), "haiku");
    }

    #[test]
    fn prefixed_model_names() {
        assert_eq!(extract_model("claude-code-cli/claude-opus-4"), "opus");
        assert_eq!(extract_model("claude-code-cli/claude-sonnet-4"), "sonnet");
        assert_eq!(extract_model("claude-code-cli/claude-haiku-4"), "haiku");
    }

    #[test]
    fn date_suffixed_model_names_pass_through_verbatim() {
        // Fully-qualified ids are not in the alias map → forwarded verbatim to
        // the CLI (which accepts them). Previously these degraded to short aliases.
        assert_eq!(
            extract_model("claude-opus-4-20250514"),
            "claude-opus-4-20250514"
        );
        assert_eq!(
            extract_model("claude-sonnet-4-5-20250929"),
            "claude-sonnet-4-5-20250929"
        );
        assert_eq!(
            extract_model("claude-haiku-4-5-20251001"),
            "claude-haiku-4-5-20251001"
        );
    }

    #[test]
    fn full_ids_pass_through_verbatim() {
        // The 1M-context suffix and exact snapshots must survive untouched.
        assert_eq!(extract_model("claude-opus-4-8[1m]"), "claude-opus-4-8[1m]");
        assert_eq!(
            extract_model("claude-sonnet-4-6-20250929"),
            "claude-sonnet-4-6-20250929"
        );
    }

    #[test]
    fn unknown_model_passes_through_verbatim() {
        assert_eq!(extract_model("gpt-4"), "gpt-4");
        assert_eq!(extract_model("unknown-model"), "unknown-model");
        assert_eq!(extract_model(""), "");
    }

    // ── messages_to_prompt ────────────────────────────────────

    #[test]
    fn single_user_message() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Some(MessageContent::Text("Hello".to_string())),
        }];
        assert_eq!(messages_to_prompt(&messages), "Hello");
    }

    #[test]
    fn system_message_wrapped_in_tags() {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: Some(MessageContent::Text("You are helpful.".to_string())),
            },
            Message {
                role: "user".to_string(),
                content: Some(MessageContent::Text("Hi".to_string())),
            },
        ];
        let prompt = messages_to_prompt(&messages);
        assert!(prompt.starts_with("<system>\nYou are helpful.\n</system>"));
        assert!(prompt.contains("Hi"));
    }

    #[test]
    fn assistant_message_wrapped_in_previous_response() {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: Some(MessageContent::Text("Hi".to_string())),
            },
            Message {
                role: "assistant".to_string(),
                content: Some(MessageContent::Text("Hello!".to_string())),
            },
            Message {
                role: "user".to_string(),
                content: Some(MessageContent::Text("How are you?".to_string())),
            },
        ];
        let prompt = messages_to_prompt(&messages);
        assert!(prompt.contains("<previous_response>\nHello!\n</previous_response>"));
        assert!(prompt.contains("How are you?"));
    }

    #[test]
    fn multipart_content() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Some(MessageContent::Parts(vec![
                ContentPart {
                    part_type: "text".to_string(),
                    text: Some("Hello ".to_string()),
                },
                ContentPart {
                    part_type: "text".to_string(),
                    text: Some("world".to_string()),
                },
                ContentPart {
                    part_type: "image_url".to_string(),
                    text: None,
                },
            ])),
        }];
        assert_eq!(messages_to_prompt(&messages), "Hello world");
    }

    #[test]
    fn none_content_produces_empty_string() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: None,
        }];
        assert_eq!(messages_to_prompt(&messages), "");
    }

    #[test]
    fn unknown_role_treated_as_user() {
        let messages = vec![Message {
            role: "tool".to_string(),
            content: Some(MessageContent::Text("tool output".to_string())),
        }];
        assert_eq!(messages_to_prompt(&messages), "tool output");
    }

    // ── openai_to_cli ────────────────────────────────────────

    #[test]
    fn openai_to_cli_extracts_all_fields() {
        let request = ChatCompletionRequest {
            model: Some("claude-sonnet-4".to_string()),
            messages: Some(vec![Message {
                role: "user".to_string(),
                content: Some(MessageContent::Text("test".to_string())),
            }]),
            stream: false,
            user: Some("session-123".to_string()),
        };
        let (model, prompt, session_id) = openai_to_cli(&request);
        assert_eq!(model, "sonnet");
        assert_eq!(prompt, "test");
        assert_eq!(session_id, Some("session-123".to_string()));
    }

    #[test]
    fn openai_to_cli_defaults_no_model() {
        let request = ChatCompletionRequest {
            model: None,
            messages: Some(vec![Message {
                role: "user".to_string(),
                content: Some(MessageContent::Text("test".to_string())),
            }]),
            stream: false,
            user: None,
        };
        let (model, _, session_id) = openai_to_cli(&request);
        assert_eq!(model, "opus");
        assert_eq!(session_id, None);
    }

    #[test]
    fn openai_to_cli_no_messages() {
        let request = ChatCompletionRequest {
            model: Some("opus".to_string()),
            messages: None,
            stream: false,
            user: None,
        };
        let (_, prompt, _) = openai_to_cli(&request);
        assert_eq!(prompt, "");
    }
}
