use anyhow::{anyhow, bail, Context, Result};
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartText, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessageArgs,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    CreateChatCompletionRequest, CreateChatCompletionRequestArgs, ImageUrl, ResponseFormat,
};
use async_openai::Client;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::resize::Processed;

pub const DEFAULT_MODEL: &str = "gpt-4o-mini";
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
/// Set to `1` to print the raw model answer to stderr before post-processing
/// (handy when a model ignores the requested output format).
pub const DEBUG_ENV: &str = "ARWE_DEBUG_VISION";

const HTML_MAX_TOKENS: u32 = 4096;
const JSON_MAX_TOKENS: u32 = 8192;
/// Layout trees deeper than this are almost certainly a model runaway; reject
/// them instead of risking a stack overflow during post-processing.
const MAX_JSON_DEPTH: usize = 128;

fn html_system_prompt() -> &'static str {
    "You are a UI layout analyzer for frontend debugging. Given a screenshot, \
describe its visual layout as a single nested HTML fragment.\n\
Rules:\n\
- Use only <div> and <span> elements; no classes, ids, styles, or attributes other than the data-* ones below.\n\
- Every element MUST carry data-x, data-y, data-w, data-h (integer pixel coordinates; origin 0,0 is the top-left of the image you see) and data-color=\"#rrggbb\" (the dominant color of the region).\n\
- Containers whose children are side by side get data-direction=\"row\"; containers whose children are stacked get data-direction=\"column\".\n\
- Text-like regions become <span> elements; blocks and containers become <div>.\n\
- Nest children inside their parent element. Never overlap siblings.\n\
Respond with only the HTML fragment: no markdown fences, no explanations, no trailing text."
}

fn json_system_prompt() -> &'static str {
    "You are a UI layout analyzer for frontend debugging. Given a screenshot, \
describe its visual layout as a JSON object.\n\
Return exactly:\n\
{\"width\":W,\"height\":H,\"root\":{...}}\n\
where W and H are the image dimensions in pixels and every node is:\n\
{\"x\":X,\"y\":Y,\"w\":W,\"h\":H,\"color\":\"#rrggbb\",\"kind\":\"group\"|\"block\"|\"text\",\"children\":[...]}\n\
Rules:\n\
- Coordinates are integer pixels; origin 0,0 is the top-left of the image you see.\n\
- Groups also carry \"direction\":\"row\" or \"column\".\n\
- Text nodes may carry \"density\" (a number 0..1).\n\
- Nest children inside their parent. Never overlap siblings.\n\
Respond with only the JSON object: no markdown fences, no explanations, no trailing text."
}

/// Build the chat completion request that asks a vision model for a layout
/// description of `data_url`. When `json_mode` is set the model is forced to
/// answer with JSON (older JSON mode, not schema-constrained).
fn build_request(
    data_url: &str,
    width: usize,
    height: usize,
    model: &str,
    json_mode: bool,
) -> Result<CreateChatCompletionRequest> {
    let (system, user_text, max_tokens) = if json_mode {
        (
            json_system_prompt(),
            format!("Image size: {}x{}px. Produce the JSON layout now.", width, height),
            JSON_MAX_TOKENS,
        )
    } else {
        (
            html_system_prompt(),
            format!("Image size: {}x{}px. Produce the layout HTML fragment now.", width, height),
            HTML_MAX_TOKENS,
        )
    };
    let system_msg = ChatCompletionRequestMessage::System(
        ChatCompletionRequestSystemMessageArgs::default()
            .content(ChatCompletionRequestSystemMessageContent::Text(system.to_string()))
            .build()?,
    );
    let user_msg = ChatCompletionRequestMessage::User(
        ChatCompletionRequestUserMessageArgs::default()
            .content(ChatCompletionRequestUserMessageContent::Array(vec![
                ChatCompletionRequestUserMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText { text: user_text },
                ),
                ChatCompletionRequestUserMessageContentPart::ImageUrl(
                    ChatCompletionRequestMessageContentPartImage {
                        image_url: ImageUrl {
                            url: data_url.to_string(),
                            detail: None,
                        },
                    },
                ),
            ]))
            .build()?,
    );
    let mut args = CreateChatCompletionRequestArgs::default();
    args.model(model)
        .messages(vec![system_msg, user_msg])
        .temperature(0.0_f32)
        .max_tokens(max_tokens);
    if json_mode {
        args.response_format(ResponseFormat::JsonObject);
    }
    Ok(args.build()?)
}

/// Call an OpenAI-compatible vision model and return the cleaned text answer
/// (fences stripped, whitespace trimmed).
pub async fn analyze(
    data_url: &str,
    width: usize,
    height: usize,
    model: &str,
    base_url: &str,
    api_key: &str,
    json_mode: bool,
) -> Result<String> {
    let request = build_request(data_url, width, height, model, json_mode)?;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key(api_key),
    );
    let response = client
        .chat()
        .create(request)
        .await
        .context("vision model request failed")?;
    let content = response
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| anyhow!("vision model '{}' returned no text content", model))?;
    if std::env::var_os(DEBUG_ENV).is_some_and(|v| v == "1") {
        eprintln!("[arwe] raw vision response:\n{}", content);
    }
    Ok(strip_fences(&content).trim().to_string())
}

/// Parse the model's JSON with the recursion limit disabled (some models emit
/// deeply nested structures), then guard depth explicitly.
fn parse_json_value(raw: &str) -> Result<Value> {
    let mut de = serde_json::Deserializer::from_str(raw);
    de.disable_recursion_limit();
    Value::deserialize(&mut de).with_context(|| "vision model did not return valid JSON")
}

fn check_depth(v: &Value, depth: usize) -> Result<()> {
    if depth > MAX_JSON_DEPTH {
        bail!(
            "vision model returned a layout nested deeper than {} levels",
            MAX_JSON_DEPTH
        );
    }
    match v {
        Value::Array(items) => {
            for item in items {
                check_depth(item, depth + 1)?;
            }
        }
        Value::Object(map) => {
            for val in map.values() {
                check_depth(val, depth + 1)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Locate the outermost balanced `{...}` object in `raw`, ignoring braces
/// inside JSON strings. Used to salvage answers where the model wrapped the
/// JSON in prose or left trailing junk.
fn extract_json_object(raw: &str) -> Option<&str> {
    let bytes = raw.as_bytes();
    let start = raw.find('{')?;
    let mut depth = 0i64;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
        } else {
            match b {
                b'"' => in_string = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&raw[start..=i]);
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// Remove a leading ```lang ... ``` (or ~~~) markdown fence if present.
pub fn strip_fences(s: &str) -> &str {
    let t = s.trim();
    for marker in ["```", "~~~"] {
        let Some(rest) = t.strip_prefix(marker) else {
            continue;
        };
        let body = if let Some(stripped) = rest.strip_prefix('\n') {
            stripped
        } else if let Some(nl) = rest.find('\n') {
            &rest[nl + 1..]
        } else {
            rest
        };
        let body = body.trim_end();
        let body = match body.rfind('\n') {
            Some(nl) if body[nl + 1..].trim_start().starts_with(marker) => &body[..nl],
            _ => body.strip_suffix(marker).unwrap_or(body),
        };
        return body.trim();
    }
    t
}

/// Turn the model's HTML answer into final output: coordinates are mapped back
/// from the image the model saw to original-image pixels.
pub fn finalize_html(raw: &str, proc: &Processed) -> String {
    crate::html::remap_coords(raw, proc)
}

/// Turn the model's JSON answer into the standard layout JSON shape
/// (`width`/`height`/`truncated`/`root`), with coordinates mapped back to
/// original-image pixels.
pub fn finalize_json(raw: &str, proc: &Processed) -> Result<String> {
    let mut v = match parse_json_value(raw) {
        Ok(v) => v,
        Err(first_err) => {
            // Some models wrap the JSON in prose or trail explanatory text;
            // salvage the outermost balanced object before giving up.
            match extract_json_object(raw).and_then(|obj| parse_json_value(obj).ok()) {
                Some(v) => v,
                None => return Err(first_err),
            }
        }
    };
    check_depth(&v, 0)?;
    if v.get("root").is_some() {
        if let Some(root) = v.get_mut("root") {
            remap_node(root, proc);
        }
        if let Some(obj) = v.as_object_mut() {
            obj.insert("width".into(), Value::from(proc.orig_width));
            obj.insert("height".into(), Value::from(proc.orig_height));
            obj.insert("truncated".into(), Value::Bool(false));
        }
    } else {
        // The model returned a bare node; wrap it into the standard shape.
        remap_node(&mut v, proc);
        let mut obj = Map::new();
        obj.insert("width".into(), Value::from(proc.orig_width));
        obj.insert("height".into(), Value::from(proc.orig_height));
        obj.insert("truncated".into(), Value::Bool(false));
        obj.insert("root".into(), v);
        v = Value::Object(obj);
    }
    Ok(serde_json::to_string(&v).expect("serializing vision JSON cannot fail"))
}

fn remap_node(v: &mut Value, proc: &Processed) {
    let Value::Object(map) = v else {
        return;
    };
    let keys = ["x", "y", "w", "h"];
    if keys.iter().all(|k| map.get(*k).and_then(Value::as_u64).is_some()) {
        let x = map["x"].as_u64().unwrap() as usize;
        let y = map["y"].as_u64().unwrap() as usize;
        let w = map["w"].as_u64().unwrap() as usize;
        let h = map["h"].as_u64().unwrap() as usize;
        let (ox, oy, ow, oh) = proc.to_original_rect(x, y, w, h);
        map.insert("x".into(), Value::from(ox));
        map.insert("y".into(), Value::from(oy));
        map.insert("w".into(), Value::from(ow));
        map.insert("h".into(), Value::from(oh));
    }
    if let Some(Value::Array(children)) = map.get_mut("children") {
        for c in children {
            remap_node(c, proc);
        }
    }
}

#[cfg(test)]
mod unit {
    use super::*;
    use crate::rgb::RgbImage;

    fn proc_scale2() -> Processed {
        let im = RgbImage::new(400, 300, vec![[255u8, 255, 255]; 400 * 300]).unwrap();
        crate::resize::process(&im, None, Some(200), 1).unwrap()
    }

    #[test]
    fn strips_markdown_fences() {
        assert_eq!(strip_fences("```html\n<div></div>\n```"), "<div></div>");
        assert_eq!(strip_fences("```json\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(strip_fences("~~~\nhello\n~~~"), "hello");
        assert_eq!(strip_fences("plain text"), "plain text");
        assert_eq!(strip_fences("```html\n<div></div>```"), "<div></div>");
        assert_eq!(strip_fences("```\n<div></div>\n```"), "<div></div>");
    }

    #[test]
    fn request_serializes_with_image_and_json_mode() {
        let req = build_request("data:image/png;base64,AAAA", 320, 240, "gpt-4o-mini", true)
            .unwrap();
        let v: Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["model"], "gpt-4o-mini");
        assert_eq!(v["temperature"], 0.0);
        assert_eq!(v["response_format"]["type"], "json_object");
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
        let parts = msgs[1]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "text");
        assert!(parts[0]["text"]
            .as_str()
            .unwrap()
            .contains("320x240"));
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(
            parts[1]["image_url"]["url"],
            "data:image/png;base64,AAAA"
        );
    }

    #[test]
    fn request_html_mode_has_no_response_format() {
        let req = build_request("data:image/png;base64,AAAA", 10, 10, "gpt-4o-mini", false)
            .unwrap();
        let v: Value = serde_json::to_value(&req).unwrap();
        assert!(v.get("response_format").is_none());
    }

    #[test]
    fn json_answers_are_remapped_and_wrapped() {
        let proc = proc_scale2();
        let raw = r##"{"root":{"x":0,"y":0,"w":200,"h":150,"color":"#000000","kind":"group","direction":"row","children":[{"x":0,"y":0,"w":100,"h":150,"color":"#ff0000","kind":"block","children":[]},{"x":100,"y":0,"w":100,"h":150,"color":"#0000ff","kind":"block","children":[]}]}}"##;
        let out = finalize_json(raw, &proc).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["width"], 400);
        assert_eq!(v["height"], 300);
        assert_eq!(v["truncated"], false);
        assert_eq!(v["root"]["w"], 400);
        assert_eq!(v["root"]["h"], 300);
        let kids = v["root"]["children"].as_array().unwrap();
        assert_eq!(kids[0]["x"], 0);
        assert_eq!(kids[0]["w"], 200);
        assert_eq!(kids[1]["x"], 200);
        assert_eq!(kids[1]["w"], 200);
    }

    #[test]
    fn bare_node_answers_are_wrapped() {
        let proc = proc_scale2();
        let raw = r##"{"x":0,"y":0,"w":200,"h":150,"color":"#000000","kind":"block","children":[]}"##;
        let v: Value = serde_json::from_str(&finalize_json(raw, &proc).unwrap()).unwrap();
        assert_eq!(v["root"]["w"], 400);
        assert_eq!(v["width"], 400);
        assert_eq!(v["height"], 300);
    }

    #[test]
    fn invalid_json_answers_error() {
        let proc = proc_scale2();
        assert!(finalize_json("not json", &proc).is_err());
    }

    #[test]
    fn prose_wrapped_json_is_salvaged() {
        let proc = proc_scale2();
        let raw = "Here is the layout:\n```json\n{\"x\":0,\"y\":0,\"w\":200,\"h\":150,\"color\":\"#000000\",\"kind\":\"block\",\"children\":[]}\n```\nHope this helps!";
        let out = finalize_json(raw, &proc).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["root"]["w"], 400);
    }

    #[test]
    fn trailing_junk_json_is_salvaged() {
        let proc = proc_scale2();
        let raw = "{\"x\":0,\"y\":0,\"w\":200,\"h\":150,\"color\":\"#000000\",\"kind\":\"block\",\"children\":[]} and that is all";
        let out = finalize_json(raw, &proc).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["root"]["w"], 400);
    }

    #[test]
    fn extract_ignores_braces_inside_strings() {
        let raw = "prefix {\"text\":\"a{b}c\",\"children\":[{\"x\":1}]} suffix";
        assert_eq!(
            extract_json_object(raw).unwrap(),
            "{\"text\":\"a{b}c\",\"children\":[{\"x\":1}]}"
        );
        assert_eq!(extract_json_object("no braces here"), None);
    }

    #[test]
    fn overly_deep_json_is_rejected_clearly() {
        let mut v = serde_json::Value::Null;
        for _ in 0..300 {
            v = serde_json::json!({ "children": [v] });
        }
        let raw = serde_json::to_string(&v).unwrap();
        let proc = proc_scale2();
        let err = finalize_json(&raw, &proc).unwrap_err().to_string();
        assert!(err.contains("deeper than"), "{}", err);
    }

    #[test]
    fn reasonably_deep_json_still_works() {
        let mut v = serde_json::json!({
            "x": 0, "y": 0, "w": 200, "h": 150,
            "color": "#000000", "kind": "block", "children": []
        });
        for _ in 0..20 {
            v = serde_json::json!({
                "x": 0, "y": 0, "w": 200, "h": 150,
                "color": "#000000", "kind": "block", "children": [v]
            });
        }
        let raw = serde_json::to_string(&v).unwrap();
        let proc = proc_scale2();
        let out = finalize_json(&raw, &proc).unwrap();
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["root"]["w"], 400);
    }
}
