use anyhow::{anyhow, bail, Context, Result};
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartText, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessageArgs,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    CreateChatCompletionRequest, CreateChatCompletionRequestArgs, ImageDetail, ImageUrl,
    ResponseFormat,
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

const HTML_MAX_TOKENS: u32 = 8192;
const JSON_MAX_TOKENS: u32 = 8192;
const DESCRIBE_MAX_TOKENS: u32 = 2048;
const OCR_MAX_TOKENS: u32 = 8192;
const ASK_MAX_TOKENS: u32 = 4096;
/// Layout trees deeper than this are almost certainly a model runaway; reject
/// them instead of risking a stack overflow during post-processing.
const MAX_JSON_DEPTH: usize = 128;

/// Which output shape the vision model should produce.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VisionOutput {
    Html,
    Json,
    Describe,
    Ocr,
}

fn html_system_prompt() -> &'static str {
    "You are a UI layout analyzer for frontend debugging. Given a screenshot, \
reproduce its content as a single complete, static HTML document.\n\
Rules:\n\
- Reproduce the image completely and faithfully: layout, styles, and content exactly as rendered. Do not add any elements that are not visible in the image.\n\
- This is a static description, not an interactive application: no JavaScript and no functional controls. Represent inputs, buttons, and other controls as styled <div> elements.\n\
- Use semantic class names and a <style> block whose CSS (colors, fonts, spacing, borders, sizes) visually recreates the screenshot. Scale every pixel size to the original image dimensions given below.\n\
- Transcribe visible text EXACTLY as rendered, character for character, including punctuation and spacing. Never correct, complete, paraphrase, or infer text. If a character or word is not clearly readable, write [unreadable] in its place. Never invent text that is not visibly present.\n\
- Only describe what you can actually see; never invent sections, buttons, or content.\n\
- After the closing </html> tag, add a blank line and then a natural-language description of the image content in detail: purpose, sections, text, and visual style.\n\
Respond with only the HTML document followed by the description: no markdown fences, no other explanations."
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
- Text nodes MUST carry \"text\": transcribe the visible text EXACTLY, character for character, including punctuation and spacing. Never correct, complete, paraphrase, or infer text; write \"[unreadable]\" for characters or words that are not clearly readable; never invent text that is not visibly present. Text nodes may also carry \"density\" (a number 0..1).\n\
- Only include nodes for elements you can actually see; never invent blocks, groups, or text.\n\
- Nest children inside their parent. Never overlap siblings.\n\
Respond with only the JSON object: no markdown fences, no explanations, no trailing text."
}

fn describe_system_prompt() -> &'static str {
    "You are a UI layout analyzer for frontend debugging. Given a screenshot, \
describe its layout in natural language.\n\
Include:\n\
- The overall purpose of the page or UI, only if it is clear from what is visible.\n\
- The main sections and their hierarchy (e.g. header, navigation, main content, sidebar, footer) and how they are arranged.\n\
- Notable content: headings, buttons, forms, menus, and any text that matters — transcribe visible text exactly, and write [unreadable] for anything not clearly readable.\n\
- Visual style observations: color scheme, spacing, alignment, fonts, borders, shadows, and anything unusual.\n\
- Only describe what you can clearly see. Never invent sections, features, or text that are not visibly present; if text is unreadable, say so instead of guessing.\n\
Respond with a few plain-text paragraphs: no markdown fences, no HTML or JSON, no coordinates."
}

fn ocr_system_prompt() -> &'static str {
    "You are an OCR engine for UI screenshots. Given a screenshot, transcribe \
every visible text region.\n\
Return exactly:\n\
{\"width\":W,\"height\":H,\"lines\":[{\"x\":X,\"y\":Y,\"w\":W,\"h\":H,\"text\":\"...\"}]}\n\
Rules:\n\
- Coordinates are integer pixels; origin 0,0 is the top-left of the image you see.\n\
- Transcribe text EXACTLY as rendered, character for character, including punctuation and spacing. Never correct, complete, paraphrase, or infer text.\n\
- If a character or word is not clearly readable, write [unreadable] in its place. Never invent text that is not visibly present.\n\
- One line per text region, top-to-bottom then left-to-right.\n\
Respond with only the JSON object: no markdown fences, no explanations, no trailing text."
}

/// Build the chat completion request that asks a vision model for a layout
/// description of `data_url` in the requested [`VisionOutput`] shape.
fn build_request(
    data_url: &str,
    width: usize,
    height: usize,
    orig_width: usize,
    orig_height: usize,
    model: &str,
    output: VisionOutput,
) -> Result<CreateChatCompletionRequest> {
    let (system, user_text, max_tokens, json_mode) = match output {
        VisionOutput::Json => (
            json_system_prompt(),
            format!("Image size: {}x{}px. Produce the JSON layout now.", width, height),
            JSON_MAX_TOKENS,
            true,
        ),
        VisionOutput::Html => (
            html_system_prompt(),
            format!(
                "The original image is {}x{}px; the image you see is scaled to fit. Reproduce the layout HTML document at the original pixel size now.",
                orig_width, orig_height
            ),
            HTML_MAX_TOKENS,
            false,
        ),
        VisionOutput::Describe => (
            describe_system_prompt(),
            format!("Image size: {}x{}px. Describe this layout now.", width, height),
            DESCRIBE_MAX_TOKENS,
            false,
        ),
        VisionOutput::Ocr => (
            ocr_system_prompt(),
            format!(
                "Image size: {}x{}px. Transcribe all visible text now.",
                width, height
            ),
            OCR_MAX_TOKENS,
            true,
        ),
    };
    let detail = match output {
        VisionOutput::Html | VisionOutput::Json | VisionOutput::Ocr => Some(ImageDetail::High),
        VisionOutput::Describe => None,
    };
    let system_msg = ChatCompletionRequestMessage::System(
        ChatCompletionRequestSystemMessageArgs::default()
            .content(ChatCompletionRequestSystemMessageContent::Text(
                system.to_string(),
            ))
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
                            detail,
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
    proc: &Processed,
    model: &str,
    base_url: &str,
    api_key: &str,
    output: VisionOutput,
) -> Result<String> {
    let request = build_request(
        data_url,
        proc.image.width,
        proc.image.height,
        proc.orig_width,
        proc.orig_height,
        model,
        output,
    )?;
    send_request(request, base_url, api_key, model).await
}

/// Send a chat-completions request and return the cleaned text answer (fences
/// stripped, whitespace trimmed), printing the raw answer to stderr when
/// `ARWE_DEBUG_VISION=1`.
async fn send_request(
    request: CreateChatCompletionRequest,
    base_url: &str,
    api_key: &str,
    model: &str,
) -> Result<String> {
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

/// Send the current image to a vision model with a fully agent-specified
/// prompt (used by the explore `ask` primitive). A minimal system message
/// provides context; the agent's prompt is sent verbatim as the user message.
pub async fn ask(
    proc: &Processed,
    prompt: &str,
    model: &str,
    base_url: &str,
    api_key: &str,
) -> Result<String> {
    let data_url = crate::ocr::png_data_url(&proc.image)?;
    let request = ask_request(&data_url, model, prompt)?;
    send_request(request, base_url, api_key, model).await
}

fn ask_request(data_url: &str, model: &str, prompt: &str) -> Result<CreateChatCompletionRequest> {
    let system = "You are an image analysis assistant for a text-only agent. \
Analyze the attached image according to the user's request. Answer directly and \
concisely; do not invent content that is not visible in the image.";
    let system_msg = ChatCompletionRequestMessage::System(
        ChatCompletionRequestSystemMessageArgs::default()
            .content(ChatCompletionRequestSystemMessageContent::Text(
                system.to_string(),
            ))
            .build()?,
    );
    let user_msg = ChatCompletionRequestMessage::User(
        ChatCompletionRequestUserMessageArgs::default()
            .content(ChatCompletionRequestUserMessageContent::Array(vec![
                ChatCompletionRequestUserMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText {
                        text: prompt.to_string(),
                    },
                ),
                ChatCompletionRequestUserMessageContentPart::ImageUrl(
                    ChatCompletionRequestMessageContentPartImage {
                        image_url: ImageUrl {
                            url: data_url.to_string(),
                            detail: Some(ImageDetail::High),
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
        .max_tokens(ASK_MAX_TOKENS);
    Ok(args.build()?)
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
    normalize_nodes(&mut v);
    Ok(serde_json::to_string(&v).expect("serializing vision JSON cannot fail"))
}

fn remap_node(v: &mut Value, proc: &Processed) {
    let Value::Object(map) = v else {
        return;
    };
    let keys = ["x", "y", "w", "h"];
    if keys
        .iter()
        .all(|k| map.get(*k).and_then(Value::as_u64).is_some())
    {
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

/// Clamp degenerate boxes (w or h == 0) to 1px so downstream consumers never
/// see zero-size nodes from a hallucinated or sloppy model answer.
fn normalize_nodes(v: &mut Value) {
    let Value::Object(map) = v else {
        return;
    };
    for k in ["w", "h"] {
        if let Some(val) = map.get_mut(k) {
            if let Some(n) = val.as_u64() {
                if n == 0 {
                    *val = Value::from(1u64);
                }
            }
        }
    }
    if let Some(root) = map.get_mut("root") {
        normalize_nodes(root);
    }
    if let Some(Value::Array(children)) = map.get_mut("children") {
        for c in children.iter_mut() {
            normalize_nodes(c);
        }
    }
}

/// Parse the model's OCR JSON answer into local-style
/// [`crate::ocr::OcrLine`]s. Coordinates stay in the image the model saw
/// (processed-image space); rendering maps them back to original pixels.
/// Vision output carries no confidence, so `conf` is 0.0.
pub fn finalize_ocr(raw: &str) -> Result<Vec<crate::ocr::OcrLine>> {
    let mut v = match parse_json_value(raw) {
        Ok(v) => v,
        Err(first_err) => {
            match extract_json_object(raw).and_then(|obj| parse_json_value(obj).ok()) {
                Some(v) => v,
                None => return Err(first_err),
            }
        }
    };
    check_depth(&v, 0)?;
    let lines = v
        .get_mut("lines")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| anyhow!("vision OCR answer has no \"lines\" array"))?;
    let mut out = Vec::with_capacity(lines.len());
    for line in lines {
        let Value::Object(map) = line else {
            continue;
        };
        let Some(x) = map.get("x").and_then(Value::as_u64) else {
            continue;
        };
        let Some(y) = map.get("y").and_then(Value::as_u64) else {
            continue;
        };
        let Some(w) = map.get("w").and_then(Value::as_u64) else {
            continue;
        };
        let Some(h) = map.get("h").and_then(Value::as_u64) else {
            continue;
        };
        let text = map
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        out.push(crate::ocr::OcrLine {
            x: x as usize,
            y: y as usize,
            w: (w.max(1)) as usize,
            h: (h.max(1)) as usize,
            text,
            conf: 0.0,
        });
    }
    Ok(out)
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
        let req = build_request(
            "data:image/png;base64,AAAA",
            320,
            240,
            640,
            480,
            "gpt-4o-mini",
            VisionOutput::Json,
        )
        .unwrap();
        let v: Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["model"], "gpt-4o-mini");
        assert_eq!(v["temperature"], 0.0);
        assert_eq!(v["response_format"]["type"], "json_object");
        assert_eq!(v["max_tokens"], 8192);
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
        let parts = msgs[1]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "text");
        assert!(parts[0]["text"].as_str().unwrap().contains("320x240"));
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,AAAA");
        assert_eq!(parts[1]["image_url"]["detail"], "high");
    }

    #[test]
    fn request_html_mode_has_no_response_format() {
        let req = build_request(
            "data:image/png;base64,AAAA",
            10,
            10,
            1795,
            1242,
            "gpt-4o-mini",
            VisionOutput::Html,
        )
        .unwrap();
        let v: Value = serde_json::to_value(&req).unwrap();
        assert!(v.get("response_format").is_none());
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs[1]["content"][1]["image_url"]["detail"], "high");
        let system = msgs[0]["content"].as_str().unwrap();
        assert!(
            system.contains("complete, static HTML document"),
            "html prompt must ask for a full document: {}",
            system
        );
        assert!(
            system.contains("natural-language description"),
            "html prompt must append a description"
        );
        assert!(system.contains("<style>"), "html prompt must allow CSS");
        let user_text = msgs[1]["content"][0]["text"].as_str().unwrap();
        assert!(
            user_text.contains("original image is 1795x1242px"),
            "html request must target original pixel size: {}",
            user_text
        );
    }

    #[test]
    fn describe_request_is_plain_text_with_token_cap() {
        let req = build_request(
            "data:image/png;base64,AAAA",
            320,
            240,
            640,
            480,
            "gpt-4o-mini",
            VisionOutput::Describe,
        )
        .unwrap();
        let v: Value = serde_json::to_value(&req).unwrap();
        assert!(v.get("response_format").is_none());
        assert_eq!(v["max_tokens"], 2048);
        let msgs = v["messages"].as_array().unwrap();
        assert!(msgs[1]["content"][1]["image_url"]["detail"].is_null());
        let system = msgs[0]["content"].as_str().unwrap();
        assert!(
            system.contains("natural language"),
            "describe prompt: {}",
            system
        );
        assert!(system.contains("no HTML or JSON"));
    }

    #[test]
    fn ocr_request_is_json_mode_with_high_detail() {
        let req = build_request(
            "data:image/png;base64,AAAA",
            320,
            240,
            640,
            480,
            "gpt-4o-mini",
            VisionOutput::Ocr,
        )
        .unwrap();
        let v: Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["response_format"]["type"], "json_object");
        assert_eq!(v["max_tokens"], 8192);
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs[1]["content"][1]["image_url"]["detail"], "high");
        let system = msgs[0]["content"].as_str().unwrap();
        assert!(system.contains("[unreadable]"), "ocr prompt: {}", system);
        assert!(system.contains("Never invent"));
    }

    #[test]
    fn finalize_ocr_parses_lines_and_clamps() {
        let raw = r##"{"width":200,"height":150,"lines":[{"x":10,"y":20,"w":100,"h":30,"text":"Hello"},{"x":5,"y":80,"w":0,"h":20,"text":"[unreadable]"}]}"##;
        let lines = finalize_ocr(raw).unwrap();
        assert_eq!(lines.len(), 2);
        assert_eq!(
            (lines[0].x, lines[0].y, lines[0].w, lines[0].h),
            (10, 20, 100, 30)
        );
        assert_eq!(lines[0].text, "Hello");
        assert_eq!(lines[1].w, 1, "zero-size boxes must be clamped");
        assert_eq!(lines[1].text, "[unreadable]");
    }

    #[test]
    fn finalize_ocr_rejects_missing_lines() {
        assert!(finalize_ocr("{\"width\":1}").is_err());
    }

    #[test]
    fn ask_request_sends_prompt_verbatim() {
        let req = ask_request(
            "data:image/png;base64,AAAA",
            "gpt-4o-mini",
            "what color is the button?",
        )
        .unwrap();
        let v: Value = serde_json::to_value(&req).unwrap();
        assert!(v.get("response_format").is_none());
        assert_eq!(v["max_tokens"], 4096);
        let msgs = v["messages"].as_array().unwrap();
        let system = msgs[0]["content"].as_str().unwrap();
        assert!(
            system.contains("image analysis assistant"),
            "ask must use a minimal system prompt: {}",
            system
        );
        assert_eq!(msgs[1]["content"][0]["text"], "what color is the button?");
        assert_eq!(msgs[1]["content"][1]["image_url"]["detail"], "high");
    }

    #[test]
    fn prompts_forbid_hallucination() {
        for prompt in [
            html_system_prompt(),
            json_system_prompt(),
            describe_system_prompt(),
        ] {
            assert!(
                prompt.contains("[unreadable]"),
                "prompt must define the marker"
            );
            assert!(
                prompt.to_lowercase().contains("never invent"),
                "prompt must forbid fabrication"
            );
        }
        assert!(
            html_system_prompt().contains("no JavaScript"),
            "html prompt must forbid interaction logic"
        );
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
    fn json_clamps_degenerate_nodes() {
        let proc = proc_scale2();
        let raw = r##"{"width":200,"height":150,"root":{"x":0,"y":0,"w":200,"h":150,"color":"#000000","kind":"group","direction":"row","children":[{"x":0,"y":0,"w":0,"h":10,"color":"#ffffff","kind":"block","children":[]}]}}"##;
        let out = finalize_json(raw, &proc).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        let kid = &v["root"]["children"][0];
        assert_eq!(kid["w"], 1);
        assert_eq!(kid["h"], 20);
    }

    #[test]
    fn bare_node_answers_are_wrapped() {
        let proc = proc_scale2();
        let raw =
            r##"{"x":0,"y":0,"w":200,"h":150,"color":"#000000","kind":"block","children":[]}"##;
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
