use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::Command;
use std::thread;

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_arwe"))
}

fn png_base64(w: u32, h: u32) -> String {
    let flat = vec![255u8; (w * h * 3) as usize];
    let buf = image::RgbImage::from_raw(w, h, flat).unwrap();
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(buf)
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .unwrap();
    STANDARD.encode(&bytes)
}

/// A minimal OpenAI-compatible chat completions mock. Returns `body` for the
/// first request and exits.
fn spawn_mock(body: String) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        // Read the full request (headers + body) before responding so the
        // client never sees a premature close.
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        let header_end = loop {
            let n = stream.read(&mut tmp).unwrap();
            assert!(n > 0, "client closed before sending headers");
            buf.extend_from_slice(&tmp[..n]);
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break pos + 4;
            }
        };
        let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
        let content_length: usize = head
            .lines()
            .find_map(|l| {
                let (k, v) = l.split_once(':')?;
                if k.eq_ignore_ascii_case("content-length") {
                    v.trim().parse().ok()
                } else {
                    None
                }
            })
            .unwrap_or(0);
        while buf.len() < header_end + content_length {
            let n = stream.read(&mut tmp).unwrap();
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
        }
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(resp.as_bytes()).unwrap();
    });
    (format!("http://{}/v1", addr), handle)
}

fn chat_response(content: &str) -> String {
    serde_json::json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }]
    })
    .to_string()
}

#[test]
fn vision_layout_html_remaps_coordinates() {
    let b64 = png_base64(4, 4);
    let content = "```html\n<div data-x=\"0\" data-y=\"0\" data-w=\"2\" data-h=\"2\" data-color=\"#000000\" data-direction=\"row\">\n  <div data-x=\"0\" data-y=\"0\" data-w=\"1\" data-h=\"2\" data-color=\"#ff0000\"></div>\n  <div data-x=\"1\" data-y=\"0\" data-w=\"1\" data-h=\"2\" data-color=\"#0000ff\"></div>\n</div>\n```";
    let (url, handle) = spawn_mock(chat_response(content));
    let out = bin()
        .args([
            "layout",
            "--engine",
            "vision",
            "--base-url",
            &url,
            "--api-key",
            "test",
            "--format",
            "html",
            "--base64",
            &b64,
            "--width",
            "2",
        ])
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !stdout.contains("```"),
        "fences must be stripped: {}",
        stdout
    );
    assert!(
        stdout.contains("data-x=\"0\" data-y=\"0\" data-w=\"4\" data-h=\"4\""),
        "root box not remapped: {}",
        stdout
    );
    assert!(
        stdout.contains("data-x=\"0\" data-y=\"0\" data-w=\"2\" data-h=\"4\""),
        "left column not remapped: {}",
        stdout
    );
    assert!(
        stdout.contains("data-x=\"2\" data-y=\"0\" data-w=\"2\" data-h=\"4\""),
        "right column not remapped: {}",
        stdout
    );
    handle.join().unwrap();
}

#[test]
fn vision_layout_defaults_to_html() {
    let b64 = png_base64(2, 2);
    let content =
        "<div data-x=\"0\" data-y=\"0\" data-w=\"2\" data-h=\"2\" data-color=\"#000000\"></div>";
    let (url, handle) = spawn_mock(chat_response(content));
    let out = bin()
        .args([
            "layout",
            "--engine",
            "vision",
            "--base-url",
            &url,
            "--api-key",
            "test",
            "--base64",
            &b64,
        ])
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("<div"));
    handle.join().unwrap();
}

#[test]
fn vision_layout_json_remaps_coordinates() {
    let b64 = png_base64(4, 4);
    let content = "{\"width\":2,\"height\":2,\"root\":{\"x\":0,\"y\":0,\"w\":2,\"h\":2,\"color\":\"#000000\",\"kind\":\"group\",\"direction\":\"row\",\"children\":[{\"x\":0,\"y\":0,\"w\":1,\"h\":2,\"color\":\"#ff0000\",\"kind\":\"block\",\"children\":[]},{\"x\":1,\"y\":0,\"w\":1,\"h\":2,\"color\":\"#0000ff\",\"kind\":\"block\",\"children\":[]}]}}";
    let (url, handle) = spawn_mock(chat_response(content));
    let out = bin()
        .args([
            "layout",
            "--engine",
            "vision",
            "--base-url",
            &url,
            "--api-key",
            "test",
            "--format",
            "json",
            "--base64",
            &b64,
            "--width",
            "2",
        ])
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value =
        serde_json::from_str(String::from_utf8_lossy(&out.stdout).trim()).unwrap();
    assert_eq!(v["width"], 4);
    assert_eq!(v["height"], 4);
    assert_eq!(v["truncated"], false);
    assert_eq!(v["root"]["w"], 4);
    assert_eq!(v["root"]["h"], 4);
    let kids = v["root"]["children"].as_array().unwrap();
    assert_eq!(kids[0]["x"], 0);
    assert_eq!(kids[0]["w"], 2);
    assert_eq!(kids[1]["x"], 2);
    assert_eq!(kids[1]["w"], 2);
    handle.join().unwrap();
}

#[test]
fn vision_layout_html_keeps_text_and_styles() {
    let b64 = png_base64(4, 4);
    let content = "```html\n<div data-x=\"0\" data-y=\"0\" data-w=\"2\" data-h=\"2\" data-color=\"#000000\" data-direction=\"row\" style=\"display:flex; gap:1px\">\n  <div data-x=\"0\" data-y=\"0\" data-w=\"1\" data-h=\"2\" data-color=\"#ff0000\" style=\"flex:1\"></div>\n  <span data-x=\"1\" data-y=\"0\" data-w=\"1\" data-h=\"2\" data-color=\"#0000ff\" style=\"font-size:7px\">OK</span>\n</div>\n```";
    let (url, handle) = spawn_mock(chat_response(content));
    let out = bin()
        .args([
            "layout",
            "--engine",
            "vision",
            "--base-url",
            &url,
            "--api-key",
            "test",
            "--format",
            "html",
            "--base64",
            &b64,
            "--width",
            "2",
        ])
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("style=\"display:flex; gap:2px\""),
        "style must be preserved and px scaled: {}",
        stdout
    );
    assert!(
        stdout.contains("style=\"flex:1\""),
        "non-px style values must be untouched: {}",
        stdout
    );
    assert!(
        stdout.contains("style=\"font-size:14px\""),
        "font-size must scale: {}",
        stdout
    );
    assert!(
        stdout.contains(">OK</span>"),
        "recognized text must be kept in the span body: {}",
        stdout
    );
    assert!(
        stdout.contains("data-x=\"2\" data-y=\"0\" data-w=\"2\" data-h=\"4\""),
        "span box not remapped: {}",
        stdout
    );
    handle.join().unwrap();
}

#[test]
fn vision_layout_normalizes_foreign_tags_and_keeps_markers() {
    let b64 = png_base64(4, 4);
    let content = "<div data-x=\"0\" data-y=\"0\" data-w=\"2\" data-h=\"2\" data-color=\"#000000\">\n  <input data-x=\"0\" data-y=\"0\" data-w=\"1\" data-h=\"1\" data-color=\"#ffffff\" style=\"font-size:7px\"/>\n  <span data-x=\"1\" data-y=\"0\" data-w=\"1\" data-h=\"1\" data-color=\"#ffffff\">[unreadable]</span>\n</div>";
    let (url, handle) = spawn_mock(chat_response(content));
    let out = bin()
        .args([
            "layout",
            "--engine",
            "vision",
            "--base-url",
            &url,
            "--api-key",
            "test",
            "--format",
            "html",
            "--base64",
            &b64,
            "--width",
            "2",
        ])
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("<div data-x=\"0\" data-y=\"0\" data-w=\"2\" data-h=\"2\" data-color=\"#ffffff\" style=\"font-size:14px\" />"),
        "input must be normalized to a remapped div: {}",
        stdout
    );
    assert!(
        !stdout.contains("<input"),
        "no input tag may remain: {}",
        stdout
    );
    assert!(
        stdout.contains(">[unreadable]</span>"),
        "unreadable marker must survive: {}",
        stdout
    );
    handle.join().unwrap();
}

#[test]
fn vision_layout_full_document_passes_through_with_description() {
    let b64 = png_base64(2, 2);
    let content = "```html\n<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n<style>\n  body { background:#000; }\n</style>\n</head>\n<body>\n  <div class=\"app\">你好</div>\n</body>\n</html>\n\nThis is a dark-themed app window with a single greeting.\n```";
    let (url, handle) = spawn_mock(chat_response(content));
    let out = bin()
        .args([
            "layout",
            "--engine",
            "vision",
            "--base-url",
            &url,
            "--api-key",
            "test",
            "--format",
            "html",
            "--base64",
            &b64,
        ])
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !stdout.contains("```"),
        "fences must be stripped: {}",
        stdout
    );
    assert!(stdout.contains("<!DOCTYPE html>"), "document must survive");
    assert!(stdout.contains("<style>"), "css must survive");
    assert!(
        stdout.contains("<div class=\"app\">你好</div>"),
        "document body must survive: {}",
        stdout
    );
    assert!(
        stdout.contains("This is a dark-themed app window with a single greeting."),
        "trailing natural-language description must survive: {}",
        stdout
    );
    handle.join().unwrap();
}

#[test]
fn vision_layout_json_keeps_recognized_text() {
    let b64 = png_base64(4, 4);
    let content = "{\"width\":2,\"height\":2,\"root\":{\"x\":0,\"y\":0,\"w\":2,\"h\":2,\"color\":\"#000000\",\"kind\":\"group\",\"direction\":\"row\",\"children\":[{\"x\":0,\"y\":0,\"w\":1,\"h\":2,\"color\":\"#ff0000\",\"kind\":\"block\",\"children\":[]},{\"x\":1,\"y\":0,\"w\":1,\"h\":2,\"color\":\"#0000ff\",\"kind\":\"text\",\"density\":0.7,\"text\":\"Login\",\"children\":[]}]}}";
    let (url, handle) = spawn_mock(chat_response(content));
    let out = bin()
        .args([
            "layout",
            "--engine",
            "vision",
            "--base-url",
            &url,
            "--api-key",
            "test",
            "--format",
            "json",
            "--base64",
            &b64,
            "--width",
            "2",
        ])
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    let kids = v["root"]["children"].as_array().unwrap();
    assert_eq!(kids[1]["text"], "Login");
    assert_eq!(kids[1]["x"], 2);
    assert_eq!(kids[1]["w"], 2);
    handle.join().unwrap();
}

#[test]
fn vision_layout_describe_prints_plain_text() {
    let b64 = png_base64(2, 2);
    let content = "```text\nA centered login card with a header, two input fields, and a primary button.\n```";
    let (url, handle) = spawn_mock(chat_response(content));
    let out = bin()
        .args([
            "layout",
            "--engine",
            "vision",
            "--base-url",
            &url,
            "--api-key",
            "test",
            "--format",
            "describe",
            "--base64",
            &b64,
        ])
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !stdout.contains("```"),
        "fences must be stripped: {}",
        stdout
    );
    assert!(
        stdout.contains("A centered login card with a header"),
        "describe answer must be printed: {}",
        stdout
    );
    handle.join().unwrap();
}

#[test]
fn vision_mode_rejects_local_only_options() {
    let b64 = png_base64(2, 2);
    let url = "http://127.0.0.1:1/v1";
    let out = bin()
        .args([
            "layout",
            "--engine",
            "vision",
            "--base-url",
            url,
            "--api-key",
            "x",
            "--format",
            "ascii",
            "--base64",
            &b64,
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("ascii"));

    let out = bin()
        .args([
            "layout",
            "--engine",
            "vision",
            "--base-url",
            url,
            "--api-key",
            "x",
            "--ocr",
            "--base64",
            &b64,
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--ocr"));

    let out = bin()
        .args([
            "layout", "--engine", "local", "--format", "describe", "--base64", &b64,
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("describe"));
}

#[test]
fn vision_mode_requires_api_key() {
    let b64 = png_base64(2, 2);
    // Isolate from any real user config: point at an empty temp config so no
    // profile can supply an api key.
    let dir = tempfile::tempdir().unwrap();
    let empty_cfg = dir.path().join("empty.toml");
    let out = bin()
        .args(["layout", "--engine", "vision", "--base64", &b64])
        .env("ARWE_CONFIG", &empty_cfg)
        .env_remove("ARWE_PROFILE")
        .env_remove("OPENAI_API_KEY")
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("OPENAI_API_KEY"), "stderr: {}", stderr);
}

#[test]
fn vision_ocr_plain_from_mock() {
    let b64 = png_base64(4, 4);
    let content = "{\"width\":2,\"height\":2,\"lines\":[{\"x\":0,\"y\":0,\"w\":2,\"h\":1,\"text\":\"Login\"},{\"x\":0,\"y\":1,\"w\":2,\"h\":1,\"text\":\"[unreadable]\"}]}";
    let (url, handle) = spawn_mock(chat_response(content));
    let out = bin()
        .args([
            "ocr",
            "--engine",
            "vision",
            "--base-url",
            &url,
            "--api-key",
            "test",
            "--format",
            "plain",
            "--base64",
            &b64,
            "--width",
            "2",
        ])
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(stdout, "Login\n[unreadable]\n");
    handle.join().unwrap();
}

#[test]
fn vision_ocr_json_from_mock() {
    let b64 = png_base64(4, 4);
    let content = "{\"width\":2,\"height\":2,\"lines\":[{\"x\":0,\"y\":0,\"w\":2,\"h\":1,\"text\":\"Login\"}]}";
    let (url, handle) = spawn_mock(chat_response(content));
    let out = bin()
        .args([
            "ocr",
            "--engine",
            "vision",
            "--base-url",
            &url,
            "--api-key",
            "test",
            "--format",
            "json",
            "--base64",
            &b64,
            "--width",
            "2",
        ])
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(v["width"], 4);
    assert_eq!(v["height"], 4);
    assert_eq!(v["lines"][0]["x"], 0);
    assert_eq!(v["lines"][0]["w"], 4);
    assert_eq!(v["lines"][0]["text"], "Login");
    assert!(v["lines"][0].get("conf").is_none());
    handle.join().unwrap();
}

#[test]
fn explore_ask_returns_vision_answer() {
    let b64 = png_base64(2, 2);
    let content = "```text\nThe button is blue.\n```";
    let (url, handle) = spawn_mock(chat_response(content));
    let mut child = bin()
        .args([
            "explore",
            "--engine",
            "vision",
            "--base-url",
            &url,
            "--api-key",
            "test",
            "--base64",
            &b64,
        ])
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"ask --width 2 what color is the button?\nquit\n")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("The button is blue."),
        "ask answer must be printed: {}",
        stdout
    );
    assert!(
        !stdout.contains("```"),
        "fences must be stripped: {}",
        stdout
    );
    handle.join().unwrap();
}

#[test]
fn explore_ask_requires_vision_engine() {
    let b64 = png_base64(2, 2);
    let mut child = bin()
        .args(["explore", "--engine", "convocr", "--base64", &b64])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"ask hello\nquit\n")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "session must survive errors");
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        stderr.contains("requires the vision engine"),
        "stderr: {}",
        stderr
    );
}

#[test]
fn vision_layout_uses_selected_profile() {
    let b64 = png_base64(2, 2);
    let content =
        "<div data-x=\"0\" data-y=\"0\" data-w=\"2\" data-h=\"2\" data-color=\"#000000\"></div>";
    let (url, handle) = spawn_mock(chat_response(content));
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("config.toml");
    std::fs::write(
        &cfg,
        format!(
            "[profiles.local]\nbase_url = \"{}\"\nmodel = \"mock-model\"\napi_key = \"cfg-key\"\n",
            url
        ),
    )
    .unwrap();
    let out = bin()
        .args([
            "layout",
            "--engine",
            "vision",
            "--profile",
            "local",
            "--base64",
            &b64,
        ])
        .env("ARWE_CONFIG", &cfg)
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("<div"));
    handle.join().unwrap();
}

#[test]
fn vision_layout_uses_default_profile() {
    let b64 = png_base64(2, 2);
    let content =
        "<div data-x=\"0\" data-y=\"0\" data-w=\"2\" data-h=\"2\" data-color=\"#000000\"></div>";
    let (url, handle) = spawn_mock(chat_response(content));
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("config.toml");
    std::fs::write(
        &cfg,
        format!(
            "default_profile = \"local\"\n\n[profiles.local]\nbase_url = \"{}\"\napi_key = \"cfg-key\"\n",
            url
        ),
    )
    .unwrap();
    let out = bin()
        .args(["layout", "--engine", "vision", "--base64", &b64])
        .env("ARWE_CONFIG", &cfg)
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("<div"));
    handle.join().unwrap();
}
