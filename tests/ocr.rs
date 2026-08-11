use std::process::Command;

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;

mod common;
use common::write_mock_tesseract;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_arwe"))
}

fn png_base64(pixels: &[[u8; 3]], w: u32, h: u32) -> String {
    let flat: Vec<u8> = pixels.iter().flatten().copied().collect();
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

fn white(w: u32, h: u32) -> String {
    png_base64(&vec![[255u8, 255, 255]; (w * h) as usize], w, h)
}

const TSV: &str =
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext
5\t1\t1\t1\t1\t1\t10\t20\t100\t30\t90.5\tHello
5\t1\t1\t1\t1\t2\t120\t20\t80\t30\t85.2\tworld
5\t1\t1\t2\t1\t1\t5\t80\t200\t25\t95\tSecond
5\t1\t1\t2\t1\t2\t150\t80\t60\t25\t88\tline";

#[test]
fn ocr_missing_binary_fails_friendly() {
    let b64 = white(300, 200);
    let out = bin()
        .args(["ocr", "--base64", &b64])
        .env("ARWE_TESSERACT_BIN", "C:\\no_such_dir\\tesseract.exe")
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ARWE_TESSERACT_BIN"),
        "expected friendly error mentioning the env var, got: {}",
        stderr
    );
    assert!(stderr.contains("tesseract"), "stderr: {}", stderr);
}

#[test]
fn ocr_json_from_mock_tesseract() {
    let dir = tempfile::tempdir().unwrap();
    let bat = write_mock_tesseract(&dir, TSV);
    let b64 = white(300, 200);
    let out = bin()
        .args(["ocr", "--base64", &b64, "--format", "json"])
        .env("ARWE_TESSERACT_BIN", &bat)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(v["width"], 300);
    assert_eq!(v["height"], 200);
    let lines = v["lines"].as_array().unwrap();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["text"], "Hello world");
    assert_eq!(lines[0]["x"], 10);
    assert_eq!(lines[0]["y"], 20);
    assert_eq!(lines[0]["w"], 190);
    assert_eq!(lines[0]["h"], 30);
    assert_eq!(lines[1]["text"], "Second line");
    assert_eq!(lines[1]["x"], 5);
    assert_eq!(lines[1]["y"], 80);
    assert_eq!(lines[1]["w"], 205);
}

#[test]
fn ocr_plain_from_mock_tesseract() {
    let dir = tempfile::tempdir().unwrap();
    let bat = write_mock_tesseract(&dir, TSV);
    let b64 = white(300, 200);
    let out = bin()
        .args(["ocr", "--base64", &b64])
        .env("ARWE_TESSERACT_BIN", &bat)
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "Hello world\nSecond line\n"
    );
}

#[test]
fn ocr_rejects_invalid_psm() {
    let b64 = white(2, 2);
    let out = bin()
        .args(["ocr", "--base64", &b64, "--psm", "99"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--psm"));
}

#[test]
fn layout_ocr_fills_text_nodes() {
    // Three glyph-like vertical bars on white, like tests/layout.rs.
    let mut pixels = vec![[255u8, 255, 255]; 20 * 20];
    for &x0 in &[2usize, 7, 12] {
        for x in x0..x0 + 2 {
            for y in 5..10 {
                pixels[y * 20 + x] = [0, 0, 0];
            }
        }
    }
    let b64 = png_base64(&pixels, 20, 20);
    let tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext
5\t1\t1\t1\t1\t1\t2\t5\t4\t3\t90.5\tHello
5\t1\t1\t1\t1\t2\t8\t5\t4\t3\t85.2\tworld";
    let dir = tempfile::tempdir().unwrap();
    let bat = write_mock_tesseract(&dir, tsv);
    let out = bin()
        .args(["layout", "--base64", &b64, "--ocr", "--format", "json"])
        .env("ARWE_TESSERACT_BIN", &bat)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value =
        serde_json::from_str(String::from_utf8_lossy(&out.stdout).trim()).unwrap();
    assert_eq!(v["width"], 20);
    assert_eq!(v["height"], 20);
    let text_nodes = collect_text(&v["root"]);
    assert!(
        text_nodes.iter().any(|n| n["text"] == "Hello world"),
        "expected OCR text on a text node, got: {:?}",
        text_nodes
    );
}

fn collect_text(node: &serde_json::Value) -> Vec<&serde_json::Value> {
    let mut out = Vec::new();
    collect_text_rec(node, &mut out);
    out
}

fn collect_text_rec<'a>(node: &'a serde_json::Value, out: &mut Vec<&'a serde_json::Value>) {
    if node["kind"] == "text" {
        out.push(node);
    }
    if let Some(children) = node["children"].as_array() {
        for c in children {
            collect_text_rec(c, out);
        }
    }
}
