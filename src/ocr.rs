use std::io::Write as _;
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use serde_json::{Map, Value};

use crate::layout::{Node, NodeKind};
use crate::resize::Processed;
use crate::rgb::RgbImage;

/// The Tesseract executable is looked up from `ARWE_TESSERACT_BIN` when set,
/// otherwise `tesseract` must be on PATH.
pub const TESSERACT_BIN_ENV: &str = "ARWE_TESSERACT_BIN";

/// A single OCR line in processed-image pixel coordinates.
#[derive(Clone, Debug)]
pub struct OcrLine {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    pub text: String,
    pub conf: f32,
}

/// Run Tesseract on `img` (PNG round-trip through a temp file) and return the
/// recognized lines in image pixel coordinates, sorted top-to-bottom then
/// left-to-right.
pub fn recognize(img: &RgbImage, lang: &str, psm: u32) -> Result<Vec<OcrLine>> {
    let bin = std::env::var_os(TESSERACT_BIN_ENV)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("tesseract"));

    let png = png_bytes(img)?;
    let mut tmp = tempfile::Builder::new()
        .prefix("arwe-ocr-")
        .suffix(".png")
        .tempfile()
        .context("failed to create temp file for OCR input")?;
    tmp.write_all(&png)
        .context("failed to write OCR temp image")?;
    tmp.flush().context("failed to flush OCR temp image")?;
    let input = tmp.path().to_path_buf();

    let output = Command::new(&bin)
        .arg(&input)
        .arg("stdout")
        .arg("-l")
        .arg(lang)
        .arg("--psm")
        .arg(psm.to_string())
        .arg("tsv")
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                anyhow!(
                    "could not run Tesseract ('{}'): executable not found. \
                     Install Tesseract (https://github.com/UB-Mannheim/tesseract/wiki) or point \
                     {} at the full path to tesseract(.exe)",
                    bin.display(),
                    TESSERACT_BIN_ENV
                )
            } else {
                anyhow!("failed to spawn Tesseract ('{}'): {}", bin.display(), e)
            }
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "tesseract exited with {}: {}",
            output.status,
            stderr.trim().chars().take(400).collect::<String>()
        );
    }
    let tsv = String::from_utf8_lossy(&output.stdout);
    let mut lines = parse_tsv(&tsv);
    lines.sort_by_key(|l| (l.y, l.x));
    Ok(lines)
}

fn png_bytes(img: &RgbImage) -> Result<Vec<u8>> {
    let flat: Vec<u8> = img.pixels.iter().flatten().copied().collect();
    let buf = image::RgbImage::from_raw(img.width as u32, img.height as u32, flat)
        .context("image dimensions do not match pixel count")?;
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(buf)
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .context("failed to encode image as PNG")?;
    Ok(bytes)
}

/// Build a `data:image/png;base64,...` URL from an image (used for vision APIs).
pub fn png_data_url(img: &RgbImage) -> Result<String> {
    let bytes = png_bytes(img)?;
    Ok(format!("data:image/png;base64,{}", STANDARD.encode(&bytes)))
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct LineKey {
    block: u32,
    par: u32,
    line: u32,
}

struct Word {
    key: LineKey,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    conf: f32,
    text: String,
}

/// Parse Tesseract TSV (word-level rows, `level == 5`) into lines by grouping
/// words that share block/paragraph/line numbers.
fn parse_tsv(tsv: &str) -> Vec<OcrLine> {
    let mut words: Vec<Word> = Vec::new();
    for (i, raw) in tsv.lines().enumerate() {
        if i == 0 {
            continue; // header
        }
        let cols: Vec<&str> = raw.split('\t').collect();
        if cols.len() < 12 {
            continue;
        }
        let level: i32 = match cols[0].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if level != 5 {
            continue;
        }
        let text = cols[11].trim();
        if text.is_empty() {
            continue;
        }
        let Some(x) = cols[6].parse::<usize>().ok() else {
            continue;
        };
        let Some(y) = cols[7].parse::<usize>().ok() else {
            continue;
        };
        let Some(w) = cols[8].parse::<usize>().ok() else {
            continue;
        };
        let Some(h) = cols[9].parse::<usize>().ok() else {
            continue;
        };
        let conf: f32 = cols[10].parse().unwrap_or(0.0);
        let key = LineKey {
            block: cols[2].parse().unwrap_or(0),
            par: cols[3].parse().unwrap_or(0),
            line: cols[4].parse().unwrap_or(0),
        };
        words.push(Word {
            key,
            x,
            y,
            w,
            h,
            conf,
            text: text.to_string(),
        });
    }

    // Group words by line key, preserving first-seen order.
    let mut groups: Vec<(LineKey, Vec<&Word>)> = Vec::new();
    for word in &words {
        match groups.iter_mut().find(|(k, _)| *k == word.key) {
            Some((_, list)) => list.push(word),
            None => groups.push((word.key, vec![word])),
        }
    }

    groups
        .into_iter()
        .map(|(_, list)| {
            let mut x0 = usize::MAX;
            let mut y0 = usize::MAX;
            let mut x1 = 0usize;
            let mut y1 = 0usize;
            let mut conf_sum = 0.0f32;
            let mut texts = Vec::with_capacity(list.len());
            for w in &list {
                x0 = x0.min(w.x);
                y0 = y0.min(w.y);
                x1 = x1.max(w.x + w.w);
                y1 = y1.max(w.y + w.h);
                conf_sum += w.conf;
                texts.push(w.text.clone());
            }
            OcrLine {
                x: x0,
                y: y0,
                w: x1 - x0,
                h: y1 - y0,
                text: texts.join(" "),
                conf: conf_sum / list.len() as f32,
            }
        })
        .collect()
}

/// Plain output: one line of text per OCR line.
pub fn render_plain(lines: &[OcrLine]) -> String {
    let mut s = String::new();
    for l in lines {
        s.push_str(l.text.trim());
        s.push('\n');
    }
    s
}

/// JSON output with boxes mapped back to original-image pixel coordinates:
/// `{"width","height","lines":[{"x","y","w","h","text","conf"}]}`.
pub fn render_json(proc: &Processed, lines: &[OcrLine]) -> String {
    let mut obj = Map::new();
    obj.insert("width".into(), Value::from(proc.orig_width));
    obj.insert("height".into(), Value::from(proc.orig_height));
    let arr: Vec<Value> = lines
        .iter()
        .map(|l| {
            let (x, y, w, h) = proc.to_original_rect(l.x, l.y, l.w, l.h);
            let mut o = Map::new();
            o.insert("x".into(), Value::from(x));
            o.insert("y".into(), Value::from(y));
            o.insert("w".into(), Value::from(w));
            o.insert("h".into(), Value::from(h));
            o.insert("text".into(), Value::String(l.text.trim().to_string()));
            o.insert(
                "conf".into(),
                Value::from((l.conf as f64 * 10.0).round() / 10.0),
            );
            Value::Object(o)
        })
        .collect();
    obj.insert("lines".into(), Value::Array(arr));
    serde_json::to_string(&Value::Object(obj)).expect("serializing OCR output cannot fail")
}

/// JSON output for vision OCR: same shape as [`render_json`] but without the
/// `conf` field, since vision models do not report confidence.
pub fn render_json_vision(proc: &Processed, lines: &[OcrLine]) -> String {
    let mut obj = Map::new();
    obj.insert("width".into(), Value::from(proc.orig_width));
    obj.insert("height".into(), Value::from(proc.orig_height));
    let arr: Vec<Value> = lines
        .iter()
        .map(|l| {
            let (x, y, w, h) = proc.to_original_rect(l.x, l.y, l.w, l.h);
            let mut o = Map::new();
            o.insert("x".into(), Value::from(x));
            o.insert("y".into(), Value::from(y));
            o.insert("w".into(), Value::from(w));
            o.insert("h".into(), Value::from(h));
            o.insert("text".into(), Value::String(l.text.trim().to_string()));
            Value::Object(o)
        })
        .collect();
    obj.insert("lines".into(), Value::Array(arr));
    serde_json::to_string(&Value::Object(obj)).expect("serializing OCR output cannot fail")
}

/// Attach OCR lines to the layout tree. Each line is matched to the smallest
/// node containing its center: text nodes accumulate their text, other nodes
/// get a new text child so nothing the OCR engine found is dropped.
pub fn fill_text_nodes(root: &mut Node, lines: &[OcrLine]) {
    for line in lines {
        if line.text.trim().is_empty() || line.conf < 10.0 {
            continue;
        }
        fill_line(root, line);
    }
}

fn fill_line(node: &mut Node, line: &OcrLine) {
    let cx = line.x + line.w / 2;
    let cy = line.y + line.h / 2;
    for child in node.children.iter_mut() {
        if contains_point(child, cx, cy) {
            fill_line(child, line);
            return;
        }
    }
    match node.kind {
        NodeKind::Text => {
            let text = node.text.get_or_insert_with(String::new);
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(line.text.trim());
        }
        _ => {
            let color = node.color;
            node.children.push(Node {
                x: line.x,
                y: line.y,
                w: line.w,
                h: line.h,
                kind: NodeKind::Text,
                color,
                density: None,
                text: Some(line.text.trim().to_string()),
                children: Vec::new(),
            });
        }
    }
}

fn contains_point(node: &Node, x: usize, y: usize) -> bool {
    node.x <= x && x < node.x + node.w && node.y <= y && y < node.y + node.h
}

#[cfg(test)]
mod unit {
    use super::*;

    fn sample_tsv() -> String {
        [
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
            "5\t1\t1\t1\t1\t1\t10\t20\t100\t30\t90.5\tHello",
            "5\t1\t1\t1\t1\t2\t120\t20\t80\t30\t85.2\tworld",
            "5\t1\t1\t2\t1\t1\t5\t80\t200\t25\t95\tSecond",
            "5\t1\t1\t2\t1\t2\t150\t80\t60\t25\t88\tline",
            "4\t1\t1\t1\t1\t0\t10\t20\t190\t30\t87.9\tHello world",
            "3\t1\t1\t1\t0\t0\t10\t20\t190\t30\t87.9\tHello world",
            "5\t1\t2\t1\t1\t1\t300\t300\t50\t20\t-1\t",
        ]
        .join("\n")
    }

    #[test]
    fn parses_and_groups_tsv_lines() {
        let lines = parse_tsv(&sample_tsv());
        assert_eq!(lines.len(), 2);
        let a = &lines[0];
        assert_eq!((a.x, a.y, a.w, a.h), (10, 20, 190, 30));
        assert_eq!(a.text, "Hello world");
        assert!((a.conf - 87.85).abs() < 1e-3);
        let b = &lines[1];
        assert_eq!((b.x, b.y, b.w, b.h), (5, 80, 205, 25));
        assert_eq!(b.text, "Second line");
    }

    #[test]
    fn plain_renders_text_only() {
        let lines = parse_tsv(&sample_tsv());
        assert_eq!(render_plain(&lines), "Hello world\nSecond line\n");
    }

    #[test]
    fn json_maps_back_to_original() {
        let im = crate::rgb::RgbImage::new(400, 300, vec![[255u8, 255, 255]; 400 * 300]).unwrap();
        let proc = crate::resize::process(&im, None, Some(200), 1).unwrap();
        let lines = parse_tsv(&sample_tsv());
        let v: Value = serde_json::from_str(&render_json(&proc, &lines)).unwrap();
        assert_eq!(v["width"], 400);
        assert_eq!(v["height"], 300);
        let arr = v["lines"].as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["text"], "Hello world");
        assert_eq!(arr[0]["x"], 20);
        assert_eq!(arr[0]["y"], 40);
        assert_eq!(arr[0]["w"], 380);
        assert_eq!(arr[0]["h"], 60);
        assert_eq!(arr[1]["text"], "Second line");
        assert!((arr[0]["conf"].as_f64().unwrap() - 87.8).abs() < 1e-6);
    }

    #[test]
    fn vision_json_omits_conf() {
        let im = crate::rgb::RgbImage::new(400, 300, vec![[255u8, 255, 255]; 400 * 300]).unwrap();
        let proc = crate::resize::process(&im, None, Some(200), 1).unwrap();
        let lines = vec![OcrLine {
            x: 5,
            y: 10,
            w: 100,
            h: 20,
            text: "Hi".to_string(),
            conf: 0.0,
        }];
        let v: Value = serde_json::from_str(&render_json_vision(&proc, &lines)).unwrap();
        assert_eq!(v["width"], 400);
        assert_eq!(v["lines"][0]["x"], 10);
        assert_eq!(v["lines"][0]["w"], 200);
        assert_eq!(v["lines"][0]["text"], "Hi");
        assert!(v["lines"][0].get("conf").is_none());
    }

    #[test]
    fn fill_text_nodes_attaches_and_appends() {
        let mut root = Node {
            x: 0,
            y: 0,
            w: 400,
            h: 300,
            kind: NodeKind::Block,
            color: [255, 255, 255],
            density: None,
            text: None,
            children: vec![Node {
                x: 0,
                y: 0,
                w: 400,
                h: 60,
                kind: NodeKind::Text,
                color: [255, 255, 255],
                density: Some(0.5),
                text: Some("Title".to_string()),
                children: vec![],
            }],
        };
        let lines = parse_tsv(&sample_tsv());
        fill_text_nodes(&mut root, &lines);
        assert_eq!(root.children.len(), 2);
        assert_eq!(root.children[0].text.as_deref(), Some("Title Hello world"));
        assert_eq!(root.children[1].kind, NodeKind::Text);
        assert_eq!(root.children[1].text.as_deref(), Some("Second line"));
        assert_eq!((root.children[1].x, root.children[1].y), (5, 80));
    }

    #[test]
    fn low_confidence_lines_are_skipped() {
        let mut root = Node {
            x: 0,
            y: 0,
            w: 100,
            h: 100,
            kind: NodeKind::Block,
            color: [0, 0, 0],
            density: None,
            text: None,
            children: vec![],
        };
        fill_text_nodes(
            &mut root,
            &[OcrLine {
                x: 0,
                y: 0,
                w: 10,
                h: 10,
                text: "noise".into(),
                conf: 5.0,
            }],
        );
        assert!(root.children.is_empty());
    }
}
