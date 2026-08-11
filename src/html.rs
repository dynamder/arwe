use crate::layout::{Direction, Node, NodeKind};
use crate::resize::Processed;

/// Render the layout tree as a nested HTML fragment. Coordinates are mapped to
/// original-image pixels, groups/blocks become `<div>`, text nodes become
/// `<span>` and carry their OCR text (when present) as the element body.
pub fn render(proc: &Processed, root: &Node) -> String {
    let mut out = String::new();
    render_node(proc, root, 0, &mut out);
    out
}

fn render_node(proc: &Processed, n: &Node, depth: usize, out: &mut String) {
    let (x, y, w, h) = proc.to_original_rect(n.x, n.y, n.w, n.h);
    let tag = if matches!(n.kind, NodeKind::Text) {
        "span"
    } else {
        "div"
    };
    let mut attrs = format!(
        "data-x=\"{}\" data-y=\"{}\" data-w=\"{}\" data-h=\"{}\" data-color=\"{}\"",
        x,
        y,
        w,
        h,
        hex(n.color)
    );
    if let NodeKind::Group { direction } = &n.kind {
        let d = match direction {
            Direction::Row => "row",
            Direction::Column => "column",
        };
        attrs.push_str(&format!(" data-direction=\"{}\"", d));
    }
    if n.children.is_empty() {
        indent(depth, out);
        let body = n.text.as_deref().unwrap_or("");
        out.push_str(&format!(
            "<{} {}>{}</{}>\n",
            tag,
            attrs,
            escape_html(body),
            tag
        ));
        return;
    }
    indent(depth, out);
    out.push_str(&format!("<{} {}>\n", tag, attrs));
    for child in &n.children {
        render_node(proc, child, depth + 1, out);
    }
    indent(depth, out);
    out.push_str(&format!("</{}>\n", tag));
}

fn indent(depth: usize, out: &mut String) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}

fn hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

pub fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Map the `data-x/y/w/h` attributes of every `<div>`/`<span>` tag from the
/// coordinate space of the image the vision model saw back to original-image
/// pixels. Everything else (nesting, text, other attributes) is preserved.
///
/// The parser intentionally targets the narrow HTML subset the model is told
/// to emit; unknown tags or malformed markup pass through untouched.
pub fn remap_coords(html: &str, proc: &Processed) -> String {
    let b = html.as_bytes();
    let mut out = String::with_capacity(html.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'<' {
            let rest = &b[i + 1..];
            // Closing tags like "</button>" -> "</div>" so normalization of
            // the opening tag never leaves unbalanced markup behind.
            if rest.starts_with(b"/") {
                let mut matched: Option<(&str, &str)> = None;
                for (name, normalized) in NORMALIZED_TAGS {
                    if rest[1..].starts_with(name.as_bytes()) {
                        matched = Some((name, normalized));
                        break;
                    }
                }
                if let Some((name, normalized)) = matched {
                    let name_end = i + 2 + name.len();
                    if name_end <= b.len()
                        && (name_end == b.len()
                            || matches!(b[name_end], b' ' | b'\t' | b'\r' | b'\n' | b'>'))
                    {
                        if let Some(gt) = find_tag_end(b, name_end) {
                            out.push_str(&format!("</{}>", normalized));
                            i = gt + 1;
                            continue;
                        }
                    }
                }
            }
            let mut matched: Option<(&str, &str)> = None;
            for (name, normalized) in NORMALIZED_TAGS {
                if rest.starts_with(name.as_bytes()) {
                    matched = Some((name, normalized));
                    break;
                }
            }
            if let Some((name, normalized)) = matched {
                let name_len = name.len();
                let name_end = i + 1 + name_len;
                if name_end <= b.len()
                    && (name_end == b.len()
                        || matches!(b[name_end], b' ' | b'\t' | b'\r' | b'\n' | b'/' | b'>'))
                {
                    if let Some(gt) = find_tag_end(b, name_end) {
                        let tag = &html[i..=gt];
                        out.push_str(&rewrite_tag(tag, proc, normalized));
                        i = gt + 1;
                        continue;
                    }
                }
            }
        }
        let len = utf8_len(b[i]);
        out.push_str(&html[i..i + len]);
        i += len;
    }
    out
}

/// Element names the vision model may emit against the div/span rule. They
/// are normalized to `<div>` so coordinate remapping and style scaling still
/// apply; unknown tags remain pass-through.
const NORMALIZED_TAGS: &[(&str, &str)] = &[
    ("div", "div"),
    ("span", "span"),
    ("input", "div"),
    ("button", "div"),
    ("img", "div"),
    ("textarea", "div"),
    ("select", "div"),
    ("label", "div"),
    ("a", "div"),
    ("form", "div"),
];

fn find_tag_end(b: &[u8], start: usize) -> Option<usize> {
    let mut quote: Option<u8> = None;
    for (j, &c) in b.iter().enumerate().skip(start) {
        match quote {
            Some(q) => {
                if c == q {
                    quote = None;
                }
            }
            None => match c {
                b'"' | b'\'' => quote = Some(c),
                b'>' => return Some(j),
                _ => {}
            },
        }
    }
    None
}

fn rewrite_tag(tag: &str, proc: &Processed, name: &str) -> String {
    let close = if tag.ends_with("/>") { "/>" } else { ">" };
    let core = tag.trim_end_matches(['>', '/']);
    let mut values = [None::<String>, None, None, None];
    let mut parts: Vec<String> = Vec::new();
    for tok in split_attrs(core) {
        if let Some((key, _quote, val)) = parse_attr(&tok) {
            match key {
                "data-x" => values[0] = Some(val.to_string()),
                "data-y" => values[1] = Some(val.to_string()),
                "data-w" => values[2] = Some(val.to_string()),
                "data-h" => values[3] = Some(val.to_string()),
                _ => {}
            }
        }
        parts.push(tok);
    }
    // Normalize the tag name: the first token is "<div"/"<input"/etc.
    if let Some(first) = parts.first_mut() {
        if let Some(stripped) = first.strip_prefix('<') {
            if !stripped.is_empty() {
                *first = format!("<{}", name);
            }
        }
    }
    let (Some(x), Some(y), Some(w), Some(h)) = (&values[0], &values[1], &values[2], &values[3])
    else {
        return tag.to_string();
    };
    let (Ok(x), Ok(y), Ok(w), Ok(h)) = (
        x.parse::<usize>(),
        y.parse::<usize>(),
        w.parse::<usize>(),
        h.parse::<usize>(),
    ) else {
        return tag.to_string();
    };
    let (ox, oy, ow, oh) = proc.to_original_rect(x, y, w, h);
    let (ow, oh) = (ow.max(1), oh.max(1));
    let scale = proc.scale_xy().0;
    for p in parts.iter_mut() {
        let Some((key, quote, val)) = parse_attr(p) else {
            continue;
        };
        match key {
            "data-x" => *p = rebuild_attr(key, quote, &ox.to_string()),
            "data-y" => *p = rebuild_attr(key, quote, &oy.to_string()),
            "data-w" => *p = rebuild_attr(key, quote, &ow.to_string()),
            "data-h" => *p = rebuild_attr(key, quote, &oh.to_string()),
            "style" => *p = rebuild_attr(key, quote, &scale_style_px(val, scale)),
            _ => {}
        }
    }
    let mut out = parts.join(" ");
    if close == "/>" && parts.len() > 1 {
        out.push(' ');
    }
    out.push_str(close);
    out
}

/// Split a tag's attribute list on whitespace, respecting quotes so values
/// like `style="display:flex; flex-direction:row"` stay one token.
fn split_attrs(core: &str) -> Vec<String> {
    let mut toks = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    for c in core.chars() {
        match quote {
            Some(q) => {
                cur.push(c);
                if c == q {
                    quote = None;
                }
            }
            None => {
                if c == '"' || c == '\'' {
                    quote = Some(c);
                    cur.push(c);
                } else if c.is_whitespace() {
                    if !cur.is_empty() {
                        toks.push(std::mem::take(&mut cur));
                    }
                } else {
                    cur.push(c);
                }
            }
        }
    }
    if !cur.is_empty() {
        toks.push(cur);
    }
    toks
}

/// Split an attribute token into `(key, quote_char, value)`. `quote` is the
/// quote character when the value was quoted, otherwise `None`.
fn parse_attr(tok: &str) -> Option<(&str, Option<char>, &str)> {
    let eq = tok.find('=')?;
    let key = &tok[..eq];
    let rest = &tok[eq + 1..];
    match rest.chars().next() {
        Some(q @ ('"' | '\'')) => {
            let val = rest[1..].trim_end_matches(q);
            Some((key, Some(q), val))
        }
        _ => Some((key, None, rest)),
    }
}

fn rebuild_attr(key: &str, quote: Option<char>, val: &str) -> String {
    match quote {
        Some(q) => format!("{}={}{}{}", key, q, val, q),
        None => format!("{}={}", key, val),
    }
}

/// Rescale `Npx` length values in a CSS style string from the image the vision
/// model saw back to original-image pixels. Non-px values (`0`, `%`, `em`,
/// `rem`, `auto`, …) are left untouched; positive lengths stay at least 1px.
fn scale_style_px(style: &str, factor: f64) -> String {
    if factor == 1.0 {
        return style.to_string();
    }
    let mut out = String::with_capacity(style.len());
    let mut rest = style;
    while !rest.is_empty() {
        let mut chars = rest.chars();
        let c = chars.next().expect("non-empty rest");
        let num_start = c.is_ascii_digit()
            || (c == '-' && chars.clone().next().is_some_and(|n| n.is_ascii_digit()));
        if !num_start {
            out.push(c);
            rest = &rest[c.len_utf8()..];
            continue;
        }
        let bytes = rest.as_bytes();
        let mut idx = 0;
        if bytes[0] == b'-' {
            idx = 1;
        }
        while idx < bytes.len() && bytes[idx].is_ascii_digit() {
            idx += 1;
        }
        if idx + 1 < bytes.len() && bytes[idx] == b'.' && bytes[idx + 1].is_ascii_digit() {
            idx += 1;
            while idx < bytes.len() && bytes[idx].is_ascii_digit() {
                idx += 1;
            }
        }
        let num = &rest[..idx];
        if rest[idx..].starts_with("px") {
            let v: f64 = num.parse().unwrap_or(0.0);
            let rounded = (v * factor).round();
            if v > 0.0 && rounded < 1.0 {
                out.push_str("1px");
            } else if v < 0.0 && rounded > -1.0 {
                out.push_str("-1px");
            } else {
                out.push_str(&format!("{}", rounded));
                out.push_str("px");
            }
            rest = &rest[idx + 2..];
        } else {
            out.push_str(num);
            rest = &rest[idx..];
        }
    }
    out
}

fn utf8_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first >> 5 == 0b110 {
        2
    } else if first >> 4 == 0b1110 {
        3
    } else if first >> 3 == 0b11110 {
        4
    } else {
        1
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
    fn escapes_special_chars() {
        assert_eq!(
            escape_html("<a href=\"x\">&"),
            "&lt;a href=&quot;x&quot;&gt;&amp;"
        );
    }

    #[test]
    fn renders_nested_tree() {
        let proc = proc_scale2();
        let root = Node {
            x: 0,
            y: 0,
            w: 200,
            h: 150,
            kind: NodeKind::Group {
                direction: Direction::Column,
            },
            color: [255, 255, 255],
            density: None,
            text: None,
            children: vec![
                Node {
                    x: 0,
                    y: 0,
                    w: 200,
                    h: 50,
                    kind: NodeKind::Block,
                    color: [0, 0, 0],
                    density: None,
                    text: None,
                    children: vec![],
                },
                Node {
                    x: 10,
                    y: 60,
                    w: 80,
                    h: 30,
                    kind: NodeKind::Text,
                    color: [255, 255, 255],
                    density: Some(0.5),
                    text: Some("Hello <world>".to_string()),
                    children: vec![],
                },
            ],
        };
        let html = render(&proc, &root);
        let expected = concat!(
            "<div data-x=\"0\" data-y=\"0\" data-w=\"400\" data-h=\"300\" data-color=\"#ffffff\" data-direction=\"column\">\n",
            "  <div data-x=\"0\" data-y=\"0\" data-w=\"400\" data-h=\"100\" data-color=\"#000000\"></div>\n",
            "  <span data-x=\"20\" data-y=\"120\" data-w=\"160\" data-h=\"60\" data-color=\"#ffffff\">Hello &lt;world&gt;</span>\n",
            "</div>\n",
        );
        assert_eq!(html, expected);
    }

    #[test]
    fn remap_preserves_structure_and_text() {
        let proc = proc_scale2();
        let raw = "<div data-x=\"0\" data-y=\"0\" data-w=\"200\" data-h=\"150\" data-color=\"#000000\" data-direction=\"row\">\n  <div data-x=\"0\" data-y=\"0\" data-w=\"100\" data-h=\"150\" data-color=\"#ff0000\"></div>\n  <span data-x=\"100\" data-y=\"10\" data-w=\"90\" data-h=\"20\" data-color=\"#00ff00\">Save</span>\n</div>";
        let out = remap_coords(raw, &proc);
        assert!(out.contains("data-x=\"0\" data-y=\"0\" data-w=\"400\" data-h=\"300\""));
        assert!(out.contains("data-x=\"0\" data-y=\"0\" data-w=\"200\" data-h=\"300\""));
        assert!(out.contains("data-x=\"200\" data-y=\"20\" data-w=\"180\" data-h=\"40\""));
        assert!(out.contains(">Save</span>"));
        assert!(out.contains("data-direction=\"row\""));
    }

    #[test]
    fn remap_leaves_malformed_or_foreign_tags_alone() {
        let proc = proc_scale2();
        let raw = "<div data-x=\"1\" data-y=\"2\">nope</div><table><tr><td>x</td></tr></table>";
        let out = remap_coords(raw, &proc);
        assert!(out.contains("<div data-x=\"1\" data-y=\"2\">"));
        assert!(out.contains("<table><tr><td>x</td></tr></table>"));
    }

    #[test]
    fn remap_handles_self_closing_style() {
        let proc = proc_scale2();
        let raw = "<div data-x=\"0\" data-y=\"0\" data-w=\"10\" data-h=\"10\" />";
        assert_eq!(
            remap_coords(raw, &proc),
            "<div data-x=\"0\" data-y=\"0\" data-w=\"20\" data-h=\"20\" />"
        );
    }

    #[test]
    fn remap_preserves_style_with_spaces_and_text() {
        let proc = proc_scale2();
        let raw = concat!(
            "<div data-x=\"0\" data-y=\"0\" data-w=\"10\" data-h=\"10\" data-color=\"#000000\"",
            " style=\"display:flex; flex-direction:row; gap:2px\">",
            "<span data-x=\"1\" data-y=\"1\" data-w=\"4\" data-h=\"2\" data-color=\"#ffffff\"",
            " style=\"font-size:7px\">Save</span></div>",
        );
        let out = remap_coords(raw, &proc);
        assert!(
            out.contains("style=\"display:flex; flex-direction:row; gap:4px\""),
            "style with spaces must survive remap: {}",
            out
        );
        assert!(
            out.contains("style=\"font-size:14px\""),
            "px must scale: {}",
            out
        );
        assert!(out.contains(">Save</span>"), "text must survive: {}", out);
        assert!(
            out.contains("data-x=\"0\" data-y=\"0\" data-w=\"20\" data-h=\"20\""),
            "root box not remapped: {}",
            out
        );
        assert!(
            out.contains("data-x=\"2\" data-y=\"2\" data-w=\"8\" data-h=\"4\""),
            "span box not remapped: {}",
            out
        );
    }

    #[test]
    fn remap_preserves_single_quoted_style() {
        let proc = proc_scale2();
        let raw = "<div data-x=\"0\" data-y=\"0\" data-w=\"5\" data-h=\"5\" style='font-family: \"Open Sans\", sans-serif; margin:1px'></div>";
        let out = remap_coords(raw, &proc);
        assert!(
            out.contains("style='font-family: \"Open Sans\", sans-serif; margin:2px'"),
            "single-quoted style must survive: {}",
            out
        );
    }

    #[test]
    fn remap_normalizes_foreign_ui_tags() {
        let proc = proc_scale2();
        let raw = "<div data-x=\"0\" data-y=\"0\" data-w=\"10\" data-h=\"10\" data-color=\"#000000\"><input data-x=\"0\" data-y=\"0\" data-w=\"5\" data-h=\"5\" data-color=\"#ffffff\" style=\"font-size:7px\"/><button data-x=\"5\" data-y=\"0\" data-w=\"5\" data-h=\"5\" data-color=\"#eeeeee\">Send</button></div>";
        let out = remap_coords(raw, &proc);
        assert!(
            out.contains("<div data-x=\"0\" data-y=\"0\" data-w=\"10\" data-h=\"10\" data-color=\"#ffffff\" style=\"font-size:14px\" />"),
            "input must become a remapped div: {}",
            out
        );
        assert!(
            out.contains("<div data-x=\"10\" data-y=\"0\" data-w=\"10\" data-h=\"10\" data-color=\"#eeeeee\">Send</div>"),
            "button must become a remapped div: {}",
            out
        );
        assert!(!out.contains("<input"), "no input tag may remain: {}", out);
        assert!(
            !out.contains("<button"),
            "no button tag may remain: {}",
            out
        );
    }

    #[test]
    fn remap_clamps_degenerate_boxes() {
        let proc = proc_scale2();
        let raw = "<div data-x=\"0\" data-y=\"0\" data-w=\"0\" data-h=\"10\" data-color=\"#000000\"></div>";
        let out = remap_coords(raw, &proc);
        assert!(
            out.contains("data-w=\"1\" data-h=\"20\""),
            "zero-size boxes must be clamped: {}",
            out
        );
    }

    #[test]
    fn remap_keeps_unreadable_marker() {
        let proc = proc_scale2();
        let raw = "<span data-x=\"0\" data-y=\"0\" data-w=\"10\" data-h=\"10\" data-color=\"#ffffff\">[unreadable]</span>";
        let out = remap_coords(raw, &proc);
        assert!(
            out.contains(">[unreadable]</span>"),
            "unreadable marker must survive: {}",
            out
        );
    }

    #[test]
    fn scale_style_px_scales_lengths_only() {
        let style = "font-size:7.5px;padding:0;border-radius:1px;margin:2px;width:auto;gap:-3px";
        assert_eq!(
            scale_style_px(style, 2.0),
            "font-size:15px;padding:0;border-radius:2px;margin:4px;width:auto;gap:-6px"
        );
    }

    #[test]
    fn scale_style_px_keeps_tiny_positive_lengths_at_one() {
        assert_eq!(scale_style_px("padding:1px", 0.25), "padding:1px");
        assert_eq!(scale_style_px("margin:-1px", 0.25), "margin:-1px");
        assert_eq!(scale_style_px("font-size:14px", 1.0), "font-size:14px");
    }
}
