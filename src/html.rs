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
        out.push_str(&format!("<{} {}>{}</{}>\n", tag, attrs, escape_html(body), tag));
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
            let name_len = if rest.starts_with(b"div") {
                3
            } else if rest.starts_with(b"span") {
                4
            } else {
                0
            };
            if name_len > 0 {
                let name_end = i + 1 + name_len;
                if name_end <= b.len()
                    && (name_end == b.len()
                        || matches!(
                            b[name_end],
                            b' ' | b'\t' | b'\r' | b'\n' | b'/' | b'>'
                        ))
                {
                    if let Some(gt) = find_tag_end(b, name_end) {
                        let tag = &html[i..=gt];
                        out.push_str(&rewrite_tag(tag, proc));
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

fn rewrite_tag(tag: &str, proc: &Processed) -> String {
    let close = if tag.ends_with("/>") { "/>" } else { ">" };
    let core = tag.trim_end_matches(['>', '/']);
    let mut values = [None::<String>, None, None, None];
    let mut parts: Vec<String> = Vec::new();
    for tok in core.split_whitespace() {
        if let Some(eq) = tok.find('=') {
            let key = &tok[..eq];
            let val = tok[eq + 1..].trim_matches(|c| c == '"' || c == '\'');
            match key {
                "data-x" => values[0] = Some(val.to_string()),
                "data-y" => values[1] = Some(val.to_string()),
                "data-w" => values[2] = Some(val.to_string()),
                "data-h" => values[3] = Some(val.to_string()),
                _ => {}
            }
            parts.push(format!("{}=\"{}\"", key, val));
        } else {
            parts.push(tok.to_string());
        }
    }
    let (Some(x), Some(y), Some(w), Some(h)) =
        (&values[0], &values[1], &values[2], &values[3])
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
    for p in parts.iter_mut() {
        if let Some(eq) = p.find('=') {
            match &p[..eq] {
                "data-x" => *p = format!("data-x=\"{}\"", ox),
                "data-y" => *p = format!("data-y=\"{}\"", oy),
                "data-w" => *p = format!("data-w=\"{}\"", ow),
                "data-h" => *p = format!("data-h=\"{}\"", oh),
                _ => {}
            }
        }
    }
    let mut out = parts.join(" ");
    if close == "/>" && parts.len() > 1 {
        out.push(' ');
    }
    out.push_str(close);
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
}
