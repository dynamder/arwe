use crate::layout::{Direction, LayoutResult, Node, NodeKind};
use crate::ocr::OcrLine;
use crate::resize::Processed;

/// Deterministic natural-language-style digest of an image, assembled from the
/// local layout box tree and OCR text. Used by the convocr engine for
/// `arwe describe` and the explore `describe` primitive.
pub fn digest(proc: &Processed, result: &LayoutResult, lines: &[OcrLine]) -> String {
    let mut groups = 0usize;
    let mut blocks = 0usize;
    let mut texts = 0usize;
    count(&result.root, &mut groups, &mut blocks, &mut texts);

    let mut out = String::new();
    out.push_str(&format!(
        "Image: {}x{}px.\n",
        proc.orig_width, proc.orig_height
    ));
    let root_dir = match result.root.kind {
        NodeKind::Group { direction } => match direction {
            Direction::Row => "row",
            Direction::Column => "column",
        },
        _ => "single block",
    };
    out.push_str(&format!(
        "Layout: root is a {} with {} child section(s).\n",
        root_dir,
        result.root.children.len()
    ));
    if result.truncated {
        out.push_str("Note: the box tree was truncated by the block/depth budget.\n");
    }
    out.push_str(&format!(
        "Elements: {} groups, {} blocks, {} text regions.\n",
        groups, blocks, texts
    ));
    if lines.is_empty() {
        out.push_str("Visible text: none detected.\n");
    } else {
        out.push_str("Visible text:\n");
        for l in lines {
            let t = l.text.trim();
            if t.is_empty() {
                continue;
            }
            out.push_str(&format!(
                "- \"{}\" @ ({}x{}, {}x{})\n",
                t, l.x, l.y, l.w, l.h
            ));
        }
    }
    out
}

fn count(n: &Node, groups: &mut usize, blocks: &mut usize, texts: &mut usize) {
    match n.kind {
        NodeKind::Group { .. } => *groups += 1,
        NodeKind::Block => *blocks += 1,
        NodeKind::Text => *texts += 1,
    }
    for c in &n.children {
        count(c, groups, blocks, texts);
    }
}

#[cfg(test)]
mod unit {
    use super::*;
    use crate::rgb::RgbImage;

    #[test]
    fn digest_reports_structure_and_text() {
        let img = RgbImage::new(400, 300, vec![[255u8, 255, 255]; 400 * 300]).unwrap();
        let proc = crate::resize::process(&img, None, None, 1).unwrap();
        let root = Node {
            x: 0,
            y: 0,
            w: 400,
            h: 300,
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
                    w: 400,
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
                    w: 100,
                    h: 20,
                    kind: NodeKind::Text,
                    color: [255, 255, 255],
                    density: Some(0.5),
                    text: Some("Login".to_string()),
                    children: vec![],
                },
            ],
        };
        let result = LayoutResult {
            root,
            truncated: false,
        };
        let lines = vec![OcrLine {
            x: 10,
            y: 60,
            w: 100,
            h: 20,
            text: "Login".to_string(),
            conf: 90.0,
        }];
        let d = digest(&proc, &result, &lines);
        assert!(d.contains("Image: 400x300px."), "{}", d);
        assert!(
            d.contains("root is a column with 2 child section(s)"),
            "{}",
            d
        );
        assert!(
            d.contains("Elements: 1 groups, 1 blocks, 1 text regions."),
            "{}",
            d
        );
        assert!(d.contains("\"Login\" @ (10x60, 100x20)"), "{}", d);
    }
}
