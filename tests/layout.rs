use arwe::layout::{self, Direction, NodeKind};
use arwe::resize;
use arwe::rgb::RgbImage;

fn img(w: usize, h: usize, pixels: Vec<[u8; 3]>) -> RgbImage {
    RgbImage::new(w, h, pixels).unwrap()
}

fn solid(w: usize, h: usize, c: [u8; 3]) -> RgbImage {
    img(w, h, vec![c; w * h])
}

fn analyze(im: &RgbImage) -> layout::LayoutResult {
    layout::analyze(im, layout::MAX_BLOCKS_DEFAULT, layout::MAX_DEPTH_DEFAULT)
}

fn is_group(n: &layout::Node) -> Option<Direction> {
    match n.kind {
        NodeKind::Group { direction } => Some(direction),
        _ => None,
    }
}

fn is_block(n: &layout::Node) -> bool {
    matches!(n.kind, NodeKind::Block)
}

#[test]
fn two_horizontal_bands() {
    let mut pixels = vec![[0, 0, 0]; 8 * 8];
    for i in 4..8 {
        for x in 0..8 {
            pixels[i * 8 + x] = [255, 255, 255];
        }
    }
    let r = analyze(&img(8, 8, pixels));
    assert!(!r.truncated);
    let root = &r.root;
    assert_eq!(is_group(root), Some(Direction::Column));
    assert_eq!(root.children.len(), 2);
    assert_eq!((root.children[0].x, root.children[0].y, root.children[0].w, root.children[0].h), (0, 0, 8, 4));
    assert_eq!(root.children[0].color, [0, 0, 0]);
    assert!(is_block(&root.children[0]));
    assert_eq!((root.children[1].x, root.children[1].y, root.children[1].w, root.children[1].h), (0, 4, 8, 4));
    assert_eq!(root.children[1].color, [255, 255, 255]);
}

#[test]
fn two_vertical_columns() {
    let mut pixels = vec![[0, 0, 0]; 8 * 8];
    for x in 4..8 {
        for y in 0..8 {
            pixels[y * 8 + x] = [0, 0, 255];
        }
    }
    let r = analyze(&img(8, 8, pixels));
    let root = &r.root;
    assert_eq!(is_group(root), Some(Direction::Row));
    assert_eq!(root.children.len(), 2);
    assert_eq!((root.children[0].x, root.children[0].y, root.children[0].w, root.children[0].h), (0, 0, 4, 8));
    assert_eq!(root.children[0].color, [0, 0, 0]);
    assert_eq!((root.children[1].x, root.children[1].y, root.children[1].w, root.children[1].h), (4, 0, 4, 8));
    assert_eq!(root.children[1].color, [0, 0, 255]);
}

#[test]
fn nested_header_then_sidebar_content() {
    let mut pixels = vec![[0, 0, 0]; 48 * 48];
    for y in 16..48 {
        for x in 0..48 {
            pixels[y * 48 + x] = if x < 16 { [255, 255, 255] } else { [200, 200, 200] };
        }
    }
    let r = analyze(&img(48, 48, pixels));
    let root = &r.root;
    assert_eq!(is_group(root), Some(Direction::Column));
    assert_eq!(root.children.len(), 2);
    // Header block.
    assert!(is_block(&root.children[0]));
    assert_eq!((root.children[0].x, root.children[0].y, root.children[0].w, root.children[0].h), (0, 0, 48, 16));
    assert_eq!(root.children[0].color, [0, 0, 0]);
    // Body group: sidebar + content.
    let body = &root.children[1];
    assert_eq!(is_group(body), Some(Direction::Row));
    assert_eq!(body.children.len(), 2);
    assert_eq!(body.children[0].color, [255, 255, 255]);
    assert_eq!((body.children[0].x, body.children[0].y, body.children[0].w, body.children[0].h), (0, 16, 16, 32));
    assert_eq!(body.children[1].color, [200, 200, 200]);
    assert_eq!((body.children[1].x, body.children[1].y, body.children[1].w, body.children[1].h), (16, 16, 32, 32));
}

#[test]
fn uniform_image_is_single_block() {
    let r = analyze(&solid(8, 8, [100, 100, 100]));
    assert!(!r.truncated);
    assert!(is_block(&r.root));
    assert!(r.root.children.is_empty());
    assert_eq!(r.root.color, [100, 100, 100]);
}

#[test]
fn text_band_is_marked() {
    let mut pixels = vec![[255, 255, 255]; 20 * 20];
    // Three thick vertical bars, like glyph strokes.
    for &x0 in &[2usize, 7, 12] {
        for x in x0..x0 + 2 {
            for y in 5..10 {
                pixels[y * 20 + x] = [0, 0, 0];
            }
        }
    }
    let r = analyze(&img(20, 20, pixels));
    fn find_text<'a>(n: &'a layout::Node, out: &mut Vec<&'a layout::Node>) {
        if matches!(n.kind, NodeKind::Text) {
            out.push(n);
        }
        for c in &n.children {
            find_text(c, out);
        }
    }
    let mut texts = Vec::new();
    find_text(&r.root, &mut texts);
    assert_eq!(texts.len(), 1, "expected exactly one text box");
    let t = texts[0];
    assert!((t.x as i32 - 1).abs() <= 2, "text x = {}", t.x);
    assert!((t.y as i32 - 4).abs() <= 2, "text y = {}", t.y);
    assert!(t.w >= 6 && t.h >= 4, "text box {}x{} too small", t.w, t.h);
    assert!(t.density.unwrap() > 0.0);
    // The containing root is the uniform white background.
    assert_eq!(r.root.color, [255, 255, 255]);
    assert!(r.root.children.iter().any(|c| matches!(c.kind, NodeKind::Text)));
}

#[test]
fn budget_truncation_is_reported() {
    let mut pixels = vec![[0, 0, 0]; 8 * 8];
    for i in 4..8 {
        for x in 0..8 {
            pixels[i * 8 + x] = [255, 255, 255];
        }
    }
    let im = img(8, 8, pixels);
    let r = layout::analyze(&im, 1, 6);
    assert!(r.truncated);
}

#[test]
fn json_shape_is_compact() {
    let mut pixels = vec![[0, 0, 0]; 8 * 8];
    for i in 4..8 {
        for x in 0..8 {
            pixels[i * 8 + x] = [255, 255, 255];
        }
    }
    let im = img(8, 8, pixels);
    let proc = resize::process(&im, None, None, 1).unwrap();
    let result = analyze(&im);
    let json = layout::result_json(&proc, &result);
    assert!(!json.contains(' '), "JSON must be compact, got: {}", json);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["width"], 8);
    assert_eq!(v["height"], 8);
    assert_eq!(v["truncated"], false);
    assert_eq!(v["root"]["kind"], "group");
    assert_eq!(v["root"]["direction"], "column");
    assert_eq!(v["root"]["children"].as_array().unwrap().len(), 2);
    assert_eq!(v["root"]["children"][0]["color"], "#000000");
    assert_eq!(v["root"]["children"][0]["x"], 0);
    assert_eq!(v["root"]["children"][0]["h"], 4);
}

#[test]
fn json_reports_text_density_rounded() {
    let mut pixels = vec![[255, 255, 255]; 20 * 20];
    for &x0 in &[2usize, 7, 12] {
        for x in x0..x0 + 2 {
            for y in 5..10 {
                pixels[y * 20 + x] = [0, 0, 0];
            }
        }
    }
    let im = img(20, 20, pixels);
    let proc = resize::process(&im, None, None, 1).unwrap();
    let result = analyze(&im);
    let v: serde_json::Value = serde_json::from_str(&layout::result_json(&proc, &result)).unwrap();
    let text = v["root"]["children"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["kind"] == "text")
        .expect("text node");
    assert!((text["density"].as_f64().unwrap() - 0.735).abs() < 1e-6);
}

#[test]
fn ascii_map_has_header_and_legend() {
    let mut pixels = vec![[0, 0, 0]; 8 * 8];
    for i in 4..8 {
        for x in 0..8 {
            pixels[i * 8 + x] = [255, 255, 255];
        }
    }
    let im = img(8, 8, pixels);
    let proc = resize::process(&im, None, None, 1).unwrap();
    let result = analyze(&im);
    let map = layout::ascii_map(&proc, &result, false);
    assert!(map.starts_with("# 8x8\n"));
    assert!(map.contains("A: 0,0,8,4 #000000\n"));
    assert!(map.contains("B: 0,4,8,4 #ffffff\n"));
    // 8 rows of 8 chars in the map body.
    let body: Vec<&str> = map.lines().skip(1).take(8).collect();
    assert_eq!(body.len(), 8);
    assert!(body.iter().all(|l| l.len() == 8));
}

#[test]
fn text_boxes_are_counted_in_budget() {
    let mut pixels = vec![[255, 255, 255]; 20 * 20];
    for &x0 in &[2usize, 7, 12] {
        for x in x0..x0 + 2 {
            for y in 5..10 {
                pixels[y * 20 + x] = [0, 0, 0];
            }
        }
    }
    // Budget 1: the single text box consumes it, the background block is forced.
    let r = layout::analyze(&img(20, 20, pixels), 1, 6);
    assert!(r.truncated);
}

#[test]
fn text_gap_keeps_two_bands_separate() {
    let mut pixels = vec![[255, 255, 255]; 24 * 24];
    for y in 5..8 {
        for x in 2..5 {
            pixels[y * 24 + x] = [0, 0, 0];
        }
    }
    for y in 13..16 {
        for x in 2..5 {
            pixels[y * 24 + x] = [0, 0, 0];
        }
    }
    let r = analyze(&img(24, 24, pixels));
    fn find_text<'a>(n: &'a layout::Node, out: &mut Vec<&'a layout::Node>) {
        if matches!(n.kind, NodeKind::Text) {
            out.push(n);
        }
        for c in &n.children {
            find_text(c, out);
        }
    }
    let mut texts = Vec::new();
    find_text(&r.root, &mut texts);
    assert_eq!(texts.len(), 2, "expected two separate text bands");
    let mut ys: Vec<usize> = texts.iter().map(|t| t.y).collect();
    ys.sort();
    assert!(ys[0] < 10, "first band y = {}", ys[0]);
    assert!(ys[1] >= 11, "second band y = {}", ys[1]);
}

#[test]
fn text_box_inside_block_is_attached_there() {
    // 40x40: top 16 black, bottom 24 white; gray bars fully inside the bottom half.
    let mut pixels = vec![[0, 0, 0]; 40 * 40];
    for y in 16..40 {
        for x in 0..40 {
            pixels[y * 40 + x] = [255, 255, 255];
        }
    }
    for y in 20..32 {
        for x in 12..20 {
            pixels[y * 40 + x] = [128, 128, 128];
        }
    }
    let r = analyze(&img(40, 40, pixels));
    let root = &r.root;
    assert!(matches!(root.kind, NodeKind::Group { .. }));
    assert!(!root.children.iter().any(|c| matches!(c.kind, NodeKind::Text)));
    let bottom = root.children.iter().find(|c| c.y >= 16).expect("bottom block");
    assert!(bottom.children.iter().any(|c| matches!(c.kind, NodeKind::Text)));
}

#[test]
fn text_box_straddling_split_attaches_to_group() {
    // Tall gray bars spanning the horizontal split between the black and white
    // halves: the text box crosses the boundary and attaches to the group.
    let mut pixels = vec![[0, 0, 0]; 10 * 10];
    for y in 4..10 {
        for x in 0..10 {
            pixels[y * 10 + x] = [255, 255, 255];
        }
    }
    for y in 1..9 {
        for x in 3..7 {
            pixels[y * 10 + x] = [128, 128, 128];
        }
    }
    let r = analyze(&img(10, 10, pixels));
    let root = &r.root;
    assert!(root.children.iter().any(|c| matches!(c.kind, NodeKind::Text)));
}

#[test]
fn three_column_nested_split_uses_offsets() {
    // 20x12: black header; body with white / gray / dark columns.
    let mut pixels = vec![[0, 0, 0]; 20 * 12];
    for y in 4..12 {
        for x in 0..20 {
            pixels[y * 20 + x] = if x < 6 {
                [255, 255, 255]
            } else if x < 14 {
                [150, 150, 150]
            } else {
                [50, 50, 50]
            };
        }
    }
    let r = analyze(&img(20, 12, pixels));
    let root = &r.root;
    assert!(matches!(root.kind, NodeKind::Group { .. }));
    // Body group starts at y=4 and splits into columns.
    let body = root
        .children
        .iter()
        .find(|c| c.y >= 4)
        .expect("body group");
    assert!(matches!(body.kind, NodeKind::Group { .. }));
    let inner = body
        .children
        .iter()
        .find(|c| c.x >= 6)
        .expect("inner column group");
    assert_eq!((inner.x, inner.y, inner.w, inner.h), (6, 4, 14, 8));
    assert_eq!(inner.children.len(), 2);
    assert_eq!((inner.children[0].x, inner.children[0].w), (6, 8));
    assert_eq!((inner.children[1].x, inner.children[1].w), (14, 6));
}

#[test]
fn budget_split_structure_priority() {
    // Four horizontal bands; with a tiny budget the structure side truncates.
    let mut pixels = vec![[0, 0, 0]; 16 * 16];
    let colors = [[255, 255, 255], [200, 200, 200], [150, 150, 150], [100, 100, 100]];
    for (band, c) in colors.iter().enumerate() {
        for y in band * 4..(band + 1) * 4 {
            for x in 0..16 {
                pixels[y * 16 + x] = *c;
            }
        }
    }
    let im = img(16, 16, pixels);
    let r = layout::analyze(&im, 5, 8);
    assert!(r.truncated);
    // With an exactly-full text cap, no false truncation.
    let mut pixels = vec![[255, 255, 255]; 20 * 20];
    for &x0 in &[2usize, 7, 12] {
        for x in x0..x0 + 2 {
            for y in 5..10 {
                pixels[y * 20 + x] = [0, 0, 0];
            }
        }
    }
    let r = layout::analyze(&img(20, 20, pixels), 2, 6);
    assert!(!r.truncated);
}

#[test]
fn ascii_map_body_and_ansi() {
    let mut pixels = vec![[0, 0, 0]; 8 * 8];
    for i in 4..8 {
        for x in 0..8 {
            pixels[i * 8 + x] = [255, 255, 255];
        }
    }
    let im = img(8, 8, pixels);
    let proc = resize::process(&im, None, None, 1).unwrap();
    let result = analyze(&im);
    let map = layout::ascii_map(&proc, &result, false);
    let body: Vec<&str> = map.lines().skip(1).take(8).collect();
    assert!(body[0..4].iter().all(|l| *l == "AAAAAAAA"));
    assert!(body[4..8].iter().all(|l| *l == "BBBBBBBB"));
    let ansi = layout::ascii_map(&proc, &result, true);
    assert!(ansi.contains("\u{1b}[48;2;0;0;0mA"));
    assert!(ansi.contains("\u{1b}[48;2;255;255;255mB"));
    assert!(!map.contains("\u{1b}["));
}
