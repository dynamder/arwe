use serde_json::Value;

use crate::conv;
use crate::resize::Processed;
use crate::rgb::RgbImage;

pub const MAX_BLOCKS_DEFAULT: usize = 64;
pub const MAX_DEPTH_DEFAULT: usize = 6;

const EDGE_THRESHOLD: f32 = 32.0;
const ROW_DENSITY: f64 = 0.02;
const COL_DENSITY: f64 = 0.02;
const MERGE_GAP: usize = 2;
const MIN_REGION: usize = 4;
const BOUNDARY_DIFF: i32 = 12;
const BOUNDARY_FRACTION: f64 = 0.15;
const MIN_FILL: f64 = 0.15;
const MAX_COMPONENT_AREA: f64 = 0.10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Children are laid out side by side (CSS flex-direction: row).
    Row,
    /// Children are stacked top to bottom (CSS flex-direction: column).
    Column,
}

#[derive(Clone, Debug, PartialEq)]
pub enum NodeKind {
    Group { direction: Direction },
    Block,
    Text,
}

#[derive(Clone, Debug)]
pub struct Node {
    /// Rectangle in processed-image pixel coordinates.
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    pub kind: NodeKind,
    pub color: [u8; 3],
    pub density: Option<f32>,
    /// Recognized text (populated by `arwe layout --ocr`).
    pub text: Option<String>,
    pub children: Vec<Node>,
}

pub struct LayoutResult {
    pub root: Node,
    pub truncated: bool,
}

#[derive(Clone, Copy)]
struct Rect {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

#[derive(Clone, Copy)]
enum Axis {
    Rows,
    Cols,
}

pub fn analyze(img: &RgbImage, max_blocks: usize, max_depth: usize) -> LayoutResult {
    let energy = edge_energy(img);
    let mut text_boxes = detect_text(img, &energy);
    let mut truncated = false;
    // Reserve half the leaf budget for structure, half for text boxes, so a
    // text-heavy UI cannot starve the grid decomposition.
    let text_cap = max_blocks / 2;
    let structure_budget = max_blocks - text_cap;
    if text_boxes.len() > text_cap {
        truncated = true;
        text_boxes.truncate(text_cap);
    }

    let mut mask = vec![false; img.width * img.height];
    for tb in &text_boxes {
        mark_rect(&mut mask, img.width, img.height, tb.x, tb.y, tb.w, tb.h);
    }
    let rect = Rect {
        x: 0,
        y: 0,
        w: img.width,
        h: img.height,
    };
    let mut budget = structure_budget;
    let mut root = split(
        img,
        &mask,
        rect,
        0,
        max_depth,
        &mut budget,
        &mut truncated,
    );
    for tb in text_boxes {
        attach(&mut root, tb);
    }
    LayoutResult { root, truncated }
}

fn edge_energy(img: &RgbImage) -> Vec<f32> {
    let luma: Vec<f32> = img.luma().iter().map(|v| *v as f32).collect();
    let k: Vec<f32> = vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];
    conv::convolve_plane(&luma, img.width, img.height, &k, 3, 3, 1)
        .into_iter()
        .map(|v| v.abs())
        .collect()
}

fn detect_text(img: &RgbImage, energy: &[f32]) -> Vec<Node> {
    let w = img.width;
    let h = img.height;
    let keep = text_like_mask(img, energy);
    let row_min = (w as f64 * ROW_DENSITY) as usize;
    let row_min = row_min.max(1);
    let mut rows = vec![0usize; h];
    for y in 0..h {
        for x in 0..w {
            if keep[y * w + x] {
                rows[y] += 1;
            }
        }
    }
    let mut bands: Vec<(usize, usize)> = Vec::new();
    let mut in_band = false;
    let mut start = 0usize;
    let mut last = 0usize;
    for y in 0..h {
        if rows[y] >= row_min {
            if !in_band {
                start = y;
                in_band = true;
            }
            last = y;
        } else if in_band {
            let gap_end = (y + MERGE_GAP).min(h);
            if (y..gap_end).any(|yy| rows[yy] >= row_min) {
                // keep the band open across a small gap
            } else {
                bands.push((start, last));
                in_band = false;
            }
        }
    }
    if in_band {
        bands.push((start, last));
    }

    let mut boxes = Vec::new();
    for (by0, by1) in bands {
        let bh = by1 - by0 + 1;
        let col_min = (bh as f64 * COL_DENSITY) as usize;
        let col_min = col_min.max(1);
        let mut cols = vec![0usize; w];
        for y in by0..=by1 {
            for x in 0..w {
                if keep[y * w + x] {
                    cols[x] += 1;
                }
            }
        }
        let mut segs: Vec<(usize, usize)> = Vec::new();
        let mut in_seg = false;
        let mut s0 = 0usize;
        let mut s1 = 0usize;
        for x in 0..w {
            if cols[x] >= col_min {
                if !in_seg {
                    s0 = x;
                    in_seg = true;
                }
                s1 = x;
            } else if in_seg {
                let gap_end = (x + MERGE_GAP).min(w);
                if !(x..gap_end).any(|xx| cols[xx] >= col_min) {
                    segs.push((s0, s1));
                    in_seg = false;
                }
            }
        }
        if in_seg {
            segs.push((s0, s1));
        }
        for (s0, s1) in segs {
            let bw = s1 - s0 + 1;
            let mut hits = 0usize;
            for y in by0..=by1 {
                for x in s0..=s1 {
                    if keep[y * w + x] {
                        hits += 1;
                    }
                }
            }
            let density = hits as f32 / (bw * bh) as f32;
            let color = bg_color(img, s0, by0, bw, bh);
            boxes.push(Node {
                x: s0,
                y: by0,
                w: bw,
                h: bh,
                kind: NodeKind::Text,
                color,
                density: Some(density),
                text: None,
                children: Vec::new(),
            });
        }
    }
    boxes
}

/// Threshold the Laplacian energy, then keep only compact connected components
/// whose size and fill look like glyph strokes rather than grid lines or large
/// textured regions.
fn text_like_mask(img: &RgbImage, energy: &[f32]) -> Vec<bool> {
    let w = img.width;
    let h = img.height;
    let len = w * h;
    let mut mask = vec![false; len];
    for i in 0..len {
        mask[i] = energy[i] > EDGE_THRESHOLD;
    }
    // Zero padding in the Laplacian creates artificial border edges; drop the
    // outermost 1px frame so it cannot swallow glyphs near the image edge.
    for x in 0..w {
        mask[x] = false;
        mask[(h - 1) * w + x] = false;
    }
    for y in 0..h {
        mask[y * w] = false;
        mask[y * w + w - 1] = false;
    }
    // Remove long straight runs (grid lines, dividers, borders) so they cannot
    // merge with glyph components or be mistaken for text.
    let h_run = (w / 3).max(16);
    let v_run = (h / 3).max(16);
    for y in 0..h {
        let mut run = 0usize;
        for x in 0..=w {
            if x < w && mask[y * w + x] {
                run += 1;
            } else {
                if run > h_run {
                    for xx in (x - run)..x {
                        mask[y * w + xx] = false;
                    }
                }
                run = 0;
            }
        }
    }
    for x in 0..w {
        let mut run = 0usize;
        for y in 0..=h {
            if y < h && mask[y * w + x] {
                run += 1;
            } else {
                if run > v_run {
                    for yy in (y - run)..y {
                        mask[yy * w + x] = false;
                    }
                }
                run = 0;
            }
        }
    }
    let max_w = (w / 2).max(24);
    let max_h = (h / 2).max(24);
    let max_area = ((w * h) as f64 * MAX_COMPONENT_AREA).max(64.0);
    let mut visited = vec![false; len];
    let mut keep = vec![false; len];
    for start in 0..len {
        if !mask[start] || visited[start] {
            continue;
        }
        let mut comp = Vec::new();
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(idx) = stack.pop() {
            comp.push(idx);
            let x = idx % w;
            let y = idx / w;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        continue;
                    }
                    let ni = ny as usize * w + nx as usize;
                    if mask[ni] && !visited[ni] {
                        visited[ni] = true;
                        stack.push(ni);
                    }
                }
            }
        }
        let mut x0 = usize::MAX;
        let mut x1 = 0usize;
        let mut y0 = usize::MAX;
        let mut y1 = 0usize;
        for &i in &comp {
            let x = i % w;
            let y = i / w;
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
        }
        let bw = x1 - x0 + 1;
        let bh = y1 - y0 + 1;
        let fill = comp.len() as f64 / (bw * bh) as f64;
        let plausible = bw >= 2
            && bh >= 2
            && bw * 2 >= bh
            && bh * 2 >= bw
            && bw <= max_w
            && bh <= max_h
            && comp.len() as f64 <= max_area
            && fill >= MIN_FILL;
        if plausible {
            for i in comp {
                keep[i] = true;
            }
        }
    }
    keep
}

fn bg_color(img: &RgbImage, x: usize, y: usize, w: usize, h: usize) -> [u8; 3] {
    let mut r = Vec::with_capacity(w * h);
    let mut g = Vec::with_capacity(w * h);
    let mut b = Vec::with_capacity(w * h);
    for yy in y..y + h {
        for xx in x..x + w {
            let p = img.pixels[yy * img.width + xx];
            r.push(p[0] as f32);
            g.push(p[1] as f32);
            b.push(p[2] as f32);
        }
    }
    [
        median(&r) as u8,
        median(&g) as u8,
        median(&b) as u8,
    ]
}

fn median(v: &[f32]) -> f32 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s[s.len() / 2]
}

fn mark_rect(mask: &mut [bool], w: usize, h: usize, x: usize, y: usize, rw: usize, rh: usize) {
    for yy in y..y + rh {
        for xx in x..x + rw {
            if yy < h && xx < w {
                mask[yy * w + xx] = true;
            }
        }
    }
}

fn split(
    img: &RgbImage,
    mask: &[bool],
    rect: Rect,
    depth: usize,
    max_depth: usize,
    budget: &mut usize,
    truncated: &mut bool,
) -> Node {
    let color = region_color(img, mask, rect);
    if depth >= max_depth || rect.w < MIN_REGION || rect.h < MIN_REGION {
        return make_leaf(rect, color, budget, truncated);
    }
    let axis = if depth.is_multiple_of(2) {
        Axis::Rows
    } else {
        Axis::Cols
    };
    let cut = best_cut(img, mask, rect, axis)
        .or_else(|| best_cut(img, mask, rect, other(axis)));
    let Some((used_axis, pos)) = cut else {
        return make_leaf(rect, color, budget, truncated);
    };
    let (a, b) = split_rect(rect, used_axis, pos);
    let mut children = Vec::with_capacity(2);
    for part in [a, b] {
        if *budget == 0 {
            *truncated = true;
            break;
        }
        children.push(split(
            img,
            mask,
            part,
            depth + 1,
            max_depth,
            budget,
            truncated,
        ));
    }
    if children.is_empty() {
        return make_leaf(rect, color, budget, truncated);
    }
    Node {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
        kind: NodeKind::Group {
            direction: direction_of(used_axis),
        },
        color,
        density: None,
        text: None,
        children,
    }
}

fn make_leaf(rect: Rect, color: [u8; 3], budget: &mut usize, truncated: &mut bool) -> Node {
    if *budget == 0 {
        *truncated = true;
    } else {
        *budget -= 1;
    }
    Node {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
        kind: NodeKind::Block,
        color,
        density: None,
        text: None,
        children: Vec::new(),
    }
}

fn region_color(img: &RgbImage, mask: &[bool], rect: Rect) -> [u8; 3] {
    let mut r = Vec::new();
    let mut g = Vec::new();
    let mut b = Vec::new();
    let mut any_unmasked = false;
    for y in rect.y..rect.y + rect.h {
        for x in rect.x..rect.x + rect.w {
            let idx = y * img.width + x;
            if mask[idx] {
                continue;
            }
            any_unmasked = true;
            let p = img.pixels[idx];
            r.push(p[0] as f32);
            g.push(p[1] as f32);
            b.push(p[2] as f32);
        }
    }
    if !any_unmasked {
        // Region is entirely text-masked: fall back to all pixels.
        for y in rect.y..rect.y + rect.h {
            for x in rect.x..rect.x + rect.w {
                let p = img.pixels[y * img.width + x];
                r.push(p[0] as f32);
                g.push(p[1] as f32);
                b.push(p[2] as f32);
            }
        }
    }
    if r.is_empty() {
        return [0, 0, 0];
    }
    [median(&r) as u8, median(&g) as u8, median(&b) as u8]
}

fn best_cut(img: &RgbImage, mask: &[bool], rect: Rect, axis: Axis) -> Option<(Axis, usize)> {
    let mut best: Option<(usize, f64)> = None;
    match axis {
        Axis::Rows => {
            if rect.h >= 2 * MIN_REGION {
                let lo = rect.y + MIN_REGION - 1;
                let hi = rect.y + rect.h - MIN_REGION - 1;
                for pos in lo..=hi {
                    let s = boundary_strength(img, mask, rect, pos, true);
                    if s > BOUNDARY_FRACTION && best.is_none_or(|(_, bs)| s > bs) {
                        best = Some((pos, s));
                    }
                }
            }
        }
        Axis::Cols => {
            if rect.w >= 2 * MIN_REGION {
                let lo = rect.x + MIN_REGION - 1;
                let hi = rect.x + rect.w - MIN_REGION - 1;
                for pos in lo..=hi {
                    let s = boundary_strength(img, mask, rect, pos, false);
                    if s > BOUNDARY_FRACTION && best.is_none_or(|(_, bs)| s > bs) {
                        best = Some((pos, s));
                    }
                }
            }
        }
    }
    best.map(|(pos, _)| (axis, pos))
}

fn boundary_strength(
    img: &RgbImage,
    mask: &[bool],
    rect: Rect,
    pos: usize,
    rows: bool,
) -> f64 {
    let mut hits = 0usize;
    let total = if rows { rect.w } else { rect.h };
    if rows {
        for x in rect.x..rect.x + rect.w {
            let i1 = pos * img.width + x;
            let i2 = (pos + 1) * img.width + x;
            if mask[i1] || mask[i2] {
                continue;
            }
            if channel_diff(img.pixels[i1], img.pixels[i2]) > BOUNDARY_DIFF {
                hits += 1;
            }
        }
    } else {
        for y in rect.y..rect.y + rect.h {
            let i1 = y * img.width + pos;
            let i2 = y * img.width + pos + 1;
            if mask[i1] || mask[i2] {
                continue;
            }
            if channel_diff(img.pixels[i1], img.pixels[i2]) > BOUNDARY_DIFF {
                hits += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    }
}

fn channel_diff(a: [u8; 3], b: [u8; 3]) -> i32 {
    (a[0] as i32 - b[0] as i32).abs()
        + (a[1] as i32 - b[1] as i32).abs()
        + (a[2] as i32 - b[2] as i32).abs()
}

fn split_rect(rect: Rect, axis: Axis, pos: usize) -> (Rect, Rect) {
    match axis {
        Axis::Rows => (
            Rect {
                x: rect.x,
                y: rect.y,
                w: rect.w,
                h: pos - rect.y + 1,
            },
            Rect {
                x: rect.x,
                y: pos + 1,
                w: rect.w,
                h: rect.y + rect.h - pos - 1,
            },
        ),
        Axis::Cols => (
            Rect {
                x: rect.x,
                y: rect.y,
                w: pos - rect.x + 1,
                h: rect.h,
            },
            Rect {
                x: pos + 1,
                y: rect.y,
                w: rect.x + rect.w - pos - 1,
                h: rect.h,
            },
        ),
    }
}

fn other(axis: Axis) -> Axis {
    match axis {
        Axis::Rows => Axis::Cols,
        Axis::Cols => Axis::Rows,
    }
}

fn direction_of(axis: Axis) -> Direction {
    match axis {
        // Row boundaries stack children vertically.
        Axis::Rows => Direction::Column,
        // Column boundaries place children side by side.
        Axis::Cols => Direction::Row,
    }
}

fn attach(node: &mut Node, tb: Node) {
    for child in node.children.iter_mut() {
        if contains(child, &tb) {
            attach(child, tb);
            return;
        }
    }
    node.children.push(tb);
}

fn contains(a: &Node, b: &Node) -> bool {
    a.x <= b.x
        && a.y <= b.y
        && b.x + b.w <= a.x + a.w
        && b.y + b.h <= a.y + a.h
}

fn hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

pub fn result_json(proc: &Processed, result: &LayoutResult) -> String {
    let mut obj = serde_json::Map::new();
    obj.insert("width".into(), Value::from(proc.orig_width));
    obj.insert("height".into(), Value::from(proc.orig_height));
    obj.insert("truncated".into(), Value::Bool(result.truncated));
    obj.insert("root".into(), node_json(proc, &result.root));
    serde_json::to_string(&Value::Object(obj)).expect("serializing layout cannot fail")
}

fn node_json(proc: &Processed, n: &Node) -> Value {
    let (x, y, w, h) = proc.to_original_rect(n.x, n.y, n.w, n.h);
    let mut obj = serde_json::Map::new();
    obj.insert("x".into(), Value::from(x));
    obj.insert("y".into(), Value::from(y));
    obj.insert("w".into(), Value::from(w));
    obj.insert("h".into(), Value::from(h));
    obj.insert("color".into(), Value::String(hex(n.color)));
    match &n.kind {
        NodeKind::Group { direction } => {
            obj.insert("kind".into(), Value::String("group".into()));
            obj.insert(
                "direction".into(),
                Value::String(
                    match direction {
                        Direction::Row => "row",
                        Direction::Column => "column",
                    }
                    .into(),
                ),
            );
        }
        NodeKind::Block => {
            obj.insert("kind".into(), Value::String("block".into()));
        }
        NodeKind::Text => {
            obj.insert("kind".into(), Value::String("text".into()));
            if let Some(d) = n.density {
                obj.insert("density".into(), Value::from((d * 1000.0).round() / 1000.0));
            }
            if let Some(t) = &n.text {
                if !t.trim().is_empty() {
                    obj.insert("text".into(), Value::String(t.clone()));
                }
            }
        }
    }
    let children: Vec<Value> = n.children.iter().map(|c| node_json(proc, c)).collect();
    obj.insert("children".into(), Value::Array(children));
    Value::Object(obj)
}

pub fn ascii_map(proc: &Processed, result: &LayoutResult, ansi: bool) -> String {
    let w = proc.image.width;
    let h = proc.image.height;
    let mut leaves: Vec<&Node> = Vec::new();
    collect_leaves(&result.root, &mut leaves);
    let mut chars = vec![b'?'; w * h];
    let mut colors = vec![[0u8; 3]; w * h];
    for (i, leaf) in leaves.iter().enumerate() {
        let ch = legend_char(i) as u8;
        for y in leaf.y..leaf.y + leaf.h {
            for x in leaf.x..leaf.x + leaf.w {
                let idx = y * w + x;
                if idx < chars.len() {
                    chars[idx] = ch;
                    colors[idx] = leaf.color;
                }
            }
        }
    }
    let mut s = String::new();
    s.push_str(&format!("# {}x{}\n", proc.orig_width, proc.orig_height));
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let ch = chars[idx] as char;
            if ansi {
                let c = colors[idx];
                s.push_str(&format!(
                    "\x1b[48;2;{};{};{}m{}\x1b[0m",
                    c[0], c[1], c[2], ch
                ));
            } else {
                s.push(ch);
            }
        }
        s.push('\n');
    }
    for (i, leaf) in leaves.iter().enumerate() {
        let (x, y, lw, lh) = proc.to_original_rect(leaf.x, leaf.y, leaf.w, leaf.h);
        let text_suffix = if matches!(leaf.kind, NodeKind::Text) {
            match &leaf.text {
                Some(t) if !t.trim().is_empty() => {
                    format!(" [text \"{}\"]", truncate(t.trim(), 24))
                }
                _ => " [text]".to_string(),
            }
        } else {
            String::new()
        };
        s.push_str(&format!(
            "{}: {},{},{},{} {}{}\n",
            legend_char(i),
            x,
            y,
            lw,
            lh,
            hex(leaf.color),
            text_suffix,
        ));
    }
    s
}

fn truncate(s: &str, max_chars: usize) -> String {
    let mut out: String = s.chars().take(max_chars).collect();
    if s.chars().count() > max_chars {
        out.push('…');
    }
    out
}

fn collect_leaves<'a>(node: &'a Node, out: &mut Vec<&'a Node>) {
    if !matches!(node.kind, NodeKind::Group { .. }) {
        out.push(node);
    }
    for child in &node.children {
        collect_leaves(child, out);
    }
}

fn legend_char(i: usize) -> char {
    match i {
        0..26 => (b'A' + i as u8) as char,
        26..52 => (b'a' + (i - 26) as u8) as char,
        52..62 => (b'0' + (i - 52) as u8) as char,
        _ => '?',
    }
}

pub fn image_to_rgb(data: &crate::rgb::ImageData) -> crate::rgb::RgbImage {
    match &data.data {
        crate::rgb::ChannelData::Rgb(pixels) => {
            crate::rgb::RgbImage::new(data.width, data.height, pixels.clone())
                .expect("image data dims match")
        }
        crate::rgb::ChannelData::Mono(v) => {
            let pixels = v.iter().map(|x| [*x, *x, *x]).collect();
            crate::rgb::RgbImage::new(data.width, data.height, pixels)
                .expect("image data dims match")
        }
    }
}

#[cfg(test)]
mod unit {
    use super::*;

    fn im(w: usize, h: usize, c: [u8; 3]) -> RgbImage {
        RgbImage::new(w, h, vec![c; w * h]).unwrap()
    }

    #[test]
    fn region_color_falls_back_when_fully_masked() {
        let img = im(2, 2, [0, 0, 0]);
        let mask = vec![true; 4];
        // All pixels masked: fall back to the median of all pixels.
        assert_eq!(
            region_color(&img, &mask, Rect { x: 0, y: 0, w: 2, h: 2 }),
            [0, 0, 0]
        );
        // Mixed pixels: median, ignoring masked ones.
        let mut pixels = vec![[255, 255, 255]; 4];
        pixels[0] = [0, 0, 0];
        let img = RgbImage::new(2, 2, pixels).unwrap();
        let mask = vec![true, false, false, false];
        assert_eq!(
            region_color(&img, &mask, Rect { x: 0, y: 0, w: 2, h: 2 }),
            [255, 255, 255]
        );
    }

    #[test]
    fn boundary_strength_ignores_masked_pixels() {
        let img = im(4, 4, [0, 0, 0]);
        let rect = Rect { x: 0, y: 0, w: 4, h: 4 };
        let mut mask = vec![false; 16];
        // Mask the entire boundary row pair.
        for x in 0..4 {
            mask[4 + x] = true;
            mask[8 + x] = true;
        }
        assert_eq!(boundary_strength(&img, &mask, rect, 1, true), 0.0);
        // No mask: uniform colors give no boundary.
        assert_eq!(boundary_strength(&img, &[false; 16], rect, 1, true), 0.0);
    }

    #[test]
    fn contains_is_strict() {
        let a = Node {
            x: 0,
            y: 0,
            w: 8,
            h: 8,
            kind: NodeKind::Block,
            color: [0, 0, 0],
            density: None,
            text: None,
            children: vec![],
        };
        let inside = Node {
            x: 2,
            y: 2,
            w: 2,
            h: 2,
            kind: NodeKind::Text,
            color: [0, 0, 0],
            density: Some(0.5),
            text: None,
            children: vec![],
        };
        let straddle = Node {
            x: 6,
            y: 6,
            w: 4,
            h: 4,
            kind: NodeKind::Text,
            color: [0, 0, 0],
            density: Some(0.5),
            text: None,
            children: vec![],
        };
        assert!(contains(&a, &inside));
        assert!(!contains(&a, &straddle));
    }

    #[test]
    fn legend_char_boundaries() {
        assert_eq!(legend_char(0), 'A');
        assert_eq!(legend_char(25), 'Z');
        assert_eq!(legend_char(26), 'a');
        assert_eq!(legend_char(51), 'z');
        assert_eq!(legend_char(52), '0');
        assert_eq!(legend_char(61), '9');
        assert_eq!(legend_char(62), '?');
    }

    #[test]
    fn median_prefers_upper_middle() {
        assert_eq!(median(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 3.0);
    }
}
