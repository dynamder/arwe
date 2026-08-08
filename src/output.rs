use std::fmt::Write as _;

use crate::rgb::{ChannelData, ImageData};

/// Compact matrix serialization, no whitespace.
///
/// RGB:  `[[[r,g,b],...],...]`
/// Mono: `[[v,...],...]`
pub fn matrix(data: &ImageData) -> String {
    let mut s = String::new();
    match &data.data {
        ChannelData::Rgb(pixels) => matrix_rgb(&mut s, pixels, data.width),
        ChannelData::Mono(values) => matrix_mono(&mut s, values, data.width),
    }
    s
}

fn matrix_rgb(s: &mut String, pixels: &[[u8; 3]], width: usize) {
    s.push('[');
    for (i, p) in pixels.iter().enumerate() {
        if i % width == 0 {
            if i != 0 {
                s.push_str("],");
            }
            s.push('[');
        }
        let _ = write!(s, "[{},{},{}]", p[0], p[1], p[2]);
        if i % width != width - 1 {
            s.push(',');
        }
    }
    s.push_str("]]");
}

fn matrix_mono(s: &mut String, values: &[u8], width: usize) {
    s.push('[');
    for (i, v) in values.iter().enumerate() {
        if i % width == 0 {
            if i != 0 {
                s.push_str("],");
            }
            s.push('[');
        }
        let _ = write!(s, "{}", v);
        if i % width != width - 1 {
            s.push(',');
        }
    }
    s.push_str("]]");
}

/// ASCII art. RGB input is converted with Rec.601 luma; mono is used as-is.
pub fn ascii(data: &ImageData, ramp: &str) -> String {
    let values: Vec<u8> = match &data.data {
        ChannelData::Rgb(pixels) => pixels.iter().map(luma).collect(),
        ChannelData::Mono(v) => v.clone(),
    };
    ascii_from_values(&values, data.width, ramp)
}

/// ASCII art with 24-bit ANSI background colors (RGB pixels keep their color,
/// mono pixels render as grayscale backgrounds).
pub fn ascii_ansi(data: &ImageData, ramp: &str) -> String {
    let pairs: Vec<(u8, [u8; 3])> = match &data.data {
        ChannelData::Rgb(pixels) => pixels.iter().map(|p| (luma(p), *p)).collect(),
        ChannelData::Mono(v) => v.iter().map(|x| (*x, [*x, *x, *x])).collect(),
    };
    let n = ramp.chars().count().max(1);
    let mut s = String::new();
    for (i, (v, color)) in pairs.iter().enumerate() {
        if i % data.width == 0 && i != 0 {
            s.push('\n');
        }
        let idx = (*v as usize) * (n - 1) / 255;
        let ch = ramp.chars().nth(idx).expect("ramp is non-empty");
        let fg = if *v >= 128 { 30 } else { 37 };
        s.push_str(&format!(
            "\x1b[48;2;{};{};{};{}m{}\x1b[0m",
            color[0], color[1], color[2], fg, ch
        ));
    }
    s.push('\n');
    s
}

fn ascii_from_values(values: &[u8], width: usize, ramp: &str) -> String {
    let n = ramp.chars().count().max(1);
    let mut s = String::new();
    for (i, v) in values.iter().enumerate() {
        if i % width == 0 && i != 0 {
            s.push('\n');
        }
        let idx = (*v as usize) * (n - 1) / 255;
        s.push(ramp.chars().nth(idx).expect("ramp is non-empty"));
    }
    s.push('\n');
    s
}

fn luma(p: &[u8; 3]) -> u8 {
    (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32).round() as u8
}
