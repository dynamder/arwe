use std::io::Cursor;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use base64::engine::general_purpose::{STANDARD, URL_SAFE};
use base64::Engine as _;
use image::ImageReader;

use crate::capture;
use crate::rgb::RgbImage;

pub enum InputSource {
    Path(PathBuf),
    Base64(String),
    Stdin,
    /// Capture the primary monitor (`None`) or the Nth monitor (`Some(n)`).
    Screen(Option<usize>),
    /// Capture the visible window whose title contains the keyword.
    Window(String),
}

pub fn load(source: &InputSource) -> Result<RgbImage> {
    match source {
        InputSource::Screen(n) => capture::capture_screen(*n),
        InputSource::Window(kw) => capture::capture_window(kw),
        InputSource::Path(p) => {
            let bytes = std::fs::read(p)
                .with_context(|| format!("failed to read image file '{}'", p.display()))?;
            decode_image(&bytes)
        }
        InputSource::Base64(s) => decode_image(&decode_base64(s)?),
        InputSource::Stdin => {
            let mut text = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut text)?;
            decode_image(&decode_base64(text.trim())?)
        }
    }
}

fn decode_base64(s: &str) -> Result<Vec<u8>> {
    let s = s.trim();
    // Accept data URIs like "data:image/png;base64,...."
    let s = match s.find("base64,") {
        Some(i) => &s[i + "base64,".len()..],
        None => s,
    };
    STANDARD
        .decode(s)
        .or_else(|_| URL_SAFE.decode(s))
        .with_context(|| "failed to decode base64 input (tried standard and URL-safe alphabets)")
}

fn decode_image(bytes: &[u8]) -> Result<RgbImage> {
    let img = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .with_context(|| "could not detect image format from input bytes")?
        .decode()
        .with_context(|| "failed to decode image")?;
    let rgb = img.to_rgb8();
    let width = rgb.width() as usize;
    let height = rgb.height() as usize;
    let pixels = rgb
        .as_raw()
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    RgbImage::new(width, height, pixels)
}

pub fn require_exactly_one<'a>(choices: impl IntoIterator<Item = (&'a str, bool)>) -> Result<()> {
    let choices: Vec<(&str, bool)> = choices.into_iter().collect();
    let chosen: Vec<&str> = choices
        .iter()
        .filter(|(_, on)| *on)
        .map(|(name, _)| *name)
        .collect();
    match chosen.len() {
        1 => Ok(()),
        0 => {
            let names: Vec<&str> = choices.iter().map(|(name, _)| *name).collect();
            bail!(
                "exactly one of these flags is required: {}",
                names.join(", ")
            )
        }
        _ => bail!(
            "conflicting flags: {} (choose exactly one)",
            chosen.join(", ")
        ),
    }
}
