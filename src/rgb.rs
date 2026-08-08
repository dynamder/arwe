use anyhow::{bail, Result};

/// A decoded image held as row-major RGB pixels.
#[derive(Clone, Debug)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<[u8; 3]>,
}

/// Output pixel data: full RGB, or a single grayscale value per pixel.
#[derive(Clone, Debug)]
pub enum ChannelData {
    Rgb(Vec<[u8; 3]>),
    Mono(Vec<u8>),
}

#[derive(Clone, Debug)]
pub struct ImageData {
    pub width: usize,
    pub height: usize,
    pub data: ChannelData,
}

impl RgbImage {
    pub fn new(width: usize, height: usize, pixels: Vec<[u8; 3]>) -> Result<Self> {
        if pixels.len() != width * height {
            bail!(
                "pixel count {} does not match {}x{}",
                pixels.len(),
                width,
                height
            );
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn rgb_data(&self) -> ImageData {
        ImageData {
            width: self.width,
            height: self.height,
            data: ChannelData::Rgb(self.pixels.clone()),
        }
    }

    /// Rec. 601 luma: 0.299R + 0.587G + 0.114B.
    pub fn luma(&self) -> Vec<u8> {
        self.pixels
            .iter()
            .map(|p| to_u8(0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32))
            .collect()
    }

    pub fn luma_data(&self) -> ImageData {
        ImageData {
            width: self.width,
            height: self.height,
            data: ChannelData::Mono(self.luma()),
        }
    }

    /// R, G, B planes as f32, each row-major.
    pub fn channels_f32(&self) -> [Vec<f32>; 3] {
        let mut c = [
            vec![0.0; self.pixels.len()],
            vec![0.0; self.pixels.len()],
            vec![0.0; self.pixels.len()],
        ];
        for (i, p) in self.pixels.iter().enumerate() {
            c[0][i] = p[0] as f32;
            c[1][i] = p[1] as f32;
            c[2][i] = p[2] as f32;
        }
        c
    }

    pub fn pixel(&self, x: usize, y: usize) -> Option<[u8; 3]> {
        if x < self.width && y < self.height {
            Some(self.pixels[y * self.width + x])
        } else {
            None
        }
    }
}

/// Round to nearest, then clamp to 0..=255.
pub fn to_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}
