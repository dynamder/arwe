use anyhow::{bail, Result};

use crate::rgb::RgbImage;

/// Crop region in original-image pixel coordinates.
#[derive(Clone, Copy, Debug)]
pub struct CropRect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

/// A processed image plus the mapping back to original pixel coordinates.
#[derive(Clone, Debug)]
pub struct Processed {
    pub image: RgbImage,
    pub orig_width: usize,
    pub orig_height: usize,
    offset_x: usize,
    offset_y: usize,
    crop_w: usize,
    crop_h: usize,
    scale_x: f64,
    scale_y: f64,
}

impl Processed {
    pub fn to_original(&self, x: usize, y: usize) -> (usize, usize) {
        let ox = (x as f64 * self.scale_x).round() as usize + self.offset_x;
        let oy = (y as f64 * self.scale_y).round() as usize + self.offset_y;
        (ox.min(self.orig_width - 1), oy.min(self.orig_height - 1))
    }

    pub fn to_original_rect(
        &self,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    ) -> (usize, usize, usize, usize) {
        let (ox, oy) = self.to_original(x, y);
        let ow = ((w as f64) * self.scale_x).round() as usize;
        let oh = ((h as f64) * self.scale_y).round() as usize;
        (
            ox,
            oy,
            ow.min(self.orig_width - ox),
            oh.min(self.orig_height - oy),
        )
    }

    /// Map an original-image coordinate to the processed image, clamping to the
    /// nearest in-range pixel. Returns None when outside the crop region.
    pub fn to_processed(&self, x: usize, y: usize) -> Option<(usize, usize)> {
        if x < self.offset_x || y < self.offset_y {
            return None;
        }
        let rx = x - self.offset_x;
        let ry = y - self.offset_y;
        if rx >= self.crop_w || ry >= self.crop_h {
            return None;
        }
        let px = ((rx as f64 / self.scale_x).round() as usize).min(self.image.width - 1);
        let py = ((ry as f64 / self.scale_y).round() as usize).min(self.image.height - 1);
        Some((px, py))
    }

    /// Map an original-image rectangle to the processed image. Returns None when
    /// the rectangle lies entirely outside the crop region.
    pub fn to_processed_rect(
        &self,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    ) -> Option<(usize, usize, usize, usize)> {
        let (x0, y0) = self.to_processed(x, y)?;
        let x1 = x.saturating_add(w.saturating_sub(1));
        let y1 = y.saturating_add(h.saturating_sub(1));
        let (px1, py1) = self.to_processed(x1, y1)?;
        if px1 < x0 || py1 < y0 {
            return None;
        }
        Some((x0, y0, px1 - x0 + 1, py1 - y0 + 1))
    }
}

/// Pipeline: crop (original coords) -> area-average resize to `width` -> subsample by `stride`.
pub fn process(
    img: &RgbImage,
    crop: Option<CropRect>,
    width: Option<usize>,
    stride: usize,
) -> Result<Processed> {
    if stride == 0 {
        bail!("stride must be >= 1");
    }
    let cr = match crop {
        Some(c) => c,
        None => CropRect {
            x: 0,
            y: 0,
            w: img.width,
            h: img.height,
        },
    };
    if cr.w == 0
        || cr.h == 0
        || cr.x >= img.width
        || cr.y >= img.height
        || cr.x + cr.w > img.width
        || cr.y + cr.h > img.height
    {
        bail!(
            "crop {}x{}+{}+{} is outside the {}x{} image",
            cr.w,
            cr.h,
            cr.x,
            cr.y,
            img.width,
            img.height
        );
    }
    let cropped = crop_image(img, cr)?;
    let resized = match width {
        Some(0) => bail!("--width must be >= 1"),
        Some(w) if w != cropped.width => resize_width(&cropped, w),
        _ => cropped.clone(),
    };
    let subsampled = if stride > 1 {
        subsample(&resized, stride)
    } else {
        resized
    };
    let scale_x = cropped.width as f64 / subsampled.width as f64;
    let scale_y = cropped.height as f64 / subsampled.height as f64;
    Ok(Processed {
        image: subsampled,
        orig_width: img.width,
        orig_height: img.height,
        offset_x: cr.x,
        offset_y: cr.y,
        crop_w: cr.w,
        crop_h: cr.h,
        scale_x,
        scale_y,
    })
}

pub fn crop_image(img: &RgbImage, cr: CropRect) -> Result<RgbImage> {
    if cr.w == 0 || cr.h == 0 || cr.x + cr.w > img.width || cr.y + cr.h > img.height {
        bail!("crop region out of bounds");
    }
    let mut pixels = Vec::with_capacity(cr.w * cr.h);
    for y in cr.y..cr.y + cr.h {
        let row = &img.pixels[y * img.width + cr.x..y * img.width + cr.x + cr.w];
        pixels.extend_from_slice(row);
    }
    RgbImage::new(cr.w, cr.h, pixels)
}

fn resize_width(img: &RgbImage, width: usize) -> RgbImage {
    let height = ((img.height as f64) * (width as f64) / (img.width as f64))
        .round()
        .max(1.0) as u32;
    let height = height.max(1) as usize;
    box_average(img, width, height)
}

/// Exact area-average (box) resize with integer-aligned source rectangles.
fn box_average(img: &RgbImage, width: usize, height: usize) -> RgbImage {
    let mut pixels = Vec::with_capacity(width * height);
    for oy in 0..height {
        let sy0 = (oy * img.height) / height;
        let sy1 = ((oy + 1) * img.height) / height;
        for ox in 0..width {
            let sx0 = (ox * img.width) / width;
            let sx1 = ((ox + 1) * img.width) / width;
            let mut sum = [0u32; 3];
            let mut count = 0u32;
            for y in sy0..sy1 {
                for x in sx0..sx1 {
                    let p = img.pixels[y * img.width + x];
                    sum[0] += p[0] as u32;
                    sum[1] += p[1] as u32;
                    sum[2] += p[2] as u32;
                    count += 1;
                }
            }
            let c = [
                ((sum[0] as f32 / count as f32).round()) as u8,
                ((sum[1] as f32 / count as f32).round()) as u8,
                ((sum[2] as f32 / count as f32).round()) as u8,
            ];
            pixels.push(c);
        }
    }
    RgbImage::new(width, height, pixels).expect("resize dims match")
}

fn subsample(img: &RgbImage, stride: usize) -> RgbImage {
    let w = (img.width - 1) / stride + 1;
    let h = (img.height - 1) / stride + 1;
    let mut pixels = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            pixels.push(img.pixels[(y * stride) * img.width + x * stride]);
        }
    }
    RgbImage::new(w, h, pixels).expect("subsample dims match")
}

pub fn parse_crop(s: &str) -> Result<CropRect, String> {
    let parts: Vec<&str> = s.split(',').map(str::trim).collect();
    if parts.len() != 4 {
        return Err("--crop must be X,Y,W,H".into());
    }
    let mut nums = Vec::with_capacity(4);
    for p in parts {
        nums.push(
            p.parse::<usize>()
                .map_err(|_| format!("invalid --crop value '{}'", p))?,
        );
    }
    if nums[2] == 0 || nums[3] == 0 {
        return Err("--crop width and height must be > 0".into());
    }
    Ok(CropRect {
        x: nums[0],
        y: nums[1],
        w: nums[2],
        h: nums[3],
    })
}

pub fn parse_point(s: &str) -> Result<(usize, usize), String> {
    let parts: Vec<&str> = s.split(',').map(str::trim).collect();
    if parts.len() != 2 {
        return Err("--sample must be X,Y".into());
    }
    let x = parts[0]
        .parse::<usize>()
        .map_err(|_| format!("invalid --sample value '{}'", parts[0]))?;
    let y = parts[1]
        .parse::<usize>()
        .map_err(|_| format!("invalid --sample value '{}'", parts[1]))?;
    Ok((x, y))
}
