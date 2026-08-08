use anyhow::{bail, Result};

use crate::kernels::Kernel;
use crate::rgb::{to_u8, ChannelData, ImageData, RgbImage};

/// 2D convolution with zero padding and no normalization.
///
/// Padding is distributed asymmetrically ((kh-1)/2 before, rest after) so
/// that with stride 1 the output has the same size as the input.
pub fn convolve_plane(
    plane: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
    kh: usize,
    kw: usize,
    stride: usize,
) -> Vec<f32> {
    assert_eq!(kernel.len(), kh * kw);
    let out_w = (width - 1) / stride + 1;
    let out_h = (height - 1) / stride + 1;
    let top = (kh - 1) / 2;
    let left = (kw - 1) / 2;
    let mut out = vec![0.0; out_w * out_h];

    for oy in 0..out_h {
        let y0 = oy * stride;
        for ox in 0..out_w {
            let x0 = ox * stride;
            let mut acc = 0.0;
            for ky in 0..kh {
                let iy = y0 as isize + ky as isize - top as isize;
                if iy < 0 || iy >= height as isize {
                    continue; // zero padding
                }
                let row = iy as usize * width;
                for kx in 0..kw {
                    let ix = x0 as isize + kx as isize - left as isize;
                    if ix < 0 || ix >= width as isize {
                        continue; // zero padding
                    }
                    acc += kernel[ky * kw + kx] * plane[row + ix as usize];
                }
            }
            out[oy * out_w + ox] = acc;
        }
    }
    out
}

fn out_size(n: usize, stride: usize) -> usize {
    (n - 1) / stride + 1
}

fn mono_data(img: &RgbImage, values: &[f32], stride: usize) -> ImageData {
    ImageData {
        width: out_size(img.width, stride),
        height: out_size(img.height, stride),
        data: ChannelData::Mono(values.iter().map(|v| to_u8(*v)).collect()),
    }
}

fn require_2d(kernel: &Kernel) -> Result<()> {
    if kernel.channels != 1 {
        bail!(
            "kernel '{}' is cross-channel ({} channels); expected a 2D per-channel kernel",
            kernel.name,
            kernel.channels
        );
    }
    Ok(())
}

/// Apply a 2D kernel to each of R, G, B independently. Output is RGB.
pub fn convolve_rgb(img: &RgbImage, kernel: &Kernel, stride: usize) -> Result<ImageData> {
    require_2d(kernel)?;
    let [r, g, b] = img.channels_f32();
    let or = convolve_plane(
        &r,
        img.width,
        img.height,
        &kernel.values,
        kernel.height,
        kernel.width,
        stride,
    );
    let og = convolve_plane(
        &g,
        img.width,
        img.height,
        &kernel.values,
        kernel.height,
        kernel.width,
        stride,
    );
    let ob = convolve_plane(
        &b,
        img.width,
        img.height,
        &kernel.values,
        kernel.height,
        kernel.width,
        stride,
    );
    let pixels: Vec<[u8; 3]> = (0..or.len())
        .map(|i| [to_u8(or[i]), to_u8(og[i]), to_u8(ob[i])])
        .collect();
    Ok(ImageData {
        width: out_size(img.width, stride),
        height: out_size(img.height, stride),
        data: ChannelData::Rgb(pixels),
    })
}

/// Convert to grayscale first, then apply a 2D kernel. Output is single-channel.
pub fn convolve_luma(img: &RgbImage, kernel: &Kernel, stride: usize) -> Result<ImageData> {
    require_2d(kernel)?;
    let plane: Vec<f32> = img.luma().iter().map(|v| *v as f32).collect();
    let out = convolve_plane(
        &plane,
        img.width,
        img.height,
        &kernel.values,
        kernel.height,
        kernel.width,
        stride,
    );
    Ok(mono_data(img, &out, stride))
}

/// Cross-channel convolution: kernel shape is [3][kh][kw] and mixes R, G, B.
/// Output is single-channel.
pub fn convolve_cross(img: &RgbImage, kernel: &Kernel, stride: usize) -> Result<ImageData> {
    if kernel.channels != 3 {
        bail!(
            "kernel '{}' has {} channels; cross-channel kernels must have exactly 3 (RGB)",
            kernel.name,
            kernel.channels
        );
    }
    let planes = img.channels_f32();
    let kh = kernel.height;
    let kw = kernel.width;
    let out_w = out_size(img.width, stride);
    let out_h = out_size(img.height, stride);
    let top = (kh - 1) / 2;
    let left = (kw - 1) / 2;
    let mut out = vec![0.0; out_w * out_h];

    for oy in 0..out_h {
        let y0 = oy * stride;
        for ox in 0..out_w {
            let x0 = ox * stride;
            let mut acc = 0.0;
            for (c, plane) in planes.iter().enumerate() {
                for ky in 0..kh {
                    let iy = y0 as isize + ky as isize - top as isize;
                    if iy < 0 || iy >= img.height as isize {
                        continue;
                    }
                    let row = iy as usize * img.width;
                    for kx in 0..kw {
                        let ix = x0 as isize + kx as isize - left as isize;
                        if ix < 0 || ix >= img.width as isize {
                            continue;
                        }
                        acc += kernel.values[c * kh * kw + ky * kw + kx]
                            * plane[row + ix as usize];
                    }
                }
            }
            out[oy * out_w + ox] = acc;
        }
    }
    Ok(mono_data(img, &out, stride))
}
