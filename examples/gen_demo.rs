//! Generate a small demo PNG for trying out arwe.
use image::{ImageBuffer, Rgb};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = (64, 48);
    let mut img = ImageBuffer::new(w, h);

    // Background gradient.
    for (x, y, p) in img.enumerate_pixels_mut() {
        let r = (x * 255 / (w - 1)) as u8;
        let g = (y * 255 / (h - 1)) as u8;
        *p = Rgb([r, g, 128]);
    }

    // White circle.
    let (cx, cy, radius) = (32.0f32, 20.0f32, 12.0f32);
    for y in 0..h {
        for x in 0..w {
            let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
            if d <= radius {
                img.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }
    }

    // Black square.
    for y in 6..18 {
        for x in 44..58 {
            img.put_pixel(x, y, Rgb([0, 0, 0]));
        }
    }

    img.save("demo.png")?;
    println!("wrote demo.png ({}x{})", w, h);
    Ok(())
}
