use std::fs;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

pub const MAX_KERNEL_DIM: usize = 256;

#[derive(Clone, Debug)]
pub struct Kernel {
    pub name: String,
    pub description: String,
    /// 1 = per-channel 2D kernel (applied to R, G, B separately),
    /// 3 = cross-channel kernel (mixes RGB into one channel).
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub values: Vec<f32>,
}

pub struct Preset {
    pub name: &'static str,
    pub description: &'static str,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub values: &'static [f32],
}

pub const PRESETS: &[Preset] = &[
    Preset {
        name: "identity",
        description: "no-op 3x3 kernel",
        channels: 1,
        height: 3,
        width: 3,
        values: &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    },
    Preset {
        name: "box3",
        description: "3x3 box blur (sum 9, not normalized)",
        channels: 1,
        height: 3,
        width: 3,
        values: &[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    },
    Preset {
        name: "gaussian3",
        description: "3x3 Gaussian blur (sum 16)",
        channels: 1,
        height: 3,
        width: 3,
        values: &[1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0],
    },
    Preset {
        name: "gaussian5",
        description: "5x5 Gaussian blur (sum 256)",
        channels: 1,
        height: 5,
        width: 5,
        values: &[
            1.0, 4.0, 6.0, 4.0, 1.0, 4.0, 16.0, 24.0, 16.0, 4.0, 6.0, 24.0, 36.0, 24.0, 6.0,
            4.0, 16.0, 24.0, 16.0, 4.0, 1.0, 4.0, 6.0, 4.0, 1.0,
        ],
    },
    Preset {
        name: "sharpen",
        description: "3x3 sharpen (sum 1)",
        channels: 1,
        height: 3,
        width: 3,
        values: &[0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
    },
    Preset {
        name: "sobel-x",
        description: "Sobel horizontal edges (sum 0)",
        channels: 1,
        height: 3,
        width: 3,
        values: &[-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0],
    },
    Preset {
        name: "sobel-y",
        description: "Sobel vertical edges (sum 0)",
        channels: 1,
        height: 3,
        width: 3,
        values: &[-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0],
    },
    Preset {
        name: "laplacian",
        description: "4-neighbour Laplacian (sum 0)",
        channels: 1,
        height: 3,
        width: 3,
        values: &[0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0],
    },
    Preset {
        name: "edge",
        description: "8-neighbour Laplacian edge detector (sum 0)",
        channels: 1,
        height: 3,
        width: 3,
        values: &[
            -1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0,
        ],
    },
    Preset {
        name: "emboss",
        description: "emboss relief (sum 1)",
        channels: 1,
        height: 3,
        width: 3,
        values: &[-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0],
    },
    Preset {
        name: "luma",
        description: "1x1 cross-channel: Rec.601 grayscale weights",
        channels: 3,
        height: 1,
        width: 1,
        values: &[0.299, 0.587, 0.114],
    },
    Preset {
        name: "sobel-x-rgb",
        description: "3x3 cross-channel Sobel-X weighted by Rec.601 luma (sum 0)",
        channels: 3,
        height: 3,
        width: 3,
        values: &[
            -0.299, 0.0, 0.299, -0.598, 0.0, 0.598, -0.299, 0.0, 0.299, // R
            -0.587, 0.0, 0.587, -1.174, 0.0, 1.174, -0.587, 0.0, 0.587, // G
            -0.114, 0.0, 0.114, -0.228, 0.0, 0.228, -0.114, 0.0, 0.114, // B
        ],
    },
];

pub fn presets() -> &'static [Preset] {
    PRESETS
}

pub fn preset(name: &str) -> Result<Kernel> {
    let want = name.to_ascii_lowercase();
    PRESETS
        .iter()
        .find(|p| p.name.to_ascii_lowercase() == want)
        .map(|p| Kernel {
            name: p.name.to_string(),
            description: p.description.to_string(),
            channels: p.channels,
            height: p.height,
            width: p.width,
            values: p.values.to_vec(),
        })
        .ok_or_else(|| anyhow!("unknown preset '{}'", name))
}

pub fn presets_json() -> String {
    let mut map = serde_json::Map::new();
    for p in PRESETS {
        let mut m = serde_json::Map::new();
        m.insert("description".into(), Value::String(p.description.into()));
        m.insert("channels".into(), Value::Number(p.channels.into()));
        m.insert("kernel".into(), kernel_json_value(p));
        map.insert(p.name.into(), Value::Object(m));
    }
    serde_json::to_string(&Value::Object(map)).expect("serializing presets cannot fail")
}

fn kernel_json_value(k: &Preset) -> Value {
    let mut out = Vec::new();
    if k.channels == 1 {
        for y in 0..k.height {
            let mut row = Vec::new();
            for x in 0..k.width {
                row.push(Value::from(k.values[y * k.width + x] as f64));
            }
            out.push(Value::Array(row));
        }
    } else {
        for c in 0..k.channels {
            let mut plane = Vec::new();
            for y in 0..k.height {
                let mut row = Vec::new();
                for x in 0..k.width {
                    row.push(Value::from(
                        k.values[c * k.height * k.width + y * k.width + x] as f64,
                    ));
                }
                plane.push(Value::Array(row));
            }
            out.push(Value::Array(plane));
        }
    }
    Value::Array(out)
}

/// Parse a kernel from a JSON string.
///
/// Accepted shapes:
/// - 2D array of rows: `[[-1,0,1],[-2,0,2],[-1,0,1]]` -> per-channel.
/// - 3D array of channel planes: `[[[r...]],[[g...]],[[b...]]]` -> cross-channel.
/// - Object with a `"kernel"` field containing either of the above.
pub fn from_json(s: &str) -> Result<Kernel> {
    let v: Value =
        serde_json::from_str(s).with_context(|| "kernel JSON failed to parse")?;
    let v = match v {
        Value::Object(map) => match map.get("kernel") {
            Some(k) => k.clone(),
            None => bail!("kernel JSON object must contain a \"kernel\" field"),
        },
        other => other,
    };
    parse_kernel_value(v)
}

fn parse_kernel_value(v: Value) -> Result<Kernel> {
    let arr = match v {
        Value::Array(a) => a,
        _ => bail!("kernel must be a JSON array of rows (2D) or channels of rows (3D)"),
    };
    if arr.is_empty() {
        bail!("kernel must not be empty");
    }
    let first = arr
        .first()
        .ok_or_else(|| anyhow!("kernel must not be empty"))?;
    match first {
        Value::Array(row) => match row.first() {
            Some(Value::Array(_)) => parse_3d(arr),
            _ => parse_2d(arr),
        },
        _ => bail!("kernel must be an array of arrays"),
    }
}

fn parse_2d(arr: Vec<Value>) -> Result<Kernel> {
    let mut values = Vec::new();
    let h = arr.len();
    let mut w: Option<usize> = None;
    for row in arr {
        let r = match row {
            Value::Array(r) => r,
            _ => bail!("kernel rows must be arrays of numbers"),
        };
        if r.is_empty() {
            bail!("kernel rows must not be empty");
        }
        match w {
            Some(w) if w != r.len() => bail!("kernel rows have inconsistent widths"),
            None => w = Some(r.len()),
            _ => {}
        }
        for x in r {
            values.push(number(&x)?);
        }
    }
    let w = w.expect("non-empty rows");
    check_dims(h, w)?;
    Ok(Kernel {
        name: "custom".into(),
        description: "custom per-channel kernel".into(),
        channels: 1,
        height: h,
        width: w,
        values,
    })
}

fn parse_3d(arr: Vec<Value>) -> Result<Kernel> {
    if arr.len() != 3 {
        bail!(
            "cross-channel kernels must have exactly 3 channels (R, G, B), got {}",
            arr.len()
        );
    }
    let mut values = Vec::new();
    let mut h: Option<usize> = None;
    let mut w: Option<usize> = None;
    for ch in arr {
        let plane = match ch {
            Value::Array(p) => p,
            _ => bail!("kernel channels must be arrays of rows"),
        };
        if plane.is_empty() {
            bail!("kernel channels must not be empty");
        }
        match h {
            Some(h) if h != plane.len() => bail!("kernel channels have inconsistent heights"),
            None => h = Some(plane.len()),
            _ => {}
        }
        for row in plane {
            let r = match row {
                Value::Array(r) => r,
                _ => bail!("kernel rows must be arrays of numbers"),
            };
            if r.is_empty() {
                bail!("kernel rows must not be empty");
            }
            match w {
                Some(w) if w != r.len() => bail!("kernel rows have inconsistent widths"),
                None => w = Some(r.len()),
                _ => {}
            }
            for x in r {
                values.push(number(&x)?);
            }
        }
    }
    let (h, w) = (h.expect("non-empty channels"), w.expect("non-empty rows"));
    check_dims(h, w)?;
    Ok(Kernel {
        name: "custom".into(),
        description: "custom cross-channel kernel".into(),
        channels: 3,
        height: h,
        width: w,
        values,
    })
}

fn number(v: &Value) -> Result<f32> {
    v.as_f64()
        .map(|f| f as f32)
        .ok_or_else(|| anyhow!("kernel values must be numbers, got {}", v))
}

fn check_dims(h: usize, w: usize) -> Result<()> {
    if h == 0 || w == 0 {
        bail!("kernel dimensions must be at least 1x1");
    }
    if h > MAX_KERNEL_DIM || w > MAX_KERNEL_DIM {
        bail!(
            "kernel too large ({}x{}, max {}x{})",
            h,
            w,
            MAX_KERNEL_DIM,
            MAX_KERNEL_DIM
        );
    }
    Ok(())
}

/// Load a kernel from a file. Accepts JSON (see `from_json`) or a plain-text
/// matrix: one row per line, numbers separated by whitespace, commas or `|`.
pub fn from_file(path: &Path) -> Result<Kernel> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read kernel file '{}'", path.display()))?;
    if let Ok(k) = from_json(&text) {
        return Ok(k);
    }
    parse_plain_text(&text).with_context(|| {
        format!(
            "kernel file '{}' is neither valid JSON nor a plain number matrix",
            path.display()
        )
    })
}

fn parse_plain_text(text: &str) -> Result<Kernel> {
    let mut values = Vec::new();
    let mut h = 0usize;
    let mut w: Option<usize> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }
        let row: Vec<f32> = line
            .split(|c: char| c.is_whitespace() || c == ',' || c == '|')
            .filter(|t| !t.is_empty())
            .map(|t| {
                t.parse::<f32>()
                    .map_err(|_| anyhow!("invalid number '{}'", t))
            })
            .collect::<Result<_>>()?;
        if row.is_empty() {
            continue;
        }
        match w {
            Some(w) if w != row.len() => bail!("plain-text kernel rows must have equal length"),
            None => w = Some(row.len()),
            _ => {}
        }
        values.extend(row);
        h += 1;
    }
    let w = w.ok_or_else(|| anyhow!("plain-text kernel has no rows"))?;
    check_dims(h, w)?;
    Ok(Kernel {
        name: "custom".into(),
        description: "custom per-channel kernel".into(),
        channels: 1,
        height: h,
        width: w,
        values,
    })
}
