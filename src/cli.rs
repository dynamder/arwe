use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, ValueEnum};

use crate::capture;
use crate::config;
use crate::conv;
use crate::describe;
use crate::explore;
use crate::html;
use crate::image_io::{self, InputSource};
use crate::kernels::{self, Kernel};
use crate::layout;
use crate::ocr;
use crate::output;
use crate::resize::{self, CropRect, Processed};
use crate::vision;

pub const DEFAULT_RAMP: &str = " .:-=+*#%@";

#[derive(Args)]
pub struct InputArgs {
    /// Image file path (PNG/JPEG/BMP/GIF).
    #[arg(short, long)]
    pub input: Option<PathBuf>,

    /// Base64-encoded image data (data-URI prefixes are accepted).
    #[arg(long)]
    pub base64: Option<String>,

    /// Read base64-encoded image data from stdin.
    #[arg(long)]
    pub stdin: bool,

    /// Capture the primary monitor (--screen) or the Nth monitor (--screen N)
    /// directly into the pipeline; no temporary file is written.
    #[arg(long, num_args = 0..=1)]
    pub screen: Option<Option<usize>>,

    /// Capture the visible window whose title contains this keyword
    /// (case-insensitive substring match; run `arwe windows` to list).
    #[arg(long)]
    pub window: Option<String>,

    /// Resize to this output width (area-average), keeping aspect ratio.
    #[arg(long)]
    pub width: Option<usize>,

    /// Crop region in original pixels: X,Y,W,H (applied before --width/--stride).
    #[arg(long, value_parser = resize::parse_crop)]
    pub crop: Option<CropRect>,

    /// Sample pixel colors at original-image coordinates; replaces normal output.
    #[arg(long = "sample", value_parser = resize::parse_point)]
    pub sample: Vec<(usize, usize)>,
}

impl InputArgs {
    pub fn source(&self) -> Result<InputSource> {
        image_io::require_exactly_one([
            ("--input", self.input.is_some()),
            ("--base64", self.base64.is_some()),
            ("--stdin", self.stdin),
            ("--screen", self.screen.is_some()),
            ("--window", self.window.is_some()),
        ])?;
        if let Some(p) = &self.input {
            Ok(InputSource::Path(p.clone()))
        } else if let Some(b) = &self.base64 {
            Ok(InputSource::Base64(b.clone()))
        } else if self.stdin {
            Ok(InputSource::Stdin)
        } else if let Some(n) = self.screen {
            Ok(InputSource::Screen(n))
        } else {
            Ok(InputSource::Window(
                self.window.clone().expect("checked above"),
            ))
        }
    }
}

fn load_processed(args: &InputArgs, stride: usize) -> Result<Processed> {
    let img = image_io::load(&args.source()?)?;
    resize::process(&img, args.crop, args.width, stride)
}

/// Print `x,y #rrggbb` lines for --sample points (original coordinates).
/// Returns true when sampling replaced normal output.
fn emit_samples(args: &InputArgs, proc: &Processed) -> Result<bool> {
    if args.sample.is_empty() {
        return Ok(false);
    }
    for (x, y) in &args.sample {
        let (px, py) = proc
            .to_processed(*x, *y)
            .ok_or_else(|| anyhow!("sample point ({},{}) is outside the cropped image", x, y))?;
        let c = proc.image.pixel(px, py).expect("mapped pixel is in bounds");
        println!("{},{} #{:02x}{:02x}{:02x}", x, y, c[0], c[1], c[2]);
    }
    Ok(true)
}

#[derive(Args)]
pub struct MatrixArgs {
    #[command(flatten)]
    pub input: InputArgs,

    /// Convert to grayscale first and output single-channel values.
    #[arg(long)]
    pub grayscale: bool,

    /// Subsample by taking every Nth pixel (after --width).
    #[arg(long, default_value_t = 1)]
    pub stride: usize,
}

#[derive(Args)]
pub struct AsciiArgs {
    #[command(flatten)]
    pub input: InputArgs,

    /// Character ramp for luminance, darkest to brightest.
    #[arg(long, default_value = DEFAULT_RAMP)]
    pub ramp: String,

    /// Subsample by taking every Nth pixel (after --width).
    #[arg(long, default_value_t = 1)]
    pub stride: usize,

    /// Use 24-bit ANSI background colors.
    #[arg(long)]
    pub ansi: bool,
}

#[derive(Args)]
pub struct ConvolveArgs {
    #[command(flatten)]
    pub input: InputArgs,

    /// Preset kernel name (run `arwe presets` to list).
    #[arg(long)]
    pub preset: Option<String>,

    /// Path to a kernel file (JSON, or plain whitespace/comma-separated rows).
    #[arg(long)]
    pub kernel_file: Option<PathBuf>,

    /// Inline kernel as JSON (2D array for per-channel, 3D for cross-channel).
    #[arg(long)]
    pub kernel_json: Option<String>,

    /// Convolution stride (must be >= 1).
    #[arg(long, default_value_t = 1)]
    pub stride: usize,

    /// Convert to grayscale first, then apply a 2D kernel (single-channel output).
    #[arg(long)]
    pub grayscale: bool,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Matrix)]
    pub format: OutputFormat,

    /// Character ramp used when --format ascii, darkest to brightest.
    #[arg(long, default_value = DEFAULT_RAMP)]
    pub ramp: String,

    /// Use 24-bit ANSI background colors for --format ascii.
    #[arg(long)]
    pub ansi: bool,
}

impl ConvolveArgs {
    fn kernel(&self) -> Result<Kernel> {
        image_io::require_exactly_one([
            ("--preset", self.preset.is_some()),
            ("--kernel-file", self.kernel_file.is_some()),
            ("--kernel-json", self.kernel_json.is_some()),
        ])?;
        if let Some(name) = &self.preset {
            kernels::preset(name)
        } else if let Some(path) = &self.kernel_file {
            kernels::from_file(path)
        } else {
            kernels::from_json(self.kernel_json.as_deref().expect("checked above"))
        }
    }
}

#[derive(Args)]
pub struct LayoutArgs {
    #[command(flatten)]
    pub input: InputArgs,

    /// Layout analysis engine: vision model, or the deterministic
    /// convolution+OCR backend (alias: local).
    #[arg(long, value_enum, default_value_t = Engine::ConvOcr)]
    pub engine: Engine,

    /// Run local OCR (Tesseract) and attach recognized text to text nodes
    /// (local engine only).
    #[arg(long)]
    pub ocr: bool,

    /// OCR language(s), e.g. eng, chi_sim, eng+chi_sim (local engine with --ocr).
    #[arg(long, default_value = "eng")]
    pub lang: String,

    /// Tesseract page segmentation mode 0-13 (local engine with --ocr).
    #[arg(long, default_value_t = 3)]
    pub psm: u32,

    /// Vision profile from the config file (see `arwe config`).
    #[arg(long)]
    pub profile: Option<String>,

    /// Vision model ID (--engine vision; falls back to the profile, then
    /// OPENAI_MODEL, then the built-in default).
    #[arg(long)]
    pub model: Option<String>,

    /// OpenAI-compatible API base URL (--engine vision; falls back to the
    /// profile, then OPENAI_BASE_URL, then the built-in default).
    #[arg(long)]
    pub base_url: Option<String>,

    /// OpenAI API key (--engine vision; falls back to the profile, then
    /// OPENAI_API_KEY).
    #[arg(long)]
    pub api_key: Option<String>,

    /// Subsample by taking every Nth pixel (after --width).
    #[arg(long, default_value_t = 1)]
    pub stride: usize,

    /// Optional preset kernel applied as preprocessing.
    #[arg(long)]
    pub preset: Option<String>,

    /// Optional kernel file (JSON, or plain rows of numbers) applied as preprocessing.
    #[arg(long)]
    pub kernel_file: Option<PathBuf>,

    /// Optional inline kernel JSON applied as preprocessing.
    #[arg(long)]
    pub kernel_json: Option<String>,

    /// Output format.
    #[arg(long, value_enum, default_value_t = LayoutFormat::Both)]
    pub format: LayoutFormat,

    /// Maximum number of leaf boxes (blocks + text).
    #[arg(long, default_value_t = layout::MAX_BLOCKS_DEFAULT)]
    pub max_blocks: usize,

    /// Maximum recursion depth of grid decomposition.
    #[arg(long, default_value_t = layout::MAX_DEPTH_DEFAULT)]
    pub max_depth: usize,

    /// ANSI colored ASCII block map.
    #[arg(long)]
    pub ansi: bool,
}

impl LayoutArgs {
    fn kernel(&self) -> Result<Option<Kernel>> {
        let n = [
            self.preset.is_some(),
            self.kernel_file.is_some(),
            self.kernel_json.is_some(),
        ]
        .iter()
        .filter(|b| **b)
        .count();
        if n > 1 {
            bail!("at most one of --preset, --kernel-file, --kernel-json is allowed");
        }
        if n == 0 {
            return Ok(None);
        }
        let k = if let Some(name) = &self.preset {
            kernels::preset(name)?
        } else if let Some(path) = &self.kernel_file {
            kernels::from_file(path)?
        } else {
            kernels::from_json(self.kernel_json.as_deref().expect("checked above"))?
        };
        Ok(Some(k))
    }
}

#[derive(Args)]
pub struct PresetsArgs {
    /// Print presets as compact JSON including kernel matrices.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args)]
pub struct ConfigArgs {
    /// Print only the resolved config file path.
    #[arg(long)]
    pub path: bool,

    /// List configured vision profiles (names only; never keys).
    #[arg(long)]
    pub list: bool,
}

#[derive(Args)]
pub struct OcrArgs {
    #[command(flatten)]
    pub input: InputArgs,

    /// OCR engine: vision model, or the deterministic convolution+OCR backend
    /// (alias: local).
    #[arg(long, value_enum, default_value_t = Engine::ConvOcr)]
    pub engine: Engine,

    /// Vision profile from the config file (--engine vision).
    #[arg(long)]
    pub profile: Option<String>,

    /// Vision model ID (--engine vision; falls back to profile/env/default).
    #[arg(long)]
    pub model: Option<String>,

    /// OpenAI-compatible API base URL (--engine vision).
    #[arg(long)]
    pub base_url: Option<String>,

    /// OpenAI API key (--engine vision).
    #[arg(long)]
    pub api_key: Option<String>,

    /// OCR language(s) for Tesseract, e.g. eng, chi_sim, eng+chi_sim
    /// (convocr only).
    #[arg(long, default_value = "eng")]
    pub lang: String,

    /// Tesseract page segmentation mode 0-13 (convocr only).
    #[arg(long, default_value_t = 3)]
    pub psm: u32,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OcrFormat::Plain)]
    pub format: OcrFormat,
}

#[derive(Args)]
pub struct WindowsArgs {
    /// Print the window list as compact JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(ValueEnum, Clone, Copy)]
pub enum OutputFormat {
    Matrix,
    Ascii,
}

#[derive(ValueEnum, Clone, Copy, PartialEq, Eq)]
pub enum LayoutFormat {
    Json,
    Ascii,
    Both,
    Html,
    Describe,
}

#[derive(ValueEnum, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    /// Vision model backend (requires a configured provider).
    Vision,
    /// Deterministic backend: convolution kernels + local OCR (+ recursive
    /// grid decomposition for layout).
    #[value(name = "convocr", alias = "local")]
    ConvOcr,
}

#[derive(ValueEnum, Clone, Copy)]
pub enum OcrFormat {
    Plain,
    Json,
}

pub fn run_matrix(a: &MatrixArgs) -> Result<()> {
    let proc = load_processed(&a.input, a.stride)?;
    if emit_samples(&a.input, &proc)? {
        return Ok(());
    }
    let data = if a.grayscale {
        proc.image.luma_data()
    } else {
        proc.image.rgb_data()
    };
    println!("{}", output::matrix(&data));
    Ok(())
}

pub fn run_ascii(a: &AsciiArgs) -> Result<()> {
    if a.ramp.is_empty() {
        bail!("--ramp must not be empty");
    }
    let proc = load_processed(&a.input, a.stride)?;
    if emit_samples(&a.input, &proc)? {
        return Ok(());
    }
    let data = proc.image.luma_data();
    if a.ansi {
        print!("{}", output::ascii_ansi(&data, &a.ramp));
    } else {
        print!("{}", output::ascii(&data, &a.ramp));
    }
    Ok(())
}

pub fn run_convolve(a: &ConvolveArgs) -> Result<()> {
    if a.stride == 0 {
        bail!("--stride must be >= 1");
    }
    if matches!(a.format, OutputFormat::Ascii) && a.ramp.is_empty() {
        bail!("--ramp must not be empty");
    }
    let proc = load_processed(&a.input, 1)?;
    if emit_samples(&a.input, &proc)? {
        return Ok(());
    }
    let kernel = a.kernel()?;
    let data = if a.grayscale {
        conv::convolve_luma(&proc.image, &kernel, a.stride)?
    } else if kernel.channels == 1 {
        conv::convolve_rgb(&proc.image, &kernel, a.stride)?
    } else {
        conv::convolve_cross(&proc.image, &kernel, a.stride)?
    };
    match a.format {
        OutputFormat::Matrix => println!("{}", output::matrix(&data)),
        OutputFormat::Ascii => {
            if a.ansi {
                print!("{}", output::ascii_ansi(&data, &a.ramp));
            } else {
                print!("{}", output::ascii(&data, &a.ramp));
            }
        }
    }
    Ok(())
}

pub async fn run_layout(a: &LayoutArgs) -> Result<()> {
    if a.max_blocks == 0 {
        bail!("--max-blocks must be >= 1");
    }
    if a.max_depth == 0 {
        bail!("--max-depth must be >= 1");
    }
    if a.psm > 13 {
        bail!("--psm must be between 0 and 13");
    }
    if a.ocr && a.engine == Engine::Vision {
        bail!("--ocr is only available with --engine convocr");
    }
    // The vision engine receives the image at full stride; --stride is a
    // token-control lever for the local engine only.
    let stride = if a.engine == Engine::Vision {
        1
    } else {
        a.stride
    };
    let proc = load_processed(&a.input, stride)?;
    if emit_samples(&a.input, &proc)? {
        return Ok(());
    }
    let mut format = a.format;
    if a.engine == Engine::Vision && format == LayoutFormat::Both {
        format = LayoutFormat::Html;
    }
    if a.engine == Engine::Vision && format == LayoutFormat::Ascii {
        bail!("--format ascii is only available with --engine convocr");
    }
    if a.engine == Engine::ConvOcr && format == LayoutFormat::Describe {
        bail!("--format describe is only available with --engine vision");
    }
    // Validate kernel flags up front even when the engine ignores the kernel,
    // so conflicts or unknown presets never fail silently.
    let kernel = a.kernel()?;

    match a.engine {
        Engine::Vision => {
            let env = |k: &str| std::env::var(k).ok().filter(|s| !s.is_empty());
            let vision_cfg = config::resolve_vision(
                a.model.as_deref(),
                a.base_url.as_deref(),
                a.api_key.as_deref(),
                a.profile.as_deref(),
                env,
            )?;
            let api_key = vision_cfg.api_key.clone().ok_or_else(|| {
                anyhow!(
                    "an API key is required for --engine vision (pass --api-key, set \
                     OPENAI_API_KEY, or add an api_key to the selected profile in the config \
                     file; see `arwe config`)"
                )
            })?;
            let data_url = ocr::png_data_url(&proc.image)?;
            let output = match format {
                LayoutFormat::Html => vision::VisionOutput::Html,
                LayoutFormat::Json => vision::VisionOutput::Json,
                LayoutFormat::Describe => vision::VisionOutput::Describe,
                _ => unreachable!("format validated above"),
            };
            let raw = vision::analyze(
                &data_url,
                &proc,
                &vision_cfg.model,
                &vision_cfg.base_url,
                &api_key,
                output,
            )
            .await?;
            match format {
                LayoutFormat::Html => println!("{}", vision::finalize_html(&raw, &proc)),
                LayoutFormat::Json => match vision::finalize_json(&raw, &proc) {
                    Ok(s) => println!("{}", s),
                    Err(e) => bail!(
                        "{}\nHint: set ARWE_DEBUG_VISION=1 to inspect the raw model answer; \
                         if the model does not emit strict JSON, use --format html instead.",
                        e
                    ),
                },
                LayoutFormat::Describe => println!("{}", raw),
                _ => unreachable!("format validated above"),
            }
            Ok(())
        }
        Engine::ConvOcr => {
            let image = match kernel {
                Some(k) => {
                    let data = if k.channels == 1 {
                        conv::convolve_rgb(&proc.image, &k, 1)?
                    } else {
                        conv::convolve_cross(&proc.image, &k, 1)?
                    };
                    layout::image_to_rgb(&data)
                }
                None => proc.image.clone(),
            };
            let mut result = layout::analyze(&image, a.max_blocks, a.max_depth);
            if a.ocr {
                let lines = ocr_lines_for_layout(a, &proc)?;
                ocr::fill_text_nodes(&mut result.root, &lines);
            }
            match format {
                LayoutFormat::Html => println!("{}", html::render(&proc, &result.root)),
                LayoutFormat::Json => println!("{}", layout::result_json(&proc, &result)),
                LayoutFormat::Ascii => print!("{}", layout::ascii_map(&proc, &result, a.ansi)),
                LayoutFormat::Both => {
                    println!("{}", layout::result_json(&proc, &result));
                    println!();
                    print!("{}", layout::ascii_map(&proc, &result, a.ansi));
                }
                LayoutFormat::Describe => unreachable!("validated above"),
            }
            Ok(())
        }
    }
}

pub fn run_config(a: &ConfigArgs) -> Result<()> {
    let env = |k: &str| std::env::var(k).ok().filter(|s| !s.is_empty());
    if a.path {
        match config::config_path(&env) {
            Some(p) => println!("{}", p.display()),
            None => bail!(
                "could not determine a config directory (set {})",
                config::CONFIG_ENV
            ),
        }
        return Ok(());
    }
    let cfg = config::load(&env)?;
    let mut names: Vec<&str> = cfg.profiles.keys().map(String::as_str).collect();
    names.sort();
    if a.list {
        for n in names {
            println!("{}", n);
        }
        return Ok(());
    }
    match config::config_path(&env) {
        Some(p) => println!("config file: {}", p.display()),
        None => println!("config file: <none> (set {})", config::CONFIG_ENV),
    }
    if names.is_empty() {
        println!("profiles: (none)");
    } else {
        println!("profiles: {}", names.join(", "));
    }
    if let Some(d) = &cfg.default_profile {
        println!("default profile: {}", d);
    }
    Ok(())
}

/// Run OCR on the original (unresized) cropped image and return lines mapped to
/// processed-image coordinates, ready to fill layout text nodes.
fn ocr_lines_for_layout(a: &LayoutArgs, proc: &Processed) -> Result<Vec<ocr::OcrLine>> {
    let img = image_io::load(&a.input.source()?)?;
    let crop = a.input.crop;
    let (offset_x, offset_y) = crop.map(|c| (c.x, c.y)).unwrap_or((0, 0));
    let crop_img = match crop {
        Some(cr) => resize::crop_image(&img, cr)?,
        None => img,
    };
    let mut lines = ocr::recognize(&crop_img, &a.lang, a.psm)?;
    for l in &mut lines {
        l.x += offset_x;
        l.y += offset_y;
    }
    let mut mapped = Vec::with_capacity(lines.len());
    for l in lines {
        if let Some((x, y, w, h)) = proc.to_processed_rect(l.x, l.y, l.w, l.h) {
            mapped.push(ocr::OcrLine {
                x,
                y,
                w,
                h,
                text: l.text,
                conf: l.conf,
            });
        }
    }
    Ok(mapped)
}

pub async fn run_ocr(a: &OcrArgs) -> Result<()> {
    if a.psm > 13 {
        bail!("--psm must be between 0 and 13");
    }
    let proc = load_processed(&a.input, 1)?;
    if emit_samples(&a.input, &proc)? {
        return Ok(());
    }
    match a.engine {
        Engine::ConvOcr => {
            let lines = ocr::recognize(&proc.image, &a.lang, a.psm)?;
            match a.format {
                OcrFormat::Plain => print!("{}", ocr::render_plain(&lines)),
                OcrFormat::Json => println!("{}", ocr::render_json(&proc, &lines)),
            }
        }
        Engine::Vision => {
            let env = |k: &str| std::env::var(k).ok().filter(|s| !s.is_empty());
            let vision_cfg = config::resolve_vision(
                a.model.as_deref(),
                a.base_url.as_deref(),
                a.api_key.as_deref(),
                a.profile.as_deref(),
                env,
            )?;
            let api_key = vision_cfg.api_key.clone().ok_or_else(|| {
                anyhow!(
                    "an API key is required for --engine vision (pass --api-key, set \
                     OPENAI_API_KEY, or add an api_key to the selected profile in the config \
                     file; see `arwe config`)"
                )
            })?;
            let data_url = ocr::png_data_url(&proc.image)?;
            let raw = vision::analyze(
                &data_url,
                &proc,
                &vision_cfg.model,
                &vision_cfg.base_url,
                &api_key,
                vision::VisionOutput::Ocr,
            )
            .await?;
            let lines = vision::finalize_ocr(&raw)?;
            match a.format {
                OcrFormat::Plain => print!("{}", ocr::render_plain(&lines)),
                OcrFormat::Json => println!("{}", ocr::render_json_vision(&proc, &lines)),
            }
        }
    }
    Ok(())
}

#[derive(Args)]
pub struct DescribeArgs {
    #[command(flatten)]
    pub input: InputArgs,

    /// Description engine: vision model, or a deterministic digest built from
    /// the box tree + OCR text (alias: local).
    #[arg(long, value_enum, default_value_t = Engine::Vision)]
    pub engine: Engine,

    /// Vision profile from the config file (--engine vision).
    #[arg(long)]
    pub profile: Option<String>,

    /// Vision model ID (--engine vision; falls back to profile/env/default).
    #[arg(long)]
    pub model: Option<String>,

    /// OpenAI-compatible API base URL (--engine vision).
    #[arg(long)]
    pub base_url: Option<String>,

    /// OpenAI API key (--engine vision).
    #[arg(long)]
    pub api_key: Option<String>,

    /// OCR language(s) for Tesseract (convocr digest).
    #[arg(long, default_value = "eng")]
    pub lang: String,

    /// Tesseract page segmentation mode 0-13 (convocr digest).
    #[arg(long, default_value_t = 3)]
    pub psm: u32,
}

pub async fn run_describe(a: &DescribeArgs) -> Result<()> {
    if a.psm > 13 {
        bail!("--psm must be between 0 and 13");
    }
    let proc = load_processed(&a.input, 1)?;
    if emit_samples(&a.input, &proc)? {
        return Ok(());
    }
    match a.engine {
        Engine::Vision => {
            let env = |k: &str| std::env::var(k).ok().filter(|s| !s.is_empty());
            let vision_cfg = config::resolve_vision(
                a.model.as_deref(),
                a.base_url.as_deref(),
                a.api_key.as_deref(),
                a.profile.as_deref(),
                env,
            )?;
            let api_key = vision_cfg.api_key.clone().ok_or_else(|| {
                anyhow!(
                    "an API key is required for --engine vision (pass --api-key, set \
                     OPENAI_API_KEY, or add an api_key to the selected profile in the config \
                     file; see `arwe config`)"
                )
            })?;
            let data_url = ocr::png_data_url(&proc.image)?;
            let raw = vision::analyze(
                &data_url,
                &proc,
                &vision_cfg.model,
                &vision_cfg.base_url,
                &api_key,
                vision::VisionOutput::Describe,
            )
            .await?;
            println!("{}", raw);
        }
        Engine::ConvOcr => {
            let result = layout::analyze(
                &proc.image,
                layout::MAX_BLOCKS_DEFAULT,
                layout::MAX_DEPTH_DEFAULT,
            );
            let lines = ocr::recognize(&proc.image, &a.lang, a.psm)?;
            println!("{}", describe::digest(&proc, &result, &lines));
        }
    }
    Ok(())
}

#[derive(Args)]
pub struct ExploreArgs {
    #[command(flatten)]
    pub input: InputArgs,

    /// Default engine for the describe/ask primitives: vision model, or the
    /// deterministic convolution+OCR backend (alias: local).
    #[arg(long, value_enum, default_value_t = Engine::ConvOcr)]
    pub engine: Engine,

    /// Vision profile from the config file.
    #[arg(long)]
    pub profile: Option<String>,

    /// Vision model ID (falls back to profile/env/default).
    #[arg(long)]
    pub model: Option<String>,

    /// OpenAI-compatible API base URL.
    #[arg(long)]
    pub base_url: Option<String>,

    /// OpenAI API key.
    #[arg(long)]
    pub api_key: Option<String>,

    /// Default OCR language(s) for Tesseract (convocr primitives).
    #[arg(long, default_value = "eng")]
    pub lang: String,

    /// Default Tesseract page segmentation mode 0-13.
    #[arg(long, default_value_t = 3)]
    pub psm: u32,
}

pub fn run_explore(a: &ExploreArgs) -> Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(explore::run(a))
}

pub fn run_presets(a: &PresetsArgs) -> Result<()> {
    if a.json {
        println!("{}", kernels::presets_json());
    } else {
        for p in kernels::presets() {
            println!("{:<12} {}", p.name, p.description);
        }
        println!("\nUse `arwe presets --json` to get kernel matrices.");
    }
    Ok(())
}

pub fn run_windows(a: &WindowsArgs) -> Result<()> {
    let windows = capture::list_windows()?;
    if a.json {
        println!(
            "{}",
            serde_json::to_string(&windows).context("serializing window list")?
        );
    } else {
        for w in &windows {
            println!(
                "[{}] {} ({}) {}x{} @({},{}){}",
                w.index,
                w.title,
                w.app_name,
                w.width,
                w.height,
                w.x,
                w.y,
                if w.focused { " [focused]" } else { "" }
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> InputArgs {
        InputArgs {
            input: None,
            base64: None,
            stdin: false,
            screen: None,
            window: None,
            width: None,
            crop: None,
            sample: Vec::new(),
        }
    }

    #[test]
    fn exactly_one_input_source_required() {
        assert!(base().source().is_err());
    }

    #[test]
    fn screenshot_sources_conflict_with_file_inputs() {
        let a = InputArgs {
            input: Some("a.png".into()),
            screen: Some(None),
            ..base()
        };
        assert!(a.source().is_err());

        let a = InputArgs {
            stdin: true,
            window: Some("x".into()),
            ..base()
        };
        assert!(a.source().is_err());

        let a = InputArgs {
            screen: Some(Some(0)),
            window: Some("x".into()),
            ..base()
        };
        assert!(a.source().is_err());
    }

    #[test]
    fn screen_and_window_sources_resolve() {
        let a = InputArgs {
            screen: Some(None),
            ..base()
        };
        assert!(matches!(a.source(), Ok(InputSource::Screen(None))));

        let a = InputArgs {
            screen: Some(Some(1)),
            ..base()
        };
        assert!(matches!(a.source(), Ok(InputSource::Screen(Some(1)))));

        let a = InputArgs {
            window: Some("Chrome".into()),
            ..base()
        };
        assert!(matches!(a.source(), Ok(InputSource::Window(t)) if t == "Chrome"));
    }
}
