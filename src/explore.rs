use std::io::{IsTerminal, Write};

use anyhow::{anyhow, bail, Context, Result};

use crate::cli::{Engine, ExploreArgs, DEFAULT_RAMP};
use crate::config;
use crate::conv;
use crate::image_io;
use crate::kernels;
use crate::layout;
use crate::ocr;
use crate::output;
use crate::resize::{self, CropRect};
use crate::rgb::RgbImage;
use crate::vision;

/// Stateful exploration session: the loaded image plus the current viewport
/// (region in original-image pixels). Primitives act on the current region.
struct Session {
    img: RgbImage,
    region: Option<CropRect>,
    engine: Engine,
    lang: String,
    psm: u32,
    profile: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
}

enum RegionFormat {
    Ascii,
    Matrix,
}

/// Run the explore REPL: read commands from stdin until EOF or `quit`.
/// stdout carries only command results; the prompt and errors go to stderr.
pub async fn run(a: &ExploreArgs) -> Result<()> {
    let img = image_io::load(&a.input.source()?)?;
    let mut session = Session {
        img,
        region: None,
        engine: a.engine,
        lang: a.lang.clone(),
        psm: a.psm,
        profile: a.profile.clone(),
        model: a.model.clone(),
        base_url: a.base_url.clone(),
        api_key: a.api_key.clone(),
    };
    let stdin = std::io::stdin();
    let interactive = stdin.is_terminal();
    let mut line = String::new();
    loop {
        if interactive {
            eprint!("explore> ");
            std::io::stderr().flush().context("flushing prompt")?;
        }
        line.clear();
        let n = stdin.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        match run_command(&mut session, text).await {
            Ok(true) => break,
            Ok(false) => {}
            Err(e) => eprintln!("error: {}", e),
        }
    }
    Ok(())
}

/// Returns `true` when the session should exit.
async fn run_command(s: &mut Session, text: &str) -> Result<bool> {
    let mut toks = text.split_whitespace();
    let cmd = toks.next().unwrap_or("");
    match cmd {
        "quit" | "exit" => return Ok(true),
        "help" => {
            print_help();
            return Ok(false);
        }
        "pixel" => {
            let rest = text["pixel".len()..].trim();
            let (x, y) =
                resize::parse_point(rest).map_err(|e| anyhow!("pixel needs X,Y: {}", e))?;
            let c = s.img.pixel(x, y).ok_or_else(|| {
                anyhow!(
                    "pixel ({},{}) is outside the {}x{} image",
                    x,
                    y,
                    s.img.width,
                    s.img.height
                )
            })?;
            println!("{},{} #{:02x}{:02x}{:02x}", x, y, c[0], c[1], c[2]);
        }
        "region" | "zoom" => cmd_region(s, text[cmd.len()..].trim())?,
        "reset" => {
            s.region = None;
            println!("region: full {}x{}", s.img.width, s.img.height);
        }
        "kernel" => cmd_kernel(s, text["kernel".len()..].trim())?,
        "ocr" => cmd_ocr(s, text["ocr".len()..].trim())?,
        "describe" => cmd_describe(s, text["describe".len()..].trim()).await?,
        "ask" => cmd_ask(s, text["ask".len()..].trim()).await?,
        other => bail!("unknown command '{}' (try help)", other),
    }
    Ok(false)
}

fn print_help() {
    println!("explore primitives (stdout carries only results):");
    println!("  pixel X,Y");
    println!("  region [X,Y,W,H] [--width N] [--ascii|--matrix]");
    println!("  zoom X,Y,W,H");
    println!("  reset");
    println!("  kernel <preset|json> [--stride N] [--ascii|--matrix]");
    println!("  ocr [--lang L] [--psm N]");
    println!("  describe [--engine vision|convocr] [--width N]");
    println!("  ask [--engine vision] [--width N] <prompt>");
    println!("  help | quit");
}

fn cmd_region(s: &mut Session, rest: &str) -> Result<()> {
    let toks: Vec<&str> = rest.split_whitespace().collect();
    let mut coords: Option<CropRect> = None;
    let mut width: Option<usize> = None;
    let mut fmt = RegionFormat::Ascii;
    let mut render = false;
    let mut i = 0;
    if let Some(first) = toks.first() {
        if !first.starts_with("--") {
            coords = Some(
                resize::parse_crop(first).map_err(|e| anyhow!("region needs X,Y,W,H: {}", e))?,
            );
            i = 1;
        }
    }
    while i < toks.len() {
        match toks[i] {
            "--width" => {
                i += 1;
                let v = toks
                    .get(i)
                    .ok_or_else(|| anyhow!("--width needs a value"))?;
                width = Some(v.parse().context("--width must be a number")?);
                render = true;
            }
            "--ascii" => {
                fmt = RegionFormat::Ascii;
                render = true;
            }
            "--matrix" => {
                fmt = RegionFormat::Matrix;
                render = true;
            }
            other => bail!("unknown region flag '{}'", other),
        }
        i += 1;
    }
    if let Some(c) = coords {
        s.region = Some(c);
    }
    match s.region {
        Some(r) => println!("region: {},{} {}x{}", r.x, r.y, r.w, r.h),
        None => println!("region: full {}x{}", s.img.width, s.img.height),
    }
    if render {
        let proc = resize::process(&s.img, s.region, width, 1)?;
        render_region(&proc, &fmt);
    }
    Ok(())
}

fn cmd_kernel(s: &mut Session, rest: &str) -> Result<()> {
    let toks: Vec<&str> = rest.split_whitespace().collect();
    let mut kernel_arg: Option<String> = None;
    let mut stride = 1usize;
    let mut fmt = RegionFormat::Ascii;
    let mut i = 0;
    if let Some(first) = toks.first() {
        if !first.starts_with("--") {
            kernel_arg = Some(first.to_string());
            i = 1;
        }
    }
    while i < toks.len() {
        match toks[i] {
            "--stride" => {
                i += 1;
                let v = toks
                    .get(i)
                    .ok_or_else(|| anyhow!("--stride needs a value"))?;
                stride = v.parse().context("--stride must be a number")?;
            }
            "--ascii" => fmt = RegionFormat::Ascii,
            "--matrix" => fmt = RegionFormat::Matrix,
            other => bail!("unknown kernel flag '{}'", other),
        }
        i += 1;
    }
    let arg = kernel_arg.ok_or_else(|| anyhow!("kernel needs a preset name or inline JSON"))?;
    let kernel = if arg.starts_with('[') || arg.starts_with('{') {
        kernels::from_json(&arg)?
    } else {
        kernels::preset(&arg)?
    };
    let proc = resize::process(&s.img, s.region, None, 1)?;
    let data = if kernel.channels == 1 {
        conv::convolve_rgb(&proc.image, &kernel, stride)?
    } else {
        conv::convolve_cross(&proc.image, &kernel, stride)?
    };
    render_region_data(&data, &fmt);
    Ok(())
}

fn cmd_ocr(s: &mut Session, rest: &str) -> Result<()> {
    let toks: Vec<&str> = rest.split_whitespace().collect();
    let mut lang = s.lang.clone();
    let mut psm = s.psm;
    let mut i = 0;
    while i < toks.len() {
        match toks[i] {
            "--lang" => {
                i += 1;
                lang = toks
                    .get(i)
                    .ok_or_else(|| anyhow!("--lang needs a value"))?
                    .to_string();
            }
            "--psm" => {
                i += 1;
                let v = toks.get(i).ok_or_else(|| anyhow!("--psm needs a value"))?;
                psm = v.parse().context("--psm must be a number")?;
            }
            other => bail!("unknown ocr flag '{}'", other),
        }
        i += 1;
    }
    if psm > 13 {
        bail!("--psm must be between 0 and 13");
    }
    let img = match s.region {
        Some(r) => resize::crop_image(&s.img, r)?,
        None => s.img.clone(),
    };
    let lines = ocr::recognize(&img, &lang, psm)?;
    print!("{}", ocr::render_plain(&lines));
    Ok(())
}

async fn cmd_describe(s: &mut Session, rest: &str) -> Result<()> {
    let toks: Vec<&str> = rest.split_whitespace().collect();
    let mut engine = s.engine;
    let mut width: Option<usize> = None;
    let mut i = 0;
    while i < toks.len() {
        match toks[i] {
            "--engine" => {
                i += 1;
                let v = toks
                    .get(i)
                    .ok_or_else(|| anyhow!("--engine needs a value"))?;
                engine = parse_engine(v)?;
            }
            "--width" => {
                i += 1;
                let v = toks
                    .get(i)
                    .ok_or_else(|| anyhow!("--width needs a value"))?;
                width = Some(v.parse().context("--width must be a number")?);
            }
            other => bail!("unknown describe flag '{}'", other),
        }
        i += 1;
    }
    let proc = resize::process(&s.img, s.region, width, 1)?;
    match engine {
        Engine::Vision => {
            let cfg = resolve_vision(s)?;
            let api_key = cfg
                .api_key
                .clone()
                .ok_or_else(|| anyhow!("an API key is required for --engine vision (pass --api-key, set OPENAI_API_KEY, or add an api_key to the selected profile in the config file; see `arwe config`)"))?;
            let data_url = ocr::png_data_url(&proc.image)?;
            let raw = vision::analyze(
                &data_url,
                &proc,
                &cfg.model,
                &cfg.base_url,
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
            let lines = ocr::recognize(&proc.image, &s.lang, s.psm)?;
            print!("{}", crate::describe::digest(&proc, &result, &lines));
        }
    }
    Ok(())
}

async fn cmd_ask(s: &mut Session, rest: &str) -> Result<()> {
    let toks: Vec<&str> = rest.split_whitespace().collect();
    let mut engine = s.engine;
    let mut width: Option<usize> = None;
    let mut prompt_toks: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        match toks[i] {
            "--engine" => {
                i += 1;
                let v = toks
                    .get(i)
                    .ok_or_else(|| anyhow!("--engine needs a value"))?;
                engine = parse_engine(v)?;
            }
            "--width" => {
                i += 1;
                let v = toks
                    .get(i)
                    .ok_or_else(|| anyhow!("--width needs a value"))?;
                width = Some(v.parse().context("--width must be a number")?);
            }
            other if other.starts_with("--") => bail!("unknown ask flag '{}'", other),
            other => prompt_toks.push(other),
        }
        i += 1;
    }
    if engine != Engine::Vision {
        bail!(
            "ask requires the vision engine (start explore with --engine vision or pass --engine vision)"
        );
    }
    if prompt_toks.is_empty() {
        bail!("ask needs a prompt after the command");
    }
    let prompt = prompt_toks.join(" ");
    let proc = resize::process(&s.img, s.region, width, 1)?;
    let cfg = resolve_vision(s)?;
    let api_key = cfg
        .api_key
        .clone()
        .ok_or_else(|| anyhow!("an API key is required for --engine vision (pass --api-key, set OPENAI_API_KEY, or add an api_key to the selected profile in the config file; see `arwe config`)"))?;
    let answer = vision::ask(&proc, &prompt, &cfg.model, &cfg.base_url, &api_key).await?;
    println!("{}", answer);
    Ok(())
}

fn render_region(proc: &resize::Processed, fmt: &RegionFormat) {
    match fmt {
        RegionFormat::Ascii => {
            print!("{}", output::ascii(&proc.image.luma_data(), DEFAULT_RAMP))
        }
        RegionFormat::Matrix => println!("{}", output::matrix(&proc.image.rgb_data())),
    }
}

fn render_region_data(data: &crate::rgb::ImageData, fmt: &RegionFormat) {
    match fmt {
        RegionFormat::Ascii => print!("{}", output::ascii(data, DEFAULT_RAMP)),
        RegionFormat::Matrix => println!("{}", output::matrix(data)),
    }
}

fn parse_engine(v: &str) -> Result<Engine> {
    match v {
        "vision" => Ok(Engine::Vision),
        "convocr" | "local" => Ok(Engine::ConvOcr),
        _ => bail!("unknown engine '{}' (vision|convocr)", v),
    }
}

fn resolve_vision(s: &Session) -> Result<config::VisionConfig> {
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    config::resolve_vision(
        s.model.as_deref(),
        s.base_url.as_deref(),
        s.api_key.as_deref(),
        s.profile.as_deref(),
        env,
    )
}
