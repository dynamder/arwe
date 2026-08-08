use anyhow::Result;
use clap::{Parser, Subcommand};

use arwe::cli::{
    self, AsciiArgs, ConfigArgs, ConvolveArgs, LayoutArgs, MatrixArgs, OcrArgs, PresetsArgs,
    WindowsArgs,
};

#[derive(Parser)]
#[command(
    name = "arwe",
    version,
    about = "Artificial Whale Eye: eyes for text-only LLMs (DeepSeek et al.).\nConvolve RGB matrices and render them as compact matrices, ASCII art,\nHTML layout skeletons, or OCR text."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Output the image as a compact RGB matrix.
    Matrix(MatrixArgs),
    /// Render the image as ASCII art.
    Ascii(AsciiArgs),
    /// Convolve the image with a preset or custom kernel.
    Convolve(ConvolveArgs),
    /// Analyze UI/page layout: recursive grid decomposition into nested boxes.
    Layout(LayoutArgs),
    /// Recognize text in the image with local Tesseract.
    Ocr(OcrArgs),
    /// Show the vision config file location and profiles.
    Config(ConfigArgs),
    /// List available preset kernels.
    Presets(PresetsArgs),
    /// List windows that can be captured with --window.
    Windows(WindowsArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Matrix(a) => cli::run_matrix(&a),
        Command::Ascii(a) => cli::run_ascii(&a),
        Command::Convolve(a) => cli::run_convolve(&a),
        Command::Layout(a) => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            rt.block_on(cli::run_layout(&a))
        }
        Command::Ocr(a) => cli::run_ocr(&a),
        Command::Config(a) => cli::run_config(&a),
        Command::Presets(a) => cli::run_presets(&a),
        Command::Windows(a) => cli::run_windows(&a),
    }
}
