use std::io::Write;
use std::process::{Command, Stdio};

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_arwe"))
}

fn png_base64(pixels: &[[u8; 3]], w: u32, h: u32) -> String {
    let flat: Vec<u8> = pixels.iter().flatten().copied().collect();
    let buf = image::RgbImage::from_raw(w, h, flat).unwrap();
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(buf)
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .unwrap();
    STANDARD.encode(&bytes)
}

fn run_ok(args: &[&str]) -> String {
    let out = bin().args(args).output().unwrap();
    assert!(
        out.status.success(),
        "args {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn run_fails(args: &[&str]) {
    let out = bin().args(args).output().unwrap();
    assert!(
        !out.status.success(),
        "args {:?} unexpectedly succeeded: {}",
        args,
        String::from_utf8_lossy(&out.stdout)
    );
}

fn run_fails_with(args: &[&str], needle: &str) {
    let out = bin().args(args).output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success() && stderr.contains(needle),
        "args {:?}: expected failure mentioning {:?}, got status {} stderr: {}",
        args,
        needle,
        out.status,
        stderr
    );
}

fn rgb2x2() -> String {
    png_base64(
        &[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
        2,
        2,
    )
}

#[test]
fn cli_matrix_rgb() {
    let out = run_ok(&["matrix", "--base64", &rgb2x2()]);
    assert_eq!(out, "[[[255,0,0],[0,255,0]],[[0,0,255],[255,255,255]]]\n");
}

#[test]
fn cli_matrix_grayscale() {
    let out = run_ok(&["matrix", "--base64", &rgb2x2(), "--grayscale"]);
    assert_eq!(out, "[[76,150],[29,255]]\n");
}

#[test]
fn cli_ascii() {
    let b64 = png_base64(&[[0, 0, 0], [128, 128, 128], [255, 255, 255]], 3, 1);
    let out = run_ok(&["ascii", "--base64", &b64]);
    assert_eq!(out, " =@\n");
}

#[test]
fn cli_convolve_custom_json_stride() {
    // Non-uniform 4x4 image, identity kernel, stride 2: output is every other pixel.
    let pixels: Vec<[u8; 3]> = (1..=16u8).map(|v| [v, v, v]).collect();
    let b64 = png_base64(&pixels, 4, 4);
    let out = run_ok(&[
        "convolve",
        "--base64",
        &b64,
        "--kernel-json",
        "[[0,0,0],[0,1,0],[0,0,0]]",
        "--stride",
        "2",
    ]);
    assert_eq!(out, "[[[1,1,1],[3,3,3]],[[9,9,9],[11,11,11]]]\n");
}

#[test]
fn cli_convolve_preset_ascii() {
    let b64 = png_base64(
        &[[0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255]],
        2,
        2,
    );
    let out = run_ok(&[
        "convolve", "--base64", &b64, "--preset", "sobel-x", "--format", "ascii",
    ]);
    let valid: Vec<char> = " .:-=+*#%@\n".chars().collect();
    assert!(out.chars().all(|c| valid.contains(&c)));
    assert_eq!(out.len(), 6); // 2 rows of 2 chars plus 2 newlines
}

#[test]
fn cli_convolve_cross_luma() {
    let b64 = png_base64(&[[255, 0, 0], [0, 255, 0]], 2, 1);
    let out = run_ok(&["convolve", "--base64", &b64, "--preset", "luma"]);
    assert_eq!(out, "[[76,150]]\n");
}

#[test]
fn cli_stdin() {
    let b64 = rgb2x2();
    let mut child = bin()
        .args(["matrix", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b64.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[[[255,0,0],[0,255,0]],[[0,0,255],[255,255,255]]]\n"
    );
}

#[test]
fn cli_presets() {
    let out = run_ok(&["presets"]);
    assert!(out.contains("sobel-x"));
    assert!(out.contains("luma"));
    let json = run_ok(&["presets", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["luma"]["channels"], 3);
    assert_eq!(v["sobel-x"]["kernel"][1][2], 2.0);
}

#[test]
fn cli_error_paths() {
    let b64 = rgb2x2();
    // Missing input.
    run_fails_with(&["matrix"], "exactly one of these flags is required");
    // Conflicting inputs.
    run_fails_with(
        &["matrix", "--input", "a.png", "--base64", &b64],
        "conflicting flags",
    );
    // Screenshot flags participate in the same exclusivity check.
    run_fails_with(
        &["matrix", "--screen", "--input", "a.png"],
        "conflicting flags",
    );
    run_fails_with(
        &["matrix", "--window", "x", "--screen"],
        "conflicting flags",
    );
    // Missing kernel selector.
    run_fails(&["convolve", "--base64", &b64]);
    // Unknown preset.
    run_fails(&["convolve", "--preset", "nope", "--base64", &b64]);
    // Stride zero.
    run_fails(&[
        "convolve", "--preset", "sobel-x", "--base64", &b64, "--stride", "0",
    ]);
    // Grayscale mode with a cross-channel kernel.
    run_fails(&[
        "convolve",
        "--preset",
        "luma",
        "--grayscale",
        "--base64",
        &b64,
    ]);
    // Invalid base64.
    run_fails(&["matrix", "--base64", "not-base64!!!"]);
    // Base64 that is not an image.
    let not_image = STANDARD.encode(b"hello");
    run_fails(&["ascii", "--base64", &not_image]);
    // Empty ramp.
    run_fails(&["ascii", "--base64", &b64, "--ramp", ""]);
}

fn half4x4() -> String {
    let row = [[255, 0, 0], [255, 0, 0], [0, 0, 255], [0, 0, 255]];
    png_base64(
        &row.iter().cycle().take(16).copied().collect::<Vec<_>>(),
        4,
        4,
    )
}

fn grad16() -> String {
    png_base64(&(1..=16u8).map(|v| [v, v, v]).collect::<Vec<_>>(), 4, 4)
}

fn half8x8() -> String {
    let row = [
        [255, 0, 0],
        [255, 0, 0],
        [255, 0, 0],
        [255, 0, 0],
        [0, 0, 255],
        [0, 0, 255],
        [0, 0, 255],
        [0, 0, 255],
    ];
    png_base64(
        &row.iter().cycle().take(64).copied().collect::<Vec<_>>(),
        8,
        8,
    )
}

#[test]
fn cli_layout_json() {
    let b64 = half8x8();
    let out = run_ok(&["layout", "--base64", &b64, "--format", "json"]);
    let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
    assert_eq!(v["width"], 8);
    assert_eq!(v["height"], 8);
    assert_eq!(v["root"]["kind"], "group");
    assert_eq!(v["root"]["direction"], "row");
    assert_eq!(v["root"]["children"].as_array().unwrap().len(), 2);
}

#[test]
fn cli_layout_ascii_map() {
    let b64 = half8x8();
    let out = run_ok(&["layout", "--base64", &b64, "--format", "ascii"]);
    assert!(out.starts_with("# 8x8\n"));
    assert!(out.contains("A: 0,0,4,8 #ff0000\n"));
    assert!(out.contains("B: 4,0,4,8 #0000ff\n"));
}

#[test]
fn cli_layout_both_outputs() {
    let b64 = half8x8();
    let out = run_ok(&["layout", "--base64", &b64]);
    assert!(out.contains("\"root\""));
    assert!(out.contains("# 8x8\n"));
}

#[test]
fn cli_matrix_width_resizes() {
    let b64 = half4x4();
    let out = run_ok(&["matrix", "--base64", &b64, "--width", "2"]);
    assert_eq!(out, "[[[255,0,0],[0,0,255]],[[255,0,0],[0,0,255]]]\n");
}

#[test]
fn cli_matrix_crop() {
    let b64 = grad16();
    let out = run_ok(&["matrix", "--base64", &b64, "--crop", "1,1,2,2"]);
    assert_eq!(out, "[[[6,6,6],[7,7,7]],[[10,10,10],[11,11,11]]]\n");
}

#[test]
fn cli_matrix_stride() {
    let b64 = grad16();
    let out = run_ok(&["matrix", "--base64", &b64, "--stride", "2"]);
    assert_eq!(out, "[[[1,1,1],[3,3,3]],[[9,9,9],[11,11,11]]]\n");
}

#[test]
fn cli_sample_outputs_original_coordinates() {
    let b64 = grad16();
    let out = run_ok(&[
        "matrix", "--base64", &b64, "--stride", "2", "--sample", "2,0", "--sample", "0,3",
    ]);
    assert_eq!(out, "2,0 #030303\n0,3 #090909\n");
}

#[test]
fn cli_layout_global_flag_errors() {
    let b64 = half4x4();
    run_fails(&["matrix", "--base64", &b64, "--width", "0"]);
    run_fails(&["matrix", "--base64", &b64, "--crop", "9,9,2,2"]);
    run_fails(&["matrix", "--base64", &b64, "--sample", "99,99"]);
    run_fails(&["layout", "--base64", &b64, "--max-blocks", "0"]);
    run_fails(&["layout", "--base64", &b64, "--max-depth", "0"]);
    run_fails(&[
        "layout",
        "--base64",
        &b64,
        "--preset",
        "sobel-x",
        "--kernel-json",
        "[[1]]",
    ]);
}

#[test]
fn cli_layout_with_preset_preprocessing() {
    let b64 = half8x8();
    let out = run_ok(&[
        "layout", "--base64", &b64, "--preset", "sobel-x", "--format", "json",
    ]);
    let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
    assert_eq!(v["width"], 8);
}
