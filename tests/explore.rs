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

fn grad4x4() -> String {
    let pixels: Vec<[u8; 3]> = (1..=16u8).map(|v| [v, v, v]).collect();
    png_base64(&pixels, 4, 4)
}

fn run_with_stdin(args: &[&str], stdin: &str) -> std::process::Output {
    let mut child = bin()
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(stdin.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn explore_primitive_commands_via_pipe() {
    let b64 = grad4x4();
    let stdin = concat!(
        "pixel 0,0\n",
        "region 0,0,2,2 --width 2 --matrix\n",
        "zoom 1,1,2,2\n",
        "region\n",
        "reset\n",
        "kernel sobel-x --stride 2 --ascii\n",
        "help\n",
        "quit\n",
    );
    let out = run_with_stdin(&["explore", "--base64", &b64], stdin);
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("0,0 #010101"), "pixel output: {}", stdout);
    assert!(
        stdout.contains("[[[1,1,1],[2,2,2]],[[5,5,5],[6,6,6]]]"),
        "region matrix output: {}",
        stdout
    );
    assert!(stdout.contains("region: 1,1 2x2"), "zoom state: {}", stdout);
    assert!(
        stdout.contains("region: full 4x4"),
        "reset state: {}",
        stdout
    );
    assert!(
        stdout.contains("explore primitives"),
        "help output: {}",
        stdout
    );
    assert!(
        !String::from_utf8_lossy(&out.stderr).contains("error:"),
        "no errors expected: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn explore_invalid_command_does_not_kill_session() {
    let b64 = grad4x4();
    let out = run_with_stdin(&["explore", "--base64", &b64], "bogus\npixel 0,0\nquit\n");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(out.status.success(), "session must survive: {}", stderr);
    assert!(
        stderr.contains("unknown command 'bogus'"),
        "stderr: {}",
        stderr
    );
    assert!(stdout.contains("0,0 #010101"), "stdout: {}", stdout);
}

#[test]
fn explore_ocr_uses_mock_tesseract() {
    let tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n5\t1\t1\t1\t1\t1\t10\t20\t100\t30\t90.5\tHello\n5\t1\t1\t1\t1\t2\t120\t20\t80\t30\t85.2\tworld";
    let dir = tempfile::tempdir().unwrap();
    let bat = dir.path().join("tess.bat");
    let lines: Vec<String> = tsv.lines().map(|l| format!("echo {}", l)).collect();
    let content = format!("@echo off\r\n{}\r\n", lines.join("\r\n"));
    std::fs::write(&bat, content).unwrap();
    let b64 = grad4x4();
    let mut child = bin()
        .args(["explore", "--base64", &b64])
        .env("ARWE_TESSERACT_BIN", &bat)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"ocr\nquit\n")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("Hello world"), "ocr output: {}", stdout);
}
