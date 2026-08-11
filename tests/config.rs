use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_arwe"))
}

fn config_toml() -> &'static str {
    r#"
default_profile = "local"

[profiles.openai]
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"
api_key = "sk-super-secret"

[profiles.local]
base_url = "http://127.0.0.1:8000/v1"
model = "qwen2.5-vl-7b-instruct"
"#
}

#[test]
fn config_path_flag_prints_override_path() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("my-config.toml");
    std::fs::write(&cfg, config_toml()).unwrap();
    let out = bin()
        .args(["config", "--path"])
        .env("ARWE_CONFIG", &cfg)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        cfg.display().to_string()
    );
}

#[test]
fn config_list_shows_profiles_but_never_keys() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("config.toml");
    std::fs::write(&cfg, config_toml()).unwrap();
    let out = bin()
        .args(["config", "--list"])
        .env("ARWE_CONFIG", &cfg)
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let names: Vec<&str> = stdout.lines().collect();
    assert!(names.contains(&"openai"), "stdout: {}", stdout);
    assert!(names.contains(&"local"), "stdout: {}", stdout);
    assert!(
        !stdout.contains("sk-super-secret"),
        "keys leaked: {}",
        stdout
    );
}

#[test]
fn config_default_prints_summary() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("config.toml");
    std::fs::write(&cfg, config_toml()).unwrap();
    let out = bin()
        .args(["config"])
        .env("ARWE_CONFIG", &cfg)
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("config file:"));
    assert!(stdout.contains("profiles: local, openai"));
    assert!(stdout.contains("default profile: local"));
    assert!(!stdout.contains("sk-super-secret"));
}

#[test]
fn config_parse_error_is_friendly() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("config.toml");
    std::fs::write(&cfg, "[profiles.openai\nbase_url =").unwrap();
    let out = bin()
        .args(["config"])
        .env("ARWE_CONFIG", &cfg)
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("config"), "stderr: {}", stderr);
    assert!(stderr.contains("parse"), "stderr: {}", stderr);
}

#[test]
fn unknown_profile_fails_with_available_names() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("config.toml");
    std::fs::write(&cfg, config_toml()).unwrap();
    let b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
    let out = bin()
        .args([
            "layout",
            "--engine",
            "vision",
            "--profile",
            "nope",
            "--base64",
            b64,
        ])
        .env("ARWE_CONFIG", &cfg)
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("'nope'"), "stderr: {}", stderr);
    assert!(stderr.contains("openai"), "stderr: {}", stderr);
    assert!(stderr.contains("local"), "stderr: {}", stderr);
}
