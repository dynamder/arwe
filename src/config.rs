use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

use crate::vision;

/// Environment variable pointing at the config file directly.
pub const CONFIG_ENV: &str = "ARWE_CONFIG";
/// Environment variable selecting the default profile (weaker than --profile).
pub const PROFILE_ENV: &str = "ARWE_PROFILE";

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub default_profile: Option<String>,
    #[serde(default)]
    pub profiles: HashMap<String, Profile>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Profile {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
}

/// Fully resolved vision settings for one call.
#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub model: String,
    pub base_url: String,
    pub api_key: Option<String>,
}

/// Resolved config file location:
/// `ARWE_CONFIG` when set, otherwise the platform config directory
/// (Windows `%APPDATA%\arwe\config.toml`, Linux
/// `$XDG_CONFIG_HOME`/`~/.config`/arwe/config.toml, macOS
/// `~/Library/Application Support/arwe/config.toml`).
pub fn config_path(env: &impl Fn(&str) -> Option<String>) -> Option<PathBuf> {
    if let Some(p) = env(CONFIG_ENV) {
        return Some(PathBuf::from(p));
    }
    #[cfg(target_os = "macos")]
    if let Some(home) = env("HOME") {
        return Some(
            PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("arwe")
                .join("config.toml"),
        );
    }
    #[cfg(windows)]
    if let Some(appdata) = env("APPDATA") {
        return Some(PathBuf::from(appdata).join("arwe").join("config.toml"));
    }
    #[cfg(unix)]
    {
        if let Some(xdg) = env("XDG_CONFIG_HOME") {
            return Some(PathBuf::from(xdg).join("arwe").join("config.toml"));
        }
        if let Some(home) = env("HOME") {
            return Some(
                PathBuf::from(home)
                    .join(".config")
                    .join("arwe")
                    .join("config.toml"),
            );
        }
    }
    None
}

/// Load the config file. A missing file is not an error — it yields an empty
/// config so vision still works through flags and environment variables.
pub fn load(env: &impl Fn(&str) -> Option<String>) -> Result<Config> {
    let Some(path) = config_path(env) else {
        return Ok(Config::default());
    };
    if !path.exists() {
        return Ok(Config::default());
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read config '{}'", path.display()))?;
    parse(&text).with_context(|| format!("failed to parse config '{}'", path.display()))
}

pub fn parse(text: &str) -> Result<Config> {
    Ok(toml::from_str(text)?)
}

/// Resolve vision settings with the precedence
/// `--model/--base-url/--api-key` > selected profile > environment variables
/// (`OPENAI_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`) > built-in defaults.
/// The profile is picked from `--profile`, then `ARWE_PROFILE`, then the
/// config file's `default_profile`.
pub fn resolve_vision(
    cli_model: Option<&str>,
    cli_base_url: Option<&str>,
    cli_api_key: Option<&str>,
    cli_profile: Option<&str>,
    env: impl Fn(&str) -> Option<String>,
) -> Result<VisionConfig> {
    let config = load(&env)?;
    let profile_name = cli_profile
        .map(str::to_string)
        .or_else(|| env(PROFILE_ENV))
        .or_else(|| config.default_profile.clone());
    let profile = match &profile_name {
        Some(name) => {
            let path = config_path(&env)
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "<none>".to_string());
            let p = config.profiles.get(name).ok_or_else(|| {
                let mut names: Vec<&str> = config.profiles.keys().map(String::as_str).collect();
                names.sort();
                let available = if names.is_empty() {
                    "none".to_string()
                } else {
                    names.join(", ")
                };
                anyhow!(
                    "vision profile '{}' not found in config '{}' (available: {})",
                    name,
                    path,
                    available
                )
            })?;
            Some(p)
        }
        None => None,
    };
    let model = cli_model
        .map(str::to_string)
        .or_else(|| profile.and_then(|p| p.model.clone()))
        .or_else(|| env("OPENAI_MODEL").filter(|s| !s.is_empty()))
        .unwrap_or_else(|| vision::DEFAULT_MODEL.to_string());
    let base_url = cli_base_url
        .map(str::to_string)
        .or_else(|| profile.and_then(|p| p.base_url.clone()))
        .or_else(|| env("OPENAI_BASE_URL").filter(|s| !s.is_empty()))
        .unwrap_or_else(|| vision::DEFAULT_BASE_URL.to_string());
    let api_key = cli_api_key
        .map(str::to_string)
        .or_else(|| profile.and_then(|p| p.api_key.clone()))
        .or_else(|| env("OPENAI_API_KEY").filter(|s| !s.is_empty()));
    Ok(VisionConfig {
        model,
        base_url,
        api_key,
    })
}

#[cfg(test)]
mod unit {
    use super::*;

    fn env_from<'a>(
        map: &'a std::collections::HashMap<&'static str, String>,
    ) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| map.get(k).cloned()
    }

    fn empty_env() -> impl Fn(&str) -> Option<String> {
        |_| None
    }

    #[test]
    fn parses_profiles_and_default() {
        let cfg = parse(
            r#"
default_profile = "local"

[profiles.openai]
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"
api_key = "sk-123"

[profiles.local]
base_url = "http://127.0.0.1:8000/v1"
"#,
        )
        .unwrap();
        assert_eq!(cfg.default_profile.as_deref(), Some("local"));
        assert_eq!(cfg.profiles.len(), 2);
        assert_eq!(cfg.profiles["openai"].api_key.as_deref(), Some("sk-123"));
        assert_eq!(cfg.profiles["local"].model, None);
    }

    #[test]
    fn invalid_toml_is_an_error() {
        assert!(parse("[profiles.openai\nbase_url =").is_err());
    }

    #[test]
    fn cli_flags_win_over_profile_and_env() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
[profiles.p]
base_url = "http://profile/v1"
model = "profile-model"
api_key = "profile-key"
"#,
        )
        .unwrap();
        let mut map = std::collections::HashMap::new();
        map.insert("ARWE_CONFIG", path.display().to_string());
        map.insert("OPENAI_MODEL", "env-model".to_string());
        let env = env_from(&map);
        let v = resolve_vision(
            Some("cli-model"),
            Some("http://cli/v1"),
            Some("cli-key"),
            Some("p"),
            env,
        )
        .unwrap();
        assert_eq!(v.model, "cli-model");
        assert_eq!(v.base_url, "http://cli/v1");
        assert_eq!(v.api_key.as_deref(), Some("cli-key"));
    }

    #[test]
    fn profile_wins_over_env() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
[profiles.p]
base_url = "http://profile/v1"
model = "profile-model"
api_key = "profile-key"
"#,
        )
        .unwrap();
        let mut map = std::collections::HashMap::new();
        map.insert("ARWE_CONFIG", path.display().to_string());
        map.insert("OPENAI_API_KEY", "env-key".to_string());
        let v = resolve_vision(None, None, None, Some("p"), env_from(&map)).unwrap();
        assert_eq!(v.model, "profile-model");
        assert_eq!(v.base_url, "http://profile/v1");
        assert_eq!(v.api_key.as_deref(), Some("profile-key"));
    }

    #[test]
    fn env_falls_back_when_no_profile() {
        let mut map = std::collections::HashMap::new();
        map.insert("OPENAI_MODEL", "env-model".to_string());
        map.insert("OPENAI_BASE_URL", "http://env/v1".to_string());
        map.insert("OPENAI_API_KEY", "env-key".to_string());
        let v = resolve_vision(None, None, None, None, env_from(&map)).unwrap();
        assert_eq!(v.model, "env-model");
        assert_eq!(v.base_url, "http://env/v1");
        assert_eq!(v.api_key.as_deref(), Some("env-key"));
    }

    #[test]
    fn defaults_when_nothing_configured() {
        let v = resolve_vision(None, None, None, None, empty_env()).unwrap();
        assert_eq!(v.model, vision::DEFAULT_MODEL);
        assert_eq!(v.base_url, vision::DEFAULT_BASE_URL);
        assert_eq!(v.api_key, None);
    }

    #[test]
    fn default_profile_is_used_without_selection() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
default_profile = "p"

[profiles.p]
base_url = "http://default-profile/v1"
"#,
        )
        .unwrap();
        let mut map = std::collections::HashMap::new();
        map.insert("ARWE_CONFIG", path.display().to_string());
        let v = resolve_vision(None, None, None, None, env_from(&map)).unwrap();
        assert_eq!(v.base_url, "http://default-profile/v1");
    }

    #[test]
    fn unknown_profile_reports_available_names() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
[profiles.alpha]
base_url = "http://a/v1"
[profiles.beta]
base_url = "http://b/v1"
"#,
        )
        .unwrap();
        let mut map = std::collections::HashMap::new();
        map.insert("ARWE_CONFIG", path.display().to_string());
        let err = resolve_vision(None, None, None, Some("nope"), env_from(&map))
            .unwrap_err()
            .to_string();
        assert!(err.contains("'nope'"), "{}", err);
        assert!(err.contains("alpha"), "{}", err);
        assert!(err.contains("beta"), "{}", err);
    }

    #[test]
    fn config_path_prefers_arwe_config() {
        let mut map = std::collections::HashMap::new();
        map.insert("ARWE_CONFIG", "C:\\custom\\cfg.toml".to_string());
        assert_eq!(
            config_path(&env_from(&map)),
            Some(PathBuf::from("C:\\custom\\cfg.toml"))
        );
    }
}
