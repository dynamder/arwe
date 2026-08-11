use std::path::PathBuf;

/// Write a mock `tesseract` executable that echoes `tsv` back on stdout.
///
/// Windows gets a `.bat`; Unix gets an executable shell script, so the same
/// test can run in CI on every platform.
pub fn write_mock_tesseract(dir: &tempfile::TempDir, tsv: &str) -> PathBuf {
    #[cfg(windows)]
    let (name, content) = {
        let lines: Vec<String> = tsv.lines().map(|l| format!("echo {}", l)).collect();
        let content = format!("@echo off\r\n{}\r\n", lines.join("\r\n"));
        ("tess.bat", content)
    };
    #[cfg(not(windows))]
    let (name, content) = {
        let lines: Vec<String> = tsv.lines().map(|l| format!("echo '{}'", l)).collect();
        let content = format!("#!/bin/sh\n{}\n", lines.join("\n"));
        ("tess", content)
    };
    let path = dir.path().join(name);
    std::fs::write(&path, content).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();
    }
    path
}
