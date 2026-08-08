//! Built-in screen/window capture via xcap.
//!
//! Captures are decoded straight into the in-memory [`RgbImage`] pipeline —
//! no temporary PNG files are ever written.

use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use xcap::{Monitor, Window};

use crate::rgb::RgbImage;

/// A capturable window: the same shape shown by `arwe windows` and used for
/// `--window` title matching.
#[derive(Clone, Debug, Serialize)]
pub struct WindowInfo {
    /// Position in the sorted window list (informational only).
    pub index: usize,
    pub title: String,
    pub app_name: String,
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub focused: bool,
}

/// A monitor as seen by xcap, in (x, y)-sorted order.
#[derive(Clone, Debug)]
pub struct MonitorInfo {
    pub index: usize,
    pub name: String,
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub primary: bool,
}

/// Capture the primary monitor, or the Nth monitor in (x, y)-sorted order.
pub fn capture_screen(requested: Option<usize>) -> Result<RgbImage> {
    let monitors = collect_monitors()?;
    let infos: Vec<MonitorInfo> = monitors.iter().map(|(i, _)| i.clone()).collect();
    let idx = select_monitor(&infos, requested)?;
    let (info, monitor) = &monitors[idx];
    let img = monitor
        .capture_image()
        .with_context(|| format!("failed to capture monitor '{}'", info.name))?;
    rgba_to_rgb(img.width(), img.height(), img.as_raw())
        .with_context(|| format!("failed to convert capture of monitor '{}'", info.name))
}

/// Capture the single visible window whose title contains `keyword`
/// (case-insensitive substring match).
pub fn capture_window(keyword: &str) -> Result<RgbImage> {
    let windows = Window::all().context("failed to enumerate windows")?;
    let mut pairs: Vec<(WindowInfo, Window)> = windows
        .iter()
        .filter_map(|w| window_info(w).map(|info| (info, w.clone())))
        .collect();
    pairs.sort_by_key(|(info, _)| window_sort_key(info));
    let infos: Vec<WindowInfo> = pairs.iter().map(|(i, _)| i.clone()).collect();
    let idx = match_window(&infos, keyword)?;
    let (info, window) = &pairs[idx];
    let img = window
        .capture_image()
        .with_context(|| format!("failed to capture window '{}'", info.title))?;
    rgba_to_rgb(img.width(), img.height(), img.as_raw())
        .with_context(|| format!("failed to convert capture of window '{}'", info.title))
}

/// List capturable (non-minimized, non-empty-title) windows, sorted by title.
pub fn list_windows() -> Result<Vec<WindowInfo>> {
    let windows = Window::all().context("failed to enumerate windows")?;
    let infos: Vec<WindowInfo> = windows.iter().filter_map(window_info).collect();
    Ok(sort_and_index(infos))
}

/// Pure selector: `None` picks the primary monitor, `Some(n)` picks the nth
/// monitor in (x, y)-sorted order.
pub fn select_monitor(infos: &[MonitorInfo], requested: Option<usize>) -> Result<usize> {
    match requested {
        None => infos
            .iter()
            .position(|m| m.primary)
            .ok_or_else(|| anyhow!("no primary monitor found")),
        Some(n) => {
            if n < infos.len() {
                Ok(n)
            } else {
                bail!(
                    "monitor index {} is out of range ({} monitor(s): {})",
                    n,
                    infos.len(),
                    describe_monitors(infos)
                )
            }
        }
    }
}

/// Pure matcher: exactly one visible window title must contain `keyword`
/// (case-insensitive). Returns the index into the sorted `infos` slice.
pub fn match_window(infos: &[WindowInfo], keyword: &str) -> Result<usize> {
    let kw = keyword.trim().to_lowercase();
    if kw.is_empty() {
        bail!("--window needs a non-empty title keyword");
    }
    let hits: Vec<usize> = infos
        .iter()
        .enumerate()
        .filter(|(_, i)| i.title.to_lowercase().contains(&kw))
        .map(|(idx, _)| idx)
        .collect();
    match hits.len() {
        0 => bail!(
            "no visible window title contains '{}' (run `arwe windows` to list)",
            keyword
        ),
        1 => Ok(hits[0]),
        _ => {
            let titles: Vec<&str> = hits.iter().map(|&i| infos[i].title.as_str()).collect();
            bail!(
                "'{}' matches {} windows: {} (narrow the keyword or run `arwe windows`)",
                keyword,
                hits.len(),
                titles.join(", ")
            )
        }
    }
}

/// Sort windows by (lowercased title, lowercased app name) and renumber.
pub fn sort_and_index(mut infos: Vec<WindowInfo>) -> Vec<WindowInfo> {
    infos.sort_by_key(window_sort_key);
    for (i, info) in infos.iter_mut().enumerate() {
        info.index = i;
    }
    infos
}

/// Convert an RGBA frame to the project's RGB pipeline. Alpha is dropped;
/// xcap returns straight (non-premultiplied) RGBA.
fn rgba_to_rgb(width: u32, height: u32, raw: &[u8]) -> Result<RgbImage> {
    let w = width as usize;
    let h = height as usize;
    if raw.len() != w * h * 4 {
        bail!(
            "RGBA buffer length {} does not match {}x{}",
            raw.len(),
            w,
            h
        );
    }
    let pixels = raw.chunks_exact(4).map(|c| [c[0], c[1], c[2]]).collect();
    RgbImage::new(w, h, pixels)
}

fn collect_monitors() -> Result<Vec<(MonitorInfo, Monitor)>> {
    let mut monitors = Monitor::all().context("failed to enumerate monitors")?;
    monitors.sort_by_key(|m| (m.x().unwrap_or(0), m.y().unwrap_or(0)));
    let mut out = Vec::with_capacity(monitors.len());
    for (i, m) in monitors.into_iter().enumerate() {
        let info = MonitorInfo {
            index: i,
            name: m.name().unwrap_or_else(|_| format!("monitor-{}", i)),
            x: m.x().unwrap_or(0),
            y: m.y().unwrap_or(0),
            width: m.width().unwrap_or(0),
            height: m.height().unwrap_or(0),
            primary: m.is_primary().unwrap_or(false),
        };
        out.push((info, m));
    }
    Ok(out)
}

/// Build a [`WindowInfo`] from an xcap window, skipping windows that cannot be
/// captured meaningfully (empty titles, minimized windows).
fn window_info(w: &Window) -> Option<WindowInfo> {
    let title = w.title().unwrap_or_default();
    if title.trim().is_empty() || w.is_minimized().unwrap_or(true) {
        return None;
    }
    Some(WindowInfo {
        index: 0,
        title,
        app_name: w.app_name().unwrap_or_default(),
        x: w.x().unwrap_or(0),
        y: w.y().unwrap_or(0),
        width: w.width().unwrap_or(0),
        height: w.height().unwrap_or(0),
        focused: w.is_focused().unwrap_or(false),
    })
}

fn window_sort_key(info: &WindowInfo) -> (String, String) {
    (info.title.to_lowercase(), info.app_name.to_lowercase())
}

fn describe_monitors(infos: &[MonitorInfo]) -> String {
    infos
        .iter()
        .map(|m| {
            format!(
                "{}: '{}' {}x{} @({},{}){}",
                m.index,
                m.name,
                m.width,
                m.height,
                m.x,
                m.y,
                if m.primary { " (primary)" } else { "" }
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn win(index: usize, title: &str) -> WindowInfo {
        WindowInfo {
            index,
            title: title.to_string(),
            app_name: "app".to_string(),
            x: 0,
            y: 0,
            width: 100,
            height: 100,
            focused: false,
        }
    }

    fn monitor(index: usize, primary: bool) -> MonitorInfo {
        MonitorInfo {
            index,
            name: format!("monitor-{}", index),
            x: 0,
            y: 0,
            width: 1920,
            height: 1080,
            primary,
        }
    }

    #[test]
    fn rgba_to_rgb_drops_alpha() {
        let img = rgba_to_rgb(1, 1, &[10, 20, 30, 255]).unwrap();
        assert_eq!(img.width, 1);
        assert_eq!(img.height, 1);
        assert_eq!(img.pixels, vec![[10, 20, 30]]);
    }

    #[test]
    fn rgba_to_rgb_handles_multiple_rows_and_any_alpha() {
        let raw: Vec<u8> = (0..16u8).collect(); // 2x2 RGBA
        let img = rgba_to_rgb(2, 2, &raw).unwrap();
        assert_eq!(
            img.pixels,
            vec![[0, 1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14]]
        );
    }

    #[test]
    fn rgba_to_rgb_rejects_bad_length() {
        assert!(rgba_to_rgb(2, 2, &[0, 1, 2, 3]).is_err());
    }

    #[test]
    fn select_monitor_none_picks_primary() {
        let infos = vec![monitor(0, false), monitor(1, true)];
        assert_eq!(select_monitor(&infos, None).unwrap(), 1);
    }

    #[test]
    fn select_monitor_by_index_and_out_of_range() {
        let infos = vec![monitor(0, false), monitor(1, true)];
        assert_eq!(select_monitor(&infos, Some(0)).unwrap(), 0);
        assert_eq!(select_monitor(&infos, Some(1)).unwrap(), 1);
        let err = select_monitor(&infos, Some(2)).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("out of range"));
        assert!(msg.contains("monitor-1"));
    }

    #[test]
    fn select_monitor_missing_primary() {
        let infos = vec![monitor(0, false), monitor(1, false)];
        let err = select_monitor(&infos, None).unwrap_err();
        assert!(err.to_string().contains("no primary monitor"));
    }

    #[test]
    fn match_window_finds_unique_case_insensitively() {
        let infos = sort_and_index(vec![
            win(0, "Visual Studio Code"),
            win(1, "Google Chrome - GitHub"),
        ]);
        assert_eq!(match_window(&infos, "chrome").unwrap(), 0);
        assert_eq!(match_window(&infos, "CHROME").unwrap(), 0);
    }

    #[test]
    fn match_window_no_hit() {
        let infos = vec![win(0, "Terminal")];
        let err = match_window(&infos, "firefox").unwrap_err();
        assert!(err.to_string().contains("firefox"));
    }

    #[test]
    fn match_window_multiple_hits() {
        let infos = vec![win(0, "Chrome - A"), win(1, "Chrome - B")];
        let err = match_window(&infos, "chrome").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("2 windows"));
        assert!(msg.contains("Chrome - A"));
    }

    #[test]
    fn match_window_empty_keyword_rejected() {
        let infos = vec![win(0, "Terminal")];
        assert!(match_window(&infos, "  ").is_err());
    }

    #[test]
    fn sort_and_index_orders_case_insensitively() {
        let infos = sort_and_index(vec![
            win(0, "zeta"),
            win(1, "Alpha"),
            win(2, "alpha"),
        ]);
        assert_eq!(infos[0].title, "Alpha");
        assert_eq!(infos[1].title, "alpha");
        assert_eq!(infos[2].title, "zeta");
        assert_eq!(infos[0].index, 0);
        assert_eq!(infos[1].index, 1);
        assert_eq!(infos[2].index, 2);
    }
}
