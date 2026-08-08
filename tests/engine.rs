use std::io::Cursor;

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;

use arwe::conv;
use arwe::image_io::{self, InputSource};
use arwe::kernels;
use arwe::output;
use arwe::rgb::{ChannelData, ImageData, RgbImage};

fn img(w: usize, h: usize, pixels: Vec<[u8; 3]>) -> RgbImage {
    RgbImage::new(w, h, pixels).unwrap()
}

fn kernel(name: &str) -> kernels::Kernel {
    kernels::preset(name).unwrap()
}

fn mono(data: &ImageData) -> &[u8] {
    match &data.data {
        ChannelData::Mono(v) => v,
        ChannelData::Rgb(_) => panic!("expected mono output"),
    }
}

fn rgb(data: &ImageData) -> &[[u8; 3]] {
    match &data.data {
        ChannelData::Rgb(v) => v,
        ChannelData::Mono(_) => panic!("expected rgb output"),
    }
}

#[test]
fn identity_keeps_pixels() {
    let im = img(2, 2, vec![[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]]);
    let out = conv::convolve_rgb(&im, &kernel("identity"), 1).unwrap();
    assert_eq!(out.width, 2);
    assert_eq!(out.height, 2);
    assert_eq!(rgb(&out), &im.pixels);
}

#[test]
fn sobel_x_detects_vertical_edge() {
    // Left half 0, right half 255: a vertical edge.
    let row = [[0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255]];
    let im = img(4, 4, row.iter().cycle().take(16).copied().collect());
    let out = conv::convolve_luma(&im, &kernel("sobel-x"), 1).unwrap();
    // Pixel straddling the edge saturates high (raw sum 1020 -> 255).
    assert_eq!(mono(&out)[4 + 1], 255);
    // Uniform white area at the far side yields 0 (raw sum -765 -> clamp 0).
    assert_eq!(mono(&out)[12 + 3], 0);
}

#[test]
fn zero_padding_allows_1x1_image() {
    let im = img(1, 1, vec![[100, 100, 100]]);
    let out = conv::convolve_luma(&im, &kernel("box3"), 1).unwrap();
    assert_eq!(out.width, 1);
    assert_eq!(out.height, 1);
    assert_eq!(mono(&out)[0], 100);
}

#[test]
fn output_clamps_to_255() {
    let im = img(2, 2, vec![[255, 255, 255]; 4]);
    let out = conv::convolve_luma(&im, &kernel("box3"), 1).unwrap();
    assert_eq!(mono(&out), &[255, 255, 255, 255]);
}

#[test]
fn stride_halves_output_size() {
    let im = img(4, 4, vec![[1, 1, 1]; 16]);
    let out = conv::convolve_rgb(&im, &kernel("identity"), 2).unwrap();
    assert_eq!((out.width, out.height), (2, 2));
    assert_eq!(rgb(&out), &[[1, 1, 1]; 4]);
}

#[test]
fn rounding_is_nearest() {
    // 0.1 * 255 = 25.5 -> rounds to 26.
    let im = img(1, 1, vec![[255, 255, 255]]);
    let k = kernels::from_json("[[0.1]]").unwrap();
    let out = conv::convolve_luma(&im, &k, 1).unwrap();
    assert_eq!(mono(&out)[0], 26);
}

#[test]
fn cross_channel_luma_kernel_matches_rec601() {
    let im = img(1, 1, vec![[255, 0, 0]]);
    let out = conv::convolve_cross(&im, &kernel("luma"), 1).unwrap();
    assert_eq!(mono(&out)[0], 76); // 0.299 * 255 = 76.245 -> 76
}

#[test]
fn cross_channel_mixes_channels() {
    let k = kernels::from_json("[[[1,1]],[[0,0]],[[0,0]]]").unwrap();
    assert_eq!(k.channels, 3);
    assert_eq!((k.height, k.width), (1, 2));
    // Only the R plane participates; a 1x2 kernel sums two R pixels.
    let im = img(2, 1, vec![[10, 200, 0], [20, 200, 0]]);
    let out = conv::convolve_cross(&im, &k, 1).unwrap();
    assert_eq!(out.width, 2);
    assert_eq!(mono(&out), &[30, 20]);
}

#[test]
fn json_2d_kernel_parses() {
    let k = kernels::from_json("[[-1,0,1],[-2,0,2],[-1,0,1]]").unwrap();
    assert_eq!(k.channels, 1);
    assert_eq!((k.height, k.width), (3, 3));
    assert_eq!(k.values.len(), 9);
}

#[test]
fn json_object_kernel_parses() {
    let k = kernels::from_json(r#"{"kernel": [[1,2],[3,4]]}"#).unwrap();
    assert_eq!((k.height, k.width), (2, 2));
    assert_eq!(k.values, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn json_bad_shapes_rejected() {
    assert!(kernels::from_json("[[1,2],[3]]").is_err());
    assert!(kernels::from_json("[1,2,3]").is_err());
    assert!(kernels::from_json("[[[1]],[[2]]]").is_err()); // 2 channels, not 3
    assert!(kernels::from_json("[[\"a\"]]").is_err());
}

#[test]
fn plain_text_kernel_file_parses() {
    let dir = std::env::temp_dir();
    let path = dir.join("arwe_test_kernel.txt");
    std::fs::write(&path, "# sobel x\n-1 0 1\n-2,0,2\n-1 | 0 | 1\n").unwrap();
    let k = kernels::from_file(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(k.name, "custom");
    assert_eq!((k.height, k.width), (3, 3));
}

#[test]
fn matrix_output_is_compact() {
    let im = img(2, 2, vec![[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]);
    let s = output::matrix(&im.rgb_data());
    assert_eq!(s, "[[[255,0,0],[0,255,0]],[[0,0,255],[255,255,255]]]");
    assert!(!s.contains(' '));
}

#[test]
fn matrix_mono_output_is_compact() {
    let data = ImageData {
        width: 2,
        height: 2,
        data: ChannelData::Mono(vec![1, 2, 3, 4]),
    };
    assert_eq!(output::matrix(&data), "[[1,2],[3,4]]");
}

#[test]
fn ascii_mapping_uses_ramp() {
    let data = ImageData {
        width: 3,
        height: 1,
        data: ChannelData::Mono(vec![0, 128, 255]),
    };
    assert_eq!(output::ascii(&data, " .:-=+*#%@"), " =@\n");
}

#[test]
fn ascii_rgb_uses_luma_and_row_breaks() {
    let data = ImageData {
        width: 5,
        height: 1,
        data: ChannelData::Rgb(vec![
            [0, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ]),
    };
    assert_eq!(output::ascii(&data, " .:-=+*#%@"), " @:+.\n");
}

#[test]
fn ascii_ansi_rgb_exact() {
    let data = ImageData {
        width: 2,
        height: 2,
        data: ChannelData::Rgb(vec![
            [64, 0, 0],
            [255, 255, 255],
            [64, 0, 0],
            [255, 255, 255],
        ]),
    };
    assert_eq!(
        output::ascii_ansi(&data, " .:-=+*#%@"),
        "\u{1b}[48;2;64;0;0;37m \u{1b}[0m\u{1b}[48;2;255;255;255;30m@\u{1b}[0m\n\
         \u{1b}[48;2;64;0;0;37m \u{1b}[0m\u{1b}[48;2;255;255;255;30m@\u{1b}[0m\n"
    );
}

#[test]
fn ascii_ansi_mono_uses_gray_backgrounds() {
    let data = ImageData {
        width: 2,
        height: 1,
        data: ChannelData::Mono(vec![64, 200]),
    };
    let s = output::ascii_ansi(&data, " .:-=+*#%@");
    assert!(s.contains("\u{1b}[48;2;64;64;64;37m"));
    assert!(s.contains("\u{1b}[48;2;200;200;200;30m"));
}

#[test]
fn pixel_bounds() {
    let im = img(2, 2, vec![[1, 1, 1]; 4]);
    assert_eq!(im.pixel(0, 0), Some([1, 1, 1]));
    assert_eq!(im.pixel(1, 1), Some([1, 1, 1]));
    assert_eq!(im.pixel(2, 0), None);
    assert_eq!(im.pixel(0, 2), None);
}

#[test]
fn base64_png_round_trip() {
    let buf = image::RgbImage::from_raw(
        2,
        2,
        vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255],
    )
    .unwrap();
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(buf)
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .unwrap();
    let b64 = STANDARD.encode(&bytes);
    let loaded = image_io::load(&InputSource::Base64(b64)).unwrap();
    assert_eq!(loaded.width, 2);
    assert_eq!(loaded.height, 2);
    assert_eq!(
        loaded.pixels,
        vec![[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]
    );
}

#[test]
fn data_uri_base64_round_trip() {
    let buf = image::RgbImage::from_raw(1, 1, vec![7, 8, 9]).unwrap();
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(buf)
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .unwrap();
    let uri = format!("data:image/png;base64,{}", STANDARD.encode(&bytes));
    let loaded = image_io::load(&InputSource::Base64(uri)).unwrap();
    assert_eq!(loaded.pixels, vec![[7, 8, 9]]);
}

#[test]
fn even_kernel_top_padding() {
    // 2x1 vertical kernel: top padding is (2-1)/2 = 0, so the top row counts.
    let k = kernels::from_json("[[1],[1]]").unwrap();
    let im = img(1, 2, vec![[10, 10, 10], [20, 20, 20]]);
    let out = conv::convolve_luma(&im, &k, 1).unwrap();
    assert_eq!(mono(&out), &[30, 20]);
}

#[test]
fn even_kernel_left_padding() {
    // 1x2 horizontal kernel: left padding is 0.
    let k = kernels::from_json("[[1,1]]").unwrap();
    let im = img(2, 1, vec![[10, 10, 10], [20, 20, 20]]);
    let out = conv::convolve_luma(&im, &k, 1).unwrap();
    assert_eq!(mono(&out), &[30, 20]);
}

#[test]
fn cross_channel_even_kernel() {
    // 2x2 cross-channel kernel, only the R plane participates.
    let k = kernels::from_json(
        "[[[1,1],[1,1]],[[0,0],[0,0]],[[0,0],[0,0]]]",
    )
    .unwrap();
    let im = img(
        2,
        2,
        vec![[10, 0, 0], [20, 0, 0], [30, 0, 0], [40, 0, 0]],
    );
    let out = conv::convolve_cross(&im, &k, 1).unwrap();
    assert_eq!(mono(&out), &[100, 60, 70, 40]);
}

#[test]
fn cross_channel_multi_channel_kernel() {
    // Values in all three channels so channel-plane indexing matters.
    let k = kernels::from_json(
        "[[[1,0],[0,0]],[[0,0],[0,1]],[[0,0],[0,0]]]",
    )
    .unwrap();
    let im = img(
        2,
        2,
        vec![[10, 1, 0], [20, 2, 0], [30, 3, 0], [40, 4, 0]],
    );
    let out = conv::convolve_cross(&im, &k, 1).unwrap();
    assert_eq!(mono(&out), &[14, 20, 30, 40]);
}

#[test]
fn cross_channel_asymmetric_with_padding() {
    // R plane has a vertical edge; kernel is sobel-x on R only. Expected values
    // computed by hand, including zero-padded border pixels.
    let k = kernels::from_json(
        "[[[-1,0,1],[-2,0,2],[-1,0,1]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]",
    )
    .unwrap();
    let im = img(
        3,
        3,
        vec![
            [0, 0, 0],
            [0, 0, 0],
            [10, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [20, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [30, 0, 0],
        ],
    );
    let out = conv::convolve_cross(&im, &k, 1).unwrap();
    assert_eq!(mono(&out), &[0, 40, 0, 0, 80, 0, 0, 80, 0]);
}

#[test]
fn cross_channel_stride_2() {
    let im = img(4, 4, (1..=16u8).map(|v| [v, v, v]).collect());
    let out = conv::convolve_cross(&im, &kernel("luma"), 2).unwrap();
    assert_eq!(mono(&out), &[1, 3, 9, 11]);
}

#[test]
fn channel_mode_errors() {
    let im = img(1, 1, vec![[1, 2, 3]]);
    // Cross-channel kernels are not valid for per-channel or grayscale modes.
    assert!(conv::convolve_rgb(&im, &kernel("luma"), 1).is_err());
    assert!(conv::convolve_luma(&im, &kernel("luma"), 1).is_err());
    // 2D kernels are not valid for cross-channel mode.
    assert!(conv::convolve_cross(&im, &kernel("sobel-x"), 1).is_err());
}

#[test]
fn preset_values_are_exact() {
    let cases: Vec<(&str, Vec<f32>)> = vec![
        (
            "identity",
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        ("box3", vec![1.0; 9]),
        (
            "gaussian3",
            vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0],
        ),
        (
            "sharpen",
            vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
        ),
        (
            "sobel-x",
            vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0],
        ),
        (
            "sobel-y",
            vec![-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0],
        ),
        (
            "laplacian",
            vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0],
        ),
        (
            "edge",
            vec![-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0],
        ),
        (
            "emboss",
            vec![-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0],
        ),
        (
            "sobel-x-rgb",
            vec![
                -0.299, 0.0, 0.299, -0.598, 0.0, 0.598, -0.299, 0.0, 0.299, -0.587, 0.0,
                0.587, -1.174, 0.0, 1.174, -0.587, 0.0, 0.587, -0.114, 0.0, 0.114, -0.228,
                0.0, 0.228, -0.114, 0.0, 0.114,
            ],
        ),
    ];
    for (name, vals) in cases {
        let k = kernels::preset(name).unwrap();
        assert_eq!(k.values, vals, "preset {}", name);
    }
    let luma = kernels::preset("luma").unwrap();
    assert_eq!(luma.channels, 3);
    assert!((luma.values[0] - 0.299).abs() < 1e-6);
    assert!((luma.values[1] - 0.587).abs() < 1e-6);
    assert!((luma.values[2] - 0.114).abs() < 1e-6);
}

#[test]
fn presets_list_and_json() {
    let ps = kernels::presets();
    assert!(!ps.is_empty());
    for name in ["identity", "sobel-x", "sobel-y", "luma", "sobel-x-rgb"] {
        assert!(ps.iter().any(|p| p.name == name), "missing preset {}", name);
    }
    let v: serde_json::Value = serde_json::from_str(&kernels::presets_json()).unwrap();
    assert_eq!(
        v["sobel-x"]["kernel"],
        serde_json::json!([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    );
    assert_eq!(v["sobel-x"]["channels"], 1);
    assert_eq!(v["luma"]["channels"], 3);
    let luma_r = v["luma"]["kernel"][0][0][0].as_f64().unwrap();
    assert!((luma_r - 0.299).abs() < 1e-6);
    // 3D kernel serialization must preserve channel-plane offsets: each plane
    // has distinct values, so a wrong offset changes the reported matrix.
    let kernel = &v["sobel-x-rgb"]["kernel"];
    assert_eq!(v["sobel-x-rgb"]["channels"], 3);
    assert!((kernel[0][1][0].as_f64().unwrap() + 0.598).abs() < 1e-6);
    assert!((kernel[1][1][0].as_f64().unwrap() + 1.174).abs() < 1e-6);
    assert!((kernel[2][1][0].as_f64().unwrap() + 0.228).abs() < 1e-6);
    assert!(kernel[0] != kernel[1]);
    assert!(kernel[1] != kernel[2]);
}

#[test]
fn kernel_size_limits_enforced() {
    let row = |n: usize| format!("[[{}]]", vec!["1"; n].join(","));
    assert!(kernels::from_json(&row(256)).is_ok());
    assert!(kernels::from_json(&row(257)).is_err());
    let tall = |n: usize| format!("[{}]", vec!["[1]"; n].join(","));
    assert!(kernels::from_json(&tall(256)).is_ok());
    assert!(kernels::from_json(&tall(257)).is_err());
}

#[test]
fn parse_3d_rejects_inconsistent_shapes() {
    // Width mismatch between channels.
    assert!(kernels::from_json("[[[1,2]],[[1]],[[1,2]]]").is_err());
    // Height mismatch between channels.
    assert!(kernels::from_json("[[[1],[2]],[[1]],[[1]]]").is_err());
}

#[test]
fn plain_text_unequal_rows_rejected() {
    let dir = std::env::temp_dir();
    let path = dir.join("arwe_test_bad_kernel.txt");
    std::fs::write(&path, "1 2 3\n4 5\n").unwrap();
    let r = kernels::from_file(&path);
    std::fs::remove_file(&path).unwrap();
    assert!(r.is_err());
}
