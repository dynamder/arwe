use arwe::resize::{self, CropRect};
use arwe::rgb::RgbImage;

fn img(w: usize, h: usize, pixels: Vec<[u8; 3]>) -> RgbImage {
    RgbImage::new(w, h, pixels).unwrap()
}

fn grad4x4() -> RgbImage {
    img(4, 4, (1..=16u8).map(|v| [v, v, v]).collect())
}

#[test]
fn crop_is_exact() {
    let p = resize::process(&grad4x4(), Some(CropRect { x: 1, y: 1, w: 2, h: 2 }), None, 1).unwrap();
    assert_eq!(p.image.width, 2);
    assert_eq!(p.image.height, 2);
    assert_eq!(p.image.pixels, vec![[6, 6, 6], [7, 7, 7], [10, 10, 10], [11, 11, 11]]);
}

#[test]
fn crop_out_of_bounds_is_error() {
    assert!(resize::process(&grad4x4(), Some(CropRect { x: 3, y: 3, w: 2, h: 2 }), None, 1).is_err());
    assert!(resize::process(&grad4x4(), Some(CropRect { x: 0, y: 0, w: 0, h: 4 }), None, 1).is_err());
}

#[test]
fn resize_uniform_blocks_are_exact() {
    let im = img(
        4,
        4,
        [[255, 0, 0], [255, 0, 0], [0, 0, 255], [0, 0, 255]]
            .iter()
            .cycle()
            .take(16)
            .copied()
            .collect(),
    );
    let p = resize::process(&im, None, Some(2), 1).unwrap();
    assert_eq!(p.image.width, 2);
    assert_eq!(p.image.height, 2);
    assert_eq!(p.image.pixels, vec![[255, 0, 0], [0, 0, 255], [255, 0, 0], [0, 0, 255]]);
}

#[test]
fn resize_gradient_is_area_average() {
    let p = resize::process(&grad4x4(), None, Some(2), 1).unwrap();
    assert_eq!(p.image.width, 2);
    assert_eq!(p.image.height, 2);
    assert_eq!(p.image.pixels, vec![[4, 4, 4], [6, 6, 6], [12, 12, 12], [14, 14, 14]]);
}

#[test]
fn subsample_takes_every_nth() {
    let p = resize::process(&grad4x4(), None, None, 2).unwrap();
    assert_eq!((p.image.width, p.image.height), (2, 2));
    assert_eq!(p.image.pixels, vec![[1, 1, 1], [3, 3, 3], [9, 9, 9], [11, 11, 11]]);
}

#[test]
fn sample_maps_original_coordinates() {
    let p = resize::process(&grad4x4(), None, None, 2).unwrap();
    assert_eq!(p.to_processed(0, 0), Some((0, 0)));
    assert_eq!(p.to_processed(2, 0), Some((1, 0)));
    // Original (0,3) rounds/clamps to processed (0,1).
    assert_eq!(p.to_processed(0, 3), Some((0, 1)));
    assert_eq!(p.to_original(0, 0), (0, 0));
    assert_eq!(p.to_original(1, 1), (2, 2));
}

#[test]
fn sample_outside_crop_is_none() {
    let p = resize::process(&grad4x4(), Some(CropRect { x: 1, y: 1, w: 2, h: 2 }), None, 1).unwrap();
    assert_eq!(p.to_processed(0, 0), None);
    assert_eq!(p.to_processed(1, 1), Some((0, 0)));
    assert_eq!(p.to_original_rect(0, 0, 2, 2), (1, 1, 2, 2));
}

#[test]
fn parse_crop_validation() {
    assert!(resize::parse_crop("10,20,30,40").is_ok());
    assert!(resize::parse_crop("10,20,0,40").is_err());
    assert!(resize::parse_crop("1,2,3").is_err());
    assert!(resize::parse_crop("a,b,c,d").is_err());
}

#[test]
fn parse_point_validation() {
    assert_eq!(resize::parse_point("3,4").unwrap(), (3, 4));
    assert!(resize::parse_point("3").is_err());
    assert!(resize::parse_point("x,y").is_err());
}

#[test]
fn crop_image_guards() {
    let im = grad4x4();
    assert!(resize::crop_image(&im, CropRect { x: 0, y: 0, w: 0, h: 4 }).is_err());
    assert!(resize::crop_image(&im, CropRect { x: 0, y: 0, w: 4, h: 0 }).is_err());
    assert!(resize::crop_image(&im, CropRect { x: 1, y: 0, w: 4, h: 1 }).is_err());
    assert!(resize::crop_image(&im, CropRect { x: 0, y: 1, w: 1, h: 4 }).is_err());
    assert!(resize::crop_image(&im, CropRect { x: 1, y: 1, w: 2, h: 2 }).is_ok());
}

#[test]
fn mapping_with_crop_and_scale() {
    // 10x10 image, crop (2,3,6,6), resize to width 3 => scale 2 in both axes.
    let im = img(10, 10, (1..=100u8).map(|v| [v, v, v]).collect());
    let p = resize::process(
        &im,
        Some(CropRect { x: 2, y: 3, w: 6, h: 6 }),
        Some(3),
        1,
    )
    .unwrap();
    assert_eq!((p.image.width, p.image.height), (3, 3));
    assert_eq!(p.to_processed(3, 3), Some((1, 0)));
    assert_eq!(p.to_processed(2, 3), Some((0, 0)));
    assert_eq!(p.to_processed(7, 8), Some((2, 2)));
    assert_eq!(p.to_processed(3, 4), Some((1, 1)));
    assert_eq!(p.to_processed(8, 8), None);
    assert_eq!(p.to_processed(1, 3), None);
    assert_eq!(p.to_original(3, 0), (8, 3));
    assert_eq!(p.to_original_rect(3, 0, 1, 1), (8, 3, 2, 2));
    // Clamping at the far edge.
    assert_eq!(p.to_original(3, 3), (8, 9));
    assert_eq!(p.to_original(4, 0), (9, 3));
    assert_eq!(p.to_original(0, 4), (2, 9));
    assert_eq!(p.to_original_rect(4, 0, 2, 1), (9, 3, 1, 2));
    assert_eq!(p.to_original_rect(0, 4, 2, 2), (2, 9, 4, 1));
}

#[test]
fn subsample_odd_dimensions() {
    let im = img(5, 5, (1..=25u8).map(|v| [v, v, v]).collect());
    let p = resize::process(&im, None, None, 2).unwrap();
    assert_eq!((p.image.width, p.image.height), (3, 3));
    assert_eq!(
        p.image.pixels,
        vec![[1, 1, 1], [3, 3, 3], [5, 5, 5], [11, 11, 11], [13, 13, 13], [15, 15, 15], [21, 21, 21], [23, 23, 23], [25, 25, 25]]
    );
}
