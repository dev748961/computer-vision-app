"""Tests for reusable OpenCV transformations."""

import unittest

import numpy as np

from vision_app.transforms import (
    apply_color_overlay,
    apply_gaussian_blur,
    convert_to_grayscale,
    crop_image,
    draw_rectangle,
    draw_text,
    resize_image,
    pixelate_region,
)


class TransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((20, 30, 3), dtype=np.uint8)
        self.image[:, :, 1] = 100
        self.original = self.image.copy()

    def test_resize_uses_requested_dimensions_without_changing_source(self) -> None:
        resized = resize_image(self.image, 12, 8)

        self.assertEqual(resized.shape, (8, 12, 3))
        np.testing.assert_array_equal(self.image, self.original)

    def test_grayscale_has_two_dimensions(self) -> None:
        grayscale = convert_to_grayscale(self.image)

        self.assertEqual(grayscale.shape, (20, 30))
        self.assertTrue(np.all(grayscale == 59))
        np.testing.assert_array_equal(self.image, self.original)

    def test_crop_returns_requested_region_and_rejects_invalid_bounds(self) -> None:
        cropped = crop_image(self.image, 3, 4, 8, 6)

        self.assertEqual(cropped.shape, (6, 8, 3))
        self.assertFalse(np.shares_memory(cropped, self.image))
        for coordinates in ((-1, 0, 2, 2), (0, -1, 2, 2), (29, 0, 2, 1), (0, 19, 1, 2), (0, 0, 0, 1)):
            with self.subTest(coordinates=coordinates):
                with self.assertRaises(ValueError):
                    crop_image(self.image, *coordinates)

    def test_blur_preserves_dimensions_and_source(self) -> None:
        image_with_detail = self.image.copy()
        image_with_detail[8:12, 12:18] = 255
        original = image_with_detail.copy()

        blurred = apply_gaussian_blur(image_with_detail, kernel_size=3)

        self.assertEqual(blurred.shape, image_with_detail.shape)
        self.assertFalse(np.array_equal(blurred, image_with_detail))
        np.testing.assert_array_equal(image_with_detail, original)
        with self.assertRaisesRegex(ValueError, "must be odd"):
            apply_gaussian_blur(self.image, kernel_size=2)

    def test_rectangle_and_text_draw_on_copies(self) -> None:
        rectangle = draw_rectangle(self.image, 2, 3, 8, 6, color=(0, 0, 255), thickness=1)
        labeled = draw_text(self.image, "Test", 2, 15, color=(255, 255, 255))

        self.assertTrue(np.any(rectangle != self.image))
        self.assertTrue(np.any(labeled != self.image))
        np.testing.assert_array_equal(self.image, self.original)
        with self.assertRaisesRegex(ValueError, "Text coordinates must be inside"):
            draw_text(self.image, "Test", 30, 15)

    def test_overlay_blends_bgr_color_only_inside_region(self) -> None:
        overlaid = apply_color_overlay(
            self.image, 5, 6, 4, 3, color=(0, 0, 200), alpha=0.5
        )

        self.assertTrue(np.all(overlaid[6:9, 5:9, 2] == 100))
        np.testing.assert_array_equal(overlaid[:6], self.image[:6])
        np.testing.assert_array_equal(self.image, self.original)
        with self.assertRaisesRegex(ValueError, "alpha must be"):
            apply_color_overlay(self.image, 0, 0, 2, 2, alpha=1.1)

    def test_pixelate_region_changes_only_requested_area_and_preserves_source(self) -> None:
        detailed = self.image.copy()
        detailed[5:15, 5:15] = 255
        original = detailed.copy()

        result = pixelate_region(detailed, 5, 5, 10, 10, pixel_size=5)

        self.assertEqual(result.shape, detailed.shape)
        np.testing.assert_array_equal(result[:5], detailed[:5])
        np.testing.assert_array_equal(result[15:], detailed[15:])
        np.testing.assert_array_equal(detailed, original)
        with self.assertRaisesRegex(ValueError, "fit entirely"):
            pixelate_region(detailed, 29, 0, 2, 2)
        with self.assertRaisesRegex(ValueError, "pixel_size"):
            pixelate_region(detailed, 0, 0, 2, 2, pixel_size=0)


if __name__ == "__main__":
    unittest.main()
