"""Tests for batch processing and input-image protection."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vision_app.detection import Detection, DetectionError
from vision_app.pipeline import process_detections, process_images, transform_detection_regions, process_region_transforms


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_processes_image_to_separate_file_without_changing_source(self) -> None:
        input_directory = self.root / "images"
        input_directory.mkdir()
        source_path = input_directory / "sample.png"
        source_image = np.zeros((32, 40, 3), dtype=np.uint8)
        source_image[8:20, 10:30] = (20, 80, 140)
        self.assertTrue(cv2.imwrite(str(source_path), source_image))
        original_bytes = source_path.read_bytes()

        processed, failures = process_images([source_path], input_directory)

        self.assertEqual(failures, [])
        self.assertEqual(processed, [input_directory / "sample_processed.png"])
        self.assertTrue(processed[0].is_file())
        self.assertEqual(source_path.read_bytes(), original_bytes)
        self.assertIsNotNone(cv2.imread(str(processed[0])))

    def test_unreadable_image_failure_does_not_stop_batch(self) -> None:
        input_directory = self.root / "input"
        input_directory.mkdir()
        unreadable_path = input_directory / "bad.png"
        unreadable_path.write_text("not an image", encoding="utf-8")
        readable_path = input_directory / "good.png"
        self.assertTrue(cv2.imwrite(str(readable_path), np.zeros((8, 8, 3), dtype=np.uint8)))

        processed, failures = process_images(
            [unreadable_path, readable_path], self.root / "output"
        )

        self.assertEqual(processed, [self.root / "output" / "good_processed.png"])
        self.assertEqual([path for path, _ in failures], [unreadable_path])
        self.assertIn("Could not read input image", failures[0][1])

    def test_existing_processed_file_is_not_overwritten(self) -> None:
        input_directory = self.root / "input"
        output_directory = self.root / "output"
        input_directory.mkdir()
        source_path = input_directory / "sample.png"
        self.assertTrue(cv2.imwrite(str(source_path), np.zeros((8, 8, 3), dtype=np.uint8)))
        output_directory.mkdir()
        previous_output = output_directory / "sample_processed.png"
        previous_output.write_bytes(b"keep existing output")

        processed, failures = process_images([source_path], output_directory)

        self.assertEqual(processed, [])
        self.assertEqual([path for path, _ in failures], [source_path])
        self.assertEqual(previous_output.read_bytes(), b"keep existing output")
        self.assertTrue(source_path.is_file())

    def test_detection_saves_annotated_copy_and_preserves_source(self) -> None:
        from unittest.mock import Mock, patch

        input_directory = self.root / "input"
        output_directory = self.root / "output"
        input_directory.mkdir()
        source_path = input_directory / "sample.png"
        source_image = np.zeros((40, 50, 3), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(source_path), source_image))
        original_bytes = source_path.read_bytes()

        with (
            patch("vision_app.pipeline.load_yolo_model", return_value=Mock()),
            patch(
                "vision_app.pipeline.detect_objects",
                return_value=[Detection(0, "person", 0.9, (5, 5, 30, 30))],
            ),
        ):
            detected, failures = process_detections([source_path], output_directory)

        self.assertEqual(failures, [])
        self.assertEqual(detected, [output_directory / "sample_detected.png"])
        annotated = cv2.imread(str(detected[0]))
        self.assertIsNotNone(annotated)
        self.assertTrue(np.any(annotated != source_image))
        self.assertEqual(source_path.read_bytes(), original_bytes)

    def test_detection_failure_on_one_image_does_not_stop_the_batch(self) -> None:
        from unittest.mock import Mock, patch

        input_directory = self.root / "input"
        input_directory.mkdir()
        first_image = input_directory / "first.png"
        second_image = input_directory / "second.png"
        for path in (first_image, second_image):
            self.assertTrue(cv2.imwrite(str(path), np.zeros((12, 12, 3), dtype=np.uint8)))

        with (
            patch("vision_app.pipeline.load_yolo_model", return_value=Mock()),
            patch(
                "vision_app.pipeline.detect_objects",
                side_effect=[DetectionError("bad prediction"), []],
            ),
        ):
            detected, failures = process_detections(
                [first_image, second_image], self.root / "output"
            )

        self.assertEqual(detected, [self.root / "output" / "second_detected.png"])
        self.assertEqual([path for path, _ in failures], [first_image])
        self.assertIn("bad prediction", failures[0][1])

    def test_existing_detection_output_is_not_overwritten(self) -> None:
        from unittest.mock import Mock, patch

        input_directory = self.root / "input"
        output_directory = self.root / "output"
        input_directory.mkdir()
        output_directory.mkdir()
        source_path = input_directory / "sample.jpg"
        self.assertTrue(cv2.imwrite(str(source_path), np.zeros((8, 8, 3), dtype=np.uint8)))
        existing_output = output_directory / "sample_detected.jpg"
        existing_output.write_bytes(b"keep this output")

        with patch("vision_app.pipeline.load_yolo_model", return_value=Mock()):
            detected, failures = process_detections([source_path], output_directory)

        self.assertEqual(detected, [])
        self.assertEqual([path for path, _ in failures], [source_path])
        self.assertEqual(existing_output.read_bytes(), b"keep this output")

    def test_model_loading_error_is_reported_for_each_detection_input(self) -> None:
        from unittest.mock import patch

        first_image = self.root / "first.png"
        second_image = self.root / "second.png"
        with patch(
            "vision_app.pipeline.load_yolo_model",
            side_effect=DetectionError("weights unavailable"),
        ):
            detected, failures = process_detections(
                [first_image, second_image], self.root / "output"
            )

        self.assertEqual(detected, [])
        self.assertEqual([path for path, _ in failures], [first_image, second_image])
        self.assertTrue(all("weights unavailable" in error for _, error in failures))

    def test_region_transforms_filter_case_insensitively_and_process_multiple_boxes(self) -> None:
        image = np.full((40, 50, 3), 100, dtype=np.uint8)
        original = image.copy()
        detections = [
            Detection(2, "car", 0.9, (25, 10, 40, 25)),
            Detection(0, "Person", 0.8, (2, 2, 17, 17)),
            Detection(1, "bicycle", 0.7, (0, 0, 8, 8)),
        ]
        result, matched = transform_detection_regions(
            image, detections, "overlay", ["PERSON", "Car"], overlay_alpha=1.0
        )
        self.assertEqual([item.class_name for item in matched], ["Person", "car"])
        self.assertTrue(np.all(result[2:17, 2:17] == 0))
        self.assertTrue(np.all(result[10:25, 25:40] == 0))
        np.testing.assert_array_equal(image, original)

    def test_region_transform_clips_boxes_skips_invalid_and_orders_overlaps(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[2:18, 2:18] = 100
        detections = [
            Detection(0, "person", 0.9, (5, 5, 15, 15)),
            Detection(0, "person", 0.8, (-5, -5, 8, 8)),
            Detection(0, "person", 0.7, (30, 30, 40, 40)),
            Detection(0, "person", 0.6, (3, 3, 3, 8)),
        ]
        result, matched = transform_detection_regions(image, detections, "pixelate", ["person"], pixel_size=4)
        reversed_result, _ = transform_detection_regions(
            image, list(reversed(detections)), "pixelate", ["person"], pixel_size=4
        )
        self.assertEqual(len(matched), 2)
        np.testing.assert_array_equal(result, reversed_result)
        self.assertTrue(np.array_equal(image, np.pad(np.full((16, 16, 3), 100, dtype=np.uint8), ((2,2),(2,2),(0,0)))))
        self.assertEqual(result.shape, image.shape)

    def test_blur_region_does_not_change_pixels_outside_box(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[8:24, 8:24] = 255
        original = image.copy()
        result, matched = transform_detection_regions(
            image, [Detection(0, "person", 0.9, (8, 8, 24, 24))], "blur", ["person"], blur_kernel=5
        )
        self.assertEqual(len(matched), 1)
        np.testing.assert_array_equal(result[:8], original[:8])
        np.testing.assert_array_equal(image, original)

    def test_region_batch_reports_unmatched_and_protects_outputs_and_continues(self) -> None:
        from unittest.mock import Mock, patch

        input_directory = self.root / "input"
        output_directory = self.root / "output"
        input_directory.mkdir()
        paths = [input_directory / "first.png", input_directory / "second.png"]
        for path in paths:
            self.assertTrue(cv2.imwrite(str(path), np.full((24, 24, 3), 80, dtype=np.uint8)))
        original_bytes = paths[0].read_bytes()
        existing = output_directory / "first_overlay.png"
        output_directory.mkdir()
        existing.write_bytes(b"keep")
        with (
            patch("vision_app.pipeline.load_yolo_model", return_value=Mock()),
            patch("vision_app.pipeline.detect_objects", side_effect=[
                [Detection(0, "person", 0.9, (1, 1, 20, 20))],
                [Detection(0, "dog", 0.9, (1, 1, 20, 20))],
            ]),
        ):
            detected, saved, failures, unmatched = process_region_transforms(
                paths, output_directory, ["person"], "overlay"
            )
        self.assertEqual(
            detected,
            [output_directory / "first_detected.png", output_directory / "second_detected.png"],
        )
        self.assertEqual(saved, [])
        self.assertEqual([path for path, _ in failures], [paths[0]])
        self.assertIn("Refusing to overwrite", failures[0][1])
        self.assertEqual(unmatched, [paths[1]])
        self.assertEqual(existing.read_bytes(), b"keep")
        self.assertEqual(paths[0].read_bytes(), original_bytes)

    def test_empty_detections_save_annotated_copy_and_return_unmatched(self) -> None:
        from unittest.mock import Mock, patch

        input_directory = self.root / "input"
        output_directory = self.root / "output"
        input_directory.mkdir()
        source = input_directory / "empty.png"
        image = np.full((20, 24, 3), 80, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(source), image))

        with (
            patch("vision_app.pipeline.load_yolo_model", return_value=Mock()),
            patch("vision_app.pipeline.detect_objects", return_value=[]) as infer,
        ):
            detected, transformed, failures, unmatched = process_region_transforms(
                [source], output_directory, ["person"], "blur"
            )

        self.assertEqual(detected, [output_directory / "empty_detected.png"])
        self.assertEqual(transformed, [])
        self.assertEqual(failures, [])
        self.assertEqual(unmatched, [source])
        infer.assert_called_once()
        np.testing.assert_array_equal(cv2.imread(str(detected[0])), image)

    def test_region_inference_failure_does_not_stop_later_images(self) -> None:
        from unittest.mock import Mock, patch

        input_directory = self.root / "input"
        output_directory = self.root / "output"
        input_directory.mkdir()
        paths = [input_directory / "failed.png", input_directory / "good.png"]
        for path in paths:
            self.assertTrue(cv2.imwrite(str(path), np.full((24, 24, 3), 80, dtype=np.uint8)))
        person = Detection(0, "person", 0.9, (2, 2, 20, 20))

        with (
            patch("vision_app.pipeline.load_yolo_model", return_value=Mock()),
            patch(
                "vision_app.pipeline.detect_objects",
                side_effect=[DetectionError("bad prediction"), [person]],
            ) as infer,
        ):
            detected, transformed, failures, unmatched = process_region_transforms(
                paths, output_directory, ["person"], "pixelate"
            )

        self.assertEqual([path for path, _ in failures], [paths[0]])
        self.assertIn("bad prediction", failures[0][1])
        self.assertEqual(detected, [output_directory / "good_detected.png"])
        self.assertEqual(transformed, [output_directory / "good_pixelate.png"])
        self.assertEqual(unmatched, [])
        self.assertEqual(infer.call_count, 2)

    def test_region_transform_failure_does_not_stop_later_images(self) -> None:
        from unittest.mock import Mock, patch

        input_directory = self.root / "input"
        output_directory = self.root / "output"
        input_directory.mkdir()
        paths = [input_directory / "failed.png", input_directory / "good.png"]
        for path in paths:
            self.assertTrue(cv2.imwrite(str(path), np.full((24, 24, 3), 80, dtype=np.uint8)))
        person = Detection(0, "person", 0.9, (2, 2, 20, 20))
        transformed_image = np.full((24, 24, 3), 120, dtype=np.uint8)

        with (
            patch("vision_app.pipeline.load_yolo_model", return_value=Mock()),
            patch("vision_app.pipeline.detect_objects", return_value=[person]),
            patch(
                "vision_app.pipeline.transform_detection_regions",
                side_effect=[ValueError("simulated transform failure"), (transformed_image, [person])],
            ) as transform,
        ):
            detected, transformed, failures, unmatched = process_region_transforms(
                paths, output_directory, ["person"], "overlay"
            )

        self.assertEqual([path for path, _ in failures], [paths[0]])
        self.assertIn("simulated transform failure", failures[0][1])
        self.assertEqual(
            detected,
            [output_directory / "failed_detected.png", output_directory / "good_detected.png"],
        )
        self.assertEqual(transformed, [output_directory / "good_overlay.png"])
        self.assertEqual(unmatched, [])
        self.assertEqual(transform.call_count, 2)

    def test_region_run_infers_once_per_image_and_reuses_detections(self) -> None:
        from unittest.mock import Mock, patch

        input_directory = self.root / "input"
        output_directory = self.root / "output"
        input_directory.mkdir()
        paths = [input_directory / "one.png", input_directory / "two.png"]
        for path in paths:
            self.assertTrue(cv2.imwrite(str(path), np.full((24, 24, 3), 80, dtype=np.uint8)))
        detection = Detection(0, "person", 0.9, (2, 2, 20, 20))
        model = Mock()
        with (
            patch("vision_app.pipeline.load_yolo_model", return_value=model) as load_model,
            patch("vision_app.pipeline.detect_objects", return_value=[detection]) as infer,
        ):
            detected, transformed, failures, unmatched = process_region_transforms(
                paths, output_directory, ["person"], "pixelate"
            )

        self.assertEqual(failures, [])
        self.assertEqual(unmatched, [])
        self.assertEqual(len(detected), 2)
        self.assertEqual(len(transformed), 2)
        load_model.assert_called_once_with()
        self.assertEqual(infer.call_count, len(paths))
        for call_args, path in zip(infer.call_args_list, paths):
            np.testing.assert_array_equal(call_args.args[0], cv2.imread(str(path)))


if __name__ == "__main__":
    unittest.main()
