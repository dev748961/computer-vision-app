"""Command-line validation tests."""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from main import main


class CliDirectoryValidationTests(unittest.TestCase):
    def _assert_invalid_directory_argument(
        self, arguments: list[str], option_name: str
    ) -> None:
        error_output = io.StringIO()
        with redirect_stderr(error_output), redirect_stdout(io.StringIO()):
            exit_code = main(arguments)

        self.assertEqual(exit_code, 1)
        self.assertIn(
            f"{option_name} must not be empty or whitespace-only.",
            error_output.getvalue(),
        )

    def test_rejects_blank_and_whitespace_input_directories(self) -> None:
        for value in ("", " \t "):
            with self.subTest(value=repr(value)):
                with (
                    patch("main.find_image_files") as find_image_files,
                    patch("main.process_images") as process_images,
                ):
                    self._assert_invalid_directory_argument(
                        ["--input-dir", value, "--output-dir", "output"],
                        "--input-dir",
                    )

                find_image_files.assert_not_called()
                process_images.assert_not_called()

    def test_detection_option_runs_and_reports_annotated_outputs(self) -> None:
        output = io.StringIO()
        image = Path("data/input/sample.png")
        detected = Path("output/sample_detected.png")
        with (
            patch("main.find_image_files", return_value=[image]),
            patch("main.process_images", return_value=([], [])) as process_images,
            patch("main.process_detections", return_value=([detected], [])) as process_detections,
            redirect_stdout(output),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = main(
                ["--input-dir", "data/input", "--output-dir", "output", "--detect", "--confidence", "0.6"]
            )

        self.assertEqual(exit_code, 0)
        process_images.assert_called_once()
        process_detections.assert_called_once_with(
            [image], "output", confidence_threshold=0.6
        )
        self.assertIn(f"Detected: {detected}", output.getvalue())

    def test_day_two_cli_default_does_not_run_detection(self) -> None:
        with (
            patch("main.find_image_files", return_value=[]),
            patch("main.process_images", return_value=([], [])) as process_images,
            patch("main.process_detections") as process_detections,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = main(["--input-dir", "input", "--output-dir", "output"])

        self.assertEqual(exit_code, 0)
        process_images.assert_called_once_with([], "output")
        process_detections.assert_not_called()

    def test_detection_failures_are_reported_with_nonzero_exit(self) -> None:
        error_output = io.StringIO()
        image = Path("sample.png")
        with (
            patch("main.find_image_files", return_value=[image]),
            patch("main.process_images", return_value=([], [])),
            patch("main.process_detections", return_value=([], [(image, "weights unavailable")])),
            redirect_stdout(io.StringIO()),
            redirect_stderr(error_output),
        ):
            exit_code = main(
                ["--input-dir", "input", "--output-dir", "output", "--detect"]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("Detection failed: sample.png: weights unavailable", error_output.getvalue())

    def test_out_of_range_confidence_is_rejected_before_processing(self) -> None:
        error_output = io.StringIO()
        with (
            patch("main.find_image_files") as find_image_files,
            patch("main.process_images") as process_images,
            redirect_stdout(io.StringIO()),
            redirect_stderr(error_output),
        ):
            exit_code = main(
                ["--input-dir", "input", "--output-dir", "output", "--detect", "--confidence", "1.1"]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("Confidence threshold must be a finite number", error_output.getvalue())
        find_image_files.assert_not_called()
        process_images.assert_not_called()

    def test_rejects_blank_and_whitespace_output_directories(self) -> None:
        for value in ("", " \t "):
            with self.subTest(value=repr(value)):
                with (
                    patch("main.find_image_files") as find_image_files,
                    patch("main.process_images") as process_images,
                ):
                    self._assert_invalid_directory_argument(
                        ["--input-dir", "data\\input", "--output-dir", value],
                        "--output-dir",
                    )

                find_image_files.assert_not_called()
                process_images.assert_not_called()

    def test_region_transform_requires_detection_and_classes(self) -> None:
        for options, message in (
            (["--region-transform", "blur", "--classes", "person"], "requires --detect"),
            (["--detect", "--region-transform", "blur"], "requires --classes"),
            (["--detect", "--region-transform", "blur", "--classes", " , "], "at least one class"),
        ):
            errors = io.StringIO()
            with (
                redirect_stderr(errors),
                redirect_stdout(io.StringIO()),
                patch("main.find_image_files") as find_files,
            ):
                code = main(["--input-dir", "input", "--output-dir", "output", *options])
            self.assertEqual(code, 1)
            self.assertIn(message, errors.getvalue())
            find_files.assert_not_called()

    def test_region_options_call_region_pipeline_and_report_unmatched(self) -> None:
        output = io.StringIO()
        source = Path("input/sample.png")
        destination = Path("output/sample_blur.png")
        with (
            patch("main.find_image_files", return_value=[source]),
            patch("main.process_images", return_value=([], [])),
            patch("main.process_detections", return_value=([], [])) as process_detections,
            patch("main.process_region_transforms", return_value=([], [destination], [], [source])) as process_regions,
            redirect_stdout(output),
            redirect_stderr(io.StringIO()),
        ):
            code = main([
                "--input-dir", "input", "--output-dir", "output", "--detect",
                "--region-transform", "blur", "--classes", "Person,car", "--blur-kernel", "7",
            ])
        self.assertEqual(code, 0)
        process_detections.assert_not_called()
        process_regions.assert_called_once_with(
            [source], "output", ["Person", "car"], "blur",
            confidence_threshold=0.25, blur_kernel=7, pixel_size=12,
        )
        self.assertIn("No detections matched", output.getvalue())

    def test_region_cli_reports_empty_detection_result_without_failing(self) -> None:
        output = io.StringIO()
        source = Path("input/no_objects.png")
        annotated = Path("output/no_objects_detected.png")
        with (
            patch("main.find_image_files", return_value=[source]),
            patch("main.process_images", return_value=([], [])),
            patch("main.process_detections") as process_detections,
            patch("main.process_region_transforms", return_value=([annotated], [], [], [source])),
            redirect_stdout(output),
            redirect_stderr(io.StringIO()),
        ):
            code = main([
                "--input-dir", "input", "--output-dir", "output", "--detect",
                "--region-transform", "blur", "--classes", "person",
            ])

        self.assertEqual(code, 0)
        process_detections.assert_not_called()
        self.assertIn(f"Detected: {annotated}", output.getvalue())
        self.assertIn(f"No detections matched selected classes for {source}.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
