"""Tests for image file discovery and error handling."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vision_app.image_io import (
    create_output_directory,
    find_image_files,
    read_image,
    save_image,
)


class ImageIoTests(unittest.TestCase):
    def setUp(self) -> None:
        # Keep temporary test files under the project, where this environment
        # allows the test process to create and remove them.
        self.temporary_directory = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_finds_supported_extensions_case_insensitively(self) -> None:
        input_directory = self.root / "input"
        input_directory.mkdir()
        names = ["c.webp", "a.JPG", "b.jpeg", "d.PNG", "e.bmp"]
        for name in names:
            (input_directory / name).touch()
        (input_directory / "notes.txt").touch()
        (input_directory / "nested.jpg").mkdir()

        found = find_image_files(input_directory)

        self.assertEqual([path.name for path in found], sorted(names, key=str.casefold))

    def test_missing_input_directory_raises_clear_error(self) -> None:
        missing_directory = self.root / "missing"

        with self.assertRaisesRegex(FileNotFoundError, "Input directory does not exist"):
            find_image_files(missing_directory)

    def test_input_path_that_is_a_file_raises_clear_error(self) -> None:
        input_file = self.root / "not-a-directory.txt"
        input_file.touch()

        with self.assertRaisesRegex(NotADirectoryError, "Input path is not a directory"):
            find_image_files(input_file)

    def test_creates_output_directory_and_missing_parents(self) -> None:
        output_directory = self.root / "results" / "images"

        created_directory = create_output_directory(output_directory)

        self.assertEqual(created_directory, output_directory)
        self.assertTrue(output_directory.is_dir())

    def test_output_path_that_is_a_file_raises_clear_error(self) -> None:
        output_file = self.root / "output.txt"
        output_file.touch()

        with self.assertRaisesRegex(NotADirectoryError, "Output path is not a directory"):
            create_output_directory(output_file)

    def test_reads_a_valid_image(self) -> None:
        import cv2

        image_path = self.root / "valid.png"
        source_image = np.zeros((3, 4, 3), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(image_path), source_image))

        loaded_image = read_image(image_path)

        self.assertIsNotNone(loaded_image)
        self.assertEqual(loaded_image.shape, source_image.shape)

    def test_unreadable_image_returns_none_without_raising(self) -> None:
        image_path = self.root / "unreadable.png"

        with patch("vision_app.image_io.cv2.imread", return_value=None):
            self.assertIsNone(read_image(image_path))

    def test_saves_image_without_replacing_existing_files(self) -> None:
        image_path = self.root / "saved.png"
        image = np.zeros((3, 4, 3), dtype=np.uint8)

        saved_path = save_image(image, image_path)

        self.assertEqual(saved_path, image_path)
        self.assertTrue(image_path.is_file())
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            save_image(image, image_path)

    def test_save_reports_unsupported_extension_and_missing_directory(self) -> None:
        image = np.zeros((3, 4, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "Unsupported output image format"):
            save_image(image, self.root / "image.gif")
        with self.assertRaisesRegex(FileNotFoundError, "Output directory does not exist"):
            save_image(image, self.root / "missing" / "image.png")

    def test_save_reports_failed_opencv_write(self) -> None:
        image = np.zeros((3, 4, 3), dtype=np.uint8)

        with patch("vision_app.image_io.cv2.imwrite", return_value=False):
            with self.assertRaisesRegex(OSError, "OpenCV could not save image"):
                save_image(image, self.root / "image.png")


if __name__ == "__main__":
    unittest.main()
