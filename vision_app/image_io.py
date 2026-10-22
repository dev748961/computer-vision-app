"""Helpers for finding, reading, and preparing image files."""

from pathlib import Path

import cv2
import numpy as np


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_image_files(directory: str | Path) -> list[Path]:
    """Return supported image files directly inside *directory*, sorted by name.

    Raises a clear path error when the directory is missing, invalid, or cannot
    be read. Subdirectories are not searched recursively.
    """
    try:
        input_directory = Path(directory)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid input directory: {directory!r}") from error

    if not input_directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_directory}")
    if not input_directory.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_directory}")

    try:
        image_files = [
            path
            for path in input_directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
    except OSError as error:
        raise OSError(f"Unable to read input directory '{input_directory}': {error}") from error

    return sorted(image_files, key=lambda path: path.name.casefold())


def create_output_directory(directory: str | Path) -> Path:
    """Create the output directory, including any missing parent directories."""
    try:
        output_directory = Path(directory)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid output directory: {directory!r}") from error

    if output_directory.exists() and not output_directory.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_directory}")

    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OSError(
            f"Unable to create output directory '{output_directory}': {error}"
        ) from error

    return output_directory


def read_image(image_path: str | Path) -> np.ndarray | None:
    """Read an image as a color array, or return None if it cannot be decoded."""
    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    except (cv2.error, OSError, ValueError):
        return None

    return image


def save_image(image: np.ndarray, image_path: str | Path) -> Path:
    """Save an image without replacing an existing file.

    The parent directory must already exist. Raises a clear error for invalid
    image data, paths, or failed OpenCV writes.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Image must be a non-empty NumPy array.")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3, 4)):
        raise ValueError("Image must have one, three, or four channels.")

    try:
        output_path = Path(image_path)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid output image path: {image_path!r}") from error

    if output_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported output image format: {output_path.suffix or '(none)'}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output_path.parent}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing image: {output_path}")

    try:
        was_written = cv2.imwrite(str(output_path), image)
    except (cv2.error, OSError, ValueError) as error:
        raise OSError(f"Unable to save image to '{output_path}': {error}") from error

    if not was_written:
        raise OSError(f"OpenCV could not save image to '{output_path}'.")

    return output_path
