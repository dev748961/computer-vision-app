"""Reusable image transformations that leave their input arrays unchanged."""

import math
from numbers import Integral, Real

import cv2
import numpy as np


def _validate_image(image: np.ndarray) -> None:
    """Check that an image is a non-empty grayscale, BGR, or BGRA array."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")
    if image.size == 0:
        raise ValueError("Image must not be empty.")
    if image.ndim not in (2, 3):
        raise ValueError("Image must be a 2D grayscale or 3D color array.")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError("Color images must have one, three, or four channels.")


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _validate_region(
    image: np.ndarray, x: int, y: int, width: int, height: int
) -> tuple[int, int, int, int]:
    """Validate a rectangular region and return its integer coordinates."""
    _validate_image(image)
    for value, name in ((x, "x"), (y, "y")):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} coordinate must be an integer.")
    width = _positive_integer(width, "width")
    height = _positive_integer(height, "height")
    x, y = int(x), int(y)
    image_height, image_width = image.shape[:2]
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError("Region coordinates must fit entirely inside the image.")
    return x, y, width, height


def _validate_bgr_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Validate and normalize a three-channel BGR color."""
    if not isinstance(color, (tuple, list)) or len(color) != 3:
        raise ValueError("Color must be a three-value BGR tuple.")
    if any(
        isinstance(channel, bool)
        or not isinstance(channel, Integral)
        or not 0 <= channel <= 255
        for channel in color
    ):
        raise ValueError("BGR color channels must be integers from 0 to 255.")
    return tuple(int(channel) for channel in color)


def _validate_bgr_image(image: np.ndarray) -> None:
    _validate_image(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("This operation requires a three-channel BGR image.")


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an image to the requested positive pixel dimensions."""
    _validate_image(image)
    width = _positive_integer(width, "width")
    height = _positive_integer(height, "height")
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Return a copy of the region at (x, y) with the given pixel size."""
    x, y, width, height = _validate_region(image, x, y, width, height)
    return image[y : y + height, x : x + width].copy()


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale, BGR, or BGRA image to a new grayscale array."""
    _validate_image(image)
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 1:
        return image[:, :, 0].copy()
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur using a positive odd square kernel size."""
    _validate_image(image)
    kernel_size = _positive_integer(kernel_size, "kernel_size")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)


def pixelate_region(
    image: np.ndarray, x: int, y: int, width: int, height: int, pixel_size: int = 12
) -> np.ndarray:
    """Pixelate a rectangular region with square blocks of approximately pixel_size.

    The region must be fully inside the image. A new image is returned and the
    input array is never changed.
    """
    _validate_bgr_image(image)
    x, y, width, height = _validate_region(image, x, y, width, height)
    pixel_size = _positive_integer(pixel_size, "pixel_size")
    region = image[y : y + height, x : x + width]
    small_width = max(1, (width + pixel_size - 1) // pixel_size)
    small_height = max(1, (height + pixel_size - 1) // pixel_size)
    small = cv2.resize(region, (small_width, small_height), interpolation=cv2.INTER_AREA)
    pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    result = image.copy()
    result[y : y + height, x : x + width] = pixelated
    return result


def draw_rectangle(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a BGR rectangle around a pixel region on a copy of the image."""
    _validate_bgr_image(image)
    x, y, width, height = _validate_region(image, x, y, width, height)
    color = _validate_bgr_color(color)
    thickness = _positive_integer(thickness, "thickness")
    result = image.copy()
    cv2.rectangle(result, (x, y), (x + width - 1, y + height - 1), color, thickness)
    return result


def draw_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.7,
    thickness: int = 1,
) -> np.ndarray:
    """Draw BGR text at its baseline position on a copy of the image."""
    _validate_bgr_image(image)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must not be empty.")
    for value, name in ((x, "x"), (y, "y")):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"Text {name} coordinate must be an integer.")
    x, y = int(x), int(y)
    image_height, image_width = image.shape[:2]
    if not 0 <= x < image_width or not 0 <= y < image_height:
        raise ValueError("Text coordinates must be inside the image.")
    if isinstance(font_scale, bool) or not isinstance(font_scale, Real):
        raise ValueError("font_scale must be a positive number.")
    font_scale = float(font_scale)
    if not math.isfinite(font_scale) or font_scale <= 0:
        raise ValueError("font_scale must be a positive finite number.")
    color = _validate_bgr_color(color)
    thickness = _positive_integer(thickness, "thickness")

    result = image.copy()
    cv2.putText(
        result,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return result


def apply_color_overlay(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.3,
) -> np.ndarray:
    """Blend a solid BGR color over a rectangular region on a copy of the image."""
    _validate_bgr_image(image)
    x, y, width, height = _validate_region(image, x, y, width, height)
    color = _validate_bgr_color(color)
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise ValueError("alpha must be a number between 0 and 1.")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be a finite number between 0 and 1.")

    result = image.copy()
    region = result[y : y + height, x : x + width]
    color_layer = np.full_like(region, color)
    region[:] = cv2.addWeighted(region, 1 - alpha, color_layer, alpha, 0)
    return result
