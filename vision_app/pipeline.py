"""Batch processing for the Day 2 image workflow and optional YOLO detection."""

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from vision_app.detection import (
    Detection,
    DetectionError,
    detect_objects,
    load_yolo_model,
    validate_confidence_threshold,
)
from vision_app.image_io import create_output_directory, read_image, save_image
from vision_app.transforms import (
    apply_color_overlay,
    apply_gaussian_blur,
    draw_rectangle,
    draw_text,
    pixelate_region,
    resize_image,
)


MAX_IMAGE_SIDE = 800


def process_image(image_path: str | Path, output_directory: str | Path) -> Path:
    """Apply the default Day 2 sequence and save a separate result.

    The sequence fits the image within 800 by 800 pixels, applies a light
    3-by-3 Gaussian blur, adds a translucent blue overlay to the center, then
    marks that region with a green rectangle and a small label. The source
    image is loaded once and never written to.
    """
    source_path = Path(image_path)
    output_directory = Path(output_directory)
    image = read_image(source_path)
    if image is None:
        raise ValueError(f"Could not read input image: {source_path}")

    image_height, image_width = image.shape[:2]
    scale = min(1.0, MAX_IMAGE_SIDE / max(image_width, image_height))
    resized_width = max(1, round(image_width * scale))
    resized_height = max(1, round(image_height * scale))
    result = resize_image(image, resized_width, resized_height)
    result = apply_gaussian_blur(result, kernel_size=3)

    result_height, result_width = result.shape[:2]
    overlay_width = max(1, result_width // 2)
    overlay_height = max(1, result_height // 2)
    overlay_x = (result_width - overlay_width) // 2
    overlay_y = (result_height - overlay_height) // 2
    result = apply_color_overlay(
        result,
        overlay_x,
        overlay_y,
        overlay_width,
        overlay_height,
        color=(255, 0, 0),
        alpha=0.2,
    )
    result = draw_rectangle(
        result,
        overlay_x,
        overlay_y,
        overlay_width,
        overlay_height,
        color=(0, 255, 0),
        thickness=2,
    )
    result = draw_text(
        result,
        "Processed",
        5,
        min(24, result_height - 1),
        color=(255, 255, 255),
        font_scale=0.5,
        thickness=1,
    )

    output_name = f"{source_path.stem}_processed{source_path.suffix.lower()}"
    output_path = output_directory / output_name
    if output_path.resolve() == source_path.resolve():
        raise ValueError(f"Refusing to overwrite input image: {source_path}")
    return save_image(result, output_path)


def process_images(
    image_paths: Iterable[str | Path], output_directory: str | Path
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Process each image independently and return saved paths and failures."""
    output_directory = create_output_directory(output_directory)
    processed_paths: list[Path] = []
    failures: list[tuple[Path, str]] = []

    for image_path in image_paths:
        source_path = Path(image_path)
        try:
            processed_paths.append(process_image(source_path, output_directory))
        except (OSError, ValueError, cv2.error) as error:
            failures.append((source_path, str(error)))

    return processed_paths, failures


def annotate_detections(
    image: np.ndarray, detections: Iterable[Detection]
) -> np.ndarray:
    """Return a copy of *image* marked with object boxes and confidence labels."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Image must be a non-empty NumPy array.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be a three-channel BGR array.")

    result = image.copy()
    height, width = result.shape[:2]
    label_rows: dict[tuple[int, int], int] = {}
    for detection in detections:
        x1, y1, x2, y2 = (round(value) for value in detection.bbox)
        x1 = min(max(x1, 0), width - 1)
        y1 = min(max(y1, 0), height - 1)
        x2 = min(max(x2, x1), width - 1)
        y2 = min(max(y2, y1), height - 1)
        result = draw_rectangle(
            result,
            x1,
            y1,
            x2 - x1 + 1,
            y2 - y1 + 1,
            color=(0, 255, 0),
            thickness=2,
        )
        label = f"{detection.class_name} {detection.confidence:.2f}"
        label_group = (x1 // 24, y1 // 24)
        row = label_rows.get(label_group, 0)
        label_rows[label_group] = row + 1
        text_y = y1 - 5 - row * 16
        if text_y < 12:
            text_y = min(height - 1, y1 + 15 + row * 16)
        result = draw_text(result, label, x1, text_y, color=(0, 255, 0), font_scale=0.5)
    return result


def process_detections(
    image_paths: Iterable[str | Path],
    output_directory: str | Path,
    confidence_threshold: float = 0.25,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Detect objects in each image and save separate annotated outputs."""
    paths = [Path(path) for path in image_paths]
    output_directory = create_output_directory(output_directory)
    try:
        threshold = validate_confidence_threshold(confidence_threshold)
    except ValueError as error:
        return [], [(path, str(error)) for path in paths]

    if not paths:
        return [], []

    try:
        model = load_yolo_model()
    except (DetectionError, ValueError) as error:
        return [], [(path, str(error)) for path in paths]

    detected_paths: list[Path] = []
    failures: list[tuple[Path, str]] = []
    for source_path in paths:
        try:
            image = read_image(source_path)
            if image is None:
                raise ValueError(f"Could not read input image: {source_path}")
            detections = detect_objects(image, threshold, model=model)
            annotated = annotate_detections(image, detections)
            output_path = output_directory / f"{source_path.stem}_detected{source_path.suffix.lower()}"
            if output_path.resolve() == source_path.resolve():
                raise ValueError(f"Refusing to overwrite input image: {source_path}")
            detected_paths.append(save_image(annotated, output_path))
        except (DetectionError, OSError, ValueError, cv2.error) as error:
            failures.append((source_path, str(error)))

    return detected_paths, failures


def transform_detection_regions(
    image: np.ndarray,
    detections: Iterable[Detection],
    transform: str,
    selected_classes: Iterable[str],
    *,
    blur_kernel: int = 15,
    pixel_size: int = 12,
    overlay_color: tuple[int, int, int] = (0, 0, 0),
    overlay_alpha: float = 1.0,
) -> tuple[np.ndarray, list[Detection]]:
    """Apply one transform to selected detection boxes in stable top-left order.

    Boxes are rounded outward and clipped to image bounds. Invalid, empty, and
    wholly out-of-bounds boxes are skipped. Overlapping boxes are transformed
    sequentially in (top, left, bottom, right, class id) order.
    """
    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be a non-empty three-channel BGR array.")
    if transform not in {"blur", "pixelate", "overlay"}:
        raise ValueError("Transform must be blur, pixelate, or overlay.")
    classes = {name.strip().casefold() for name in selected_classes if isinstance(name, str) and name.strip()}
    if not classes:
        raise ValueError("At least one object class must be selected.")
    if transform == "blur" and (
        isinstance(blur_kernel, bool) or not isinstance(blur_kernel, int)
        or blur_kernel <= 0 or blur_kernel % 2 == 0
    ):
        raise ValueError("blur_kernel must be a positive odd integer.")
    if transform == "pixelate" and (
        isinstance(pixel_size, bool) or not isinstance(pixel_size, int) or pixel_size <= 0
    ):
        raise ValueError("pixel_size must be a positive integer.")

    height, width = image.shape[:2]
    candidates = [d for d in detections if d.class_name.casefold() in classes]
    candidates.sort(key=lambda d: (d.bbox[1], d.bbox[0], d.bbox[3], d.bbox[2], d.class_id))
    result = image.copy()
    matched: list[Detection] = []
    for detection in candidates:
        try:
            x1, y1, x2, y2 = detection.bbox
            if any(not np.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                continue
            left, top = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
            right, bottom = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
            if right <= left or bottom <= top:
                continue
            args = (result, left, top, right - left, bottom - top)
            if transform == "blur":
                # Blur the current crop so overlaps have deterministic cumulative effects.
                blurred_region = apply_gaussian_blur(
                    result[top:bottom, left:right].copy(), kernel_size=blur_kernel
                )
                result[top:bottom, left:right] = blurred_region
            elif transform == "pixelate":
                result = pixelate_region(*args, pixel_size=pixel_size)
            else:
                result = apply_color_overlay(*args, color=overlay_color, alpha=overlay_alpha)
            matched.append(detection)
        except (TypeError, ValueError, cv2.error):
            continue
    return result, matched


def process_region_transforms(
    image_paths: Iterable[str | Path],
    output_directory: str | Path,
    selected_classes: Iterable[str],
    transform: str,
    confidence_threshold: float = 0.25,
    *,
    blur_kernel: int = 15,
    pixel_size: int = 12,
) -> tuple[list[Path], list[Path], list[tuple[Path, str]], list[Path]]:
    """Detect once per image and save annotated and selected-region outputs.

    Return annotated paths, transformed paths, failures, and images without a
    matching region. The same detections are reused for both output images.
    """
    paths = [Path(path) for path in image_paths]
    output_directory = create_output_directory(output_directory)
    try:
        threshold = validate_confidence_threshold(confidence_threshold)
    except ValueError as error:
        return [], [], [(path, str(error)) for path in paths], []
    if not paths:
        return [], [], [], []
    try:
        model = load_yolo_model()
    except (DetectionError, ValueError) as error:
        return [], [], [(path, str(error)) for path in paths], []

    detected_paths: list[Path] = []
    saved: list[Path] = []
    failures: list[tuple[Path, str]] = []
    unmatched: list[Path] = []
    for source_path in paths:
        try:
            image = read_image(source_path)
            if image is None:
                raise ValueError(f"Could not read input image: {source_path}")
            detections = detect_objects(image, threshold, model=model)
        except (DetectionError, OSError, ValueError, cv2.error) as error:
            failures.append((source_path, str(error)))
            continue

        # Preserve the ordinary --detect output while sharing this inference.
        try:
            annotated = annotate_detections(image, detections)
            detected_output = output_directory / f"{source_path.stem}_detected{source_path.suffix.lower()}"
            if detected_output.resolve() == source_path.resolve():
                raise ValueError(f"Refusing to overwrite input image: {source_path}")
            detected_paths.append(save_image(annotated, detected_output))
        except (DetectionError, OSError, ValueError, cv2.error) as error:
            failures.append((source_path, str(error)))

        try:
            result, matched = transform_detection_regions(
                image, detections, transform, selected_classes,
                blur_kernel=blur_kernel, pixel_size=pixel_size,
            )
            if not matched:
                unmatched.append(source_path)
                continue
            # Annotate only the detections actually transformed.
            result = annotate_detections(result, matched)
            output_path = output_directory / f"{source_path.stem}_{transform}{source_path.suffix.lower()}"
            if output_path.resolve() == source_path.resolve():
                raise ValueError(f"Refusing to overwrite input image: {source_path}")
            saved.append(save_image(result, output_path))
        except (DetectionError, OSError, ValueError, cv2.error) as error:
            failures.append((source_path, str(error)))
    return detected_paths, saved, failures, unmatched
