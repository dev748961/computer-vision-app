"""Object detection helpers backed by a small pretrained Ultralytics YOLO model."""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

import numpy as np


DEFAULT_MODEL = "yolo11n.pt"


class DetectionError(RuntimeError):
    """Raised when a YOLO model cannot be loaded or run."""


@dataclass(frozen=True)
class Detection:
    """One detected object with pixel coordinates in xyxy order."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, Integral) or self.class_id < 0:
            raise ValueError("Detection class_id must be a non-negative integer.")
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("Detection class_name must not be empty.")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, Real):
            raise ValueError("Detection confidence must be a number from 0 to 1.")
        if not math.isfinite(float(self.confidence)) or not 0 <= self.confidence <= 1:
            raise ValueError("Detection confidence must be a finite number from 0 to 1.")
        if not isinstance(self.bbox, (tuple, list)) or len(self.bbox) != 4:
            raise ValueError("Detection bbox must contain four xyxy coordinates.")
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) for value in self.bbox):
            raise ValueError("Detection bbox coordinates must be finite numbers.")
        x1, y1, x2, y2 = self.bbox
        if x2 < x1 or y2 < y1:
            raise ValueError("Detection bbox maximum coordinates must not be less than minimum coordinates.")


def validate_confidence_threshold(confidence_threshold: float) -> float:
    """Return a valid confidence threshold between zero and one."""
    if isinstance(confidence_threshold, bool) or not isinstance(confidence_threshold, Real):
        raise ValueError("Confidence threshold must be a number between 0 and 1.")
    threshold = float(confidence_threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Confidence threshold must be a finite number between 0 and 1.")
    return threshold


def _to_list(value: Any) -> list[Any]:
    """Convert a tensor-like value to a plain, one-dimensional Python list."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return [value]
    values = list(value)
    while len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = list(values[0])
    return values


def parse_yolo_results(results: Any) -> list[Detection]:
    """Parse Ultralytics prediction results into simple Detection objects."""
    if results is None:
        raise ValueError("YOLO returned no result object.")
    if not isinstance(results, (list, tuple)):
        results = [results]

    detections: list[Detection] = []
    try:
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            names = result.names
            for box in boxes:
                raw_class_id = float(_to_list(box.cls)[0])
                if not math.isfinite(raw_class_id) or not raw_class_id.is_integer():
                    raise ValueError("YOLO returned an invalid class identifier.")
                class_id = int(raw_class_id)
                confidence = float(_to_list(box.conf)[0])
                coordinates = _to_list(box.xyxy)
                if len(coordinates) != 4:
                    raise ValueError("YOLO returned a bounding box without four coordinates.")
                if isinstance(names, dict):
                    class_name = names[class_id]
                else:
                    class_name = names[class_id]
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=str(class_name),
                        confidence=confidence,
                        bbox=tuple(float(value) for value in coordinates),
                    )
                )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, ValueError) and str(error).startswith("YOLO returned"):
            raise
        raise ValueError(f"Could not parse YOLO detection results: {error}") from error
    return detections


def load_yolo_model(model_name: str = DEFAULT_MODEL) -> Any:
    """Load the small pretrained YOLO11 nano model, reporting setup errors clearly."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Model name or path must not be empty.")
    try:
        from ultralytics import YOLO

        return YOLO(model_name)
    except Exception as error:
        raise DetectionError(f"Unable to load YOLO model '{model_name}': {error}") from error


def detect_objects(
    image: np.ndarray,
    confidence_threshold: float = 0.25,
    model: Any | None = None,
) -> list[Detection]:
    """Detect objects in a BGR image and return names, scores, and xyxy boxes.

    A caller may pass an already-loaded model for reuse or testing. Without one,
    the lightweight pretrained ``yolo11n.pt`` model is loaded on demand.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Image must be a non-empty NumPy array.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be a three-channel BGR array.")
    threshold = validate_confidence_threshold(confidence_threshold)
    detector = model if model is not None else load_yolo_model()
    try:
        results = detector.predict(
            source=image, conf=threshold, device="cpu", verbose=False
        )
    except Exception as error:
        if isinstance(error, DetectionError):
            raise
        raise DetectionError(f"YOLO inference failed: {error}") from error
    try:
        return parse_yolo_results(results)
    except ValueError as error:
        raise DetectionError(str(error)) from error
