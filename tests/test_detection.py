"""Offline tests for YOLO result parsing and input validation."""

import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import sys

import numpy as np

from vision_app.detection import (
    Detection,
    DetectionError,
    detect_objects,
    load_yolo_model,
    parse_yolo_results,
)


class FakeBox:
    def __init__(self, class_id: int, confidence: float, coordinates: list[float]):
        self.cls = np.array([class_id], dtype=np.float32)
        self.conf = np.array([confidence], dtype=np.float32)
        self.xyxy = np.array([coordinates], dtype=np.float32)


class FakeResult:
    def __init__(self, boxes: list[FakeBox] | None, names: dict[int, str] | None = None):
        self.boxes = boxes
        self.names = names or {0: "person", 2: "car"}


class DetectionTests(unittest.TestCase):
    def test_parses_classes_scores_and_box_coordinates(self) -> None:
        parsed = parse_yolo_results(
            [FakeResult([FakeBox(0, 0.91, [1, 2, 30, 40]), FakeBox(2, 0.75, [5, 6, 20, 25])])]
        )

        self.assertEqual(parsed[0], Detection(0, "person", float(np.float32(0.91)), (1.0, 2.0, 30.0, 40.0)))
        self.assertEqual(parsed[1].class_name, "car")
        self.assertAlmostEqual(parsed[1].confidence, 0.75)

    def test_parses_results_with_no_boxes(self) -> None:
        self.assertEqual(parse_yolo_results([FakeResult(None)]), [])

    def test_rejects_invalid_result_fields(self) -> None:
        invalid_class = FakeBox(0, 0.5, [1, 2, 3, 4])
        invalid_class.cls = np.array([0.5])
        with self.assertRaisesRegex(ValueError, "invalid class identifier"):
            parse_yolo_results([FakeResult([invalid_class])])

        with self.assertRaisesRegex(ValueError, "no result object"):
            parse_yolo_results(None)

    def test_detection_model_receives_confidence_and_returns_parsed_objects(self) -> None:
        model = Mock()
        model.predict.return_value = [FakeResult([FakeBox(0, 0.8, [0, 1, 8, 9])])]

        found = detect_objects(np.zeros((10, 10, 3), dtype=np.uint8), 0.6, model=model)

        self.assertEqual(len(found), 1)
        model.predict.assert_called_once()
        self.assertEqual(model.predict.call_args.kwargs["conf"], 0.6)
        self.assertEqual(model.predict.call_args.kwargs["device"], "cpu")

    def test_invalid_images_or_thresholds_are_rejected_before_inference(self) -> None:
        model = Mock()
        with self.assertRaisesRegex(ValueError, "non-empty NumPy array"):
            detect_objects(np.array([]), model=model)
        with self.assertRaisesRegex(ValueError, "three-channel BGR"):
            detect_objects(np.zeros((4, 4), dtype=np.uint8), model=model)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            detect_objects(np.zeros((4, 4, 3), dtype=np.uint8), 1.1, model=model)
        model.predict.assert_not_called()

    def test_model_loading_and_inference_errors_are_clear(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        with patch("vision_app.detection.load_yolo_model", side_effect=DetectionError("weight file missing")):
            with self.assertRaisesRegex(DetectionError, "weight file missing"):
                detect_objects(image)

        model = Mock()
        model.predict.side_effect = RuntimeError("device problem")
        with self.assertRaisesRegex(DetectionError, "YOLO inference failed: device problem"):
            detect_objects(image, model=model)

        fake_ultralytics = SimpleNamespace(
            YOLO=Mock(side_effect=RuntimeError("weight download failed"))
        )
        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            with self.assertRaisesRegex(DetectionError, "Unable to load YOLO model.*weight download failed"):
                load_yolo_model()

    def test_detection_values_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "confidence"):
            Detection(0, "person", 1.2, (1, 2, 3, 4))
        with self.assertRaisesRegex(ValueError, "bbox"):
            Detection(0, "person", 0.5, (4, 2, 3, 4))


if __name__ == "__main__":
    unittest.main()
