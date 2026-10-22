"""Command-line entry point for image processing and optional YOLO detection."""

import argparse
import sys
from collections.abc import Sequence

from vision_app.detection import validate_confidence_threshold
from vision_app.image_io import find_image_files
from vision_app.pipeline import process_detections, process_images, process_region_transforms


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Process supported images and optionally detect objects with YOLO."
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory containing input images."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for processed image files."
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Also run pretrained YOLO detection and save annotated images.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum YOLO confidence from 0 to 1 (default: 0.25).",
    )
    parser.add_argument(
        "--region-transform", choices=("blur", "pixelate", "overlay"),
        help="Transform selected YOLO bounding-box regions.",
    )
    parser.add_argument(
        "--classes", help="Comma-separated YOLO class names to transform (for example: person,car).",
    )
    parser.add_argument(
        "--blur-kernel", type=int, default=15,
        help="Odd Gaussian blur kernel size for --region-transform blur (default: 15).",
    )
    parser.add_argument(
        "--pixel-size", type=int, default=12,
        help="Approximate pixel block size for --region-transform pixelate (default: 12).",
    )
    return parser


def _validate_directory_argument(value: str, option_name: str) -> str:
    """Reject blank directory arguments before they are interpreted as paths."""
    if not value.strip():
        raise ValueError(f"{option_name} must not be empty or whitespace-only.")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Process images and, when requested, save YOLO-annotated copies."""
    args = build_parser().parse_args(argv)

    try:
        input_directory = _validate_directory_argument(args.input_dir, "--input-dir")
        output_argument = _validate_directory_argument(args.output_dir, "--output-dir")
        if args.region_transform and not args.detect:
            raise ValueError("--region-transform requires --detect.")
        if args.region_transform and not args.classes:
            raise ValueError("--region-transform requires --classes with at least one class name.")
        if args.classes and not args.region_transform:
            raise ValueError("--classes can only be used with --region-transform.")
        classes = []
        if args.region_transform:
            classes = [item.strip() for item in args.classes.split(",") if item.strip()]
            if not classes:
                raise ValueError("--classes must contain at least one class name.")
            if args.region_transform == "blur" and (args.blur_kernel <= 0 or args.blur_kernel % 2 == 0):
                raise ValueError("--blur-kernel must be a positive odd integer.")
            if args.region_transform == "pixelate" and args.pixel_size <= 0:
                raise ValueError("--pixel-size must be a positive integer.")
        if args.detect:
            validate_confidence_threshold(args.confidence)
        image_files = find_image_files(input_directory)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Found {len(image_files)} supported image(s) in '{args.input_dir}':")
    for image_path in image_files:
        print(f"- {image_path}")

    try:
        processed_paths, failures = process_images(image_files, output_argument)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    for output_path in processed_paths:
        print(f"Processed: {output_path}")
    for input_path, error in failures:
        print(f"Failed: {input_path}: {error}", file=sys.stderr)

    detected_count = 0
    if args.detect and not args.region_transform:
        detected_paths, detection_failures = process_detections(
            image_files, output_argument, confidence_threshold=args.confidence
        )
        for output_path in detected_paths:
            print(f"Detected: {output_path}")
        for input_path, error in detection_failures:
            print(f"Detection failed: {input_path}: {error}", file=sys.stderr)
        detected_count = len(detected_paths)
        failures.extend(detection_failures)

    if args.region_transform:
        detected_paths, region_paths, region_failures, unmatched_paths = process_region_transforms(
            image_files,
            output_argument,
            classes,
            args.region_transform,
            confidence_threshold=args.confidence,
            blur_kernel=args.blur_kernel,
            pixel_size=args.pixel_size,
        )
        for output_path in detected_paths:
            print(f"Detected: {output_path}")
        detected_count = len(detected_paths)
        for output_path in region_paths:
            print(f"Transformed: {output_path}")
        for input_path in unmatched_paths:
            print(f"No detections matched selected classes for {input_path}.")
        for input_path, error in region_failures:
            print(f"Region transform failed: {input_path}: {error}", file=sys.stderr)
        failures.extend(region_failures)
        print(f"Saved {len(region_paths)} transformed image(s) to '{output_argument}'.")

    print(f"Saved {len(processed_paths)} processed image(s) to '{output_argument}'.")
    if args.detect:
        print(f"Saved {detected_count} detection image(s) to '{output_argument}'.")

    if failures:
        print(f"Failed images: {len(failures)}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
