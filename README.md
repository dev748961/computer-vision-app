# Computer Vision Application

A beginner-friendly Python application for batch image processing with OpenCV and optional CPU object detection with a pretrained Ultralytics YOLO11 nano model. It can annotate detections and blur, pixelate, or cover selected detected regions.

## Prerequisites

- Windows 64-bit and PowerShell
- Python 3.13.7 (the project environment has been verified with this version)
- Internet access to install packages and, when weights are not available locally, obtain the pretrained YOLO model
- No NVIDIA GPU is required; inference is configured to use the CPU

## Setup

Run these commands from the project root. The first command uses the Python 3.13 launcher to create `.venv`; activate it before installing dependencies.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The first install command selects CPU builds of PyTorch and torchvision. `requirements.txt` pins OpenCV, NumPy, PyTorch, torchvision, and Ultralytics. If PowerShell blocks activation, bypass activation and use the environment's interpreter directly (this does not change the execution policy):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the application

The CLI reads images directly from the specified input directory (not recursively) and creates the output directory if it does not exist. It recognizes `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp` files, case-insensitively. Other files and subdirectories are ignored. Unreadable images are reported and do not prevent the remaining images from being processed.

Run from the project root so the example paths resolve:

### Default OpenCV processing

```powershell
python main.py --input-dir data\input --output-dir output_default
```

Each readable image is fit within 800 by 800 pixels, blurred with a 3-by-3 Gaussian kernel, given a translucent blue overlay over its center, and marked with a green rectangle and the text `Processed`. The saved filename ends in `_processed`.

### YOLO object detection

```powershell
python main.py --input-dir data\input --output-dir output_detection --detect --confidence 0.4
```

`--detect` saves a separate image with bounding boxes, class names, and confidence values; its filename ends in `_detected`. The minimum confidence defaults to `0.25` and must be between 0 and 1. Detection uses CPU inference.

### Transform selected detected regions

Region transformations require both `--detect` and `--classes`. Class names are comma-separated pretrained model labels and are matched without regard to case. A region-transform run saves the default `_processed` image, the `_detected` annotation, and (when at least one selected class has a valid region) an additional image with the transformation suffix.

Blur detected persons (the blur kernel must be a positive odd number; default `15`):

```powershell
python main.py --input-dir data\input --output-dir output_blur --detect --region-transform blur --classes person --blur-kernel 15
```

Pixelate detected persons (default approximate pixel block size `12`):

```powershell
python main.py --input-dir data\input --output-dir output_pixelation --detect --region-transform pixelate --classes person --pixel-size 12
```

Apply a solid black overlay to detected persons (overlay color and opacity are fixed by the current implementation):

```powershell
python main.py --input-dir data\input --output-dir output_black_overlay --detect --region-transform overlay --classes person
```

The available CLI arguments are:

| Argument | Purpose |
| --- | --- |
| `--input-dir` | Required directory containing input images. |
| `--output-dir` | Required directory for saved results. |
| `--detect` | Run YOLO and save annotated copies; required for region transformations. |
| `--confidence` | YOLO minimum confidence from 0 to 1; default `0.25`. |
| `--region-transform` | One of `blur`, `pixelate`, or `overlay`. Requires `--detect` and `--classes`. |
| `--classes` | Comma-separated class names to transform, such as `person,car`. Requires `--region-transform`. |
| `--blur-kernel` | Positive odd Gaussian kernel size for blur; default `15`. |
| `--pixel-size` | Positive approximate pixel block size for pixelation; default `12`. |

If no detection matches the selected classes, the CLI reports that no detections matched; it still saves the annotated detection image and skips the transformed image. Partially out-of-bounds boxes are clipped and invalid or empty regions are skipped. Overlapping regions are transformed in a stable top-to-bottom, then left-to-right order. Processing errors are reported per image while the batch continues.

## Output protection

The application never writes to input images. It refuses to overwrite an existing output file and reports the affected image as a failure. Use a fresh output directory for each run, as shown in the examples, to avoid collisions with results from earlier runs. Root-level directories whose names begin with `output` are ignored by `.gitignore`.

## Model weights

The detector requests `yolo11n.pt`. This project’s `.gitignore` excludes `*.pt` model files, so weights are not included in source control. If the weights are not already available locally, Ultralytics attempts to download them when detection is first run; internet access is then required. The project does not train a model.

## Project structure and architecture

```text
computer-vision-app/
├── main.py                 # Command-line arguments and workflow coordination
├── requirements.txt        # Pinned runtime dependencies
├── vision_app/
│   ├── image_io.py         # Discover, read, and safely save images
│   ├── detection.py        # Load YOLO, run inference, parse detections
│   ├── transforms.py      # Reusable OpenCV image and region operations
│   └── pipeline.py        # Batch processing, annotation, and region transforms
├── data/input/             # Input images
├── output/                 # Example output directory
└── tests/                  # unittest coverage for I/O, CLI, detection, pipeline, transforms
```

`main.py` validates CLI options and connects the reusable functions. `image_io.py` handles supported files and safe output writes. `detection.py` wraps CPU YOLO inference and returns validated detection records. `transforms.py` implements operations that return new arrays. `pipeline.py` coordinates batch processing, uses one inference per image during detection-plus-region-transform runs, saves distinct outputs, and records individual failures.

## Known limitations

- YOLO can miss objects or classify them incorrectly.
- A detection box may be incomplete or imprecise, so a transformation may leave parts of an object visible or affect nearby pixels.
- Inference runs on the CPU and can be slow, particularly for large images or batches.
- Blur, pixelation, and overlays operate only inside detected regions. They do not guarantee privacy or anonymization.
- The model recognizes only the classes supported by its pretrained weights; class selection uses those labels.

## Tests

The automated tests use Python’s built-in `unittest` framework and mock YOLO inference where appropriate, so they do not require a model download. From the project root with `.venv` activated, run:

```powershell
python -m unittest discover -s tests -v
```

Alternatively, run the tests without activating the environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
