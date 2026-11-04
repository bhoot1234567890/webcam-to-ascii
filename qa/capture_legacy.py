"""One-time capture of the unmodified webcam renderer and its real helpers.

Never run this from normal tests. An existing output directory is an error,
including an incomplete capture: investigate it rather than replacing answers.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib.metadata
from io import StringIO
import json
from pathlib import Path
import platform
import runpy
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from qa.fixture_io import (
    load_array, load_bytes, load_manifest, load_text, sha256, write_array, write_blob,
)

ROOT = Path(__file__).resolve().parents[1]
SIGMA_PAIRS = [(1, 2), (0.5, 1.5), (2, 3), (1, 1), (2, 1), (0.5, 5)]
DEFAULTS = {
    "maxWidthChars": 150, "maxHeightLines": 200,
    "sigma1": 1, "sigma2": 2, "edgeIndexThreshold": 90,
}


class FiniteCapture:
    """Supply the real fixture once, then the real loop's end-of-stream."""

    def __init__(self, frame):
        self.frame = frame
        self.reads = 0
        self.released = False

    def read(self):
        self.reads += 1
        if self.reads == 1:
            return True, self.frame
        return False, None

    def release(self):
        self.released = True


def capture_frame(source: Path, original: bytes, image: Image.Image):
    frame = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    adapter = FiniteCapture(frame)

    def video_capture(device):
        if device != 0:
            raise ValueError(f"Legacy capture unexpectedly requested device {device}")
        return adapter

    if source.read_bytes() != original:
        raise ValueError(f"{source}: renderer changed before baseline capture")
    stdout = StringIO()
    # Acquisition and GUI entry points only. The renderer's numerical processing,
    # resizing, loop, imports, helpers and print statements execute unchanged.
    with patch.object(cv2, "VideoCapture", video_capture), \
            patch.object(cv2, "imshow", lambda *_: None), \
            patch.object(cv2, "waitKey", lambda *_: -1), \
            patch.object(cv2, "destroyAllWindows", lambda: None), \
            redirect_stdout(stdout):
        namespace = runpy.run_path(str(source), run_name="__legacy_baseline__")
    if source.read_bytes() != original:
        raise ValueError(f"{source}: renderer changed during baseline capture")
    if adapter.reads != 2 or not adapter.released:
        raise ValueError("Legacy loop did not consume exactly one frame and release acquisition")
    printed = stdout.getvalue()
    if not printed.endswith("\n"):
        raise ValueError("Legacy stdout missing exactly removable final print newline")
    text = printed[:-1]  # Remove exactly one final print newline, never strip spaces.
    width, height = int(namespace["new_width"]), int(namespace["new_height"])
    rows = text.split("\n")
    if len(rows) != height or any(len(row) != width for row in rows):
        raise ValueError("Legacy stdout dimensions disagree with the original resize loop")
    return text, width, height, namespace


def sobel_outputs(namespace, pixels, out):
    magnitude, radians, angles = namespace["sobel_edge_detection_with_angle"](pixels)
    # Independent diagnostic operations are the actual imported OpenCV functions.
    gx = namespace["cv2"].Sobel(pixels, cv2.CV_64F, 1, 0, ksize=3)
    gy = namespace["cv2"].Sobel(pixels, cv2.CV_64F, 0, 1, ksize=3)
    return {
        "gx": write_array(out, gx), "gy": write_array(out, gy),
        "magnitude": write_array(out, magnitude),
        "angles": write_array(out, angles), "angleRadians": write_array(out, radians),
    }


def freeze(out: Path, images_manifest: Path, source: Path) -> dict:
    if out.exists():
        raise ValueError(f"{out}: refusing to overwrite existing legacy capture")
    images = load_manifest(images_manifest)
    original = source.read_bytes()
    out.mkdir(parents=True, exist_ok=False)
    dependencies = {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in (
            "numpy", "Pillow", "scikit-image", "opencv-python-headless", "scipy",
        )},
    }
    manifest = {
        "schemaVersion": 1,
        "captureMethod": "runpy.run_path; one actual PNG as BGR then end-of-stream; acquisition/GUI patches only",
        "source": write_blob(out, "source/webcam ascii.py", original),
        "inputManifest": write_blob(out, "source/images-manifest.json", images_manifest.read_bytes()),
        "dependencies": dependencies, "defaults": DEFAULTS,
        "fullImages": [], "components": [], "sobelCases": [], "rgbCases": [],
        "gridCases": [],
    }
    inputs = []
    namespace = None
    for fixture in images["fixtures"]:
        path = images_manifest.parent / fixture["path"]
        # load_manifest already checks each PNG hash; decode from its checked bytes.
        from io import BytesIO
        image = Image.open(BytesIO(load_bytes(images_manifest.parent, fixture))).convert("RGB")
        image.load()
        if image.size != (fixture["width"], fixture["height"]):
            raise ValueError(f"{path}: decoded PNG dimensions disagree with manifest")
        text, width, height, namespace = capture_frame(source, original, image)
        manifest["fullImages"].append({
            "id": fixture["id"], "inputSha256": fixture["sha256"],
            "sourceWidth": image.width, "sourceHeight": image.height,
            "width": width, "height": height,
            "text": write_blob(out, f"text/{fixture['id']}.txt", text.encode("utf-8")),
            "gray": write_array(out, namespace["gray_image_np"]),
        })
        manifest["gridCases"].append({
            "id": fixture["id"], "sourceWidth": image.width, "sourceHeight": image.height,
            "settings": {"maxWidthChars": 150, "maxHeightLines": 200},
            "width": width, "height": height,
        })
        gray = np.asarray(image.convert("L"))
        inputs.append((fixture["id"], gray))
        manifest["rgbCases"].append({
            "id": fixture["id"], "inputSha256": fixture["sha256"],
            "rgb": write_array(out, np.asarray(image)),
            "rgba": write_array(out, np.asarray(image.convert("RGBA"))),
            "gray": write_array(out, gray),
        })
    if namespace is None:
        raise ValueError(f"{images_manifest}: empty canonical image list")
    for tiny in images["componentInputs"]:
        inputs.append((tiny["id"], load_array(images_manifest.parent, tiny["input"])))
    for fixture_id, pixels in inputs:
        input_spec = write_array(out, pixels)
        for sigma1, sigma2 in SIGMA_PAIRS:
            # Invoke the original helper, not a transcription of its operations.
            dog = namespace["apply_dog"](pixels, sigma1=sigma1, sigma2=sigma2)
            gaussian = namespace["gaussian"]
            blur1 = gaussian(pixels, sigma=sigma1)
            blur2 = gaussian(pixels, sigma=sigma2)
            manifest["components"].append({
                "id": f"{fixture_id}@{sigma1},{sigma2}", "fixtureId": fixture_id,
                "input": input_spec, "settings": {"sigma1": sigma1, "sigma2": sigma2},
                "outputs": {
                    "blur1": write_array(out, blur1), "blur2": write_array(out, blur2),
                    "dog": write_array(out, dog), **sobel_outputs(namespace, dog, out),
                },
            })
    # Original grayscale steps and their polarities exercise >255 Sobel wrapping;
    # the tiny-axis cases also constrain OpenCV's special border handling.
    sobel_ids = {"step-horizontal", "step-vertical", "step-diagonal-down", "step-diagonal-up", "checker4"}
    for fixture_id, pixels in inputs:
        if fixture_id not in sobel_ids and not fixture_id.startswith("tiny-"):
            continue
        for polarity, values in [("positive", pixels), ("negative", 255 - pixels)]:
            manifest["sobelCases"].append({
                "id": f"{fixture_id}-{polarity}", "input": write_array(out, values),
                "outputs": sobel_outputs(namespace, values, out),
            })
    degrees = set(float(value) for value in np.arange(-720, 720.5, 0.5))
    degrees.update([-135, -90, -45, 45, 90, 135, 0, 180])
    for center in [0, 22.5, 67.5, 112.5, 157.5, 202.5, 360]:
        for translate in [-360, 0, 360]:
            for epsilon in [-1e-7, 0, 1e-7]:
                degrees.add(center + translate + epsilon)
    angle_input = np.array(sorted(degrees), dtype=np.float64)
    codes = np.array([ord(namespace["sobel_angle_to_char"](value)) for value in angle_input], dtype=np.uint8)
    manifest["angles"] = {
        "input": write_array(out, angle_input), "codes": write_array(out, codes),
        "explicitProbes": [{"degrees": value, "char": namespace["sobel_angle_to_char"](value)}
                           for value in [-90, 22.5, 67.5, 180]],
    }
    lookup = np.array([(np.abs(namespace["normalized_ink_values"] - value)).argmin()
                       for value in range(256)], dtype=np.uint8)
    manifest["ramp"] = {
        "inkUsage": namespace["ink_usage"],
        "normalized": write_array(out, namespace["normalized_ink_values"]),
        "indices": write_array(out, lookup),
        "codes": write_array(out, np.array([ord(namespace["characters"][value]) for value in lookup], dtype=np.uint8)),
    }
    all_gray = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
    gray_image = Image.fromarray(all_gray)
    manifest["rgbCases"].append({
        "id": "all-grayscale-values", "rgb": write_array(out, all_gray),
        "rgba": write_array(out, np.asarray(gray_image.convert("RGBA"))),
        "gray": write_array(out, np.asarray(gray_image.convert("L"))),
    })
    if source.read_bytes() != original:
        raise ValueError(f"{source}: renderer changed before completing baseline")
    encoded = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    write_blob(out, "manifest.json", encoded)
    verified = load_manifest(out / "manifest.json")
    # Check semantic constraints in addition to every file hash and binary length.
    if len(verified["components"]) != len(inputs) * len(SIGMA_PAIRS):
        raise ValueError(f"{out}: incomplete component matrix")
    for entry in verified["fullImages"]:
        rows = load_text(out, entry["text"]).split("\n")
        if len(rows) != entry["height"] or any(len(row) != entry["width"] for row in rows):
            raise ValueError(f"{out}: malformed frozen text grid {entry['id']}")
    if not any(np.max(np.hypot(load_array(out, case["outputs"]["gx"]),
                              load_array(out, case["outputs"]["gy"]))) > 255
               for case in verified["sobelCases"]):
        raise ValueError(f"{out}: Sobel fixtures did not exercise magnitude wrapping")
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "qa/fixtures/legacy")
    args = parser.parse_args()
    manifest = freeze(args.out.resolve(), ROOT / "qa/fixtures/images/manifest.json", ROOT / "webcam ascii.py")
    print(f"Verified immutable original baseline: {len(manifest['fullImages'])} images, "
          f"{len(manifest['components'])} Gaussian/DoG/Sobel cases, "
          f"{len(manifest['sobelCases'])} independent Sobel cases; "
          f"source SHA-256 {manifest['source']['sha256']}")


if __name__ == "__main__":
    main()
