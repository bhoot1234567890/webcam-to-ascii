"""Export verified, real Python stage answers; never modify frozen fixtures."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
import sys
import unittest

import cv2
import numpy as np
from PIL import Image

from qa.fixture_io import (
    fixture_by_id, load_array, load_bytes, load_manifest, sha256,
    write_array, write_blob,
)

ROOT = Path(__file__).resolve().parents[1]
PLANES = (
    "gray", "blur1", "blur2", "dog", "gx", "gy", "magnitude", "angles",
    "shadeCodes", "edgeCodes",
)


def settings(width=150, height=200, sigma1=1, sigma2=2, threshold=90):
    return dict(maxWidthChars=width, maxHeightLines=height, sigma1=sigma1,
                sigma2=sigma2, edgeIndexThreshold=threshold)


PROFILES = {"comparison": settings(60, 80), "default": settings()}


def _regressions():
    suite = unittest.defaultTestLoader.loadTestsFromName("qa.python_tests.test_reference")
    if suite.countTestCases() == 0:
        raise RuntimeError("qa.python_tests.test_reference: no regression tests found")
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Python regressions failed; reference export refused")


def export_reference(out: Path, fixture_id: str | None = None) -> dict:
    """Run regressions before writing any derived answer and validate the result."""
    out = out.resolve()
    artifacts = (ROOT / "qa/artifacts").resolve()
    if not out.is_relative_to(artifacts) or out == artifacts:
        raise ValueError("Reference output must be a directory under qa/artifacts")
    images_path = ROOT / "qa/fixtures/images/manifest.json"
    legacy_path = ROOT / "qa/fixtures/legacy/manifest.json"
    images = load_manifest(images_path)
    legacy = load_manifest(legacy_path)
    if fixture_id is not None:
        fixture_by_id(images, fixture_id)
    _regressions()

    # Import only after the frozen-regression gate has succeeded.
    import ascii_core as core

    image_root, legacy_root = images_path.parent, legacy_path.parent
    selected = [entry for entry in images["fixtures"]
                if fixture_id is None or entry["id"] == fixture_id]

    def copy_frozen(value):
        if isinstance(value, dict):
            if "dtype" in value:
                return write_array(out, load_array(legacy_root, value))
            if "path" in value:
                data = load_bytes(legacy_root, value)
                return write_blob(out, f"blobs/{sha256(data)}", data)
            return {key: copy_frozen(item) for key, item in value.items()}
        if isinstance(value, list):
            return [copy_frozen(item) for item in value]
        return value

    def pack(result):
        data = result["text"].encode("utf-8")
        return dict(width=int(result["width"]), height=int(result["height"]),
                    outputs={key: write_array(out, result[key]) for key in PLANES},
                    text=write_blob(out, f"text/{sha256(data)}.txt", data))

    manifest = dict(
        schemaVersion=1,
        canonicalIds=[entry["id"] for entry in images["fixtures"]],
        profiles=PROFILES,
        dependencies={"python": platform.python_version(), **{
            name: version(name) for name in
            ("numpy", "Pillow", "scikit-image", "opencv-python-headless", "scipy")}},
        machine=dict(system=platform.system(), release=platform.release(),
                     machine=platform.machine(), processor=platform.processor()),
        provenance=dict(
            method="ascii_core after frozen original Python unittest regressions",
            goldenSourceSha256=legacy["source"]["sha256"],
            goldenManifestSha256=sha256(legacy_path.read_bytes()),
            imageManifestSha256=sha256(images_path.read_bytes()),
            coreSha256=sha256((ROOT / "ascii_core.py").read_bytes()),
        ),
        components=[], sobelCases=[], fullImages=[], gridCases=[], resizeCases=[],
        angles=copy_frozen(legacy["angles"]), ramp=copy_frozen(legacy["ramp"]),
        rgbCases=copy_frozen(legacy["rgbCases"]),
    )
    if fixture_id is not None:
        manifest["selectedFixtureId"] = fixture_id
    for case in legacy["components"]:
        if fixture_id is not None and case["fixtureId"] not in {
                fixture_id, *(entry["id"] for entry in images["componentInputs"])}:
            continue
        gray = load_array(legacy_root, case["input"])
        normalized = settings(sigma1=case["settings"]["sigma1"],
                              sigma2=case["settings"]["sigma2"])
        result = core.analyze_gray(gray, sigma1=normalized["sigma1"],
                                   sigma2=normalized["sigma2"], edge_index_threshold=90)
        manifest["components"].append(dict(
            id=case["id"], fixtureId=case["fixtureId"], settings=normalized,
            input=write_array(out, gray), **pack(result)))

    for case in legacy["sobelCases"]:
        dog = load_array(legacy_root, case["input"])
        magnitude, radians, degrees = core.sobel_edge_detection_with_angle(dog)
        outputs = dict(gx=cv2.Sobel(dog, cv2.CV_64F, 1, 0, ksize=3),
                       gy=cv2.Sobel(dog, cv2.CV_64F, 0, 1, ksize=3),
                       magnitude=magnitude, angles=degrees, radians=radians)
        manifest["sobelCases"].append(dict(
            id=case["id"], input=write_array(out, dog),
            outputs={key: write_array(out, plane) for key, plane in outputs.items()}))

    def pack_resize(identity, image, width, height, **metadata):
        # RGB 8-bit Lanczos is the production sampling contract. Alpha is
        # composited before resizing, never filtered as a fourth colour channel.
        rgba = image.convert("RGBA")
        black = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        rgb = Image.alpha_composite(black, rgba).convert("RGB")
        resized = rgb.resize((width, height), Image.Resampling.LANCZOS).convert("RGBA")
        manifest["resizeCases"].append(dict(
            id=identity, srcWidth=rgb.width, srcHeight=rgb.height,
            width=width, height=height, **metadata,
            input=write_array(out, np.asarray(rgb.convert("RGBA"), dtype=np.uint8)),
            outputs=dict(rgba=write_array(out, np.asarray(resized, dtype=np.uint8)))))

    for fixture in selected:
        # Hash-checked bytes, not a second unchecked path-based image open.
        image_bytes = load_bytes(image_root, fixture)
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            rgba = source.convert("RGBA")
            image = Image.alpha_composite(
                Image.new("RGBA", rgba.size, (0, 0, 0, 255)), rgba).convert("RGB")
        for profile, config in PROFILES.items():
            result = core.render_image(
                image, max_width_chars=config["maxWidthChars"],
                max_height_lines=config["maxHeightLines"], sigma1=config["sigma1"],
                sigma2=config["sigma2"], edge_index_threshold=config["edgeIndexThreshold"])
            identity = f"{profile}:{fixture['id']}"
            manifest["fullImages"].append(dict(
                id=identity, fixtureId=fixture["id"], profile=profile,
                sourceWidth=image.width, sourceHeight=image.height,
                inputSha256=fixture["sha256"], settings=config, **pack(result)))
            manifest["gridCases"].append(dict(
                id=identity, sourceWidth=image.width, sourceHeight=image.height,
                settings=config, width=result["width"], height=result["height"]))
            pack_resize(identity, image, result["width"], result["height"],
                        fixtureId=fixture["id"], profile=profile)

    # Non-gray RGB channels make pass ordering and intermediate clipping visible.
    # Exercise single-axis passes, identity, enlargement and reduction separately
    # from the aspect-preserving canonical grids.
    for label, src_width, src_height, targets in (
        ("tiny-1x1", 1, 1, ((1, 1), (7, 9), (1, 13), (11, 1))),
        ("tiny-1x17", 1, 17, ((1, 17), (1, 5), (1, 31), (9, 7))),
        ("tiny-17x1", 17, 1, ((17, 1), (5, 1), (31, 1), (7, 9))),
        ("awkward", 13, 11, ((5, 3), (29, 23), (7, 11), (13, 5),
                                 (23, 11), (13, 19), (13, 11))),
    ):
        y, x = np.indices((src_height, src_width), dtype=np.int64)
        rgb = np.stack(((x * 73 + y * 29 + 17) % 256,
                        (x * 11 + y * 151 + 103) % 256,
                        (x * 197 + y * 47 + 239) % 256), axis=-1).astype(np.uint8)
        image = Image.fromarray(rgb)
        for width, height in targets:
            pack_resize(f"boundary:{label}:{width}x{height}", image, width, height)

    for identity, width, height in (
        ("square", 40, 40), ("tiny-square", 1, 1),
        ("tiny-portrait", 1, 17), ("tiny-landscape", 17, 1),
        ("zero-width", 0, 1), ("negative-height", 1, -1),
        ("extreme-portrait", 1, 1000000), ("extreme-landscape", 1000000, 1),
    ):
        case = dict(id=f"boundary:{identity}", sourceWidth=width, sourceHeight=height,
                    settings=settings())
        try:
            case["width"], case["height"] = core.grid_dimensions(width, height)
        except ValueError as exc:
            case["error"] = str(exc)
        manifest["gridCases"].append(case)

    out.mkdir(parents=True, exist_ok=True)
    destination = out / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Fail closed: no manifest is returned to consumers until all blobs validate.
    return load_manifest(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "qa/artifacts/reference")
    parser.add_argument("--fixture", help="Export one known canonical image ID")
    args = parser.parse_args(argv)
    try:
        manifest = export_reference(args.out, args.fixture)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Reference export failed: {exc}\n")
    print(f"Exported {len(manifest['fullImages'])} full-image references to {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
