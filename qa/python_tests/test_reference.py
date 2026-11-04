"""Regression gates against immutable, pre-extraction Python answers."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

import ascii_core as core
from qa.fixture_io import fixture_by_id, load_array, load_manifest, load_text, sha256

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "qa/fixtures/images"
LEGACY = ROOT / "qa/fixtures/legacy"


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.images = load_manifest(IMAGES / "manifest.json")
        cls.legacy = load_manifest(LEGACY / "manifest.json")

    def array(self, spec):
        return load_array(LEGACY, spec)

    def equal(self, actual, expected, label):
        np.testing.assert_array_equal(actual, expected, err_msg=label)

    def test_full_images(self):
        self.assertEqual(len(self.legacy["fullImages"]), 16)
        for frozen in self.legacy["fullImages"]:
            with self.subTest(fixture=frozen["id"]):
                source = fixture_by_id(self.images, frozen["id"])
                self.assertEqual(source["sha256"], frozen["inputSha256"])
                with Image.open(IMAGES / source["path"]) as image:
                    actual = core.render_image(image)
                self.assertEqual((actual["width"], actual["height"]), (frozen["width"], frozen["height"]))
                self.equal(actual["gray"], self.array(frozen["gray"]), "full-image gray")
                self.assertEqual(actual["text"], load_text(LEGACY, frozen["text"]))
                self.assertFalse(actual["text"].endswith("\n"))

    def test_original_components(self):
        cases = self.legacy["components"]
        self.assertEqual(len(cases), 114)
        self.assertEqual({tuple(c["settings"].values()) for c in cases}, {(1, 2), (.5, 1.5), (2, 3), (1, 1), (2, 1), (.5, 5)})
        for case in cases:
            with self.subTest(component=case["id"]):
                gray = self.array(case["input"])
                result = core.analyze_gray(gray, **case["settings"])
                for key in ("blur1", "blur2", "dog", "gx", "gy", "magnitude", "angles"):
                    self.equal(result[key], self.array(case["outputs"][key]), key)
                dog = core.apply_dog(gray, **case["settings"])
                self.equal(dog, self.array(case["outputs"]["dog"]), "apply_dog")
                for actual, key in zip(core.sobel_edge_detection_with_angle(dog), ("magnitude", "angleRadians", "angles")):
                    self.equal(actual, self.array(case["outputs"][key]), key)

    def test_composition_from_frozen_independent_planes(self):
        ramp = self.array(self.legacy["ramp"]["normalized"])
        codes = np.array([ord(pair[0]) for pair in self.legacy["ramp"]["inkUsage"]], dtype=np.uint8)
        lookup = np.array([np.argmin(np.abs(ramp - value)) for value in range(256)])
        for case in self.legacy["components"]:
            if case["settings"] != {"sigma1": 1, "sigma2": 2}:
                continue
            gray = self.array(case["input"])
            magnitude = self.array(case["outputs"]["magnitude"])
            angles = self.array(case["outputs"]["angles"])
            shade = codes[lookup[gray]]
            glyphs = np.array([ord(core.sobel_angle_to_char(a)) for a in angles.flat], dtype=np.uint8).reshape(gray.shape)
            for threshold in (0, 90, 95):
                with self.subTest(composition=case["id"], threshold=threshold):
                    selected = np.where(lookup[magnitude] < threshold, glyphs, ord(" "))
                    expected = np.where(selected != ord(" "), selected, shade)
                    text = "\n".join("".join(chr(c) for c in row) for row in expected)
                    result = core.analyze_gray(gray, edge_index_threshold=threshold)
                    self.equal(result["shadeCodes"], shade, "shading")
                    self.equal(result["edgeCodes"], selected, "threshold edge selection")
                    self.assertEqual(result["text"], text)

    def test_independent_sobel_polarities_wrap_and_tiny_axes(self):
        crossed_255 = False
        for case in self.legacy["sobelCases"]:
            with self.subTest(sobel=case["id"]):
                outputs = case["outputs"]
                for actual, key in zip(core.sobel_edge_detection_with_angle(self.array(case["input"])), ("magnitude", "angleRadians", "angles")):
                    self.equal(actual, self.array(outputs[key]), key)
                gx, gy = self.array(outputs["gx"]), self.array(outputs["gy"])
                raw = np.sqrt(gx * gx + gy * gy)
                crossed_255 |= bool(np.any(raw > 255))
                self.equal(raw.astype(np.uint8), self.array(outputs["magnitude"]), "uint8 wrap")
        self.assertTrue(crossed_255)

    def test_original_angles(self):
        frozen = self.legacy["angles"]
        actual = np.array([ord(core.sobel_angle_to_char(float(a))) for a in self.array(frozen["input"])], dtype=np.uint8)
        self.equal(actual, self.array(frozen["codes"]), "angle sweep and boundary probes")
        for probe in frozen["explicitProbes"]:
            self.assertEqual(core.sobel_angle_to_char(probe["degrees"]), probe["char"])

    def test_ramp_and_first_index_ties(self):
        frozen = self.legacy["ramp"]
        self.assertEqual([list(pair) for pair in core.ink_usage], frozen["inkUsage"])
        self.equal(core.normalized_ink_values, self.array(frozen["normalized"]), "normalized original ramp")
        normalized = np.asarray(core.normalized_ink_values)
        indices = np.array([np.argmin(np.abs(normalized - value)) for value in range(256)], dtype=np.uint8)
        self.equal(indices, self.array(frozen["indices"]), "exhaustive argmin")
        gray = np.arange(256, dtype=np.uint8).reshape(1, 256)
        result = core.analyze_gray(gray, edge_index_threshold=0)
        self.equal(result["shadeCodes"].reshape(-1), self.array(frozen["codes"]), "production shading")
        self.assertEqual(result["text"], "".join(chr(c) for c in self.array(frozen["codes"])))
        for value in normalized:
            tied = np.flatnonzero(normalized == value)
            self.assertEqual(int(np.argmin(np.abs(normalized - value))), int(tied[0]))

    def test_original_rgb_to_pillow_gray(self):
        for case in self.legacy["rgbCases"]:
            with self.subTest(rgb=case["id"]):
                rgb = self.array(case["rgb"])
                self.equal(np.asarray(Image.fromarray(rgb).convert("L")), self.array(case["gray"]), "Pillow grayscale")
                self.equal(self.array(case["rgba"])[..., :3], rgb, "RGB source channels")

    def test_grid_defaults_branches_and_boundaries(self):
        for case in self.legacy["gridCases"]:
            self.assertEqual(core.grid_dimensions(case["sourceWidth"], case["sourceHeight"]), (case["width"], case["height"]))
        for width, height, expected in [(4, 2, (120, 30)), (2, 4, (80, 80)), (4, 4, (160, 80)), (1, 1, (160, 80))]:
            self.assertEqual(core.grid_dimensions(width, height, 60, 80), expected)
        message = "Image aspect ratio is too extreme for this resolution"
        for width, height in [(0, 1), (1, 0), (-1, 1), (1, -1), (100000, 1), (1, 100000)]:
            with self.subTest(width=width, height=height), self.assertRaisesRegex(ValueError, message):
                core.grid_dimensions(width, height)

    def test_invalid_inputs_and_valid_retry(self):
        valid = np.zeros((3, 4), dtype=np.uint8)
        for bad in [None, [], np.zeros(3, dtype=np.uint8), np.zeros((1, 1, 1), dtype=np.uint8), np.zeros((0, 2), dtype=np.uint8), np.zeros((2, 0), dtype=np.uint8), valid.astype(float), valid.astype(np.int16)]:
            with self.subTest(gray=repr(bad)), self.assertRaisesRegex(ValueError, "Expected a nonempty 2D uint8 image"):
                core.analyze_gray(bad)
        for key in ("sigma1", "sigma2"):
            for value in (0, -1, float("nan"), float("inf"), -float("inf")):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "Sigma must be finite and positive"):
                    core.analyze_gray(valid, **{key: value})
        for value in (-1, 96, 1.5, float("nan"), float("inf"), "90", None, True):
            with self.subTest(threshold=value), self.assertRaisesRegex(ValueError, "Edge index threshold must be an integer from 0 to 95"):
                core.analyze_gray(valid, edge_index_threshold=value)
        for threshold in (0, 90, 95):
            result = core.analyze_gray(valid, sigma1=2, sigma2=1, edge_index_threshold=threshold)
            self.assertEqual(result["text"], "    \n    \n    ")

    def test_viewer_import_is_inert(self):
        with patch.object(cv2, "VideoCapture", side_effect=AssertionError("camera started on import")), patch.object(cv2, "imshow", side_effect=AssertionError("GUI opened on import")):
            runpy.run_path(str(ROOT / "webcam ascii.py"), run_name="import_smoke")

    def run_viewer(self, frame):
        class Capture:
            def __init__(self):
                self.reads = 0
                self.released = False
            def isOpened(self):
                return True
            def read(self):
                self.reads += 1
                return (True, frame.copy()) if self.reads == 1 else (False, None)
            def release(self):
                self.released = True
        capture = Capture()
        output = io.StringIO()
        with patch.object(cv2, "VideoCapture", return_value=capture), patch.object(cv2, "imshow"), patch.object(cv2, "waitKey", return_value=-1), patch.object(cv2, "destroyAllWindows") as destroy, contextlib.redirect_stdout(output):
            try:
                runpy.run_path(str(ROOT / "webcam ascii.py"), run_name="__main__")
            finally:
                self.assertTrue(capture.released, "viewer did not release camera")
                destroy.assert_called_once_with()
        return output.getvalue()

    def test_viewer_finite_actual_png(self):
        frozen = fixture_by_id(self.legacy, "astronaut-portrait", "fullImages")
        source = fixture_by_id(self.images, frozen["id"])
        with Image.open(IMAGES / source["path"]) as image:
            frame = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        self.assertEqual(self.run_viewer(frame), load_text(LEGACY, frozen["text"]) + "\n")

    def test_viewer_processing_failure_cleans_up(self):
        # A real extreme-aspect acquisition reaches and fails the production grid policy.
        with self.assertRaisesRegex(ValueError, "Image aspect ratio is too extreme for this resolution"):
            self.run_viewer(np.zeros((1, 100000, 3), dtype=np.uint8))


class ManifestFailureTests(unittest.TestCase):
    def test_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "array.u8"
            manifest_path = root / "manifest.json"
            good = b"\x00\x01\x02\x03"
            spec = {"path": "array.u8", "bytes": 4, "sha256": sha256(good), "dtype": "u8", "shape": [2, 2]}
            def manifest(specification=spec):
                return {"schemaVersion": 1, "fixtures": [{"id": "one", "input": specification}]}
            def save(value):
                manifest_path.write_text(json.dumps(value), encoding="utf-8")
            blob.write_bytes(good)
            save(manifest())
            loaded = load_manifest(manifest_path)
            self.assertEqual(fixture_by_id(loaded, "one")["id"], "one")
            with self.assertRaises(ValueError):
                fixture_by_id(loaded, "unknown")
            with self.assertRaises(ValueError):
                fixture_by_id(loaded, "one", "missing")
            for data in (b"\x00\x01", b"\x00\x01\x02\xff"):
                blob.write_bytes(data)
                with self.assertRaises(ValueError):
                    load_manifest(manifest_path)
            blob.unlink()
            with self.assertRaises(ValueError):
                load_manifest(manifest_path)
            blob.write_bytes(good)
            for changes in ({"path": "../array.u8"}, {"path": str(blob.resolve())}, {"shape": [3, 2]}, {"shape": [0, 4]}, {"dtype": "float32"}, {"sha256": "bad"}, {"bytes": "4"}):
                with self.subTest(changes=changes):
                    save(manifest({**spec, **changes}))
                    with self.assertRaises(ValueError):
                        load_manifest(manifest_path)
            for malformed in ({}, {"schemaVersion": 2, "fixtures": [{"id": "one"}]}, {"schemaVersion": 1, "fixtures": []}, {"schemaVersion": 1, "fixtures": [{"id": "one"}, {"id": "one"}]}):
                save(malformed)
                with self.assertRaises(ValueError):
                    load_manifest(manifest_path)
            manifest_path.write_text("{broken", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manifest(manifest_path)
            manifest_path.unlink()
            with self.assertRaises(ValueError):
                load_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
