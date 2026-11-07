"""Validate the complete real-browser A/B matrix before publishing an artifact index.

This only inventories and checks evidence. It does not manufacture visual judgments or
assert exact Canvas/Pillow output equality. Run after all Playwright producers finish.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
import sys

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    "black", "midgray", "white", "ramp", "step-horizontal", "step-vertical",
    "step-diagonal-down", "step-diagonal-up", "impulse-center", "impulse-corner",
    "checker4", "noise", "rgb-patches", "astronaut", "astronaut-portrait", "astronaut-landscape",
)
DEFAULT = ("astronaut", "astronaut-portrait", "step-diagonal-down", "step-diagonal-up")
MATRIX = [("comparison", name) for name in CANONICAL] + [("default", name) for name in DEFAULT]
CENTERS = ((0.5, 0.25), (0.5, 0.5), (0.5, 0.75), (0.1, 0.5), (0.9, 0.5))
PROFILES = {
    "comparison": {"maxWidthChars": 60, "maxHeightLines": 80, "sigma1": 1, "sigma2": 2, "edgeIndexThreshold": 90},
    "default": {"maxWidthChars": 50, "maxHeightLines": 65, "sigma1": 1, "sigma2": 2, "edgeIndexThreshold": 90},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def inside(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), f"Invalid relative path: {relative!r}")
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root.resolve()), f"Path escapes evidence root: {relative}")
    require(path.is_file(), f"Not a file: {path}")
    return path


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON {value}: {path}")))


def checked_blob(root, spec):
    require(isinstance(spec, dict), f"Malformed file specification in {root}")
    path = inside(root, spec.get("path"))
    data = path.read_bytes()
    require(type(spec.get("bytes")) is int and spec["bytes"] == len(data), f"Byte count mismatch: {path}")
    require(spec.get("sha256") == digest(data), f"SHA-256 mismatch: {path}")
    return data


def read_png(path):
    with Image.open(path) as image:
        require(image.format == "PNG", f"Not a PNG: {path}")
        image.verify()
    with Image.open(path) as image:
        image.load()
        require(image.width > 0 and image.height > 0, f"Empty PNG: {path}")
        return image.convert("RGB")


def same_pixels(actual, expected, message):
    require(actual.size == expected.size and ImageChops.difference(actual, expected).getbbox() is None, message)


def text_metrics(left, right, columns, rows):
    a, b = left.split("\n"), right.split("\n")
    per_row, per_column = [0] * rows, [0] * columns
    first = None
    for y in range(rows):
        for x in range(columns):
            if a[y][x] != b[y][x]:
                per_row[y] += 1
                per_column[x] += 1
                if first is None:
                    first = {"x": x, "y": y, "leftCode": ord(a[y][x]), "rightCode": ord(b[y][x])}
    mismatches = sum(per_row)
    cells = columns * rows
    return {
        "schemaVersion": 1, "columns": columns, "rows": rows, "cells": cells,
        "mismatchCount": mismatches, "mismatchFraction": mismatches / cells,
        "equalFraction": 1 - mismatches / cells, "firstMismatch": first,
        "perRow": per_row, "perColumn": per_column,
        "leftHistogram": dict(Counter(left.replace("\n", ""))),
        "rightHistogram": dict(Counter(right.replace("\n", ""))),
    }


def check_case(directory, project, profile, fixture_id, fixture, reference, reference_root, reference_hash):
    metadata = load_json(directory / "metadata.json")
    context = f"{profile}/{fixture_id}"
    require(metadata.get("schemaVersion") == 1, f"Unknown metadata schema: {context}")
    for key, expected in (("project", project), ("profile", profile), ("fixtureId", fixture_id),
                          ("settings", PROFILES[profile]), ("columns", reference["width"]), ("rows", reference["height"])):
        require(metadata.get(key) == expected, f"Metadata {key} mismatch: {context}")
    require(isinstance(metadata.get("createdAt"), str) and isinstance(metadata.get("browserVersion"), str), f"Missing run/browser identity: {context}")
    columns, rows = reference["width"], reference["height"]
    required_files = {"source.png", "left.txt", "right.txt", "left.png", "right.png", "pair.png", "heatmap.png", "metrics.json", "answer-key.json"}
    if profile == "default":
        required_files.update(f"crop-{index}-{side}.png" for index in range(1, 6) for side in ("left", "right", "pair"))
    files = metadata.get("files")
    require(isinstance(files, dict) and set(files) == required_files, f"Missing/unknown catalog files: {context}")
    require({path.name for path in directory.iterdir()} == required_files | {"metadata.json"}, f"Missing/unknown case files: {context}")
    pngs = {}
    for name, spec in files.items():
        require(spec.get("path") == name, f"Catalog name/path mismatch: {context}/{name}")
        checked_blob(directory, spec)
        if name.endswith(".png"):
            image = read_png(inside(directory, name))
            require(image.size == (spec.get("width"), spec.get("height")), f"PNG dimensions disagree with catalog: {context}/{name}")
            pngs[name] = image
    source = metadata.get("source", {})
    require(source == {"file": "source.png", "fixturePath": fixture["path"], "sha256": fixture["sha256"],
                       "width": fixture["width"], "height": fixture["height"]}, f"Source metadata mismatch: {context}")
    require(files["source.png"]["sha256"] == fixture["sha256"], f"Copied source differs: {context}")
    require(pngs["source.png"].size == (fixture["width"], fixture["height"]), f"Source dimensions mismatch: {context}")
    require(reference.get("inputSha256") == fixture["sha256"], f"Reference source mismatch: {context}")
    require(reference.get("settings") == PROFILES[profile], f"Reference settings mismatch: {context}")
    answer = load_json(directory / "answer-key.json")
    permutation = digest(f"20261104:{profile}:{fixture_id}".encode())
    python_side = "left" if int(permutation[:2], 16) & 1 == 0 else "right"
    candidate_side = "right" if python_side == "left" else "left"
    require(answer == {"schemaVersion": 1, "profile": profile, "fixtureId": fixture_id, "seed": "20261104",
                       "permutationSha256": permutation, "python": python_side, "candidate": candidate_side}, f"Incorrect physical permutation key: {context}")
    texts = {side: (directory / f"{side}.txt").read_text(encoding="utf-8") for side in ("left", "right")}
    for side, text in texts.items():
        lines = text.split("\n")
        require(len(lines) == rows and all(len(line) == columns and all(32 <= ord(c) <= 126 for c in line) for line in lines), f"Incomplete/wrapped text: {context}/{side}")
    python_bytes = checked_blob(reference_root, reference["text"])
    require(texts[python_side].encode() == python_bytes, f"Reference text differs from independently exported Python: {context}")
    require(metadata.get("reference") == {"id": f"{profile}:{fixture_id}", "textSha256": digest(python_bytes), "manifestSha256": reference_hash}, f"Reference provenance mismatch: {context}")
    commit = metadata.get("commit", {})
    require(type(commit.get("renderId")) is int and commit["renderId"] > 0 and
            type(commit.get("sourceGeneration")) is int and commit["sourceGeneration"] > 0, f"Missing committed render identity: {context}")
    settings_key = json.dumps(PROFILES[profile], separators=(",", ":"))
    for key, value in (("columns", columns), ("rows", rows), ("sourceKind", "image"),
                       ("sourceName", Path(fixture["path"]).name), ("settingsKey", settings_key),
                       ("textSha256", digest(texts[candidate_side].encode()))):
        require(commit.get(key) == value, f"Commit {key} mismatch: {context}")
    metrics = load_json(directory / "metrics.json")
    for key, value in text_metrics(texts["left"], texts["right"], columns, rows).items():
        require(metrics.get(key) == value, f"Symmetric metric {key} disagrees with actual texts: {context}")
    require(isinstance(metrics.get("interpretation"), str), f"Missing metric interpretation: {context}")
    presentation = metadata.get("presentation", {})
    geometry = presentation.get("left", {})
    require(geometry == presentation.get("right") and presentation.get("devicePixelRatio") == 1, f"Unequal cell geometry/device scale: {context}")
    require("Roboto Mono" in geometry.get("fontFamily", "") and geometry.get("fontSize") == "12px" and
            geometry.get("whiteSpace") == "pre" and geometry.get("transform") == "none", f"Wrong font/wrapping/transform: {context}")
    line_height = float(geometry["lineHeight"].removesuffix("px"))
    require(geometry.get("bottomGutterPx") == 2, f"Missing equal descender-ink gutter: {context}")
    require(math.isfinite(line_height) and line_height > 0 and
            abs(geometry["height"] - geometry["bottomGutterPx"] - rows * line_height) < 1 and
            abs(geometry["width"] - columns * line_height / 2) < 1, f"Grid cell geometry mismatch: {context}")
    require(geometry["scrollWidth"] <= math.ceil(geometry["width"]) and
            geometry["scrollHeight"] <= math.ceil(geometry["height"]), f"Clipped grid: {context}")
    require(pngs["left.png"].size == pngs["right.png"].size, f"Unequal screenshot sizes: {context}")
    for side in ("left", "right"):
        size = pngs[f"{side}.png"].size
        require(abs(size[0] - geometry["width"]) <= 2 and abs(size[1] - geometry["height"]) <= 2, f"Screenshot resized/clipped: {context}/{side}")
    pair_geometry = presentation.get("pair", {})
    require(all(abs(actual - pair_geometry.get(key, -100)) <= 2 for actual, key in zip(pngs["pair.png"].size, ("width", "height"))), f"Pair screenshot resized/clipped: {context}")
    require(pngs["pair.png"].width >= pngs["left.png"].width * 2 + 24 and
            pngs["pair.png"].height > pngs["left.png"].height, f"Pair does not contain both complete grids: {context}")
    heatmap = Image.new("RGB", (columns * 4, rows * 4), "#10171d")
    pixels = heatmap.load()
    a, b = texts["left"].split("\n"), texts["right"].split("\n")
    for y in range(rows):
        for x in range(columns):
            if a[y][x] != b[y][x]:
                for dy in range(4):
                    for dx in range(4):
                        pixels[x * 4 + dx, y * 4 + dy] = (255, 134, 102)
    same_pixels(pngs["heatmap.png"], heatmap, f"Heatmap disagrees with text: {context}")
    crops = metadata.get("crops")
    require(isinstance(crops, list) and len(crops) == (5 if profile == "default" else 0), f"Incomplete crop matrix: {context}")
    for index, crop in enumerate(crops, 1):
        cx, cy = CENTERS[index - 1]
        full_width, full_height = pngs["left.png"].size
        width, height = min(512, full_width), min(512, full_height)
        # JavaScript Math.round is floor(x + 0.5), not Python's ties-to-even round.
        x = max(0, min(full_width - width, math.floor(cx * full_width - width / 2 + 0.5)))
        y = max(0, min(full_height - height, math.floor(cy * full_height - height / 2 + 0.5)))
        paths = {side: f"crop-{index}-{side}.png" for side in ("left", "right", "pair")}
        require(crop == {"index": index, "center": [cx, cy], "box": {"x": x, "y": y, "width": width, "height": height}, "files": paths}, f"Crop is not the prescribed fixed region: {context}/{index}")
        for side in ("left", "right"):
            expected = pngs[f"{side}.png"].crop((x, y, x + width, y + height))
            same_pixels(pngs[paths[side]], expected, f"Crop pixels resampled/mismatched: {context}/{index}/{side}")
        pair = Image.new("RGB", (width * 2 + 16, height), "#10171d")
        pair.paste(pngs[paths["left"]], (0, 0))
        pair.paste(pngs[paths["right"]], (width + 16, 0))
        same_pixels(pngs[paths["pair"]], pair, f"Crop pair pixels mismatched: {context}/{index}")
    prefix = f"{project}/{profile}/{fixture_id}/"
    return {
        "id": f"{project}/{profile}/{fixture_id}", "profile": profile, "fixtureId": fixture_id,
        "settings": PROFILES[profile], "columns": columns, "rows": rows,
        "source": prefix + "source.png", "left": prefix + "left.png", "right": prefix + "right.png",
        "pair": prefix + "pair.png", "leftText": prefix + "left.txt", "rightText": prefix + "right.txt",
        "metrics": prefix + "metrics.json", "heatmap": prefix + "heatmap.png",
        "metadata": prefix + "metadata.json", "answerKey": prefix + "answer-key.json",
        "mismatchCount": metrics["mismatchCount"], "mismatchFraction": metrics["mismatchFraction"],
        "crops": [{"index": crop["index"], "center": crop["center"], "box": crop["box"],
                   **{side: prefix + path for side, path in crop["files"].items()}} for crop in crops],
    }


def aggregate(root, reference_root):
    root = root.resolve(strict=True)
    reference_root = reference_root.resolve(strict=True)
    project = "ab-chromium"
    projects = {path.name for path in root.iterdir() if path.is_dir()}
    require(projects == {project}, f"Missing/unknown A/B projects: {sorted(projects)}")
    project_root = root / project
    require({path.name for path in project_root.iterdir()} == set(PROFILES), "Missing/unknown profile directories")
    for profile, expected in (("comparison", CANONICAL), ("default", DEFAULT)):
        entries = {path.name for path in (project_root / profile).iterdir()}
        require(entries == set(expected) and all((project_root / profile / name).is_dir() for name in entries), f"Missing/unknown cases in {profile}: {sorted(entries)}")
    image_root = ROOT / "qa/fixtures/images"
    image_manifest = load_json(image_root / "manifest.json")
    fixture_list = image_manifest.get("fixtures", [])
    require(len(fixture_list) == len(CANONICAL) and {item["id"] for item in fixture_list} == set(CANONICAL), "Incomplete/unknown canonical image manifest")
    fixtures = {entry["id"]: entry for entry in fixture_list}
    for fixture in fixture_list:
        checked_blob(image_root, fixture)
    reference_manifest_path = reference_root / "manifest.json"
    reference_manifest = load_json(reference_manifest_path)
    require(reference_manifest.get("schemaVersion") == 1 and reference_manifest.get("profiles") == PROFILES, "Unknown reference schema/profiles")
    full_images = reference_manifest.get("fullImages", [])
    require(len(full_images) == 32 and {entry["id"] for entry in full_images} == {f"{profile}:{name}" for profile in PROFILES for name in CANONICAL}, "Incomplete/unknown full-image reference matrix")
    references = {entry["id"]: entry for entry in full_images}
    reference_hash = digest(reference_manifest_path.read_bytes())
    cases = [check_case(project_root / profile / name, project, profile, name, fixtures[name], references[f"{profile}:{name}"], reference_root, reference_hash)
             for profile, name in MATRIX]
    manifest = {
        "schemaVersion": 1, "project": project, "caseCount": len(cases),
        "referenceManifestSha256": reference_hash,
        "interpretation": "Complete evidence inventory, not a visual-review verdict. Raw Canvas/Pillow character mismatch is symmetric and descriptive only. Judge images are source and neutral pair/crop images, never answer keys or provenance metadata.",
        "cases": cases,
    }
    cards = []
    for case in cases:
        e = lambda value: escape(str(value), quote=True)
        crops = " ".join(f'<a href="{e(crop["pair"])}">Region {crop["index"]}</a>' for crop in case["crops"])
        cards.append(f'''<article id="{e(case['profile'])}-{e(case['fixtureId'])}">
<h2>{e(case['profile'])} / {e(case['fixtureId'])}</h2>
<p>{case['columns']} columns × {case['rows']} rows · raw character mismatch {case['mismatchCount']} ({case['mismatchFraction']:.4%})</p>
<p><a href="{e(case['source'])}">Source</a> · <a href="{e(case['left'])}">Left full resolution</a> · <a href="{e(case['right'])}">Right full resolution</a> · <a href="{e(case['pair'])}">Pair full resolution</a> · <a href="{e(case['heatmap'])}">Mismatch heatmap</a></p>
<p><a href="{e(case['leftText'])}">Left text</a> · <a href="{e(case['rightText'])}">Right text</a> · <a href="{e(case['metrics'])}">Metrics</a> · <a href="{e(case['metadata'])}">Commit/provenance metadata (unblinded)</a></p>
<p>{crops}</p><details><summary>Answer key — open only after blinded judgments</summary><a href="{e(case['answerKey'])}">Decode placement</a></details>
<div class="scroll"><img src="{e(case['pair'])}" alt="Neutral Left and Right full grids for {e(case['fixtureId'])}" loading="lazy"></div>
</article>''')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ASCII A/B evidence</title>
<style>body{margin:24px;background:#10171d;color:#e6edf1;font:16px system-ui}a{color:#a7e8ce}article{border-top:1px solid #465966;padding:24px 0}.scroll{overflow:auto;max-height:1000px;margin-top:20px}img{display:block;max-width:none;height:auto}p{line-height:1.6}details{margin:16px 0}</style>
<h1>ASCII A/B evidence — 20 complete cases</h1><p>Neutral Left/Right placement is deterministic and physically permuted. Images retain full-resolution 12px local-font grids; scroll rather than scale. Each default case has all five prescribed 512px regions, clipped only to grid bounds.</p>
<p>Raw Python/browser mismatch is descriptive, not a quality score or parity claim. Browser acquisition uses independently verified JavaScript Lanczos sampling. This index validates evidence completeness only; independent/direct visual review remains a separate gate. Keep provenance and answer keys out of blinded judge inputs.</p><p><a href="manifest.json">Relative-path evidence manifest</a></p>
''' + "\n".join(cards) + "</html>\n"
    # Producers never write shared outputs. Publish only after every case passes.
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "index.html").write_text(html, encoding="utf-8")
    print(f"Validated {len(cases)} A/B cases and 20 fixed crop triplets: {root / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "qa/artifacts/ab")
    parser.add_argument("--reference", type=Path, default=ROOT / "qa/artifacts/reference")
    args = parser.parse_args()
    try:
        aggregate(args.root, args.reference)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f"A/B evidence validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
