"""Read immutable fixture manifests and verify every referenced byte before use."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

DTYPES = {"u8": np.dtype("u1"), "f64le": np.dtype("<f8")}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file(root: Path, spec: dict) -> Path:
    if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
        raise ValueError(f"{root}: malformed file specification")
    relative = Path(spec["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"{root}: fixture path escapes manifest directory: {relative}")
    return path


def load_bytes(root: Path | str, spec: dict) -> bytes:
    path = _file(Path(root), spec)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{path}: unable to read fixture: {exc}") from exc
    length = spec.get("bytes")
    digest = spec.get("sha256")
    if type(length) is not int or length < 0 or len(data) != length:
        raise ValueError(f"{path}: fixture byte length mismatch (expected {length}, got {len(data)})")
    if not isinstance(digest, str) or len(digest) != 64 or sha256(data) != digest:
        raise ValueError(f"{path}: fixture SHA-256 mismatch")
    return data


def load_array(root: Path | str, spec: dict) -> np.ndarray:
    data = load_bytes(root, spec)
    dtype = DTYPES.get(spec.get("dtype"))
    shape = spec.get("shape")
    if dtype is None or not isinstance(shape, list) or not shape:
        raise ValueError(f"{spec.get('path')}: malformed array dtype/shape")
    size = 1
    for dimension in shape:
        if type(dimension) is not int or dimension <= 0:
            raise ValueError(f"{spec['path']}: invalid array dimension {dimension}")
        size *= dimension
    if size * dtype.itemsize != len(data):
        raise ValueError(f"{spec['path']}: array shape disagrees with byte length")
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def load_text(root: Path | str, spec: dict) -> str:
    try:
        return load_bytes(root, spec).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{spec.get('path')}: invalid UTF-8 text") from exc


def load_manifest(path: Path | str) -> dict:
    """Return JSON only after recursively checking all file hashes/array lengths.

    Paths in every file/array specification are relative to this manifest's
    directory. Missing, truncated, modified, malformed and escaping files fail.
    No data generation, network access or fixture replacement happens here.
    """
    path = Path(path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: unable to read manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1:
        raise ValueError(f"{path}: unsupported or malformed fixture manifest")
    section_names = (
        "fixtures", "componentInputs", "fullImages", "components",
        "sobelCases", "rgbCases", "gridCases",
    )
    if "fixtures" not in manifest and "fullImages" not in manifest:
        raise ValueError(f"{path}: missing canonical or legacy fixture section")
    for name in section_names:
        if name not in manifest:
            continue
        entries = manifest[name]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{path}: missing or empty fixture section {name}")
        ids = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
                raise ValueError(f"{path}: malformed fixture ID in {name}")
            ids.append(entry["id"])
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: duplicate fixture ID in {name}")

    def visit(value):
        if isinstance(value, dict):
            if any(key in value for key in ("path", "dtype", "shape", "sha256", "bytes")):
                if "dtype" in value or "shape" in value:
                    load_array(path.parent, value)
                else:
                    load_bytes(path.parent, value)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(manifest)
    return manifest


def fixture_by_id(manifest: dict, fixture_id: str, section: str = "fixtures") -> dict:
    entries = manifest.get(section)
    if not isinstance(entries, list):
        raise ValueError(f"Missing fixture section: {section}")
    found = [entry for entry in entries if entry.get("id") == fixture_id]
    if len(found) != 1:
        raise ValueError(f"Unknown or duplicate fixture ID in {section}: {fixture_id}")
    return found[0]


def write_blob(root: Path, relative: str, data: bytes) -> dict:
    """Generation-only helper; identical content is never rewritten."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"{path}: refusing to replace different fixture bytes")
    else:
        path.write_bytes(data)
    return {"path": relative, "sha256": sha256(data), "bytes": len(data)}


def write_array(root: Path, array: np.ndarray) -> dict:
    array = np.asarray(array)
    if array.dtype == np.uint8:
        dtype = "u8"
        stored = np.ascontiguousarray(array, dtype=np.uint8)
    elif np.issubdtype(array.dtype, np.floating):
        dtype = "f64le"
        stored = np.ascontiguousarray(array, dtype="<f8")
    else:
        raise ValueError(f"Unsupported frozen array dtype: {array.dtype}")
    data = stored.tobytes(order="C")
    spec = write_blob(root, f"arrays/{sha256(data)}.{dtype}", data)
    return {**spec, "dtype": dtype, "shape": list(stored.shape)}
