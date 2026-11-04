"""Numerical core of the original live renderer; no camera or GUI side effects."""
from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.filters import gaussian

ink_usage = json.loads(Path(__file__).with_name("ink-usage.json").read_text(encoding="utf-8"))
characters = [char for char, _ in ink_usage]
ink_values = np.array([ink for _, ink in ink_usage])
normalized_ink_values = 255 * (ink_values - ink_values.min()) / (ink_values.max() - ink_values.min())
_RAMP_LOOKUP = np.array([(np.abs(normalized_ink_values - value)).argmin()
                         for value in range(256)], dtype=np.uint8)
_CHARACTER_CODES = np.array([ord(char) for char in characters], dtype=np.uint8)
_ANGLE_MAPPING = {-135: "/", -90: "-", -45: "\\", 45: "/", 90: "-", 135: "\\", 0: " ", 180: "|"}


def _validate_sigmas(sigma1, sigma2):
    for sigma in (sigma1, sigma2):
        if (isinstance(sigma, bool) or not isinstance(sigma, Real)
                or not math.isfinite(sigma) or sigma <= 0):
            raise ValueError("Sigma must be finite and positive")


def _validate_threshold(threshold):
    if isinstance(threshold, bool) or not isinstance(threshold, Integral) or not 0 <= threshold <= 95:
        raise ValueError("Edge index threshold must be an integer from 0 to 95")


def grid_dimensions(width, height, max_width_chars=150, max_height_lines=200) -> tuple[int, int]:
    for dimension in (width, height, max_width_chars, max_height_lines):
        if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension <= 0:
            raise ValueError("Image aspect ratio is too extreme for this resolution")
    if width > height:
        new_width = max_width_chars
        new_height = int(new_width * (height / width))
    else:
        new_height = max_height_lines
        new_width = int(new_height * (width / height))
    new_width *= 2
    if new_width <= 0 or new_height <= 0:
        raise ValueError("Image aspect ratio is too extreme for this resolution")
    return int(new_width), int(new_height)


def _blur_pair(image, sigma1, sigma2):
    _validate_sigmas(sigma1, sigma2)
    if image.ndim == 3:
        return (gaussian(image, sigma=sigma1, channel_axis=-1),
                gaussian(image, sigma=sigma2, channel_axis=-1))
    return gaussian(image, sigma=sigma1), gaussian(image, sigma=sigma2)


def _dog_from_blurs(blur1, blur2):
    return np.uint8(np.clip(blur1 - blur2, 0, 1) * 255)


def apply_dog(image, sigma1=1, sigma2=2):
    return _dog_from_blurs(*_blur_pair(image, sigma1, sigma2))


def _sobel_planes(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.uint8(np.sqrt(gx**2 + gy**2))
    radians = np.arctan2(gy, gx)
    return gx, gy, magnitude, radians, np.degrees(radians)


def sobel_edge_detection_with_angle(image):
    _, _, magnitude, radians, degrees = _sobel_planes(image)
    return magnitude, radians, degrees


def sobel_angle_to_char(degree):
    # Preserve the original negative keys and distance expression, including quirks.
    degree = degree % 360
    closest = min(_ANGLE_MAPPING, key=lambda key: min(abs(degree - key), 360 - abs(degree - key)))
    return _ANGLE_MAPPING[closest]


def analyze_gray(gray, *, sigma1=1, sigma2=2, edge_index_threshold=90) -> dict:
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8 or gray.size == 0:
        raise ValueError("Expected a nonempty 2D uint8 image")
    _validate_threshold(edge_index_threshold)
    blur1, blur2 = _blur_pair(gray, sigma1, sigma2)
    dog = _dog_from_blurs(blur1, blur2)
    gx, gy, magnitude, _, angles = _sobel_planes(dog)
    shade_codes = _CHARACTER_CODES[_RAMP_LOOKUP[gray]]
    degrees = angles % 360
    best = np.full(gray.shape, np.inf)
    edge_codes = np.full(gray.shape, ord(" "), dtype=np.uint8)
    for key, char in _ANGLE_MAPPING.items():
        distance = np.abs(degrees - key)
        distance = np.minimum(distance, 360 - distance)
        nearer = distance < best
        edge_codes[nearer] = ord(char)
        best[nearer] = distance[nearer]
    edge_codes[_RAMP_LOOKUP[magnitude] >= edge_index_threshold] = ord(" ")
    combined = np.where(edge_codes == ord(" "), shade_codes, edge_codes)
    text = "\n".join(row.tobytes().decode("ascii") for row in combined)
    height, width = gray.shape
    return {"width": width, "height": height, "gray": gray,
            "blur1": blur1, "blur2": blur2, "dog": dog, "gx": gx, "gy": gy,
            "magnitude": magnitude, "angles": angles, "shadeCodes": shade_codes,
            "edgeCodes": edge_codes, "text": text}


def render_image(image, *, max_width_chars=150, max_height_lines=200,
                 sigma1=1, sigma2=2, edge_index_threshold=90) -> dict:
    width, height = grid_dimensions(image.width, image.height, max_width_chars, max_height_lines)
    _validate_sigmas(sigma1, sigma2)
    _validate_threshold(edge_index_threshold)
    # The browser acquisition surface is opaque black, including transparent uploads.
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    gray = np.array(resized.convert("L"))
    return analyze_gray(gray, sigma1=sigma1, sigma2=sigma2,
                        edge_index_threshold=edge_index_threshold)
