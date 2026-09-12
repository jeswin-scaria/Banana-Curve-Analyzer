"""
Image transforms for stability testing: resize, rotation, brightness/contrast,
noise, and margin crop. Used by tests/test_stability.py and scripts/benchmark.py
to probe how much analyze_banana's output varies under transformations that do
not change the banana's real-world curvature.
"""
from typing import Tuple
import numpy as np
import cv2


def resize_image(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def rotate_image(img: np.ndarray, angle_deg: float, border_value: Tuple[int, int, int] = (240, 240, 240)) -> np.ndarray:
    """Rotate about the image center, expanding the canvas so nothing clips."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    m[0, 2] += (new_w / 2.0) - cx
    m[1, 2] += (new_h / 2.0) - cy

    return cv2.warpAffine(img, m, (new_w, new_h), borderValue=border_value)


def adjust_brightness(img: np.ndarray, delta: int) -> np.ndarray:
    return np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    mean = img.astype(np.float64).mean()
    out = (img.astype(np.float64) - mean) * factor + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def add_gaussian_noise(img: np.ndarray, sigma: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def crop_with_margin(img: np.ndarray, margin_fraction: float) -> np.ndarray:
    """Crop `margin_fraction` off each edge. Caller is responsible for ensuring
    the subject stays fully inside the cropped region."""
    h, w = img.shape[:2]
    dy, dx = int(h * margin_fraction), int(w * margin_fraction)
    return img[dy:h - dy, dx:w - dx].copy()
