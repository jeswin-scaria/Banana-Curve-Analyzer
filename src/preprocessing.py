"""
Image preprocessing and segmentation module for Banana Curve Analyzer.
Provides robust methods to isolate the single banana from various backgrounds
including wooden tables, cutting boards, countertops, and white surfaces.
"""

from typing import Optional, Tuple, Union
import io
import numpy as np
import cv2
from PIL import Image

from .config import (
    DEFAULT_HSV_LOWER,
    DEFAULT_HSV_UPPER,
    MORPH_KERNEL_SIZE,
    MIN_CONTOUR_AREA,
)


def load_image(source: Union[str, bytes, io.BytesIO, np.ndarray]) -> np.ndarray:
    """
    Load an image from a file path, byte stream, or existing array into an RGB numpy array.

    Args:
        source: File path, file-like byte buffer, or numpy array.

    Returns:
        np.ndarray: Image in RGB format (uint8, shape [H, W, 3]).
    """
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            return cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        elif source.shape[2] == 4:
            return cv2.cvtColor(source, cv2.COLOR_RGBA2RGB)
        return source

    if isinstance(source, (bytes, bytearray)):
        nparr = np.frombuffer(source, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image from provided bytes.")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if hasattr(source, "read"):
        source.seek(0)
        file_bytes = np.asarray(bytearray(source.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image from uploaded file.")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if isinstance(source, str):
        img_bgr = cv2.imread(source, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image at path: {source}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    raise TypeError(f"Unsupported image source type: {type(source)}")


def compute_smart_banana_saliency(image_rgb: np.ndarray) -> np.ndarray:
    """
    Compute a normalized banana saliency response combining CIE LAB and HSV color spaces.
    Discriminates vibrant yellow/green bananas from wooden tables, shadows, and neutral backgrounds.

    - In CIE LAB:
        b* channel measures blue (-128) to yellow (+127). Bananas have high b* (> 150 in uint8).
        a* channel measures green (-128) to red (+127). Bananas have negative or near-zero a* (<= 135 in uint8).
        Wood surfaces have lower b* (130-150) and higher a* (> 135, reddish/brown).
    - In HSV:
        Yellow bananas have Hue in [18, 38] and high Saturation.
        Green bananas have Hue in [35, 85].
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    a_chan = lab[:, :, 1].astype(np.float32)
    b_chan = lab[:, :, 2].astype(np.float32)

    # 1. Ripe Yellow Component: High b* in LAB, low a* (not red/brown), and yellow hue
    yellow_b_strength = np.maximum(0.0, b_chan - 145.0)
    yellow_hue_gate = (h >= 15) & (h <= 40)
    yellow_a_gate = a_chan <= 136.0  # rejects reddish-brown wood grain
    yellow_score = yellow_b_strength * s * v * yellow_hue_gate * yellow_a_gate

    # 2. Green / Unripe Component: Green hue in HSV and low a* (green side of LAB)
    green_hue_gate = (h > 35) & (h <= 85)
    green_a_strength = np.maximum(0.0, 132.0 - a_chan)
    green_score = green_a_strength * s * v * green_hue_gate

    # Combined score
    total_score = yellow_score + (1.2 * green_score)
    max_val = np.max(total_score)

    if max_val < 1e-4:
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    normalized = np.clip((total_score / max_val) * 255.0, 0, 255).astype(np.uint8)
    return normalized


def segment_banana(
    image_rgb: np.ndarray,
    method: str = "auto",
    hsv_lower: Optional[Tuple[int, int, int]] = None,
    hsv_upper: Optional[Tuple[int, int, int]] = None,
    min_area: int = MIN_CONTOUR_AREA,
    morph_kernel_size: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Segment the single banana from the image and return a cleaned binary mask
    and the contour of the banana.

    Args:
        image_rgb: RGB image (uint8, [H, W, 3]).
        method: 'auto' (smart multi-space), 'wood' (strict table filter),
                'hsv', 'otsu', or 'saturation'.
        hsv_lower: Optional custom lower HSV bound.
        hsv_upper: Optional custom upper HSV bound.
        min_area: Minimum contour area in pixels.
        morph_kernel_size: Optional custom morphological kernel size.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]:
            - binary_mask: uint8 image with 255 for banana and 0 for background.
            - largest_contour: Largest contour array (or None if no banana found).
    """
    h, w = image_rgb.shape[:2]
    k_size = morph_kernel_size or max(5, int(min(h, w) / 90) | 1)

    raw_mask: Optional[np.ndarray] = None

    if method in ("auto", "wood"):
        # Use Smart Multi-Space (LAB + HSV) Saliency
        saliency = compute_smart_banana_saliency(image_rgb)
        if cv2.countNonZero(saliency) > min_area:
            # Otsu thresholding on the banana saliency map
            _, raw_mask = cv2.threshold(saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # In 'wood' mode or if table bleeding occurs, apply strict saturation gating
            if method == "wood":
                hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
                lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
                strict_gate = (hsv[:, :, 1] >= 80) & (lab[:, :, 2] >= 155) & (lab[:, :, 1] <= 135)
                raw_mask = raw_mask & (strict_gate.astype(np.uint8) * 255)

    if raw_mask is None or cv2.countNonZero(raw_mask) < min_area:
        # Fallback / explicit HSV segmentation
        lower = np.array(hsv_lower if hsv_lower is not None else DEFAULT_HSV_LOWER, dtype=np.uint8)
        upper = np.array(hsv_upper if hsv_upper is not None else DEFAULT_HSV_UPPER, dtype=np.uint8)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        raw_mask = cv2.inRange(hsv, lower, upper)

    if method == "saturation" or (method == "auto" and (raw_mask is None or cv2.countNonZero(raw_mask) < min_area)):
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1]
        _, raw_mask = cv2.threshold(s_channel, 60, 255, cv2.THRESH_BINARY)

    if method == "otsu" or (method == "auto" and (raw_mask is None or cv2.countNonZero(raw_mask) < min_area)):
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        corners = [blurred[0, 0], blurred[0, -1], blurred[-1, 0], blurred[-1, -1]]
        avg_corner = float(np.mean(corners))
        if avg_corner > 127:
            _, raw_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, raw_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if raw_mask is None or cv2.countNonZero(raw_mask) == 0:
        return np.zeros((h, w), dtype=np.uint8), None

    # Morphological cleanup:
    # 1. Opening: cuts thin bridges connecting banana to wood grain or background reflections
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, open_kernel, iterations=2)

    # 2. Closing: fills small brown spots and internal peel blemishes
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size * 2, k_size * 2))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # Find external contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), dtype=np.uint8), None

    # Pick the largest contour by area (representing the single banana)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < min_area:
        return np.zeros((h, w), dtype=np.uint8), None

    # Create a solid, clean mask containing strictly the largest contour
    final_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return final_mask, largest_contour
