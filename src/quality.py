"""
Shape-plausibility scoring, image-quality checks, and the explainable confidence
score for Banana Curve Analyzer.

None of this is a machine-learning model. Every value here is a deterministic
function of measurable image/geometry properties (contour area, solidity,
elongation, blur variance, etc.), documented so each number can be explained
to a judge on request.
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np
import cv2

from .config import (
    PLAUSIBILITY_WEIGHT_AREA,
    PLAUSIBILITY_WEIGHT_SOLIDITY,
    PLAUSIBILITY_WEIGHT_ELONGATION,
    PLAUSIBILITY_WEIGHT_EXTENT,
    PLAUSIBILITY_WEIGHT_SMOOTHNESS,
    PLAUSIBILITY_WEIGHT_FRAME_COVERAGE,
    BANANA_ELONGATION_RANGE,
    BANANA_EXTENT_RANGE,
    FRAME_COVERAGE_MAX_PLAUSIBLE,
    BLUR_VARIANCE_MIN,
    CONTRAST_STD_MIN,
    MIN_BANANA_AREA_FRACTION,
    MULTI_BLOB_AREA_RATIO,
    CONFIDENCE_WEIGHT_SEGMENTATION,
    CONFIDENCE_WEIGHT_SKELETON_CONTINUITY,
    CONFIDENCE_WEIGHT_ENDPOINT_ALIGNMENT,
    CONFIDENCE_WEIGHT_SPLINE_FIT,
    CONFIDENCE_WEIGHT_IMAGE_QUALITY,
)


def _range_score(value: float, low: float, high: float, softness: float = 0.5) -> float:
    """1.0 inside [low, high], decaying smoothly outside it. `softness` controls
    how many range-widths of overshoot it takes to decay to ~37% (1/e)."""
    if low <= value <= high:
        return 1.0
    span = max(high - low, 1e-6)
    if value < low:
        overshoot = (low - value) / span
    else:
        overshoot = (value - high) / span
    return float(np.exp(-overshoot / max(softness, 1e-6)))


def _high_side_score(value: float, threshold: float, softness: float = 0.15) -> float:
    """1.0 up to `threshold`, decaying sharply above it (small softness = fast
    cutoff). Used where exceeding the threshold is a strong, not a soft, signal."""
    if value <= threshold:
        return 1.0
    overshoot = (value - threshold) / max(threshold, 1e-6)
    return float(np.exp(-overshoot / max(softness, 1e-6)))


def score_contour_plausibility(
    contour: np.ndarray,
    image_shape: Tuple[int, int],
) -> Dict[str, float]:
    """
    Score how plausible a contour is as "the banana" using only measurable shape
    properties -- no color/position information (that's already been used to
    generate the candidate mask).

    Sub-scores (each 0-1):
        area_score:        contour area relative to the full frame (bigger, up to
                            a point, is more plausible than a tiny fleck).
        solidity_score:     area / convex-hull area. Bananas are fairly solid
                            (no deep concavities), but this alone can't reject a
                            filled rectangle (also solidity ~1.0).
        elongation_score:   long/short side of the minAreaRect, mapped against
                            BANANA_ELONGATION_RANGE. A banana is a long, thin
                            shape, not a square blob.
        extent_score:       area / minAreaRect area, mapped against
                            BANANA_EXTENT_RANGE. A tapered banana fills only
                            part of its own bounding rectangle; a filled
                            rectangle fills essentially all of it (~1.0) and is
                            downweighted here even though its solidity is high.
        smoothness_score:   hull perimeter / contour perimeter. Close to 1.0 for
                            a smooth boundary, lower for a jagged/noisy one.
        frame_coverage_score: penalizes a contour whose own bounding box covers
                            most of the frame (config.FRAME_COVERAGE_MAX_PLAUSIBLE).
                            This is the check that catches "the whole image" or a
                            background+banana blob merged into one region: since
                            cv2.findContours(RETR_EXTERNAL) discards holes, a mask
                            that is "the entire frame with a banana-shaped hole in
                            it" produces an outer contour whose area/elongation/
                            extent/solidity/smoothness can look deceptively close
                            to a clean rectangle -- this is the one signal that
                            still catches it.

    Returns a dict of the sub-scores plus the combined "plausibility_score" (0-1).
    """
    h, w = image_shape[:2]
    frame_area = float(h * w)
    area = float(cv2.contourArea(contour))

    if area < 1e-6 or len(contour) < 5:
        return {
            "area": area, "solidity": 0.0, "elongation": 0.0, "extent": 0.0,
            "smoothness": 0.0, "frame_coverage": 0.0, "area_score": 0.0,
            "solidity_score": 0.0, "elongation_score": 0.0, "extent_score": 0.0,
            "smoothness_score": 0.0, "frame_coverage_score": 0.0,
            "plausibility_score": 0.0,
        }

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 1e-6 else 0.0

    rect = cv2.minAreaRect(contour)
    (rw, rh) = rect[1]
    long_side, short_side = max(rw, rh), max(min(rw, rh), 1e-6)
    elongation = long_side / short_side
    rect_area = max(rw * rh, 1e-6)
    extent = area / rect_area

    perimeter = cv2.arcLength(contour, True)
    hull_perimeter = cv2.arcLength(hull, True)
    smoothness = hull_perimeter / perimeter if perimeter > 1e-6 else 0.0

    _, _, bbox_w, bbox_h = cv2.boundingRect(contour)
    frame_coverage = (bbox_w * bbox_h) / frame_area if frame_area > 0 else 0.0

    area_score = float(np.clip(area / (frame_area * 0.02), 0.0, 1.0))
    solidity_score = float(np.clip(solidity / 0.85, 0.0, 1.0))
    elongation_score = _range_score(elongation, *BANANA_ELONGATION_RANGE)
    extent_score = _range_score(extent, *BANANA_EXTENT_RANGE)
    smoothness_score = float(np.clip(smoothness, 0.0, 1.0))
    frame_coverage_score = _high_side_score(frame_coverage, FRAME_COVERAGE_MAX_PLAUSIBLE)

    plausibility_score = (
        PLAUSIBILITY_WEIGHT_AREA * area_score
        + PLAUSIBILITY_WEIGHT_SOLIDITY * solidity_score
        + PLAUSIBILITY_WEIGHT_ELONGATION * elongation_score
        + PLAUSIBILITY_WEIGHT_EXTENT * extent_score
        + PLAUSIBILITY_WEIGHT_SMOOTHNESS * smoothness_score
        + PLAUSIBILITY_WEIGHT_FRAME_COVERAGE * frame_coverage_score
    )

    return {
        "area": area,
        "solidity": solidity,
        "elongation": elongation,
        "extent": extent,
        "smoothness": smoothness,
        "frame_coverage": frame_coverage,
        "area_score": area_score,
        "solidity_score": solidity_score,
        "elongation_score": elongation_score,
        "extent_score": extent_score,
        "smoothness_score": smoothness_score,
        "frame_coverage_score": frame_coverage_score,
        "plausibility_score": float(np.clip(plausibility_score, 0.0, 1.0)),
    }


def assess_image_quality(image_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Pre-segmentation image-quality signals, computable before any banana is found:
        blur_score:    variance of the Laplacian (higher = sharper).
        contrast_score: standard deviation of grayscale intensity.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast_score = float(gray.std())

    is_blurry = blur_score < BLUR_VARIANCE_MIN
    is_low_contrast = contrast_score < CONTRAST_STD_MIN
    quality_score = float(np.clip(
        0.5 * np.clip(blur_score / (BLUR_VARIANCE_MIN * 2.0), 0.0, 1.0)
        + 0.5 * np.clip(contrast_score / (CONTRAST_STD_MIN * 2.0), 0.0, 1.0),
        0.0, 1.0,
    ))

    return {
        "blur_score": blur_score,
        "contrast_score": contrast_score,
        "is_blurry": is_blurry,
        "is_low_contrast": is_low_contrast,
        "quality_score": quality_score,
    }


def assess_shape_quality_flags(
    mask: np.ndarray,
    contour: np.ndarray,
    image_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """
    Post-segmentation quality signals that need the winning mask/contour:
        is_too_small:              banana area is a tiny fraction of the frame.
        touches_border:             contour touches the image edge (possible
                                    partial/cropped-off banana).
        possible_multiple_bananas: a second connected component exists whose
                                    area is a significant fraction of the
                                    winning one's (could be a second banana, or
                                    a hand/other object that passed color
                                    thresholding).
    """
    h, w = image_shape[:2]
    frame_area = float(h * w)
    area = float(cv2.contourArea(contour))
    area_fraction = area / frame_area if frame_area > 0 else 0.0

    x, y, cw, ch = cv2.boundingRect(contour)
    touches_border = x <= 1 or y <= 1 or (x + cw) >= (w - 1) or (y + ch) >= (h - 1)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    component_areas = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True) if num_labels > 1 else []
    secondary_component_count = 0
    possible_multiple_bananas = False
    if len(component_areas) >= 2 and component_areas[0] > 0:
        secondary_component_count = sum(
            1 for a in component_areas[1:] if a / component_areas[0] >= MULTI_BLOB_AREA_RATIO
        )
        possible_multiple_bananas = secondary_component_count > 0

    return {
        "area_fraction": area_fraction,
        "is_too_small": area_fraction < MIN_BANANA_AREA_FRACTION,
        "touches_border": bool(touches_border),
        "possible_multiple_bananas": possible_multiple_bananas,
        "secondary_component_count": secondary_component_count,
    }


def compute_confidence_score(
    segmentation_score: float,
    skeleton_continuity_score: float,
    endpoint_alignment_score: float,
    spline_fit_quality_score: float,
    image_quality_score: float,
) -> float:
    """
    Weighted combination of pipeline-stage quality signals -> 0-100.

    This is NOT a calibrated probability or an ML model's confidence -- it is a
    documented, deterministic weighted sum of the measurable signals above,
    using CONFIDENCE_WEIGHT_* from config.py. Weights follow the pipeline's own
    stated priority order (segmentation > centerline/endpoints > smoothing/quality).
    """
    weighted = (
        CONFIDENCE_WEIGHT_SEGMENTATION * segmentation_score
        + CONFIDENCE_WEIGHT_SKELETON_CONTINUITY * skeleton_continuity_score
        + CONFIDENCE_WEIGHT_ENDPOINT_ALIGNMENT * endpoint_alignment_score
        + CONFIDENCE_WEIGHT_SPLINE_FIT * spline_fit_quality_score
        + CONFIDENCE_WEIGHT_IMAGE_QUALITY * image_quality_score
    )
    return float(np.clip(weighted, 0.0, 1.0) * 100.0)
