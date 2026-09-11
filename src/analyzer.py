"""
Curvature calculation and analysis module for Banana Curve Analyzer.
Implements the core project-defined formula:
Curve Score = ((Centerline Path Length - Straight-Line Distance) / Straight-Line Distance) * 100
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.interpolate import splprep, splev

from .config import (
    STRAIGHT_THRESHOLD,
    CURVED_THRESHOLD,
    CATEGORY_STRAIGHT,
    CATEGORY_CURVED,
    CATEGORY_HIGHLY_CURVED,
    SPLINE_SMOOTHING,
    SPLINE_NUM_POINTS,
)
from .preprocessing import segment_banana
from .skeleton import extract_skeleton, trace_longest_centerline


@dataclass
class CurvatureAnalysisResult:
    """Stores all metrics, coordinates, and classification results for a banana image."""
    success: bool
    message: str = ""
    is_curved: bool = False
    curve_score: float = 0.0
    category: str = CATEGORY_STRAIGHT
    path_length: float = 0.0
    chord_distance: float = 0.0
    max_deflection: float = 0.0
    deflection_ratio: float = 0.0
    endpoint_1: Optional[Tuple[float, float]] = None
    endpoint_2: Optional[Tuple[float, float]] = None
    apex_point: Optional[Tuple[float, float]] = None
    apex_chord_foot: Optional[Tuple[float, float]] = None
    centerline_points: Optional[np.ndarray] = None
    contour: Optional[np.ndarray] = None
    binary_mask: Optional[np.ndarray] = None
    local_curvatures: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert scalar results and metrics to a dictionary for JSON reporting."""
        return {
            "success": self.success,
            "message": self.message,
            "is_curved": self.is_curved,
            "curve_score": round(self.curve_score, 2),
            "category": self.category,
            "path_length_px": round(self.path_length, 2),
            "chord_distance_px": round(self.chord_distance, 2),
            "max_deflection_px": round(self.max_deflection, 2),
            "deflection_ratio": round(self.deflection_ratio, 4),
            "endpoint_1": [round(c, 2) for c in self.endpoint_1] if self.endpoint_1 else None,
            "endpoint_2": [round(c, 2) for c in self.endpoint_2] if self.endpoint_2 else None,
            "apex_point": [round(c, 2) for c in self.apex_point] if self.apex_point else None,
        }


def fit_centerline_spline(
    ordered_points: List[Tuple[float, float]],
    smoothing: float = SPLINE_SMOOTHING,
    num_points: int = SPLINE_NUM_POINTS,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Fit a smooth parametric B-spline to the ordered centerline points to eliminate
    discrete pixel-grid step artifacts and calculate smooth continuous derivatives.

    Args:
        ordered_points: List of (x, y) coordinates along the skeleton centerline.
        smoothing: Spline smoothing parameter (s in splprep).
        num_points: Number of points to sample along the spline curve.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]:
            - sampled_points: [num_points, 2] array of (x, y) coordinates.
            - local_curvatures: Array of local curvature values kappa along the path.
    """
    if len(ordered_points) < 2:
        return np.array(ordered_points, dtype=np.float64), None

    pts = np.array(ordered_points, dtype=np.float64)

    # Remove consecutive duplicate points if any
    diff = np.diff(pts, axis=0)
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(diff != 0, axis=1)
    pts = pts[keep]

    if len(pts) < 4:
        # Fallback to linear interpolation if too few points for cubic spline
        t_in = np.linspace(0, 1, len(pts))
        t_out = np.linspace(0, 1, num_points)
        x_out = np.interp(t_out, t_in, pts[:, 0])
        y_out = np.interp(t_out, t_in, pts[:, 1])
        sampled = np.column_stack([x_out, y_out])
        return sampled, None

    x, y = pts[:, 0], pts[:, 1]
    k = min(3, len(pts) - 1)

    try:
        tck, u = splprep([x, y], s=smoothing * len(pts), k=k)
        u_fine = np.linspace(0, 1, num_points)
        x_spl, y_spl = splev(u_fine, tck)

        # First and second derivatives for curvature calculation
        dx, dy = splev(u_fine, tck, der=1)
        ddx, ddy = splev(u_fine, tck, der=2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx ** 2 + dy ** 2) ** 1.5
        # Avoid division by zero
        safe_denom = np.where(denominator < 1e-9, 1e-9, denominator)
        kappa = numerator / safe_denom

        sampled = np.column_stack([x_spl, y_spl])
        return sampled, kappa
    except Exception:
        # Fallback to direct linear interpolation if splprep encounters a singular matrix
        t_in = np.linspace(0, 1, len(pts))
        t_out = np.linspace(0, 1, num_points)
        x_out = np.interp(t_out, t_in, pts[:, 0])
        y_out = np.interp(t_out, t_in, pts[:, 1])
        sampled = np.column_stack([x_out, y_out])
        return sampled, None


def calculate_chord_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate the straight-line Euclidean distance between two endpoints.
    D = sqrt((x2 - x1)^2 + (y2 - y1)^2)
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(np.hypot(dx, dy))


def calculate_path_length(points: np.ndarray) -> float:
    """
    Calculate the cumulative path length along an ordered sequence of coordinates.
    L = sum(sqrt(dx^2 + dy^2))
    """
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    segment_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    return float(np.sum(segment_lengths))


def calculate_curve_score(path_length: float, chord_distance: float) -> float:
    """
    Compute project-defined Curve Score:
    Score = ((Path Length - Chord Distance) / Chord Distance) * 100

    Clamped to >= 0.0 to prevent minor negative values due to floating-point rounding.
    """
    if chord_distance <= 1e-6:
        return 0.0
    score = ((path_length - chord_distance) / chord_distance) * 100.0
    return max(0.0, float(score))


def classify_curvature(
    curve_score: float,
    straight_threshold: float = STRAIGHT_THRESHOLD,
    curved_threshold: float = CURVED_THRESHOLD,
) -> Tuple[bool, str]:
    """
    Classify curvature based on numerical score.

    Returns:
        Tuple[bool, str]: (is_curved, category_name)
    """
    if curve_score < straight_threshold:
        return False, CATEGORY_STRAIGHT
    elif curve_score <= curved_threshold:
        return True, CATEGORY_CURVED
    else:
        return True, CATEGORY_HIGHLY_CURVED


def calculate_max_deflection(
    centerline_points: np.ndarray,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> Tuple[float, Tuple[float, float], Tuple[float, float]]:
    """
    Calculate the maximum perpendicular deflection (sagitta) of the centerline
    away from the straight chord line connecting p1 and p2.

    Returns:
        Tuple:
            - max_deflection: Maximum distance in pixels.
            - apex_point: Coordinates (x, y) on the centerline of maximum deflection.
            - apex_chord_foot: Point (x, y) on the chord line directly below the apex.
    """
    x1, y1 = p1
    x2, y2 = p2
    chord_len = np.hypot(x2 - x1, y2 - y1)

    if chord_len < 1e-6 or len(centerline_points) == 0:
        return 0.0, p1, p1

    # Unit vector along chord and perpendicular vector
    ux = (x2 - x1) / chord_len
    uy = (y2 - y1) / chord_len

    pts = centerline_points
    vx = pts[:, 0] - x1
    vy = pts[:, 1] - y1

    # Scalar projection along chord: t = v . u
    t = vx * ux + vy * uy

    # Foot of perpendicular on the chord line
    foot_x = x1 + t * ux
    foot_y = y1 + t * uy

    # Perpendicular distances
    distances = np.hypot(pts[:, 0] - foot_x, pts[:, 1] - foot_y)
    max_idx = int(np.argmax(distances))

    max_dist = float(distances[max_idx])
    apex = (float(pts[max_idx, 0]), float(pts[max_idx, 1]))
    foot = (float(foot_x[max_idx]), float(foot_y[max_idx]))

    return max_dist, apex, foot


def analyze_banana(
    image_rgb: np.ndarray,
    segmentation_method: str = "auto",
    hsv_lower: Optional[Tuple[int, int, int]] = None,
    hsv_upper: Optional[Tuple[int, int, int]] = None,
    smoothing: float = SPLINE_SMOOTHING,
    straight_threshold: float = STRAIGHT_THRESHOLD,
    curved_threshold: float = CURVED_THRESHOLD,
    morph_kernel_size: Optional[int] = None,
) -> CurvatureAnalysisResult:
    """
    End-to-end banana curvature analysis pipeline:
    1. Segments banana foreground.
    2. Extracts 1-pixel skeleton.
    3. Finds endpoints and traces main centerline path.
    4. Smooths centerline with B-spline.
    5. Calculates path length, chord distance, Curve Score, and max deflection.
    6. Categorizes curvature into Straight, Curved, or Highly Curved.

    Args:
        image_rgb: RGB input image.
        segmentation_method: 'auto' (smart multi-space), 'wood', 'hsv', 'otsu', or 'saturation'.
        hsv_lower: Optional custom HSV lower bound.
        hsv_upper: Optional custom HSV upper bound.
        smoothing: Spline smoothing parameter.
        straight_threshold: Upper limit for 'Straight' classification.
        curved_threshold: Upper limit for 'Curved' classification.
        morph_kernel_size: Optional kernel size for morphological operations.

    Returns:
        CurvatureAnalysisResult: Comprehensive analysis metrics and data structures.
    """
    # 1. Segment banana
    mask, contour = segment_banana(
        image_rgb,
        method=segmentation_method,
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        morph_kernel_size=morph_kernel_size,
    )

    if contour is None or np.count_nonzero(mask) == 0:
        return CurvatureAnalysisResult(
            success=False,
            message="No banana detected in the image. Try adjusting the segmentation method or thresholds.",
        )

    # 2. Extract skeleton
    skeleton = extract_skeleton(mask)
    if np.count_nonzero(skeleton) < 2:
        return CurvatureAnalysisResult(
            success=False,
            message="Could not extract a valid skeleton from the detected contour.",
            contour=contour,
            binary_mask=mask,
        )

    # 3. Trace longest centerline
    ordered_points, p1, p2 = trace_longest_centerline(skeleton)
    if not ordered_points or p1 is None or p2 is None:
        return CurvatureAnalysisResult(
            success=False,
            message="Failed to trace end-to-end centerline from skeleton.",
            contour=contour,
            binary_mask=mask,
        )

    # 4. Fit spline and compute smooth centerline
    centerline_points, local_curvatures = fit_centerline_spline(
        ordered_points,
        smoothing=smoothing,
        num_points=SPLINE_NUM_POINTS,
    )

    # Endpoints from smooth curve
    ep1 = (float(centerline_points[0, 0]), float(centerline_points[0, 1]))
    ep2 = (float(centerline_points[-1, 0]), float(centerline_points[-1, 1]))

    # 5. Calculate geometric metrics
    chord_distance = calculate_chord_distance(ep1, ep2)
    path_length = calculate_path_length(centerline_points)
    curve_score = calculate_curve_score(path_length, chord_distance)
    is_curved, category = classify_curvature(
        curve_score,
        straight_threshold=straight_threshold,
        curved_threshold=curved_threshold,
    )

    # Max deflection (sagitta)
    max_deflection, apex_point, apex_foot = calculate_max_deflection(
        centerline_points, ep1, ep2
    )
    deflection_ratio = max_deflection / chord_distance if chord_distance > 0 else 0.0

    return CurvatureAnalysisResult(
        success=True,
        message="Analysis completed successfully.",
        is_curved=is_curved,
        curve_score=curve_score,
        category=category,
        path_length=path_length,
        chord_distance=chord_distance,
        max_deflection=max_deflection,
        deflection_ratio=deflection_ratio,
        endpoint_1=ep1,
        endpoint_2=ep2,
        apex_point=apex_point,
        apex_chord_foot=apex_foot,
        centerline_points=centerline_points,
        contour=contour,
        binary_mask=mask,
        local_curvatures=local_curvatures,
    )
