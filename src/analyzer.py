"""
Curvature calculation and analysis module for Banana Curve Analyzer.
Implements the core project-defined formula:
Curve Score = ((Centerline Path Length - Straight-Line Distance) / Straight-Line Distance) * 100
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
from scipy.interpolate import splprep, splev

from .config import (
    STRAIGHT_THRESHOLD,
    CURVED_THRESHOLD,
    CATEGORY_STRAIGHT,
    CATEGORY_CURVED,
    CATEGORY_HIGHLY_CURVED,
    SPLINE_SMOOTHING,
    SPLINE_NUM_POINTS,
    ANALYSIS_MAX_DIMENSION,
    SPLINE_ADAPTIVE_WINDOW,
    SPLINE_ADAPTIVE_SIGMA_FLOOR,
    SPLINE_ADAPTIVE_S_MIN_MULTIPLIER,
    SPLINE_ADAPTIVE_S_MAX_MULTIPLIER,
    CONFIDENCE_STATUS_THRESHOLD,
    STATUS_HIGH_CONFIDENCE,
    STATUS_LOW_CONFIDENCE,
    QUALITY_REJECT_BLUR_VARIANCE,
    QUALITY_REJECT_PLAUSIBILITY_MAX,
    FAILURE_STAGE_SEGMENTATION_FAILED,
    FAILURE_STAGE_CENTERLINE_FAILED,
    FAILURE_STAGE_ENDPOINT_FAILED,
    FAILURE_STAGE_SPLINE_FAILED,
    FAILURE_STAGE_GEOMETRY_FAILED,
    FAILURE_STAGE_QUALITY_REJECTED,
)
from .preprocessing import segment_banana_scored
from .skeleton import (
    extract_skeleton,
    trace_longest_centerline,
    prune_skeleton_spurs,
    estimate_adaptive_spur_threshold,
    extend_centerline_to_boundary,
)
from .quality import assess_image_quality, assess_shape_quality_flags, compute_confidence_score


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
    # --- Diagnostics added for explainability (all additive, safe defaults) ---
    failure_stage: Optional[str] = None
    quality_status: str = "UNKNOWN"
    confidence_score: float = 0.0
    segmentation_method: Optional[str] = None
    segmentation_score: float = 0.0
    quality_flags: Optional[Dict[str, Any]] = None
    mean_curvature: float = 0.0
    integrated_absolute_curvature: float = 0.0
    curvature_variance: float = 0.0
    endpoint_alignment_score: float = 0.0
    spline_fit_residual: float = 0.0

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
            "failure_stage": self.failure_stage,
            "quality_status": self.quality_status,
            "confidence_score": round(self.confidence_score, 1),
            "segmentation_method": self.segmentation_method,
            "segmentation_score": round(self.segmentation_score, 3),
            "quality_flags": self.quality_flags,
            "mean_curvature": round(self.mean_curvature, 6),
            "integrated_absolute_curvature_deg": round(float(np.degrees(self.integrated_absolute_curvature)), 2),
            "curvature_variance": round(self.curvature_variance, 8),
            "endpoint_alignment_score": round(self.endpoint_alignment_score, 3),
            "spline_fit_residual_px": round(self.spline_fit_residual, 3),
        }


def _estimate_centerline_noise_sigma(pts: np.ndarray, window: int = SPLINE_ADAPTIVE_WINDOW) -> float:
    """
    Estimate per-point pixel noise as the RMS residual of the raw centerline
    against a light moving average of itself. This is a coarse local-jitter
    estimate (pixel-grid/skeletonization noise), not a claim about the true
    banana boundary -- it exists only to set an appropriately-sized spline
    smoothing factor for *this* centerline instead of one fixed constant.
    """
    n = len(pts)
    if n < window + 1:
        return SPLINE_ADAPTIVE_SIGMA_FLOOR

    kernel = np.ones(window) / window
    smoothed_x = np.convolve(pts[:, 0], kernel, mode="valid")
    smoothed_y = np.convolve(pts[:, 1], kernel, mode="valid")

    trim = window // 2
    raw_x = pts[trim: trim + len(smoothed_x), 0]
    raw_y = pts[trim: trim + len(smoothed_y), 1]

    residuals = np.hypot(raw_x - smoothed_x, raw_y - smoothed_y)
    sigma = float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) > 0 else 0.0
    return max(sigma, SPLINE_ADAPTIVE_SIGMA_FLOOR)


def fit_centerline_spline(
    ordered_points: List[Tuple[float, float]],
    smoothing: Optional[float] = None,
    num_points: int = SPLINE_NUM_POINTS,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Fit a smooth parametric B-spline to the ordered centerline points to eliminate
    discrete pixel-grid step artifacts and calculate smooth continuous derivatives.

    Args:
        ordered_points: List of (x, y) coordinates along the skeleton centerline.
        smoothing: Spline smoothing parameter (s in splprep). If None (default),
            the smoothing factor is estimated adaptively from this centerline's
            own noise level and point count (SciPy's guidance: a good s is on
            the order of s ~= m * sigma^2). Pass an explicit float to use the
            fixed s = smoothing * len(points) behavior instead.
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

    if smoothing is None:
        sigma = _estimate_centerline_noise_sigma(pts)
        s_value = float(np.clip(
            (sigma ** 2) * len(pts),
            SPLINE_ADAPTIVE_S_MIN_MULTIPLIER * len(pts),
            SPLINE_ADAPTIVE_S_MAX_MULTIPLIER * len(pts),
        ))
    else:
        s_value = smoothing * len(pts)

    try:
        tck, u = splprep([x, y], s=s_value, k=k)
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

    NOTE: straight_threshold/curved_threshold are project-defined cutoffs for
    this demo (see config.CLASSIFICATION_THRESHOLDS_NOTE) -- not a scientific
    or industry standard.

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


def _compute_secondary_curvature_metrics(
    centerline_points: np.ndarray,
    local_curvatures: Optional[np.ndarray],
    path_length: float,
) -> Tuple[float, float, float]:
    """
    Secondary, explainable curvature metrics computed alongside (never replacing)
    the primary Curve Score:
        mean_curvature:                arc-length-weighted average of kappa.
        integrated_absolute_curvature: total turning angle (radians) = integral
                                        of kappa ds along the path.
        curvature_variance:            arc-length-weighted variance of kappa.

    Arc-length weighting matters because the 200 spline samples are uniform in
    the spline parameter u, not in arc length -- a plain np.mean/np.var over
    the raw kappa array would misrepresent these wherever samples bunch up.
    """
    if local_curvatures is None or len(centerline_points) < 2:
        return 0.0, 0.0, 0.0

    diffs = np.diff(centerline_points, axis=0)
    ds = np.hypot(diffs[:, 0], diffs[:, 1])
    kappa_avg = (local_curvatures[:-1] + local_curvatures[1:]) / 2.0

    integrated_absolute_curvature = float(np.sum(kappa_avg * ds))
    mean_curvature = integrated_absolute_curvature / max(path_length, 1e-9)
    curvature_variance = float(np.average((local_curvatures - mean_curvature) ** 2))

    return mean_curvature, integrated_absolute_curvature, curvature_variance


def _endpoint_alignment_score(
    contour: np.ndarray,
    ep1: Tuple[float, float],
    ep2: Tuple[float, float],
) -> float:
    """
    Confidence signal, not a correction: are the chosen centerline endpoints
    close to the contour's true extreme points along its principal axis (via
    PCA)? A banana's two tips should lie near the extremes of its longest axis;
    a low score here means the traced path may have picked a suboptimal route
    (e.g. a leftover noisy branch), and should lower overall confidence rather
    than silently be trusted.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 3:
        return 0.0

    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, int(np.argmax(eigvals))]

    projections = centered @ principal_axis
    axis_length = float(projections.max() - projections.min())
    if axis_length < 1e-6:
        return 0.0

    ep1_proj = float(np.dot(np.array(ep1) - mean, principal_axis))
    ep2_proj = float(np.dot(np.array(ep2) - mean, principal_axis))
    proj_min, proj_max = float(projections.min()), float(projections.max())

    # Endpoints should align with OPPOSITE extremes of the axis, not the same one.
    dist_ep1_min_ep2_max = abs(ep1_proj - proj_min) + abs(ep2_proj - proj_max)
    dist_ep1_max_ep2_min = abs(ep1_proj - proj_max) + abs(ep2_proj - proj_min)
    total_offset = min(dist_ep1_min_ep2_max, dist_ep1_max_ep2_min)

    return float(np.clip(1.0 - total_offset / axis_length, 0.0, 1.0))


def _spline_fit_residual(raw_points: np.ndarray, sampled_points: np.ndarray) -> float:
    """
    Mean nearest-neighbor distance from each raw (pre-spline) centerline point
    to the fitted, smoothed curve -- a measure of how much the spline actually
    changed the shape versus just denoising pixel-grid jitter. Large values mean
    the fitted curve strayed noticeably from the traced skeleton path.
    """
    if len(raw_points) == 0 or len(sampled_points) == 0:
        return 0.0
    diffs = raw_points[:, None, :] - sampled_points[None, :, :]
    dists = np.hypot(diffs[..., 0], diffs[..., 1])
    return float(np.mean(dists.min(axis=1)))


def analyze_banana(
    image_rgb: np.ndarray,
    segmentation_method: str = "auto",
    hsv_lower: Optional[Tuple[int, int, int]] = None,
    hsv_upper: Optional[Tuple[int, int, int]] = None,
    smoothing: Optional[float] = None,
    straight_threshold: float = STRAIGHT_THRESHOLD,
    curved_threshold: float = CURVED_THRESHOLD,
    morph_kernel_size: Optional[int] = None,
    alpha_mask: Optional[np.ndarray] = None,
) -> CurvatureAnalysisResult:
    """
    End-to-end banana curvature analysis pipeline:
    1. Assesses raw image quality (blur, contrast).
    2. Segments banana foreground from multiple scored candidates.
    3. Extracts and prunes the 1-pixel skeleton.
    4. Finds endpoints and traces main centerline path.
    5. Smooths centerline with an adaptively-smoothed B-spline.
    6. Calculates path length, chord distance, Curve Score, deflection, and
       secondary curvature metrics.
    7. Categorizes curvature into Straight, Curved, or Highly Curved.
    8. Computes an explainable confidence score and HIGH/LOW quality status.

    Args:
        image_rgb: RGB input image.
        segmentation_method: 'auto' (smart multi-space), 'wood', 'hsv', 'otsu', or 'saturation'.
        hsv_lower: Optional custom HSV lower bound.
        hsv_upper: Optional custom HSV upper bound.
        smoothing: Spline smoothing parameter. None (default) uses adaptive smoothing.
        straight_threshold: Upper limit for 'Straight' classification (project-defined).
        curved_threshold: Upper limit for 'Curved' classification (project-defined).
        morph_kernel_size: Optional kernel size for morphological operations.

    Returns:
        CurvatureAnalysisResult: Comprehensive analysis metrics and data structures.
    """
    # Blur/contrast detection is resolution-sensitive (downsampling itself
    # blurs an image), so image quality is always assessed on the original,
    # full-resolution photo -- only the expensive geometry stages below run on
    # a capped-resolution working copy.
    image_quality = assess_image_quality(image_rgb)

    h_orig, w_orig = image_rgb.shape[:2]
    scale = min(1.0, ANALYSIS_MAX_DIMENSION / max(h_orig, w_orig)) if max(h_orig, w_orig) > 0 else 1.0
    if scale < 1.0:
        work_w, work_h = max(1, int(round(w_orig * scale))), max(1, int(round(h_orig * scale)))
        work_image = cv2.resize(image_rgb, (work_w, work_h), interpolation=cv2.INTER_AREA)
        work_alpha = (
            cv2.resize(alpha_mask, (work_w, work_h), interpolation=cv2.INTER_NEAREST)
            if alpha_mask is not None else None
        )
    else:
        work_image = image_rgb
        work_alpha = alpha_mask
    inv_scale = 1.0 / scale

    # 1. Segment banana (multi-candidate, shape-plausibility-scored) on the working copy
    mask_work, contour_work, seg_method, seg_score, _all_candidates = segment_banana_scored(
        work_image,
        method=segmentation_method,
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        morph_kernel_size=morph_kernel_size,
        alpha_mask=work_alpha,
    )

    if contour_work is None or np.count_nonzero(mask_work) == 0:
        return CurvatureAnalysisResult(
            success=False,
            message="No banana detected in the image. Try adjusting the segmentation method or thresholds.",
            failure_stage=FAILURE_STAGE_SEGMENTATION_FAILED,
            quality_flags={"image_quality": image_quality},
        )

    # Scale the winning mask/contour back up to the original resolution --
    # everything returned to the caller (and used for reporting/visualization)
    # is in original-image pixel coordinates, regardless of the working scale.
    if scale < 1.0:
        mask = cv2.resize(mask_work, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        contour = np.round(contour_work.astype(np.float64) * inv_scale).astype(np.int32)
    else:
        mask, contour = mask_work, contour_work

    shape_quality = assess_shape_quality_flags(mask, contour, (h_orig, w_orig))
    quality_flags = {"image_quality": image_quality, "shape_quality": shape_quality}

    # Hard quality floor: a contour WAS found, but blur + shape-uncertainty are
    # both bad enough that reporting a confident-looking number would mislead.
    if image_quality["blur_score"] < QUALITY_REJECT_BLUR_VARIANCE and seg_score < QUALITY_REJECT_PLAUSIBILITY_MAX:
        return CurvatureAnalysisResult(
            success=False,
            message="Image quality too low for a reliable measurement (severe blur combined with an uncertain banana outline).",
            failure_stage=FAILURE_STAGE_QUALITY_REJECTED,
            contour=contour,
            binary_mask=mask,
            segmentation_method=seg_method,
            segmentation_score=seg_score,
            quality_flags=quality_flags,
        )

    # 2. Extract and prune skeleton -- still on the working-resolution mask,
    # since skeletonize cost scales with pixel count.
    skeleton_raw = extract_skeleton(mask_work)
    if np.count_nonzero(skeleton_raw) < 2:
        return CurvatureAnalysisResult(
            success=False,
            message="Could not extract a valid skeleton from the detected contour.",
            failure_stage=FAILURE_STAGE_CENTERLINE_FAILED,
            contour=contour,
            binary_mask=mask,
            segmentation_method=seg_method,
            segmentation_score=seg_score,
            quality_flags=quality_flags,
        )

    spur_threshold = estimate_adaptive_spur_threshold(mask_work, skeleton_raw)
    skeleton = prune_skeleton_spurs(skeleton_raw, spur_threshold)
    if np.count_nonzero(skeleton) < 2:
        skeleton = skeleton_raw  # pruning over-pruned everything; fall back to raw

    raw_count = np.count_nonzero(skeleton_raw)
    skeleton_continuity_score = float(np.count_nonzero(skeleton) / raw_count) if raw_count > 0 else 0.0
    skeleton_continuity_score = float(np.clip(skeleton_continuity_score, 0.0, 1.0))

    # 3. Trace longest centerline (working resolution)
    ordered_points_work, p1, p2 = trace_longest_centerline(skeleton)
    if not ordered_points_work or p1 is None or p2 is None:
        return CurvatureAnalysisResult(
            success=False,
            message="Failed to trace end-to-end centerline from skeleton.",
            failure_stage=FAILURE_STAGE_ENDPOINT_FAILED,
            contour=contour,
            binary_mask=mask,
            segmentation_method=seg_method,
            segmentation_score=seg_score,
            quality_flags=quality_flags,
        )

    # 4. Fit spline and compute smooth centerline (working resolution)
    centerline_points_work, local_curvatures_work = fit_centerline_spline(
        ordered_points_work,
        smoothing=smoothing,
        num_points=SPLINE_NUM_POINTS,
    )

    if (
        centerline_points_work is None
        or len(centerline_points_work) < 2
        or not np.all(np.isfinite(centerline_points_work))
    ):
        return CurvatureAnalysisResult(
            success=False,
            message="Spline fitting produced a degenerate centerline.",
            failure_stage=FAILURE_STAGE_SPLINE_FAILED,
            contour=contour,
            binary_mask=mask,
            segmentation_method=seg_method,
            segmentation_score=seg_score,
            quality_flags=quality_flags,
        )

    # Scale the fitted centerline back to original-image pixel coordinates.
    # Local curvature kappa has units of 1/length, so it scales by `scale`
    # (not inv_scale) when lengths scale by inv_scale.
    centerline_points = centerline_points_work * inv_scale
    local_curvatures = local_curvatures_work * scale if local_curvatures_work is not None else None

    # Endpoints from smooth curve, in original-image pixel coordinates
    ep1 = (float(centerline_points[0, 0]), float(centerline_points[0, 1]))
    ep2 = (float(centerline_points[-1, 0]), float(centerline_points[-1, 1]))

    # 5. Calculate geometric metrics (original-image pixel units)
    chord_distance = calculate_chord_distance(ep1, ep2)

    if chord_distance < 1e-3:
        return CurvatureAnalysisResult(
            success=False,
            message="Degenerate geometry: banana endpoints coincide.",
            failure_stage=FAILURE_STAGE_GEOMETRY_FAILED,
            contour=contour,
            binary_mask=mask,
            segmentation_method=seg_method,
            segmentation_score=seg_score,
            quality_flags=quality_flags,
        )

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

    mean_curvature, integrated_absolute_curvature, curvature_variance = _compute_secondary_curvature_metrics(
        centerline_points, local_curvatures, path_length
    )

    endpoint_alignment_score = _endpoint_alignment_score(contour, ep1, ep2)
    # Nearest-neighbor structure is preserved under uniform scaling, so the
    # working-resolution residual just needs the same inv_scale factor rather
    # than recomputing nearest-neighbor search on the full-resolution points.
    spline_fit_residual = _spline_fit_residual(
        np.array(ordered_points_work, dtype=np.float64), centerline_points_work
    ) * inv_scale
    spline_fit_quality_score = 1.0 / (1.0 + spline_fit_residual / 3.0)

    confidence_score = compute_confidence_score(
        segmentation_score=seg_score,
        skeleton_continuity_score=skeleton_continuity_score,
        endpoint_alignment_score=endpoint_alignment_score,
        spline_fit_quality_score=spline_fit_quality_score,
        image_quality_score=image_quality["quality_score"],
    )
    quality_status = (
        STATUS_HIGH_CONFIDENCE if confidence_score >= CONFIDENCE_STATUS_THRESHOLD else STATUS_LOW_CONFIDENCE
    )

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
        failure_stage=None,
        quality_status=quality_status,
        confidence_score=confidence_score,
        segmentation_method=seg_method,
        segmentation_score=seg_score,
        quality_flags=quality_flags,
        mean_curvature=mean_curvature,
        integrated_absolute_curvature=integrated_absolute_curvature,
        curvature_variance=curvature_variance,
        endpoint_alignment_score=endpoint_alignment_score,
        spline_fit_residual=spline_fit_residual,
    )
