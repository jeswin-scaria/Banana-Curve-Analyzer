"""
Configuration settings and default parameters for Banana Curve Analyzer.
"""

from typing import Dict, Tuple

# Curvature Classification Thresholds (in percentage %)
# Curve Score = ((Path Length - Chord Distance) / Chord Distance) * 100
STRAIGHT_THRESHOLD: float = 5.0      # Score < 5.0% is classified as Straight
CURVED_THRESHOLD: float = 20.0       # 5.0% <= Score <= 20.0% is Curved; > 20.0% is Highly Curved

# These cutoffs are project-defined for this demo -- no scientific/industry standard
# for banana curvature classification exists, and they were not statistically derived
# from labeled data. Report them to judges as such, not as an authoritative standard.
CLASSIFICATION_THRESHOLDS_NOTE: str = (
    "STRAIGHT_THRESHOLD and CURVED_THRESHOLD are project-defined cutoffs chosen for this "
    "hackathon demo. No scientific/industry standard for banana curvature classification "
    "exists; these were not statistically derived from labeled data."
)

# Classification Labels
CATEGORY_STRAIGHT = "Straight"
CATEGORY_CURVED = "Curved"
CATEGORY_HIGHLY_CURVED = "Highly Curved"

# HSV Color Ranges for Banana Segmentation
# OpenCV Hue is [0, 179], Saturation is [0, 255], Value is [0, 255]
# Covers yellow (approx 18-35) and green/unripe shades (approx 35-85)
DEFAULT_HSV_LOWER = (15, 40, 40)
DEFAULT_HSV_UPPER = (85, 255, 255)

# Minimum HSV saturation (0-1 normalized) for a pixel to count as "colored" in
# the banana saliency map. Used as a boolean gate, NOT a continuous multiplier:
# a glossy studio-lit banana has a bright specular highlight running down its
# body where saturation drops sharply even though the pixel is still clearly
# part of the banana. Multiplying the LAB-yellowness signal by saturation
# directly (as a continuous factor) compounds two measures of the same
# "how washed-out is this" property and collapses highlighted regions to a
# near-zero score well before they'd look non-yellow to a human -- gating
# instead of multiplying keeps highlighted banana surface correctly attached
# to the rest of the mask while still rejecting truly gray/white backgrounds.
SALIENCY_MIN_SATURATION_GATE: float = 0.12

MIN_CONTOUR_AREA: int = 1000  # Minimum pixel area to consider as banana

# The full geometric pipeline (segmentation, morphology, skeletonization,
# centerline tracing) runs on a working copy capped at this many pixels on the
# longest side, then all coordinates are scaled back to the original image's
# resolution before being reported. Morphology and skeletonize cost scale with
# raw pixel count, and a modern phone photo can be 10+ megapixels -- without
# this cap, analysis time scales into multiple seconds per image. Curve Score
# is a ratio of two lengths (scale-invariant), so this trades a small amount of
# sub-pixel precision in the traced centerline for a large, necessary speedup.
ANALYSIS_MAX_DIMENSION: int = 1200

# Centerline Spline Smoothing Parameters
SPLINE_SMOOTHING: float = 2.0
SPLINE_NUM_POINTS: int = 200  # Number of points to sample along continuous spline

# --- Segmentation candidate scoring (contour shape-plausibility) ---
# How many candidates (by area) each segmentation strategy contributes for scoring,
# instead of that strategy's single largest contour "winning" automatically.
SEGMENTATION_CANDIDATES_PER_STRATEGY: int = 2
# Weighted combination of shape-plausibility sub-scores -> plausibility_score (0-1).
# Elongation and extent are weighted highest because together they are what separates
# a tapered, curved banana silhouette from a large, blocky, non-banana blob.
PLAUSIBILITY_WEIGHT_AREA: float = 0.10
PLAUSIBILITY_WEIGHT_SOLIDITY: float = 0.15
PLAUSIBILITY_WEIGHT_ELONGATION: float = 0.30
PLAUSIBILITY_WEIGHT_EXTENT: float = 0.20
PLAUSIBILITY_WEIGHT_SMOOTHNESS: float = 0.10
PLAUSIBILITY_WEIGHT_FRAME_COVERAGE: float = 0.15
# A banana's bounding min-area-rect is expected to be long and thin (elongation =
# long_side / short_side) and to fill only a fraction of that rectangle (extent =
# area / rect_area) because of its taper -- unlike a filled rectangle (extent ~1.0).
BANANA_ELONGATION_RANGE: Tuple[float, float] = (2.5, 7.0)
BANANA_EXTENT_RANGE: Tuple[float, float] = (0.40, 0.78)
# A real photo's subject essentially never has its own axis-aligned bounding box
# cover most of the frame in both dimensions at once -- photographers leave
# margin. A contour whose bbox exceeds this fraction of the frame area is very
# likely the whole image (or a background+banana blob merged into one region),
# not a photographed banana -- this is what actually catches the case a filled
# rectangle's elongation/extent alone can miss: RETR_EXTERNAL discards holes, so
# "the frame with a banana-shaped hole in it" scores deceptively similar to a
# clean rectangle on every OTHER shape metric.
FRAME_COVERAGE_MAX_PLAUSIBLE: float = 0.65
# Tie-breaking priority multiplier applied only when *choosing* the winning
# candidate (never to the stored plausibility_score itself). Color-aware
# strategies are semantically meaningful for "is this a banana" -- generic
# brightness-based Otsu thresholding is not (it would happily pick out a dark
# mug on a light table). This mirrors the original single-strategy cascade's
# fallback order: color-based methods first, plain brightness thresholding last.
SEGMENTATION_METHOD_PRIORITY: Dict[str, float] = {
    # A cut-out image's alpha channel is the object's real silhouette rather
    # than an estimate of it, so it outranks every colour heuristic -- but it is
    # still shape-scored, so a nonsense alpha channel cannot hijack the result.
    "alpha_channel": 1.30,
    "grabcut_refined": 1.05,
    "saliency_strict": 1.02,
    "saliency_wood": 1.01,
    "saliency_otsu": 1.00,
    "hsv_inrange": 0.97,
    "saturation": 0.93,
    "gray_otsu": 0.65,
}

# --- Skeleton spur pruning ---
# A banana skeleton should topologically be a single unbranched path; any branch
# point is presumptively noise. Branches shorter than this length (in pixels) are
# pruned before endpoint/centerline tracing.
SPUR_MIN_LENGTH_FLOOR_PX: float = 5.0
SPUR_LENGTH_WIDTH_MULTIPLIER: float = 1.5
SPUR_PRUNE_MAX_ITERATIONS: int = 10

# --- Adaptive spline smoothing (used when smoothing=None is passed) ---
# Noise sigma is estimated from the residual of raw centerline points against a
# light moving average; s is then set following SciPy's guidance that a good
# smoothing factor is on the order of s ~= m * sigma^2 (m = number of points).
SPLINE_ADAPTIVE_WINDOW: int = 5
SPLINE_ADAPTIVE_SIGMA_FLOOR: float = 0.3
# Floor matches the legacy fixed SPLINE_SMOOTHING (2.0): a clean, low-noise
# centerline (the common case for synthetic test shapes) gets exactly the
# historical amount of smoothing, never less -- adaptive smoothing only ever
# scales *up* from there as measured noise increases, so it never introduces
# under-smoothing regressions relative to the previous fixed-constant behavior.
SPLINE_ADAPTIVE_S_MIN_MULTIPLIER: float = 2.0
SPLINE_ADAPTIVE_S_MAX_MULTIPLIER: float = 50.0

# --- Image quality gates ---
BLUR_VARIANCE_MIN: float = 50.0
CONTRAST_STD_MIN: float = 15.0
QUALITY_REJECT_BLUR_VARIANCE: float = 15.0
QUALITY_REJECT_PLAUSIBILITY_MAX: float = 0.35
MIN_BANANA_AREA_FRACTION: float = 0.005
MULTI_BLOB_AREA_RATIO: float = 0.15

# --- Confidence score weights (documented heuristic, NOT a calibrated probability) ---
# Weighted to match the pipeline's own priority order: segmentation reliability
# matters most, then centerline/endpoint reliability, then smoothing/image quality.
CONFIDENCE_WEIGHT_SEGMENTATION: float = 0.30
CONFIDENCE_WEIGHT_SKELETON_CONTINUITY: float = 0.20
CONFIDENCE_WEIGHT_ENDPOINT_ALIGNMENT: float = 0.20
CONFIDENCE_WEIGHT_SPLINE_FIT: float = 0.15
CONFIDENCE_WEIGHT_IMAGE_QUALITY: float = 0.15
CONFIDENCE_STATUS_THRESHOLD: float = 70.0

# --- Failure stages / status labels ---
FAILURE_STAGE_SEGMENTATION_FAILED = "SEGMENTATION_FAILED"
FAILURE_STAGE_CENTERLINE_FAILED = "CENTERLINE_FAILED"
FAILURE_STAGE_ENDPOINT_FAILED = "ENDPOINT_FAILED"
FAILURE_STAGE_SPLINE_FAILED = "SPLINE_FAILED"
FAILURE_STAGE_GEOMETRY_FAILED = "GEOMETRY_FAILED"
FAILURE_STAGE_QUALITY_REJECTED = "QUALITY_REJECTED"
STATUS_HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
STATUS_LOW_CONFIDENCE = "LOW_CONFIDENCE"

# Visualization Colors (BGR for OpenCV, RGB/Hex for Streamlit/Matplotlib)
COLORS: Dict[str, Tuple[int, int, int]] = {
    "contour": (0, 220, 0),        # Vibrant Green
    "centerline": (255, 180, 0),   # Bright Cyan/Sky Blue in BGR (0, 180, 255) -> BGR: (255, 180, 0)
    "chord": (0, 140, 255),        # Deep Orange in BGR (0, 140, 255)
    "endpoint_1": (0, 255, 0),     # Bright Green
    "endpoint_2": (0, 255, 255),   # Bright Yellow
    "deflection": (255, 0, 255),   # Magenta
}

CATEGORY_COLORS: Dict[str, str] = {
    CATEGORY_STRAIGHT: "#22c55e",       # Green
    CATEGORY_CURVED: "#f59e0b",         # Amber/Orange
    CATEGORY_HIGHLY_CURVED: "#ef4444",  # Crimson/Red
}
