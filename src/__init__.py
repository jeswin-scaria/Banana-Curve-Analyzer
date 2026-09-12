"""
Banana Curve Analyzer - Computer Vision & Geometry Package.
"""

from .config import (
    STRAIGHT_THRESHOLD,
    CURVED_THRESHOLD,
    CATEGORY_STRAIGHT,
    CATEGORY_CURVED,
    CATEGORY_HIGHLY_CURVED,
)
from .preprocessing import load_image, segment_banana, segment_banana_scored
from .skeleton import extract_skeleton, trace_longest_centerline, prune_skeleton_spurs
from .analyzer import (
    CurvatureAnalysisResult,
    analyze_banana,
    calculate_chord_distance,
    calculate_path_length,
    calculate_curve_score,
    classify_curvature,
)
from .visualization import (
    create_annotated_overlay,
    create_isolated_shape_overlay,
    create_diagnostic_figure,
    compute_center_of_curvature,
)

__all__ = [
    "STRAIGHT_THRESHOLD",
    "CURVED_THRESHOLD",
    "CATEGORY_STRAIGHT",
    "CATEGORY_CURVED",
    "CATEGORY_HIGHLY_CURVED",
    "load_image",
    "segment_banana",
    "segment_banana_scored",
    "extract_skeleton",
    "trace_longest_centerline",
    "prune_skeleton_spurs",
    "CurvatureAnalysisResult",
    "analyze_banana",
    "calculate_chord_distance",
    "calculate_path_length",
    "calculate_curve_score",
    "classify_curvature",
    "create_annotated_overlay",
    "create_isolated_shape_overlay",
    "create_diagnostic_figure",
    "compute_center_of_curvature",
]
