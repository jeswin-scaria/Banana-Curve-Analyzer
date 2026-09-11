"""
Configuration settings and default parameters for Banana Curve Analyzer.
"""

from typing import Dict, Tuple

# Curvature Classification Thresholds (in percentage %)
# Curve Score = ((Path Length - Chord Distance) / Chord Distance) * 100
STRAIGHT_THRESHOLD: float = 5.0      # Score < 5.0% is classified as Straight
CURVED_THRESHOLD: float = 20.0       # 5.0% <= Score <= 20.0% is Curved; > 20.0% is Highly Curved

# Classification Labels
CATEGORY_STRAIGHT = "Straight"
CATEGORY_CURVED = "Curved"
CATEGORY_HIGHLY_CURVED = "Highly Curved"

# HSV Color Ranges for Banana Segmentation
# OpenCV Hue is [0, 179], Saturation is [0, 255], Value is [0, 255]
# Covers yellow (approx 18-35) and green/unripe shades (approx 35-85)
DEFAULT_HSV_LOWER = (15, 40, 40)
DEFAULT_HSV_UPPER = (85, 255, 255)

# Morphological cleanup settings
MORPH_KERNEL_SIZE: int = 5
MIN_CONTOUR_AREA: int = 1000  # Minimum pixel area to consider as banana

# Centerline Spline Smoothing Parameters
SPLINE_SMOOTHING: float = 2.0
SPLINE_NUM_POINTS: int = 200  # Number of points to sample along continuous spline

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
