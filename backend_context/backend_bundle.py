"""
🍌 CHILL ETHAKAAA — BANANA GEOMETRY LAB
Unified Backend Bundle (Self-Contained Single-File Module)
Combines config, preprocessing, skeletonization, analysis, and visualization.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import io
import os
import argparse
import json

import numpy as np
import cv2
from PIL import Image
import networkx as nx
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ===========================================================================
# 1. CONFIGURATION & CONSTANTS
# ===========================================================================

STRAIGHT_THRESHOLD: float = 5.0      # Score < 5.0% is Straight
CURVED_THRESHOLD: float = 20.0       # 5.0% <= Score <= 20.0% is Curved; > 20.0% is Highly Curved

CATEGORY_STRAIGHT = "Straight"
CATEGORY_CURVED = "Curved"
CATEGORY_HIGHLY_CURVED = "Highly Curved"

DEFAULT_HSV_LOWER = (15, 40, 40)
DEFAULT_HSV_UPPER = (85, 255, 255)

MORPH_KERNEL_SIZE: int = 5
MIN_CONTOUR_AREA: int = 1000

SPLINE_SMOOTHING: float = 2.0
SPLINE_NUM_POINTS: int = 200

COLORS: Dict[str, Tuple[int, int, int]] = {
    "contour": (0, 220, 0),        # Vibrant Green
    "centerline": (255, 180, 0),   # Cyan/Sky Blue in BGR
    "chord": (0, 140, 255),        # Deep Orange in BGR
    "endpoint_1": (0, 255, 0),     # Bright Green
    "endpoint_2": (0, 255, 255),   # Bright Yellow
    "deflection": (255, 0, 255),   # Magenta
}

CATEGORY_COLORS: Dict[str, str] = {
    CATEGORY_STRAIGHT: "#22c55e",
    CATEGORY_CURVED: "#f59e0b",
    CATEGORY_HIGHLY_CURVED: "#ef4444",
}


# ===========================================================================
# 2. PREPROCESSING & SEGMENTATION
# ===========================================================================

def load_image(source: Union[str, bytes, io.BytesIO, np.ndarray]) -> np.ndarray:
    """Load an image from path, buffer, or array into uint8 RGB format."""
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
    """Combines CIE LAB (b* high, a* wood filter) and HSV for robust segmentation."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    a_chan = lab[:, :, 1].astype(np.float32)
    b_chan = lab[:, :, 2].astype(np.float32)

    # 1. Ripe Yellow: High b* in LAB, low a* (not red/brown wood), yellow hue
    yellow_b_strength = np.maximum(0.0, b_chan - 145.0)
    yellow_hue_gate = (h >= 15) & (h <= 40)
    yellow_a_gate = a_chan <= 136.0
    yellow_score = yellow_b_strength * s * v * yellow_hue_gate * yellow_a_gate

    # 2. Green / Unripe: Green hue in HSV and low a* (green side of LAB)
    green_hue_gate = (h > 35) & (h <= 85)
    green_a_strength = np.maximum(0.0, 132.0 - a_chan)
    green_score = green_a_strength * s * v * green_hue_gate

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
    """Segments banana foreground, cleans boundaries, and extracts contour."""
    h, w = image_rgb.shape[:2]
    k_size = morph_kernel_size or max(5, int(min(h, w) / 90) | 1)

    raw_mask: Optional[np.ndarray] = None

    if method in ("auto", "wood"):
        saliency = compute_smart_banana_saliency(image_rgb)
        if cv2.countNonZero(saliency) > min_area:
            _, raw_mask = cv2.threshold(saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if method == "wood":
                hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
                lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
                strict_gate = (hsv[:, :, 1] >= 80) & (lab[:, :, 2] >= 155) & (lab[:, :, 1] <= 135)
                raw_mask = raw_mask & (strict_gate.astype(np.uint8) * 255)

    if raw_mask is None or cv2.countNonZero(raw_mask) < min_area:
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

    # Morphological opening (sever bridges) and closing (seal blemishes)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, open_kernel, iterations=2)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size * 2, k_size * 2))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), dtype=np.uint8), None

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < min_area:
        return np.zeros((h, w), dtype=np.uint8), None

    final_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return final_mask, largest_contour


# ===========================================================================
# 3. SKELETONIZATION & CENTERLINE GRAPH TRACING
# ===========================================================================

def extract_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """1-pixel wide morphological skeletonization."""
    bool_mask = binary_mask > 0
    if not np.any(bool_mask):
        return np.zeros_like(binary_mask, dtype=np.uint8)
    skel = skeletonize(bool_mask)
    return (skel.astype(np.uint8)) * 255


def build_skeleton_graph(skeleton: np.ndarray) -> nx.Graph:
    """Constructs 8-connected weighted spatial graph from skeleton pixels."""
    g = nx.Graph()
    rows, cols = np.where(skeleton > 0)
    pixel_set = set(zip(rows, cols))

    neighbors = [
        (-1, -1, np.sqrt(2)), (-1, 0, 1.0), (-1, 1, np.sqrt(2)),
        (0, -1, 1.0),                        (0, 1, 1.0),
        (1, -1, np.sqrt(2)),  (1, 0, 1.0),  (1, 1, np.sqrt(2)),
    ]

    for r, c in pixel_set:
        g.add_node((r, c))
        for dr, dc, weight in neighbors:
            nr, nc = r + dr, c + dc
            if (nr, nc) in pixel_set:
                g.add_edge((r, c), (nr, nc), weight=weight)

    return g


def trace_longest_centerline(
    skeleton: np.ndarray,
) -> Tuple[List[Tuple[float, float]], Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Finds the longest geodesic path through skeleton graph, pruning small spurs."""
    if np.count_nonzero(skeleton) < 2:
        return [], None, None

    g = build_skeleton_graph(skeleton)
    if len(g) < 2:
        return [], None, None

    largest_cc = max(nx.connected_components(g), key=len)
    subg = g.subgraph(largest_cc).copy()
    endpoints = [node for node in subg.nodes() if subg.degree(node) == 1]

    best_path = []
    if len(endpoints) >= 2:
        max_dist = -1.0
        best_pair = (endpoints[0], endpoints[1])

        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                u, v = endpoints[i], endpoints[j]
                try:
                    dist = nx.shortest_path_length(subg, u, v, weight="weight")
                    if dist > max_dist:
                        max_dist = dist
                        best_pair = (u, v)
                except nx.NetworkXNoPath:
                    continue

        try:
            best_path = nx.shortest_path(subg, best_pair[0], best_pair[1], weight="weight")
        except nx.NetworkXNoPath:
            best_path = list(subg.nodes())
    elif len(endpoints) == 1:
        u = endpoints[0]
        lengths = nx.single_source_dijkstra_path_length(subg, u, weight="weight")
        farthest_node = max(lengths, key=lengths.get)
        best_path = nx.shortest_path(subg, u, farthest_node, weight="weight")
    else:
        nodes = list(subg.nodes())
        if len(nodes) >= 2:
            coords = np.array(nodes)
            diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            sq_dists = np.sum(diffs ** 2, axis=-1)
            i, j = np.unravel_index(np.argmax(sq_dists), sq_dists.shape)
            u, v = nodes[i], nodes[j]
            try:
                best_path = nx.shortest_path(subg, u, v, weight="weight")
            except nx.NetworkXNoPath:
                best_path = nodes
        else:
            best_path = nodes

    if not best_path:
        return [], None, None

    ordered_points = [(float(c), float(r)) for (r, c) in best_path]
    return ordered_points, ordered_points[0], ordered_points[-1]


# ===========================================================================
# 4. GEOMETRIC ANALYSIS & CURVATURE COMPUTATION
# ===========================================================================

@dataclass
class CurvatureAnalysisResult:
    """Analysis results data container."""
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
    """Parametric B-spline fitting for continuous curve representation."""
    if len(ordered_points) < 2:
        return np.array(ordered_points, dtype=np.float64), None

    pts = np.array(ordered_points, dtype=np.float64)
    diff = np.diff(pts, axis=0)
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(diff != 0, axis=1)
    pts = pts[keep]

    if len(pts) < 4:
        t_in = np.linspace(0, 1, len(pts))
        t_out = np.linspace(0, 1, num_points)
        x_out = np.interp(t_out, t_in, pts[:, 0])
        y_out = np.interp(t_out, t_in, pts[:, 1])
        return np.column_stack([x_out, y_out]), None

    x, y = pts[:, 0], pts[:, 1]
    k = min(3, len(pts) - 1)

    try:
        tck, u = splprep([x, y], s=smoothing * len(pts), k=k)
        u_fine = np.linspace(0, 1, num_points)
        x_spl, y_spl = splev(u_fine, tck)

        dx, dy = splev(u_fine, tck, der=1)
        ddx, ddy = splev(u_fine, tck, der=2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx ** 2 + dy ** 2) ** 1.5
        safe_denom = np.where(denominator < 1e-9, 1e-9, denominator)
        kappa = numerator / safe_denom

        return np.column_stack([x_spl, y_spl]), kappa
    except Exception:
        t_in = np.linspace(0, 1, len(pts))
        t_out = np.linspace(0, 1, num_points)
        x_out = np.interp(t_out, t_in, pts[:, 0])
        y_out = np.interp(t_out, t_in, pts[:, 1])
        return np.column_stack([x_out, y_out]), None


def calculate_chord_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def calculate_path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def calculate_curve_score(path_length: float, chord_distance: float) -> float:
    """Curve Score = ((L - D) / D) * 100"""
    if chord_distance <= 1e-6:
        return 0.0
    score = ((path_length - chord_distance) / chord_distance) * 100.0
    return max(0.0, float(score))


def classify_curvature(
    curve_score: float,
    straight_threshold: float = STRAIGHT_THRESHOLD,
    curved_threshold: float = CURVED_THRESHOLD,
) -> Tuple[bool, str]:
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
    """Computes perpendicular sagitta distance from chord line to apex."""
    x1, y1 = p1
    x2, y2 = p2
    chord_len = np.hypot(x2 - x1, y2 - y1)

    if chord_len < 1e-6 or len(centerline_points) == 0:
        return 0.0, p1, p1

    ux = (x2 - x1) / chord_len
    uy = (y2 - y1) / chord_len

    pts = centerline_points
    vx = pts[:, 0] - x1
    vy = pts[:, 1] - y1

    t = vx * ux + vy * uy
    foot_x = x1 + t * ux
    foot_y = y1 + t * uy

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
    """Comprehensive end-to-end analysis pipeline."""
    # 1. Segment banana foreground
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
            message="No banana detected in the image. Try adjusting the lighting or segmentation method.",
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

    # 3. Trace longest path
    ordered_points, p1, p2 = trace_longest_centerline(skeleton)
    if not ordered_points or p1 is None or p2 is None:
        return CurvatureAnalysisResult(
            success=False,
            message="Failed to trace end-to-end centerline from skeleton.",
            contour=contour,
            binary_mask=mask,
        )

    # 4. Parametric spline fitting
    centerline_points, local_curvatures = fit_centerline_spline(
        ordered_points,
        smoothing=smoothing,
        num_points=SPLINE_NUM_POINTS,
    )

    ep1 = (float(centerline_points[0, 0]), float(centerline_points[0, 1]))
    ep2 = (float(centerline_points[-1, 0]), float(centerline_points[-1, 1]))

    # 5. Compute metrics
    chord_distance = calculate_chord_distance(ep1, ep2)
    path_length = calculate_path_length(centerline_points)
    curve_score = calculate_curve_score(path_length, chord_distance)
    is_curved, category = classify_curvature(
        curve_score,
        straight_threshold=straight_threshold,
        curved_threshold=curved_threshold,
    )

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


# ===========================================================================
# 5. VISUALIZATION & DIAGNOSTICS
# ===========================================================================

def draw_dashed_line(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 10,
) -> None:
    dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if dist < 1e-3:
        return
    dashes = int(dist / dash_length)
    for i in range(0, dashes, 2):
        start_ratio = i / dashes
        end_ratio = min(1.0, (i + 1) / dashes)
        p_start = (int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio), int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio))
        p_end = (int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio), int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio))
        cv2.line(img, p_start, p_end, color, thickness, cv2.LINE_AA)


def create_annotated_overlay(
    image_rgb: np.ndarray,
    result: CurvatureAnalysisResult,
    show_contour: bool = True,
    show_centerline: bool = True,
    show_chord: bool = True,
    show_endpoints: bool = True,
    show_deflection: bool = True,
    show_info_card: bool = False,
) -> np.ndarray:
    """Renders scientific vector overlay onto input image."""
    annotated = image_rgb.copy()
    h, w = annotated.shape[:2]

    scale = max(1, int(np.sqrt(h * w) / 500))
    line_thick = max(2, scale * 2)
    point_radius = max(5, scale * 4)

    if not result.success:
        cv2.putText(annotated, result.message or "No banana detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 60, 60), 2, cv2.LINE_AA)
        return annotated

    if show_contour and result.contour is not None:
        cv2.drawContours(annotated, [result.contour], -1, (40, 220, 40), max(1, line_thick - 1), cv2.LINE_AA)

    if show_chord and result.endpoint_1 and result.endpoint_2:
        p1 = (int(round(result.endpoint_1[0])), int(round(result.endpoint_1[1])))
        p2 = (int(round(result.endpoint_2[0])), int(round(result.endpoint_2[1])))
        draw_dashed_line(annotated, p1, p2, (255, 140, 0), thickness=line_thick, dash_length=max(6, scale * 6))

    if show_centerline and result.centerline_points is not None and len(result.centerline_points) > 1:
        pts = np.round(result.centerline_points).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=False, color=(0, 215, 255), thickness=line_thick + 1, lineType=cv2.LINE_AA)

    if show_deflection and result.apex_point and result.apex_chord_foot and result.max_deflection > 2.0:
        apex = (int(round(result.apex_point[0])), int(round(result.apex_point[1])))
        foot = (int(round(result.apex_chord_foot[0])), int(round(result.apex_chord_foot[1])))
        draw_dashed_line(annotated, apex, foot, (255, 50, 215), thickness=max(1, line_thick - 1), dash_length=max(4, scale * 4))
        cv2.circle(annotated, apex, max(3, point_radius - 2), (255, 50, 215), -1, cv2.LINE_AA)

    if show_endpoints and result.endpoint_1 and result.endpoint_2:
        p1 = (int(round(result.endpoint_1[0])), int(round(result.endpoint_1[1])))
        p2 = (int(round(result.endpoint_2[0])), int(round(result.endpoint_2[1])))
        cv2.circle(annotated, p1, point_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(annotated, p1, point_radius, (34, 197, 94), -1, cv2.LINE_AA)
        cv2.circle(annotated, p2, point_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(annotated, p2, point_radius, (234, 179, 8), -1, cv2.LINE_AA)

    return annotated


def create_diagnostic_figure(
    image_rgb: np.ndarray,
    result: CurvatureAnalysisResult,
    theme: str = "light",
) -> plt.Figure:
    """Creates a 4-panel diagnostic Matplotlib figure."""
    is_light = theme == "light"
    fig_bg = "#ffffff" if is_light else "#0f172a"
    text_color = "#0f172a" if is_light else "#ffffff"
    sub_text_color = "#475569" if is_light else "#94a3b8"
    chart_bg = "#f8fafc" if is_light else "#1e293b"
    grid_color = "#e2e8f0" if is_light else "#475569"
    spine_color = "#cbd5e1" if is_light else "#334155"

    fig = plt.figure(figsize=(12, 8), facecolor=fig_bg)
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.3)

    # 1. Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_rgb)
    ax1.set_title("1. Input Image", color=text_color, fontsize=12, pad=8, fontweight="bold")
    ax1.axis("off")

    # 2. Binary Mask
    ax2 = fig.add_subplot(gs[0, 1])
    if result.binary_mask is not None:
        ax2.imshow(result.binary_mask, cmap="viridis" if is_light else "magma")
        ax2.set_title("2. Segmentation Mask", color=text_color, fontsize=12, pad=8, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No Mask Available", color="gray", ha="center", va="center")
    ax2.axis("off")

    # 3. Geometric Overlay
    ax3 = fig.add_subplot(gs[1, 0])
    annotated = create_annotated_overlay(image_rgb, result, show_info_card=False)
    ax3.imshow(annotated)
    badge = f"Curve Score: {result.curve_score:.1f}% ({result.category})"
    ax3.set_title(f"3. Geometric Overlay\n{badge}", color=text_color, fontsize=11, pad=8, fontweight="bold")
    ax3.axis("off")

    # 4. Local Curvature or Metrics Chart
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(chart_bg)

    if result.success and result.local_curvatures is not None and len(result.local_curvatures) > 0:
        norm_length = np.linspace(0, 100, len(result.local_curvatures))
        ax4.plot(norm_length, result.local_curvatures * 1000, color="#0284c7", linewidth=2.5, label="Curvature (10⁻³ px⁻¹)")
        ax4.fill_between(norm_length, result.local_curvatures * 1000, color="#38bdf8", alpha=0.25)
        ax4.set_xlabel("Normalized Centerline Path (%)", color=sub_text_color, fontsize=10, fontweight="bold")
        ax4.set_ylabel("Curvature κ (x10⁻³)", color=sub_text_color, fontsize=10, fontweight="bold")
        ax4.set_title("4. Centerline Curvature Profile", color=text_color, fontsize=12, pad=8, fontweight="bold")
        ax4.tick_params(colors=sub_text_color, labelsize=8)
        for spine in ax4.spines.values():
            spine.set_color(spine_color)
        ax4.grid(True, linestyle="--", alpha=0.6, color=grid_color)
    else:
        metrics = ["Chord Distance", "Arc Length"]
        values = [result.chord_distance, result.path_length]
        bars = ax4.bar(metrics, values, color=["#f59e0b", "#0284c7"], width=0.45)
        ax4.set_ylabel("Length (pixels)", color=sub_text_color, fontsize=10, fontweight="bold")
        ax4.set_title("4. Length Comparison", color=text_color, fontsize=12, pad=8, fontweight="bold")
        ax4.tick_params(colors=sub_text_color, labelsize=9)
        for spine in ax4.spines.values():
            spine.set_color(spine_color)
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(
                f"{height:.1f} px",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=text_color,
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    return fig


# ===========================================================================
# 6. CLI ENTRYPOINT
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="CHILL ETHAKAAA Banana Curve Analyzer CLI")
    parser.add_argument("--image", "-i", required=True, help="Path to input banana image")
    parser.add_argument("--output", "-o", default=None, help="Optional path to save annotated output image")
    parser.add_argument("--method", "-m", default="auto", choices=["auto", "hsv", "otsu", "saturation", "wood"], help="Segmentation method")
    parser.add_argument("--json", "-j", action="store_true", help="Output metrics as JSON")
    args = parser.parse_args()

    img_rgb = load_image(args.image)
    result = analyze_banana(img_rgb, segmentation_method=args.method)

    if not result.success:
        print(f"Analysis Failed: {result.message}")
        return

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("========================================")
        print("    CHILL ETHAKAAA - GEOMETRY LAB       ")
        print("========================================")
        print(f"Specimen File:      {args.image}")
        print(f"Curve Score:        {result.curve_score:.2f}%")
        print(f"Classification:     {result.category}")
        print(f"Arc Length (L):     {result.path_length:.2f} px")
        print(f"Chord Distance (D): {result.chord_distance:.2f} px")
        print(f"Max Deflection (δ): {result.max_deflection:.2f} px")
        print("========================================")

    if args.output:
        annotated_rgb = create_annotated_overlay(img_rgb, result)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output, annotated_bgr)
        print(f"Saved annotated inspection frame to: {args.output}")


if __name__ == "__main__":
    main()
