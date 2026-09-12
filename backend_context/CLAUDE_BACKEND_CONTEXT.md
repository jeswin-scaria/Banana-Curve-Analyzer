# 🍌 CHILL ETHAKAAA — Backend Architecture & Complete Source Context

> **Banana Geometry Lab (CHILL ETHAKAAA)**  
> *"Measuring what never needed to be measured."*  
> A computer-vision and morphological analysis engine for calculating the geometric curvature of bananas.

---

## 1. Executive Summary & Mathematical Architecture

The **CHILL ETHAKAAA** backend is a modular, high-precision computer vision pipeline built with **OpenCV, NumPy, SciPy, Scikit-Image, and NetworkX**. It calculates an objective, scale-invariant **Curve Score** that quantifies how much a banana deviates from a straight line.

### Core Curvature Metric Formula
$$\text{Curve Score} = \left(\frac{L - D}{D}\right) \times 100$$

Where:
- **$L$ (Arc Length / Centerline Path Length)**: Cumulative continuous Euclidean path length along the smoothed medial-axis spline of the banana (in pixels).
- **$D$ (Chord Distance)**: Direct straight-line Euclidean distance between the two terminal endpoints (apices) of the banana (in pixels).
- **$\delta$ (Max Deflection / Sagitta)**: Maximum perpendicular distance from the straight chord line to the banana's spine.
- **Curvature Classification**:
  - $\text{Score} < 5.0\%$ $\rightarrow$ **Straight**
  - $5.0\% \le \text{Score} \le 20.0\%$ $\rightarrow$ **Curved**
  - $\text{Score} > 20.0\%$ $\rightarrow$ **Highly Curved**

---

## 2. 6-Stage Computer Vision Pipeline

```
[ Input RGB Image ]
        │
        ▼
[ Stage 1: Smart Saliency & Segmentation ]
  • CIE LAB b* (yellow) & a* (wood grain rejection)
  • HSV chrominance gating (Hue 15°-85°)
  • Otsu dynamic thresholding
        │
        ▼
[ Stage 2: Morphological Cleanup ]
  • Morphological Opening (removes table reflection bridges)
  • Morphological Closing (heals brown spots/blemishes)
  • External contour isolation (picks largest area)
        │
        ▼
[ Stage 3: Topological Skeletonization ]
  • Medial axis thinning (1-pixel wide skeleton)
  • 8-connected NetworkX graph construction
  • Dijkstra / Geodesic longest path tracing (prunes spurs)
        │
        ▼
[ Stage 4: Parametric B-Spline Smoothing ]
  • Parametric B-spline fit via scipy.interpolate.splprep
  • Eliminates discrete pixel-grid staircasing
  • Analytic 1st & 2nd derivatives for local curvature κ
        │
        ▼
[ Stage 5: Metric Extraction ]
  • Numerical path integration -> Arc Length (L)
  • Terminal endpoints -> Chord Distance (D)
  • Perpendicular projection -> Max Deflection (δ)
        │
        ▼
[ Stage 6: Scoring & Output Generation ]
  • Curve Score calculation
  • Curvature classification
  • Annotated vector overlay & Matplotlib diagnostics
```

---

## 3. Directory Layout

```
Banana-Curve-Analyzer/
├── src/
│   ├── __init__.py           # Package exports & public API
│   ├── config.py             # System thresholds, constants & color palettes
│   ├── preprocessing.py      # Image I/O & multi-space segmentation
│   ├── skeleton.py           # Morphological skeletonization & graph pathfinding
│   ├── analyzer.py           # Spline fitting, geometric metrics & scoring
│   ├── visualization.py      # Vector overlay rendering & diagnostic plots
│   └── cli.py                # Command-line interface
├── samples/                  # Curvature test images (straight, curved, highly curved)
├── tests/                    # Unit & synthetic test suites
├── app.py                    # Streamlit Laboratory Web Application
└── requirements.txt          # Python dependencies
```

---

## 4. Complete Backend Source Code

---

### `requirements.txt`
```text
streamlit>=1.30.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
scipy>=1.11.0
scikit-image>=0.21.0
matplotlib>=3.7.0
pillow>=10.0.0
networkx>=3.1
```

---

### `src/__init__.py`
```python
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
from .preprocessing import load_image, segment_banana
from .skeleton import extract_skeleton, trace_longest_centerline
from .analyzer import (
    CurvatureAnalysisResult,
    analyze_banana,
    calculate_chord_distance,
    calculate_path_length,
    calculate_curve_score,
    classify_curvature,
)
from .visualization import create_annotated_overlay, create_diagnostic_figure

__all__ = [
    "STRAIGHT_THRESHOLD",
    "CURVED_THRESHOLD",
    "CATEGORY_STRAIGHT",
    "CATEGORY_CURVED",
    "CATEGORY_HIGHLY_CURVED",
    "load_image",
    "segment_banana",
    "extract_skeleton",
    "trace_longest_centerline",
    "CurvatureAnalysisResult",
    "analyze_banana",
    "calculate_chord_distance",
    "calculate_path_length",
    "calculate_curve_score",
    "classify_curvature",
    "create_annotated_overlay",
    "create_diagnostic_figure",
]
```

---

### `src/config.py`
```python
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
```

---

### `src/preprocessing.py`
```python
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
```

---

### `src/skeleton.py`
```python
"""
Skeletonization and centerline extraction module for Banana Curve Analyzer.
Extracts the topological medial axis and traces the ordered tip-to-tip centerline.
"""

from typing import List, Tuple, Optional
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize


def extract_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    Compute the 1-pixel wide morphological skeleton of the binary mask.

    Args:
        binary_mask: Binary mask image (uint8, with values 0 and 255).

    Returns:
        np.ndarray: Skeleton image as uint8 (0 and 255).
    """
    bool_mask = binary_mask > 0
    if not np.any(bool_mask):
        return np.zeros_like(binary_mask, dtype=np.uint8)

    skel = skeletonize(bool_mask)
    return (skel.astype(np.uint8)) * 255


def build_skeleton_graph(skeleton: np.ndarray) -> nx.Graph:
    """
    Build a networkx graph from the 8-connected skeleton pixels.
    Edge weights represent Euclidean distance between adjacent pixels (1.0 or sqrt(2)).

    Args:
        skeleton: Skeleton image (uint8, 0 and 255).

    Returns:
        nx.Graph: Graph where nodes are (row, col) tuples.
    """
    g = nx.Graph()
    rows, cols = np.where(skeleton > 0)
    pixel_set = set(zip(rows, cols))

    # 8-neighborhood offsets and Euclidean step lengths
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
    """
    Find the longest geodesic path through the skeleton graph, effectively
    pruning minor spurs and selecting the true tip-to-tip banana centerline.

    Args:
        skeleton: Binary skeleton image (uint8).

    Returns:
        Tuple:
            - ordered_points: List of (x, y) coordinates in order from tip 1 to tip 2.
            - endpoint_1: (x, y) coordinates of the first endpoint.
            - endpoint_2: (x, y) coordinates of the second endpoint.
    """
    if np.count_nonzero(skeleton) < 2:
        return [], None, None

    g = build_skeleton_graph(skeleton)
    if len(g) < 2:
        return [], None, None

    # Focus on the largest connected component
    largest_cc = max(nx.connected_components(g), key=len)
    subg = g.subgraph(largest_cc).copy()

    # Find endpoints (degree == 1)
    endpoints = [node for node in subg.nodes() if subg.degree(node) == 1]

    best_path = []
    if len(endpoints) >= 2:
        # Find the pair of endpoints that has the longest shortest path
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
        # Single endpoint (e.g. lasso or tail): find farthest node from that endpoint
        u = endpoints[0]
        lengths = nx.single_source_dijkstra_path_length(subg, u, weight="weight")
        farthest_node = max(lengths, key=lengths.get)
        best_path = nx.shortest_path(subg, u, farthest_node, weight="weight")
    else:
        # No degree 1 nodes (e.g. a loop): find two nodes with maximum distance
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

    # Convert nodes from (row, col) to standard Cartesian (x, y) = (col, row)
    ordered_points = [(float(c), float(r)) for (r, c) in best_path]
    endpoint_1 = ordered_points[0]
    endpoint_2 = ordered_points[-1]

    return ordered_points, endpoint_1, endpoint_2
```

---

### `src/analyzer.py`
```python
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
        safe_denom = np.where(denominator < 1e-9, 1e-9, denominator)
        kappa = numerator / safe_denom

        sampled = np.column_stack([x_spl, y_spl])
        return sampled, kappa
    except Exception:
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
    """
    End-to-end banana curvature analysis pipeline.
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
```

---

### `src/visualization.py`
```python
"""
Visualization module for Banana Curve Analyzer.
Provides high-clarity OpenCV image overlays and multi-panel Matplotlib diagnostic figures.
"""

from typing import Optional, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .config import COLORS, CATEGORY_COLORS, CATEGORY_STRAIGHT, CATEGORY_CURVED, CATEGORY_HIGHLY_CURVED
from .analyzer import CurvatureAnalysisResult


def draw_dashed_line(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 10,
) -> None:
    """Draw a dashed line between two points in-place on an OpenCV image."""
    dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if dist < 1e-3:
        return
    dashes = int(dist / dash_length)
    for i in range(0, dashes, 2):
        start_ratio = i / dashes
        end_ratio = min(1.0, (i + 1) / dashes)
        p_start = (
            int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio),
            int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio),
        )
        p_end = (
            int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio),
            int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio),
        )
        cv2.line(img, p_start, p_end, color, thickness, cv2.LINE_AA)


def create_annotated_overlay(
    image_rgb: np.ndarray,
    result: CurvatureAnalysisResult,
    show_contour: bool = True,
    show_centerline: bool = True,
    show_chord: bool = True,
    show_endpoints: bool = True,
    show_deflection: bool = True,
    show_info_card: bool = True,
) -> np.ndarray:
    """
    Produce an annotated RGB image highlighting detected geometry:
    - Green outer contour of the banana
    - Glowing cyan centerline
    - Dashed orange chord (straight-line distance)
    - Distinct colored endpoints
    - Maximum deflection (sagitta) line
    """
    annotated = image_rgb.copy()
    h, w = annotated.shape[:2]

    scale = max(1, int(np.sqrt(h * w) / 500))
    line_thick = max(2, scale * 2)
    point_radius = max(5, scale * 4)

    if not result.success:
        cv2.putText(
            annotated,
            result.message or "No banana detected",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 60, 60),
            2,
            cv2.LINE_AA,
        )
        return annotated

    # 1. Draw contour outline
    if show_contour and result.contour is not None:
        cv2.drawContours(annotated, [result.contour], -1, (40, 220, 40), max(1, line_thick - 1), cv2.LINE_AA)

    # 2. Draw straight chord line
    if show_chord and result.endpoint_1 and result.endpoint_2:
        p1 = (int(round(result.endpoint_1[0])), int(round(result.endpoint_1[1])))
        p2 = (int(round(result.endpoint_2[0])), int(round(result.endpoint_2[1])))
        draw_dashed_line(annotated, p1, p2, (255, 140, 0), thickness=line_thick, dash_length=max(6, scale * 6))

    # 3. Draw centerline curve
    if show_centerline and result.centerline_points is not None and len(result.centerline_points) > 1:
        pts = np.round(result.centerline_points).astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=False, color=(0, 215, 255), thickness=line_thick + 1, lineType=cv2.LINE_AA)

    # 4. Draw maximum deflection line
    if show_deflection and result.apex_point and result.apex_chord_foot and result.max_deflection > 2.0:
        apex = (int(round(result.apex_point[0])), int(round(result.apex_point[1])))
        foot = (int(round(result.apex_chord_foot[0])), int(round(result.apex_chord_foot[1])))
        draw_dashed_line(annotated, apex, foot, (255, 50, 215), thickness=max(1, line_thick - 1), dash_length=max(4, scale * 4))
        cv2.circle(annotated, apex, max(3, point_radius - 2), (255, 50, 215), -1, cv2.LINE_AA)

    # 5. Draw endpoints
    if show_endpoints and result.endpoint_1 and result.endpoint_2:
        p1 = (int(round(result.endpoint_1[0])), int(round(result.endpoint_1[1])))
        p2 = (int(round(result.endpoint_2[0])), int(round(result.endpoint_2[1])))
        cv2.circle(annotated, p1, point_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(annotated, p1, point_radius, (34, 197, 94), -1, cv2.LINE_AA)
        cv2.circle(annotated, p2, point_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(annotated, p2, point_radius, (234, 179, 8), -1, cv2.LINE_AA)

    # 6. Draw semi-transparent information card in corner
    if show_info_card:
        card_w, card_h = int(min(320, w * 0.45)), int(min(140, h * 0.35))
        overlay = annotated.copy()
        cv2.rectangle(overlay, (15, 15), (15 + card_w, 15 + card_h), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (15, 15), (15 + card_w, 15 + card_h), (60, 60, 70), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        line_spacing = 24

        cat_color = (34, 197, 94) if result.category == CATEGORY_STRAIGHT else (
            (234, 179, 8) if result.category == CATEGORY_CURVED else (239, 68, 68)
        )

        cv2.putText(annotated, f"Curve Score: {result.curve_score:.2f}%", (25, 42), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Category: {result.category}", (25, 42 + line_spacing), font, font_scale, cat_color, 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Path Length: {result.path_length:.1f} px", (25, 42 + line_spacing * 2), font, font_scale - 0.1, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Chord Dist:  {result.chord_distance:.1f} px", (25, 42 + line_spacing * 3), font, font_scale - 0.1, (200, 200, 200), 1, cv2.LINE_AA)

    return annotated


def create_diagnostic_figure(
    image_rgb: np.ndarray,
    result: CurvatureAnalysisResult,
    theme: str = "light",
) -> plt.Figure:
    """
    Generate a 4-panel diagnostic Matplotlib figure displaying:
    1. Original Image
    2. Binary Segmentation Mask
    3. Centerline & Chord Geometric Overlay
    4. Local Curvature Profile along path length
    """
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
```

---

### `src/cli.py`
```python
"""
Command-line interface (CLI) for Banana Curve Analyzer.
Usage:
    python -m src.cli --image samples/curved_banana.png
    python -m src.cli --image samples/curved_banana.png --output annotated.png
"""

import argparse
import json
import cv2
from .preprocessing import load_image
from .analyzer import analyze_banana
from .visualization import create_annotated_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Banana Curve Analyzer CLI")
    parser.add_argument("--image", "-i", required=True, help="Path to input banana image")
    parser.add_argument("--output", "-o", default=None, help="Optional path to save annotated output image")
    parser.add_argument("--method", "-m", default="auto", choices=["auto", "hsv", "otsu", "saturation"], help="Segmentation method")
    parser.add_argument("--json", "-j", action="store_true", help="Print results formatted as JSON")
    args = parser.parse_args()

    img_rgb = load_image(args.image)
    result = analyze_banana(img_rgb, segmentation_method=args.method)

    if not result.success:
        print(f"Error: {result.message}")
        return

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("========================================")
        print("        BANANA CURVE ANALYZER           ")
        print("========================================")
        print(f"File:               {args.image}")
        print(f"Curve Score:        {result.curve_score:.2f}%")
        print(f"Curvature Category: {result.category}")
        print(f"Is Curved:          {'Yes' if result.is_curved else 'No'}")
        print(f"Path Length (L):    {result.path_length:.2f} px")
        print(f"Chord Distance (D): {result.chord_distance:.2f} px")
        print(f"Max Deflection:     {result.max_deflection:.2f} px")
        print(f"Deflection Ratio:   {result.deflection_ratio:.4f}")
        print("========================================")

    if args.output:
        annotated_rgb = create_annotated_overlay(img_rgb, result)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output, annotated_bgr)
        print(f"Annotated output image saved to: {args.output}")


if __name__ == "__main__":
    main()
```
