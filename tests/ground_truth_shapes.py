"""
Synthetic banana-like shapes with independently-computed ground-truth geometry.

Ground truth (arc length, chord, max deflection, Curve Score) is derived from each
shape's closed-form parametric definition using scipy.integrate.quad (arc length)
and dense independent sampling (max deflection) at 20x the pipeline's own 200-sample
spline resolution. None of this reuses calculate_path_length/calculate_max_deflection
from src.analyzer, so tests built on these shapes validate the CV + geometry pipeline
against a truth computed outside of it, not a circular self-check.
"""
from typing import Callable, Dict, Tuple
import numpy as np
import cv2
from scipy.integrate import quad


def _rasterize_tapered_centerline(
    xy: np.ndarray,
    width: int,
    height: int,
    max_thickness: int = 34,
    min_thickness: int = 14,
    background: Tuple[int, int, int] = (240, 240, 240),
    color: Tuple[int, int, int] = (250, 210, 30),
) -> np.ndarray:
    """Draw a tapered (thin-thick-thin) filled band along an ordered centerline,
    matching the taper style used by tests/test_synthetic.py's synthetic bananas."""
    img = np.full((height, width, 3), background, dtype=np.uint8)
    canvas = np.zeros((height, width), dtype=np.uint8)
    n = len(xy)
    for i, (px, py) in enumerate(xy):
        norm_pos = i / max(1, n - 1)
        thickness = int((max_thickness - min_thickness) * np.sin(np.pi * norm_pos) + min_thickness)
        cv2.circle(canvas, (int(round(px)), int(round(py))), thickness, 255, -1)
    mask = canvas > 0
    img[mask] = color
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def _analytical_geometry(
    x_of_t: Callable[[float], float],
    y_of_t: Callable[[float], float],
    dxdt: Callable[[float], float],
    dydt: Callable[[float], float],
    n_dense: int = 4000,
) -> Dict[str, float]:
    """Ground-truth arc length (quad on the analytical speed function), chord
    (closed-form endpoint distance), max perpendicular deflection from the chord
    (dense independent sampling), and the resulting Curve Score."""
    arc_length, _ = quad(lambda t: float(np.hypot(dxdt(t), dydt(t))), 0.0, 1.0, limit=200)

    x0, y0 = x_of_t(0.0), y_of_t(0.0)
    x1, y1 = x_of_t(1.0), y_of_t(1.0)
    chord = float(np.hypot(x1 - x0, y1 - y0))

    t_dense = np.linspace(0.0, 1.0, n_dense)
    xs = np.array([x_of_t(t) for t in t_dense])
    ys = np.array([y_of_t(t) for t in t_dense])

    if chord < 1e-9:
        max_deflection = 0.0
    else:
        ux, uy = (x1 - x0) / chord, (y1 - y0) / chord
        vx, vy = xs - x0, ys - y0
        proj = vx * ux + vy * uy
        foot_x, foot_y = x0 + proj * ux, y0 + proj * uy
        dists = np.hypot(xs - foot_x, ys - foot_y)
        max_deflection = float(np.max(dists))

    curve_score = ((arc_length - chord) / chord) * 100.0 if chord > 1e-9 else 0.0

    return {
        "arc_length": float(arc_length),
        "chord_distance": chord,
        "max_deflection": max_deflection,
        "curve_score": float(max(0.0, curve_score)),
    }


def straight_line_shape(length: float = 300.0, width: int = 700, height: int = 700):
    """A perfectly straight centerline. Expected Curve Score ~= 0."""
    cx, cy = width / 2.0, height / 2.0
    x0, y0 = cx, cy - length / 2.0
    x1, y1 = cx, cy + length / 2.0

    x_of_t = lambda t: x0 + t * (x1 - x0)
    y_of_t = lambda t: y0 + t * (y1 - y0)
    dxdt = lambda t: (x1 - x0)
    dydt = lambda t: (y1 - y0)

    t = np.linspace(0.0, 1.0, 300)
    xy = np.column_stack([x_of_t(t), y_of_t(t)])
    img = _rasterize_tapered_centerline(xy, width, height)
    return img, _analytical_geometry(x_of_t, y_of_t, dxdt, dydt)


def circular_arc_shape(radius: float, theta_total: float, width: int = 700, height: int = 700):
    """A circular arc of given radius and total angular span (radians)."""
    cx, cy = width / 2.0, height / 2.0

    def angle(t):
        return -theta_total / 2.0 + t * theta_total

    x_of_t = lambda t: cx + radius * np.sin(angle(t))
    y_of_t = lambda t: cy + radius * (1.0 - np.cos(angle(t)))
    dxdt = lambda t: radius * np.cos(angle(t)) * theta_total
    dydt = lambda t: radius * np.sin(angle(t)) * theta_total

    t = np.linspace(0.0, 1.0, 300)
    xy = np.column_stack([x_of_t(t), y_of_t(t)])
    img = _rasterize_tapered_centerline(xy, width, height)
    return img, _analytical_geometry(x_of_t, y_of_t, dxdt, dydt)


def quadratic_shape(a: float, half_span: float, width: int = 700, height: int = 700):
    """A parabolic arc y = a*(x-cx)^2 over x in [cx-half_span, cx+half_span]."""
    cx, cy = width / 2.0, height / 2.0
    x_of_t = lambda t: (cx - half_span) + t * (2.0 * half_span)
    y_of_t = lambda t: cy + a * (x_of_t(t) - cx) ** 2
    dxdt = lambda t: 2.0 * half_span
    dydt = lambda t: 2.0 * a * (x_of_t(t) - cx) * dxdt(t)

    t = np.linspace(0.0, 1.0, 300)
    xy = np.column_stack([x_of_t(t), y_of_t(t)])
    img = _rasterize_tapered_centerline(xy, width, height)
    return img, _analytical_geometry(x_of_t, y_of_t, dxdt, dydt)


def cubic_bezier_shape(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    width: int = 700,
    height: int = 700,
):
    """A cubic Bezier curve with the given four control points (absolute pixel coords)."""
    p0a, p1a, p2a, p3a = (np.array(p, dtype=float) for p in (p0, p1, p2, p3))

    def point(t):
        return ((1 - t) ** 3) * p0a + 3 * ((1 - t) ** 2) * t * p1a + 3 * (1 - t) * (t ** 2) * p2a + (t ** 3) * p3a

    def deriv(t):
        return (3 * (1 - t) ** 2) * (p1a - p0a) + (6 * (1 - t) * t) * (p2a - p1a) + (3 * t ** 2) * (p3a - p2a)

    x_of_t = lambda t: point(t)[0]
    y_of_t = lambda t: point(t)[1]
    dxdt = lambda t: deriv(t)[0]
    dydt = lambda t: deriv(t)[1]

    t = np.linspace(0.0, 1.0, 300)
    xy = np.array([point(tt) for tt in t])
    img = _rasterize_tapered_centerline(xy, width, height)
    return img, _analytical_geometry(x_of_t, y_of_t, dxdt, dydt)


def all_ground_truth_shapes() -> Dict[str, Tuple[np.ndarray, Dict[str, float]]]:
    """Convenience bundle of one representative shape per family, used by both
    tests/test_ground_truth.py and scripts/benchmark.py so both stay in sync."""
    return {
        "straight_line": straight_line_shape(length=300.0),
        "circular_arc_gentle": circular_arc_shape(radius=1500.0, theta_total=0.3),
        "circular_arc_moderate": circular_arc_shape(radius=220.0, theta_total=1.2),
        "circular_arc_strong": circular_arc_shape(radius=220.0, theta_total=2.4),
        "quadratic": quadratic_shape(a=0.0015, half_span=150.0),
        "cubic_bezier": cubic_bezier_shape(
            p0=(190, 410), p1=(190, 230), p2=(510, 230), p3=(510, 410)
        ),
    }
