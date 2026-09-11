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
    - On-screen summary HUD card

    Args:
        image_rgb: Input RGB image.
        result: Analysis result dataclass.
        show_contour: Whether to draw contour outline.
        show_centerline: Whether to draw centerline path.
        show_chord: Whether to draw straight chord line.
        show_endpoints: Whether to draw endpoint markers.
        show_deflection: Whether to draw deflection line.
        show_info_card: Whether to draw the top-left summary card.

    Returns:
        np.ndarray: Annotated RGB image.
    """
    annotated = image_rgb.copy()
    h, w = annotated.shape[:2]

    # Dynamic line and marker scaling based on image resolution
    scale = max(1, int(np.sqrt(h * w) / 500))
    line_thick = max(2, scale * 2)
    point_radius = max(5, scale * 4)

    if not result.success:
        # Draw error notification banner
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
        # Outer rings
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
) -> plt.Figure:
    """
    Generate a 4-panel diagnostic Matplotlib figure displaying:
    1. Original Image
    2. Binary Segmentation Mask
    3. Centerline & Chord Geometric Overlay
    4. Local Curvature Profile along path length
    """
    fig = plt.figure(figsize=(12, 8), facecolor="#0f172a")
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.3)

    # 1. Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_rgb)
    ax1.set_title("1. Input Image", color="white", fontsize=12, pad=8)
    ax1.axis("off")

    # 2. Binary Mask
    ax2 = fig.add_subplot(gs[0, 1])
    if result.binary_mask is not None:
        ax2.imshow(result.binary_mask, cmap="magma")
        ax2.set_title("2. Segmentation Mask", color="white", fontsize=12, pad=8)
    else:
        ax2.text(0.5, 0.5, "No Mask Available", color="gray", ha="center", va="center")
    ax2.axis("off")

    # 3. Geometric Overlay
    ax3 = fig.add_subplot(gs[1, 0])
    annotated = create_annotated_overlay(image_rgb, result, show_info_card=False)
    ax3.imshow(annotated)
    badge = f"Curve Score: {result.curve_score:.1f}% ({result.category})"
    ax3.set_title(f"3. Geometric Overlay\n{badge}", color="white", fontsize=11, pad=8)
    ax3.axis("off")

    # 4. Local Curvature or Metrics Chart
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#1e293b")

    if result.success and result.local_curvatures is not None and len(result.local_curvatures) > 0:
        norm_length = np.linspace(0, 100, len(result.local_curvatures))
        ax4.plot(norm_length, result.local_curvatures * 1000, color="#38bdf8", linewidth=2.0, label="Curvature (10⁻³ px⁻¹)")
        ax4.fill_between(norm_length, result.local_curvatures * 1000, color="#38bdf8", alpha=0.2)
        ax4.set_xlabel("Normalized Centerline Path (%)", color="#94a3b8", fontsize=10)
        ax4.set_ylabel("Curvature κ (x10⁻³)", color="#94a3b8", fontsize=10)
        ax4.set_title("4. Centerline Curvature Profile", color="white", fontsize=12, pad=8)
        ax4.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax4.spines.values():
            spine.set_color("#334155")
        ax4.grid(True, linestyle="--", alpha=0.3, color="#475569")
    else:
        # Bar chart comparing Chord vs Arc Length
        metrics = ["Chord Distance", "Arc Length"]
        values = [result.chord_distance, result.path_length]
        bars = ax4.bar(metrics, values, color=["#f59e0b", "#06b6d4"], width=0.45)
        ax4.set_ylabel("Length (pixels)", color="#94a3b8", fontsize=10)
        ax4.set_title("4. Length Comparison", color="white", fontsize=12, pad=8)
        ax4.tick_params(colors="#94a3b8", labelsize=9)
        for spine in ax4.spines.values():
            spine.set_color("#334155")
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(
                f"{height:.1f} px",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="white",
                fontsize=9,
            )

    plt.tight_layout()
    return fig
