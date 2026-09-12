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


def compute_center_of_curvature(
    p1: Tuple[float, float],
    apex: Tuple[float, float],
    p2: Tuple[float, float],
) -> Tuple[Optional[Tuple[float, float]], float]:
    """
    Calculate center of curvature (circumcircle center of P1, Apex, P2) and radius R.
    Returns ((xc, yc), R).
    """
    x1, y1 = p1
    xa, ya = apex
    x2, y2 = p2

    D = 2.0 * (x1 * (ya - y2) + xa * (y2 - y1) + x2 * (y1 - ya))
    if abs(D) < 1e-5:
        return None, 0.0

    sq1 = x1**2 + y1**2
    sqa = xa**2 + ya**2
    sq2 = x2**2 + y2**2

    xc = (sq1 * (ya - y2) + sqa * (y2 - y1) + sq2 * (y1 - ya)) / D
    yc = (sq1 * (x2 - xa) + sqa * (x1 - x2) + sq2 * (xa - x1)) / D
    R = float(np.hypot(xa - xc, ya - yc))

    return (xc, yc), R


def create_annotated_overlay(
    image_rgb: np.ndarray,
    result: CurvatureAnalysisResult,
    show_contour: bool = True,
    show_centerline: bool = True,
    show_chord: bool = True,
    show_endpoints: bool = True,
    show_deflection: bool = True,
    show_info_card: bool = True,
    show_curvature_center: bool = True,
) -> np.ndarray:
    """
    Produce an annotated RGB image highlighting detected geometry:
    - Green outer contour of the banana
    - Glowing cyan centerline
    - Dashed orange chord (straight-line distance)
    - Center of Curvature (C) & Radius vector (R)
    - Distinct colored endpoints
    - Maximum deflection (sagitta) line
    - On-screen summary HUD card
    """
    annotated = image_rgb.copy()
    h, w = annotated.shape[:2]

    # Dynamic line and marker scaling based on image resolution
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

    # 4. Draw Center of Curvature & Radius R
    if show_curvature_center and result.endpoint_1 and result.endpoint_2 and result.apex_point:
        c_pt, R = compute_center_of_curvature(result.endpoint_1, result.apex_point, result.endpoint_2)
        if c_pt is not None and 10 < R < max(h, w) * 4:
            xc, yc = int(round(c_pt[0])), int(round(c_pt[1]))
            apex_i = (int(round(result.apex_point[0])), int(round(result.apex_point[1])))

            # Draw Osculating Arc/Circle
            if 0 <= xc < w and 0 <= yc < h:
                cv2.circle(annotated, (xc, yc), int(round(R)), (255, 215, 0), max(1, line_thick - 1), cv2.LINE_AA)

            # Draw Radius Vector R
            if 0 <= xc < w*2 and 0 <= yc < h*2:
                cv2.line(annotated, (xc, yc), apex_i, (255, 215, 0), line_thick, cv2.LINE_AA)
                cv2.circle(annotated, (xc, yc), point_radius, (255, 215, 0), -1, cv2.LINE_AA)
                cv2.putText(annotated, "C", (xc + 8, yc - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 215, 0), 2, cv2.LINE_AA)

    # 5. Draw maximum deflection line
    if show_deflection and result.apex_point and result.apex_chord_foot and result.max_deflection > 2.0:
        apex = (int(round(result.apex_point[0])), int(round(result.apex_point[1])))
        foot = (int(round(result.apex_chord_foot[0])), int(round(result.apex_chord_foot[1])))
        draw_dashed_line(annotated, apex, foot, (255, 50, 215), thickness=max(1, line_thick - 1), dash_length=max(4, scale * 4))
        cv2.circle(annotated, apex, max(3, point_radius - 2), (255, 50, 215), -1, cv2.LINE_AA)

    # 6. Draw endpoints
    if show_endpoints and result.endpoint_1 and result.endpoint_2:
        p1 = (int(round(result.endpoint_1[0])), int(round(result.endpoint_1[1])))
        p2 = (int(round(result.endpoint_2[0])), int(round(result.endpoint_2[1])))
        cv2.circle(annotated, p1, point_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(annotated, p1, point_radius, (34, 197, 94), -1, cv2.LINE_AA)
        cv2.circle(annotated, p2, point_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(annotated, p2, point_radius, (234, 179, 8), -1, cv2.LINE_AA)

    # 7. Draw semi-transparent information card in corner
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


def create_isolated_shape_overlay(
    image_rgb: np.ndarray,
    result: CurvatureAnalysisResult,
    show_contour: bool = True,
    show_centerline: bool = True,
    show_chord: bool = True,
    show_endpoints: bool = True,
    show_deflection: bool = True,
    show_curvature_center: bool = True,
) -> np.ndarray:
    """
    Produce a clean isolated geometric shape overlay:
    - Removes all background photos, table, pen, and noise
    - Renders pure banana shape/mask silhouette on sleek dark canvas
    - Overlays Center of Curvature (C), Radius R, Osculating Circle, Chord (D), Centerline, and Endpoints
    """
    h, w = image_rgb.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (15, 20, 30)  # Sleek dark navy laboratory background

    if not result.success:
        return canvas

    # Render isolated banana shape mask silhouette
    if result.binary_mask is not None:
        mask_bool = result.binary_mask > 0
        canvas[mask_bool] = (40, 50, 70)
        shape_fill = np.zeros_like(canvas)
        shape_fill[mask_bool] = (244, 212, 100)  # Gold banana silhouette fill
        cv2.addWeighted(shape_fill, 0.28, canvas, 0.72, 0, canvas)

    scale = max(1, int(np.sqrt(h * w) / 500))
    line_thick = max(2, scale * 2)
    point_radius = max(5, scale * 4)

    # 1. Draw contour outline
    if show_contour and result.contour is not None:
        cv2.drawContours(canvas, [result.contour], -1, (34, 197, 94), max(2, line_thick), cv2.LINE_AA)

    # 2. Draw Center of Curvature (C) & Radius (R) & Osculating Circle
    if show_curvature_center and result.endpoint_1 and result.endpoint_2 and result.apex_point:
        c_pt, R = compute_center_of_curvature(result.endpoint_1, result.apex_point, result.endpoint_2)
        if c_pt is not None and 10 < R < max(h, w) * 5:
            xc, yc = int(round(c_pt[0])), int(round(c_pt[1]))
            apex_i = (int(round(result.apex_point[0])), int(round(result.apex_point[1])))

            # Draw Osculating Circle
            cv2.circle(canvas, (xc, yc), int(round(R)), (250, 204, 21), max(1, line_thick - 1), cv2.LINE_AA)
            
            # Draw Radius Line R
            cv2.line(canvas, (xc, yc), apex_i, (250, 204, 21), line_thick, cv2.LINE_AA)
            
            # Draw Center Marker C
            cv2.circle(canvas, (xc, yc), point_radius + 2, (250, 204, 21), -1, cv2.LINE_AA)
            
            # Labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_s = 0.5 * scale
            cv2.putText(canvas, f"Center of Curvature (C)", (xc + 10, yc - 10), font, font_s, (250, 204, 21), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"Radius R={R:.0f}px", (int((xc + apex_i[0]) / 2) + 6, int((yc + apex_i[1]) / 2)), font, font_s - 0.05, (250, 204, 21), 1, cv2.LINE_AA)

    # 3. Draw straight chord line (D)
    if show_chord and result.endpoint_1 and result.endpoint_2:
        p1_i = (int(round(result.endpoint_1[0])), int(round(result.endpoint_1[1])))
        p2_i = (int(round(result.endpoint_2[0])), int(round(result.endpoint_2[1])))
        draw_dashed_line(canvas, p1_i, p2_i, (249, 115, 22), thickness=line_thick, dash_length=max(6, scale * 6))
        mid_x, mid_y = int((p1_i[0] + p2_i[0]) / 2), int((p1_i[1] + p2_i[1]) / 2)
        cv2.putText(canvas, f"Chord D={result.chord_distance:.0f}px", (mid_x - 35, mid_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48 * scale, (249, 115, 22), 2, cv2.LINE_AA)

    # 4. Draw centerline curve
    if show_centerline and result.centerline_points is not None and len(result.centerline_points) > 1:
        pts = np.round(result.centerline_points).astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=(6, 182, 212), thickness=line_thick + 2, lineType=cv2.LINE_AA)

    # 5. Draw maximum deflection line
    if show_deflection and result.apex_point and result.apex_chord_foot and result.max_deflection > 2.0:
        apex = (int(round(result.apex_point[0])), int(round(result.apex_point[1])))
        foot = (int(round(result.apex_chord_foot[0])), int(round(result.apex_chord_foot[1])))
        draw_dashed_line(canvas, apex, foot, (236, 72, 153), thickness=line_thick, dash_length=max(4, scale * 4))
        cv2.circle(canvas, apex, max(3, point_radius - 2), (236, 72, 153), -1, cv2.LINE_AA)

    # 6. Draw endpoints
    if show_endpoints and result.endpoint_1 and result.endpoint_2:
        p1_i = (int(round(result.endpoint_1[0])), int(round(result.endpoint_1[1])))
        p2_i = (int(round(result.endpoint_2[0])), int(round(result.endpoint_2[1])))
        cv2.circle(canvas, p1_i, point_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, p1_i, point_radius, (34, 197, 94), -1, cv2.LINE_AA)
        cv2.circle(canvas, p2_i, point_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, p2_i, point_radius, (234, 179, 8), -1, cv2.LINE_AA)

    return canvas


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
