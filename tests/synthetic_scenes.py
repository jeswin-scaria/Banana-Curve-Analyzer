"""
Composite adversarial scenes for segmentation-robustness testing.
"""
from typing import Dict, Optional, Tuple
import numpy as np
import cv2

from tests.test_synthetic import generate_synthetic_banana_image


def make_glossy_banana_scene(
    width: int = 600,
    height: int = 800,
    on_wood: bool = True,
    with_alpha: bool = False,
    seed: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    A studio/phone-style banana with a strong SPECULAR HIGHLIGHT running down
    its body, plus dark ripeness spots -- the two features that tear a
    colour-thresholded mask apart on real photographs. Optionally placed on a
    warm oak background (whose colour overlaps banana-yellow) or given a real
    alpha channel, as a cut-out product PNG would have.

    Returns (image_rgb, true_body_mask, alpha_or_None).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    if on_wood:
        img = np.dstack([
            np.full((height, width), 206.0), np.full((height, width), 166.0),
            np.full((height, width), 116.0),
        ]).astype(np.float32)
        img += (9.0 * np.sin(xx / 24.0) + 5.0 * np.sin(xx / 6.5 + 1.4 * np.sin(yy / 80.0)))[..., None]
        img += rng.normal(0, 3.0, img.shape)
    else:
        img = np.full((height, width, 3), 255.0, np.float32)

    t = np.linspace(0, 1, 320)
    ang = np.linspace(-0.72, 0.72, 320)
    r = 330.0
    bx = width * 0.50 - r * (1 - np.cos(ang)) + 80
    by = height * 0.50 + r * np.sin(ang) * 0.80

    body = np.zeros((height, width), np.float32)
    for i in range(len(t)):
        thick = 38.0 * np.sin(np.pi * t[i]) ** 0.6 + 6.0
        cv2.circle(body, (int(bx[i]), int(by[i])), int(thick), 1.0, -1)
    body = cv2.GaussianBlur(body, (7, 7), 0)
    body_mask = body > 0.5

    dist = cv2.distanceTransform(body_mask.astype(np.uint8), cv2.DIST_L2, 5)
    dn = dist / (dist.max() + 1e-6)
    banana = np.zeros_like(img)
    edge_col, core_col = np.array([218.0, 165.0, 44.0]), np.array([253.0, 225.0, 80.0])
    for c in range(3):
        banana[..., c] = edge_col[c] + (core_col[c] - edge_col[c]) * np.clip(dn * 1.6, 0, 1)

    # specular highlight: desaturates a wide band to near-white
    hl = np.zeros((height, width), np.float32)
    for i in range(len(t)):
        thick = 38.0 * np.sin(np.pi * t[i]) ** 0.6 + 6.0
        cv2.circle(hl, (int(bx[i] - thick * 0.32), int(by[i] - thick * 0.18)),
                   max(2, int(thick * 0.42)), 1.0, -1)
    hl = cv2.GaussianBlur(hl, (31, 31), 0)
    hl = np.clip(hl / (hl.max() + 1e-6), 0, 1) * body_mask
    for c in range(3):
        banana[..., c] = banana[..., c] * (1 - 0.93 * hl) + 255.0 * (0.93 * hl)

    # dark ripeness spots
    spots = np.zeros((height, width), np.float32)
    for _ in range(70):
        i = int(rng.integers(30, len(t) - 30))
        thick = 38.0 * np.sin(np.pi * t[i]) ** 0.6 + 6.0
        px = int(bx[i] + rng.normal(0, thick * 0.4))
        py = int(by[i] + rng.normal(0, thick * 0.4))
        cv2.circle(spots, (px, py), int(abs(rng.normal(3.0, 1.8)) + 1), 1.0, -1)
    spots = cv2.GaussianBlur(spots, (5, 5), 0) * body_mask
    for c in range(3):
        banana[..., c] = banana[..., c] * (1 - spots) + np.array([86.0, 55.0, 26.0])[c] * spots

    img[body_mask] = banana[body_mask]
    img = np.clip(img, 0, 255).astype(np.uint8)

    alpha = None
    if with_alpha:
        alpha = (body_mask.astype(np.uint8)) * 255
        # emulate an exporter that leaves black under fully-transparent pixels
        img[~body_mask] = 0
    return img, body_mask, alpha


def make_adversarial_rectangle_scene(
    canvas_width: int = 900,
    canvas_height: int = 500,
) -> Tuple[np.ndarray, Dict]:
    """
    Composite scene with a LARGE, non-tapered yellow rectangle (high extent ~1.0,
    no taper -> implausible banana shape) placed in a disjoint region from a
    SMALLER genuinely tapered, curved banana shape. The rectangle's area is
    deliberately larger than the banana's, so a segmentation stage that blindly
    trusts "largest contour wins" picks the rectangle instead of the banana.
    """
    img = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)

    # Large rectangle on the left: high extent, no taper.
    rect_w, rect_h = 170, 400
    rect_x0, rect_y0 = 60, (canvas_height - rect_h) // 2
    rect_x1, rect_y1 = rect_x0 + rect_w, rect_y0 + rect_h
    cv2.rectangle(img, (rect_x0, rect_y0), (rect_x1, rect_y1), (250, 210, 30), -1)
    rect_area = rect_w * rect_h

    # Smaller tapered, curved banana on the right.
    banana_region_w, banana_region_h = 420, canvas_height
    banana_img = generate_synthetic_banana_image(curvature="curved", width=banana_region_w, height=banana_region_h)
    x_offset = canvas_width - banana_region_w
    region = img[0:banana_region_h, x_offset:x_offset + banana_region_w]
    banana_pixels = np.any(banana_img < 235, axis=-1)
    region[banana_pixels] = banana_img[banana_pixels]

    banana_ys, banana_xs = np.where(banana_pixels)
    banana_bbox = (
        int(banana_xs.min()) + x_offset, int(banana_ys.min()),
        int(banana_xs.max()) + x_offset, int(banana_ys.max()),
    )
    banana_area_estimate = int(banana_pixels.sum())

    info = {
        "rectangle_bbox": (rect_x0, rect_y0, rect_x1, rect_y1),
        "rectangle_area": rect_area,
        "banana_bbox": banana_bbox,
        "banana_area_estimate": banana_area_estimate,
    }
    return img, info
