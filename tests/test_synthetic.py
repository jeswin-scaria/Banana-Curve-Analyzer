"""
End-to-end integration tests using synthetically generated banana shapes.
Clearly labeled as synthetic geometry tests for pipeline validation.
"""

import unittest
import numpy as np
import cv2
from src.analyzer import analyze_banana
from src.config import CATEGORY_STRAIGHT, CATEGORY_CURVED, CATEGORY_HIGHLY_CURVED


def generate_synthetic_banana_image(
    curvature: str = "curved",
    width: int = 500,
    height: int = 500,
) -> np.ndarray:
    """
    Generate a clean synthetic banana-like yellow shape on a neutral background
    for automated testing and regression verification.

    Args:
        curvature: 'straight', 'curved', or 'highly_curved'
        width: Image width
        height: Image height

    Returns:
        np.ndarray: RGB image (uint8).
    """
    # Neutral white/light-gray background
    img = np.full((height, width, 3), 240, dtype=np.uint8)

    # Centerline definition
    cx, cy = width // 2, height // 2

    if curvature == "straight":
        # Straight vertical rod
        y = np.linspace(cy - 160, cy + 160, 200)
        x = np.full_like(y, cx)
    elif curvature == "curved":
        # Moderate circular arc (~70 degrees)
        r = 220.0
        angles = np.linspace(-0.6, 0.6, 200)
        x = cx - r * (1 - np.cos(angles)) + 40
        y = cy + r * np.sin(angles)
    else:  # highly_curved
        # Strong arc (~130 degrees)
        r = 170.0
        angles = np.linspace(-1.15, 1.15, 200)
        x = cx - r * (1 - np.cos(angles)) + 80
        y = cy + r * np.sin(angles)

    pts = np.column_stack([x, y]).astype(np.int32)

    # Draw tapered banana body using varying circle thicknesses
    canvas = np.zeros((height, width), dtype=np.uint8)
    n = len(pts)
    for i, (px, py) in enumerate(pts):
        # Taper at tips, thick in center
        norm_pos = i / (n - 1)  # 0 to 1
        thickness = int(32 * np.sin(np.pi * norm_pos) + 6)
        cv2.circle(canvas, (int(px), int(py)), thickness, 255, -1)

    # Fill yellow banana color (RGB: approx (250, 215, 20) with slight green tips)
    yellow_rgb = np.array([250, 210, 30], dtype=np.uint8)
    mask = canvas > 0
    img[mask] = yellow_rgb

    # Add slight green tint at one tip (stem)
    stem_mask = np.zeros_like(canvas, dtype=bool)
    cv2.circle(canvas, (int(pts[0, 0]), int(pts[0, 1])), 10, 255, -1)
    img[canvas > 200] = np.array([80, 160, 40], dtype=np.uint8)

    # Smooth borders with Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


class TestSyntheticPipeline(unittest.TestCase):
    """Integration test suite verifying the end-to-end CV pipeline on synthetic shapes."""

    def test_pipeline_on_straight_shape(self):
        """Pipeline should detect low curvature on a straight synthetic shape."""
        img = generate_synthetic_banana_image(curvature="straight")
        result = analyze_banana(img, segmentation_method="auto")

        self.assertTrue(result.success, f"Pipeline failed: {result.message}")
        self.assertIsNotNone(result.contour)
        self.assertIsNotNone(result.centerline_points)
        self.assertIsNotNone(result.endpoint_1)
        self.assertIsNotNone(result.endpoint_2)
        self.assertLess(result.curve_score, 5.0)
        self.assertEqual(result.category, CATEGORY_STRAIGHT)

    def test_pipeline_on_curved_shape(self):
        """Pipeline should detect moderate curvature on a curved synthetic shape."""
        img = generate_synthetic_banana_image(curvature="curved")
        result = analyze_banana(img, segmentation_method="auto")

        self.assertTrue(result.success, f"Pipeline failed: {result.message}")
        self.assertTrue(result.is_curved)
        self.assertGreaterEqual(result.curve_score, 5.0)
        self.assertLessEqual(result.curve_score, 20.0)
        self.assertEqual(result.category, CATEGORY_CURVED)

    def test_pipeline_on_highly_curved_shape(self):
        """Pipeline should detect high curvature on a strongly curved synthetic shape."""
        img = generate_synthetic_banana_image(curvature="highly_curved")
        result = analyze_banana(img, segmentation_method="auto")

        self.assertTrue(result.success, f"Pipeline failed: {result.message}")
        self.assertTrue(result.is_curved)
        self.assertGreater(result.curve_score, 20.0)
        self.assertEqual(result.category, CATEGORY_HIGHLY_CURVED)

    def test_pipeline_on_wood_background(self):
        """Pipeline should successfully isolate a banana placed on a yellowish-brown wooden surface."""
        # Create a wooden texture
        w, h = 500, 500
        wood_bg = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            grain = int(12 * np.sin(y / 8.0))
            wood_bg[y, :, 0] = np.clip(190 + grain, 0, 255)  # R
            wood_bg[y, :, 1] = np.clip(140 + grain, 0, 255)  # G
            wood_bg[y, :, 2] = np.clip(85 + grain, 0, 255)   # B

        curved_banana = generate_synthetic_banana_image(curvature="curved", width=w, height=h)
        banana_pixels = np.any(curved_banana < 220, axis=-1)
        composite = wood_bg.copy()
        composite[banana_pixels] = curved_banana[banana_pixels]

        result = analyze_banana(composite, segmentation_method="auto")
        self.assertTrue(result.success, f"Failed on wood background: {result.message}")
        self.assertTrue(result.is_curved)
        self.assertEqual(result.category, CATEGORY_CURVED)
        # Curve score should remain reasonable (5% to 20%), not explode to >300%
        self.assertGreaterEqual(result.curve_score, 5.0)
        self.assertLessEqual(result.curve_score, 20.0)

    def test_pipeline_on_blank_image(self):
        """Pipeline should handle images with no banana gracefully without crashing."""
        blank_img = np.full((300, 300, 3), 255, dtype=np.uint8)
        result = analyze_banana(blank_img, segmentation_method="auto")

        self.assertFalse(result.success)
        self.assertIn("No banana detected", result.message)


if __name__ == "__main__":
    unittest.main()

