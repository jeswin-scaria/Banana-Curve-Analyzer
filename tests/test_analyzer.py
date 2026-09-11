"""
Unit tests for mathematical curvature formulas and classification logic.
"""

import unittest
import numpy as np
from src.analyzer import (
    calculate_chord_distance,
    calculate_path_length,
    calculate_curve_score,
    classify_curvature,
    calculate_max_deflection,
    fit_centerline_spline,
)
from src.config import (
    CATEGORY_STRAIGHT,
    CATEGORY_CURVED,
    CATEGORY_HIGHLY_CURVED,
    STRAIGHT_THRESHOLD,
    CURVED_THRESHOLD,
)


class TestCurvatureAnalyzer(unittest.TestCase):
    """Test suite for mathematical and analytical functions."""

    def test_straight_line_geometry(self):
        """A straight line should have equal path length and chord distance, yielding Curve Score = 0."""
        p1 = (0.0, 0.0)
        p2 = (150.0, 0.0)

        # Generate collinear points
        x = np.linspace(p1[0], p2[0], 50)
        y = np.linspace(p1[1], p2[1], 50)
        points = np.column_stack([x, y])

        chord = calculate_chord_distance(p1, p2)
        length = calculate_path_length(points)
        score = calculate_curve_score(length, chord)
        is_curved, category = classify_curvature(score)

        self.assertAlmostEqual(chord, 150.0, places=4)
        self.assertAlmostEqual(length, 150.0, places=4)
        self.assertAlmostEqual(score, 0.0, places=4)
        self.assertFalse(is_curved)
        self.assertEqual(category, CATEGORY_STRAIGHT)

    def test_circular_arc_analytical_match(self):
        """
        Verify against an analytical circular arc of radius R = 100 spanning 90 degrees (pi/2).
        - Arc length L = R * theta = 100 * pi/2 = 50 * pi ~= 157.0796
        - Chord length D = 2 * R * sin(theta/2) = 200 * sin(pi/4) = 100 * sqrt(2) ~= 141.4214
        - Curve Score = ((L - D) / D) * 100 = ((pi / (2 * sqrt(2))) - 1) * 100 ~= 11.0721%
        - Theoretical Sagitta (Max Deflection) = R * (1 - cos(theta/2)) = 100 * (1 - sqrt(2)/2) ~= 29.2893
        """
        r = 100.0
        theta_total = np.pi / 2.0  # 90 degrees

        # Angles from -pi/4 to +pi/4 so chord is vertical
        angles = np.linspace(-theta_total / 2.0, theta_total / 2.0, 200)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        arc_points = np.column_stack([x, y])

        p1 = (arc_points[0, 0], arc_points[0, 1])
        p2 = (arc_points[-1, 0], arc_points[-1, 1])

        chord = calculate_chord_distance(p1, p2)
        length = calculate_path_length(arc_points)
        score = calculate_curve_score(length, chord)

        expected_chord = 2.0 * r * np.sin(theta_total / 2.0)
        expected_length = r * theta_total
        expected_score = ((expected_length - expected_chord) / expected_chord) * 100.0
        expected_sagitta = r * (1.0 - np.cos(theta_total / 2.0))

        self.assertAlmostEqual(chord, expected_chord, places=3)
        self.assertAlmostEqual(length, expected_length, places=2)
        self.assertAlmostEqual(score, expected_score, places=2)

        # Verify sagitta (max deflection)
        max_defl, apex, foot = calculate_max_deflection(arc_points, p1, p2)
        self.assertAlmostEqual(max_defl, expected_sagitta, places=1)

    def test_classification_thresholds(self):
        """Verify boundary classification matches project requirements."""
        # Below straight threshold
        is_curved, cat = classify_curvature(0.0)
        self.assertFalse(is_curved)
        self.assertEqual(cat, CATEGORY_STRAIGHT)

        is_curved, cat = classify_curvature(STRAIGHT_THRESHOLD - 0.1)
        self.assertFalse(is_curved)
        self.assertEqual(cat, CATEGORY_STRAIGHT)

        # At and within curved range
        is_curved, cat = classify_curvature(STRAIGHT_THRESHOLD)
        self.assertTrue(is_curved)
        self.assertEqual(cat, CATEGORY_CURVED)

        is_curved, cat = classify_curvature(12.5)
        self.assertTrue(is_curved)
        self.assertEqual(cat, CATEGORY_CURVED)

        is_curved, cat = classify_curvature(CURVED_THRESHOLD)
        self.assertTrue(is_curved)
        self.assertEqual(cat, CATEGORY_CURVED)

        # Above curved threshold -> Highly Curved
        is_curved, cat = classify_curvature(CURVED_THRESHOLD + 0.1)
        self.assertTrue(is_curved)
        self.assertEqual(cat, CATEGORY_HIGHLY_CURVED)

        is_curved, cat = classify_curvature(45.0)
        self.assertTrue(is_curved)
        self.assertEqual(cat, CATEGORY_HIGHLY_CURVED)

    def test_spline_smoothing_preserves_endpoints(self):
        """Spline fitting should preserve beginning and end of the path within subpixel tolerance."""
        points = [(0.0, 0.0), (10.0, 5.0), (20.0, 15.0), (30.0, 30.0), (40.0, 50.0)]
        sampled, _ = fit_centerline_spline(points, smoothing=0.1, num_points=100)

        self.assertEqual(len(sampled), 100)
        # Check start and end proximity
        self.assertAlmostEqual(sampled[0, 0], points[0][0], delta=1.5)
        self.assertAlmostEqual(sampled[0, 1], points[0][1], delta=1.5)
        self.assertAlmostEqual(sampled[-1, 0], points[-1][0], delta=1.5)
        self.assertAlmostEqual(sampled[-1, 1], points[-1][1], delta=1.5)


if __name__ == "__main__":
    unittest.main()
