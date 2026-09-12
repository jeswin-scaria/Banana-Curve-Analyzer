"""
Ground-truth validation: runs the full CV + geometry pipeline on synthetic
shapes whose arc length, chord distance, max deflection, and Curve Score are
known independently (scipy.integrate.quad on the shape's closed-form parametric
definition, plus dense sampling for max deflection -- see
tests/ground_truth_shapes.py). This validates the pipeline's MATH against a
truth computed outside of it, independent of any real-world segmentation
difficulty.

Tolerances are looser than the pure-math unit tests in test_analyzer.py
because rasterizing a shape into an image and re-extracting it via
segmentation + skeletonization introduces a known, small "tip-rounding" bias:
skeletonization recedes slightly from a shape's rounded/tapered physical ends,
marginally shortening the measured path relative to the mathematically exact
curve. That effect is a property of the physical measurement process (also
present when measuring a real object with a real skeletonization pipeline),
not evidence of a formula error.
"""
import unittest

from src.analyzer import analyze_banana
from tests.ground_truth_shapes import all_ground_truth_shapes

ARC_LENGTH_REL_TOLERANCE_PCT = 10.0
CHORD_REL_TOLERANCE_PCT = 10.0
CURVE_SCORE_ABS_TOLERANCE_PTS = 4.0
MAX_DEFLECTION_REL_TOLERANCE_PCT = 20.0


class TestGroundTruthGeometry(unittest.TestCase):
    """One test per shape family: straight line, circular arcs (gentle/moderate/
    strong), a quadratic arc, and a cubic Bezier curve."""

    @classmethod
    def setUpClass(cls):
        cls.shapes = all_ground_truth_shapes()

    def _check_shape(self, name: str):
        img, truth = self.shapes[name]
        result = analyze_banana(img, segmentation_method="auto")

        self.assertTrue(result.success, f"[{name}] pipeline failed: {result.message}")

        if truth["chord_distance"] > 1e-6:
            arc_rel_err = abs(result.path_length - truth["arc_length"]) / truth["arc_length"] * 100.0
            chord_rel_err = abs(result.chord_distance - truth["chord_distance"]) / truth["chord_distance"] * 100.0
            self.assertLessEqual(
                arc_rel_err, ARC_LENGTH_REL_TOLERANCE_PCT,
                f"[{name}] arc length relative error {arc_rel_err:.2f}% exceeds tolerance",
            )
            self.assertLessEqual(
                chord_rel_err, CHORD_REL_TOLERANCE_PCT,
                f"[{name}] chord distance relative error {chord_rel_err:.2f}% exceeds tolerance",
            )

        curve_score_abs_err = abs(result.curve_score - truth["curve_score"])
        self.assertLessEqual(
            curve_score_abs_err, CURVE_SCORE_ABS_TOLERANCE_PTS,
            f"[{name}] Curve Score absolute error {curve_score_abs_err:.2f}pts exceeds tolerance "
            f"(measured={result.curve_score:.2f}%, truth={truth['curve_score']:.2f}%)",
        )

        if truth["max_deflection"] > 1.0:  # skip near-zero-deflection shapes (relative error is meaningless there)
            defl_rel_err = abs(result.max_deflection - truth["max_deflection"]) / truth["max_deflection"] * 100.0
            self.assertLessEqual(
                defl_rel_err, MAX_DEFLECTION_REL_TOLERANCE_PCT,
                f"[{name}] max deflection relative error {defl_rel_err:.2f}% exceeds tolerance",
            )

    def test_straight_line(self):
        self._check_shape("straight_line")

    def test_circular_arc_gentle(self):
        self._check_shape("circular_arc_gentle")

    def test_circular_arc_moderate(self):
        self._check_shape("circular_arc_moderate")

    def test_circular_arc_strong(self):
        self._check_shape("circular_arc_strong")

    def test_quadratic(self):
        self._check_shape("quadratic")

    def test_cubic_bezier(self):
        self._check_shape("cubic_bezier")


if __name__ == "__main__":
    unittest.main()
