"""
Segmentation-robustness regression tests: verifies the pipeline does not
blindly trust the largest contour when a larger, non-banana-shaped region is
present, and does not score a frame-spanning contour as plausible just because
cv2.findContours(RETR_EXTERNAL) discards hole information.
"""
import unittest
import numpy as np
import cv2

from src.preprocessing import segment_banana_scored
from src.analyzer import analyze_banana
from src.quality import score_contour_plausibility
from tests.synthetic_scenes import make_adversarial_rectangle_scene


class TestSegmentationRobustness(unittest.TestCase):
    """
    Composite scene: a large, non-tapered yellow rectangle (area ~68,000px,
    extent ~1.0 -- no taper) placed disjoint from a much smaller, genuinely
    tapered, curved banana shape (~13,000-15,000px). A segmentation stage that
    picks "whichever contour has the most pixels" would select the rectangle.
    A shape-plausibility-aware segmentation should reject it in favor of the
    smaller, banana-shaped region instead.
    """

    def test_segmentation_prefers_banana_shape_over_larger_rectangle(self):
        img, info = make_adversarial_rectangle_scene()
        mask, contour, method, score, candidates = segment_banana_scored(img, method="auto")

        self.assertIsNotNone(contour, "Segmentation found no contour at all.")
        self.assertGreater(len(candidates), 0)

        area = cv2.contourArea(contour)
        m = cv2.moments(contour)
        self.assertGreater(m["m00"], 0)
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]

        bx0, by0, bx1, by1 = info["banana_bbox"]
        self.assertTrue(
            bx0 <= cx <= bx1 and by0 <= cy <= by1,
            f"Winning contour centroid ({cx:.1f}, {cy:.1f}) falls outside the banana's "
            f"placement bbox {info['banana_bbox']} -- segmentation likely picked the rectangle.",
        )
        self.assertLess(
            area, info["rectangle_area"] * 0.5,
            "Winning contour area is not meaningfully smaller than the rectangle's -- "
            "the rectangle may have won instead of the banana shape.",
        )

    def test_full_pipeline_succeeds_on_adversarial_scene(self):
        img, _info = make_adversarial_rectangle_scene()
        result = analyze_banana(img, segmentation_method="auto")

        self.assertTrue(result.success, f"Pipeline failed on adversarial scene: {result.message}")
        self.assertTrue(result.is_curved)
        self.assertIsNotNone(result.segmentation_method)


class TestFrameCoveragePenalty(unittest.TestCase):
    """
    A contour whose own bounding box covers most of the image frame should be
    scored as implausible, even when its area/solidity/smoothness look
    otherwise reasonable. This is what catches "the whole image" or a
    background+banana blob merged into one region: cv2.findContours with
    RETR_EXTERNAL discards hole information, so a mask that is "the entire
    frame with a banana-shaped hole punched in it" can look deceptively like a
    clean shape on every OTHER metric.
    """

    def test_near_full_frame_contour_is_penalized(self):
        h, w = 1000, 750
        mask = np.zeros((h, w), dtype=np.uint8)
        margin = 30  # bbox covers ~92% of the frame in each dimension
        cv2.rectangle(mask, (margin, margin), (w - margin, h - margin), 255, -1)
        # Round the corners so this isn't a perfect 4-point rectangle (which
        # CHAIN_APPROX_SIMPLE would otherwise collapse to trivially).
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        self.assertGreater(len(contour), 4, "test setup should produce a rounded, multi-point contour")

        scores = score_contour_plausibility(contour, (h, w))
        self.assertGreater(scores["frame_coverage"], 0.65)
        self.assertLess(scores["frame_coverage_score"], 0.5)
        # A large, solid, smooth blob still accumulates some plausibility from
        # those sub-scores alone regardless of frame coverage -- what matters
        # for correct candidate selection is that this is clearly below a
        # legitimate banana candidate's score (~0.95+, see the companion test
        # below), not that it hits some arbitrary absolute floor.
        self.assertLess(
            scores["plausibility_score"], 0.65,
            "a near-full-frame contour should score well below a legitimate banana candidate",
        )

    def test_legitimate_close_up_banana_is_not_penalized(self):
        from tests.test_synthetic import generate_synthetic_banana_image

        img = generate_synthetic_banana_image(curvature="curved", width=700, height=700)
        from src.preprocessing import segment_banana_scored

        _mask, contour, _method, score, _candidates = segment_banana_scored(img, method="auto")
        self.assertIsNotNone(contour)
        scores = score_contour_plausibility(contour, img.shape[:2])
        self.assertEqual(
            scores["frame_coverage_score"], 1.0,
            "a normal, non-full-frame banana photo should not be penalized by the frame-coverage check",
        )


class TestPerformanceScaling(unittest.TestCase):
    """
    Regression guard against reintroducing resolution-dependent runtime: a
    realistic phone-photo-sized image should still analyze in well under the
    'a few seconds' budget, not scale into multiple seconds with pixel count.
    """

    def test_large_image_analyzes_quickly(self):
        import time
        from tests.test_synthetic import generate_synthetic_banana_image
        from tests.transform_utils import resize_image

        base = generate_synthetic_banana_image(curvature="curved", width=800, height=800)
        large_img = resize_image(base, 3.0)  # 2400x2400, phone-photo-scale

        start = time.perf_counter()
        result = analyze_banana(large_img, segmentation_method="auto")
        elapsed_s = time.perf_counter() - start

        self.assertTrue(result.success)
        self.assertLess(
            elapsed_s, 3.0,
            f"analyze_banana took {elapsed_s:.2f}s on a 2400x2400 image -- "
            "runtime should not scale with input resolution",
        )


if __name__ == "__main__":
    unittest.main()
