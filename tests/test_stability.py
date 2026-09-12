"""
Stability regression tests: the same synthetic curved banana, put through
resize, rotation, brightness/contrast, Gaussian noise, and margin crops, should
produce a Curve Score that varies only modestly -- these transforms don't
change the banana's real-world curvature, so large swings would mean the
measurement is unreliable rather than the shape actually changing.

Coefficient-of-variation (CV%) ceilings below are calibrated from
scripts/benchmark.py's measured baseline/improved runs (see benchmarks/*.json
and BACKEND_AUDIT.md), with margin -- they are regression guards, not claims
of a universal accuracy bound. Resize and rotation are NOT asserted on
classification stability: this particular synthetic shape's true Curve Score
sits close enough to the 5% Straight/Curved boundary that pixel-grid resampling
under rotation/resize can legitimately flip its category -- a known, disclosed
limitation of hard thresholds near a boundary, not something this test suite
claims to have solved.
"""
import unittest
import numpy as np

from src.analyzer import analyze_banana
from tests.test_synthetic import generate_synthetic_banana_image
from tests.transform_utils import (
    resize_image,
    rotate_image,
    adjust_brightness,
    adjust_contrast,
    add_gaussian_noise,
    crop_with_margin,
)

CV_CEILING_PCT = {
    "resize": 10.0,
    "rotation": 15.0,
    "brightness": 5.0,
    "contrast": 5.0,
    "noise": 5.0,
    "crop": 5.0,
}


def _curve_score_cv(scores):
    arr = np.array(scores, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return (std / mean * 100.0) if abs(mean) > 1e-9 else 0.0


class TestStability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_img = generate_synthetic_banana_image(curvature="curved", width=700, height=700)

    def _run_group(self, variants):
        scores, categories = [], []
        for _name, img in variants:
            result = analyze_banana(img, segmentation_method="auto")
            self.assertTrue(result.success, f"Pipeline failed on stability variant '{_name}': {result.message}")
            scores.append(result.curve_score)
            categories.append(result.category)
        return scores, categories

    def test_resize_stability(self):
        variants = [("0.5x", resize_image(self.base_img, 0.5)),
                    ("1.0x", self.base_img),
                    ("2.0x", resize_image(self.base_img, 2.0))]
        scores, _categories = self._run_group(variants)
        cv = _curve_score_cv(scores)
        self.assertLessEqual(cv, CV_CEILING_PCT["resize"], f"Resize Curve Score CV {cv:.2f}% exceeds ceiling")

    def test_rotation_stability(self):
        variants = [(f"{a}deg", rotate_image(self.base_img, a)) for a in (0, 45, 90, 135)]
        scores, _categories = self._run_group(variants)
        cv = _curve_score_cv(scores)
        self.assertLessEqual(cv, CV_CEILING_PCT["rotation"], f"Rotation Curve Score CV {cv:.2f}% exceeds ceiling")

    def test_brightness_stability(self):
        variants = [("-30", adjust_brightness(self.base_img, -30)), ("+30", adjust_brightness(self.base_img, 30))]
        scores, categories = self._run_group(variants)
        cv = _curve_score_cv(scores)
        self.assertLessEqual(cv, CV_CEILING_PCT["brightness"], f"Brightness Curve Score CV {cv:.2f}% exceeds ceiling")
        self.assertEqual(len(set(categories)), 1, "Brightness change should not flip classification")

    def test_contrast_stability(self):
        variants = [("0.7x", adjust_contrast(self.base_img, 0.7)), ("1.3x", adjust_contrast(self.base_img, 1.3))]
        scores, categories = self._run_group(variants)
        cv = _curve_score_cv(scores)
        self.assertLessEqual(cv, CV_CEILING_PCT["contrast"], f"Contrast Curve Score CV {cv:.2f}% exceeds ceiling")
        self.assertEqual(len(set(categories)), 1, "Contrast change should not flip classification")

    def test_noise_stability(self):
        variants = [("sigma10", add_gaussian_noise(self.base_img, 10)), ("sigma20", add_gaussian_noise(self.base_img, 20))]
        scores, categories = self._run_group(variants)
        cv = _curve_score_cv(scores)
        self.assertLessEqual(cv, CV_CEILING_PCT["noise"], f"Noise Curve Score CV {cv:.2f}% exceeds ceiling")
        self.assertEqual(len(set(categories)), 1, "Moderate noise should not flip classification")

    def test_crop_stability(self):
        variants = [("10pct", crop_with_margin(self.base_img, 0.10)), ("25pct", crop_with_margin(self.base_img, 0.25))]
        scores, categories = self._run_group(variants)
        cv = _curve_score_cv(scores)
        self.assertLessEqual(cv, CV_CEILING_PCT["crop"], f"Crop Curve Score CV {cv:.2f}% exceeds ceiling")
        self.assertEqual(len(set(categories)), 1, "Margin crop should not flip classification")


if __name__ == "__main__":
    unittest.main()
