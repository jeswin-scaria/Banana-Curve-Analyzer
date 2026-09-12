"""
Regression tests for the two failure modes seen on real uploaded photographs:

1. A glossy banana's SPECULAR HIGHLIGHT desaturates a wide band down the middle
   of the fruit. The colour saliency used to multiply LAB "yellowness" by HSV
   saturation -- two measures of the same washed-out-ness -- so highlighted
   pixels collapsed to a zero score, the mask tore in half, and only a narrow
   sliver (or just the stem) survived to be measured. The reported curve was
   then the curve of a sliver, not of the banana.

2. A cut-out product PNG's transparency was discarded on load, leaving whatever
   colour the exporter had left underneath (very often pure black) as a hard
   background, when the alpha channel was in fact an exact silhouette of the
   fruit and should simply have been used.
"""
import unittest
import numpy as np
import cv2

from src.preprocessing import load_image, segment_banana_scored
from src.analyzer import (
    analyze_banana,
    fit_centerline_spline,
    calculate_path_length,
    calculate_chord_distance,
    calculate_curve_score,
)
from src.skeleton import (
    extract_skeleton,
    trace_longest_centerline,
    prune_skeleton_spurs,
    estimate_adaptive_spur_threshold,
)
from tests.synthetic_scenes import make_glossy_banana_scene


def _score_from_perfect_mask(body_mask: np.ndarray) -> float:
    """The Curve Score the pipeline produces given a flawless mask -- the target
    any segmentation result should be measured against."""
    m = body_mask.astype(np.uint8) * 255
    skeleton = extract_skeleton(m)
    skeleton = prune_skeleton_spurs(skeleton, estimate_adaptive_spur_threshold(m, skeleton))
    ordered, _, _ = trace_longest_centerline(skeleton)
    centerline, _ = fit_centerline_spline(ordered, smoothing=None)
    length = calculate_path_length(centerline)
    chord = calculate_chord_distance(tuple(centerline[0]), tuple(centerline[-1]))
    return calculate_curve_score(length, chord)


def _recall(mask: np.ndarray, truth: np.ndarray) -> float:
    found = mask > 0
    return float(np.logical_and(found, truth).sum() / truth.sum())


class TestGlossyHighlightPhoto(unittest.TestCase):
    def test_specular_highlight_does_not_tear_the_mask(self):
        img, truth, _ = make_glossy_banana_scene(on_wood=True)
        mask, contour, method, _score, _c = segment_banana_scored(img, method="auto")

        self.assertIsNotNone(contour, "no banana found on a glossy wood-table photo")
        recall = _recall(mask, truth)
        self.assertGreater(
            recall, 0.85,
            f"segmentation recovered only {recall:.1%} of the banana ({method}); a specular "
            "highlight is tearing the mask apart again",
        )

    def test_measured_curve_matches_perfect_mask_result(self):
        img, truth, _ = make_glossy_banana_scene(on_wood=True)
        target = _score_from_perfect_mask(truth)
        result = analyze_banana(img, segmentation_method="auto")

        self.assertTrue(result.success, result.message)
        self.assertLess(
            abs(result.curve_score - target), 4.0,
            f"measured {result.curve_score:.2f}% but a perfect mask of the same banana "
            f"gives {target:.2f}% -- the mask is distorting the geometry",
        )


class TestTransparentPngPhoto(unittest.TestCase):
    def test_alpha_channel_is_used_as_the_silhouette(self):
        img, truth, alpha = make_glossy_banana_scene(on_wood=False, with_alpha=True)
        self.assertIsNotNone(alpha)

        result = analyze_banana(img, segmentation_method="auto", alpha_mask=alpha)
        self.assertTrue(result.success, result.message)
        self.assertEqual(
            result.segmentation_method, "alpha_channel",
            "a cut-out PNG's own alpha silhouette should outrank the colour heuristics",
        )

        target = _score_from_perfect_mask(truth)
        self.assertLess(
            abs(result.curve_score - target), 1.5,
            f"alpha-driven measurement {result.curve_score:.2f}% should closely match the "
            f"perfect-mask result {target:.2f}%",
        )

    def test_loader_returns_alpha_and_does_not_leave_black_background(self):
        img, _truth, alpha = make_glossy_banana_scene(on_wood=False, with_alpha=True)
        rgba = np.dstack([img, alpha])

        rgb_only = load_image(rgba)
        rgb, returned_alpha = load_image(rgba, return_alpha=True)

        self.assertEqual(rgb_only.shape[2], 3, "load_image must still return plain RGB by default")
        self.assertIsNotNone(returned_alpha, "alpha channel should be reported when present")

        transparent = alpha == 0
        self.assertTrue(
            np.all(rgb[transparent] > 200),
            "fully transparent pixels should be composited onto white, not left as the "
            "exporter's leftover (often black) colour",
        )

    def test_opaque_image_reports_no_alpha(self):
        img, _truth, _ = make_glossy_banana_scene(on_wood=True)
        _rgb, alpha = load_image(img, return_alpha=True)
        self.assertIsNone(alpha, "a fully opaque image must not present a spurious alpha mask")


if __name__ == "__main__":
    unittest.main()
