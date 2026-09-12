"""
Before/after evidence benchmark for the Banana Curve Analyzer backend.

Run this script once against the current code to capture a baseline, implement
pipeline improvements, then run it again to capture the "improved" numbers.
Both runs use the exact same synthetic ground-truth shapes, stability transforms,
and adversarial segmentation scene -- only the pipeline code differs between runs.

Usage:
    python scripts/benchmark.py --label baseline --output benchmarks/baseline.json
    python scripts/benchmark.py --label improved --output benchmarks/improved.json
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from src.analyzer import analyze_banana
from src.preprocessing import segment_banana
from tests.ground_truth_shapes import all_ground_truth_shapes
from tests.transform_utils import (
    resize_image,
    rotate_image,
    adjust_brightness,
    adjust_contrast,
    add_gaussian_noise,
    crop_with_margin,
)
from tests.synthetic_scenes import make_adversarial_rectangle_scene
from tests.test_synthetic import generate_synthetic_banana_image


def _rel_error(measured: float, truth: float) -> float:
    if abs(truth) < 1e-9:
        return 0.0 if abs(measured) < 1e-9 else float("inf")
    return abs(measured - truth) / abs(truth) * 100.0


def run_ground_truth_benchmark() -> dict:
    results = {}
    for name, (img, truth) in all_ground_truth_shapes().items():
        result = analyze_banana(img, segmentation_method="auto")
        if not result.success:
            results[name] = {"success": False, "message": result.message, "truth": truth}
            continue
        results[name] = {
            "success": True,
            "truth": truth,
            "measured": {
                "arc_length": result.path_length,
                "chord_distance": result.chord_distance,
                "max_deflection": result.max_deflection,
                "curve_score": result.curve_score,
            },
            "relative_error_pct": {
                "arc_length": _rel_error(result.path_length, truth["arc_length"]),
                "chord_distance": _rel_error(result.chord_distance, truth["chord_distance"]),
                "max_deflection": _rel_error(result.max_deflection, truth["max_deflection"]),
            },
            "curve_score_abs_error_pts": abs(result.curve_score - truth["curve_score"]),
        }
    return results


def _stability_group_stats(scores: list, ratios: list) -> dict:
    scores_arr = np.array(scores, dtype=float)
    ratios_arr = np.array(ratios, dtype=float)

    def stats(arr):
        if len(arr) == 0:
            return {"mean": None, "std": None, "cv_pct": None}
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        cv = (std / mean * 100.0) if abs(mean) > 1e-9 else 0.0
        return {"mean": mean, "std": std, "cv_pct": cv}

    return {"curve_score": stats(scores_arr), "deflection_ratio": stats(ratios_arr)}


def run_stability_benchmark() -> dict:
    base_img = generate_synthetic_banana_image(curvature="curved", width=700, height=700)

    variant_groups = {
        "resize": [("0.5x", resize_image(base_img, 0.5)), ("1.0x", base_img), ("2.0x", resize_image(base_img, 2.0))],
        "rotation": [(f"{a}deg", rotate_image(base_img, a)) for a in (0, 45, 90, 135)],
        "brightness": [("-30", adjust_brightness(base_img, -30)), ("+30", adjust_brightness(base_img, 30))],
        "contrast": [("0.7x", adjust_contrast(base_img, 0.7)), ("1.3x", adjust_contrast(base_img, 1.3))],
        "noise": [("sigma10", add_gaussian_noise(base_img, 10)), ("sigma20", add_gaussian_noise(base_img, 20))],
        "crop": [("10pct", crop_with_margin(base_img, 0.10)), ("25pct", crop_with_margin(base_img, 0.25))],
    }

    results = {}
    for group_name, variants in variant_groups.items():
        scores, ratios, categories, per_variant = [], [], [], {}
        for variant_name, img in variants:
            result = analyze_banana(img, segmentation_method="auto")
            per_variant[variant_name] = {
                "success": result.success,
                "curve_score": result.curve_score if result.success else None,
                "category": result.category if result.success else None,
            }
            if result.success:
                scores.append(result.curve_score)
                ratios.append(result.deflection_ratio)
                categories.append(result.category)
        stats = _stability_group_stats(scores, ratios)
        stats["categories_seen"] = sorted(set(categories))
        stats["per_variant"] = per_variant
        results[group_name] = stats
    return results


def run_segmentation_robustness_benchmark() -> dict:
    img, info = make_adversarial_rectangle_scene()
    mask, contour = segment_banana(img, method="auto")

    winning_area = None
    winning_centroid_inside_banana = None
    if contour is not None:
        winning_area = float(cv2.contourArea(contour))
        m = cv2.moments(contour)
        if m["m00"] > 0:
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            bx0, by0, bx1, by1 = info["banana_bbox"]
            winning_centroid_inside_banana = bool(bx0 <= cx <= bx1 and by0 <= cy <= by1)

    result = analyze_banana(img, segmentation_method="auto")

    return {
        "rectangle_area": info["rectangle_area"],
        "banana_area_estimate": info["banana_area_estimate"],
        "winning_contour_area": winning_area,
        "winning_centroid_inside_banana_bbox": winning_centroid_inside_banana,
        "picked_banana_not_rectangle": winning_centroid_inside_banana is True,
        "analyze_banana_success": result.success,
        "analyze_banana_category": result.category if result.success else None,
        "segmentation_method": getattr(result, "segmentation_method", None),
    }


def run_timing_benchmark(n_runs: int = 20) -> dict:
    img = generate_synthetic_banana_image(curvature="curved", width=800, height=800)
    # Warm-up run (first call can include import/JIT-ish overhead).
    analyze_banana(img, segmentation_method="auto")

    durations_ms = []
    for _ in range(n_runs):
        start = time.perf_counter()
        analyze_banana(img, segmentation_method="auto")
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "n_runs": n_runs,
        "mean_ms": float(np.mean(durations_ms)),
        "median_ms": float(np.median(durations_ms)),
        "max_ms": float(np.max(durations_ms)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Banana Curve Analyzer backend benchmark")
    parser.add_argument("--label", default="run", help="Label for this benchmark run (e.g. baseline, improved)")
    parser.add_argument("--output", default=None, help="Output JSON path (default: benchmarks/<label>.json)")
    args = parser.parse_args()

    output_path = args.output or os.path.join("benchmarks", f"{args.label}.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Running ground-truth accuracy benchmark...")
    ground_truth = run_ground_truth_benchmark()
    print(f"Running stability benchmark...")
    stability = run_stability_benchmark()
    print(f"Running segmentation-robustness benchmark...")
    segmentation_robustness = run_segmentation_robustness_benchmark()
    print(f"Running timing benchmark...")
    timing = run_timing_benchmark()

    payload = {
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ground_truth": ground_truth,
        "stability": stability,
        "segmentation_robustness": segmentation_robustness,
        "timing": timing,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote benchmark results to {output_path}")


if __name__ == "__main__":
    main()
