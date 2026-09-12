"""
Manual smoke test against real banana photographs.

This is deliberately NOT part of the automated test suite: there is no labeled
ground truth for real photos in this repository, and fabricating labels would
misrepresent accuracy. Run this yourself against your own photos and eyeball
the results (and the saved annotated overlays) -- it prints every number the
pipeline produces, with no pass/fail judgement.

Usage:
    python scripts/smoke_test.py --dir path/to/your/photos [--annotate-dir out/]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2

from src.preprocessing import load_image
from src.analyzer import analyze_banana
from src.visualization import create_annotated_overlay

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual smoke test on real banana photos")
    parser.add_argument("--dir", "-d", required=True, help="Directory of banana photos")
    parser.add_argument("--annotate-dir", default=None, help="Optional directory to save annotated overlays")
    parser.add_argument("--method", "-m", default="auto", choices=["auto", "wood", "hsv", "otsu", "saturation"])
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Not a directory: {args.dir}")
        return

    files = sorted(f for f in os.listdir(args.dir) if f.lower().endswith(IMAGE_EXTENSIONS))
    if not files:
        print(f"No image files found in {args.dir}")
        return

    if args.annotate_dir:
        os.makedirs(args.annotate_dir, exist_ok=True)

    header = f"{'file':30s} {'success':8s} {'score%':>8s} {'category':14s} {'conf%':>6s} {'status':15s} {'seg_method':14s} {'failure_stage':20s}"
    print(header)
    print("-" * len(header))

    for filename in files:
        path = os.path.join(args.dir, filename)
        try:
            img_rgb, alpha = load_image(path, return_alpha=True)
        except Exception as e:
            print(f"{filename:30s} could not load: {e}")
            continue

        result = analyze_banana(img_rgb, segmentation_method=args.method, alpha_mask=alpha)

        score_str = f"{result.curve_score:.2f}" if result.success else "-"
        category = result.category if result.success else "-"
        conf_str = f"{result.confidence_score:.1f}" if result.success else "-"
        status = result.quality_status if result.success else "-"
        seg_method = result.segmentation_method or "-"
        stage = result.failure_stage or "-"

        print(f"{filename:30s} {str(result.success):8s} {score_str:>8s} {category:14s} {conf_str:>6s} {status:15s} {seg_method:14s} {stage:20s}")
        if not result.success:
            print(f"    -> {result.message}")

        if args.annotate_dir:
            annotated_rgb = create_annotated_overlay(img_rgb, result)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            out_path = os.path.join(args.annotate_dir, f"annotated_{filename}")
            cv2.imwrite(out_path, annotated_bgr)

    if args.annotate_dir:
        print(f"\nAnnotated overlays saved to: {args.annotate_dir}")


if __name__ == "__main__":
    main()
