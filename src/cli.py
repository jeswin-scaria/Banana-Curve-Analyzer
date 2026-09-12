"""
Command-line interface (CLI) for Banana Curve Analyzer.
Usage:
    python -m src.cli --image samples/curved_banana.png
    python -m src.cli --image samples/curved_banana.png --output annotated.png
"""

import argparse
import json
import cv2
from .preprocessing import load_image
from .analyzer import analyze_banana
from .visualization import create_annotated_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Banana Curve Analyzer CLI")
    parser.add_argument("--image", "-i", required=True, help="Path to input banana image")
    parser.add_argument("--output", "-o", default=None, help="Optional path to save annotated output image")
    parser.add_argument(
        "--method", "-m", default="auto",
        choices=["auto", "wood", "hsv", "otsu", "saturation"],
        help="Segmentation method",
    )
    parser.add_argument("--smoothing", type=float, default=None, help="Spline smoothing factor (default: adaptive)")
    parser.add_argument("--morph-kernel-size", type=int, default=None, help="Morphological kernel size (default: adaptive)")
    parser.add_argument("--straight-threshold", type=float, default=None, help="Straight/Curved classification cutoff (project-defined; default from config)")
    parser.add_argument("--curved-threshold", type=float, default=None, help="Curved/Highly Curved classification cutoff (project-defined; default from config)")
    parser.add_argument("--json", "-j", action="store_true", help="Print results formatted as JSON")
    args = parser.parse_args()

    kwargs = {"segmentation_method": args.method, "smoothing": args.smoothing, "morph_kernel_size": args.morph_kernel_size}
    if args.straight_threshold is not None:
        kwargs["straight_threshold"] = args.straight_threshold
    if args.curved_threshold is not None:
        kwargs["curved_threshold"] = args.curved_threshold

    # Load image and run analysis
    img_rgb, alpha = load_image(args.image, return_alpha=True)
    result = analyze_banana(img_rgb, alpha_mask=alpha, **kwargs)

    if not result.success:
        print(f"Error [{result.failure_stage}]: {result.message}")
        return

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("========================================")
        print("        BANANA CURVE ANALYZER           ")
        print("========================================")
        print(f"File:               {args.image}")
        print(f"Curve Score:        {result.curve_score:.2f}%  (project-defined metric)")
        print(f"Curvature Category: {result.category}  (project-defined thresholds)")
        print(f"Is Curved:          {'Yes' if result.is_curved else 'No'}")
        print(f"Path Length (L):    {result.path_length:.2f} px  (measured)")
        print(f"Chord Distance (D): {result.chord_distance:.2f} px  (measured)")
        print(f"Max Deflection:     {result.max_deflection:.2f} px  (measured)")
        print(f"Deflection Ratio:   {result.deflection_ratio:.4f}  (secondary metric)")
        print(f"Mean Curvature:     {result.mean_curvature:.6f} px^-1  (secondary metric)")
        print("----------------------------------------")
        print(f"Segmentation Method: {result.segmentation_method}  (score={result.segmentation_score:.3f})")
        print(f"Confidence Score:    {result.confidence_score:.1f}%  ({result.quality_status}, heuristic -- not an ML probability)")
        print("========================================")

    if args.output:
        annotated_rgb = create_annotated_overlay(img_rgb, result)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output, annotated_bgr)
        print(f"Annotated output image saved to: {args.output}")


if __name__ == "__main__":
    main()
