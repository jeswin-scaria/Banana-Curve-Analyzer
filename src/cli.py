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
    parser.add_argument("--method", "-m", default="auto", choices=["auto", "hsv", "otsu", "saturation"], help="Segmentation method")
    parser.add_argument("--json", "-j", action="store_true", help="Print results formatted as JSON")
    args = parser.parse_args()

    # Load image and run analysis
    img_rgb = load_image(args.image)
    result = analyze_banana(img_rgb, segmentation_method=args.method)

    if not result.success:
        print(f"Error: {result.message}")
        return

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("========================================")
        print("        BANANA CURVE ANALYZER           ")
        print("========================================")
        print(f"File:               {args.image}")
        print(f"Curve Score:        {result.curve_score:.2f}%")
        print(f"Curvature Category: {result.category}")
        print(f"Is Curved:          {'Yes' if result.is_curved else 'No'}")
        print(f"Path Length (L):    {result.path_length:.2f} px")
        print(f"Chord Distance (D): {result.chord_distance:.2f} px")
        print(f"Max Deflection:     {result.max_deflection:.2f} px")
        print(f"Deflection Ratio:   {result.deflection_ratio:.4f}")
        print("========================================")

    if args.output:
        annotated_rgb = create_annotated_overlay(img_rgb, result)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output, annotated_bgr)
        print(f"Annotated output image saved to: {args.output}")


if __name__ == "__main__":
    main()
