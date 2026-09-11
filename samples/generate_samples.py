"""
Utility script to generate sample reference images for offline testing and Streamlit demos.
NOTE: These images are synthetically rendered geometric shapes for testing and demonstration purposes.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from tests.test_synthetic import generate_synthetic_banana_image



def generate_all_samples(output_dir: str = "samples") -> None:
    """Generate sample reference images (straight, curved, highly curved)."""
    os.makedirs(output_dir, exist_ok=True)

    samples = [
        ("straight_banana.png", "straight"),
        ("curved_banana.png", "curved"),
        ("highly_curved_banana.png", "highly_curved"),
    ]

    for filename, curvature_type in samples:
        filepath = os.path.join(output_dir, filename)
        img_rgb = generate_synthetic_banana_image(curvature=curvature_type, width=600, height=600)
        # Convert RGB to BGR for cv2.imwrite
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img_bgr)
        print(f"Generated sample: {filepath} ({curvature_type})")


if __name__ == "__main__":
    generate_all_samples()
