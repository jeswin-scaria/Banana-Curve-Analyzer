# 🍌 Banana Curve Analyzer

A computer vision and geometric analysis system that inspects an image of a single banana to determine:
1. **Whether the banana is curved** (Boolean classification).
2. **How much it is curved** (quantitative geometric deflection & path metrics).
3. **A numerical Curve Score** (percentage elongation of the arc relative to the straight chord).
4. **A simple curvature category** (**Straight**, **Curved**, or **Highly Curved**).

> **Important Note on Measurement Definition:**  
> The **Curve Score** is a project-defined geometric metric designed specifically for this application, not a universal biological or scientific standard.

---

## 📐 Mathematical Formulation

The core mathematical principle calculates how much the banana's natural curvature elongates its centerline path compared to the direct straight-line distance connecting its two ends:

$$\text{Curve Score} = \left( \frac{L - D}{D} \right) \times 100$$

Where:
- **$D$ (Straight-Line Distance / Chord)**: The Euclidean distance between the two tips $(x_1, y_1)$ and $(x_2, y_2)$:
  $$D = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$
- **$L$ (Centerline Path Length)**: The continuous arc length along the smoothed medial axis of the banana:
  $$L = \int_{u=0}^{1} \sqrt{\left(\frac{dx}{du}\right)^2 + \left(\frac{dy}{du}\right)^2} \, du$$

### Curvature Categories
- **Straight**: $\text{Curve Score} < 5.0\%$
- **Curved**: $5.0\% \le \text{Curve Score} \le 20.0\%$
- **Highly Curved**: $\text{Curve Score} > 20.0\%$

*(Threshold boundaries are configurable in `src/config.py` or directly through the interactive UI).*

### Why Continuous Spline Smoothing Matters
Discrete digital images are composed of pixel grids. Summing raw pixel-to-pixel diagonal steps yields Manhattan/Chebyshev grid artifacts ($\approx 1.414$ px per step), which artificially inflates path lengths. To achieve true continuous geometric accuracy, our pipeline fits a parametric B-spline (`scipy.interpolate.splprep`) through the extracted skeleton centerline before integrating the path length.

---

## 🔬 Computer Vision Pipeline

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│   Input Image   │ ──> │ Color & Morphological│ ──> │   Topological Medial   │
│ (Single Banana) │     │     Segmentation     │     │  Axis Skeletonization  │
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
                                                                  │
                                                                  ▼
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│ Streamlit UI /  │ <── │  Curvature Metrics   │ <── │ Graph Path Extraction  │
│ Visual Overlays │     │  & Classification    │     │   & Spline Smoothing   │
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
```

1. **Preprocessing & Segmentation (`src/preprocessing.py`)**:
   - Isolates the banana using HSV color gating (detecting yellow and unripe green peel shades) and Otsu thresholding with background adaptation.
   - Applies morphological closing to bridge peel blemishes and openings to eliminate sensor noise.
   - Extracts the largest continuous external contour corresponding to the banana.
2. **Skeleton & Centerline Extraction (`src/skeleton.py`)**:
   - Computes a 1-pixel wide morphological skeleton using `skimage.morphology.skeletonize`.
   - Constructs an 8-connected graph of skeleton pixels.
   - Traces the longest geodesic path between topological endpoints to naturally prune small peel or stem spurs.
3. **Geometric Analysis (`src/analyzer.py`)**:
   - Fits a smooth parametric B-spline to the ordered centerline points.
   - Calculates path length $L$, chord distance $D$, Curve Score, and maximum perpendicular deflection (sagitta).
   - Categorizes curvature into **Straight**, **Curved**, or **Highly Curved**.
4. **Interactive Visualization (`src/visualization.py` & `app.py`)**:
   - Renders annotated overlays: contour boundary (green), smooth centerline (cyan), straight chord (dashed orange), endpoints, and maximum deflection (magenta).
   - Provides a 4-panel diagnostic breakdown and quantitative JSON data export.

---

## 📂 Project Structure

```
Banana-Curve-Analyzer/
├── .gitignore                  # Git ignore rules for Python, virtual environments, and cache
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation and guide
├── app.py                      # Interactive Streamlit web application
├── src/
│   ├── __init__.py             # Package exports
│   ├── config.py               # Configurable thresholds, color palettes, and default settings
│   ├── preprocessing.py        # Color thresholding, Otsu segmentation, contour extraction
│   ├── skeleton.py             # Skeletonization, graph traversal, and centerline tracing
│   ├── analyzer.py             # Spline fitting, arc length, chord distance, Curve Score
│   ├── visualization.py        # OpenCV image overlays and Matplotlib diagnostic plots
│   └── cli.py                  # Command-line interface for headless execution and batch testing
├── tests/
│   ├── __init__.py
│   ├── test_analyzer.py        # Unit tests verifying mathematical calculations & circular arcs
│   └── test_synthetic.py       # Integration tests on synthetic geometric shapes
└── samples/
    ├── generate_samples.py     # Generator for demo reference shapes (clearly labeled synthetic)
    ├── straight_banana.png
    ├── curved_banana.png
    └── highly_curved_banana.png
```

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/jeswin-scaria/Banana-Curve-Analyzer.git
cd Banana-Curve-Analyzer
pip install -r requirements.txt
```

### 2. Generate Demo Samples (Optional)
To generate reference geometric test images:
```bash
python samples/generate_samples.py
```

### 3. Run the Interactive Web App
Launch the Streamlit dashboard:
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`. You can upload your own banana images or test with the built-in demo samples.

### 4. Command-Line Interface (CLI)
You can also analyze images directly from the terminal:
```bash
# Analyze an image and display metrics in the console:
python -m src.cli --image samples/curved_banana.png

# Output results as JSON:
python -m src.cli --image samples/curved_banana.png --json

# Save the annotated visual overlay to disk:
python -m src.cli --image samples/curved_banana.png --output result_overlay.png
```

---

## 🧪 Testing and Verification

Run the automated test suite with Python's built-in `unittest`:
```bash
python -m unittest discover tests -v
```

All 8 tests verify:
- Exact zero score on straight linear segments ($L = D$).
- High-precision match against analytical circular arc geometry ($\theta = \pi/2$, radius $R = 100$).
- Subpixel preservation of endpoints under B-spline smoothing.
- Correct boundary categorization (Straight, Curved, Highly Curved).
- Graceful handling of blank or non-banana images without crashing.
- End-to-end integration on synthetic reference shapes.

---

## 🛠️ Technology Stack
- **Python 3.10+**
- **OpenCV (`cv2`)**: Color space conversions, morphological operations, contour detection, drawing.
- **NumPy**: Matrix operations and Euclidean geometry.
- **SciPy**: Parametric B-spline interpolation (`splprep`, `splev`).
- **scikit-image**: Morphological skeletonization.
- **NetworkX**: Graph representation, endpoint detection, and geodesic path tracing.
- **Matplotlib**: Multi-panel diagnostic figures and curvature profiles.
- **Streamlit**: Modern, interactive web UI.
