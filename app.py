"""
Banana Curve Analyzer - Interactive Streamlit Application.
Provides automated image upload, segmentation, skeletonization,
continuous spline-based arc length calculation, and curvature classification.
"""

import os
import json
import numpy as np
import streamlit as st
from PIL import Image

from src.config import (
    STRAIGHT_THRESHOLD,
    CURVED_THRESHOLD,
    CATEGORY_STRAIGHT,
    CATEGORY_CURVED,
    CATEGORY_HIGHLY_CURVED,
    CATEGORY_COLORS,
    DEFAULT_HSV_LOWER,
    DEFAULT_HSV_UPPER,
    SPLINE_SMOOTHING,
)
from src.preprocessing import load_image
from src.analyzer import analyze_banana
from src.visualization import create_annotated_overlay, create_diagnostic_figure

# Streamlit Page Configuration
st.set_page_config(
    page_title="Banana Curve Analyzer",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished hackathon aesthetic
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #f59e0b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-container {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #334155;
        text-align: center;
    }
    .badge-straight {
        background-color: #22c55e22;
        color: #22c55e;
        border: 1px solid #22c55e;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-curved {
        background-color: #f59e0b22;
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-highly-curved {
        background-color: #ef444422;
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    # Application Title and Subtitle
    st.markdown('<div class="main-header">🍌 Banana Curve Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Computer vision pipeline to determine banana curvature, measure centerline geometry, and compute numerical Curve Score.</div>',
        unsafe_allow_html=True,
    )

    # Sidebar Navigation & Settings
    st.sidebar.header("🛠️ Input & Parameters")

    input_mode = st.sidebar.radio(
        "Select Image Source",
        ["Demo Samples", "Upload Image"],
        index=0,
    )

    selected_image = None
    image_name = "banana"

    if input_mode == "Upload Image":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Banana Image",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload an image containing a single banana.",
        )
        if uploaded_file is not None:
            selected_image = load_image(uploaded_file)
            image_name = uploaded_file.name
        else:
            st.info("👆 Please upload an image of a single banana using the sidebar.")
    else:
        # Demo samples list
        sample_options = {
            "Curved Banana (Sample)": "samples/curved_banana.png",
            "Highly Curved Banana (Sample)": "samples/highly_curved_banana.png",
            "Straight Banana (Sample)": "samples/straight_banana.png",
        }
        chosen_sample = st.sidebar.selectbox("Choose a sample image", list(sample_options.keys()))
        sample_path = sample_options[chosen_sample]
        if os.path.exists(sample_path):
            selected_image = load_image(sample_path)
            image_name = chosen_sample
        else:
            st.warning(f"Sample file not found at `{sample_path}`. Run `python samples/generate_samples.py` to create samples.")

    # Background & Surface Presets
    st.sidebar.markdown("### 🎨 Background Preset")
    preset_map = {
        "🌟 Auto (Smart Multi-Space: LAB + HSV)": "auto",
        "🪵 Wood Table / Countertop (Anti-Bleed)": "wood",
        "⚪ White / Light Surface (Otsu)": "otsu",
        "🟢 Green / Unripe Banana": "auto",
        "⚙️ Custom HSV Manual": "hsv",
    }
    selected_preset = st.sidebar.selectbox(
        "Detection Preset",
        list(preset_map.keys()),
        index=0,
        help="Auto uses LAB b* + HSV saliency to prevent wood bleeding. Use 'Wood Table' for stubborn textured wood surfaces.",
    )
    seg_method = preset_map[selected_preset]

    # Advanced Settings in Expander
    with st.sidebar.expander("⚙️ Advanced CV Settings", expanded=(selected_preset == "⚙️ Custom HSV Manual")):
        morph_kernel = st.slider(
            "Morphological Cleanup (Bridge Removal)",
            min_value=3,
            max_value=25,
            value=11 if seg_method == "wood" else 7,
            step=2,
            help="Higher values disconnect thin shadows or wood reflections touching the banana.",
        )

        h_min = st.slider("Min Hue (H)", 0, 179, 35 if selected_preset == "🟢 Green / Unripe Banana" else DEFAULT_HSV_LOWER[0])
        h_max = st.slider("Max Hue (H)", 0, 179, DEFAULT_HSV_UPPER[0])
        s_min = st.slider("Min Saturation (S)", 0, 255, 80 if seg_method == "wood" else DEFAULT_HSV_LOWER[1])
        v_min = st.slider("Min Value (V)", 0, 255, DEFAULT_HSV_LOWER[2])

        custom_lower = (h_min, s_min, v_min)
        custom_upper = (h_max, 255, 255)

        smoothing = st.slider(
            "Spline Smoothing Factor",
            min_value=0.1,
            max_value=10.0,
            value=float(SPLINE_SMOOTHING),
            step=0.2,
            help="Controls smoothness of the parametric spline centerline.",
        )

        straight_thresh = st.number_input(
            "Straight Threshold (%)",
            value=float(STRAIGHT_THRESHOLD),
            step=1.0,
            help="Scores below this are categorized as Straight.",
        )
        curved_thresh = st.number_input(
            "Curved Threshold (%)",
            value=float(CURVED_THRESHOLD),
            step=1.0,
            help="Scores between Straight and Curved Thresholds are Curved; above is Highly Curved.",
        )

    # If an image is selected, proceed with analysis
    if selected_image is not None:
        with st.spinner("Analyzing banana geometry..."):
            result = analyze_banana(
                selected_image,
                segmentation_method=seg_method,
                hsv_lower=custom_lower,
                hsv_upper=custom_upper,
                smoothing=smoothing,
                straight_threshold=straight_thresh,
                curved_threshold=curved_thresh,
                morph_kernel_size=morph_kernel,
            )

        if not result.success:
            st.error(f"⚠️ Analysis Failed: {result.message}")
            st.write("Try selecting a different preset (e.g. '🪵 Wood Table' or '⚪ White Surface') or adjusting the sliders.")
            st.image(selected_image, caption="Original Input", use_container_width=True)
            return

        # High Curve Score Advisory (helpful for users when background bleeds into contour)
        if result.curve_score > 50.0:
            st.warning(
                "💡 **High Curve Score Advisory**: If the visual overlay shows the contour leaking into a wooden table or background shadow, switch the preset to **'🪵 Wood Table / Countertop (Anti-Bleed)'** or increase **'Morphological Cleanup'** in the sidebar!"
            )

        # Top KPI Metric Cards
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                label="Curve Score",
                value=f"{result.curve_score:.2f}%",
                help="Defined as ((Path Length - Chord Distance) / Chord Distance) * 100",
            )

        with col2:
            badge_class = (
                "badge-straight"
                if result.category == CATEGORY_STRAIGHT
                else ("badge-curved" if result.category == CATEGORY_CURVED else "badge-highly-curved")
            )
            st.markdown(
                f"""
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 6px;">Category</div>
                <div class="{badge_class}">{result.category}</div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.metric(
                label="Centerline Path (L)",
                value=f"{result.path_length:.1f} px",
                help="Arc length along the centerline from tip to tip.",
            )

        with col4:
            st.metric(
                label="Chord Distance (D)",
                value=f"{result.chord_distance:.1f} px",
                help="Euclidean straight-line distance between endpoints.",
            )

        with col5:
            st.metric(
                label="Max Deflection",
                value=f"{result.max_deflection:.1f} px",
                help="Perpendicular distance from chord to deepest curve point (sagitta).",
            )

        st.markdown("---")

        # Visualization Tabs
        tab_overlay, tab_diagnostics, tab_data = st.tabs(
            ["🎯 Annotated Overlay", "🔬 Diagnostic Breakdown", "📊 Analysis Data & Export"]
        )

        with tab_overlay:
            # Display Layer Controls
            ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns(5)
            with ctrl1:
                show_cont = st.checkbox("Show Contour", value=True)
            with ctrl2:
                show_center = st.checkbox("Show Centerline", value=True)
            with ctrl3:
                show_chord = st.checkbox("Show Straight Chord", value=True)
            with ctrl4:
                show_ends = st.checkbox("Show Endpoints", value=True)
            with ctrl5:
                show_defl = st.checkbox("Show Max Deflection", value=True)

            annotated_img = create_annotated_overlay(
                selected_image,
                result,
                show_contour=show_cont,
                show_centerline=show_center,
                show_chord=show_chord,
                show_endpoints=show_ends,
                show_deflection=show_defl,
                show_info_card=True,
            )

            col_orig, col_ann = st.columns(2)
            with col_orig:
                st.image(selected_image, caption="Original Input Image", use_container_width=True)
            with col_ann:
                st.image(annotated_img, caption="Annotated Curvature Overlay", use_container_width=True)

            with st.expander("👁️ View Extracted Binary Mask", expanded=False):
                if result.binary_mask is not None:
                    st.image(
                        result.binary_mask,
                        caption="Segmented Binary Mask (White = Banana Foreground, Black = Background)",
                        use_container_width=True,
                    )

        with tab_diagnostics:
            st.markdown("### 4-Panel Computer Vision Pipeline Inspection")
            diag_fig = create_diagnostic_figure(selected_image, result)
            st.pyplot(diag_fig)

        with tab_data:
            st.markdown("### Quantitative Analysis Results")
            data_dict = result.to_dict()
            st.json(data_dict)

            # Export Button
            json_str = json.dumps(data_dict, indent=2)
            st.download_button(
                label="📥 Download JSON Report",
                data=json_str,
                file_name=f"{image_name}_curvature_report.json",
                mime="application/json",
            )

    # Educational Footer Section
    st.markdown("---")
    with st.expander("📚 Methodology & Mathematical Formulation"):
        st.markdown(
            r"""
            ### Core Formulation
            The **Curve Score** is a project-defined metric quantifying how much longer a curved path is compared to the direct straight line between its ends:
            
            $$\text{Curve Score} = \left( \frac{\text{Centerline Path Length } (L) - \text{Straight-Line Distance } (D)}{\text{Straight-Line Distance } (D)} \right) \times 100$$
            
            - **Straight-Line Distance ($D$)**: Euclidean distance between the two tips:
              $$D = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$
            - **Centerline Path Length ($L$)**: Continuous arc length obtained by numerical integration along a smoothed parametric B-spline fit to the morphological skeleton:
              $$L = \int_{u=0}^{1} \sqrt{x'(u)^2 + y'(u)^2} \, du$$
            - **Classification Thresholds**:
              - **Straight**: $\text{Curve Score} < 5.0\%$
              - **Curved**: $5.0\% \le \text{Curve Score} \le 20.0\%$
              - **Highly Curved**: $\text{Curve Score} > 20.0\%$
            - **Continuous Spline Smoothing**: Discrete pixel grids introduce staircase stepping artifacts where diagonal steps measure $\sqrt{2} \approx 1.414$ px instead of Euclidean subpixel continuity. Fitting a parametric B-spline guarantees an accurate, continuous arc-length measurement.
            """
        )


if __name__ == "__main__":
    main()
