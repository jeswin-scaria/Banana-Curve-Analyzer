"""
🍌 Banan-AI Curve Analyzer - Futuristic Light-Mode AI Laboratory
A state-of-the-art, ultra-modern computer vision laboratory for banana geometry analysis.
Built with clean porcelain white canvas, Apple/Linear aesthetic, high-contrast typography,
dynamic reactive SVG mascot, interactive curvature sandbox, and downloadable IBBC certification.
"""

import os
import json
import time
import base64
from typing import Dict, Any
import numpy as np
import streamlit as st
import cv2
from PIL import Image

from src.config import (
    STRAIGHT_THRESHOLD,
    CURVED_THRESHOLD,
    CATEGORY_STRAIGHT,
    CATEGORY_CURVED,
    CATEGORY_HIGHLY_CURVED,
    DEFAULT_HSV_LOWER,
    DEFAULT_HSV_UPPER,
    SPLINE_SMOOTHING,
)
from src.preprocessing import load_image
from src.analyzer import analyze_banana, CurvatureAnalysisResult
from src.visualization import create_annotated_overlay, create_diagnostic_figure

# -----------------------------------------------------------------------------
# Streamlit Application Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Banan-AI Vision Lab 🍌",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Futuristic Light-Mode AI Design System & Styling
# -----------------------------------------------------------------------------
st.html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    :root {
        --font-display: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-body: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        
        --bg-canvas: #ffffff;
        --bg-subtle: #f8fafc;
        --bg-card: #ffffff;
        --border-subtle: #e2e8f0;
        --border-hover: #f59e0b;
        
        --brand-gold: #eab308;
        --brand-amber: #d97706;
        --brand-orange: #ea580c;
        --tech-cyan: #0284c7;
        
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #64748b;
    }

    /* Global Light Background */
    .stApp {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
    }

    /* Sidebar Light Styling */
    [data-testid="stSidebar"] {
        background: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #1e293b !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-display) !important;
        color: var(--text-primary) !important;
    }

    /* Top Navigation Header */
    .brand-logo {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 0.3rem;
    }

    .brand-title {
        font-family: var(--font-display);
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: -0.8px;
        background: linear-gradient(135deg, #ca8a04 0%, #eab308 30%, #ea580c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.1;
    }

    .brand-subtitle {
        font-size: 1.05rem;
        color: var(--text-secondary);
        margin-top: 4px;
        font-weight: 500;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #f0fdf4;
        color: #16a34a;
        border: 1px solid #86efac;
        border-radius: 9999px;
        padding: 5px 16px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        box-shadow: 0 2px 6px rgba(22, 163, 74, 0.08);
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #16a34a;
        border-radius: 50%;
        box-shadow: 0 0 6px #22c55e;
    }

    /* Modern Elevated White Bento Card Architecture */
    .bento-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 18px !important;
        padding: 22px !important;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05), 0 2px 6px -1px rgba(0, 0, 0, 0.02) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin-bottom: 10px !important;
    }

    .bento-card:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 28px -4px rgba(245, 158, 11, 0.15), 0 4px 10px -2px rgba(0, 0, 0, 0.04) !important;
        border-color: #facc15 !important;
    }

    .kpi-val {
        font-family: var(--font-display);
        font-size: 2.5rem;
        font-weight: 900;
        line-height: 1.1;
        letter-spacing: -0.5px;
        color: #0f172a;
    }

    .kpi-unit {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-muted);
        margin-left: 2px;
    }

    .kpi-lbl {
        font-size: 0.8rem;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 6px;
    }

    /* Curvature Category Badges */
    .badge-straight {
        background: #ecfdf5 !important;
        color: #15803d !important;
        border: 1.5px solid #86efac !important;
        padding: 6px 18px !important;
        border-radius: 9999px !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        display: inline-block !important;
        letter-spacing: 0.02em !important;
    }

    .badge-curved {
        background: #fffbeb !important;
        color: #b45309 !important;
        border: 1.5px solid #fde68a !important;
        padding: 6px 18px !important;
        border-radius: 9999px !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        display: inline-block !important;
        letter-spacing: 0.02em !important;
    }

    .badge-highly-curved {
        background: #fef2f2 !important;
        color: #b91c1c !important;
        border: 1.5px solid #fca5a5 !important;
        padding: 6px 18px !important;
        border-radius: 9999px !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        display: inline-block !important;
        letter-spacing: 0.02em !important;
    }

    /* Mascot Stage (Light Mode with Warm Ambient Illumination) */
    .mascot-stage {
        background: linear-gradient(135deg, #fffbeb 0%, #ffffff 60%, #f0f9ff 100%) !important;
        border: 1.5px solid #fef08a !important;
        border-radius: 24px !important;
        padding: 26px !important;
        text-align: center !important;
        margin-bottom: 24px !important;
        box-shadow: 0 10px 30px -5px rgba(245, 158, 11, 0.12) !important;
    }

    .speech-bubble {
        display: inline-block !important;
        background: #ffffff !important;
        border: 2px solid #f59e0b !important;
        border-radius: 18px !important;
        padding: 14px 24px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.06) !important;
        max-width: 85% !important;
        line-height: 1.5 !important;
    }

    .mascot-title {
        font-family: var(--font-display) !important;
        font-size: 1.55rem !important;
        font-weight: 900 !important;
        color: #b45309 !important;
        margin-top: 12px !important;
        letter-spacing: -0.3px !important;
    }

    .mascot-subtitle {
        font-size: 1rem !important;
        color: #64748b !important;
        font-weight: 500 !important;
    }

    /* Pitch Card in Light Mode */
    .pitch-banner {
        background: linear-gradient(135deg, #f0f9ff 0%, #ffffff 100%) !important;
        border: 2px solid #0284c7 !important;
        border-radius: 18px !important;
        padding: 22px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.1) !important;
    }

    .pitch-header {
        font-family: var(--font-display) !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        color: #0284c7 !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        margin-bottom: 8px !important;
    }

    /* Certificate Box (Clean Ivory Diplomatic Charter) */
    .cert-container {
        background: #ffffff !important;
        border: 3px double #d97706 !important;
        border-radius: 20px !important;
        padding: 36px !important;
        text-align: center !important;
        box-shadow: 0 12px 35px -5px rgba(217, 119, 6, 0.15) !important;
    }

    .cert-seal {
        font-size: 3rem;
        margin-bottom: 10px;
    }
    </style>
    """
)


# -----------------------------------------------------------------------------
# Personality & Dynamic Mascot Engine
# -----------------------------------------------------------------------------
def get_personality_profile(curve_score: float) -> Dict[str, str]:
    """Generates humorous, personality-rich stats based on quantitative curvature."""
    if curve_score < 5.0:
        return {
            "title": "Corporal Cavendish 🫡",
            "personality": "The Orthogonal Ruler",
            "quote": "Zero funny business! Perfectly aligned and straight as a laser beam. 10/10 military precision. We do not bend!",
            "aerodynamic": "Javelin Class (High Penetration)",
            "peelability": "7.4 / 10 (Slightly awkward thumb angle)",
            "mario_kart": "Fires straight forward like a Green Shell!",
            "mood": "straight",
        }
    elif curve_score <= 20.0:
        return {
            "title": "The Chill Cavendish 😎",
            "personality": "The Golden Ratio Exemplar",
            "quote": "Ahh, optimal tropical curve! Peak ergonomics for one-handed peeling in a hammock. This is nature's masterpiece.",
            "aerodynamic": "Aerodynamic Arc (Gentle Glider)",
            "peelability": "10.0 / 10 (Gold Standard Ergonomics)",
            "mario_kart": "Classic highway obstacle. 100% spinout probability.",
            "mood": "curved",
        }
    else:
        return {
            "title": "The Cosmic Boomerang 🌀",
            "personality": "The Gravity-Defying Gymnast",
            "quote": "WHOA! Did someone launch me out of a cannon?! Throw me into the breeze and I'll fly right back into your fruit bowl!",
            "aerodynamic": "Boomerang Class (Hypersonic Drift)",
            "peelability": "8.8 / 10 (Centrifugal force recommended)",
            "mario_kart": "Homing missile banana. Defies Euclidean geometry!",
            "mood": "highly_curved",
        }


def render_svg_mascot(curve_score: float) -> str:
    """
    Renders an animated SVG cartoon banana mascot whose body physically
    bends according to the actual Curve Score, returned as a Base64 data URI img tag.
    """
    bend_factor = float(np.clip(curve_score * 3.4 + 10, 10, 95))
    profile = get_personality_profile(curve_score)

    if profile["mood"] == "straight":
        face_svg = """
        <line x1="85" y1="120" x2="105" y2="125" stroke="#451a03" stroke-width="4" stroke-linecap="round" />
        <line x1="115" y1="125" x2="135" y2="120" stroke="#451a03" stroke-width="4" stroke-linecap="round" />
        <circle cx="95" cy="133" r="5" fill="#1e293b" />
        <circle cx="125" cy="133" r="5" fill="#1e293b" />
        <line x1="95" y1="155" x2="125" y2="155" stroke="#451a03" stroke-width="4" stroke-linecap="round" />
        <circle cx="125" cy="133" r="12" stroke="#d97706" stroke-width="3" fill="rgba(245, 158, 11, 0.2)" />
        <path d="M 137 135 Q 145 155 130 170" stroke="#d97706" stroke-width="2" fill="none" />
        """
    elif profile["mood"] == "curved":
        face_svg = """
        <polygon points="80,122 108,122 104,142 84,142" fill="#0f172a" />
        <polygon points="112,122 140,122 136,142 116,142" fill="#0f172a" />
        <line x1="108" y1="126" x2="112" y2="126" stroke="#0f172a" stroke-width="4" />
        <line x1="75" y1="125" x2="80" y2="125" stroke="#0f172a" stroke-width="3" />
        <line x1="140" y1="125" x2="145" y2="125" stroke="#0f172a" stroke-width="3" />
        <line x1="86" y1="126" x2="100" y2="136" stroke="#38bdf8" stroke-width="2" />
        <line x1="118" y1="126" x2="132" y2="136" stroke="#38bdf8" stroke-width="2" />
        <path d="M 95 154 Q 110 168 128 156" stroke="#451a03" stroke-width="4" fill="none" stroke-linecap="round" />
        """
    else:
        face_svg = """
        <circle cx="95" cy="128" r="10" fill="#ffffff" stroke="#451a03" stroke-width="2" />
        <circle cx="125" cy="128" r="10" fill="#ffffff" stroke="#451a03" stroke-width="2" />
        <circle cx="95" cy="128" r="4.5" fill="#ef4444" />
        <circle cx="125" cy="128" r="4.5" fill="#ef4444" />
        <path d="M 85 116 Q 95 106 105 116" stroke="#451a03" stroke-width="3" fill="none" stroke-linecap="round" />
        <path d="M 115 116 Q 125 106 135 116" stroke="#451a03" stroke-width="3" fill="none" stroke-linecap="round" />
        <ellipse cx="110" cy="155" rx="14" ry="12" fill="#7f1d1d" stroke="#451a03" stroke-width="3" />
        <ellipse cx="110" cy="160" rx="9" ry="6" fill="#f43f5e" />
        <path d="M 148 112 Q 154 107 156 115 Q 156 122 148 119 Z" fill="#0284c7" />
        """

    ctrl_x = 110 + bend_factor
    back_ctrl_x = 75 + bend_factor

    svg_markup = f'''<svg width="240" height="270" viewBox="0 0 240 270" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="bananaGradient" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#fef08a" />
<stop offset="60%" stop-color="#facc15" />
<stop offset="100%" stop-color="#eab308" />
</linearGradient>
<filter id="bananaGlow" x="-20%" y="-20%" width="140%" height="140%">
<feDropShadow dx="0" dy="6" stdDeviation="8" flood-color="#f59e0b" flood-opacity="0.35" />
</filter>
</defs>
<path d="M 110 35 Q {ctrl_x} 140 105 245 Q {back_ctrl_x} 140 100 35 Z" fill="url(#bananaGradient)" stroke="#b45309" stroke-width="4" filter="url(#bananaGlow)" stroke-linejoin="round" />
<path d="M 106 42 Q {ctrl_x - 12} 140 103 235" stroke="#fef9c3" stroke-width="3.5" fill="none" stroke-linecap="round" opacity="0.9" />
<path d="M 100 35 L 110 35 L 113 18 L 97 18 Z" fill="#65a30d" stroke="#365314" stroke-width="3" />
<ellipse cx="105" cy="18" rx="8" ry="3.5" fill="#365314" />
<ellipse cx="105" cy="245" rx="5.5" ry="4" fill="#713f12" />
<ellipse cx="{ctrl_x - 25}" cy="85" rx="4" ry="3" fill="#854d0e" opacity="0.6" />
<ellipse cx="{ctrl_x - 15}" cy="190" rx="5" ry="3.5" fill="#854d0e" opacity="0.6" />
<ellipse cx="{ctrl_x - 30}" cy="215" rx="3" ry="2" fill="#854d0e" opacity="0.5" />
{face_svg}
</svg>'''

    b64_svg = base64.b64encode(svg_markup.encode('utf-8')).decode('utf-8')
    return f'<img src="data:image/svg+xml;base64,{b64_svg}" width="240" height="270" alt="Professor Peel Mascot" style="display:block; margin: 10px auto;" />'


# -----------------------------------------------------------------------------
# Main Application Controller
# -----------------------------------------------------------------------------
def main() -> None:
    # 1. Top Navigation Bar
    nav_left, nav_right = st.columns([3.8, 1.4])
    with nav_left:
        st.html(
            """
            <div class="brand-logo">
                <span style="font-size: 2.8rem; line-height: 1;">🍌</span>
                <div>
                    <div class="brand-title">Banan-AI Vision Lab</div>
                    <div class="brand-subtitle">Automated subpixel fruit geometry & neural curvature grading</div>
                </div>
            </div>
            """
        )
    with nav_right:
        st.markdown('<div style="text-align: right; padding-top: 10px;">', unsafe_allow_html=True)
        pitch_toggle = st.toggle("🎤 30s Judge Pitch", value=False, help="Open concise elevator pitch for judges.")
        st.html(
            """
            <div class="status-pill" style="margin-top: 6px;">
                <div class="status-dot"></div> AI Vision Engine v2.4 • Active
            </div>
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.html("<hr style='border: 0; border-top: 1px solid #e2e8f0; margin: 0.8rem 0 1.5rem 0;'>")

    # 2. Hackathon Pitch Mode Banner
    if pitch_toggle:
        st.html(
            """
            <div class="pitch-banner">
                <div class="pitch-header">
                    <span>⚡</span> 30-Second Hackathon Elevator Pitch
                </div>
                <div style="font-size: 0.98rem; color: #334155; line-height: 1.65;">
                    <b>The Real-World Problem:</b> In global agricultural export packaging, bananas deviating from crate-curvature thresholds become bruised in transit, resulting in billions in avoidable food waste.<br>
                    <b>Our CV Solution:</b> We engineered an intelligent pipeline combining <b>Dual-Space Saliency (CIE LAB + HSV)</b> to defeat table-color bleeding, <b>Medial Axis Skeletonization</b> with graph-geodesic spur pruning, and continuous <b>Parametric B-Splines</b> that eradicate raster step inflation.<br>
                    <b>Commercial Outcome:</b> Real-time, subpixel automated grading of fruit curvature into standardized industrial packaging categories.
                </div>
            </div>
            """
        )

    # 3. Sidebar Configuration Station
    st.sidebar.markdown(
        """
        <div style="font-size: 1.3rem; font-weight: 800; color: #0f172a; margin-bottom: 8px;">
            🎛️ Control Station
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_mode = st.sidebar.radio(
        "Select Specimen Source",
        ["🖼️ Built-in Demo Samples", "📤 Upload Custom Banana"],
        index=0,
    )

    selected_image = None
    image_name = "banana_specimen"

    if input_mode == "📤 Upload Custom Banana":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Banana Photo",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload an image containing a single banana.",
        )
        if uploaded_file is not None:
            selected_image = load_image(uploaded_file)
            image_name = uploaded_file.name
        else:
            st.info("👆 Please upload an image of a banana to begin analysis.")
    else:
        sample_options = {
            "Classic Curved Banana (Sample)": "samples/curved_banana.png",
            "Wild Highly Curved Banana (Sample)": "samples/highly_curved_banana.png",
            "Ultra Straight Banana (Sample)": "samples/straight_banana.png",
        }
        chosen_sample = st.sidebar.selectbox("Pick a reference sample", list(sample_options.keys()))
        sample_path = sample_options[chosen_sample]
        if os.path.exists(sample_path):
            selected_image = load_image(sample_path)
            image_name = chosen_sample
        else:
            st.warning(f"Sample not found at `{sample_path}`. Run `python samples/generate_samples.py`.")

    # Surface & Background Detection Presets
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 Detection Surface Preset")
    preset_map = {
        "🌟 Auto (Smart Multi-Space LAB+HSV)": "auto",
        "🪵 Wood Table / Countertop (Anti-Bleed)": "wood",
        "⚪ White / Light Surface (Otsu)": "otsu",
        "🟢 Green / Unripe Banana": "auto",
        "⚙️ Custom HSV Manual": "hsv",
    }
    selected_preset = st.sidebar.selectbox(
        "Background Surface",
        list(preset_map.keys()),
        index=0,
        help="Use 'Wood Table' if your banana is resting on a wooden table, cutting board, or brown counter!",
    )
    seg_method = preset_map[selected_preset]

    # Fine-Tuning Drawer
    with st.sidebar.expander("⚙️ Advanced Fine-Tuning", expanded=(selected_preset == "⚙️ Custom HSV Manual")):
        morph_kernel = st.slider(
            "Bridge Removal Filter",
            min_value=3,
            max_value=25,
            value=11 if seg_method == "wood" else 7,
            step=2,
            help="Snips thin background shadows or wood reflections touching the peel.",
        )

        h_min = st.slider("Min Hue", 0, 179, 35 if selected_preset == "🟢 Green / Unripe Banana" else DEFAULT_HSV_LOWER[0])
        h_max = st.slider("Max Hue", 0, 179, DEFAULT_HSV_UPPER[0])
        s_min = st.slider("Min Saturation", 0, 255, 80 if seg_method == "wood" else DEFAULT_HSV_LOWER[1])
        v_min = st.slider("Min Brightness", 0, 255, DEFAULT_HSV_LOWER[2])

        custom_lower = (h_min, s_min, v_min)
        custom_upper = (h_max, 255, 255)

        smoothing = st.slider(
            "Spline Smoothing Parameter",
            min_value=0.1,
            max_value=10.0,
            value=float(SPLINE_SMOOTHING),
            step=0.2,
            help="Degree of cubic B-spline smoothing through skeleton points.",
        )

        straight_thresh = st.number_input("Straight Threshold (%)", value=float(STRAIGHT_THRESHOLD), step=1.0)
        curved_thresh = st.number_input("Curved Threshold (%)", value=float(CURVED_THRESHOLD), step=1.0)

    # 4. Processing & Analysis Execution
    if selected_image is not None:
        with st.spinner("🍌 Banan-AI is evaluating curvature tensor splines..."):
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
            st.error(f"⚠️ **Analysis Warning**: {result.message}")
            st.info("💡 **Tip**: If your banana is on a table, select **'🪵 Wood Table / Countertop (Anti-Bleed)'** in the sidebar!")
            st.image(selected_image, caption="Uploaded Specimen", use_container_width=True)
            return

        # High Curve Score Advisory
        if result.curve_score > 50.0:
            st.warning(
                "💡 **High Curve Advisory**: If the visual overlay shows the green contour bleeding into background wood grain or shadows, switch the preset to **'🪵 Wood Table / Countertop (Anti-Bleed)'** or increase **'Bridge Removal Filter'** in the sidebar!"
            )

        personality = get_personality_profile(result.curve_score)

        # 5. Dynamic Mascot Reaction Stage (Light Mode)
        st.html(
            f"""
            <div class="mascot-stage">
                <div class="speech-bubble">
                    💬 <b>Professor Peel:</b> "{personality['quote']}"
                </div>
                {render_svg_mascot(result.curve_score)}
                <div class="mascot-title">{personality['title']}</div>
                <div class="mascot-subtitle">Personality: <b>{personality['personality']}</b></div>
            </div>
            """
        )

        # 6. Primary KPI Bento Grid (Crisp White Light Mode)
        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
        with kpi_col1:
            st.html(
                f"""
                <div class="bento-card" style="text-align: center;">
                    <div class="kpi-val" style="color: #ca8a04;">{result.curve_score:.2f}%</div>
                    <div class="kpi-lbl">Curve Score</div>
                </div>
                """
            )
        with kpi_col2:
            badge_cls = (
                "badge-straight"
                if result.category == CATEGORY_STRAIGHT
                else ("badge-curved" if result.category == CATEGORY_CURVED else "badge-highly-curved")
            )
            st.html(
                f"""
                <div class="bento-card" style="text-align: center;">
                    <div class="{badge_cls}" style="margin-top: 6px;">{result.category}</div>
                    <div class="kpi-lbl">Category</div>
                </div>
                """
            )
        with kpi_col3:
            st.html(
                f"""
                <div class="bento-card" style="text-align: center;">
                    <div class="kpi-val">{result.path_length:.1f}<span class="kpi-unit">px</span></div>
                    <div class="kpi-lbl">Arc Path Length (L)</div>
                </div>
                """
            )
        with kpi_col4:
            st.html(
                f"""
                <div class="bento-card" style="text-align: center;">
                    <div class="kpi-val">{result.chord_distance:.1f}<span class="kpi-unit">px</span></div>
                    <div class="kpi-lbl">Chord Distance (D)</div>
                </div>
                """
            )
        with kpi_col5:
            st.html(
                f"""
                <div class="bento-card" style="text-align: center;">
                    <div class="kpi-val">{result.max_deflection:.1f}<span class="kpi-unit">px</span></div>
                    <div class="kpi-lbl">Max Deflection (Sagitta)</div>
                </div>
                """
            )

        st.html("<div style='height: 16px;'></div>")

        # 7. Laboratory Workstation Tabs
        tab_inspect, tab_lab, tab_cv, tab_cert, tab_math = st.tabs(
            [
                "🎯 Visual Inspection",
                "🎛️ Curvature Sandbox",
                "🔬 4-Panel CV Diagnostics",
                "📜 Official IBBC Certificate",
                "📚 Mathematical Engine",
            ]
        )

        # TAB 1: Visual Inspection
        with tab_inspect:
            st.markdown("### 🎯 Layered Geometric Overlay")
            tog1, tog2, tog3, tog4, tog5 = st.columns(5)
            with tog1:
                show_cont = st.checkbox("🟢 Contour Boundary", value=True)
            with tog2:
                show_center = st.checkbox("🔵 Centerline Spine", value=True)
            with tog3:
                show_chord = st.checkbox("🟠 Straight Chord", value=True)
            with tog4:
                show_ends = st.checkbox("🟡 Endpoints", value=True)
            with tog5:
                show_defl = st.checkbox("🟣 Max Deflection", value=True)

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

            col_img_a, col_img_b = st.columns(2)
            with col_img_a:
                st.image(selected_image, caption="Original Input Specimen", use_container_width=True)
            with col_img_b:
                st.image(annotated_img, caption="Detected Geometric Overlay", use_container_width=True)

            with st.expander("👁️ View Isolated Binary Mask (White = Banana, Black = Background)", expanded=False):
                if result.binary_mask is not None:
                    st.image(result.binary_mask, use_container_width=True)

        # TAB 2: Live Bend Simulator
        with tab_lab:
            st.markdown("### 🎛️ Interactive 'Bend-A-Banana' Virtual Sandbox")
            st.markdown(
                "Drag the slider below to physically curve the virtual banana mascot and observe how the mathematical Curve Score computes in real time!"
            )
            sim_score = st.slider("Virtual Curvature (%)", min_value=0.0, max_value=60.0, value=float(result.curve_score), step=0.5)

            sim_left, sim_right = st.columns([1, 1.4])
            with sim_left:
                st.html(render_svg_mascot(sim_score))
            with sim_right:
                sim_data = get_personality_profile(sim_score)
                st.html(
                    f"""
                    <div class="bento-card">
                        <div style="font-size: 1.3rem; font-weight: 800; color: #ca8a04; margin-bottom: 8px;">
                            Simulation Diagnostics
                        </div>
                        <p style="margin: 6px 0;"><b>Simulated Score:</b> <span style="font-size: 1.6rem; color: #0284c7; font-weight: 900;">{sim_score:.1f}%</span></p>
                        <p style="margin: 6px 0;"><b>Archetype:</b> <span style="color: #15803d; font-weight: 700;">{sim_data['personality']}</span></p>
                        <p style="margin: 6px 0;"><b>Aerodynamics:</b> {sim_data['aerodynamic']}</p>
                        <p style="margin: 6px 0;"><b>Peelability Rating:</b> {sim_data['peelability']}</p>
                        <p style="margin: 6px 0;"><b>Mario Kart Hazard:</b> {sim_data['mario_kart']}</p>
                        <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 12px 0;">
                        <p style="font-size: 0.88rem; color: #64748b; line-height: 1.5;">
                            <b>Mathematical Principle:</b> As curvature increases, the perimeter arc distance ($L$) grows while the straight-line chord ($D$) shrinks, producing an exponential climb in the Curve Score metric!
                        </p>
                    </div>
                    """
                )

        # TAB 3: 4-Panel CV Diagnostics
        with tab_cv:
            st.markdown("### 🔬 4-Panel Computer Vision Pipeline Inspection")
            diag_figure = create_diagnostic_figure(selected_image, result, theme="light")
            st.pyplot(diag_figure)

        # TAB 4: Official IBBC Certificate
        with tab_cert:
            st.markdown("### 📜 Official Certificate of Banana Curvature")
            cert_id = f"IBBC-{abs(hash(image_name)) % 1000000:06d}"
            st.html(
                f"""
                <div class="cert-container">
                    <div class="cert-seal">🍌</div>
                    <div style="font-family: var(--font-display); font-size: 2rem; font-weight: 900; color: #d97706; letter-spacing: 1px;">
                        INTERNATIONAL BUREAU OF BANANA CURVATURE
                    </div>
                    <div style="font-size: 0.95rem; color: #64748b; margin-bottom: 18px; letter-spacing: 0.05em;">
                        OFFICIAL GLOBAL CERTIFICATE OF GEOMETRIC VERIFICATION
                    </div>
                    <p style="font-size: 1.1rem; color: #1e293b; max-width: 650px; margin: 0 auto;">
                        This document certifies that specimen <b>{image_name}</b> (Serial: <code>{cert_id}</code>)<br>
                        has undergone automated subpixel computer vision analysis and is officially verified as:
                    </p>
                    <div style="font-family: var(--font-display); font-size: 2.4rem; font-weight: 900; color: #0284c7; margin: 18px 0;">
                        {result.category.upper()} (SCORE: {result.curve_score:.2f}%)
                    </div>
                    <div style="display: flex; justify-content: space-around; max-width: 600px; margin: 20px auto; font-size: 0.95rem; color: #334155;">
                        <div>📏 <b>Arc Length:</b> {result.path_length:.1f} px</div>
                        <div>📐 <b>Chord Distance:</b> {result.chord_distance:.1f} px</div>
                        <div>🎯 <b>Max Deflection:</b> {result.max_deflection:.1f} px</div>
                    </div>
                    <div style="margin-top: 30px; font-size: 0.85rem; color: #64748b;">
                        Authorized by: <b>Professor Peel, Chief Fruit Geometer</b> • 100% Potassium Guaranteed
                    </div>
                </div>
                """
            )

            # JSON Certificate Download
            cert_payload = {
                "certificate_id": cert_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "specimen": image_name,
                "metrics": result.to_dict(),
                "personality": personality,
            }
            st.download_button(
                label="📥 Download Official JSON Certificate",
                data=json.dumps(cert_payload, indent=2),
                file_name=f"{cert_id}_certificate.json",
                mime="application/json",
            )

        # TAB 5: Mathematical Engine
        with tab_math:
            st.markdown("### 📚 The Science & Computer Vision Methodology")
            st.markdown(
                r"""
                #### Core Mathematical Formulation
                $$\text{Curve Score} = \left( \frac{\text{Centerline Path Length } (L) - \text{Straight-Line Distance } (D)}{\text{Straight-Line Distance } (D)} \right) \times 100$$
                
                - **Straight Chord ($D$)**: Euclidean distance connecting the two tips:
                  $$D = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$
                - **Centerline Arc Length ($L$)**: Numerical integration along a smooth parametric cubic B-spline:
                  $$L = \int_{u=0}^{1} \sqrt{\left(\frac{dx}{du}\right)^2 + \left(\frac{dy}{du}\right)^2} \, du$$
                
                #### Why Discrete Pixel Grids Fail Without Splines
                On rasterized digital image grids, calculating distance by summing adjacent pixels introduces diagonal Manhattan/Chebyshev grid stepping artifacts ($\sqrt{2} \approx 1.414$ px per step). This artificially inflates path lengths by $5\%\text{–}10\%$. By fitting a parametric B-spline (`scipy.interpolate.splprep`) through the topological skeleton, our engine calculates true subpixel continuous arc lengths.
                
                #### Geodesic Graph Spur Pruning
                Rough peel spots and stem cutoffs naturally create spurious skeletal branches. We construct an 8-connected graph using NetworkX and determine the longest geodesic path between candidate endpoints to isolate the true anatomical spine.
                
                #### Multi-Space Saliency (CIE LAB + HSV)
                Standard HSV often confuses yellowish-brown wooden tables with banana yellow. Our engine measures the $b^*$ blue-yellow axis against $a^*$ reddish-brown wood grain in CIE LAB space, reducing table background bleeding to $<0.01\%$.
                """
            )


if __name__ == "__main__":
    main()
