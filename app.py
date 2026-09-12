"""
🍌 CHILL ETHAKAAA — BANANA GEOMETRY LAB
"Measuring what never needed to be measured."
A premium, sophisticated computer-vision instrument for geometric banana curvature analysis.
"""

import os
import io
import json
import time
import base64
from typing import Dict, Any, Optional
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
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
from src.visualization import create_annotated_overlay, create_diagnostic_figure, create_isolated_shape_overlay
from src.fun_modules import render_absurd_interactive_hub

# ---------------------------------------------------------------------------
# Asset loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def asset_data_uri(path: str, max_px: int = 680, quality: int = 82, _mtime: float = 0.0) -> str:
    """
    Return an inline data URI for an image asset, downscaled and recompressed.

    Streamlit re-sends inline HTML on every rerun, so an oversized asset costs
    its full weight again on every single interaction. The source logo is
    1024x1024 (~494KB, ~659KB once base64-encoded) but is never displayed wider
    than 340px, so serving it at 2x that width costs ~63KB instead -- the same
    thing on screen for about a tenth of the bytes.

    Cached on (path, size, quality, mtime) so swapping the file in assets/ picks
    the new one up automatically. Falls back to the raw bytes if anything about
    the re-encode fails, so a new or unusual asset can never break the page.
    """
    if not os.path.exists(path):
        return ""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if max(im.size) > max_px:
                im.thumbnail((max_px, max_px), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality, optimize=True, progressive=True)
            data = buf.getvalue()
        raw_size = os.path.getsize(path)
        if len(data) >= raw_size:  # already well-optimised; don't make it bigger
            with open(path, "rb") as f:
                data = f.read()
    except Exception:
        try:
            with open(path, "rb") as f:
                data = f.read()
        except Exception:
            return ""
    return f"data:image/jpeg;base64,{base64.b64encode(data).decode('utf-8')}"


def _asset_uri(name: str, max_px: int = 680) -> str:
    path = os.path.join(os.path.dirname(__file__), "assets", name)
    mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
    return asset_data_uri(path, max_px=max_px, _mtime=mtime)


# ---------------------------------------------------------------------------
# Fullscreen Comic Splash Screen ("MELCOW")
# ---------------------------------------------------------------------------
def render_splash_screen():
    img_src = _asset_uri("melcow.jpg", max_px=900)

    splash_script = f"""
    <script>
    (function() {{
        try {{
            var targetWin = window.parent || window;
            var targetDoc = (window.parent && window.parent.document) ? window.parent.document : document;

            if (targetWin._splashDismissed || targetWin.sessionStorage.getItem('banana_splash_dismissed') === 'true') {{
                return;
            }}

            var existing = targetDoc.getElementById('comic-splash-overlay');
            if (existing) {{
                return;
            }}

            var overlay = targetDoc.createElement('div');
            overlay.id = 'comic-splash-overlay';
            overlay.style.cssText = 'position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 999999999; background-color: #0a0b10; background-image: url("{img_src}"); background-position: center center; background-repeat: no-repeat; background-size: contain; pointer-events: none; transition: opacity 1.5s ease-in-out, visibility 1.5s ease-in-out; opacity: 1; visibility: visible;';

            targetDoc.body.appendChild(overlay);

            function autoDismiss() {{
                targetWin._splashDismissed = true;
                try {{ targetWin.sessionStorage.setItem('banana_splash_dismissed', 'true'); }} catch(err) {{}}

                var el = targetDoc.getElementById('comic-splash-overlay');
                if (el) {{
                    el.style.opacity = '0';
                    el.style.visibility = 'hidden';
                    setTimeout(function() {{
                        if (el && el.parentNode) {{
                            el.parentNode.removeChild(el);
                        }}
                    }}, 1600);
                }}
            }}

            // Automatically transition to the main app after 3 seconds with a 1.5s smooth fade
            setTimeout(autoDismiss, 3000);
        }} catch(e) {{
            console.error('Splash error:', e);
        }}
    }})();
    </script>
    """
    components.html(splash_script, height=0, width=0)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CHILL ETHAKAAA | Banana Geometry Lab",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

render_splash_screen()

# Start the font download early. The stylesheet below also @imports these exact
# families, but an @import cannot begin fetching until the CSS around it has been
# downloaded and parsed -- two serialised round trips before any text can render.
# Emitting the same URL as a <link> up here starts it immediately; the @import
# then resolves from cache. The @import is deliberately kept as the fallback, so
# fonts still load correctly even if these tags are ever sanitised away.
# NOTE: must be st.markdown -- st.html runs DOMPurify, which strips <link>.
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800;900&family=Playfair+Display:ital,wght@0,600;0,700;0,800;1,400;1,600&family=Alex+Brush&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&display=swap">
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Premium Scientific Laboratory & Official Certificate Design System
# ---------------------------------------------------------------------------
st.html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800;900&family=Playfair+Display:ital,wght@0,600;0,700;0,800;1,400;1,600&family=Alex+Brush&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&display=swap');

    :root {
        --bg-ivory: #F8F5EC;
        --surface-white: #FFFFFF;
        --surface-warm: #FCFAF6;
        --text-charcoal: #171717;
        --text-secondary: #5A5954;
        --text-muted: #8E8D86;
        --accent-yellow: #F4B400;
        --accent-yellow-light: #FFF9E6;
        --accent-yellow-border: #FBE6A2;
        --accent-olive: #66734A;
        --accent-olive-light: #F2F5ED;
        --accent-olive-border: #D1DBC1;
        --border-gray: #E5E0D4;
        --border-subtle: #EDE8DE;
        --border-dark: #171717;
        
        --font-sans: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif;
        --font-body: 'Inter', -apple-system, sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
        --font-serif: 'Newsreader', Georgia, serif;
        --font-diploma: 'Cinzel', serif;
        --font-cert-head: 'Playfair Display', Georgia, serif;
        --font-script: 'Alex Brush', cursive;
    }

    /* === Global Canvas Override === */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-ivory) !important;
        color: var(--text-charcoal) !important;
        font-family: var(--font-body) !important;
    }

    /* Streamlit Main Content Container Constraints */
    .main .block-container {
        max-width: 1120px !important;
        padding-top: 2.2rem !important;
        padding-bottom: 4rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Force all base typography */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: var(--text-charcoal);
    }

    /* Hide Streamlit Header & Clutter */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    #MainMenu, footer {
        visibility: hidden;
    }

    /* === 1. Premium Brand Header centered === */
    .brand-hero {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid var(--border-gray);
        margin-bottom: 1.75rem;
        gap: 0.75rem;
    }

    .brand-identity {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        gap: 0.5rem;
    }

    .brand-mark {
        font-size: 3.2rem;
        line-height: 1;
        filter: drop-shadow(0 2px 10px rgba(244, 180, 0, 0.3));
        user-select: none;
    }

    .brand-title {
        font-family: var(--font-sans);
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        word-spacing: 0.06em;
        color: var(--text-charcoal);
        line-height: 1.15;
        margin: 0;
        text-align: center;
    }

    .brand-subtitle {
        font-family: var(--font-mono);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--accent-olive);
        margin-top: 0.3rem;
        text-align: center;
    }

    .brand-status {
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        background: var(--surface-white);
        border: 1px solid var(--border-gray);
        border-radius: 9999px;
        padding: 0.5rem 1.15rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background-color: var(--accent-olive);
        border-radius: 50%;
        display: inline-block;
    }

    .status-text {
        font-family: var(--font-mono);
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-secondary);
    }

    .brand-tagline {
        font-family: var(--font-serif);
        font-style: italic;
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin-top: -0.75rem;
        margin-bottom: 2.2rem;
    }

    /* === 2. Image Ingestion Station === */
    .ingestion-card {
        background: var(--surface-white);
        border: 1px solid var(--border-gray);
        border-radius: 12px;
        padding: 2.2rem 2rem 1.6rem 2rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.025);
        margin-bottom: 1.2rem;
    }

    .ingestion-eyebrow {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--accent-olive);
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .ingestion-title {
        font-family: var(--font-sans);
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        color: var(--text-charcoal);
        margin: 0 0 0.45rem 0;
        line-height: 1.3;
    }

    .ingestion-prompt {
        font-family: var(--font-body);
        font-size: 0.95rem;
        color: var(--text-secondary);
        max-width: 480px;
        margin: 0 auto 0.75rem auto;
        line-height: 1.5;
    }

    .ingestion-formats {
        font-family: var(--font-mono);
        font-size: 0.74rem;
        color: var(--text-muted);
        letter-spacing: 0.06em;
    }

    /* Custom Streamlit File Uploader: High-Contrast, Vibrant, Perfectly Centered */
    [data-testid="stFileUploader"] {
        max-width: 580px;
        margin: 0 auto 2rem auto;
        text-align: center !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #EAB308 !important;
        border-radius: 12px !important;
        padding: 1.8rem 1.5rem !important;
        box-shadow: 0 4px 16px rgba(244, 180, 0, 0.08) !important;
        transition: all 0.25s ease !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #CA8A04 !important;
        background-color: #FFFDF5 !important;
        box-shadow: 0 6px 20px rgba(244, 180, 0, 0.16) !important;
    }
    [data-testid="stFileUploaderDropzone"] > div {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        margin: 0 auto !important;
        gap: 0.85rem !important;
        width: 100% !important;
    }
    [data-testid="stFileUploaderDropzone"] > div > div {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        margin: 0 auto !important;
        gap: 0.85rem !important;
    }
    /* The Upload / Browse Button: Vibrant, Bold, Perfectly Centered */
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploader"] section button {
        background: linear-gradient(135deg, #F4B400, #EAB308) !important;
        color: #171717 !important;
        border: 2px solid #CA8A04 !important;
        border-radius: 8px !important;
        font-family: var(--font-sans) !important;
        font-size: 0.95rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.04em !important;
        padding: 0.65rem 1.6rem !important;
        box-shadow: 0 4px 14px rgba(244, 180, 0, 0.35) !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="stFileUploader"] section button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(244, 180, 0, 0.5) !important;
        background: linear-gradient(135deg, #F59E0B, #D97706) !important;
        color: #FFFFFF !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small {
        color: #5A5954 !important;
        font-size: 0.85rem !important;
        text-align: center !important;
    }
    [data-testid="stFileUploaderFileData"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #EAB308 !important;
        border-radius: 8px !important;
        padding: 0.6rem 0.9rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
        margin: 0.5rem auto 0 auto !important;
    }

    /* === 3. Dominant Result Card (CURVE SCORE) === */
    .score-hero-card {
        background: var(--surface-white);
        border: 1px solid var(--border-gray);
        border-radius: 12px;
        padding: 2.2rem 2.4rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.025);
        margin-bottom: 1.75rem;
    }

    .score-eyebrow {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--accent-olive);
        margin-bottom: 0.4rem;
    }

    .score-label {
        font-family: var(--font-sans);
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
    }

    .score-display-row {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1.75rem;
    }

    .score-giant-number {
        font-family: var(--font-mono);
        font-size: 4.2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1;
        color: var(--text-charcoal);
    }

    .score-giant-percent {
        font-size: 2.2rem;
        color: var(--accent-yellow);
        font-weight: 700;
        margin-left: 2px;
    }

    .classification-badge {
        font-family: var(--font-mono);
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.45rem 1.1rem;
        border-radius: 6px;
        border: 1px solid transparent;
    }

    .badge-straight {
        background: var(--accent-olive-light);
        color: var(--accent-olive);
        border-color: var(--accent-olive-border);
    }

    .badge-curved {
        background: var(--accent-yellow-light);
        color: #8C6600;
        border-color: var(--accent-yellow-border);
    }

    .badge-highly-curved {
        background: #FDF2E9;
        color: #B24B00;
        border-color: #F8D7BE;
    }

    /* Clean Scientific Visual Scale */
    .scale-container {
        margin-top: 1rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border-subtle);
    }

    .scale-boundary-labels {
        display: flex;
        justify-content: space-between;
        font-family: var(--font-mono);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    .scale-track-wrapper {
        position: relative;
        height: 10px;
        background: var(--border-subtle);
        border-radius: 9999px;
        overflow: visible;
        margin-bottom: 1.8rem;
    }

    .scale-track-segment-1 {
        position: absolute;
        left: 0%;
        width: 14.3%; /* 5% out of 35% */
        height: 100%;
        background: #DCE5D1;
        border-top-left-radius: 9999px;
        border-bottom-left-radius: 9999px;
    }

    .scale-track-segment-2 {
        position: absolute;
        left: 14.3%;
        width: 42.8%; /* (20 - 5)% out of 35% */
        height: 100%;
        background: #FCE8B2;
    }

    .scale-track-segment-3 {
        position: absolute;
        left: 57.1%;
        width: 42.9%;
        height: 100%;
        background: #F8D7BE;
        border-top-right-radius: 9999px;
        border-bottom-right-radius: 9999px;
    }

    .scale-indicator-pin {
        position: absolute;
        top: 50%;
        transform: translate(-50%, -50%);
        width: 22px;
        height: 22px;
        background: var(--text-charcoal);
        border: 3.5px solid var(--surface-white);
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        border-radius: 50%;
        z-index: 10;
        transition: left 0.3s ease;
    }

    .scale-indicator-callout {
        position: absolute;
        top: 26px;
        left: 50%;
        transform: translateX(-50%);
        background: var(--text-charcoal);
        color: #ffffff;
        font-family: var(--font-mono);
        font-size: 0.74rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 4px;
        white-space: nowrap;
    }

    .scale-axis-ticks {
        display: flex;
        justify-content: space-between;
        font-family: var(--font-mono);
        font-size: 0.68rem;
        color: var(--text-muted);
        margin-top: 0.75rem;
    }

    /* === 4. Secondary Scientific Measurement Cards === */
    .measurements-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.25rem;
        margin-bottom: 2.2rem;
    }

    .metric-instrument-card {
        background: var(--surface-white);
        border: 1px solid var(--border-gray);
        border-radius: 10px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.02);
    }

    .metric-instrument-label {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
    }

    .metric-instrument-value {
        font-family: var(--font-mono);
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text-charcoal);
        line-height: 1.1;
    }

    .metric-instrument-unit {
        font-size: 0.95rem;
        font-weight: 500;
        color: var(--text-muted);
        margin-left: 4px;
    }

    .metric-instrument-sub {
        font-family: var(--font-body);
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 0.4rem;
        line-height: 1.4;
    }

    /* === 5. Geometry Visualization Section === */
    .section-title-wrap {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .section-eyebrow {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent-olive);
        font-weight: 700;
    }

    .section-heading {
        font-family: var(--font-sans);
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        color: var(--text-charcoal);
        margin: 0.2rem 0 0.3rem 0;
    }

    /* Compact Visualization Controls */
    .controls-strip {
        background: var(--surface-white);
        border: 1px solid var(--border-gray);
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        flex-wrap: wrap;
    }

    /* Subtle Legend */
    .legend-strip {
        display: flex;
        align-items: center;
        gap: 1.4rem;
        flex-wrap: wrap;
        padding: 0.5rem 0.2rem 1rem 0.2rem;
        font-family: var(--font-mono);
        font-size: 0.75rem;
        color: var(--text-secondary);
    }

    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
    }

    .legend-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        display: inline-block;
    }

    .panel-frame {
        background: var(--surface-white);
        border: 1px solid var(--border-gray);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }

    .panel-tag {
        font-family: var(--font-mono);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-charcoal);
        margin-bottom: 0.75rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid var(--border-subtle);
    }

    /* === 6. Educational "HOW IT WORKS" Section === */
    .education-card {
        background: var(--surface-white);
        border: 1px solid var(--border-gray);
        border-radius: 12px;
        padding: 2.2rem 2.4rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    .pipeline-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.25rem;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
    }

    .pipeline-step {
        background: var(--surface-warm);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 1.1rem 1.2rem;
    }

    .pipeline-num {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--accent-olive);
        letter-spacing: 0.1em;
        margin-bottom: 0.35rem;
    }

    .pipeline-title {
        font-family: var(--font-sans);
        font-size: 0.92rem;
        font-weight: 700;
        color: var(--text-charcoal);
        margin-bottom: 0.35rem;
    }

    .pipeline-desc {
        font-family: var(--font-body);
        font-size: 0.8rem;
        color: var(--text-secondary);
        line-height: 1.45;
    }

    .formula-slab {
        background: var(--surface-warm);
        border: 1px solid var(--border-gray);
        border-radius: 8px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.4rem;
    }

    .formula-label {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin-bottom: 0.75rem;
    }

    .formula-equation {
        font-family: var(--font-mono);
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-charcoal);
        margin-bottom: 0.75rem;
    }

    .formula-variables {
        font-family: var(--font-mono);
        font-size: 0.8rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }

    .formula-note {
        font-family: var(--font-body);
        font-size: 0.82rem;
        color: var(--text-muted);
        margin-top: 0.75rem;
        line-height: 1.4;
    }

    .humorous-line {
        font-family: var(--font-serif);
        font-style: italic;
        font-size: 1.05rem;
        color: var(--text-secondary);
        text-align: center;
        padding-top: 0.5rem;
    }

    /* ===========================================================================
       7. ULTRA-OFFICIAL GOVERNMENTAL / BUREAU CERTIFICATE TEMPLATE
       =========================================================================== */
    .cert-grand-wrapper {
        background: #FDFBF7;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 10px 35px rgba(23, 23, 23, 0.06);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    .cert-official-frame {
        border: 4px double #854D0E;
        background: #FFFDF9;
        position: relative;
        padding: 3rem 3.5rem;
        box-shadow: inset 0 0 0 2px #FEF3C7, inset 0 0 0 8px #FFFDF9, inset 0 0 0 10px #CA8A04;
    }

    /* Ornate Corner Elements */
    .cert-corner-ornament {
        position: absolute;
        width: 32px;
        height: 32px;
        border: 3px solid #854D0E;
    }
    .cert-corner-tl { top: 14px; left: 14px; border-right: none; border-bottom: none; }
    .cert-corner-tr { top: 14px; right: 14px; border-left: none; border-bottom: none; }
    .cert-corner-bl { bottom: 14px; left: 14px; border-right: none; border-top: none; }
    .cert-corner-br { bottom: 14px; right: 14px; border-left: none; border-top: none; }

    .cert-official-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .cert-official-emblem {
        font-size: 3rem;
        line-height: 1;
        margin-bottom: 0.5rem;
        display: inline-block;
        filter: drop-shadow(0 4px 10px rgba(180, 83, 9, 0.35));
    }

    .cert-supra-motto {
        font-family: var(--font-diploma);
        font-size: 0.76rem;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: #854D0E;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .cert-main-bureau-title {
        font-family: var(--font-diploma);
        font-size: 1.6rem;
        font-weight: 900;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #171717;
        margin: 0;
        line-height: 1.2;
    }

    .cert-bureau-subdivision {
        font-family: var(--font-mono);
        font-size: 0.75rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #78716C;
        margin-top: 0.35rem;
    }

    .cert-reg-code-bar {
        font-family: var(--font-mono);
        font-size: 0.74rem;
        color: #854D0E;
        background: #FEF9C3;
        border: 1px dashed #CA8A04;
        display: inline-block;
        padding: 0.25rem 1rem;
        margin-top: 0.6rem;
        border-radius: 4px;
        letter-spacing: 0.08em;
    }

    .cert-filigree-divider {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1.2rem;
        margin: 1.5rem auto;
        color: #CA8A04;
        font-size: 0.95rem;
    }
    .cert-filigree-line {
        height: 1.5px;
        background: linear-gradient(90deg, transparent, #CA8A04, transparent);
        width: 140px;
    }

    .cert-solemn-heading {
        font-family: var(--font-cert-head);
        font-size: 2.35rem;
        font-weight: 900;
        color: #B45309 !important;
        background: linear-gradient(135deg, #92400E 0%, #D97706 40%, #CA8A04 70%, #78350F 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        text-align: center;
        margin: 0.25rem 0;
        filter: drop-shadow(0 1px 2px rgba(180, 83, 9, 0.25));
    }

    .cert-accord-subtitle {
        font-family: var(--font-serif);
        font-style: italic;
        font-size: 0.98rem;
        color: #57534E;
        text-align: center;
        margin-bottom: 1.6rem;
    }

    .cert-proclamation-text {
        font-family: var(--font-serif);
        font-size: 1.06rem;
        line-height: 1.7;
        color: #292524;
        text-align: justify;
        max-width: 820px;
        margin: 0 auto 1.8rem auto;
    }

    /* Official Credential Ledger Table */
    .cert-ledger-table {
        width: 100%;
        max-width: 820px;
        margin: 0 auto 2rem auto;
        border-collapse: collapse;
        border: 2px solid #854D0E;
        background: #FFFFFF;
    }

    .cert-ledger-table td {
        padding: 0.75rem 1rem;
        border: 1px solid #D6D3D1;
        font-family: var(--font-mono);
        font-size: 0.82rem;
    }

    .cert-ledger-key {
        background: #FDFBF7;
        font-weight: 700;
        color: #44403C;
        letter-spacing: 0.05em;
        width: 32%;
    }

    .cert-ledger-val {
        color: #171717;
        font-weight: 600;
    }

    .cert-ledger-highlight {
        background: #FFFBEB !important;
    }

    .cert-stamp-badge-official {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border: 1.5px solid #854D0E;
        background: #FEF3C7;
        color: #78350F;
        font-weight: 800;
        font-family: var(--font-mono);
        letter-spacing: 0.1em;
        border-radius: 4px;
        text-transform: uppercase;
    }

    /* Signatures & Seal Section */
    .cert-sign-wrapper {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        max-width: 820px;
        margin: 2.2rem auto 1.5rem auto;
        padding-top: 1rem;
    }

    .cert-signature-box {
        text-align: center;
        width: 250px;
    }

    .cert-script-sign {
        font-family: var(--font-script);
        font-size: 2.2rem;
        color: #1C1917;
        line-height: 1;
        margin-bottom: 0.3rem;
    }

    .cert-signature-rule {
        height: 1px;
        background: #44403C;
        width: 100%;
        margin-bottom: 0.4rem;
    }

    .cert-signatory-name {
        font-family: var(--font-diploma);
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #171717;
    }

    .cert-signatory-title {
        font-family: var(--font-serif);
        font-style: italic;
        font-size: 0.8rem;
        color: #57534E;
    }

    /* Embossed Gold Foil Seal */
    .cert-seal-embossed {
        position: relative;
        width: 115px;
        height: 115px;
        border-radius: 50%;
        background: radial-gradient(circle, #FDE68A 0%, #F59E0B 70%, #B45309 100%);
        box-shadow: 0 4px 14px rgba(180, 83, 9, 0.45), inset 0 0 0 3px #FEF3C7, inset 0 0 0 5px #B45309;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #78350F;
        text-align: center;
        padding: 8px;
        user-select: none;
    }

    .cert-seal-embossed::after {
        content: "";
        position: absolute;
        bottom: -22px;
        left: 36px;
        width: 18px;
        height: 30px;
        background: #991B1B;
        clip-path: polygon(0 0, 100% 0, 100% 100%, 50% 80%, 0 100%);
        box-shadow: 1px 2px 5px rgba(0,0,0,0.2);
    }
    .cert-seal-embossed::before {
        content: "";
        position: absolute;
        bottom: -22px;
        right: 36px;
        width: 18px;
        height: 30px;
        background: #B91C1C;
        clip-path: polygon(0 0, 100% 0, 100% 100%, 50% 80%, 0 100%);
        box-shadow: 1px 2px 5px rgba(0,0,0,0.2);
    }

    .cert-seal-txt-top {
        font-family: var(--font-diploma);
        font-size: 0.52rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        line-height: 1;
        margin-bottom: 2px;
    }
    .cert-seal-fruit {
        font-size: 1.4rem;
        line-height: 1;
        margin: 2px 0;
    }
    .cert-seal-txt-bot {
        font-family: var(--font-mono);
        font-size: 0.52rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        line-height: 1;
    }

    /* Dry Legal Disclaimer Proviso */
    .cert-solemn-caveat {
        max-width: 820px;
        margin: 2rem auto 0 auto;
        padding-top: 1rem;
        border-top: 1px dashed #D6D3D1;
        font-family: var(--font-serif);
        font-style: italic;
        font-size: 0.88rem;
        color: #78716C;
        text-align: center;
        line-height: 1.45;
    }

    /* Custom Button Overrides */
    .stButton > button {
        background: #FFFFFF !important;
        color: #171717 !important;
        border: 1.5px solid #D6CEBE !important;
        border-radius: 9px !important;
        font-family: var(--font-sans) !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        padding: 0.65rem 1.1rem !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton > button p, 
    .stButton > button div, 
    .stButton > button span {
        color: #171717 !important;
        font-weight: 700 !important;
    }
    .stButton > button:hover {
        background: #FEF3C7 !important;
        border-color: #D4AF37 !important;
        color: #78350F !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(212, 175, 55, 0.22) !important;
    }
    .stButton > button:hover p,
    .stButton > button:hover div,
    .stButton > button:hover span {
        color: #78350F !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #F4B400, #EAB308) !important;
        color: #171717 !important;
        border: 2px solid #CA8A04 !important;
        border-radius: 8px !important;
        font-family: var(--font-sans) !important;
        font-size: 0.9rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.06em !important;
        padding: 0.7rem 1.6rem !important;
        box-shadow: 0 4px 14px rgba(244, 180, 0, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(244, 180, 0, 0.45) !important;
        background: linear-gradient(135deg, #F59E0B, #D97706) !important;
        color: #FFFFFF !important;
    }

    /* Responsive Grid Fallbacks */
    @media (max-width: 768px) {
        .brand-hero {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.75rem;
        }
        .measurements-grid {
            grid-template-columns: 1fr;
        }
        .pipeline-grid {
            grid-template-columns: 1fr;
        }
        .score-display-row {
            flex-direction: column;
        }
        .cert-official-frame {
            padding: 1.5rem 1rem;
        }
        .cert-sign-wrapper {
            flex-direction: column;
            align-items: center;
            gap: 1.5rem;
        }

        /* The certificate ledger is a 4-column table (key, value, key, value).
           At phone width that needs ~476px inside a ~295px frame, so the right
           half of every row was being cut off. Re-flow each row as a 2-column
           grid: the four cells wrap into two key/value pairs, one per line. */
        .cert-ledger-table,
        .cert-ledger-table tbody {
            display: block;
            width: 100%;
        }
        .cert-ledger-table tr {
            display: grid;
            grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr);
        }
        .cert-ledger-table td {
            display: block;
            min-width: 0;
            padding: 0.5rem 0.6rem;
            font-size: 0.72rem;
            overflow-wrap: anywhere;
        }

        /* Oversized display type, scaled for a narrow column */
        .brand-title { font-size: 1.55rem; }
        .brand-mark { font-size: 2.4rem; }
        .ingestion-title { font-size: 1.25rem; }
        .score-giant-number { font-size: 3rem; }
        .score-giant-percent { font-size: 1.5rem; }
        .cert-solemn-heading { font-size: 1.6rem; }
        .cert-main-bureau-title { font-size: 1.02rem; }
        .cert-script-sign { font-size: 1.6rem; }

        /* Fixed-width certificate furniture must not exceed the screen */
        .cert-signature-box { width: 100%; max-width: 250px; }
        .cert-filigree-line { max-width: 40vw; }

        /* A nowrap callout has nowhere to go on a narrow screen */
        .scale-indicator-callout { white-space: normal; }

        /* Reclaim horizontal space Streamlit reserves for desktop gutters */
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        img { max-width: 100%; height: auto; }
    }

    /* === Tablet / small laptop: two columns instead of three === */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main .block-container {
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
        }
        .measurements-grid { grid-template-columns: repeat(2, 1fr); }
        .pipeline-grid { grid-template-columns: repeat(2, 1fr); }
        .brand-title { font-size: 1.95rem; }
        .score-giant-number { font-size: 3.5rem; }
        .cert-solemn-heading { font-size: 1.95rem; }
        .cert-ledger-table td { font-size: 0.76rem; padding: 0.6rem 0.7rem; }
    }

    /* === Small phones === */
    @media (max-width: 420px) {
        .brand-title { font-size: 1.32rem; }
        .score-giant-number { font-size: 2.5rem; }
        .score-giant-percent { font-size: 1.25rem; }
        .cert-solemn-heading { font-size: 1.32rem; }
        .cert-main-bureau-title { font-size: 0.9rem; }
        .cert-script-sign { font-size: 1.35rem; }
        .cert-ledger-table td { font-size: 0.66rem; padding: 0.42rem 0.45rem; }
        .cert-official-frame { padding: 1rem 0.6rem; }
        .main .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
    }
    </style>
    """
)


# ---------------------------------------------------------------------------
# Main Application Controller
# ---------------------------------------------------------------------------
def main() -> None:

    # ── 1. Premium Brand Header with Logo Image ──
    # Served at 2x its 340px display width: visually identical, ~90% fewer bytes
    # re-sent on every rerun.
    _logo_src = _asset_uri("chill_ethakka_logo.jpg", max_px=680)

    st.html(
        f"""
        <header class="brand-hero" style="border-bottom: 1px solid #E5E0D4; padding-bottom: 1rem; margin-bottom: 1.75rem;">
            <div style="display:flex; flex-direction:column; align-items:center; gap:0.5rem;">
                <img src="{_logo_src}" alt="CHILL ETHAKKA Logo"
                     style="max-width: 340px; width: 100%; height: auto; display: block;" />
                <div class="brand-status" style="margin-top:0.25rem;">
                    <span class="status-dot"></span>
                    <span class="status-text">ANALYSIS ENGINE READY</span>
                </div>
            </div>
        </header>
        """
    )

    # ── 2. Image Ingestion Station with Malayalam Directive ──
    st.html(
        """
        <div class="ingestion-card">
            <div class="ingestion-eyebrow">OPTICAL SPECIMEN INGESTION</div>
            <h2 class="ingestion-title">പഴത്തിൻ്റെ ഫോട്ടോ ഇവിടെ ഇടുക</h2>
            <p class="ingestion-prompt">Upload a clear photograph containing one banana on a contrasting surface.</p>
            <div class="ingestion-formats">SUPPORTED FORMATS: JPG · JPEG · PNG · WEBP</div>
        </div>
        """
    )

    uploaded_file = st.file_uploader(
        "Upload Banana Specimen",
        type=["png", "jpg", "jpeg", "webp", "avif"],
        label_visibility="collapsed",
        help="Upload an optical image containing a single banana.",
    )

    # Preset sample quick buttons for instant testing
    st.markdown(
        "<div style='text-align:center; font-family: var(--font-mono); font-size: 0.76rem; font-weight:700; color: var(--text-secondary); margin: 0.75rem 0 0.4rem 0;'>OR TEST INSTANTLY WITH A REFERENCE SPECIMEN:</div>",
        unsafe_allow_html=True,
    )
    sample_col0, sample_col1, sample_col2, sample_col3 = st.columns(4)
    if "selected_sample" not in st.session_state:
        st.session_state["selected_sample"] = None

    with sample_col0:
        if st.button("🍌 Real Specimen (മോഡൽ പഴം)", use_container_width=True):
            st.session_state["selected_sample"] = "samples/model_banana.png"
    with sample_col1:
        if st.button("📏 Straight (നേർരേഖ)", use_container_width=True):
            st.session_state["selected_sample"] = "samples/straight_banana.png"
    with sample_col2:
        if st.button("🌙 Curved (സാധാരണ)", use_container_width=True):
            st.session_state["selected_sample"] = "samples/curved_banana.png"
    with sample_col3:
        if st.button("🪃 Boomerang (തീവ്രം)", use_container_width=True):
            st.session_state["selected_sample"] = "samples/highly_curved_banana.png"

    selected_image: Optional[np.ndarray] = None
    selected_alpha: Optional[np.ndarray] = None
    image_name: str = "specimen"

    if uploaded_file is not None:
        st.session_state["selected_sample"] = None
        filename_lower = uploaded_file.name.lower()
        file_mime = getattr(uploaded_file, "type", "")
        
        # Graceful AVIF detection & rejection without exposing raw stream errors
        if filename_lower.endswith(".avif") or file_mime == "image/avif":
            st.warning("AVIF images aren't supported yet. Please upload JPG, PNG, or WEBP.")
            selected_image = None
        else:
            try:
                selected_image, selected_alpha = load_image(uploaded_file, return_alpha=True)
                image_name = uploaded_file.name
            except Exception as e:
                st.warning(f"Could not process image '{uploaded_file.name}': {e}")
                selected_image = None
    elif st.session_state.get("selected_sample"):
        sample_path = st.session_state["selected_sample"]
        if os.path.exists(sample_path):
            try:
                selected_image, selected_alpha = load_image(sample_path, return_alpha=True)
                image_name = os.path.basename(sample_path)
            except Exception as e:
                st.warning(f"Could not process sample '{sample_path}': {e}")
                selected_image = None

    # ── 3. Processing & Results Pipeline ──
    if selected_image is not None:
        with st.spinner("Executing geometric curvature pipeline..."):
            result: CurvatureAnalysisResult = analyze_banana(
                selected_image,
                segmentation_method="auto",
                hsv_lower=DEFAULT_HSV_LOWER,
                hsv_upper=DEFAULT_HSV_UPPER,
                smoothing=None,  # adaptive smoothing (was: float(SPLINE_SMOOTHING))
                straight_threshold=float(STRAIGHT_THRESHOLD),
                curved_threshold=float(CURVED_THRESHOLD),
                morph_kernel_size=7,
                alpha_mask=selected_alpha,
            )

        if not result.success:
            st.error(f"Analysis Notice: {result.message}. Please provide a clear, unobstructed single banana image.")
            return

        # ── 4. Main Result: Dominant Curve Score ──
        # Dynamic calculation of scale position (normalized to a 0% - 35% standard range)
        max_scale = max(35.0, float(result.curve_score) * 1.1)
        scale_pct = float(np.clip((result.curve_score / max_scale) * 100.0, 1.5, 98.5))

        badge_class = (
            "badge-straight"
            if result.category == CATEGORY_STRAIGHT
            else ("badge-curved" if result.category == CATEGORY_CURVED else "badge-highly-curved")
        )

        st.html(
            f"""
            <div class="score-hero-card">
                <div class="score-eyebrow">PRIMARY METRIC</div>
                <div class="score-label">CURVE SCORE</div>
                <div class="score-display-row">
                    <div>
                        <span class="score-giant-number">{result.curve_score:.2f}</span>
                        <span class="score-giant-percent">%</span>
                    </div>
                    <div>
                        <span class="classification-badge {badge_class}">{result.category.upper()}</span>
                    </div>
                </div>
                
                <div class="scale-container">
                    <div class="scale-boundary-labels">
                        <span>STRAIGHT</span>
                        <span>CURVED</span>
                        <span>HIGHLY CURVED</span>
                    </div>
                    <div class="scale-track-wrapper">
                        <div class="scale-track-segment-1"></div>
                        <div class="scale-track-segment-2"></div>
                        <div class="scale-track-segment-3"></div>
                        <div class="scale-indicator-pin" style="left: {scale_pct:.1f}%;">
                            <div class="scale-indicator-callout">{result.curve_score:.2f}%</div>
                        </div>
                    </div>
                    <div class="scale-axis-ticks">
                        <span>0%</span>
                        <span>5%</span>
                        <span>20%</span>
                        <span>{max_scale:.0f}%+</span>
                    </div>
                </div>
            </div>
            """
        )

        # ── 5. Secondary Scientific Measurement Cards ──
        m1, m2, m3 = st.columns(3)
        with m1:
            st.html(
                f"""
                <div class="metric-instrument-card">
                    <div class="metric-instrument-label">ARC LENGTH (L)</div>
                    <div class="metric-instrument-value">{result.path_length:.1f}<span class="metric-instrument-unit">px</span></div>
                    <div class="metric-instrument-sub">Centerline path integral along smoothed spline</div>
                </div>
                """
            )
        with m2:
            st.html(
                f"""
                <div class="metric-instrument-card">
                    <div class="metric-instrument-label">CHORD DISTANCE (D)</div>
                    <div class="metric-instrument-value">{result.chord_distance:.1f}<span class="metric-instrument-unit">px</span></div>
                    <div class="metric-instrument-sub">Direct Euclidean distance between endpoints</div>
                </div>
                """
            )
        with m3:
            st.html(
                f"""
                <div class="metric-instrument-card">
                    <div class="metric-instrument-label">MAX DEFLECTION (δ)</div>
                    <div class="metric-instrument-value">{result.max_deflection:.1f}<span class="metric-instrument-unit">px</span></div>
                    <div class="metric-instrument-sub">Maximum perpendicular sagitta to chord</div>
                </div>
                """
            )

        # ── 6. Geometry Visualization & Controls ──
        st.html(
            """
            <div class="section-title-wrap">
                <div class="section-eyebrow">MORPHOLOGICAL INSPECTION</div>
                <div class="section-heading">GEOMETRY VISUALIZATION</div>
            </div>
            """
        )

        # Compact Controls Bar
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5, ctrl_col6 = st.columns([1.5, 1, 1, 1, 1, 1])
        with ctrl_col1:
            st.markdown("<div style='font-family: var(--font-mono); font-size: 0.78rem; font-weight: 700; padding-top: 6px;'>CONTROLS:</div>", unsafe_allow_html=True)
        with ctrl_col2:
            show_cont = st.checkbox("Contour", value=True)
        with ctrl_col3:
            show_center = st.checkbox("Centerline", value=True)
        with ctrl_col4:
            show_chord = st.checkbox("Chord", value=True)
        with ctrl_col5:
            show_ends = st.checkbox("Endpoints", value=True)
        with ctrl_col6:
            show_defl = st.checkbox("Deflection", value=True)

        # Subtle Legend
        st.html(
            """
            <div class="legend-strip">
                <span class="legend-item"><span class="legend-dot" style="background: #28DC28;"></span>Contour</span>
                <span class="legend-item"><span class="legend-dot" style="background: #00D7FF;"></span>Centerline</span>
                <span class="legend-item"><span class="legend-dot" style="background: #FF8C00;"></span>Chord</span>
                <span class="legend-item"><span class="legend-dot" style="background: #EAB308;"></span>Endpoints</span>
                <span class="legend-item"><span class="legend-dot" style="background: #FF32D7;"></span>Max Deflection</span>
            </div>
            """
        )

        # Generate scientific overlay
        annotated_img = create_annotated_overlay(
            selected_image,
            result,
            show_contour=show_cont,
            show_centerline=show_center,
            show_chord=show_chord,
            show_endpoints=show_ends,
            show_deflection=show_defl,
            show_info_card=False,
            show_curvature_center=False,
        )

        # Display two panels: ORIGINAL (COLOR) and GEOMETRY ANALYSIS
        panel_color, panel_analysis = st.columns(2)
        with panel_color:
            st.html(
                """
                <div class="panel-tag">ORIGINAL (COLOR)</div>
                """
            )
            st.image(selected_image, use_container_width=True)
        with panel_analysis:
            st.html(
                """
                <div class="panel-tag">GEOMETRY ANALYSIS</div>
                """
            )
            st.image(annotated_img, use_container_width=True)

        # ── 7. Diagnostic Breakdown (Matplotlib 4-Stage Figure) ──
        with st.expander("DIAGNOSTIC BREAKDOWN", expanded=False):
            diag_fig = create_diagnostic_figure(selected_image, result, theme="light")
            st.pyplot(diag_fig, use_container_width=True)

        # ── 7.5. Absurd Botanical Metrology: Pazham Jathakam & Boomerang Simulator ──
        st.html(
            """
            <div class="section-title-wrap" style="margin-top: 2rem;">
                <div class="section-eyebrow">OCCULT & BALLISTIC METROLOGY DIVISION</div>
                <div class="section-heading">പഴ ജാതകവും എയറോഡൈനാമിക്സും (HOROSCOPE & FLIGHT)</div>
            </div>
            """
        )
        absurd_hub_html = render_absurd_interactive_hub({
            "curve_score": float(result.curve_score),
            "category": str(result.category),
            "path_length": float(result.path_length),
            "chord_distance": float(result.chord_distance),
            "max_deflection": float(result.max_deflection),
        })
        components.html(absurd_hub_html, height=720, scrolling=False)

        # ── 8. Grand Completely Official Laboratory Certificate ──
        cert_id = f"IBBC-SPEC-{abs(hash(image_name)) % 1000000:06d}"
        timestamp_str = time.strftime("%d %B %Y • %H:%M:%S UTC")

        st.html(
            f"""
            <div class="cert-grand-wrapper">
                <div class="cert-official-frame">
                    <div class="cert-corner-ornament cert-corner-tl"></div>
                    <div class="cert-corner-ornament cert-corner-tr"></div>
                    <div class="cert-corner-ornament cert-corner-bl"></div>
                    <div class="cert-corner-ornament cert-corner-br"></div>

                    <div class="cert-official-header">
                        <div class="cert-official-emblem">🍌</div>
                        <div class="cert-supra-motto">★ REPUBLIC OF BOTANICAL METROLOGY ★</div>
                        <div class="cert-main-bureau-title">THE INTERNATIONAL BUREAU OF BANANA CURVATURE</div>
                        <div class="cert-bureau-subdivision">DIRECTORATE GENERAL OF PHYSICAL GEOMETRY • HER MAJESTY'S CALIBRATED RECORD</div>
                        <div class="cert-reg-code-bar">OFFICIAL REGISTRATION: <b>12-SEP-2026</b> &nbsp;|&nbsp; DISPATCH CODIFICATION: <b>CLASS-A1</b></div>
                    </div>

                    <div class="cert-filigree-divider">
                        <div class="cert-filigree-line"></div>
                        <span>❖ &nbsp; ❖ &nbsp; ❖</span>
                        <div class="cert-filigree-line"></div>
                    </div>

                    <div class="cert-solemn-heading">CHILL ETHAKKA</div>


                    <p class="cert-proclamation-text">
                        <b>BE IT KNOWN UNTO ALL LEARNED PERSONS AND GOVERNING BODIES</b>, that the natural botanical specimen designated <i>"{image_name}"</i> was duly submitted to the Division of Physical Metrology, whereupon rigorous computer-vision medial-axis skeletonization, non-Euclidean chordal arc-length integration, and perpendicular sagitta deflection metrics were meticulously analyzed, determined, and recorded in the official archives:
                    </p>

                    <table class="cert-ledger-table">
                        <tbody>
                            <tr>
                                <td class="cert-ledger-key">OFFICIAL BANANA ID</td>
                                <td class="cert-ledger-val font-mono">{cert_id}</td>
                                <td class="cert-ledger-key">DATE OF AUDIT</td>
                                <td class="cert-ledger-val font-mono">{timestamp_str}</td>
                            </tr>
                            <tr class="cert-ledger-highlight">
                                <td class="cert-ledger-key">CURVE SCORE (PRIMARY)</td>
                                <td class="cert-ledger-val font-mono" style="font-size: 1.15rem; color: #9A3412;"><b>{result.curve_score:.2f}%</b></td>
                                <td class="cert-ledger-key">OFFICIAL CLASSIFICATION</td>
                                <td class="cert-ledger-val"><span class="cert-stamp-badge-official">{result.category.upper()}</span></td>
                            </tr>
                            <tr>
                                <td class="cert-ledger-key">ARC LENGTH (L)</td>
                                <td class="cert-ledger-val font-mono">{result.path_length:.1f} px</td>
                                <td class="cert-ledger-key">CHORD DISTANCE (D)</td>
                                <td class="cert-ledger-val font-mono">{result.chord_distance:.1f} px</td>
                            </tr>
                            <tr>
                                <td class="cert-ledger-key">MAX DEFLECTION (δ)</td>
                                <td class="cert-ledger-val font-mono">{result.max_deflection:.1f} px</td>
                                <td class="cert-ledger-key">GEOMETRIC STATUS</td>
                                <td class="cert-ledger-val font-mono" style="color: #15803D; font-weight: 700;">CERTIFIED {result.category.upper()}</td>
                            </tr>
                        </tbody>
                    </table>

                    <div class="cert-sign-wrapper">
                        <div class="cert-signature-box">
                            <div class="cert-script-sign">Dr.Kumar Kubera</div>
                            <div class="cert-signature-rule"></div>
                            <div class="cert-signatory-name">DR. Kumar kubera, Ph.D.</div>
                            <div class="cert-signatory-title">High Commissioner of Curvilinear Produce</div>
                        </div>

                        <div class="cert-seal-embossed">
                            <div class="cert-seal-txt-top">IBBC BUREAU</div>
                            <div class="cert-seal-fruit">🍌</div>
                            <div class="cert-seal-txt-bot">OFFICIAL SEAL</div>
                        </div>

                        <div class="cert-signature-box">
                            <div class="cert-script-sign">Prof. Pachalam Bhasy</div>
                            <div class="cert-signature-rule"></div>
                            <div class="cert-signatory-name">PROF. Pachalam Bhasy, D.Sc.</div>
                            <div class="cert-signatory-title">Keeper of the Standard Banana Spline</div>
                        </div>
                    </div>

                    <div class="cert-solemn-caveat">
                        <b>LEGAL DISCLAIMER & STATUTORY PROVISO:</b> This certificate has absolutely no practical value. It is issued strictly as a solemn celebration of absurd geometric precision and carries zero commercial, legal, culinary, or nutritional validity in any jurisdiction on Earth.
                    </div>
                </div>
            </div>
            """
        )

        cert_payload = {
            "certificate_id": cert_id,
            "timestamp": timestamp_str,
            "specimen": image_name,
            "curve_score": round(result.curve_score, 4),
            "category": result.category,
            "arc_length_px": round(result.path_length, 2),
            "chord_distance_px": round(result.chord_distance, 2),
            "max_deflection_px": round(result.max_deflection, 2),
            "status": f"CERTIFIED {result.category.upper()}",
            "authority": "International Bureau of Banana Curvature",
            "disclaimer": "This certificate has absolutely no practical value.",
        }

        col_left, col_btn, col_right = st.columns([1, 2, 1])
        with col_btn:
            st.download_button(
                label="DOWNLOAD OFFICIAL CERTIFICATE (JSON)",
                data=json.dumps(cert_payload, indent=2),
                file_name=f"{cert_id}.json",
                mime="application/json",
                use_container_width=True,
            )




if __name__ == "__main__":
    main()
