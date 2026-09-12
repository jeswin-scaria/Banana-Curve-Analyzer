"""
Image preprocessing and segmentation module for Banana Curve Analyzer.
Provides robust methods to isolate the single banana from various backgrounds
including wooden tables, cutting boards, countertops, and white surfaces.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import io
import numpy as np
import cv2
from PIL import Image, ImageOps

from .config import (
    DEFAULT_HSV_LOWER,
    DEFAULT_HSV_UPPER,
    MIN_CONTOUR_AREA,
    SALIENCY_MIN_SATURATION_GATE,
    SEGMENTATION_CANDIDATES_PER_STRATEGY,
    SEGMENTATION_METHOD_PRIORITY,
)
from .quality import score_contour_plausibility


def _split_rgba(rgba: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Split an RGB(A) array into an RGB image and an optional alpha mask.

    Transparent pixels are composited onto white rather than having their alpha
    simply dropped: PNG exporters commonly leave arbitrary colour (very often
    pure black) underneath fully-transparent pixels, and dropping alpha turns
    that into a hard black background that no colour-based segmentation can be
    expected to interpret sensibly.

    The alpha channel is returned separately because, when a cut-out image
    provides one, it IS the object's silhouette -- far more reliable than any
    colour heuristic we could infer from the pixels.
    """
    if rgba.ndim == 2:
        return cv2.cvtColor(rgba, cv2.COLOR_GRAY2RGB), None
    if rgba.shape[2] == 3:
        return rgba, None

    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3]
    # Only treat alpha as meaningful if the image is actually partly transparent.
    if np.all(alpha >= 250):
        return rgba[:, :, :3].copy(), None

    a = (alpha.astype(np.float32) / 255.0)[..., None]
    composited = np.clip(rgb * a + 255.0 * (1.0 - a), 0, 255).astype(np.uint8)
    return composited, alpha.copy()


def load_image(
    source: Union[str, bytes, io.BytesIO, np.ndarray],
    return_alpha: bool = False,
):
    """
    Load an image from a file path, byte stream, or existing array into an RGB numpy array.
    Supports Streamlit UploadedFile, PIL decoding, EXIF orientation auto-correction, and OpenCV fallbacks.

    Args:
        source: File path, file-like byte buffer, or numpy array.
        return_alpha: When True, return ``(rgb, alpha)`` where ``alpha`` is the
            source's alpha channel (uint8) if it had meaningful transparency,
            otherwise None. When False (default) only the RGB image is returned,
            preserving the original call signature.

    Returns:
        np.ndarray, or (np.ndarray, Optional[np.ndarray]) when return_alpha=True.
    """
    rgba: Optional[np.ndarray] = None

    if isinstance(source, np.ndarray):
        rgba = source
    else:
        raw_bytes: Optional[bytes] = None

        if isinstance(source, (bytes, bytearray)):
            raw_bytes = bytes(source)
        elif hasattr(source, "getvalue"):
            try:
                raw_bytes = source.getvalue()
            except Exception:
                pass

        if raw_bytes is None and hasattr(source, "read"):
            try:
                source.seek(0)
                raw_bytes = source.read()
            except Exception:
                pass

        if raw_bytes is not None and len(raw_bytes) > 0:
            # 1. Primary decoder: Pillow with EXIF auto-rotation
            try:
                pil_img = Image.open(io.BytesIO(raw_bytes))
                pil_img = ImageOps.exif_transpose(pil_img)
                has_alpha = pil_img.mode in ("RGBA", "LA") or (
                    pil_img.mode == "P" and "transparency" in pil_img.info
                )
                pil_img = pil_img.convert("RGBA" if has_alpha else "RGB")
                rgba = np.array(pil_img)
            except Exception:
                rgba = None

            # 2. Secondary decoder: OpenCV imdecode (IMREAD_UNCHANGED keeps alpha)
            if rgba is None:
                try:
                    nparr = np.frombuffer(raw_bytes, np.uint8)
                    decoded = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                    if decoded is not None:
                        if decoded.ndim == 3 and decoded.shape[2] == 4:
                            rgba = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
                        elif decoded.ndim == 3:
                            rgba = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                        else:
                            rgba = cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
                except Exception:
                    rgba = None

        if rgba is None and isinstance(source, str):
            try:
                pil_img = Image.open(source)
                pil_img = ImageOps.exif_transpose(pil_img)
                has_alpha = pil_img.mode in ("RGBA", "LA") or (
                    pil_img.mode == "P" and "transparency" in pil_img.info
                )
                pil_img = pil_img.convert("RGBA" if has_alpha else "RGB")
                rgba = np.array(pil_img)
            except Exception:
                decoded = cv2.imread(source, cv2.IMREAD_UNCHANGED)
                if decoded is None:
                    raise FileNotFoundError(f"Could not load image at path: {source}")
                if decoded.ndim == 3 and decoded.shape[2] == 4:
                    rgba = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
                elif decoded.ndim == 3:
                    rgba = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                else:
                    rgba = cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)

    if rgba is None:
        raise ValueError("Could not decode image from the provided source.")

    rgb, alpha = _split_rgba(rgba)
    return (rgb, alpha) if return_alpha else rgb


def compute_smart_banana_saliency(image_rgb: np.ndarray) -> np.ndarray:
    """
    Compute a normalized banana saliency response combining CIE LAB and HSV color spaces.
    Discriminates vibrant yellow/green bananas from wooden tables, shadows, and neutral backgrounds.

    - In CIE LAB:
        b* channel measures blue (-128) to yellow (+127). Bananas have high b* (> 150 in uint8).
        a* channel measures green (-128) to red (+127). Bananas have negative or near-zero a* (<= 135 in uint8).
        Wood surfaces have lower b* (130-150) and higher a* (> 135, reddish/brown).
    - In HSV:
        Yellow bananas have Hue in [18, 38] and high Saturation.
        Green bananas have Hue in [35, 85].
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    a_chan = lab[:, :, 1].astype(np.float32)
    b_chan = lab[:, :, 2].astype(np.float32)

    # Saturation is applied as a GATE, not as a continuous multiplier. The LAB
    # b*/a* strength terms below already measure how far a pixel sits from
    # neutral gray, so multiplying by saturation as well compounds two measures
    # of the same property: on a glossy, studio-lit or sunlit banana the
    # specular highlight running down the body desaturates those pixels and the
    # product collapses to ~0, tearing the highlight strip out of the mask and
    # leaving only a narrow, fully-saturated sliver of the fruit.
    colored_gate = s >= SALIENCY_MIN_SATURATION_GATE

    # 1. Ripe Yellow Component: High b* in LAB, low a* (not red/brown), and yellow hue
    yellow_b_strength = np.maximum(0.0, b_chan - 145.0)
    yellow_hue_gate = (h >= 15) & (h <= 40)
    yellow_a_gate = a_chan <= 136.0  # rejects reddish-brown wood grain
    yellow_score = yellow_b_strength * v * colored_gate * yellow_hue_gate * yellow_a_gate

    # 2. Green / Unripe Component: Green hue in HSV and low a* (green side of LAB)
    green_hue_gate = (h > 35) & (h <= 85)
    green_a_strength = np.maximum(0.0, 132.0 - a_chan)
    green_score = green_a_strength * v * colored_gate * green_hue_gate

    # Combined score
    total_score = yellow_score + (1.2 * green_score)
    max_val = np.max(total_score)

    if max_val < 1e-4:
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    normalized = np.clip((total_score / max_val) * 255.0, 0, 255).astype(np.uint8)
    return normalized


def _fill_interior_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill fully-enclosed holes in a binary mask, at any size.

    A glossy banana's specular highlight and its dark ripeness spots are both
    *interior* regions that fail a colour test but are unambiguously part of the
    fruit -- they are surrounded on all sides by banana. Morphological closing
    can only bridge gaps narrower than its kernel, so a wide highlight streak
    survives as a hole and tears the mask (and therefore the skeleton) apart.
    Flood-filling the background from outside the shape and inverting captures
    those regions regardless of how wide they are.
    """
    h, w = mask.shape[:2]
    # Pad by one pixel of guaranteed background so the flood fill always has a
    # valid outside seed even when the shape touches the image border.
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    ff_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:h + 1, 1:w + 1]
    return cv2.bitwise_or(mask, holes)


def _clean_binary_mask(raw_mask: np.ndarray, k_size: int) -> np.ndarray:
    """Morphological cleanup shared by every segmentation strategy:
    1. Opening: cuts thin bridges connecting banana to wood grain or background reflections.
    2. Closing: fills small brown spots and internal peel blemishes.
    3. Hole filling: recovers wide interior gaps (specular highlights) that
       closing cannot bridge.
    """
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, open_kernel, iterations=2)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size * 2, k_size * 2))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    return _fill_interior_holes(closed)


def _generate_candidate_masks(
    image_rgb: np.ndarray,
    method: str,
    hsv_lower: Optional[Tuple[int, int, int]],
    hsv_upper: Optional[Tuple[int, int, int]],
    alpha_mask: Optional[np.ndarray] = None,
) -> List[Tuple[str, np.ndarray]]:
    """
    Generate one raw (uncleaned) candidate mask per applicable segmentation strategy.

    For 'auto', every strategy runs unconditionally so all of them can be scored
    and compared -- rather than stopping at the first one that produces *any*
    mask, which is what allowed a plausible-looking but wrong region to win by
    default. Single explicit methods ('hsv', 'otsu', 'saturation', 'wood') only
    generate their own strategy's candidate.
    """
    candidates: List[Tuple[str, np.ndarray]] = []
    h, w = image_rgb.shape[:2]
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # A cut-out image's own alpha channel is the object's actual silhouette --
    # not an estimate of it -- so it is offered as a candidate ahead of every
    # colour heuristic. It is still shape-scored like any other candidate, so a
    # nonsense alpha channel cannot hijack the result.
    if alpha_mask is not None and alpha_mask.shape[:2] == (h, w):
        _, alpha_binary = cv2.threshold(alpha_mask, 127, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(alpha_binary) > 0:
            candidates.append(("alpha_channel", alpha_binary))

    if method in ("auto", "wood"):
        saliency = compute_smart_banana_saliency(image_rgb)
        if cv2.countNonZero(saliency) > 0:
            _, saliency_mask = cv2.threshold(saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            candidates.append(("saliency_otsu", saliency_mask))

            mask_strict = None
            pos_pixels = saliency[saliency > 0]
            if len(pos_pixels) > 50:
                ret_pos, _ = cv2.threshold(pos_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, mask_strict = cv2.threshold(saliency, max(int(ret_pos), 95), 255, cv2.THRESH_BINARY)
                candidates.append(("saliency_strict", mask_strict))

            lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
            strict_gate = (hsv[:, :, 1] >= 75) & (lab[:, :, 2] >= 150) & (lab[:, :, 1] <= 136)
            wood_mask = saliency_mask & (strict_gate.astype(np.uint8) * 255)
            candidates.append(("saliency_wood", wood_mask))

            # Seed GrabCut from the STRICT core, not the Otsu mask. A seed needs
            # precision, not recall: GrabCut grows outward from confident
            # foreground, so a small clean core is ideal, whereas the Otsu mask
            # can swallow large amounts of warm background (wood scores non-zero
            # on the colour saliency) and leave GrabCut nothing to model
            # background from.
            seed = mask_strict if mask_strict is not None and cv2.countNonZero(mask_strict) > 200 else saliency_mask
            refined = _grabcut_refine(image_rgb, seed)
            if refined is not None:
                candidates.append(("grabcut_refined", refined))

    if method in ("auto", "wood", "hsv"):
        lower = np.array(hsv_lower if hsv_lower is not None else DEFAULT_HSV_LOWER, dtype=np.uint8)
        upper = np.array(hsv_upper if hsv_upper is not None else DEFAULT_HSV_UPPER, dtype=np.uint8)
        candidates.append(("hsv_inrange", cv2.inRange(hsv, lower, upper)))

    if method in ("auto", "saturation"):
        s_channel = hsv[:, :, 1]
        _, sat_mask = cv2.threshold(s_channel, 60, 255, cv2.THRESH_BINARY)
        candidates.append(("saturation", sat_mask))

    if method in ("auto", "otsu"):
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        corners = [blurred[0, 0], blurred[0, -1], blurred[-1, 0], blurred[-1, -1]]
        avg_corner = float(np.mean(corners))
        if avg_corner > 127:
            _, gray_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, gray_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(("gray_otsu", gray_mask))

    return candidates


GRABCUT_MAX_DIMENSION = 500


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected component of a binary mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == largest).astype(np.uint8)) * 255


def _grabcut_refine(image_rgb: np.ndarray, seed_mask: np.ndarray, iterations: int = 5) -> Optional[np.ndarray]:
    """
    Refine a coarse color-based mask with GrabCut's iterative graph-cut
    foreground/background modeling (classical CV, part of standard OpenCV --
    no deep learning). Unlike a fixed global color threshold, GrabCut reasons
    about the local color *distributions* of likely foreground/background
    regions, so it can still separate the banana from a background that is
    close to it in raw color (e.g. light, warm wood) as long as the two are
    even slightly separable -- which a single fixed threshold cannot do.

    The color-based mask is used only to seed GrabCut's initial guess (eroded
    core = definite foreground, dilated ring = probable foreground, far
    outside = definite background); GrabCut then re-estimates the actual
    foreground/background boundary from the image itself.

    GrabCut's cost scales with total pixel count, and a phone photo can be
    10+ megapixels -- run it on a downsampled proxy (longest side capped at
    GRABCUT_MAX_DIMENSION) and scale the resulting mask back up, rather than
    on the full-resolution image, to stay within the pipeline's performance
    budget. The banana's silhouette doesn't need full resolution to refine.
    """
    h, w = image_rgb.shape[:2]
    if cv2.countNonZero(seed_mask) < 200:
        return None

    scale = min(1.0, GRABCUT_MAX_DIMENSION / max(h, w))
    if scale < 1.0:
        small_w, small_h = max(1, int(w * scale)), max(1, int(h * scale))
        work_img = cv2.resize(image_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
        work_seed = cv2.resize(seed_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    else:
        work_img, work_seed = image_rgb, seed_mask
    wh, ww = work_img.shape[:2]

    # Seed from the single largest connected component only. The raw colour
    # mask typically also contains scattered background specks (wood grain,
    # reflections); dilating those to build the "definitely background" ring
    # can cover the whole frame, leaving GrabCut with no background samples at
    # all -- at which point it labels the entire image foreground.
    work_seed = _largest_component(work_seed)
    if cv2.countNonZero(work_seed) < 100:
        return None

    dist = cv2.distanceTransform(work_seed, cv2.DIST_L2, 5)
    max_d = float(dist.max())
    if max_d < 1.0:
        return None

    sure_fg = (dist >= max(2.0, max_d * 0.40)).astype(np.uint8) * 255
    if cv2.countNonZero(sure_fg) < 5:
        return None

    # Probable foreground = the convex hull of the seed core, lightly dilated.
    # The seed is a high-precision but *narrow* core (a specular highlight or a
    # band of dark spots can split the fruit lengthwise), so simply dilating it
    # never reaches the far side of the banana. Its convex hull spans the whole
    # fruit including those gaps. The hull of a curved banana also covers some
    # background in the concave side, but that region is only marked *probable*
    # -- GrabCut's colour model reliably rejects it, which is exactly the job it
    # is here to do.
    seed_contours, _ = cv2.findContours(work_seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not seed_contours:
        return None
    hull = cv2.convexHull(np.vstack(seed_contours))
    pr_fg = np.zeros_like(work_seed)
    cv2.drawContours(pr_fg, [hull], -1, 255, thickness=cv2.FILLED)
    rad_pr = max(3, int(max_d * 0.6)) | 1
    pr_fg = cv2.dilate(pr_fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad_pr, rad_pr)), iterations=1)

    rad_bg = max(15, int(max_d * 2.5)) | 1
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad_bg, rad_bg))
    far_region = cv2.dilate(pr_fg, bg_kernel, iterations=1)

    grabcut_mask = np.full((wh, ww), cv2.GC_PR_BGD, dtype=np.uint8)
    grabcut_mask[sure_fg > 0] = cv2.GC_FGD
    grabcut_mask[(pr_fg > 0) & (sure_fg == 0)] = cv2.GC_PR_FGD
    grabcut_mask[far_region == 0] = cv2.GC_BGD

    # GrabCut needs a meaningful sample of known background to build its
    # background colour model. If the seed (plus its margin) already spans
    # nearly the whole frame there is nothing left to learn "background" from,
    # and the result degenerates to "everything is foreground".
    work_area = float(wh * ww)
    if cv2.countNonZero((grabcut_mask == cv2.GC_BGD).astype(np.uint8)) < work_area * 0.10:
        return None

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        work_bgr = cv2.cvtColor(work_img, cv2.COLOR_RGB2BGR)
        cv2.grabCut(work_bgr, grabcut_mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    refined_small = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # Reject a degenerate refinement rather than offering it as a candidate:
    # covering almost the whole frame means the foreground/background models
    # failed to separate, and near-empty means it collapsed the other way.
    refined_fraction = cv2.countNonZero(refined_small) / work_area
    if refined_fraction > 0.85 or refined_fraction < 0.005:
        return None

    if scale < 1.0:
        return cv2.resize(refined_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return refined_small


def _top_k_contours(mask: np.ndarray, k: int) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    return sorted(contours, key=cv2.contourArea, reverse=True)[:k]


def segment_banana_scored(
    image_rgb: np.ndarray,
    method: str = "auto",
    hsv_lower: Optional[Tuple[int, int, int]] = None,
    hsv_upper: Optional[Tuple[int, int, int]] = None,
    min_area: int = MIN_CONTOUR_AREA,
    morph_kernel_size: Optional[int] = None,
    top_k_per_strategy: int = SEGMENTATION_CANDIDATES_PER_STRATEGY,
    alpha_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[str], float, List[Dict[str, Any]]]:
    """
    Segment the banana using every applicable strategy, score every resulting
    contour on shape plausibility (not just area), and return the best one.

    This replaces "assume the largest contour is the banana": each strategy's
    cleaned mask contributes its top `top_k_per_strategy` contours by area, and
    every one of those candidates is scored with quality.score_contour_plausibility
    (area, solidity, elongation, extent, boundary smoothness). The globally
    highest-scoring candidate wins, even if a different candidate had more
    pixels -- so a large but implausibly-shaped blob (e.g. a background region,
    a second object) no longer automatically beats a smaller, genuinely
    banana-shaped one.

    Returns:
        Tuple:
            - binary_mask: uint8 image with 255 for the winning contour, 0 elsewhere.
            - winning_contour: the selected contour (or None if nothing plausible found).
            - winning_method: name of the strategy that produced the winning contour.
            - winning_score: that contour's plausibility_score (0-1).
            - all_candidates: every scored candidate considered, for diagnostics/benchmarking.
    """
    h, w = image_rgb.shape[:2]
    k_size = morph_kernel_size or max(5, int(min(h, w) / 90) | 1)

    frame_area = float(h * w)
    all_candidates: List[Dict[str, Any]] = []
    for strategy_name, raw_mask in _generate_candidate_masks(image_rgb, method, hsv_lower, hsv_upper, alpha_mask):
        if raw_mask is None or cv2.countNonZero(raw_mask) == 0:
            continue
        cleaned = _clean_binary_mask(raw_mask, k_size)
        for contour in _top_k_contours(cleaned, top_k_per_strategy):
            area = cv2.contourArea(contour)
            if area < min_area or area > frame_area * 0.85:
                continue
            scores = score_contour_plausibility(contour, (h, w))
            all_candidates.append({"method": strategy_name, "contour": contour, **scores})

    if not all_candidates:
        return np.zeros((h, w), dtype=np.uint8), None, None, 0.0, []

    # --- Smart candidate selection ---
    # A single banana in a photograph typically covers 1%-40% of the frame.
    # Candidates covering >50% of the frame are almost certainly the background
    # (wooden table, countertop) misidentified by a broad color strategy.
    # We hard-penalize them so that a correctly-shaped, smaller banana candidate
    # wins even when a background blob has more pixels.
    frame_area = float(h * w)
    FRAME_AREA_HARD_CEILING = 0.50  # max plausible banana coverage

    def _selection_key(c: Dict[str, Any]) -> float:
        method = c["method"]
        plaus = c["plausibility_score"]
        priority = SEGMENTATION_METHOD_PRIORITY.get(method, 1.0)
        # For highly banana-shaped gray_otsu candidates (plausibility > 0.85),
        # boost priority since they often capture the full banana silhouette
        # better than color-based methods that miss dark stem/tip regions.
        if method == "gray_otsu" and plaus >= 0.85:
            priority = max(priority, 0.98)
        base = plaus * priority
        # Penalize candidates that cover >50% of the frame -- they're backgrounds
        coverage = c["area"] / frame_area
        if coverage > FRAME_AREA_HARD_CEILING:
            base *= 0.15  # severe penalty: background, not banana
        return base

    # Tier 1: banana-shaped candidates (plausibility >= 0.70 AND not frame-sized)
    banana_candidates = [
        c for c in all_candidates
        if c["plausibility_score"] >= 0.70 and c["area"] / frame_area <= FRAME_AREA_HARD_CEILING
    ]

    # Tier 2: any candidate not covering the whole frame
    non_background = [
        c for c in all_candidates
        if c["area"] / frame_area <= FRAME_AREA_HARD_CEILING
    ]

    # Use the best available tier
    if banana_candidates:
        candidate_pool = banana_candidates
    elif non_background:
        candidate_pool = non_background
    else:
        candidate_pool = all_candidates  # ultimate fallback

    best = max(candidate_pool, key=_selection_key)

    final_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(final_mask, [best["contour"]], -1, 255, thickness=cv2.FILLED)

    return final_mask, best["contour"], best["method"], best["plausibility_score"], all_candidates


def segment_banana(
    image_rgb: np.ndarray,
    method: str = "auto",
    hsv_lower: Optional[Tuple[int, int, int]] = None,
    hsv_upper: Optional[Tuple[int, int, int]] = None,
    min_area: int = MIN_CONTOUR_AREA,
    morph_kernel_size: Optional[int] = None,
    alpha_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Backward-compatible wrapper around segment_banana_scored: returns just the
    binary mask and winning contour. See segment_banana_scored for the full
    result including the winning strategy name, plausibility score, and every
    candidate considered.
    """
    mask, contour, _method, _score, _candidates = segment_banana_scored(
        image_rgb,
        method=method,
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        min_area=min_area,
        morph_kernel_size=morph_kernel_size,
        alpha_mask=alpha_mask,
    )
    return mask, contour
