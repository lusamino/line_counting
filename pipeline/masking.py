"""
Stage 4 — Non-text masking

Connected component analysis on the binarised image to remove:
  (a) Large compact components → illustrations / figures
  (b) Uniformly very-dark components with near-zero variance → holes /
      physical damage
Small filigranes and decorated initials are intentionally preserved.

Run standalone:

    python -m pipeline.masking path/to/image.jpg
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MaskingResult:
    """Result of non-text masking."""
    text_binary: np.ndarray          # binary with non-text removed
    illustration_mask: np.ndarray    # boolean mask of removed illustration regions
    damage_mask: np.ndarray          # boolean mask of removed damage regions
    removed_components: List[dict]   # metadata for each removed component
    text_coverage: float             # fraction of text pixels remaining


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connected_components(binary: np.ndarray):
    """Return (n_labels, labels, stats, centroids) via OpenCV."""
    return cv2.connectedComponentsWithStats(binary, connectivity=8)


def _component_fill_ratio(binary: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Fraction of pixels inside the bounding box that are foreground."""
    roi = binary[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0
    return float(roi.mean()) / 255.0


def _component_gray_variance(gray: np.ndarray, mask_roi: np.ndarray) -> float:
    """Variance of grayscale values of pixels belonging to the component."""
    pixels = gray[mask_roi > 0]
    if len(pixels) == 0:
        return 0.0
    return float(np.var(pixels.astype(float)))


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def mask_non_text(
    binary: np.ndarray,
    gray: np.ndarray,
    *,
    large_component_area_frac: float = 0.003,
    fill_ratio_threshold: float = 0.45,
    damage_mean_threshold: float = 30.0,
    damage_variance_threshold: float = 50.0,
    min_initial_area: int = 200,
    max_initial_area_frac: float = 0.0005,
) -> MaskingResult:
    """Remove large illustrations and physical damage from the binary image.

    Preserves small decorative initials and filigranes.

    Parameters
    ----------
    binary : np.ndarray
        Masked binary image (foreground=255) from Stage 2.
    gray : np.ndarray
        Corresponding grayscale image (for damage detection).
    large_component_area_frac : float
        A component is "large" if its area > this fraction of total image pixels.
    fill_ratio_threshold : float
        A large component is "compact" (= illustration) if its fill ratio
        exceeds this value.
    damage_mean_threshold : float
        Grayscale mean below this → potentially dark damage region.
    damage_variance_threshold : float
        Variance below this (combined with dark mean) → damage/hole.
    min_initial_area : int
        Components smaller than this are never removed (preserve small details).
    max_initial_area_frac : float
        Small decorated initials are at most this fraction of image area; they
        are preserved even if compact.

    Returns
    -------
    MaskingResult
    """
    h, w = binary.shape
    total_pixels = h * w
    large_area_thresh = large_component_area_frac * total_pixels
    initial_area_thresh = max_initial_area_frac * total_pixels

    n, labels, stats, _ = _connected_components(binary)

    illustration_mask = np.zeros((h, w), dtype=bool)
    damage_mask = np.zeros((h, w), dtype=bool)
    removed_components: List[dict] = []

    for label_id in range(1, n):  # 0 = background
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        # Never remove tiny components (filigranes, pen strokes, etc.)
        if area < min_initial_area:
            continue

        component_mask = (labels == label_id)

        # --- Illustration / figure detection ---
        if area >= large_area_thresh:
            fill = _component_fill_ratio(binary, x, y, bw, bh)
            # Small decorated initials: compact but small — preserve them
            if area <= initial_area_thresh:
                continue
            if fill >= fill_ratio_threshold:
                illustration_mask |= component_mask
                removed_components.append({
                    "type": "illustration",
                    "label": label_id,
                    "x": x, "y": y, "w": bw, "h": bh,
                    "area": area,
                    "fill_ratio": fill,
                })
                continue

        # --- Physical damage / hole detection ---
        # Only apply to components that are notably dark in the grayscale
        comp_pixels_gray = gray[component_mask]
        if len(comp_pixels_gray) == 0:
            continue
        mean_val = float(np.mean(comp_pixels_gray))
        var_val = float(np.var(comp_pixels_gray.astype(float)))

        if mean_val < damage_mean_threshold and var_val < damage_variance_threshold:
            damage_mask |= component_mask
            removed_components.append({
                "type": "damage",
                "label": label_id,
                "x": x, "y": y, "w": bw, "h": bh,
                "area": area,
                "gray_mean": mean_val,
                "gray_var": var_val,
            })

    # Apply masks
    text_binary = binary.copy()
    text_binary[illustration_mask] = 0
    text_binary[damage_mask] = 0

    text_coverage = float(text_binary.sum()) / (255.0 * total_pixels + 1e-9)

    return MaskingResult(
        text_binary=text_binary,
        illustration_mask=illustration_mask,
        damage_mask=damage_mask,
        removed_components=removed_components,
        text_coverage=text_coverage,
    )


# ---------------------------------------------------------------------------
# RGB-aware non-text masking
# ---------------------------------------------------------------------------

def mask_non_text_fullrgb(
    bgr: np.ndarray,
    binary: np.ndarray,
    *,
    min_saturation: int = 40,
    max_dark_value: int = 60,
    red_hue_margin: int = 15,
    min_colored_area_frac: float = 0.001,
    morph_close_px: int = 25,
    min_hue_std: float = 15.0,
) -> MaskingResult:
    """Remove illustrations and figures using full RGB colour information.

    Medieval manuscript text is either black ink or red ink (rubrics).
    Illustrations, decorated borders, and figures contain many other
    colours.  This function identifies regions of colour diversity and
    removes them from the binary image.

    Algorithm
    ---------
    1. Convert BGR → HSV.
    2. Label each pixel as *coloured* when it is:
       - Sufficiently saturated  (S >= min_saturation  → not grey / white)
       - Not too dark            (V >= max_dark_value  → not black ink)
       - Outside the red hue range (not red text ink)
    3. Morphological closing joins nearby coloured pixels into blobs.
    4. Connected components are found; small blobs are discarded.
    5. Each remaining blob is accepted as an illustration only when its
       internal hue standard deviation exceeds *min_hue_std* — this
       rejects uniform off-white patches (parchment discolouration, etc.)
       that may accidentally pass the saturation filter.
    6. The accepted blobs are zeroed in the binary image.

    Parameters
    ----------
    bgr : np.ndarray
        Full-colour BGR image, aligned (same size) as *binary*.
        Typically ``result["bgr"]`` from ``preprocess()``.
    binary : np.ndarray
        Masked binary image (foreground=255) from Stage 2.
    min_saturation : int
        HSV saturation threshold; pixels below this are grey/white and
        are NOT classified as coloured.
    max_dark_value : int
        HSV value threshold; pixels below this are too dark to be
        illustration colour (they are black ink or shadow).
    red_hue_margin : int
        Hue distance from 0 (and 180) that is considered red.
        Pixels within this margin are red text ink, not illustrations.
    min_colored_area_frac : float
        Minimum blob area as a fraction of total image pixels.
    morph_close_px : int
        Closing kernel size in pixels; joins nearby coloured spots into
        a single blob.
    min_hue_std : float
        Minimum standard deviation of hue values inside a blob.  Low
        values indicate a uniformly-tinted patch (likely parchment
        discolouration) rather than a true illustration.

    Returns
    -------
    MaskingResult
        ``illustration_mask`` contains the coloured blobs removed.
        ``damage_mask`` is empty (use ``mask_non_text`` for damage).
    """
    h, w = binary.shape
    total_pixels = h * w
    min_area = max(1, int(min_colored_area_frac * total_pixels))

    # ── Build per-pixel coloured mask ────────────────────────────────────────
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int16)   # 0–179 in OpenCV
    S = hsv[:, :, 1]                     # 0–255
    V = hsv[:, :, 2]                     # 0–255

    is_saturated = S >= min_saturation
    is_bright    = V >= max_dark_value
    is_red       = (H <= red_hue_margin) | (H >= 180 - red_hue_margin)

    colored_mask = (is_saturated & is_bright & ~is_red).astype(np.uint8) * 255

    # ── Morphological closing: merge nearby coloured patches ─────────────────
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_close_px, morph_close_px)
    )
    colored_closed = cv2.morphologyEx(colored_mask, cv2.MORPH_CLOSE, kernel)

    # ── Connected-component analysis ─────────────────────────────────────────
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        colored_closed, connectivity=8
    )

    illustration_mask  = np.zeros((h, w), dtype=bool)
    removed_components: List[dict] = []

    for label_id in range(1, n):   # 0 = background
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x  = int(stats[label_id, cv2.CC_STAT_LEFT])
        y  = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        comp_mask  = labels == label_id
        hue_pixels = H[comp_mask].astype(float)

        if len(hue_pixels) == 0:
            continue

        # Reject uniform-tint patches (parchment discolouration, foxing, etc.)
        hue_std = float(np.std(hue_pixels))
        if hue_std < min_hue_std:
            continue

        illustration_mask |= comp_mask
        removed_components.append({
            "type": "illustration_color",
            "label": label_id,
            "x": x, "y": y, "w": bw, "h": bh,
            "area": area,
            "hue_std": round(hue_std, 2),
        })

    # ── Apply to binary ───────────────────────────────────────────────────────
    text_binary  = binary.copy()
    text_binary[illustration_mask] = 0
    damage_mask  = np.zeros((h, w), dtype=bool)
    text_coverage = float(text_binary.sum()) / (255.0 * total_pixels + 1e-9)

    return MaskingResult(
        text_binary=text_binary,
        illustration_mask=illustration_mask,
        damage_mask=damage_mask,
        removed_components=removed_components,
        text_coverage=text_coverage,
    )


# ---------------------------------------------------------------------------
# Kraken baseline-segmentation masking
# ---------------------------------------------------------------------------

def mask_non_text_kraken(
    bgr: np.ndarray,
    binary: np.ndarray,
    *,
    model_path: Optional[str] = None,
    device: str = "cpu",
    dilation_px: int = 8,
) -> MaskingResult:
    """Remove non-text regions using Kraken's neural baseline segmenter.

    Kraken detects text-line boundaries as polygons.  Every pixel that lies
    *outside* all detected line boundaries is treated as non-text (illustration,
    decoration, margin ornament, etc.) and zeroed in the output binary.

    Parameters
    ----------
    bgr : np.ndarray
        Full-colour BGR image aligned with *binary*.
        Typically ``result["bgr"]`` from ``preprocess()``.
    binary : np.ndarray
        Masked binary image (foreground=255) from Stage 2.
    model_path : str, optional
        Path to a Kraken ``.mlmodel`` segmentation file.
        If *None* the function looks in ``~/.kraken/`` for any ``.mlmodel``
        file.  Download a suitable model first, e.g.::

            kraken get 10.5281/zenodo.10592716   # printed Latin (default)

        For medieval manuscripts consider models from
        https://zenodo.org/communities/ocr_models
    device : str
        PyTorch device string passed to Kraken, e.g. ``"cpu"``, ``"mps"``.
    dilation_px : int
        After rasterising line polygons, dilate the text mask by this many
        pixels to give a small margin around each line boundary.

    Returns
    -------
    MaskingResult
        ``illustration_mask`` is True for every pixel outside a text line.
        ``damage_mask`` is empty; ``removed_components`` is empty (region-level
        aggregation is not performed here).

    Raises
    ------
    ImportError
        If the ``kraken`` package is not installed.
    FileNotFoundError
        If *model_path* is None and no ``.mlmodel`` can be found in
        ``~/.kraken/``.
    """
    try:
        from PIL import Image as PILImage
        from kraken import blla
        from kraken.lib import vgsl
    except ImportError as exc:
        raise ImportError(
            "kraken is required for this function.  Install it with:\n"
            "    pip install kraken"
        ) from exc

    # ── Resolve and load model ────────────────────────────────────────────────
    # When model_path is None we let Kraken use its built-in default
    # segmentation model (no separate download required).
    if model_path is not None:
        loaded_model = vgsl.TorchVGSLModel.load_model(model_path)
        if loaded_model.model_type != "segmentation":
            raise ValueError(
                f"The model at '{model_path}' is a '{loaded_model.model_type}' "
                f"model, not a segmentation model.\n"
                f"Kraken segmentation models can be found at:\n"
                f"    https://zenodo.org/communities/ocr_models\n"
                f"Leave model_path=None to use Kraken's built-in default."
            )
    else:
        loaded_model = None   # blla.segment will use its built-in default

    # ── Segment ──────────────────────────────────────────────────────────────
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_im = PILImage.fromarray(rgb)

    seg_kwargs: dict = {"device": device}
    if loaded_model is not None:
        seg_kwargs["model"] = loaded_model

    seg = blla.segment(pil_im, **seg_kwargs)

    # ── Rasterise line boundary polygons ─────────────────────────────────────
    h, w = binary.shape
    text_mask = np.zeros((h, w), dtype=np.uint8)

    # Kraken ≤4.x returns a plain dict; ≥5.x returns a Segmentation object
    if isinstance(seg, dict):
        lines = seg.get("lines", [])
        def _boundary(line):
            return line.get("boundary", [])
    else:
        lines = getattr(seg, "lines", [])
        def _boundary(line):
            b = getattr(line, "boundary", None)
            if b is None:
                return []
            # Kraken ≥5 may return list of BaselineLine objects with .boundary
            if hasattr(b, "tolist"):
                return b.tolist()
            return list(b)

    removed_components: List[dict] = []
    for line in lines:
        boundary = _boundary(line)
        if not boundary:
            continue
        pts = np.array(boundary, dtype=np.int32).reshape((-1, 1, 2))
        # Clip to image dimensions to be safe
        pts[:, :, 0] = pts[:, :, 0].clip(0, w - 1)
        pts[:, :, 1] = pts[:, :, 1].clip(0, h - 1)
        cv2.fillPoly(text_mask, [pts], 255)
        # Store boundary for downstream feature extraction
        removed_components.append({
            "type": "text_line",
            "boundary": pts[:, 0, :].tolist(),  # [[x, y], ...]
            "n_points": len(pts),
        })

    # ── Dilate text mask for a small margin around each line ─────────────────
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (dilation_px, dilation_px)
        )
        text_mask = cv2.dilate(text_mask, kernel)

    # ── Build output ──────────────────────────────────────────────────────────
    non_text_mask = text_mask == 0   # True outside all text lines

    text_binary = binary.copy()
    text_binary[non_text_mask] = 0

    illustration_mask = non_text_mask
    damage_mask = np.zeros((h, w), dtype=bool)
    total_pixels = h * w
    text_coverage = float(text_binary.sum()) / (255.0 * total_pixels + 1e-9)

    return MaskingResult(
        text_binary=text_binary,
        illustration_mask=illustration_mask,
        damage_mask=damage_mask,
        removed_components=removed_components,
        text_coverage=text_coverage,
    )


# ---------------------------------------------------------------------------
# __main__ test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pipeline.preprocessing import preprocess
    from pipeline.layout import detect_layout

    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if img_path is None:
        exemplars_dir = Path(__file__).parent.parent / "data" / "exemplars"
        candidates = sorted(exemplars_dir.glob("*.jpg"))
        img_path = next(
            (p for p in candidates if "figure" in p.name or "stain" in p.name),
            candidates[0] if candidates else None,
        )
    if img_path is None:
        print("No exemplar images found.")
        sys.exit(1)

    print(f"Processing: {img_path}")
    prep = preprocess(img_path)
    result = mask_non_text(prep["masked"], prep["gray"])

    n_ill = sum(1 for c in result.removed_components if c["type"] == "illustration")
    n_dmg = sum(1 for c in result.removed_components if c["type"] == "damage")
    print(f"  Removed illustrations: {n_ill}")
    print(f"  Removed damage regions: {n_dmg}")
    print(f"  Text coverage: {result.text_coverage:.3f}")

    overlay = cv2.cvtColor(result.text_binary, cv2.COLOR_GRAY2BGR)
    overlay[result.illustration_mask] = (0, 0, 200)    # red for illustrations
    overlay[result.damage_mask] = (0, 200, 200)        # yellow for damage

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(prep["masked"], cmap="gray")
    axes[0].set_title("Input (masked binary)")
    axes[1].imshow(result.text_binary, cmap="gray")
    axes[1].set_title("After non-text masking")
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Removed: red=illustration, yellow=damage")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("/tmp/masking_test.png", dpi=100)
    print("Saved preview to /tmp/masking_test.png")
    plt.show()
