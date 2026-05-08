"""
Stage 1 — Binarisation (Sauvola adaptive thresholding + deskew)
Stage 2 — Page extraction (crop border, detect binding, mask strips)

Each public function is independently testable; run this file directly
to exercise Stages 1–2 on a single image:

    python -m pipeline.preprocessing path/to/image.jpg
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from skimage.transform import rotate
from scipy.ndimage import label as nd_label


# ---------------------------------------------------------------------------
# Stage 1 helpers
# ---------------------------------------------------------------------------

def binarise(gray: np.ndarray, window_size: int = 51, k: float = 0.2) -> np.ndarray:
    """Return a binary uint8 image (0=background, 255=foreground) using
    Sauvola adaptive thresholding.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel uint8 grayscale image.
    window_size : int
        Local window size for Sauvola (must be odd).
    k : float
        Sauvola sensitivity parameter.

    Returns
    -------
    np.ndarray
        Binary uint8 image with foreground=255.
    """
    if window_size % 2 == 0:
        window_size += 1
    thresh = threshold_sauvola(gray, window_size=window_size, k=k)
    binary = (gray < thresh).astype(np.uint8) * 255
    return binary


def _projection_variance(image: np.ndarray, angle: float) -> float:
    """Compute variance of the horizontal projection profile after rotating
    by *angle* degrees.  Higher variance → lines are more horizontal."""
    rotated = rotate(image, angle, resize=False, mode="constant", cval=0)
    profile = rotated.sum(axis=1)
    return float(np.var(profile))


def deskew(
    binary: np.ndarray,
    angle_range: float = 5.0,
    angle_step: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """Deskew a binary image by maximising the variance of the horizontal
    projection profile.

    Parameters
    ----------
    binary : np.ndarray
        Binary uint8 image (foreground=255).
    angle_range : float
        Search ±angle_range degrees around 0.
    angle_step : float
        Angular resolution of the search.

    Returns
    -------
    deskewed : np.ndarray
        Deskewed binary image.
    best_angle : float
        Rotation angle applied (degrees, counter-clockwise).
    """
    angles = np.arange(-angle_range, angle_range + angle_step, angle_step)
    variances = [_projection_variance(binary, a) for a in angles]
    best_angle = float(angles[np.argmax(variances)])
    if abs(best_angle) < angle_step / 2:
        return binary.copy(), 0.0
    deskewed = rotate(
        binary, best_angle, resize=False, mode="constant", cval=0,
        preserve_range=True,
    ).astype(np.uint8)
    return deskewed, best_angle


# ---------------------------------------------------------------------------
# Stage 2 helpers
# ---------------------------------------------------------------------------

def _crop_to_page(gray: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop away the dark camera/scanner border by finding the largest contour.

    Returns
    -------
    cropped : np.ndarray
        Cropped grayscale image.
    rect : (x, y, w, h)
        Bounding rectangle of the largest contour in the *original* image.
    """
    # Threshold: anything darker than ~50 grey is "border"
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    # Close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = gray.shape
        return gray.copy(), (0, 0, w, h)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    cropped = gray[y : y + h, x : x + w]
    return cropped, (x, y, w, h)


def _detect_binding_side(gray: np.ndarray, probe_frac: float = 0.1) -> str:
    """Detect which side of the page contains the binding strip.

    The binding side has a wide high-density dark vertical band; the
    opposite edge has only a thin sliver of the facing page.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale page image (after border crop).
    probe_frac : float
        Fraction of image width to sample on each side.

    Returns
    -------
    'left' or 'right'
    """
    h, w = gray.shape
    probe_w = max(1, int(w * probe_frac))

    left_strip = gray[:, :probe_w]
    right_strip = gray[:, w - probe_w :]

    # Dark pixels = value < 120
    left_dark = np.sum(left_strip < 120)
    right_dark = np.sum(right_strip < 120)

    return "right" if left_dark >= right_dark else "left"


def remove_black_margin(
    gray: np.ndarray,
    binding_side: str,
    dark_threshold: int = 120,
    max_dark_frac: float = 0.50,
    search_frac: float = 0.40,
    extra_px: int = 10,
) -> int:
    """Return the width of the large black patch on the binding edge.

    The binding side of a scanned manuscript page carries a solid dark
    margin (scanner shadow / physical fold).  This function scans
    column-by-column from that edge inward and locates where the black
    patch ends: the first column in which fewer than *max_dark_frac* of
    rows are darker than *dark_threshold* is considered the start of the
    actual page surface.  Everything from the edge up to that column is
    the margin to remove.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image (cropped + deskewed).
    binding_side : str
        'left' or 'right' — the opposite side on which the black patch appears.
    dark_threshold : int
        A pixel is considered 'black margin' if its gray value is below
        this value.
    max_dark_frac : float
        A column still belongs to the black margin when at least this
        fraction of its rows are dark.  The margin ends at the first
        column below this fraction.
    search_frac : float
        Maximum fraction of image width to scan from the binding edge.
    extra_px : int
        Additional pixels to mask inward beyond the detected edge.  Positive
        values push the mask further into the page (right if the black patch
        is on the left, left if it is on the right).  Default 0.

    Returns
    -------
    int
        Number of pixels to mask from the binding edge.
    """
    h, w = gray.shape
    max_search = int(w * search_frac)
    col_dark = (gray < dark_threshold).mean(axis=0)  # shape (w,)

    if binding_side == "right":
        for i in range(max_search):
            if col_dark[i] < max_dark_frac:
                return min(w, i + extra_px)
        return min(w, max_search + extra_px)
    else:
        for i in range(max_search):
            col = w - 1 - i
            if col_dark[col] < max_dark_frac:
                return min(w, i + extra_px)
        return min(w, max_search + extra_px)


def remove_top_bottom_black(
    gray: np.ndarray,
    dark_threshold: int = 120,
    max_dark_frac: float = 0.50,
    search_frac: float = 0.20,
    ruler_search_frac: float = 0.08,
    ruler_min_std: float = 20.0,
    ruler_max_mean: float = 180.0,
    extra_px: int = 10,
) -> Tuple[int, int, int]:
    """Return the heights of the top black patch, ruler band, and bottom black patch.

    Scans row-by-row from the top (and bottom) edge.  A row belongs to the
    solid black border when more than *max_dark_frac* of its columns are
    darker than *dark_threshold*.  After the top black patch ends, a short
    ruler search window is inspected: rows that are neither solid-dark nor
    typical parchment (they have a mean brightness below *ruler_max_mean*
    AND horizontal standard deviation above *ruler_min_std*, characteristic
    of a scale-bar's alternating black/white marks) are counted as ruler
    rows and also masked.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image (cropped + deskewed).
    dark_threshold : int
        Pixel value below which a pixel is considered black.
    max_dark_frac : float
        A row is solid black when at least this fraction of its columns are
        dark.  Solid-black rows are masked as black border.
    search_frac : float
        Maximum fraction of image height to scan from each edge.
    ruler_search_frac : float
        Fraction of image height to inspect for a ruler just below the top
        black patch.
    ruler_min_std : float
        Minimum horizontal standard deviation for a row to be considered
        part of the ruler (ruler has alternating dark/bright segments).
    ruler_max_mean : float
        Maximum row mean brightness for a ruler row (rules out white rows).
    extra_px : int
        Additional rows to mask inward beyond each detected edge — downward
        for the top margin and upward for the bottom margin.  Default 0.

    Returns
    -------
    (top_margin, ruler_height, bottom_margin) : Tuple[int, int, int]
        top_margin    — rows of solid black at the top (plus extra_px).
        ruler_height  — rows of ruler just below the top black patch (0 if
                        no ruler is detected).
        bottom_margin — rows of solid black at the bottom (plus extra_px).
    """
    h, w = gray.shape
    max_search = int(h * search_frac)
    row_dark = (gray < dark_threshold).mean(axis=1)   # shape (h,)
    row_mean = gray.mean(axis=1)
    row_std  = gray.std(axis=1)

    # ── Top black patch ──────────────────────────────────────────────────────
    top_margin = 0
    for i in range(max_search):
        if row_dark[i] >= max_dark_frac:
            top_margin = i + 1
        else:
            break  # first non-dark row ends the solid patch

    # ── Ruler band (just below top black patch) ───────────────────────────────
    ruler_height = 0
    ruler_search_end = top_margin + int(h * ruler_search_frac)
    for i in range(top_margin, min(ruler_search_end, h)):
        if row_std[i] >= ruler_min_std and row_mean[i] <= ruler_max_mean:
            ruler_height += 1
        else:
            break  # first non-ruler row ends the band

    # ── Bottom black patch ───────────────────────────────────────────────────
    bottom_margin = 0
    for i in range(max_search):
        row = h - 1 - i
        if row_dark[row] >= max_dark_frac:
            bottom_margin = i + 1
        else:
            break

    return min(h, top_margin + extra_px), ruler_height, min(h, bottom_margin + extra_px)


def mask_page_borders(
    gray: np.ndarray,
    binary: np.ndarray,
    dark_threshold: int = 120,
) -> Tuple[np.ndarray, dict]:
    """Detect the binding-side dark margin and remove it from the binary image.

    Step 1 — Detect which side (left/right) carries the dark binding margin
             by comparing dark-pixel density in narrow probes on each edge.
    Step 2 — Scan from that edge inward, column by column, to find where
             the dark margin ends.  Everything from the edge up to that
             column is zeroed in the binary image.
    Step 3 — Scan row-by-row from the top and bottom to remove the solid
             black patches on those edges, plus the ruler band that sits
             immediately below the top black patch.

    Parameters
    ----------
    gray : np.ndarray
        Deskewed grayscale image (used for all detections).
    binary : np.ndarray
        Deskewed binary image (foreground=255).

    Returns
    -------
    masked_binary : np.ndarray
        Binary image with dark border regions zeroed.
    info : dict
        binding_side   : 'left' or 'right'
        margin_width   : columns masked from the binding edge
        top_margin     : rows masked from the top (black patch)
        ruler_height   : rows masked below top_margin (ruler band)
        bottom_margin  : rows masked from the bottom (black patch)
    """
    h, w = binary.shape
    masked = binary.copy()

    # ── Step 1: which side has the dark binding margin? ──────────────────────
    binding_side = _detect_binding_side(gray)

    # ── Step 2: find and remove the lateral dark margin ──────────────────────
    margin_width = remove_black_margin(gray, binding_side,
                                       dark_threshold=dark_threshold)

    if binding_side == "right":
        masked[:, :margin_width] = 0
    else:
        masked[:, w - margin_width :] = 0

    # ── Step 3: find and remove top/bottom black patches + ruler ─────────────
    top_margin, ruler_height, bottom_margin = remove_top_bottom_black(
        gray, dark_threshold=dark_threshold)

    masked[:top_margin, :] = 0
    if bottom_margin:
        masked[h - bottom_margin :, :] = 0

    info = {
        "binding_side": binding_side,
        "margin_width": margin_width,
        "top_margin": top_margin,
        "ruler_height": ruler_height,
        "bottom_margin": bottom_margin,
    }
    return masked, info


def detect_binding(
    gray: np.ndarray,
    masked: np.ndarray,
    border_info: dict,
    strip_frac: float = 0.20,
    smooth_frac: float = 0.10,
    prominence_frac: float = 0.05,
) -> Tuple[np.ndarray, int]:
    """Locate the physical fold and refine the binding-side mask.

    Examines the inner 20 % of columns on the binding side of the gray
    image.  The strip contains two text populations separated by a bright
    (ink-free) fold: facing-page text bleed near the edge and the current
    page's first (or last) column further in.  Column-wise mean gray values
    are smoothed and their peaks are detected — a peak is a *white valley*
    in the ink density, i.e. the physical fold.

    - Left binding: the **leftmost** white peak → fold position from the
      left edge.  Everything left of it is masked.
    - Right binding: the **rightmost** white peak → fold position from the
      right edge.  Everything right of it is masked.

    Falls back to ``border_info['margin_width']`` when no clear peak is
    found (e.g. the facing-page bleed is absent).

    Parameters
    ----------
    gray : np.ndarray
        Deskewed grayscale image.
    masked : np.ndarray
        Binary image already partially masked by ``mask_page_borders``.
    border_info : dict
        Output of ``mask_page_borders``; must contain ``'binding_side'``
        and ``'margin_width'``.
    strip_frac : float
        Fraction of image width to examine on the binding side (default 20 %).
    smooth_frac : float
        Smoothing window as a fraction of the strip width.
    prominence_frac : float
        Minimum peak prominence as a fraction of the strip's gray range.

    Returns
    -------
    updated_masked : np.ndarray
        Binary image with the refined binding strip additionally zeroed.
    binding_width : int
        Total columns masked from the binding edge (dark margin + fold gap).
    """
    from scipy.signal import find_peaks
    from scipy.ndimage import uniform_filter1d

    h, w = gray.shape
    binding_side = border_info["binding_side"]
    fallback = border_info["margin_width"]

    strip_w = max(4, int(w * strip_frac))

    if binding_side == "left":
        strip = gray[:, :strip_w]
    else:
        strip = gray[:, w - strip_w :]

    # Column-wise mean gray: high value = bright/white = low ink
    col_mean = strip.mean(axis=0)

    smooth_w = max(3, int(strip_w * smooth_frac))
    if smooth_w % 2 == 0:
        smooth_w += 1
    col_mean_s = uniform_filter1d(col_mean, size=smooth_w)

    peak_range = col_mean_s.max() - col_mean_s.min()
    if peak_range == 0:
        return masked, fallback

    # Peaks = white (low-ink) columns = candidate fold locations
    peaks, _ = find_peaks(
        col_mean_s,
        prominence=peak_range * prominence_frac,
    )

    if len(peaks) == 0:
        return masked, fallback

    result = masked.copy()

    if binding_side == "left":
        # Leftmost white peak = fold; mask everything to its left
        binding_width = int(peaks[0])
        result[:, :binding_width] = 0
    else:
        # Rightmost white peak (strip-local) = fold; mask everything to its right
        # strip-local index 0 = far from binding, strip_w-1 = near binding
        valley_local = int(peaks[-1])
        binding_width = strip_w - valley_local
        result[:, w - binding_width :] = 0

    return result, binding_width


def detect_binding_valley(
    gray: np.ndarray,
    masked: np.ndarray,
    border_info: dict,
    strip_frac: float = 0.20,
    valley_frac: float = 0.05,
    smooth_frac: float = 0.05,
) -> Tuple[np.ndarray, int]:
    """Locate the physical fold using a column-sum valley on the binary masked image.

    Examines the inner *strip_frac* of columns on the binding side of the
    **masked binary** image.  Column sums represent ink density: a bright
    gap between facing-page bleed and page text produces a deep valley
    (low column sum = few foreground pixels).

    Algorithm:
    1. Extract a strip of width ``strip_frac * image_width`` from the
       binding side of *masked*.
    2. Compute the column sum across all rows.
    3. Apply a uniform (box) smoothing of width ``smooth_frac * strip_w``
       columns to merge adjacent low-density columns into a coherent valley.
    4. Find the threshold at the *valley_frac* quantile of the smoothed sums.
    5. Among all columns whose smoothed sum falls at or below that threshold
       (the deepest valleys), choose the column **closest to the binding edge**
       as the fold position.
    6. Mask every pixel between the binding edge and the fold, then return.

    Falls back to ``border_info['margin_width']`` when no columns pass the
    valley threshold (i.e. the strip is uniformly dark or uniformly white).

    Parameters
    ----------
    gray : np.ndarray
        Deskewed grayscale image (not used for detection, kept for API
        consistency with ``detect_binding``).
    masked : np.ndarray
        Binary image already partially masked by ``mask_page_borders``
        (foreground=255).
    border_info : dict
        Output of ``mask_page_borders``; must contain ``'binding_side'``
        and ``'margin_width'``.
    strip_frac : float
        Fraction of image width to examine on the binding side.  Default 0.20.
    valley_frac : float
        Quantile threshold for selecting valley columns.  Columns whose
        smoothed sum is at or below this quantile are valley candidates.
        Default 0.05 (bottom 5 %).
    smooth_frac : float
        Smoothing window width as a fraction of the strip width.  A uniform
        (box) filter of this size is applied to the column sums before valley
        detection, merging nearby low-density columns.  Set to 0.0 to
        disable smoothing.  Default 0.05.

    Returns
    -------
    updated_masked : np.ndarray
        Binary image with the binding strip up to the fold additionally zeroed.
    binding_width : int
        Total columns masked from the binding edge.
    """
    from scipy.ndimage import uniform_filter1d

    h, w = masked.shape
    binding_side = border_info["binding_side"]
    fallback = border_info["margin_width"]

    strip_w = max(4, int(w * strip_frac))

    if binding_side == "left":
        strip = masked[:, :strip_w]
    else:
        strip = masked[:, w - strip_w:]

    col_sums = strip.astype(np.float64).sum(axis=0)  # shape (strip_w,)

    # Optional smoothing: merge adjacent columns before valley detection
    if smooth_frac > 0.0:
        smooth_w = max(3, int(strip_w * smooth_frac))
        col_sums = uniform_filter1d(col_sums, size=smooth_w)

    # Valley threshold: the valley_frac quantile of (smoothed) column sums
    threshold = np.quantile(col_sums, valley_frac)

    valley_cols = np.where(col_sums <= threshold)[0]
    if valley_cols.size == 0:
        return masked, fallback

    result = masked.copy()

    if binding_side == "left":
        # Strip spans columns [0, strip_w).  The column closest to the left
        # (binding) edge is the one with the smallest strip-local index.
        fold_local = int(valley_cols.min())
        binding_width = fold_local
        result[:, :binding_width] = 0
    else:
        # Strip spans columns [w - strip_w, w).  The column closest to the
        # right (binding) edge has the largest strip-local index.
        fold_local = int(valley_cols.max())
        binding_width = strip_w - fold_local
        result[:, w - binding_width:] = 0

    return result, binding_width


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def preprocess(
    image_path: str | Path,
    sauvola_window: int = 51,
    sauvola_k: float = 0.2,
    deskew_range: float = 5.0,
    dark_threshold: int = 120,
    model_path: str | None = None,
    device: str = "cpu",
) -> dict:
    """Full Stages 1–2 preprocessing pipeline.

    Parameters
    ----------
    image_path : str or Path
        Path to the input image (any format readable by OpenCV).
    sauvola_window : int
        Sauvola window size.
    sauvola_k : float
        Sauvola k parameter.
    deskew_range : float
        ±degrees for deskew search.

    Returns
    -------
    dict with keys:
        gray        : grayscale uint8 image, deskewed and cropped to page content
        bgr         : colour uint8 image (BGR), deskewed and cropped to page content
        binary      : binarised uint8 image, deskewed and cropped to page content
        binary_desk : same as ``binary`` (alias kept for API compatibility)
        masked      : binary image with border/binding regions zeroed, cropped
        deskew_angle: float, rotation angle applied
        border_info : dict from mask_page_borders; also contains ``crop_rect``
                      as ``(x, y, w, h)`` in deskewed image coordinates
    """
    path = Path(image_path)
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Stage 1 — binarise
    binary = binarise(gray_full, window_size=sauvola_window, k=sauvola_k)

    # Stage 1b — deskew
    binary_desk, angle = deskew(binary, angle_range=deskew_range)
    # Apply the same rotation to both grayscale and BGR for coherent downstream use
    if abs(angle) > 0.1:
        from skimage.transform import rotate as sk_rotate
        gray = sk_rotate(
            gray_full, angle, resize=False, mode="constant", cval=255,
            preserve_range=True,
        ).astype(np.uint8)
        bgr_desk = sk_rotate(
            bgr, angle, resize=False, mode="constant", cval=255,
            preserve_range=True,
        ).astype(np.uint8)
    else:
        gray = gray_full.copy()
        bgr_desk = bgr.copy()

    # Stage 2 — mask dark borders (lateral + top/bottom)
    masked, border_info = mask_page_borders(gray, binary_desk,
                                            dark_threshold=dark_threshold)

    # Stage 2 — detect fold valley and refine the binding-side mask
    #masked, binding_width = detect_binding(gray, masked, border_info)
    masked, binding_width = detect_binding_valley(gray, masked, border_info)
    border_info["binding_width"] = binding_width

    # Stage 2b — if binding is tiny, re-detect using Kraken text mask only ──
    # A binding_width < 10 px means the valley finder was likely confused by
    # figure ink or facing-page bleed-through.  Running Kraken on the current
    # image gives us a text-only binary that exposes the true fold.
    if binding_width < 10:
        print(
            f"[preprocess] binding_width={binding_width} < 10 px — "
            "re-detecting binding using Kraken text mask."
        )
        from .masking import mask_non_text_kraken as _mntk
        nz = cv2.findNonZero(masked)
        if nz is not None:
            cx, cy, cw, ch = cv2.boundingRect(nz)
        else:
            cy, cx = 0, 0
            ch, cw = masked.shape  
        bgr_desk_pre    = bgr_desk[cy : cy + ch, cx : cx + cw]
        masked_pre      = masked[cy : cy + ch, cx : cx + cw]  
        gray_pre        = gray[cy : cy + ch, cx : cx + cw]            
        _mres = _mntk(bgr_desk_pre, masked_pre, model_path=model_path, device=device, dilation_px = 8)
        _text_only = _mres.text_binary  # uint8 0/255, ink inside text polygons only
        masked_pre, binding_width = detect_binding_valley(gray_pre, _text_only, border_info)
        border_info["binding_width"] = binding_width
        # Re-apply the new binding strip to the full masked image so that figure
        # pixels between the binding edge and the fold are also zeroed out.
        if border_info["binding_side"] == "left":
            masked[:, :binding_width] = 0
        else:
            h_tmp, w_tmp = masked.shape
            masked[:, w_tmp - binding_width:] = 0

    # ── Crop all images to the tight bounding box of the masked content ──────
    # Using the non-zero region of the masked binary guarantees every returned
    # image is aligned and contains only the usable page area.
    nz = cv2.findNonZero(masked)
    if nz is not None:
        cx, cy, cw, ch = cv2.boundingRect(nz)
    else:
        cy, cx = 0, 0
        ch, cw = masked.shape
    border_info["crop_rect"] = (cx, cy, cw, ch)

    gray        = gray[cy : cy + ch, cx : cx + cw]
    bgr_desk    = bgr_desk[cy : cy + ch, cx : cx + cw]
    binary_desk = binary_desk[cy : cy + ch, cx : cx + cw]
    masked      = masked[cy : cy + ch, cx : cx + cw]
    # binary (pre-deskew) shares the same spatial layout as the deskewed images
    # when rotate(resize=False) is used; the coordinates are directly applicable.
    binary      = binary[cy : cy + ch, cx : cx + cw]

    return {
        "gray": gray,
        "bgr": bgr_desk,
        "binary": binary,
        "binary_desk": binary_desk,
        "masked": masked,
        "deskew_angle": angle,
        "border_info": border_info,
    }


# ---------------------------------------------------------------------------
# __main__ test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if img_path is None:
        # Default: first exemplar found
        exemplars = sorted(
            Path(__file__).parent.parent / "data" / "exemplars"
        )
        img_path = next(
            (
                p
                for p in (Path(__file__).parent.parent / "data" / "exemplars").iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif"}
            ),
            None,
        )
        if img_path is None:
            print("No exemplar images found.")
            sys.exit(1)

    print(f"Processing: {img_path}")
    result = preprocess(img_path)
    print(f"  Deskew angle : {result['deskew_angle']:.2f}°")
    print(f"  Crop rect    : {result['border_info']['crop_rect']}")
    print(f"  Border info  : {result['border_info']}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(result["gray"], cmap="gray")
    axes[0].set_title("Grayscale (cropped)")
    axes[1].imshow(result["binary_desk"], cmap="gray")
    axes[1].set_title(f"Binary deskewed ({result['deskew_angle']:.1f}°)")
    axes[2].imshow(result["masked"], cmap="gray")
    axes[2].set_title(f"Masked — binding {result['border_info']['binding_side']} (fold at {result['border_info']['binding_width']}px)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("/tmp/preprocess_test.png", dpi=100)
    print("Saved preview to /tmp/preprocess_test.png")
    plt.show()
