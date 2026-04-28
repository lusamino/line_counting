#!/usr/bin/env python3
"""Streamlit app — calibrate preprocessing (detect_binding_valley) parameters.

Run with:
    streamlit run 0_ParametersPreprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.preprocessing import (
    binarise,
    deskew,
    remove_black_margin,
    remove_top_bottom_black,
    detect_binding_valley,
    _detect_binding_side,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Preprocessing Parameter Calibration",
    layout="wide",
    page_icon="🔧",
)
st.title("🔧 Preprocessing Parameter Calibration — detect_binding_valley")

# ---------------------------------------------------------------------------
# Cached slow steps (binarise + deskew) — keyed by file path
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Binarising & deskewing…")
def load_and_prepare(img_path: str):
    """Load image, binarise and deskew. Returns (gray, bgr, binary_desk, angle)."""
    bgr = cv2.imread(img_path)
    if bgr is None:
        return None
    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary = binarise(gray_full)
    binary_desk, angle = deskew(binary)
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
    return gray, bgr_desk, binary_desk, angle


def run_border_detection(
    gray: np.ndarray,
    binary_desk: np.ndarray,
    # remove_black_margin params
    rbm_dark_threshold: int,
    rbm_max_dark_frac: float,
    rbm_search_frac: float,
    rbm_extra_px: int,
    # remove_top_bottom_black params
    rtb_dark_threshold: int,
    rtb_max_dark_frac: float,
    rtb_search_frac: float,
    rtb_ruler_search_frac: float,
    rtb_ruler_min_std: float,
    rtb_ruler_max_mean: float,
    rtb_extra_px: int,
    # detect_binding_valley params
    db_strip_frac: float,
    db_valley_frac: float,
    db_smooth_frac: float,
):
    """Run mask_page_borders + detect_binding_valley with custom parameters."""
    h, w = binary_desk.shape
    masked = binary_desk.copy()

    binding_side = _detect_binding_side(gray)

    margin_width = remove_black_margin(
        gray, binding_side,
        dark_threshold=rbm_dark_threshold,
        max_dark_frac=rbm_max_dark_frac,
        search_frac=rbm_search_frac,
        extra_px=rbm_extra_px,
    )
    if binding_side == "right":
        masked[:, :margin_width] = 0
    else:
        masked[:, w - margin_width:] = 0

    top_margin, ruler_height, bottom_margin = remove_top_bottom_black(
        gray,
        dark_threshold=rtb_dark_threshold,
        max_dark_frac=rtb_max_dark_frac,
        search_frac=rtb_search_frac,
        ruler_search_frac=rtb_ruler_search_frac,
        ruler_min_std=rtb_ruler_min_std,
        ruler_max_mean=rtb_ruler_max_mean,
        extra_px=rtb_extra_px,
    )
    masked[:top_margin, :] = 0
    if bottom_margin:
        masked[h - bottom_margin:, :] = 0

    border_info = {
        "binding_side": binding_side,
        "margin_width": margin_width,
        "top_margin": top_margin,
        "ruler_height": ruler_height,
        "bottom_margin": bottom_margin,
    }

    masked, binding_width = detect_binding_valley(
        gray, masked, border_info,
        strip_frac=db_strip_frac,
        valley_frac=db_valley_frac,
        smooth_frac=db_smooth_frac,
    )
    border_info["binding_width"] = binding_width

    # Crop to non-zero region
    nz = cv2.findNonZero(masked)
    if nz is not None:
        cx, cy, cw, ch = cv2.boundingRect(nz)
    else:
        cy, cx = 0, 0
        ch, cw = masked.shape

    return border_info, (cx, cy, cw, ch)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "images" not in st.session_state:
    st.session_state.images: list[Path] = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📁 Image folder")
    folder_str = st.text_input("Folder path", value="data/exemplars")
    if st.button("Load images"):
        folder = Path(folder_str)
        if not folder.is_absolute():
            folder = PROJECT_ROOT / folder
        if not folder.exists():
            st.error(f"Folder not found: {folder}")
        else:
            found = sorted(
                p for p in folder.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
            )
            st.session_state.images = found
            st.success(f"Found {len(found)} image(s)")

    if st.session_state.images:
        n_images = len(st.session_state.images)
        img_names = [p.name for p in st.session_state.images]

        # Apply any pending navigation request BEFORE the widget is instantiated
        # (Streamlit forbids writing to a widget key after it has been rendered)
        if "_pending_img" in st.session_state:
            st.session_state["img_selectbox"] = st.session_state.pop("_pending_img")

        # Initialise selectbox key if missing or stale
        if st.session_state.get("img_selectbox") not in img_names:
            st.session_state["img_selectbox"] = img_names[0]

        selected_name = st.selectbox("Image", img_names, key="img_selectbox")
        current_idx = img_names.index(selected_name)

        # Navigation arrows
        nav_left, nav_mid, nav_right = st.columns([1, 2, 1])
        with nav_left:
            if st.button("◀", use_container_width=True, disabled=current_idx == 0):
                st.session_state["_pending_img"] = img_names[current_idx - 1]
                st.rerun()
        with nav_mid:
            st.caption(f"{current_idx + 1} / {n_images}")
        with nav_right:
            if st.button("▶", use_container_width=True, disabled=current_idx == n_images - 1):
                st.session_state["_pending_img"] = img_names[current_idx + 1]
                st.rerun()

        import random as _random
        if st.button("🎲 Random image", use_container_width=True):
            st.session_state["_pending_img"] = img_names[_random.randrange(n_images)]
            st.rerun()

        selected_path = st.session_state.images[current_idx]
    else:
        selected_path = None
        st.info("Load a folder to begin.")

    st.divider()

    # ── remove_black_margin ───────────────────────────────────────────────
    st.header("🖤 remove_black_margin")
    rbm_dark_threshold = st.slider(
        "dark_threshold", 20, 200, 80, 5,
        help="Pixel value below which a pixel is 'black'.",
    )
    rbm_max_dark_frac = st.slider(
        "max_dark_frac", 0.05, 1.0, 0.50, 0.05,
        help="A column still belongs to the margin when ≥ this fraction of rows are dark.",
    )
    rbm_search_frac = st.slider(
        "search_frac", 0.05, 0.80, 0.40, 0.05,
        help="Maximum fraction of image width to scan from the binding edge.",
    )
    rbm_extra_px = st.slider(
        "extra_px", 0, 200, 10, 5,
        help="Extra pixels to mask inward beyond the detected black margin edge.",
    )

    st.divider()

    # ── remove_top_bottom_black ───────────────────────────────────────────
    st.header("⬛ remove_top_bottom_black")
    rtb_dark_threshold = st.slider(
        "dark_threshold (top/bot)", 20, 200, 80, 5,
        key="rtb_dark",
        help="Pixel value below which a pixel is 'black'.",
    )
    rtb_max_dark_frac = st.slider(
        "max_dark_frac (top/bot)", 0.05, 1.0, 0.50, 0.05,
        key="rtb_mdf",
        help="A row is solid black when ≥ this fraction of columns are dark.",
    )
    rtb_search_frac = st.slider(
        "search_frac (top/bot)", 0.05, 0.50, 0.20, 0.05,
        key="rtb_sf",
        help="Maximum fraction of image height to scan from each edge.",
    )
    rtb_ruler_search_frac = st.slider(
        "ruler_search_frac", 0.01, 0.20, 0.08, 0.01,
        help="Fraction of height to look for ruler just below the top black patch.",
    )
    rtb_ruler_min_std = st.slider(
        "ruler_min_std", 0.0, 80.0, 20.0, 1.0,
        help="Minimum row std for a row to count as ruler (alternating marks).",
    )
    rtb_ruler_max_mean = st.slider(
        "ruler_max_mean", 50.0, 255.0, 180.0, 5.0,
        help="Maximum row mean brightness for a ruler row.",
    )
    rtb_extra_px = st.slider(
        "extra_px (top/bot)", 0, 200, 10, 5,
        key="rtb_ep",
        help="Extra rows masked inward: downward for top margin, upward for bottom.",
    )

    st.divider()

    # ── detect_binding_valley ─────────────────────────────────────────────
    st.header("📎 detect_binding_valley")
    db_strip_frac = st.slider(
        "strip_frac", 0.05, 0.50, 0.20, 0.01,
        help="Fraction of image width to examine on the binding side.",
    )
    db_valley_frac = st.slider(
        "valley_frac", 0.01, 0.30, 0.05, 0.01,
        help="Quantile threshold: columns at or below this ink-density quantile are valley candidates.",
    )
    db_smooth_frac = st.slider(
        "smooth_frac", 0.0, 0.30, 0.05, 0.01,
        help="Smoothing window as a fraction of the strip width (0 = no smoothing).",
    )

    st.divider()
    st.caption(
        f"`rbm_dark_threshold={rbm_dark_threshold}` · "
        f"`rbm_max_dark_frac={rbm_max_dark_frac}` · "
        f"`rbm_search_frac={rbm_search_frac}` · "
        f"`rbm_extra_px={rbm_extra_px}` · "
        f"`rtb_dark_threshold={rtb_dark_threshold}` · "
        f"`rtb_max_dark_frac={rtb_max_dark_frac}` · "
        f"`rtb_search_frac={rtb_search_frac}` · "
        f"`rtb_ruler_search_frac={rtb_ruler_search_frac}` · "
        f"`rtb_ruler_min_std={rtb_ruler_min_std}` · "
        f"`rtb_ruler_max_mean={rtb_ruler_max_mean}` · "
        f"`rtb_extra_px={rtb_extra_px}` · "
        f"`db_strip_frac={db_strip_frac}` · "
        f"`db_valley_frac={db_valley_frac}` · "
        f"`db_smooth_frac={db_smooth_frac}`"
    )

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

if selected_path is None:
    st.info("👈 Load a folder and select an image to begin.")
    st.stop()

prepared = load_and_prepare(str(selected_path))
if prepared is None:
    st.error(f"Cannot read image: {selected_path}")
    st.stop()

gray, bgr_desk, binary_desk, deskew_angle = prepared

# Run border detection with current params
border_info, (cx, cy, cw, ch) = run_border_detection(
    gray, binary_desk,
    rbm_dark_threshold, rbm_max_dark_frac, rbm_search_frac, rbm_extra_px,
    rtb_dark_threshold, rtb_max_dark_frac, rtb_search_frac,
    rtb_ruler_search_frac, rtb_ruler_min_std, rtb_ruler_max_mean, rtb_extra_px,
    db_strip_frac, db_valley_frac, db_smooth_frac,
)

# Cropped BGR result
bgr_cropped = bgr_desk[cy: cy + ch, cx: cx + cw]

# Build annotated version of the full deskewed image showing detected margins
annotated = bgr_desk.copy()
h_full, w_full = annotated.shape[:2]
binding_side = border_info["binding_side"]
binding_width = border_info["binding_width"]
top_margin = border_info["top_margin"]
ruler_height = border_info["ruler_height"]
bottom_margin = border_info["bottom_margin"]
margin_width = border_info["margin_width"]

ALPHA = 0.35
overlay = annotated.copy()

# Binding-side black margin (red tint)
if binding_side == "right":
    cv2.rectangle(overlay, (0, 0), (margin_width, h_full), (0, 0, 180), -1)
    cv2.rectangle(overlay, (0, 0), (binding_width, h_full), (0, 0, 255), -1)
else:
    cv2.rectangle(overlay, (w_full - margin_width, 0), (w_full, h_full), (0, 0, 180), -1)
    cv2.rectangle(overlay, (w_full - binding_width, 0), (w_full, h_full), (0, 0, 255), -1)

# Top margin (blue tint)
cv2.rectangle(overlay, (0, 0), (w_full, top_margin), (200, 50, 0), -1)
# Ruler band (yellow tint)
if ruler_height:
    cv2.rectangle(overlay, (0, top_margin), (w_full, top_margin + ruler_height), (0, 200, 200), -1)
# Bottom margin (blue tint)
if bottom_margin:
    cv2.rectangle(overlay, (0, h_full - bottom_margin), (w_full, h_full), (200, 50, 0), -1)

cv2.addWeighted(overlay, ALPHA, annotated, 1 - ALPHA, 0, annotated)

# Draw crop-box outline in green
cv2.rectangle(annotated, (cx, cy), (cx + cw, cy + ch), (0, 220, 0), 3)

# ── Layout ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Original (with detected margins)")
    st.image(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        use_container_width=True,
        caption=(
            f"Deskew: {deskew_angle:.1f}°  |  "
            f"Binding: {binding_side}  |  "
            f"margin={margin_width}px  →  fold={binding_width}px  |  "
            f"top={top_margin}px  ruler={ruler_height}px  bottom={bottom_margin}px"
        ),
    )

with col_right:
    st.subheader("Cropped result")
    st.image(
        cv2.cvtColor(bgr_cropped, cv2.COLOR_BGR2RGB),
        use_container_width=True,
        caption=f"Crop: x={cx} y={cy} w={cw} h={ch}",
    )
