"""
Interactive tuner for mask_non_text_fullrgb parameters.

Run:
    streamlit run tune_masking.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.preprocessing import preprocess
from pipeline.masking import mask_non_text_fullrgb

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Colour Masking Tuner",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Colour Masking Parameter Tuner")

# ---------------------------------------------------------------------------
# Image list
# ---------------------------------------------------------------------------

EXEMPLARS_DIR = PROJECT_ROOT / "data" / "exemplars"
image_paths = sorted(EXEMPLARS_DIR.glob("*.jpg"))
n_images = len(image_paths)

if n_images == 0:
    st.error(f"No .jpg images found in {EXEMPLARS_DIR}")
    st.stop()

image_names = [p.stem for p in image_paths]

# ---------------------------------------------------------------------------
# Session state — image index
# ---------------------------------------------------------------------------

if "img_idx" not in st.session_state:
    st.session_state.img_idx = 0

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Image navigation")

    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("◀", help="Previous image", use_container_width=True):
            st.session_state.img_idx = (st.session_state.img_idx - 1) % n_images
    with col_info:
        st.markdown(
            f"<div style='text-align:center; padding-top:6px'>"
            f"{st.session_state.img_idx + 1} / {n_images}</div>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button("▶", help="Next image", use_container_width=True):
            st.session_state.img_idx = (st.session_state.img_idx + 1) % n_images

    selected_name = st.selectbox(
        "Jump to image",
        image_names,
        index=st.session_state.img_idx,
    )
    # Selectbox can also drive the index
    st.session_state.img_idx = image_names.index(selected_name)

    st.divider()

    # ── Parameter sliders ──────────────────────────────────────────────────
    st.header("mask_non_text_fullrgb parameters")

    st.markdown("**Pixel classification**")
    min_saturation = st.slider(
        "min_saturation",
        min_value=0, max_value=255, value=40,
        help="Pixels with S < this are grey/white (not coloured).",
    )
    max_dark_value = st.slider(
        "max_dark_value",
        min_value=0, max_value=255, value=60,
        help="Pixels with V < this are too dark — treated as black ink.",
    )
    red_hue_margin = st.slider(
        "red_hue_margin",
        min_value=0, max_value=30, value=15,
        help=(
            "Hue distance from 0° / 180° that counts as red (rubric) ink. "
            "Increase to exclude more red from the coloured mask."
        ),
    )

    st.markdown("**Region detection**")
    morph_close_px = st.slider(
        "morph_close_px",
        min_value=1, max_value=120, value=25,
        help="Morphological closing kernel size — joins nearby coloured spots.",
    )
    min_colored_area_frac = st.slider(
        "min_colored_area_frac (×10⁻⁴)",
        min_value=1, max_value=100, value=10,
        help="Minimum blob area as fraction of image (×10⁻⁴). Raise to ignore small specks.",
    ) * 1e-4

    st.markdown("**Diversity filter**")
    min_hue_std = st.slider(
        "min_hue_std",
        min_value=0.0, max_value=60.0, value=15.0, step=0.5,
        help=(
            "Minimum hue std-dev inside a blob. "
            "Low values = uniform tint (parchment discolouration, foxing) — rejected."
        ),
    )

    st.divider()
    st.markdown("**Display options**")
    show_table = st.checkbox("Show blob details table", value=False)

# ---------------------------------------------------------------------------
# Load and preprocess (cached per image)
# ---------------------------------------------------------------------------

img_path = image_paths[st.session_state.img_idx]


@st.cache_data(show_spinner="Preprocessing…")
def load_prep(path_str: str) -> dict:
    return preprocess(path_str)


prep = load_prep(str(img_path))
bgr          = prep["bgr"]
binary_masked = prep["masked"]

# ---------------------------------------------------------------------------
# Run masking (fast — no caching, runs live on every slider change)
# ---------------------------------------------------------------------------

result = mask_non_text_fullrgb(
    bgr,
    binary_masked,
    min_saturation=min_saturation,
    max_dark_value=max_dark_value,
    red_hue_margin=red_hue_margin,
    min_colored_area_frac=min_colored_area_frac,
    morph_close_px=morph_close_px,
    min_hue_std=min_hue_std,
)

# ---------------------------------------------------------------------------
# Recompute intermediate per-pixel mask for visualisation
# ---------------------------------------------------------------------------

hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
H_ch = hsv[:, :, 0].astype(np.int16)
S_ch = hsv[:, :, 1]
V_ch = hsv[:, :, 2]

per_pixel_colored = (
    (S_ch >= min_saturation)
    & (V_ch >= max_dark_value)
    & ~((H_ch <= red_hue_margin) | (H_ch >= 180 - red_hue_margin))
)

# ---------------------------------------------------------------------------
# Build visualisation images
# ---------------------------------------------------------------------------

rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
h_img, w_img = rgb.shape[:2]

# Panel 2: per-pixel coloured mask overlaid as orange tint
colored_vis = rgb.copy()
colored_vis[per_pixel_colored] = (
    (rgb[per_pixel_colored].astype(np.int16) // 2 + np.array([255 // 2, 100 // 2, 0], dtype=np.int16))
    .clip(0, 255)
    .astype(np.uint8)
)

# Panel 3: illustration blobs as semi-transparent red fill + contours
blob_vis = rgb.copy()
red_layer = np.zeros_like(rgb)
red_layer[result.illustration_mask] = (220, 30, 30)
blob_vis = cv2.addWeighted(rgb, 0.55, red_layer, 0.45, 0)

# Draw contours on top for sharp boundaries
ill_uint8 = result.illustration_mask.astype(np.uint8) * 255
contours, _ = cv2.findContours(ill_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(blob_vis, contours, -1, (255, 80, 0), 2)

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

n_blobs = len(result.removed_components)
total_pixels = h_img * w_img
blob_area_frac = float(result.illustration_mask.sum()) / total_pixels

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.subheader(img_path.stem)

metric_cols = st.columns(3)
metric_cols[0].metric("Blobs detected", n_blobs)
metric_cols[1].metric("Area removed", f"{blob_area_frac:.2%}")
metric_cols[2].metric("Text coverage", f"{result.text_coverage:.2%}")

st.divider()

# Row 1 — colour views
r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    st.caption("Original colour image")
    st.image(rgb, use_container_width=True)
with r1c2:
    st.caption("Per-pixel coloured mask (orange = coloured pixel)")
    st.image(colored_vis, use_container_width=True)
with r1c3:
    st.caption("Illustration blobs (red = removed region)")
    st.image(blob_vis, use_container_width=True)

st.divider()

# Row 2 — binary views
r2c1, r2c2 = st.columns(2)
with r2c1:
    st.caption("Binary — before colour masking")
    st.image(binary_masked, use_container_width=True)
with r2c2:
    st.caption("Binary — after colour masking")
    st.image(result.text_binary, use_container_width=True)

# Optional blob table
if show_table and n_blobs > 0:
    st.divider()
    st.subheader("Detected blobs")
    df = pd.DataFrame(result.removed_components)
    df = df.drop(columns=["type", "label"], errors="ignore")
    df["area_frac"] = (df["area"] / total_pixels).map("{:.4f}".format)
    st.dataframe(df, use_container_width=True)
elif show_table and n_blobs == 0:
    st.info("No blobs detected with current parameters.")
