#!/usr/bin/env python3
"""Streamlit app — tune postprocessing parameters interactively.

Run with:
    streamlit run 0_TuningPostprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.stages import (
    PreprocessResult,
    SegmentKrakenResult,
    PostprocessResult,
    preprocess_page,
    segment_kraken,
    postprocess,
)
from run_pipeline import process_image, _DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Postprocessing Parameter Tuning",
    layout="wide",
    page_icon="📜",
)
st.title("📜 Postprocessing Parameter Tuning")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "images" not in st.session_state:
    st.session_state.images: list[Path] = []

# Cache: img_path_str → {"pre": PreprocessResult, "seg": SegmentKrakenResult}
if "seg_cache" not in st.session_state:
    st.session_state.seg_cache: dict = {}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📁 Image folder")
    folder_str = st.text_input("Folder path", value="data/exemplars")
    if st.button("Load images"):
        folder = Path(folder_str)
        if not folder.exists():
            st.error(f"Folder not found: {folder}")
        else:
            found = sorted(
                p for p in folder.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
            )
            st.session_state.images = found
            st.success(f"Found {len(found)} image(s)")

    st.divider()

    st.header("⚙️ Segmentation settings")
    model_path = st.text_input("Kraken model path", value=_DEFAULT_MODEL)
    device = st.selectbox("Device", ["cpu", "mps", "cuda"], index=1)

    st.divider()

    st.header("🎛️ Postprocessing parameters")
    corner_fraction = st.slider(
        "Corner fraction",
        min_value=0.00, max_value=0.30, value=0.10, step=0.01,
        help=(
            "Fraction of image width × height that defines the corner zone "
            "adjacent to the binding spine. Detections whose centroid falls "
            "inside these corners are removed."
        ),
    )
    min_dimension_px = st.slider(
        "Min dimension (px)",
        min_value=0, max_value=100, value=20, step=1,
        help=(
            "Minimum allowed size for the shorter side of any bounding box. "
            "Boxes narrower than this threshold are removed."
        ),
    )
    min_gutter_fraction = st.slider(
        "Min gutter fraction",
        min_value=0.30, max_value=0.95, value=0.50, step=0.05,
        help=(
            "Minimum fraction of lines that must be non-spanning "
            "(entirely left or right of the candidate gutter) "
            "for a double-column gutter to be declared."
        ),
    )
    single_col_threshold = st.slider(
        "Single-column threshold",
        min_value=0.30, max_value=0.95, value=0.70, step=0.05,
        help=(
            "Fraction of lines needed to trigger the single-column guard. "
            "If at least this fraction of lines are full-width (extending "
            "200 px past the page centre on both sides), OR if this fraction "
            "of line centroids fall on the same side of the page centre, "
            "the page is treated as single-column and no gutter is searched."
        ),
    )

# ---------------------------------------------------------------------------
# Guard: no images loaded yet
# ---------------------------------------------------------------------------

if not st.session_state.images:
    st.info("Enter a folder path in the sidebar and click **Load images** to begin.")
    st.stop()

# ---------------------------------------------------------------------------
# Image selector
# ---------------------------------------------------------------------------

image_names = [p.name for p in st.session_state.images]
selected_name = st.selectbox("Select image to inspect", image_names)
selected_path = next(p for p in st.session_state.images if p.name == selected_name)
cache_key = str(selected_path)

# ---------------------------------------------------------------------------
# Segmentation (run once per image, cached in session_state)
# ---------------------------------------------------------------------------

seg_col, _ = st.columns([1, 4])
with seg_col:
    run_seg = st.button("▶ Segment this image", type="primary")

if run_seg:
    with st.spinner(f"Running Kraken on {selected_name} — this may take a moment …"):
        try:
            pre = preprocess_page(selected_path)
            seg = segment_kraken(pre, model_path=model_path, device=device)
            st.session_state.seg_cache[cache_key] = {"pre": pre, "seg": seg}
            st.success(
                f"Segmentation complete — "
                f"{seg.n_lines} lines, {seg.n_figures} figures detected"
            )
        except Exception as exc:
            st.error(f"Segmentation failed: {exc}")

if cache_key not in st.session_state.seg_cache:
    st.info("Click **▶ Segment this image** to run Kraken on this page.")
    st.stop()

pre: PreprocessResult   = st.session_state.seg_cache[cache_key]["pre"]
seg: SegmentKrakenResult = st.session_state.seg_cache[cache_key]["seg"]

# ---------------------------------------------------------------------------
# Apply postprocessing with current slider values (fast)
# ---------------------------------------------------------------------------

post: PostprocessResult = postprocess(
    pre, seg,
    corner_fraction=corner_fraction,
    min_dimension_px=min_dimension_px,
    min_gutter_fraction=min_gutter_fraction,
    single_col_threshold=single_col_threshold,
)

# ---------------------------------------------------------------------------
# Filter statistics
# ---------------------------------------------------------------------------

st.markdown("### Filter statistics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Lines — raw",  seg.n_lines)
m2.metric(
    "Lines — kept", post.n_lines,
    delta=f"−{post.n_lines_removed_corner + post.n_lines_removed_narrow} removed",
    delta_color="inverse",
)
m3.metric("Figures — raw",  seg.n_figures)
m4.metric(
    "Figures — kept", post.n_figures,
    delta=f"−{post.n_figures_removed_corner + post.n_figures_removed_narrow} removed",
    delta_color="inverse",
)

with st.expander("Removal breakdown"):
    st.markdown(
        f"**Lines** — corner filter: {post.n_lines_removed_corner}, "
        f"narrow filter: {post.n_lines_removed_narrow}  \n"
        f"**Figures** — corner filter: {post.n_figures_removed_corner}, "
        f"narrow filter: {post.n_figures_removed_narrow}"
    )

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _bgr2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _plot_original(pre: PreprocessResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(_bgr2rgb(pre.bgr))
    ax.set_title(
        f"Original (deskewed {pre.deskew_angle:+.1f}°, "
        f"binding: {pre.binding_side})",
        fontsize=10,
    )
    ax.axis("off")
    fig.tight_layout()
    return fig


def _plot_lines(pre: PreprocessResult, post: PostprocessResult) -> plt.Figure:
    """Filtered Kraken line polygons overlaid on the BGR image."""
    canvas = _bgr2rgb(pre.bgr).copy()
    for boundary in post.line_boundaries:
        pts = np.array(boundary, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 200, 255), thickness=3)
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(canvas)
    ax.set_title(f"Line boundaries (kept: {post.n_lines})", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


def _plot_overlay(
    pre: PreprocessResult,
    seg: SegmentKrakenResult,
    post: PostprocessResult,
) -> plt.Figure:
    """Text (blue tint) + figure ink (red tint) + kept figure bbox outlines."""
    overlay = np.stack([pre.gray, pre.gray, pre.gray], axis=-1).astype(np.float32)

    # Blue tint on text-mask region
    text_bool = seg.text_mask > 0
    overlay[text_bool, 2] = np.clip(overlay[text_bool, 2] + 80, 0, 255)
    overlay[text_bool, 0] = overlay[text_bool, 0] * 0.5

    # Red tint on figure ink
    fig_bool = seg.figure_binary > 0
    overlay[fig_bool, 0] = np.clip(overlay[fig_bool, 0] + 80, 0, 255)
    overlay[fig_bool, 2] = overlay[fig_bool, 2] * 0.5

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(overlay.astype(np.uint8))

    # Kept figure bounding boxes as green outlines
    for x, y, w, h in post.figure_bboxes:
        ax.add_patch(
            mpatches.Rectangle(
                (x, y), w, h,
                linewidth=1.5, edgecolor="lime", facecolor="none",
            )
        )

    ax.legend(
        handles=[
            mpatches.Patch(facecolor=(0.1, 0.1, 0.9), label="text region"),
            mpatches.Patch(facecolor=(0.9, 0.1, 0.1), label="figure ink"),
            mpatches.Patch(fill=False, edgecolor="lime", label="kept figure bbox"),
        ],
        loc="lower right",
        fontsize=8,
    )
    ax.set_title(
        f"Text + figures overlay (figures kept: {post.n_figures})", fontsize=10
    )
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Three-column visualisation
# ---------------------------------------------------------------------------

st.markdown("### Visualisation")
pc1, pc2, pc3 = st.columns(3)

with pc1:
    fig = _plot_original(pre)
    st.pyplot(fig)
    plt.close(fig)

with pc2:
    fig = _plot_lines(pre, post)
    st.pyplot(fig)
    plt.close(fig)

with pc3:
    fig = _plot_overlay(pre, seg, post)
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Process all images with current parameters
# ---------------------------------------------------------------------------

st.divider()
st.markdown("### Process all images with current parameters")
st.caption(
    f"`corner_fraction = {corner_fraction}` · "
    f"`min_dimension_px = {min_dimension_px}` · "
    f"`min_gutter_fraction = {min_gutter_fraction}` · "
    f"`single_col_threshold = {single_col_threshold}`"
)
st.info(
    "This runs the full pipeline (preprocess → segment → postprocess) on every "
    "image in the folder and saves a `.pkl` file alongside each image. "
    "The embedding stage is skipped for speed."
)

if st.button("🚀 Process all images", type="primary"):
    images = st.session_state.images
    progress = st.progress(0, text="Starting …")
    errors: list[str] = []

    for i, img_path in enumerate(images):
        progress.progress(
            (i + 1) / len(images),
            text=f"[{i + 1}/{len(images)}] {img_path.name}",
        )
        try:
            process_image(
                img_path,
                model_path=model_path,
                device=device,
                corner_fraction=corner_fraction,
                min_dimension_px=min_dimension_px,
                min_gutter_fraction=min_gutter_fraction,
                single_col_threshold=single_col_threshold,
                compute_embed=False,
            )
        except Exception as exc:
            errors.append(f"{img_path.name}: {exc}")

    progress.empty()

    if errors:
        st.warning(f"Completed with {len(errors)} error(s):")
        for err in errors:
            st.text(err)
    else:
        st.success(
            f"✅ All {len(images)} images processed. "
            f"Results saved as `.pkl` files alongside each image."
        )
