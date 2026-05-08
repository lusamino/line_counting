"""
1_ValidateCounting.py — Streamlit app for validating processed image line counts.

Layout
------
  Left sidebar  — folder selector, current image info, line count, validated
                  count input, Validate button, and Prev / Next navigation.
  Centre        — annotated result image (red line polygons, semi-transparent
                  yellow figure mask, green gutter line, thickness 4).
  Right column  — summary table (Image | Line counting | Validated line counting);
                  validated rows are highlighted in green.

Run
---
    streamlit run 1_ValidateCounting.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Validate Line Counting",
    layout="wide",
    initial_sidebar_state="expanded",
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_images_with_pkl(folder: Path) -> List[Path]:
    """Return sorted list of images in *folder* that have a matching .pkl in results/."""
    results_dir = folder / "results"
    if not results_dir.is_dir():
        return []
    imgs = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            if (results_dir / p.with_suffix(".pkl").name).is_file():
                imgs.append(p)
    return imgs


def _load_pkl(img_path: Path) -> Optional[dict]:
    pkl_path = img_path.parent / "results" / img_path.with_suffix(".pkl").name
    if not pkl_path.is_file():
        return None
    with open(pkl_path, "rb") as fh:
        return pickle.load(fh)


CSV_NAME = "validation_counting.csv"


def _csv_path(folder: Path) -> Path:
    return folder / "results" / CSV_NAME


def _load_csv(folder: Path) -> tuple[dict[str, int], dict[str, int]]:
    """Return ({name: validated_count}, {name: line_count}) from CSV, or ({}, {}) if absent."""
    csv_path = _csv_path(folder)
    if not csv_path.is_file():
        return {}, {}
    df = pd.read_csv(csv_path, dtype={"image": str, "line_counting": "Int64", "line_counting_validated": "Int64"})
    validated: dict[str, int] = {}
    line_counts: dict[str, int] = {}
    for _, row in df.iterrows():
        name = str(row["image"])
        if pd.notna(row.get("line_counting")):
            line_counts[name] = int(row["line_counting"])
        if pd.notna(row.get("line_counting_validated")):
            validated[name] = int(row["line_counting_validated"])
    return validated, line_counts


def _save_csv(folder: Path, images: List[Path], line_counts: dict, validated: dict) -> None:
    """Write (or overwrite) validation_counting.csv from in-memory dicts — no pkl loading."""
    rows = []
    for img in images:
        n_lines = line_counts.get(img.name)
        if n_lines is None:
            continue
        v = validated.get(img.name)
        rows.append({
            "image": img.name,
            "line_counting": n_lines,
            "line_counting_validated": int(v) if v is not None else "",
        })
    csv_path = _csv_path(folder)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _render_annotated(data: dict) -> np.ndarray:
    """Return an RGB ndarray with line polygons, figure mask, and gutter line."""
    pre  = data["pre"]
    seg  = data["seg"]
    post = data["post"]

    annotated = pre.bgr.copy()

    # Semi-transparent yellow fill for figure ink pixels
    if seg.figure_binary is not None and seg.figure_binary.any():
        fig_mask = seg.figure_binary > 0
        overlay  = annotated.copy()
        overlay[fig_mask] = (0, 220, 220)          # BGR yellow
        cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

    # Red line-boundary polygons
    for boundary in post.line_boundaries:
        pts = np.array(boundary, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 0, 220), thickness=2)

    # Green gutter line (thickness 4) when double-column detected
    if (
        post.is_double_column
        and post.gutter_x is not None
        and post.gutter_y_min is not None
        and post.gutter_y_max is not None
    ):
        cv2.line(
            annotated,
            (post.gutter_x, post.gutter_y_min),
            (post.gutter_x, post.gutter_y_max),
            (0, 220, 0),
            thickness=4,
        )

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def _build_summary(images: List[Path], line_counts: dict, validated: dict) -> pd.DataFrame:
    """Build the summary table entirely from in-memory dicts — no pkl loading."""
    rows = []
    for img in images:
        n_lines = line_counts.get(img.name)
        if n_lines is None:
            continue
        v = validated.get(img.name)
        rows.append({
            "Image": img.name,
            "Line counting": n_lines,
            "Validated line counting": int(v) if v is not None else None,
        })
    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=["Image", "Line counting", "Validated line counting"])
    df["Validated line counting"] = pd.array(
        df["Validated line counting"], dtype=pd.Int64Dtype()
    )
    return df


# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------

for _key, _default in {"folder": None, "images": [], "validated": {}, "line_counts": {}, "img_idx": 0}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ---------------------------------------------------------------------------
# Left sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Validate Line Counting")

    folder_input = st.text_input(
        "Image folder",
        value=str(st.session_state.folder) if st.session_state.folder else "",
        placeholder="/path/to/images",
    )

    if st.button("Load folder", use_container_width=True):
        folder = Path(folder_input.strip())
        if folder.is_dir():
            imgs = _find_images_with_pkl(folder)
            if imgs:
                validated_map, line_counts_map = _load_csv(folder)
                # If no CSV exists yet, bootstrap line_counts from pkl files
                if not line_counts_map:
                    for img in imgs:
                        d = _load_pkl(img)
                        if d is not None:
                            line_counts_map[img.name] = d["post"].n_lines
                    # Create the CSV immediately so future renders are fast
                    _save_csv(folder, imgs, line_counts_map, validated_map)
                first_unvalidated_idx = next(
                    (i for i, img in enumerate(imgs) if img.name not in validated_map),
                    0,
                )
                st.session_state.folder      = folder
                st.session_state.images      = imgs
                st.session_state.validated   = validated_map
                st.session_state.line_counts = line_counts_map
                st.session_state.img_idx     = first_unvalidated_idx
                st.success(f"{len(imgs)} images found.")
            else:
                st.warning("No images with associated .pkl found in results/.")
        else:
            st.error("Directory does not exist.")

    images: List[Path] = st.session_state.images

    if images:
        st.divider()

        img_names = [img.name for img in images]

        # Selectbox for direct navigation — no widget key, driven by img_idx
        selected = st.selectbox(
            "Jump to image",
            options=img_names,
            index=st.session_state.img_idx,
        )
        new_idx = img_names.index(selected)
        if new_idx != st.session_state.img_idx:
            st.session_state.img_idx = new_idx

        idx = st.session_state.img_idx

        img_path = images[idx]
        data     = _load_pkl(img_path)

        st.markdown(f"**{idx + 1} / {len(images)}** — `{img_path.name}`")

        if data is not None:
            post     = data["post"]
            n_lines  = post.n_lines
            existing = st.session_state.validated.get(img_path.name)

            st.metric("Line counting", n_lines)

            validated_val = st.number_input(
                "Validated line counting",
                min_value=0,
                value=int(existing) if existing is not None else n_lines,
                step=1,
                key=f"val_{idx}",
            )

            st.divider()

            already_validated = existing is not None
            if already_validated:
                if st.button("↺ Reset", type="secondary", use_container_width=True):
                    st.session_state.validated.pop(img_path.name, None)
                    _save_csv(st.session_state.folder, images, st.session_state.line_counts, st.session_state.validated)
                    st.rerun()
            else:
                if st.button("✓ Validate", type="primary", use_container_width=True):
                    st.session_state.validated[img_path.name] = int(validated_val)
                    _save_csv(st.session_state.folder, images, st.session_state.line_counts, st.session_state.validated)
                    next_i = idx + 1
                    st.session_state.img_idx = next_i if next_i < len(images) else idx
                    st.rerun()

            col_p, col_n = st.columns(2)
            with col_p:
                if st.button("← Prev", disabled=(idx == 0), use_container_width=True):
                    st.session_state.img_idx = idx - 1
                    st.rerun()
            with col_n:
                if st.button("Next →", disabled=(idx >= len(images) - 1), use_container_width=True):
                    st.session_state.img_idx = idx + 1
                    st.rerun()

# ---------------------------------------------------------------------------
# Main area — centre image + right summary table
# ---------------------------------------------------------------------------

images: List[Path] = st.session_state.images

if not images:
    st.info("Use the sidebar to select a folder with processed images.")
    st.stop()

img_names = [img.name for img in images]
idx       = st.session_state.img_idx
img_path  = images[idx]
data      = _load_pkl(img_path)

col_center, col_right = st.columns([2, 3], gap="large")

with col_center:
    if data is not None:
        st.image(_render_annotated(data), caption=img_path.name, use_container_width=True)
    else:
        st.error(f"Could not load pipeline result for {img_path.name}.")

with col_right:
    st.subheader("All images")

    df = _build_summary(images, st.session_state.line_counts, st.session_state.validated)

    def _row_style(row: pd.Series) -> list:
        if pd.notna(row["Validated line counting"]):
            return ["background-color: #d4edda; color: #155724"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(_row_style, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=600)
