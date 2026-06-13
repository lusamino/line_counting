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

# Boolean flag columns stored in the flags dict (order fixes sidebar layout: 2 per row)
_FLAG_COLS = ("interesting", "recheck", "hole_text", "quire_mark", "mix_layout", "tail_rhyme")
_FLAG_LABELS = {
    "interesting": "Interesting", "recheck": "Recheck",
    "hole_text": "Hole text",     "quire_mark": "Quire mark",
    "mix_layout": "Mix layout",   "tail_rhyme": "Tail rhyme",
}
_NAV_FLAG_OPTIONS = list(_FLAG_COLS) + ["two_columns"]

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


def _load_csv(folder: Path) -> tuple[dict, dict, dict, dict]:
    """Return (validated, line_counts, flags, two_columns) from CSV, or ({}, {}, {}, {}) if absent.

    flags: {name: {"interesting": bool, "recheck": bool}}
    two_columns: {name: bool}
    """
    csv_path = _csv_path(folder)
    if not csv_path.is_file():
        return {}, {}, {}, {}
    df = pd.read_csv(csv_path, dtype={"image": str, "line_counting": "Int64", "line_counting_validated": "Int64"})
    validated: dict[str, int] = {}
    line_counts: dict[str, int] = {}
    flags: dict[str, dict[str, bool]] = {}
    two_columns: dict[str, bool] = {}
    for _, row in df.iterrows():
        name = str(row["image"])
        if pd.notna(row.get("line_counting")):
            line_counts[name] = int(row["line_counting"])
        if pd.notna(row.get("line_counting_validated")):
            validated[name] = int(row["line_counting_validated"])
        f = {}
        for flag in _FLAG_COLS:
            if flag in row and pd.notna(row[flag]):
                f[flag] = str(row[flag]).strip().lower() == "true"
        if f:
            flags[name] = f
        if "two_columns" in row and pd.notna(row["two_columns"]):
            two_columns[name] = str(row["two_columns"]).strip().lower() == "true"
    return validated, line_counts, flags, two_columns


def _save_csv(
    folder: Path,
    images: List[Path],
    line_counts: dict,
    validated: dict,
    flags: dict,
    two_columns: dict,
) -> None:
    """Write (or overwrite) validation_counting.csv from in-memory dicts — no pkl loading."""
    rows = []
    for img in images:
        n_lines = line_counts.get(img.name)
        if n_lines is None:
            continue
        v = validated.get(img.name)
        f = flags.get(img.name, {})
        row = {
            "image": img.name,
            "line_counting": n_lines,
            "line_counting_validated": int(v) if v is not None else "",
            "two_columns": two_columns.get(img.name, False),
        }
        for flag in _FLAG_COLS:
            row[flag] = f.get(flag, False)
        rows.append(row)
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


def _display_names(names: List[str]) -> List[str]:
    """Strip the common prefix and suffix from a list of names for compact display."""
    if len(names) <= 1:
        return list(names)
    pfx = 0
    for chars in zip(*names):
        if len(set(chars)) == 1:
            pfx += 1
        else:
            break
    tails = [n[pfx:] for n in names]
    sfx = 0
    for chars in zip(*[t[::-1] for t in tails]):
        if len(set(chars)) == 1:
            sfx += 1
        else:
            break
    result = []
    for n in names:
        short = n[pfx: len(n) - sfx if sfx else None].strip("_-. ")
        result.append(short or n)
    return result


def _nav_target(
    images: List[Path], current: int, direction: int,
    mode: str, validated: dict, line_counts: dict,
    flags: Optional[dict] = None,
    two_columns: Optional[dict] = None,
    nav_flag: Optional[str] = None,
) -> Optional[int]:
    """Return the target index for Prev (direction=-1) or Next (direction=1), or None if none exists."""
    candidates = range(current + direction, -1 if direction == -1 else len(images), direction)
    for i in candidates:
        name = images[i].name
        if mode == "Normal":
            return i
        elif mode == "To validate":
            if name not in validated:
                return i
        elif mode == "Modified":
            if name in validated and validated[name] != line_counts.get(name):
                return i
        elif mode == "By flag" and nav_flag is not None:
            if nav_flag == "two_columns":
                if (two_columns or {}).get(name, False):
                    return i
            else:
                if (flags or {}).get(name, {}).get(nav_flag, False):
                    return i
    return None


def _build_summary(
    images: List[Path],
    line_counts: dict,
    validated: dict,
    flags: dict,
    two_columns: dict,
) -> pd.DataFrame:
    """Build the summary table entirely from in-memory dicts — no pkl loading."""
    rows = []
    for img in images:
        n_lines = line_counts.get(img.name)
        if n_lines is None:
            continue
        v = validated.get(img.name)
        f = flags.get(img.name, {})
        row = {
            "Image": img.name,
            "Line counting": n_lines,
            "Validated line counting": int(v) if v is not None else None,
            "two_columns": two_columns.get(img.name, False),
        }
        for flag in _FLAG_COLS:
            row[flag] = f.get(flag, False)
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=["Image", "Line counting", "Validated line counting",
                                    "two_columns"] + list(_FLAG_COLS))
    df["Validated line counting"] = pd.array(
        df["Validated line counting"], dtype=pd.Int64Dtype()
    )
    return df


# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------

for _key, _default in {
    "folder": None, "images": [], "validated": {}, "line_counts": {},
    "flags": {}, "two_columns": {}, "img_idx": 0,
    "nav_mode": "Normal", "nav_flag": _NAV_FLAG_OPTIONS[0],
}.items():
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

    if st.button("Load folder", width='stretch'):
        folder = Path(folder_input.strip())
        if folder.is_dir():
            imgs = _find_images_with_pkl(folder)
            if imgs:
                validated_map, line_counts_map, flags_map, two_cols_map = _load_csv(folder)
                # Bootstrap line_counts (and two_columns) from pkl files if missing
                needs_save = False
                if not line_counts_map:
                    for img in imgs:
                        d = _load_pkl(img)
                        if d is not None:
                            line_counts_map[img.name] = d["post"].n_lines
                    needs_save = True
                if not two_cols_map:
                    for img in imgs:
                        d = _load_pkl(img)
                        if d is not None:
                            two_cols_map[img.name] = bool(d["post"].is_double_column)
                    needs_save = True
                if needs_save:
                    _save_csv(folder, imgs, line_counts_map, validated_map,
                              flags_map, two_cols_map)
                first_unvalidated_idx = next(
                    (i for i, img in enumerate(imgs) if img.name not in validated_map),
                    0,
                )
                st.session_state.folder      = folder
                st.session_state.images      = imgs
                st.session_state.validated   = validated_map
                st.session_state.line_counts = line_counts_map
                st.session_state.flags       = flags_map
                st.session_state.two_columns = two_cols_map
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
        display_names = _display_names(img_names)

        # Selectbox for direct navigation — no widget key, driven by img_idx
        selected_display = st.selectbox(
            "Jump to image",
            options=display_names,
            index=st.session_state.img_idx,
        )
        new_idx = display_names.index(selected_display)
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

            # ── Page flags ────────────────────────────────────────────────────
            cur_flags = st.session_state.flags.get(img_path.name, {})
            flag_values: dict[str, bool] = {}
            for _fi in range(0, len(_FLAG_COLS), 2):
                _ca, _cb = st.columns(2)
                with _ca:
                    _f = _FLAG_COLS[_fi]
                    flag_values[_f] = st.checkbox(
                        _FLAG_LABELS[_f], value=cur_flags.get(_f, False),
                        key=f"{_f}_{idx}",
                    )
                if _fi + 1 < len(_FLAG_COLS):
                    with _cb:
                        _f = _FLAG_COLS[_fi + 1]
                        flag_values[_f] = st.checkbox(
                            _FLAG_LABELS[_f], value=cur_flags.get(_f, False),
                            key=f"{_f}_{idx}",
                        )
            flag_two_cols = st.checkbox(
                "Two columns",
                value=st.session_state.two_columns.get(img_path.name, False),
                key=f"two_columns_{idx}",
            )
            if st.button("Save flags", key="save_flags_btn", width='stretch'):
                st.session_state.flags[img_path.name] = flag_values
                st.session_state.two_columns[img_path.name] = flag_two_cols
                _save_csv(
                    st.session_state.folder, images,
                    st.session_state.line_counts, st.session_state.validated,
                    st.session_state.flags, st.session_state.two_columns,
                )
                st.rerun()

            st.divider()

            already_validated = existing is not None
            if already_validated:
                if st.button("↺ Reset", type="secondary", width='stretch'):
                    st.session_state.validated.pop(img_path.name, None)
                    _save_csv(
                        st.session_state.folder, images,
                        st.session_state.line_counts, st.session_state.validated,
                        st.session_state.flags, st.session_state.two_columns,
                    )
                    st.rerun()
            else:
                if st.button("✓ Validate", type="primary", width='stretch'):
                    st.session_state.validated[img_path.name] = int(validated_val)
                    _save_csv(
                        st.session_state.folder, images,
                        st.session_state.line_counts, st.session_state.validated,
                        st.session_state.flags, st.session_state.two_columns,
                    )
                    next_i = idx + 1
                    st.session_state.img_idx = next_i if next_i < len(images) else idx
                    st.rerun()

            # ── Navigation ────────────────────────────────────────────────────
            _NAV_MODES = ["Normal", "To validate", "Modified", "By flag"]
            st.session_state.nav_mode = st.radio(
                "Navigation mode",
                options=_NAV_MODES,
                index=_NAV_MODES.index(st.session_state.nav_mode),
                horizontal=False,
            )

            if st.session_state.nav_mode == "By flag":
                _cur_nf = st.session_state.nav_flag
                if _cur_nf not in _NAV_FLAG_OPTIONS:
                    _cur_nf = _NAV_FLAG_OPTIONS[0]
                st.session_state.nav_flag = st.selectbox(
                    "Navigate by",
                    options=_NAV_FLAG_OPTIONS,
                    index=_NAV_FLAG_OPTIONS.index(_cur_nf),
                )

            _nav_kw = dict(
                flags=st.session_state.flags,
                two_columns=st.session_state.two_columns,
                nav_flag=st.session_state.nav_flag,
            )
            col_p, col_n = st.columns(2)
            prev_target = _nav_target(images, idx, -1, st.session_state.nav_mode,
                                      st.session_state.validated, st.session_state.line_counts,
                                      **_nav_kw)
            next_target = _nav_target(images, idx, +1, st.session_state.nav_mode,
                                      st.session_state.validated, st.session_state.line_counts,
                                      **_nav_kw)
            with col_p:
                if st.button("← Prev", disabled=(prev_target is None), width='stretch'):
                    st.session_state.img_idx = prev_target
                    st.rerun()
            with col_n:
                if st.button("Next →", disabled=(next_target is None), width='stretch'):
                    st.session_state.img_idx = next_target
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
        st.image(_render_annotated(data), caption=img_path.name, width='stretch')
    else:
        st.error(f"Could not load pipeline result for {img_path.name}.")

with col_right:
    st.subheader("All images")

    df = _build_summary(
        images, st.session_state.line_counts, st.session_state.validated,
        st.session_state.flags, st.session_state.two_columns,
    )

    def _row_style(row: pd.Series) -> list:
        if pd.notna(row["Validated line counting"]):
            return ["background-color: #d4edda; color: #155724"] * len(row)
        return [""] * len(row)

    def _flag_cell_style(val: object) -> str:
        if val is True or val == "True":
            return "background-color: #ffc107; color: #212529; font-weight: bold"
        return ""

    def _two_col_cell_style(val: object) -> str:
        if val is True or val == "True":
            return "background-color: #cce5ff; color: #004085; font-weight: bold"
        return ""

    styled = (
        df.style
        .apply(_row_style, axis=1)
        .map(_flag_cell_style, subset=list(_FLAG_COLS))
        .map(_two_col_cell_style, subset=["two_columns"])
    )
    st.dataframe(styled, width='stretch', hide_index=True, height=600)

    n_validated = sum(1 for img in images if img.name in st.session_state.validated)
    st.metric("Validated pages", f"{n_validated} / {len(images)}")
