"""
3_ValidateDecorations.py — Streamlit app for validating detected decorations.

Layout
------
  Left sidebar  — folder selector, per-category +/- counts, Validate button,
                  navigation mode selector, Prev / Next.
  Centre        — pre-rendered *_decorations.jpg for the current page.
  Right column  — summary table (Image | extracted_<cat> | validated_<cat> …);
                  validated rows highlighted green.

Run
---
    streamlit run 3_ValidateDecorations.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Validate Decorations",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Categories excluded from the validation UI (noise classes)
_SKIP = {"other", "other_interesting"}

CSV_NAME = "validation_decorations.csv"

# ---------------------------------------------------------------------------
# Helpers — data loading / saving
# ---------------------------------------------------------------------------

def _find_json_files(folder: Path) -> List[Path]:
    return sorted(folder.glob("*_decorations.json"))


def _load_extracted(json_files: List[Path]) -> dict[str, dict[str, int]]:
    """Parse all JSON files → {image_key: {category: count}}."""
    extracted = {}
    for jf in json_files:
        with open(jf) as fh:
            detections = json.load(fh)
        counts = Counter(
            d["category"] for d in detections if d["category"] not in _SKIP
        )
        extracted[jf.name] = dict(counts)
    return extracted


def _derive_categories(extracted: dict) -> List[str]:
    """Sorted list of all categories that appear across all pages."""
    cats = set()
    for counts in extracted.values():
        cats.update(counts.keys())
    return sorted(cats - _SKIP)


def _csv_path(folder: Path) -> Path:
    return folder / CSV_NAME


def _load_validated(folder: Path, categories: List[str]) -> tuple[dict, dict]:
    """Load validated counts and flags from CSV.

    Returns ({image_key: {cat: count}}, {image_key: {flag: bool}}).
    """
    csv = _csv_path(folder)
    if not csv.is_file():
        return {}, {}
    df = pd.read_csv(csv, dtype=str)
    validated: dict[str, dict[str, int]] = {}
    flags: dict[str, dict[str, bool]] = {}
    for _, row in df.iterrows():
        key = str(row["image"])
        counts = {}
        for cat in categories:
            col = f"validated_{cat}"
            if col in row and pd.notna(row[col]):
                try:
                    counts[cat] = int(row[col])
                except ValueError:
                    pass
        if counts:
            validated[key] = counts
        f = {}
        for flag in ("interesting", "recheck"):
            if flag in row and pd.notna(row[flag]):
                f[flag] = str(row[flag]).strip().lower() == "true"
        if f:
            flags[key] = f
    return validated, flags


def _save_validated(
    folder: Path,
    json_files: List[Path],
    extracted: dict,
    validated: dict,
    categories: List[str],
    flags: dict,
) -> None:
    rows = []
    for jf in json_files:
        key = jf.name
        row = {"image": key}
        for cat in categories:
            row[f"extracted_{cat}"] = extracted.get(key, {}).get(cat, 0)
        v = validated.get(key)
        for cat in categories:
            row[f"validated_{cat}"] = v.get(cat, 0) if v is not None else ""
        f = flags.get(key, {})
        row["interesting"] = f.get("interesting", False)
        row["recheck"] = f.get("recheck", False)
        rows.append(row)
    _csv_path(folder).write_text(
        pd.DataFrame(rows).to_csv(index=False)
    )


# ---------------------------------------------------------------------------
# Helpers — navigation
# ---------------------------------------------------------------------------

def _nav_target(
    json_files: List[Path],
    current: int,
    direction: int,
    mode: str,
    validated: dict,
    extracted: dict,
    nav_class: str,
) -> Optional[int]:
    candidates = range(current + direction,
                       -1 if direction == -1 else len(json_files),
                       direction)
    for i in candidates:
        key = json_files[i].name
        if mode == "Normal":
            return i
        elif mode == "To validate":
            if key not in validated:
                return i
        elif mode == "By class":
            if extracted.get(key, {}).get(nav_class, 0) > 0:
                return i
    return None


# ---------------------------------------------------------------------------
# Helpers — display
# ---------------------------------------------------------------------------

def _build_summary(
    json_files: List[Path],
    extracted: dict,
    validated: dict,
    categories: List[str],
    flags: dict,
) -> pd.DataFrame:
    rows = []
    for jf in json_files:
        key = jf.name
        display = key.replace("_decorations.json", "")
        row = {"Image": display}
        for cat in categories:
            row[f"extracted_{cat}"] = extracted.get(key, {}).get(cat, 0)
        v = validated.get(key)
        for cat in categories:
            row[f"validated_{cat}"] = v.get(cat, 0) if v is not None else None
        f = flags.get(key, {})
        row["interesting"] = f.get("interesting", False)
        row["recheck"] = f.get("recheck", False)
        rows.append(row)
    df = pd.DataFrame(rows)
    for cat in categories:
        df[f"validated_{cat}"] = pd.array(df[f"validated_{cat}"], dtype=pd.Int64Dtype())
    return df


def _row_style(row: pd.Series, categories: List[str]) -> list:
    is_validated = any(pd.notna(row.get(f"validated_{cat}")) for cat in categories)
    if not is_validated:
        return [""] * len(row)
    # Check if any validated count differs from extracted
    modified = any(
        pd.notna(row.get(f"validated_{cat}"))
        and row.get(f"validated_{cat}") != row.get(f"extracted_{cat}", 0)
        for cat in categories
    )
    color = "#fff3cd; color: #856404" if modified else "#d4edda; color: #155724"
    return [f"background-color: {color}"] * len(row)


# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "folder":     None,
    "json_files": [],
    "extracted":  {},
    "validated":  {},
    "categories": [],
    "flags":      {},
    "img_idx":    0,
    "nav_mode":   "Normal",
    "nav_class":  None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Left sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Validate Decorations")

    folder_input = st.text_input(
        "Results folder",
        value=str(st.session_state.folder) if st.session_state.folder else "",
        placeholder="/path/to/results",
    )

    if st.button("Load folder", width="stretch"):
        folder = Path(folder_input.strip())
        if folder.is_dir():
            jfiles = _find_json_files(folder)
            if jfiles:
                ext = _load_extracted(jfiles)
                cats = _derive_categories(ext)
                val, flg = _load_validated(folder, cats)
                first_unvalidated = next(
                    (i for i, jf in enumerate(jfiles) if jf.name not in val), 0
                )
                st.session_state.folder     = folder
                st.session_state.json_files = jfiles
                st.session_state.extracted  = ext
                st.session_state.validated  = val
                st.session_state.categories = cats
                st.session_state.flags      = flg
                st.session_state.img_idx    = first_unvalidated
                st.session_state.nav_class  = cats[0] if cats else None
                st.success(f"{len(jfiles)} pages found.")
            else:
                st.warning("No *_decorations.json files found.")
        else:
            st.error("Directory does not exist.")

    json_files: List[Path] = st.session_state.json_files
    categories: List[str]  = st.session_state.categories

    if json_files:
        st.divider()

        # ── Direct jump ───────────────────────────────────────────────────────
        display_names = [jf.name.replace("_decorations.json", "") for jf in json_files]
        selected = st.selectbox(
            "Jump to image",
            options=display_names,
            index=st.session_state.img_idx,
        )
        new_idx = display_names.index(selected)
        if new_idx != st.session_state.img_idx:
            st.session_state.img_idx = new_idx

        idx      = st.session_state.img_idx
        jf       = json_files[idx]
        key      = jf.name
        ext_now  = st.session_state.extracted.get(key, {})
        val_now  = st.session_state.validated.get(key)

        st.markdown(f"**{idx + 1} / {len(json_files)}** — `{display_names[idx]}`")

        # ── Per-category +/- inputs ───────────────────────────────────────────
        st.subheader("Counts")
        new_validated: dict[str, int] = {}
        for cat in categories:
            extracted_count = ext_now.get(cat, 0)
            default = val_now.get(cat, extracted_count) if val_now is not None else extracted_count
            col_label, col_input = st.columns([3, 2])
            with col_label:
                st.markdown(f"**{cat}**")
                st.caption(f"Detected: {extracted_count}")
            with col_input:
                new_validated[cat] = st.number_input(
                    cat,
                    min_value=0,
                    value=int(default),
                    step=1,
                    label_visibility="collapsed",
                    key=f"val_{idx}_{cat}",
                )

        # ── Add new category ──────────────────────────────────────────────────
        with st.expander("Add category"):
            new_cat_input = st.text_input("Name", key="new_cat_input", placeholder="e.g. rubric")
            if st.button("Add", key="add_cat_btn"):
                cat_norm = new_cat_input.strip().lower().replace(" ", "_")
                if cat_norm and cat_norm not in st.session_state.categories:
                    st.session_state.categories.append(cat_norm)
                    _save_validated(
                        st.session_state.folder, json_files,
                        st.session_state.extracted, st.session_state.validated,
                        st.session_state.categories, st.session_state.flags,
                    )
                    st.rerun()
                elif cat_norm in st.session_state.categories:
                    st.warning("Category already exists.")

        # ── Page flags ────────────────────────────────────────────────────────
        cur_flags = st.session_state.flags.get(key, {})
        col_i, col_r = st.columns(2)
        with col_i:
            flag_interesting = st.checkbox(
                "Interesting", value=cur_flags.get("interesting", False), key=f"interesting_{idx}"
            )
        with col_r:
            flag_recheck = st.checkbox(
                "Recheck", value=cur_flags.get("recheck", False), key=f"recheck_{idx}"
            )
        if st.button("Save flags", key="save_flags_btn", width="stretch"):
            st.session_state.flags[key] = {"interesting": flag_interesting, "recheck": flag_recheck}
            _save_validated(
                st.session_state.folder, json_files,
                st.session_state.extracted, st.session_state.validated,
                categories, st.session_state.flags,
            )
            st.rerun()

        st.divider()

        # ── Validate / Reset ──────────────────────────────────────────────────
        already_validated = val_now is not None
        if already_validated:
            if st.button("↺ Reset", type="secondary", width="stretch"):
                st.session_state.validated.pop(key, None)
                _save_validated(
                    st.session_state.folder, json_files,
                    st.session_state.extracted, st.session_state.validated,
                    categories, st.session_state.flags,
                )
                st.rerun()
        else:
            if st.button("✓ Validate", type="primary", width="stretch"):
                st.session_state.validated[key] = new_validated
                _save_validated(
                    st.session_state.folder, json_files,
                    st.session_state.extracted, st.session_state.validated,
                    categories, st.session_state.flags,
                )
                next_i = idx + 1
                st.session_state.img_idx = next_i if next_i < len(json_files) else idx
                st.rerun()

        # ── Navigation ────────────────────────────────────────────────────────
        st.session_state.nav_mode = st.radio(
            "Navigation mode",
            options=["Normal", "To validate", "By class"],
            index=["Normal", "To validate", "By class"].index(st.session_state.nav_mode),
        )

        if st.session_state.nav_mode == "By class" and categories:
            current_class = st.session_state.nav_class or categories[0]
            st.session_state.nav_class = st.selectbox(
                "Class",
                options=categories,
                index=categories.index(current_class) if current_class in categories else 0,
            )

        prev_target = _nav_target(json_files, idx, -1,
                                   st.session_state.nav_mode, st.session_state.validated,
                                   st.session_state.extracted, st.session_state.nav_class or "")
        next_target = _nav_target(json_files, idx, +1,
                                   st.session_state.nav_mode, st.session_state.validated,
                                   st.session_state.extracted, st.session_state.nav_class or "")

        col_p, col_n = st.columns(2)
        with col_p:
            if st.button("← Prev", disabled=(prev_target is None), width="stretch"):
                st.session_state.img_idx = prev_target
                st.rerun()
        with col_n:
            if st.button("Next →", disabled=(next_target is None), width="stretch"):
                st.session_state.img_idx = next_target
                st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

json_files: List[Path] = st.session_state.json_files
categories: List[str]  = st.session_state.categories

if not json_files:
    st.info("Use the sidebar to select a results folder containing *_decorations.json files.")
    st.stop()

idx = st.session_state.img_idx
jf  = json_files[idx]
jpg = jf.with_name(jf.name.replace(".json", ".jpg"))

col_center, col_right = st.columns([2, 3], gap="large")

with col_center:
    if jpg.is_file():
        st.image(str(jpg), caption=jf.name.replace("_decorations.json", ""), width="stretch")
    else:
        st.warning(f"Annotated image not found: {jpg.name}")

with col_right:
    st.subheader("All pages")

    df = _build_summary(json_files, st.session_state.extracted,
                        st.session_state.validated, categories, st.session_state.flags)

    def _flag_cell_style(val: object) -> str:
        if val is True or val == "True":
            return "background-color: #ffc107; color: #212529; font-weight: bold"
        return ""

    styled = (
        df.style
        .apply(lambda row: _row_style(row, categories), axis=1)
        .map(_flag_cell_style, subset=["interesting", "recheck"])
    )
    st.dataframe(styled, width="stretch", hide_index=True, height=580)

    n_validated = sum(1 for jf in json_files if jf.name in st.session_state.validated)
    st.metric("Validated pages", f"{n_validated} / {len(json_files)}")
