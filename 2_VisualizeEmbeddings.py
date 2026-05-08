"""
2_VisualizeEmbeddings.py — Streamlit app for exploring page embeddings in 3D.

Layout
------
  Left sidebar  — folder selector, embedding-type checkboxes, UMAP parameters,
                  dark/light toggle, "Compute" button.
  Centre        — interactive 3D Plotly scatter; click a point to select it
                  (selected point turns into a large orange diamond).
  Right column  — selectbox (stays in sync with click), annotated result image,
                  anomaly score, structural features table.

Run
---
    streamlit run 2_VisualizeEmbeddings.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.embeddings import anomaly_scores, compute_umap_3d

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

STRUCT_NAMES = [
    "binding_left", "binding_right", "deskew_angle",
    "margin_width/W", "top_margin/H", "ruler_height/H",
    "bottom_margin/H", "binding_width/W", "H/W",
    "n_lines/H", "text_coverage", "ink_retention",
    "fig_norm", "fig_coverage", "fig_mean_area", "fig_std_area",
    "post_n_lines/H", "removed_corner/seg_n", "removed_narrow/seg_n",
    "post_fig_norm", "fig_rm_corner/seg_n", "fig_rm_narrow/seg_n",
    "is_double_col", "gutter_x/W",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scan_folder(folder: Path):
    """Return list of records for every image that has a matching pkl."""
    results_dir = folder / "results"
    records = []
    for img_path in sorted(folder.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        pkl_path = results_dir / img_path.with_suffix(".pkl").name
        if not pkl_path.is_file():
            continue
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)
        emb = data.get("embedding")
        if emb is None:
            continue
        records.append({
            "filename":       img_path.name,
            "img_path":       str(img_path),
            "embedding":      emb,
            "has_structural": emb.structural_vec.size > 0,
            "has_vit_rgb":    emb.vit_rgb_vec.size > 0,
            "has_vit_mask":   emb.vit_mask_vec.size > 0,
        })
    return records


def _compute(records, use_structural, use_vit_rgb, use_vit_mask,
             n_neighbors, min_dist):
    """Filter records, rebuild combined_vec, run anomaly + UMAP."""
    def has_req(r):
        if use_structural and not r["has_structural"]:
            return False
        if use_vit_rgb and not r["has_vit_rgb"]:
            return False
        if use_vit_mask and not r["has_vit_mask"]:
            return False
        return True

    active = [r for r in records if has_req(r)]

    for r in active:
        emb = r["embedding"]
        parts = []
        if use_structural:
            parts.append(emb.structural_vec)
        if use_vit_rgb:
            parts.append(emb.vit_rgb_vec)
        if use_vit_mask:
            parts.append(emb.vit_mask_vec)
        emb.combined_vec = np.concatenate(parts)

    emb_list = [r["embedding"] for r in active]
    anomaly_scores(emb_list)
    compute_umap_3d(emb_list, n_neighbors=n_neighbors, min_dist=min_dist)

    xs           = [r["embedding"].umap_xyz[0] for r in active]
    ys           = [r["embedding"].umap_xyz[1] for r in active]
    zs           = [r["embedding"].umap_xyz[2] for r in active]
    filenames    = [r["filename"]               for r in active]
    anomaly_list = [r["embedding"].anomaly_score for r in active]

    struct_data = []
    for r in active:
        emb = r["embedding"]
        d = {}
        if emb.structural_vec.size > 0:
            for j, name in enumerate(STRUCT_NAMES[: emb.structural_vec.size]):
                d[name] = round(float(emb.structural_vec[j]), 4)
        struct_data.append(d)

    return active, xs, ys, zs, filenames, anomaly_list, struct_data


def _build_figure(xs, ys, zs, filenames, anomaly_list,
                  sel_idx: int, dark: bool) -> go.Figure:
    """Two-trace figure: muted background points + highlighted selected point.

    Each point carries its original index in ``customdata`` so that the
    correct record can be identified regardless of which trace was clicked.
    """
    n        = len(xs)
    template = "plotly_dark" if dark else "plotly_white"
    unsel    = [i for i in range(n) if i != sel_idx]

    fig = go.Figure()

    # ── Trace 0: all non-selected points ─────────────────────────────────
    if unsel:
        fig.add_trace(go.Scatter3d(
            x=[xs[i] for i in unsel],
            y=[ys[i] for i in unsel],
            z=[zs[i] for i in unsel],
            mode="markers",
            marker=dict(
                size=5,
                color=[anomaly_list[i] for i in unsel],
                colorscale="RdBu_r",
                cmin=0.0, cmax=1.0,
                opacity=0.70,
                colorbar=dict(title="Anomaly<br>score", thickness=12,
                              len=0.5, tickformat=".2f"),
            ),
            text=[filenames[i] for i in unsel],
            customdata=[[i] for i in unsel],           # wrapped so Plotly serialises as list
            hovertemplate="<b>%{text}</b><br>anomaly: %{marker.color:.3f}<extra></extra>",
            showlegend=False,
        ))

    # ── Trace 1: selected point (orange diamond, larger) ─────────────────
    border = "white" if dark else "#333"
    fig.add_trace(go.Scatter3d(
        x=[xs[sel_idx]],
        y=[ys[sel_idx]],
        z=[zs[sel_idx]],
        mode="markers",
        marker=dict(
            size=14,
            color="orange",
            symbol="diamond",
            line=dict(color=border, width=2),
            opacity=1.0,
        ),
        text=[filenames[sel_idx]],
        customdata=[[sel_idx]],                        # wrapped consistently
        hovertemplate="<b>%{text}</b> ★<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        template=template,
        scene=dict(
            xaxis=dict(title="UMAP-1", showspikes=False),
            yaxis=dict(title="UMAP-2", showspikes=False),
            zaxis=dict(title="UMAP-3", showspikes=False),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600,
        uirevision="umap",   # keeps camera across reruns
    )
    return fig


def _load_display_image(img_path: str, results_dir: Path,
                        max_height: int = 900):
    """Load the annotated right half of the _result image (original | annotated)."""
    p = Path(img_path)
    result_path = results_dir / (p.stem + "_result" + p.suffix)
    if result_path.is_file():
        img = cv2.imread(str(result_path))
        if img is not None:
            img = img[:, img.shape[1] // 2:]   # crop to annotated half
    else:
        img = cv2.imread(str(p))
    if img is None:
        return None
    h, w = img.shape[:2]
    if h > max_height:
        scale = max_height / h
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "folder":       "",
    "records":      [],
    "load_error":   "",
    "computed":     False,
    "active":       [],
    "xs": [], "ys": [], "zs": [],
    "filenames":    [],
    "anomaly_list": [],
    "struct_data":  [],
    "sel_idx":      0,
    "dark_mode":    False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Apply any pending click (stored by the PREVIOUS rerun, before building fig)
# ---------------------------------------------------------------------------
# When the user clicks a Plotly point, on_select triggers a rerun.  In that
# rerun we detect the click and store "pending_click".  We then call
# st.rerun() so that in the *next* rerun we can update sel_idx HERE, before
# _build_figure() is called, so the highlight is always in sync.
if "pending_click" in st.session_state:
    st.session_state["sel_idx"] = st.session_state.pop("pending_click")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Visualize Embeddings",
    layout="wide",
    page_icon="🗺️",
)
st.markdown("### Page Embeddings — 3D UMAP Explorer")

# ---------------------------------------------------------------------------
# Dark-mode CSS (must be injected before sidebar to avoid flash)
# ---------------------------------------------------------------------------

dark_mode = st.session_state["dark_mode"]   # read before sidebar widget
if dark_mode:
    st.markdown(
        """<style>
        [data-testid="stAppViewContainer"] > .main { background-color: #0e1117; }
        [data-testid="stSidebar"]                  { background-color: #161b22; }
        </style>""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Configuration")

    dark_mode = st.toggle("🌙 Dark mode", value=st.session_state["dark_mode"])
    st.session_state["dark_mode"] = dark_mode

    st.divider()

    folder_input = st.text_input(
        "Images folder",
        value=st.session_state.folder,
        placeholder="/path/to/images",
    )

    load_btn = st.button("Load folder", use_container_width=True)

    if load_btn:
        folder = Path(folder_input)
        _rd = folder / "results"
        if not folder.is_dir():
            st.session_state.load_error = f"Folder not found: {folder}"
            st.session_state.records = []
        elif not _rd.is_dir():
            st.session_state.load_error = f"No results/ subfolder in {folder}"
            st.session_state.records = []
        else:
            with st.spinner("Scanning pkl files…"):
                recs = _scan_folder(folder)
            st.session_state.folder     = folder_input
            st.session_state.records    = recs
            st.session_state.load_error = ""
            st.session_state.computed   = False
            st.session_state.sel_idx    = 0

    if st.session_state.load_error:
        st.error(st.session_state.load_error)

    records = st.session_state.records
    if records:
        n_total = len(records)
        n_s = sum(r["has_structural"] for r in records)
        n_r = sum(r["has_vit_rgb"]    for r in records)
        n_m = sum(r["has_vit_mask"]   for r in records)
        st.success(f"Loaded **{n_total}** images with embeddings")

        st.divider()
        st.subheader("Embedding types")
        use_structural = st.checkbox(
            f"Structural  ({n_s}/{n_total})", value=True, disabled=n_s == 0)
        use_vit_rgb = st.checkbox(
            f"ViT RGB  ({n_r}/{n_total})", value=n_r > 0, disabled=n_r == 0)
        use_vit_mask = st.checkbox(
            f"ViT Mask  ({n_m}/{n_total})", value=False, disabled=n_m == 0)

        st.divider()
        st.subheader("UMAP parameters")
        n_neighbors = st.slider("n_neighbors", 2, 50, 10, key="nb")
        min_dist    = st.slider("min_dist", 0.0, 0.99, 0.1, 0.01, key="md")

        st.divider()
        compute_btn = st.button(
            "Compute UMAP", type="primary", use_container_width=True)

        if compute_btn:
            if not (use_structural or use_vit_rgb or use_vit_mask):
                st.error("Select at least one embedding type.")
            else:
                try:
                    with st.spinner("Running 3D UMAP (may take a moment)…"):
                        active, xs, ys, zs, fnames, alist, sdata = _compute(
                            records,
                            use_structural, use_vit_rgb, use_vit_mask,
                            n_neighbors, min_dist,
                        )
                    st.session_state.active       = active
                    st.session_state.xs           = xs
                    st.session_state.ys           = ys
                    st.session_state.zs           = zs
                    st.session_state.filenames    = fnames
                    st.session_state.anomaly_list = alist
                    st.session_state.struct_data  = sdata
                    st.session_state.computed        = True
                    st.session_state.sel_idx         = 0
                    st.session_state["_prev_click_set"] = []  # reset on recompute
                except Exception as exc:
                    st.error(f"Compute failed: {exc}")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

if not st.session_state.computed:
    st.info(
        "1. Enter the path to your images folder and click **Load folder**.\n"
        "2. Choose embedding types and UMAP parameters.\n"
        "3. Click **Compute UMAP**."
    )
    st.stop()

xs           = st.session_state.xs
ys           = st.session_state.ys
zs           = st.session_state.zs
filenames    = st.session_state.filenames
anomaly_list = st.session_state.anomaly_list
struct_data  = st.session_state.struct_data
active       = st.session_state.active
n            = len(filenames)

# Clamp in case data was recomputed with fewer points
if st.session_state.sel_idx >= n:
    st.session_state.sel_idx = 0
sel_idx = st.session_state.sel_idx

folder_path = Path(st.session_state.folder)
results_dir = folder_path / "results"

# Build figure with current sel_idx highlighted
fig = _build_figure(xs, ys, zs, filenames, anomaly_list, sel_idx, dark_mode)

col_plot, col_img = st.columns([3, 2], gap="medium")

# ── Plot column ───────────────────────────────────────────────────────────────
with col_plot:
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="umap3d",
    )
    # Detect which point was clicked.  Plotly 3D accumulates selections, so
    # we diff against the previous selection set to find the *new* point.
    # customdata is stored as [[i]] so it comes back as a list — unwrap it.
    def _cd(p):
        raw = p.get("customdata")
        if raw is None:
            return None
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return None

    try:
        pts = event.selection.points or []
    except Exception:
        pts = []

    if pts:
        cur_set  = {_cd(p) for p in pts} - {None}
        prev_set = set(st.session_state.get("_prev_click_set", []))
        st.session_state["_prev_click_set"] = list(cur_set)
        new_pts  = cur_set - prev_set
        if new_pts:
            new_idx = next(iter(new_pts))
            if new_idx != sel_idx:
                st.session_state["pending_click"] = new_idx
                st.rerun()

# ── Info / image column ───────────────────────────────────────────────────────
with col_img:
    # key="sel_idx" binds the selectbox directly to session state, so:
    #   • writing st.session_state["sel_idx"] (from a click) updates what it shows
    #   • the user changing the selectbox updates st.session_state["sel_idx"] and
    #     triggers an automatic rerun — no manual st.rerun() needed
    st.selectbox(
        "Selected page",
        options=list(range(n)),
        format_func=lambda i: filenames[i],
        key="sel_idx",
    )
    sel_idx = st.session_state["sel_idx"]   # re-read after widget (may have changed)

    rec = active[sel_idx]
    st.metric("Anomaly score", f"{anomaly_list[sel_idx]:.4f}")

    img_rgb = _load_display_image(rec["img_path"], results_dir)
    if img_rgb is not None:
        st.image(img_rgb, use_container_width=True)
    else:
        st.warning("Could not load result image.")

    if struct_data[sel_idx]:
        with st.expander("Structural features", expanded=False):
            df = pd.DataFrame(
                list(struct_data[sel_idx].items()),
                columns=["Feature", "Value"],
            )
            st.dataframe(df, hide_index=True, use_container_width=True)
