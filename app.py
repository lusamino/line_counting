"""
Stage 8 — Streamlit frontend

Run:
    streamlit run app.py

Features:
  • Upload / select an exemplar image
  • Side-by-side original + annotated image with numbered bounding boxes
  • Layout info, line counts, anomaly flags
  • UMAP scatter plot with current page highlighted; click to navigate
  • Validation panel: accept / reject / edit individual line bounding boxes
  • Export validated results to CSV
  • Anomaly ranking sidebar
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.preprocessing import preprocess
from pipeline.layout import detect_layout
from pipeline.masking import mask_non_text
from pipeline.segmentation import segment_lines, LineResult
from pipeline.embeddings import (
    compute_embedding,
    anomaly_scores,
    compute_umap,
    PageEmbedding,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXEMPLARS_DIR = PROJECT_ROOT / "data" / "exemplars"
BBOX_COLOURS = [
    (220, 53, 69),    # red
    (13, 110, 253),   # blue
    (25, 135, 84),    # green
    (255, 153, 0),    # orange
    (102, 16, 242),   # purple
]
ANOMALY_COLOUR = (255, 0, 255)  # magenta

# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "current_image": None,       # filename string
        "pipeline_cache": {},        # filename → pipeline result dict
        "embeddings_cache": {},      # filename → PageEmbedding
        "validation": {},            # filename → {line_key → status}
        "umap_computed": False,
        "all_embeddings": [],        # List[PageEmbedding] after scoring
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Pipeline runner (cached per image)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_pipeline(img_path_str: str, method: str = "hpp") -> dict:
    """Run all pipeline stages on a single image (cached by path + method)."""
    img_path = Path(img_path_str)
    prep = preprocess(img_path)
    layout = detect_layout(prep["masked"])
    masking = mask_non_text(prep["masked"], prep["gray"])
    seg = segment_lines(masking.text_binary, layout.columns, method=method)
    return {
        "prep": prep,
        "layout": layout,
        "masking": masking,
        "seg": seg,
    }


@st.cache_data(show_spinner=False)
def run_embedding(img_path_str: str, method: str = "hpp") -> Optional[PageEmbedding]:
    """Compute embedding for a single image (cached)."""
    try:
        result = run_pipeline(img_path_str, method)
        prep = result["prep"]
        layout = result["layout"]
        masking = result["masking"]
        seg = result["seg"]
        emb = compute_embedding(
            filename=Path(img_path_str).name,
            layout_type=layout.layout_type,
            per_column_counts=seg.per_column_counts,
            line_heights=[l.height for l in seg.lines],
            removed_components=masking.removed_components,
            text_coverage=masking.text_coverage,
            image_shape=prep["gray"].shape,
            column_separator=layout.column_separator,
            columns=layout.columns,
            text_binary=masking.text_binary,
            binary_for_vit=prep["binary_desk"],
        )
        return emb
    except Exception as exc:
        st.warning(f"Embedding failed for {Path(img_path_str).name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Image annotation helper
# ---------------------------------------------------------------------------

def annotate_image(
    gray: np.ndarray,
    lines: List[LineResult],
    validation: Dict[str, str],  # line_key → 'accepted'|'rejected'|'pending'
    alpha: float = 0.3,
) -> np.ndarray:
    """Return an RGB uint8 image with coloured bounding boxes + line numbers."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    overlay = rgb.copy()

    for line in lines:
        key = f"{line.column_index}_{line.line_index}"
        status = validation.get(key, "pending")

        if line.is_anomalous:
            colour = ANOMALY_COLOUR
        else:
            colour = BBOX_COLOURS[line.column_index % len(BBOX_COLOURS)]

        if status == "rejected":
            colour = (128, 128, 128)  # grey out rejected

        x0, y0, x1, y1 = line.bbox
        cv2.rectangle(overlay, (x0, y0), (x1, y1), colour, 2)
        label = f"{line.column_index+1}.{line.line_index+1}"
        cv2.putText(overlay, label, (x0 + 2, max(y0 + 12, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)

    # Blend overlay for accepted boxes
    result = cv2.addWeighted(overlay, 1 - alpha, rgb, alpha, 0)
    return result


# ---------------------------------------------------------------------------
# UMAP plot via Plotly
# ---------------------------------------------------------------------------

def make_umap_figure(
    all_embeddings: List[PageEmbedding],
    current_filename: Optional[str],
):
    import plotly.graph_objects as go

    fnames = [e.filename for e in all_embeddings]
    scores = [e.anomaly_score for e in all_embeddings]
    xs = [e.umap_xy[0] if e.umap_xy else 0.0 for e in all_embeddings]
    ys = [e.umap_xy[1] if e.umap_xy else 0.0 for e in all_embeddings]

    marker_colours = [
        f"rgb({int(255 * s)}, {int(255 * (1 - s))}, 80)" for s in scores
    ]
    marker_sizes = [10 if fn == current_filename else 7 for fn in fnames]
    marker_symbols = ["star" if fn == current_filename else "circle" for fn in fnames]
    border_widths = [3 if fn == current_filename else 1 for fn in fnames]

    fig = go.Figure(
        go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker=dict(
                color=marker_colours,
                size=marker_sizes,
                symbol=marker_symbols,
                line=dict(color="black", width=border_widths),
            ),
            text=fnames,
            hovertemplate="<b>%{text}</b><br>anomaly=%.3f<extra></extra>",
            customdata=scores,
        )
    )
    fig.update_layout(
        title="UMAP — all processed pages",
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Export to CSV
# ---------------------------------------------------------------------------

def build_export_df(
    result: dict,
    validation: Dict[str, str],
    embedding: Optional[PageEmbedding],
) -> pd.DataFrame:
    rows = []
    for line in result["seg"].lines:
        key = f"{line.column_index}_{line.line_index}"
        status = validation.get(key, "pending")
        is_validated = 1 if status in {"accepted", "rejected"} else 0
        x0, y0, x1, y1 = line.bbox
        rows.append({
            "filename": Path(result.get("path", "")).name if "path" in result else "",
            "layout_type": result["layout"].layout_type,
            "column": line.column_index,
            "line_index": line.line_index,
            "x_min": x0,
            "y_min": y0,
            "x_max": x1,
            "y_max": y1,
            "height": line.height,
            "is_anomalous": int(line.is_anomalous),
            "method": line.method,
            "validation_status": status,
            "is_validated": is_validated,
            "anomaly_score": round(
                embedding.anomaly_score if embedding else 0.0, 4
            ),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Medieval Manuscript Line Counter",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()

    # ---------- Sidebar ----------
    with st.sidebar:
        st.title("⚙ Settings")
        method = st.radio("Segmentation method", ["hpp", "kraken"], index=0)
        compute_embed = st.checkbox("Compute embeddings / anomaly scores", value=True)

        st.divider()
        st.subheader("📂 Image selection")

        exemplar_files = sorted(
            p.name for p in EXEMPLARS_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif"}
        )
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif"])
        selected_name = st.selectbox(
            "Or select an exemplar", ["— select —"] + exemplar_files
        )

        if uploaded is not None:
            # Save to a temp location
            tmp_path = PROJECT_ROOT / "data" / "exemplars" / uploaded.name
            tmp_path.write_bytes(uploaded.read())
            st.session_state["current_image"] = uploaded.name
        elif selected_name != "— select —":
            st.session_state["current_image"] = selected_name

        st.divider()
        # Anomaly ranking
        st.subheader("🚨 Anomaly ranking")
        if st.session_state["all_embeddings"]:
            ranked = sorted(
                st.session_state["all_embeddings"],
                key=lambda e: e.anomaly_score,
                reverse=True,
            )
            for e in ranked:
                score = e.anomaly_score
                colour = "🔴" if score > 0.7 else ("🟡" if score > 0.4 else "🟢")
                label = f"{colour} {e.filename[:40]} ({score:.2f})"
                if st.button(label, key=f"rank_{e.filename}", use_container_width=True):
                    st.session_state["current_image"] = e.filename
        else:
            st.info("Process images to see anomaly ranking.")

    # ---------- Main area ----------
    current = st.session_state.get("current_image")
    if current is None:
        st.info("Select or upload an image to begin.")
        return

    img_path = EXEMPLARS_DIR / current
    if not img_path.exists():
        st.error(f"Image not found: {img_path}")
        return

    # Run pipeline
    with st.spinner(f"Running pipeline on {current} …"):
        result = run_pipeline(str(img_path), method=method)

    layout = result["layout"]
    masking = result["masking"]
    seg = result["seg"]
    prep = result["prep"]

    # Embedding
    embedding: Optional[PageEmbedding] = None
    if compute_embed:
        with st.spinner("Computing embedding …"):
            embedding = run_embedding(str(img_path), method=method)
        if embedding is not None:
            st.session_state["embeddings_cache"][current] = embedding
            # Collect all embeddings and re-score
            all_embs = list(st.session_state["embeddings_cache"].values())
            if len(all_embs) >= 2:
                anomaly_scores(all_embs)
            if len(all_embs) >= 4 and not st.session_state["umap_computed"]:
                compute_umap(all_embs)
                st.session_state["umap_computed"] = True
            st.session_state["all_embeddings"] = all_embs

    # Validation state
    if current not in st.session_state["validation"]:
        st.session_state["validation"][current] = {}
    val = st.session_state["validation"][current]

    # ---------- Header metrics ----------
    st.title(f"📜 {current}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Layout", layout.layout_type)
    c2.metric("Total lines", len(seg.lines))
    for i, cnt in enumerate(seg.per_column_counts):
        (c3 if i == 0 else c4).metric(f"Col {i+1} lines", cnt)
    c5.metric("Anomalous lines", seg.anomalous_count)
    if embedding:
        c5.metric("Anomaly score", f"{embedding.anomaly_score:.3f}")

    if seg.fallback_triggered:
        st.warning("⚠ Kraken fallback was triggered for one or more columns.")

    # ---------- Image panels ----------
    col_orig, col_annot = st.columns(2)
    with col_orig:
        st.subheader("Original")
        orig_bgr = cv2.imread(str(img_path))
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        st.image(orig_rgb, use_container_width=True)

    with col_annot:
        st.subheader("Detected lines")
        annotated = annotate_image(prep["gray"], seg.lines, val)
        st.image(annotated, use_container_width=True)

    # ---------- UMAP plot ----------
    all_embs = st.session_state["all_embeddings"]
    if all_embs and all(e.umap_xy is not None for e in all_embs):
        st.subheader("🗺 UMAP — layout space")
        fig = make_umap_figure(all_embs, current)
        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                                selection_mode="points", key="umap_chart")
        # Handle click-to-navigate
        if event and hasattr(event, "selection") and event.selection.get("points"):
            clicked_idx = event.selection["points"][0].get("point_index")
            if clicked_idx is not None and 0 <= clicked_idx < len(all_embs):
                clicked_name = all_embs[clicked_idx].filename
                if clicked_name != current:
                    st.session_state["current_image"] = clicked_name
                    st.rerun()

    # ---------- Validation panel ----------
    st.subheader("✅ Validation panel")
    st.caption("Accept or reject individual line bounding boxes. "
               "Edit bbox coordinates inline.")

    # Group by column
    col_groups: Dict[int, List[LineResult]] = {}
    for line in seg.lines:
        col_groups.setdefault(line.column_index, []).append(line)

    for col_idx in sorted(col_groups):
        with st.expander(f"Column {col_idx + 1} ({len(col_groups[col_idx])} lines)", expanded=False):
            for line in col_groups[col_idx]:
                key = f"{line.column_index}_{line.line_index}"
                status = val.get(key, "pending")
                lcol1, lcol2, lcol3, lcol4, lcol5 = st.columns([1, 3, 2, 2, 2])
                with lcol1:
                    st.write(f"**L{line.line_index + 1}**")
                    if line.is_anomalous:
                        st.caption("⚠ anom.")
                with lcol2:
                    x0, y0, x1, y1 = line.bbox
                    new_coords = st.text_input(
                        "bbox", value=f"{x0},{y0},{x1},{y1}",
                        key=f"bbox_{key}", label_visibility="collapsed"
                    )
                    # Parse and update bbox if changed
                    try:
                        parts = [int(v.strip()) for v in new_coords.split(",")]
                        if len(parts) == 4 and tuple(parts) != line.bbox:
                            line.bbox = tuple(parts)
                            line.height = parts[3] - parts[1]
                    except ValueError:
                        pass
                with lcol3:
                    st.write(f"h={line.height}px")
                with lcol4:
                    if st.button("✓ Accept", key=f"acc_{key}", type="primary"):
                        val[key] = "accepted"
                        st.rerun()
                with lcol5:
                    if st.button("✗ Reject", key=f"rej_{key}"):
                        val[key] = "rejected"
                        st.rerun()

    # ---------- Export ----------
    st.subheader("💾 Export results")
    if st.button("Export current page to CSV"):
        df = build_export_df({"layout": layout, "seg": seg, "path": img_path}, val, embedding)
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button(
            label=f"Download {current}_results.csv",
            data=csv_bytes,
            file_name=f"{Path(current).stem}_results.csv",
            mime="text/csv",
        )

    if st.button("Export all validated pages to CSV"):
        all_rows_dfs = []
        for fname, fval in st.session_state["validation"].items():
            fpath = EXEMPLARS_DIR / fname
            if not fpath.exists():
                continue
            try:
                res = run_pipeline(str(fpath), method=method)
                emb = st.session_state["embeddings_cache"].get(fname)
                df_part = build_export_df({"layout": res["layout"], "seg": res["seg"], "path": fpath}, fval, emb)
                all_rows_dfs.append(df_part)
            except Exception:
                pass
        if all_rows_dfs:
            combined = pd.concat(all_rows_dfs, ignore_index=True)
            csv_bytes = combined.to_csv(index=False).encode()
            st.download_button(
                label="Download all_results.csv",
                data=csv_bytes,
                file_name="all_results.csv",
                mime="text/csv",
            )
        else:
            st.warning("No validated data found.")


if __name__ == "__main__":
    main()
