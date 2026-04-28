#!/usr/bin/env python3
"""
CLI entry point — process one or all exemplar images through the full pipeline.

Usage:
    # Process a single image
    python run_pipeline.py data/exemplars/some_image.jpg

    # Process all exemplars and write results to CSV
    python run_pipeline.py --all --output results.csv

    # Use Apple Silicon GPU
    python run_pipeline.py --all --device mps

    # Skip the embedding stage (faster, no GPU needed)
    python run_pipeline.py --all --no-embed
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np


# Ensure the project root is on the path when called as a script
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.stages import (
    PreprocessResult, preprocess_page,
    SegmentKrakenResult, segment_kraken,
    PostprocessResult, postprocess,
)
from pipeline.embeddings import (
    compute_embedding,
    anomaly_scores,
    compute_umap,
    PageEmbedding,
)

# Default Kraken model path (override via --model-path)
_DEFAULT_MODEL = (
    "/Users/luissalamanca/Library/Application Support/htrmopo/"
    "97665cf3-f83d-5594-8855-f28d3af9df7a/blla.mlmodel"
)

# ---------------------------------------------------------------------------
# Result image helper
# ---------------------------------------------------------------------------

def _save_result_image(
    img_path: Path,
    pre: "PreprocessResult",
    seg: "SegmentKrakenResult",
    post: "PostprocessResult",
    results_dir: Optional[Path] = None,
) -> None:
    """Save a side-by-side image: original BGR | annotated overlay.

    The annotated panel shows:
    - Red polygons for each kept Kraken text-line boundary.
    - A semi-transparent yellow fill over figure/illustration regions.

    Output file: <stem>_result<suffix> written to *results_dir* (or the
    same folder as the source image when *results_dir* is None).
    """
    import cv2 as _cv2
    import numpy as _np

    bgr = pre.bgr.copy()

    # ── Annotated panel ───────────────────────────────────────────────────
    annotated = bgr.copy()

    # Semi-transparent yellow mask over figure ink pixels
    if seg.figure_binary is not None and seg.figure_binary.any():
        fig_mask = seg.figure_binary > 0
        overlay = annotated.copy()
        overlay[fig_mask] = (0, 220, 220)   # BGR yellow
        _cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

    # Red line-boundary polygons
    for boundary in post.line_boundaries:
        pts = _np.array(boundary, dtype=_np.int32).reshape(-1, 1, 2)
        _cv2.polylines(annotated, [pts], isClosed=True, color=(0, 0, 220), thickness=2)

    # Gutter line (green, thickness 4) when double-column detected
    if post.is_double_column and post.gutter_x is not None:
        _cv2.line(
            annotated,
            (post.gutter_x, post.gutter_y_min),
            (post.gutter_x, post.gutter_y_max),
            (0, 220, 0),   # BGR green
            thickness=4,
        )

    # ── Side-by-side ──────────────────────────────────────────────────────
    h = max(bgr.shape[0], annotated.shape[0])
    if bgr.shape[0] != h:
        bgr = _cv2.resize(bgr, (int(bgr.shape[1] * h / bgr.shape[0]), h))
    if annotated.shape[0] != h:
        annotated = _cv2.resize(annotated, (int(annotated.shape[1] * h / annotated.shape[0]), h))

    combined = _np.hstack([bgr, annotated])

    out_dir = results_dir if results_dir is not None else img_path.parent
    out_path = out_dir / (img_path.stem + "_result" + img_path.suffix)
    _cv2.imwrite(str(out_path), combined)


# ---------------------------------------------------------------------------
# Processing one image
# ---------------------------------------------------------------------------

import signal as _signal
import contextlib as _contextlib


class _SegmentTimeout(Exception):
    """Raised when segment_kraken exceeds its walltime."""


@_contextlib.contextmanager
def _walltime(seconds: int):
    """Context manager that raises _SegmentTimeout after *seconds* seconds."""
    def _handler(signum, frame):
        raise _SegmentTimeout(f"segment_kraken timed out after {seconds}s")

    old_handler = _signal.signal(_signal.SIGALRM, _handler)
    _signal.alarm(seconds)
    try:
        yield
    finally:
        _signal.alarm(0)
        _signal.signal(_signal.SIGALRM, old_handler)


def _results_dir(img_path: Path) -> Path:
    """Return (and create) a 'results/' subfolder next to the image."""
    d = img_path.parent / "results"
    d.mkdir(exist_ok=True)
    return d


def _record_failure(img_path: Path, reason: str) -> None:
    """Append the filename + reason to results/failed.txt."""
    failed_txt = _results_dir(img_path) / "failed.txt"
    with open(failed_txt, "a") as f:
        f.write(f"{img_path.name}\t{reason}\n")


def process_image(
    img_path: Path,
    model_path: str = _DEFAULT_MODEL,
    device: str = "cpu",
    corner_fraction: float = 0.10,
    min_dimension_px: int = 20,
    compute_embed: bool = True,
    use_vit_mask: bool = False,
    min_gutter_fraction: float = 0.50,
    single_col_threshold: float = 0.70,
    segment_walltime: int = 30,
) -> Optional[dict]:
    """Run the full pipeline on a single image.

    Returns a result dict with keys:
        filename, path, pre, seg, post, embedding
    or None on failure.

    Outputs are written to a ``results/`` subfolder inside the image's
    parent directory (created automatically).  If ``segment_kraken`` does
    not complete within *segment_walltime* seconds, or raises an exception,
    the filename is appended to ``results/failed.txt`` and None is returned.
    """
    results_dir = _results_dir(img_path)

    # ── Preprocessing ────────────────────────────────────────────────────
    try:
        pre = preprocess_page(img_path, model_path=model_path, device=device)
    except Exception as exc:
        reason = f"preprocess error: {exc}"
        print(f"[ERROR] {img_path.name}: {reason}", file=sys.stderr)
        traceback.print_exc()
        _record_failure(img_path, reason)
        return None

    # ── Segmentation (with walltime) ─────────────────────────────────────
    try:
        with _walltime(segment_walltime):
            seg = segment_kraken(pre, model_path=model_path, device=device)
    except _SegmentTimeout as exc:
        reason = str(exc)
        print(f"[TIMEOUT] {img_path.name}: {reason}", file=sys.stderr)
        _record_failure(img_path, reason)
        return None
    except Exception as exc:
        reason = f"segment error: {exc}"
        print(f"[ERROR] {img_path.name}: {reason}", file=sys.stderr)
        traceback.print_exc()
        _record_failure(img_path, reason)
        return None

    # ── Postprocessing + embedding ────────────────────────────────────────
    try:
        post = postprocess(pre, seg,
                           corner_fraction=corner_fraction,
                           min_dimension_px=min_dimension_px,
                           min_gutter_fraction=min_gutter_fraction,
                           single_col_threshold=single_col_threshold)

        embedding: Optional[PageEmbedding] = None
        if compute_embed:
            embedding = compute_embedding(
                filename=img_path.name,
                pre=pre,
                seg=seg,
                post=post,
                use_structural=True,
                use_vit_rgb=True,
                use_vit_mask=use_vit_mask,
            )

        result = {
            "filename": img_path.name,
            "path": img_path,
            "pre": pre,
            "seg": seg,
            "post": post,
            "embedding": embedding,
        }

        # Save pickle and result image inside results/
        pkl_path = results_dir / img_path.with_suffix(".pkl").name
        with open(pkl_path, "wb") as f:
            pickle.dump(result, f)

        _save_result_image(img_path, pre, seg, post, results_dir=results_dir)

        return result

    except Exception as exc:
        print(f"[ERROR] {img_path.name}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# CSV export  (one row per page)
# ---------------------------------------------------------------------------

FIELDNAMES = [
    # Identity
    "filename",
    # Preprocess
    "image_h", "image_w", "deskew_angle", "binding_side",
    "margin_width", "top_margin", "ruler_height", "bottom_margin", "binding_width",
    # Segment (raw Kraken output)
    "n_lines_raw", "text_coverage", "text_px_kept", "text_px_input",
    "n_figures_raw", "figure_coverage",
    # Postprocess (after corner + narrow filtering)
    "n_lines", "n_lines_removed_corner", "n_lines_removed_narrow",
    "n_figures", "n_figures_removed_corner", "n_figures_removed_narrow",
    # Embedding / anomaly
    "anomaly_score",
]


def result_to_row(result: dict) -> dict:
    """Convert a pipeline result dict to a single CSV row."""
    pre:  PreprocessResult  = result["pre"]
    seg:  SegmentKrakenResult = result["seg"]
    post: PostprocessResult = result["post"]
    emb:  Optional[PageEmbedding] = result["embedding"]

    return {
        "filename":       result["filename"],
        # Preprocess
        "image_h":        pre.image_h,
        "image_w":        pre.image_w,
        "deskew_angle":   round(pre.deskew_angle, 3),
        "binding_side":   pre.binding_side,
        "margin_width":   pre.margin_width,
        "top_margin":     pre.top_margin,
        "ruler_height":   pre.ruler_height,
        "bottom_margin":  pre.bottom_margin,
        "binding_width":  pre.binding_width,
        # Segment
        "n_lines_raw":    seg.n_lines,
        "text_coverage":  round(seg.text_coverage, 4),
        "text_px_kept":   seg.text_px_kept,
        "text_px_input":  seg.text_px_input,
        "n_figures_raw":  seg.n_figures,
        "figure_coverage": round(seg.figure_coverage, 4),
        # Postprocess
        "n_lines":                  post.n_lines,
        "n_lines_removed_corner":   post.n_lines_removed_corner,
        "n_lines_removed_narrow":   post.n_lines_removed_narrow,
        "n_figures":                post.n_figures,
        "n_figures_removed_corner": post.n_figures_removed_corner,
        "n_figures_removed_narrow": post.n_figures_removed_narrow,
        # Anomaly
        "anomaly_score": round(emb.anomaly_score if emb else 0.0, 4),
    }


def write_csv(rows: List[dict], output_path: Path) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Medieval manuscript line-counting pipeline"
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to a single image to process.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all images in the exemplars directory.",
    )
    parser.add_argument(
        "--dir",
        default=str(PROJECT_ROOT / "data" / "exemplars"),
        help="Directory containing exemplar images (default: data/exemplars).",
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help="Output CSV file path (default: results.csv).",
    )
    parser.add_argument(
        "--model-path",
        default=_DEFAULT_MODEL,
        help="Path to the Kraken .mlmodel segmentation file.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device string: 'cpu', 'mps', 'cuda' (default: cpu).",
    )
    parser.add_argument(
        "--corner-fraction",
        type=float,
        default=0.10,
        help="Corner zone size as fraction of image dimensions (default: 0.10).",
    )
    parser.add_argument(
        "--min-dimension-px",
        type=int,
        default=70,
        help="Min short-side pixels to keep a bbox (default: 70).",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip the embedding / anomaly scoring stage.",
    )
    parser.add_argument(
        "--vit-mask",
        action="store_true",
        help="Also compute a ViT embedding from the text+figure overlay image.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-page feature details.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compute_embed = not args.no_embed

    # Collect images
    if args.all:
        exemplars_dir = Path(args.dir)
        images = sorted(
            p for p in exemplars_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
            and not p.stem.endswith("_result")
        )
        if not images:
            print(f"No images found in {exemplars_dir}", file=sys.stderr)
            sys.exit(1)
    elif args.image:
        images = [Path(args.image)]
    else:
        print("Provide an image path or use --all.  See --help.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(images)} image(s)  device={args.device} …")

    results = []
    for img_path in images:
        print(f"  → {img_path.name}")
        result = process_image(
            img_path,
            model_path=args.model_path,
            device=args.device,
            corner_fraction=args.corner_fraction,
            min_dimension_px=args.min_dimension_px,
            compute_embed=compute_embed,
            use_vit_mask=args.vit_mask,
        )
        if result is None:
            continue

        seg  = result["seg"]
        post = result["post"]
        print(
            f"     lines={seg.n_lines} (kept {post.n_lines})  "
            f"figures={seg.n_figures} (kept {post.n_figures})  "
            f"text_cov={seg.text_coverage:.2%}"
        )
        if args.verbose:
            pre = result["pre"]
            print(
                f"     binding={pre.binding_side}  "
                f"skew={pre.deskew_angle:.2f}°  "
                f"size={pre.image_w}×{pre.image_h}"
            )
            print(
                f"     removed_corner lines={post.n_lines_removed_corner} "
                f"figs={post.n_figures_removed_corner}  "
                f"removed_narrow lines={post.n_lines_removed_narrow} "
                f"figs={post.n_figures_removed_narrow}"
            )

        results.append(result)

    # Anomaly scoring across all pages
    if compute_embed and results:
        embeddings = [r["embedding"] for r in results if r["embedding"] is not None]
        if embeddings:
            print(f"\nScoring {len(embeddings)} embeddings …")
            anomaly_scores(embeddings)
            if len(embeddings) >= 4:
                compute_umap(embeddings)

            print("\n--- Anomaly ranking (most anomalous first) ---")
            for e in sorted(embeddings, key=lambda e: e.anomaly_score, reverse=True):
                print(f"  {e.anomaly_score:.3f}  {e.filename}")

    # Write CSV
    rows = [result_to_row(r) for r in results]
    if rows:
        write_csv(rows, Path(args.output))
    else:
        print("No results to write.")


if __name__ == "__main__":
    main()
