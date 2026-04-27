#!/usr/bin/env python3
"""
CLI entry point — process one or all exemplar images through the full pipeline.

Usage:
    # Process a single image
    python run_pipeline.py data/exemplars/some_image.jpg

    # Process all exemplars and write results to CSV
    python run_pipeline.py --all --output results.csv

    # Use Kraken segmentation instead of HPP
    python run_pipeline.py --all --method kraken

    # Skip the embedding stage (faster, no GPU needed)
    python run_pipeline.py --all --no-embed
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np


# Ensure the project root is on the path when called as a script
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.preprocessing import preprocess
from pipeline.layout import detect_layout
from pipeline.masking import mask_non_text
from pipeline.segmentation import segment_lines
from pipeline.embeddings import (
    compute_embedding,
    anomaly_scores,
    compute_umap,
    PageEmbedding,
)


# ---------------------------------------------------------------------------
# Processing one image
# ---------------------------------------------------------------------------

def process_image(
    img_path: Path,
    method: str = "hpp",
    compute_embed: bool = True,
) -> Optional[dict]:
    """Run the full pipeline on a single image.

    Returns a result dict or None on failure.
    """
    try:
        # Stages 1–2
        prep = preprocess(img_path)

        # Stage 3
        layout = detect_layout(prep["masked"])

        # Stage 4
        masking = mask_non_text(prep["masked"], prep["gray"])

        # Stages 5–6
        seg = segment_lines(masking.text_binary, layout.columns, method=method)

        # Stage 7 (optional)
        embedding: Optional[PageEmbedding] = None
        if compute_embed:
            embedding = compute_embedding(
                filename=img_path.name,
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

        return {
            "filename": img_path.name,
            "path": img_path,
            "prep": prep,
            "layout": layout,
            "masking": masking,
            "seg": seg,
            "embedding": embedding,
        }

    except Exception as exc:
        print(f"[ERROR] {img_path.name}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def results_to_rows(result: dict) -> List[dict]:
    """Convert a pipeline result dict to a list of CSV row dicts."""
    rows = []
    for line in result["seg"].lines:
        x0, y0, x1, y1 = line.bbox
        rows.append({
            "filename": result["filename"],
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
            "deskew_angle": round(result["prep"]["deskew_angle"], 2),
            "anomaly_score": round(
                result["embedding"].anomaly_score if result["embedding"] else 0.0, 4
            ),
            "is_validated": 0,
        })
    return rows


FIELDNAMES = [
    "filename", "layout_type", "column", "line_index",
    "x_min", "y_min", "x_max", "y_max", "height",
    "is_anomalous", "method", "deskew_angle",
    "anomaly_score", "is_validated",
]


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
        "--exemplars-dir",
        default=str(PROJECT_ROOT / "data" / "exemplars"),
        help="Directory containing exemplar images (default: data/exemplars).",
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help="Output CSV file path (default: results.csv).",
    )
    parser.add_argument(
        "--method",
        choices=["hpp", "kraken"],
        default="hpp",
        help="Segmentation method: 'hpp' (default) or 'kraken'.",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip the embedding / anomaly scoring stage.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-line details.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compute_embed = not args.no_embed

    # Collect images to process
    if args.all:
        exemplars_dir = Path(args.exemplars_dir)
        images = sorted(
            p for p in exemplars_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        )
        if not images:
            print(f"No images found in {exemplars_dir}", file=sys.stderr)
            sys.exit(1)
    elif args.image:
        images = [Path(args.image)]
    else:
        print("Provide an image path or use --all.  See --help.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(images)} image(s) with method={args.method} …")

    results = []
    for img_path in images:
        print(f"  → {img_path.name}")
        result = process_image(img_path, method=args.method, compute_embed=compute_embed)
        if result is None:
            continue

        layout = result["layout"]
        seg = result["seg"]
        print(
            f"     layout={layout.layout_type}  "
            f"lines={len(seg.lines)}  "
            f"fallback={seg.fallback_triggered}  "
            f"anomalous={seg.anomalous_count}"
        )
        if args.verbose:
            for line in seg.lines:
                print(f"       col={line.column_index} line={line.line_index} "
                      f"bbox={line.bbox} h={line.height} {'[!]' if line.is_anomalous else ''}")

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
            ranked = sorted(embeddings, key=lambda e: e.anomaly_score, reverse=True)
            for e in ranked:
                print(f"  {e.anomaly_score:.3f}  {e.filename}")

    # Write CSV
    all_rows = []
    for r in results:
        all_rows.extend(results_to_rows(r))

    if all_rows:
        write_csv(all_rows, Path(args.output))
    else:
        print("No results to write.")


if __name__ == "__main__":
    main()
