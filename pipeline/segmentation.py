"""
Stages 5–6 — Line segmentation and bounding boxes

Stage 5: Horizontal Projection Profile (HPP) with Kraken fallback.
Stage 6: Tight bounding boxes per detected line with vertical padding.

Two segmentation methods are always available and selectable via the
`method` parameter ('hpp' | 'kraken').  When `method='hpp'`, automatic
fallback to Kraken occurs if > 15% of lines are anomalously tall.

Run standalone:

    python -m pipeline.segmentation path/to/image.jpg
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from .layout import ColumnRegion


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LineResult:
    """A single detected text line."""
    column_index: int
    line_index: int                        # 0-based within its column
    bbox: Tuple[int, int, int, int]        # (x_min, y_min, x_max, y_max)
    height: int
    is_anomalous: bool = False
    method: str = "hpp"                    # 'hpp' or 'kraken'


@dataclass
class SegmentationResult:
    """Full segmentation result for one page."""
    lines: List[LineResult]
    method_used: str                       # 'hpp' or 'kraken'
    fallback_triggered: bool
    anomalous_count: int
    median_line_height: float
    per_column_counts: List[int]


# ---------------------------------------------------------------------------
# HPP segmentation helpers
# ---------------------------------------------------------------------------

def _horizontal_projection(binary_region: np.ndarray) -> np.ndarray:
    """Sum foreground pixels row-wise for a binary region."""
    return binary_region.sum(axis=1).astype(float)


def _detect_line_centres(
    hpp: np.ndarray,
    min_peak_distance: int = 5,
    min_peak_height_frac: float = 0.02,
) -> np.ndarray:
    """Detect line centres from the horizontal projection profile.

    Parameters
    ----------
    hpp : np.ndarray
        1-D horizontal projection profile.
    min_peak_distance : int
        Minimum distance between adjacent peaks (pixels).
    min_peak_height_frac : float
        Minimum peak height as a fraction of the maximum value.

    Returns
    -------
    np.ndarray
        Array of peak positions (y-coordinates within the region).
    """
    if hpp.max() == 0:
        return np.array([], dtype=int)
    min_h = hpp.max() * min_peak_height_frac
    peaks, _ = find_peaks(hpp, height=min_h, distance=min_peak_distance)
    return peaks


def _line_bbox_from_centre(
    binary_region: np.ndarray,
    y_centre: int,
    half_window: int,
    x_offset: int,
    y_offset: int,
    v_padding: int = 2,
) -> Tuple[int, int, int, int]:
    """Compute a tight bounding box for a line centred at *y_centre*.

    Scans within ±half_window rows of the centre, collects all foreground
    pixel columns, and returns (x_min, y_min, x_max, y_max) in *image*
    coordinates (applying x_offset and y_offset).

    v_padding : int
        Extra pixels added above and below to capture ascenders/descenders.
    """
    h_reg, w_reg = binary_region.shape
    y0 = max(0, y_centre - half_window)
    y1 = min(h_reg, y_centre + half_window + 1)

    band = binary_region[y0:y1, :]
    if band.max() == 0:
        # Fall back to full band extent
        return (
            x_offset,
            y_offset + y0,
            x_offset + w_reg,
            y_offset + y1,
        )

    rows_with_ink = np.where(band.sum(axis=1) > 0)[0]
    cols_with_ink = np.where(band.sum(axis=0) > 0)[0]

    y_min_local = rows_with_ink[0]
    y_max_local = rows_with_ink[-1]
    x_min_local = cols_with_ink[0]
    x_max_local = cols_with_ink[-1]

    return (
        x_offset + x_min_local,
        y_offset + y0 + max(0, y_min_local - v_padding),
        x_offset + x_max_local + 1,
        y_offset + y0 + min(y1 - y0, y_max_local + v_padding + 1),
    )


def _segment_column_hpp(
    text_binary: np.ndarray,
    col: ColumnRegion,
    col_idx: int,
    v_padding: int = 2,
    min_peak_distance: int = 5,
    min_peak_height_frac: float = 0.02,
) -> List[LineResult]:
    """HPP line segmentation for a single column region."""
    region = text_binary[col.y_start : col.y_end, col.x_start : col.x_end]
    hpp = _horizontal_projection(region)
    peaks = _detect_line_centres(hpp, min_peak_distance, min_peak_height_frac)

    if len(peaks) == 0:
        return []

    # Estimate half-window from median inter-peak spacing
    if len(peaks) > 1:
        spacings = np.diff(peaks)
        half_w = max(2, int(np.median(spacings) // 2))
    else:
        half_w = max(2, region.shape[0] // 4)

    lines: List[LineResult] = []
    for line_idx, yc in enumerate(peaks):
        bbox = _line_bbox_from_centre(
            region, yc, half_w,
            x_offset=col.x_start,
            y_offset=col.y_start,
            v_padding=v_padding,
        )
        h = bbox[3] - bbox[1]
        lines.append(LineResult(
            column_index=col_idx,
            line_index=line_idx,
            bbox=bbox,
            height=h,
            method="hpp",
        ))
    return lines


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def _flag_anomalies(
    lines: List[LineResult],
    anomaly_height_factor: float = 2.5,
) -> Tuple[List[LineResult], float, int]:
    """Flag lines whose height > anomaly_height_factor × median.

    Returns (updated_lines, median_height, anomalous_count).
    Large initials that are isolated (neighbours are clean) are flagged but
    excluded from triggering Kraken fallback.
    """
    if not lines:
        return lines, 0.0, 0

    heights = np.array([l.height for l in lines])
    median_h = float(np.median(heights))
    threshold = anomaly_height_factor * median_h

    for line in lines:
        line.is_anomalous = line.height > threshold

    anomalous_count = sum(1 for l in lines if l.is_anomalous)
    return lines, median_h, anomalous_count


def _is_isolated_initial(
    line: LineResult,
    all_lines: List[LineResult],
    neighbour_window: int = 2,
) -> bool:
    """Return True if an anomalously tall line is surrounded by clean lines
    (= likely a large decorated initial rather than a segmentation failure)."""
    if not line.is_anomalous:
        return False
    col_lines = sorted(
        [l for l in all_lines if l.column_index == line.column_index],
        key=lambda l: l.line_index,
    )
    idx = next((i for i, l in enumerate(col_lines) if l is line), None)
    if idx is None:
        return False
    neighbours = col_lines[max(0, idx - neighbour_window) : idx + neighbour_window + 1]
    clean_neighbours = [l for l in neighbours if l is not line and not l.is_anomalous]
    return len(clean_neighbours) >= min(neighbour_window, len(neighbours) - 1)


# ---------------------------------------------------------------------------
# Kraken fallback
# ---------------------------------------------------------------------------

def _segment_column_kraken(
    text_binary: np.ndarray,
    col: ColumnRegion,
    col_idx: int,
) -> List[LineResult]:
    """Kraken-based line segmentation for a single column region.

    Uses kraken.blla.segment (baseline segmentation) on the column crop.
    Falls back to an empty list if kraken is unavailable or fails.
    """
    try:
        from PIL import Image
        import kraken.blla as blla
        import kraken.lib.models as kmodels
    except ImportError:
        print("[WARNING] kraken not installed; returning empty segmentation.", file=sys.stderr)
        return []

    region = text_binary[col.y_start : col.y_end, col.x_start : col.x_end]
    # kraken expects a PIL image; convert binary → RGB
    pil_img = Image.fromarray(region).convert("RGB")

    try:
        seg = blla.segment(pil_img)
    except Exception as exc:
        print(f"[WARNING] Kraken segmentation failed: {exc}", file=sys.stderr)
        return []

    lines: List[LineResult] = []
    for line_idx, line in enumerate(seg.lines):
        # Kraken returns baseline + bounding polygon; use bbox
        xs = [pt[0] for pt in line.boundary]
        ys = [pt[1] for pt in line.boundary]
        x_min = col.x_start + min(xs)
        y_min = col.y_start + min(ys)
        x_max = col.x_start + max(xs)
        y_max = col.y_start + max(ys)
        h = y_max - y_min
        lines.append(LineResult(
            column_index=col_idx,
            line_index=line_idx,
            bbox=(x_min, y_min, x_max, y_max),
            height=h,
            method="kraken",
        ))
    return lines


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def segment_lines(
    text_binary: np.ndarray,
    columns: List[ColumnRegion],
    method: str = "hpp",
    anomaly_height_factor: float = 2.5,
    fallback_threshold: float = 0.15,
    v_padding: int = 2,
    min_peak_distance: int = 5,
    min_peak_height_frac: float = 0.02,
) -> SegmentationResult:
    """Segment text lines in each column of the page.

    Parameters
    ----------
    text_binary : np.ndarray
        Binary image after non-text masking (Stage 4).
    columns : List[ColumnRegion]
        Column regions from Stage 3.
    method : str
        'hpp' or 'kraken'.  When 'hpp', Kraken fallback is applied
        per-column if anomaly rate > fallback_threshold.
    anomaly_height_factor : float
        Multiplier of median height to flag anomalous lines.
    fallback_threshold : float
        Fraction of anomalous lines that triggers Kraken fallback.
    v_padding : int
        Vertical pixel padding for bounding boxes.
    min_peak_distance : int
        HPP: minimum distance between adjacent line peaks.
    min_peak_height_frac : float
        HPP: minimum peak height fraction.

    Returns
    -------
    SegmentationResult
    """
    all_lines: List[LineResult] = []
    fallback_triggered = False
    per_column_counts: List[int] = []

    for col_idx, col in enumerate(columns):
        if method == "kraken":
            col_lines = _segment_column_kraken(text_binary, col, col_idx)
        else:
            col_lines = _segment_column_hpp(
                text_binary, col, col_idx,
                v_padding=v_padding,
                min_peak_distance=min_peak_distance,
                min_peak_height_frac=min_peak_height_frac,
            )

        # Flag anomalies on this column's lines
        col_lines, _, anomalous = _flag_anomalies(col_lines, anomaly_height_factor)

        # Identify isolated initials (they won't count toward fallback)
        isolated_initials = [l for l in col_lines if _is_isolated_initial(l, col_lines)]
        non_isolated_anomalous = anomalous - len(isolated_initials)

        # Decide on fallback
        if (
            method == "hpp"
            and len(col_lines) > 0
            and non_isolated_anomalous / len(col_lines) > fallback_threshold
        ):
            print(
                f"[INFO] Column {col_idx}: {non_isolated_anomalous}/{len(col_lines)} "
                f"anomalous lines → falling back to Kraken.",
                file=sys.stderr,
            )
            col_lines = _segment_column_kraken(text_binary, col, col_idx)
            col_lines, _, _ = _flag_anomalies(col_lines, anomaly_height_factor)
            fallback_triggered = True

        all_lines.extend(col_lines)
        per_column_counts.append(len(col_lines))

    # Global anomaly stats
    all_lines, median_h, total_anomalous = _flag_anomalies(all_lines, anomaly_height_factor)
    method_used = "kraken" if (method == "kraken" or fallback_triggered) else "hpp"

    return SegmentationResult(
        lines=all_lines,
        method_used=method_used,
        fallback_triggered=fallback_triggered,
        anomalous_count=total_anomalous,
        median_line_height=median_h,
        per_column_counts=per_column_counts,
    )


# ---------------------------------------------------------------------------
# __main__ test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    from pipeline.preprocessing import preprocess
    from pipeline.layout import detect_layout
    from pipeline.masking import mask_non_text

    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if img_path is None:
        exemplars_dir = Path(__file__).parent.parent / "data" / "exemplars"
        candidates = sorted(exemplars_dir.glob("*.jpg"))
        img_path = candidates[0] if candidates else None
    if img_path is None:
        print("No exemplar images found.")
        sys.exit(1)

    print(f"Processing: {img_path}")
    prep = preprocess(img_path)
    layout = detect_layout(prep["masked"])
    masking = mask_non_text(prep["masked"], prep["gray"])
    seg = segment_lines(masking.text_binary, layout.columns)

    print(f"  Method used   : {seg.method_used}")
    print(f"  Fallback      : {seg.fallback_triggered}")
    print(f"  Total lines   : {len(seg.lines)}")
    print(f"  Per-column    : {seg.per_column_counts}")
    print(f"  Median height : {seg.median_line_height:.1f}px")
    print(f"  Anomalous     : {seg.anomalous_count}")

    fig, ax = plt.subplots(figsize=(10, 14))
    ax.imshow(masking.text_binary, cmap="gray")
    colours = ["red", "blue", "green", "orange", "cyan"]
    for line in seg.lines:
        x0, y0, x1, y1 = line.bbox
        colour = colours[line.column_index % len(colours)]
        edgecolour = "magenta" if line.is_anomalous else colour
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=1, edgecolor=edgecolour, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(x0, y0, str(line.line_index), fontsize=5, color=colour)
    ax.set_title(
        f"{Path(img_path).name}\n"
        f"Layout: {layout.layout_type} | Lines: {len(seg.lines)} | "
        f"Anomalous: {seg.anomalous_count}"
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("/tmp/segmentation_test.png", dpi=120)
    print("Saved preview to /tmp/segmentation_test.png")
    plt.show()
