"""Postprocessing filters applied to segmentation results.

remove_corner_bboxes  — discard line / figure boxes whose centroid falls inside
                        the binding-side corners of the page.
remove_narrow_bboxes  — discard line / figure boxes that are too thin in either
                        dimension (likely ruling lines, folds, or scan artefacts).

Both functions operate on plain Python lists (not dataclasses) so they can be
used independently without importing heavy pipeline objects.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# remove_corner_bboxes
# ---------------------------------------------------------------------------

def remove_corner_bboxes(
    line_boundaries: List[List],
    figure_bboxes: List[Tuple[int, int, int, int]],
    image_h: int,
    image_w: int,
    binding_side: str,
    corner_fraction: float = 0.10,
) -> Tuple[List[List], List[Tuple[int, int, int, int]], int, int]:
    """Remove bounding boxes whose centroid falls inside a corner of the page.

    The corners that are tested depend on the *binding side* — the two corners
    adjacent to the binding (top-binding and bottom-binding) are considered
    unreliable because the page curves inward near the spine, producing
    detection artefacts.

    Parameters
    ----------
    line_boundaries : list of [[x, y], …]
        Per-line polygon vertices as returned by ``SegmentKrakenResult``.
    figure_bboxes : list of (x, y, w, h)
        Per-figure bounding boxes as returned by ``SegmentKrakenResult``.
    image_h, image_w : int
        Pixel dimensions of the deskewed image.
    binding_side : str
        ``"left"`` or ``"right"``.  Any other value → no corners removed.
    corner_fraction : float
        Fraction of image width *and* height that defines each corner square.
        Default is 0.10 (10 %).

    Returns
    -------
    filtered_lines, filtered_figures, n_lines_removed, n_figures_removed
    """
    if binding_side not in ("left", "right"):
        return line_boundaries, figure_bboxes, 0, 0

    corner_w = image_w * corner_fraction
    corner_h = image_h * corner_fraction

    def _in_corner(cx: float, cy: float) -> bool:
        """True if centroid (cx, cy) is inside one of the opposite to the two binding corners."""
        near_binding = (cx < corner_w) if binding_side == "right" else (cx > image_w - corner_w)
        near_top    = cy < corner_h
        near_bottom = cy > image_h - corner_h
        return near_binding and (near_top or near_bottom)

    # ── Filter line boundaries ────────────────────────────────────────────────
    kept_lines, removed_lines = [], 0
    for boundary in line_boundaries:
        pts = np.array(boundary, dtype=float)  # (N, 2)
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        if _in_corner(cx, cy):
            removed_lines += 1
        else:
            kept_lines.append(boundary)

    # ── Filter figure bboxes ──────────────────────────────────────────────────
    kept_figures, removed_figures = [], 0
    for x, y, w, h in figure_bboxes:
        cx, cy = x + w / 2.0, y + h / 2.0
        if _in_corner(cx, cy):
            removed_figures += 1
        else:
            kept_figures.append((x, y, w, h))

    return kept_lines, kept_figures, removed_lines, removed_figures


# ---------------------------------------------------------------------------
# remove_narrow_bboxes
# ---------------------------------------------------------------------------

def remove_narrow_bboxes(
    line_boundaries: List[List],
    figure_bboxes: List[Tuple[int, int, int, int]],
    min_dimension_px: int = 20,
) -> Tuple[List[List], List[Tuple[int, int, int, int]], int, int]:
    """Remove bounding boxes that are thinner than a minimum pixel size.

    A box is considered *narrow* when its shorter side (min of width and
    height) is below *min_dimension_px*.  This removes:

    * Thin horizontal artefacts — e.g. ruling lines, folds, scan edges
      where a polygon is very tall but only a few pixels wide.
    * Small fragments — stray marks with both dimensions below the threshold.

    Parameters
    ----------
    line_boundaries : list of [[x, y], …]
        Per-line polygon vertices.
    figure_bboxes : list of (x, y, w, h)
        Per-figure bounding boxes.
    min_dimension_px : int
        Minimum allowed size for the *shorter* side of the bounding box.
        Default is 20 px — sensible for typical manuscript scans at 300–400 dpi
        where a real text line is at least ~30 px tall.

    Returns
    -------
    filtered_lines, filtered_figures, n_lines_removed, n_figures_removed
    """
    # ── Filter line boundaries ────────────────────────────────────────────────
    kept_lines, removed_lines = [], 0
    for boundary in line_boundaries:
        pts = np.array(boundary, dtype=float)  # (N, 2)
        w = float(pts[:, 0].max() - pts[:, 0].min())
        h = float(pts[:, 1].max() - pts[:, 1].min())
        if min(w, h) < min_dimension_px:
            removed_lines += 1
        else:
            kept_lines.append(boundary)

    # ── Filter figure bboxes ──────────────────────────────────────────────────
    kept_figures, removed_figures = [], 0
    for x, y, w, h in figure_bboxes:
        if min(w, h) < min_dimension_px:
            removed_figures += 1
        else:
            kept_figures.append((x, y, w, h))

    return kept_lines, kept_figures, removed_lines, removed_figures
