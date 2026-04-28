"""Postprocessing filters applied to segmentation results.

remove_corner_bboxes  — discard line / figure boxes whose centroid falls inside
                        the binding-side corners of the page.
remove_narrow_bboxes  — discard line / figure boxes that are too thin in either
                        dimension (likely ruling lines, folds, or scan artefacts).

Both functions operate on plain Python lists (not dataclasses) so they can be
used independently without importing heavy pipeline objects.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

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
        if w < min_dimension_px:
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


# ---------------------------------------------------------------------------
# detect_and_split_gutter
# ---------------------------------------------------------------------------

def _clip_poly_left(
    pts: List[List[float]], x_cut: float
) -> List[List[float]]:
    """Sutherland-Hodgman clip: keep the portion of *pts* with x ≤ x_cut."""
    result: List[List[float]] = []
    n = len(pts)
    for i in range(n):
        curr = pts[i]
        nxt  = pts[(i + 1) % n]
        curr_in = curr[0] <= x_cut
        nxt_in  = nxt[0]  <= x_cut
        if curr_in:
            result.append(curr)
        if curr_in != nxt_in:
            dx = nxt[0] - curr[0]
            if abs(dx) > 1e-9:
                t = (x_cut - curr[0]) / dx
                result.append([x_cut, curr[1] + t * (nxt[1] - curr[1])])
    return result


def _clip_poly_right(
    pts: List[List[float]], x_cut: float
) -> List[List[float]]:
    """Sutherland-Hodgman clip: keep the portion of *pts* with x ≥ x_cut."""
    result: List[List[float]] = []
    n = len(pts)
    for i in range(n):
        curr = pts[i]
        nxt  = pts[(i + 1) % n]
        curr_in = curr[0] >= x_cut
        nxt_in  = nxt[0]  >= x_cut
        if curr_in:
            result.append(curr)
        if curr_in != nxt_in:
            dx = nxt[0] - curr[0]
            if abs(dx) > 1e-9:
                t = (x_cut - curr[0]) / dx
                result.append([x_cut, curr[1] + t * (nxt[1] - curr[1])])
    return result


def detect_and_split_gutter(
    line_boundaries: List[List],
    image_w: int,
    image_h: int,
    min_gutter_fraction: float = 0.60,
    central_zone: Tuple[float, float] = (0.35, 0.65),
    min_gutter_width_frac: float = 0.01,
    gutter_margin_px: int = 3,
    single_col_center_px: int = 250,
    single_col_threshold: float = 0.70,
    split_overflow_px: int = 100,
    gutter_y_tolerance_px: int = 20,
) -> Tuple[List[List], "Optional[int]", "Optional[int]", "Optional[int]", bool, int]:
    """Detect a central column-separator gutter and split crossed line polygons.

    The algorithm:

    0. **Single-column guard** — two independent checks, either of which
       short-circuits gutter detection:

       a. *Full-width lines* — if ≥ ``single_col_threshold`` of lines extend
          at least ``single_col_center_px`` pixels past the page centre on
          both sides, the page is single-column.
       b. *Centroid skew* — if ≥ ``single_col_threshold`` of line centroids
          (midpoint of x0 and x1) fall on the same side of the page centre x,
          the page is a single (possibly offset) column.
    1. Build a 1-D horizontal coverage histogram — for each x position, count
       how many line bounding-boxes include it.
    2. Find the x band with *minimum* coverage inside ``central_zone``.  That
       is the gutter candidate.
    3. Count lines that lie entirely to the left or right of the candidate
       (non-spanning).  If that fraction ≥ ``min_gutter_fraction`` (default
       60 %), declare the page double-column.
    4. Refine ``gutter_x`` as the midpoint between the median right-edge of
       left-column lines and the median left-edge of right-column lines.
    5. Verify the actual gap is at least ``min_gutter_width_frac * image_w``.
    6. Set ``gutter_y_min``/``gutter_y_max`` from contributing lines.
    7. Split only line polygons that overflow the gutter by at least
       ``split_overflow_px`` on *both* sides (i.e. x_min ≤ gutter_x −
       split_overflow_px *and* x_max ≥ gutter_x + split_overflow_px).
       Each half is inset by ``gutter_margin_px`` so there is a clean gap.

    Parameters
    ----------
    line_boundaries : list of [[x, y], …]
        Per-line polygon vertex lists.
    image_w, image_h : int
        Page pixel dimensions.
    min_gutter_fraction : float
        Minimum fraction of lines that must be non-spanning for a gutter to be
        declared.  Default 0.60 (60 %).
    central_zone : (float, float)
        (lo, hi) fractions of image width within which the gutter must lie.
        Default (0.25, 0.75).
    min_gutter_width_frac : float
        Minimum gutter width as a fraction of image width.  Default 0.01.
    gutter_margin_px : int
        Number of pixels to inset from the gutter centre when clipping split
        polygons.  The left half is clipped at ``gutter_x - gutter_margin_px``
        and the right half at ``gutter_x + gutter_margin_px``.  Default 3.
    single_col_center_px : int
        Minimum number of pixels a line must extend past the page centre on
        both sides to count as a "full-width" single-column line.  Default 200.
    single_col_threshold : float
        If at least this fraction of lines are full-width (per the criterion
        above), the page is treated as single-column and no gutter is
        searched.  Default 0.60.
    split_overflow_px : int
        A spanning line is only split if it extends at least this many pixels
        past the gutter on both sides.  Lines whose bounding box barely
        crosses the gutter (e.g. a long line whose end just reaches the
        centre) are left intact.  Default 100.
    gutter_y_tolerance_px : int
        A spanning line is only split if its y range overlaps the gutter's
        y span (``gutter_y_min`` … ``gutter_y_max``) or falls within this
        many pixels of either edge.  Polygons that span the gutter x but
        lie entirely above or below the gutter are left intact.  Default 20.

    Returns
    -------
    new_line_boundaries : list
        Line polygons after splitting.  Non-split polygons are unchanged.
    gutter_x : int or None
        Detected gutter x position in pixels.
    gutter_y_min : int or None
        Top of the gutter span (min y of contributing lines).
    gutter_y_max : int or None
        Bottom of the gutter span (max y of contributing lines).
    is_double_column : bool
    n_lines_split : int
        Number of polygons that were split.
    """
    _no_gutter = (line_boundaries, None, None, None, False, 0)

    n = len(line_boundaries)
    if n < 3:
        return _no_gutter

    # 1. Per-line bounding boxes
    bboxes: List[Tuple[float, float, float, float]] = []
    for boundary in line_boundaries:
        pts = np.array(boundary, dtype=float)
        bboxes.append((
            float(pts[:, 0].min()),
            float(pts[:, 0].max()),
            float(pts[:, 1].min()),
            float(pts[:, 1].max()),
        ))

    # 0. Single-column guard — either condition is sufficient.
    cx = image_w / 2.0

    # (a) Full-width lines: bbox extends single_col_center_px past centre on both sides.
    n_full_width = sum(
        1 for x0, x1, _, _ in bboxes
        if x0 <= cx - single_col_center_px and x1 >= cx + single_col_center_px
    )
    if n_full_width / n >= single_col_threshold:
        print(f"Single-column guard: {n_full_width} / {n} full-width lines")
        return _no_gutter

    # (b) Centroid skew: if ≥ threshold of centroids are on the same side of cx,
    #     the page has a single (possibly offset) text block.
    centroids_x = [(x0 + x1) / 2.0 for x0, x1, _, _ in bboxes]
    n_left_of_cx  = sum(1 for c in centroids_x if c < cx)
    n_right_of_cx = n - n_left_of_cx
    if max(n_left_of_cx, n_right_of_cx) / n >= single_col_threshold:
        print(f"Single-column guard: {max(n_left_of_cx, n_right_of_cx)} / {n} lines on one side of centre")
        return _no_gutter

    # 2. Coverage histogram (~200 bins)
    resolution = max(1, image_w // 200)
    n_bins = image_w // resolution + 2
    cov = np.zeros(n_bins, dtype=float)
    for x0, x1, _, _ in bboxes:
        lo = max(0, int(x0) // resolution)
        hi = min(n_bins - 1, int(x1) // resolution)
        cov[lo: hi + 1] += 1.0

    # 3. Valley in central zone
    cx_lo = max(0, int(central_zone[0] * image_w) // resolution)
    cx_hi = min(n_bins - 1, int(central_zone[1] * image_w) // resolution)
    if cx_lo >= cx_hi:
        return _no_gutter

    valley_rel  = int(np.argmin(cov[cx_lo: cx_hi + 1]))
    valley_bin  = cx_lo + valley_rel
    candidate_x = valley_bin * resolution + resolution // 2

    # 4. Non-spanning count
    n_left     = sum(1 for x0, x1, *_ in bboxes if x1 <= candidate_x)
    n_right    = sum(1 for x0, x1, *_ in bboxes if x0 >= candidate_x)
    n_non_span = n_left + n_right

    if n_non_span / n < min_gutter_fraction:
        return _no_gutter

    # 5. Refine gutter_x
    right_edges = [x1 for x0, x1, _, _ in bboxes if x1 <= candidate_x]
    left_edges  = [x0 for x0, x1, _, _ in bboxes if x0 >= candidate_x]

    if right_edges and left_edges:
        med_right = float(np.median(right_edges))
        med_left  = float(np.median(left_edges))
        if med_left - med_right < image_w * min_gutter_width_frac:
            return _no_gutter
        gutter_x = int((med_right + med_left) / 2.0)
    elif right_edges:
        gutter_x = int(float(np.median(right_edges)) + image_w * min_gutter_width_frac)
    elif left_edges:
        gutter_x = int(float(np.median(left_edges)) - image_w * min_gutter_width_frac)
    else:
        return _no_gutter

    # 6. y range from contributing (non-spanning) lines
    contrib_y = [
        (y0, y1)
        for x0, x1, y0, y1 in bboxes
        if x1 <= gutter_x or x0 >= gutter_x
    ]
    gutter_y_min = int(min(y0 for y0, _ in contrib_y))
    gutter_y_max = int(max(y1 for _, y1 in contrib_y))

    # 7. Split polygons that genuinely straddle the gutter.
    # Only split when the bounding box overflows the gutter by at least
    # split_overflow_px on both sides, so that lines whose end merely reaches
    # the gutter centre are left intact.
    # Each half is inset by gutter_margin_px to leave a clean gap.
    left_cut  = float(gutter_x - gutter_margin_px)
    right_cut = float(gutter_x + gutter_margin_px)
    new_boundaries: List[List] = []
    n_split = 0
    for i, boundary in enumerate(line_boundaries):
        x0, x1, y0, y1 = bboxes[i]
        genuinely_spans = (
            x0 <= gutter_x - split_overflow_px
            and x1 >= gutter_x + split_overflow_px
        )
        # Only split polygons that are intersected by the gutter vertically,
        # or close enough to its top/bottom edge.
        near_gutter_y = (
            y0 <= gutter_y_max + gutter_y_tolerance_px
            and y1 >= gutter_y_min - gutter_y_tolerance_px
        )
        if genuinely_spans and near_gutter_y:
            pts_f = [[float(p[0]), float(p[1])] for p in boundary]
            left_poly  = _clip_poly_left(pts_f, left_cut)
            right_poly = _clip_poly_right(pts_f, right_cut)
            if len(left_poly) >= 3:
                new_boundaries.append([[int(p[0]), int(p[1])] for p in left_poly])
            if len(right_poly) >= 3:
                new_boundaries.append([[int(p[0]), int(p[1])] for p in right_poly])
            n_split += 1
        else:
            new_boundaries.append(boundary)

    return new_boundaries, gutter_x, gutter_y_min, gutter_y_max, True, n_split
