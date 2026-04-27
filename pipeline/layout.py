"""
Stage 3 — Layout detection

Divides the masked binary image into horizontal slabs and uses the
vertical projection profile of each slab to detect the number of text
columns (1 or 2).  Handles mixed layouts (transition from one layout
to another mid-page).

Run standalone:

    python -m pipeline.layout path/to/image.jpg
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ColumnRegion:
    """A single text-column region within the page."""
    x_start: int
    x_end: int
    y_start: int
    y_end: int

    @property
    def width(self) -> int:
        return self.x_end - self.x_start

    @property
    def height(self) -> int:
        return self.y_end - self.y_start

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x_start, self.y_start, self.x_end, self.y_end)


@dataclass
class LayoutResult:
    """Result of layout detection for a full page."""
    layout_type: str                  # 'single' | 'double' | 'mixed'
    columns: List[ColumnRegion]       # one or two ColumnRegion per zone
    transition_row: Optional[int]     # y-pixel where layout changes (mixed only)
    upper_layout: Optional[str]       # layout type of upper zone (mixed only)
    lower_layout: Optional[str]       # layout type of lower zone (mixed only)
    column_separator: Optional[Tuple[int, int]]  # (x_start, x_end) of separator
    slab_votes: List[int] = field(default_factory=list)  # columns per slab


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _vertical_projection(binary_slab: np.ndarray) -> np.ndarray:
    """Sum foreground pixels column-wise."""
    return binary_slab.sum(axis=0).astype(float)


def _count_columns_in_profile(
    profile: np.ndarray,
    min_valley_depth_frac: float = 0.30,
    min_valley_width: int = 10,
) -> Tuple[int, Optional[Tuple[int, int]]]:
    """Detect whether a vertical projection profile corresponds to 1 or 2
    text columns by looking for a clear valley in the middle third of the
    profile.

    Returns
    -------
    n_cols : int
        1 or 2.
    separator : (x_start, x_end) or None
        Column separator pixel range (only when 2 columns detected).
    """
    w = len(profile)
    if w < 30:
        return 1, None

    # Restrict search to middle 50% to avoid false valleys at margins
    lo = w // 4
    hi = 3 * w // 4
    mid_profile = profile[lo:hi]

    if mid_profile.max() == 0:
        return 1, None

    # Normalise
    norm = mid_profile / (mid_profile.max() + 1e-9)

    # Find valleys (inverted peaks)
    peaks, props = find_peaks(
        -norm,
        height=-( 1 - min_valley_depth_frac),
        width=min_valley_width,
    )
    if len(peaks) == 0:
        return 1, None

    # Use the deepest valley closest to the centre
    centre = len(norm) // 2
    valley_depths = -norm[peaks]
    # Score: depth + closeness to centre
    scores = valley_depths - 0.5 * (np.abs(peaks - centre) / centre)
    best = peaks[np.argmax(scores)]
    best_global = best + lo

    # Determine separator extent
    half_w = props["widths"][np.argmax(scores)] / 2
    sep_start = max(0, int(best_global - half_w))
    sep_end = min(w - 1, int(best_global + half_w))

    return 2, (sep_start, sep_end)


def _analyse_slab(
    binary: np.ndarray,
    y_start: int,
    y_end: int,
) -> Tuple[int, Optional[Tuple[int, int]]]:
    """Analyse a single horizontal slab of the binary image."""
    slab = binary[y_start:y_end, :]
    profile = _vertical_projection(slab)
    return _count_columns_in_profile(profile)


def _majority_vote(votes: List[int]) -> int:
    if not votes:
        return 1
    return 1 if votes.count(1) >= votes.count(2) else 2


def _find_transition_row(
    votes: List[int], slab_height: int
) -> Optional[int]:
    """Return the y-pixel of the first layout change, or None if uniform."""
    for i in range(1, len(votes)):
        if votes[i] != votes[i - 1]:
            return i * slab_height
    return None


def _column_regions_from_separator(
    binary: np.ndarray,
    sep: Optional[Tuple[int, int]],
    y_start: int,
    y_end: int,
    n_cols: int,
    x_margin: int = 5,
) -> List[ColumnRegion]:
    """Build ColumnRegion objects given separator info."""
    h, w = binary.shape
    if n_cols == 1 or sep is None:
        return [ColumnRegion(x_start=0, x_end=w, y_start=y_start, y_end=y_end)]
    sep_start, sep_end = sep
    left = ColumnRegion(
        x_start=0,
        x_end=max(0, sep_start - x_margin),
        y_start=y_start,
        y_end=y_end,
    )
    right = ColumnRegion(
        x_start=min(w, sep_end + x_margin),
        x_end=w,
        y_start=y_start,
        y_end=y_end,
    )
    return [left, right]


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def detect_layout(
    masked_binary: np.ndarray,
    n_slabs: int = 20,
    min_valley_depth_frac: float = 0.30,
    min_valley_width: int = 10,
) -> LayoutResult:
    """Detect column layout from a masked binary page image.

    Parameters
    ----------
    masked_binary : np.ndarray
        Binary image with non-page regions already zeroed.
    n_slabs : int
        Number of horizontal slabs to analyse.
    min_valley_depth_frac : float
        Minimum relative valley depth to consider a column separator.
    min_valley_width : int
        Minimum pixel width of the valley.

    Returns
    -------
    LayoutResult
    """
    h, w = masked_binary.shape
    slab_h = max(1, h // n_slabs)

    votes: List[int] = []
    separators: List[Optional[Tuple[int, int]]] = []

    for i in range(n_slabs):
        y0 = i * slab_h
        y1 = min(h, (i + 1) * slab_h)
        n_cols, sep = _analyse_slab(masked_binary, y0, y1)
        votes.append(n_cols)
        separators.append(sep)

    transition = _find_transition_row(votes, slab_h)

    # Aggregate separator: median of detected separators
    valid_seps = [s for s in separators if s is not None]
    agg_sep: Optional[Tuple[int, int]] = None
    if valid_seps:
        starts = np.median([s[0] for s in valid_seps])
        ends = np.median([s[1] for s in valid_seps])
        agg_sep = (int(starts), int(ends))

    if transition is None:
        # Uniform layout
        n_cols = _majority_vote(votes)
        layout_type = "single" if n_cols == 1 else "double"
        columns = _column_regions_from_separator(
            masked_binary, agg_sep if n_cols == 2 else None, 0, h, n_cols
        )
        return LayoutResult(
            layout_type=layout_type,
            columns=columns,
            transition_row=None,
            upper_layout=None,
            lower_layout=None,
            column_separator=agg_sep if n_cols == 2 else None,
            slab_votes=votes,
        )
    else:
        # Mixed layout — split at transition row
        upper_votes = votes[: transition // slab_h]
        lower_votes = votes[transition // slab_h :]
        upper_n = _majority_vote(upper_votes)
        lower_n = _majority_vote(lower_votes)

        upper_seps = [s for v, s in zip(votes, separators) if v == 2 and separators.index(s) < transition // slab_h]
        lower_seps = [s for v, s in zip(votes, separators) if v == 2 and separators.index(s) >= transition // slab_h]

        def _agg(seps):
            valid = [s for s in seps if s is not None]
            if not valid:
                return None
            return (int(np.median([s[0] for s in valid])), int(np.median([s[1] for s in valid])))

        upper_sep = _agg(upper_seps) if upper_n == 2 else None
        lower_sep = _agg(lower_seps) if lower_n == 2 else None

        upper_cols = _column_regions_from_separator(
            masked_binary, upper_sep, 0, transition, upper_n
        )
        lower_cols = _column_regions_from_separator(
            masked_binary, lower_sep, transition, h, lower_n
        )

        upper_label = "single" if upper_n == 1 else "double"
        lower_label = "single" if lower_n == 1 else "double"

        return LayoutResult(
            layout_type="mixed",
            columns=upper_cols + lower_cols,
            transition_row=transition,
            upper_layout=upper_label,
            lower_layout=lower_label,
            column_separator=agg_sep,
            slab_votes=votes,
        )


# ---------------------------------------------------------------------------
# __main__ test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from pipeline.preprocessing import preprocess

    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if img_path is None:
        exemplars_dir = Path(__file__).parent.parent / "data" / "exemplars"
        candidates = sorted(exemplars_dir.glob("*.jpg"))
        img_path = next((p for p in candidates if "simple" in p.name), candidates[0] if candidates else None)
    if img_path is None:
        print("No exemplar images found.")
        sys.exit(1)

    print(f"Processing: {img_path}")
    prep = preprocess(img_path)
    layout = detect_layout(prep["masked"])

    print(f"  Layout type   : {layout.layout_type}")
    print(f"  Columns       : {len(layout.columns)}")
    print(f"  Transition row: {layout.transition_row}")
    print(f"  Separator     : {layout.column_separator}")
    print(f"  Slab votes    : {layout.slab_votes}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    axes[0].imshow(prep["masked"], cmap="gray")
    axes[0].set_title("Masked binary")

    axes[1].imshow(prep["masked"], cmap="gray")
    axes[1].set_title(f"Layout: {layout.layout_type}")
    colours = ["red", "blue", "green", "orange"]
    for i, col in enumerate(layout.columns):
        rect = patches.Rectangle(
            (col.x_start, col.y_start),
            col.width,
            col.height,
            linewidth=2,
            edgecolor=colours[i % len(colours)],
            facecolor="none",
        )
        axes[1].add_patch(rect)
        axes[1].text(
            col.x_start + 5, col.y_start + 20,
            f"Col {i+1}", color=colours[i % len(colours)], fontsize=10,
        )
    if layout.transition_row:
        axes[1].axhline(layout.transition_row, color="yellow", linewidth=2, linestyle="--")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("/tmp/layout_test.png", dpi=100)
    print("Saved preview to /tmp/layout_test.png")
    plt.show()
