"""Plotting helpers for the manuscript pipeline stages.

Each function accepts the result dataclass of a specific stage and returns
a ``matplotlib.figure.Figure`` ready for display or saving.  Passing the
wrong type raises a clear ``TypeError``.

Usage
-----
    from pipeline.plots import plot_preprocess, plot_segment_kraken

    fig = plot_preprocess(pre)
    plt.show()                   # or fig.savefig("preprocess.png", ...)

    fig = plot_segment_kraken(seg, pre=pre)
    plt.show()
"""

from __future__ import annotations

from typing import Optional

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .stages import PreprocessResult, SegmentKrakenResult, PostprocessResult


# ---------------------------------------------------------------------------
# Standalone panel helpers
# ---------------------------------------------------------------------------

def plot_kraken_polygons(
    result: SegmentKrakenResult,
    pre: PreprocessResult,
    ax: Optional["plt.Axes"] = None,
    figsize: tuple = (8, 10),
) -> plt.Figure:
    """Draw Kraken line-boundary polygons overlaid on the BGR image.

    Can be used standalone (creates a new figure) or embedded inside a larger
    grid by passing an existing *ax*.

    Parameters
    ----------
    result : SegmentKrakenResult
    pre    : PreprocessResult  (provides the BGR image)
    ax     : existing Axes to draw into; if None a new figure is created.
    figsize : used only when *ax* is None.

    Returns
    -------
    matplotlib.figure.Figure  (the figure that owns *ax*)
    """
    if not isinstance(result, SegmentKrakenResult):
        raise TypeError(f"Expected SegmentKrakenResult, got {type(result).__name__}")
    if not isinstance(pre, PreprocessResult):
        raise TypeError(f"Expected PreprocessResult, got {type(pre).__name__}")

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    poly_img = _bgr2rgb(pre.bgr).copy()
    for boundary in result.line_boundaries:
        pts = np.array(boundary, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(poly_img, [pts], isClosed=True, color=(0, 180, 255), thickness=3)

    ax.imshow(poly_img)
    ax.set_title("Kraken line boundaries", fontsize=11)
    ax.set_xlabel(f"{result.n_lines} lines detected", fontsize=9)
    _off(ax)

    if own_fig:
        fig.tight_layout()
    return fig


def plot_combined_overlay(
    result: SegmentKrakenResult,
    pre: PreprocessResult,
    ax: Optional["plt.Axes"] = None,
    figsize: tuple = (8, 10),
) -> plt.Figure:
    """Draw a combined overlay: text region (blue tint) + figure ink (red tint).

    Can be used standalone (creates a new figure) or embedded inside a larger
    grid by passing an existing *ax*.

    Parameters
    ----------
    result : SegmentKrakenResult
    pre    : PreprocessResult  (provides the grayscale image)
    ax     : existing Axes to draw into; if None a new figure is created.
    figsize : used only when *ax* is None.

    Returns
    -------
    matplotlib.figure.Figure  (the figure that owns *ax*)
    """
    if not isinstance(result, SegmentKrakenResult):
        raise TypeError(f"Expected SegmentKrakenResult, got {type(result).__name__}")
    if not isinstance(pre, PreprocessResult):
        raise TypeError(f"Expected PreprocessResult, got {type(pre).__name__}")

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    combined = _gray2rgb(pre.gray).astype(np.float32)
    text_bool = result.text_mask > 0
    fig_bool  = result.figure_binary > 0
    # Blue tint for text region
    combined[text_bool, 2] = np.clip(combined[text_bool, 2] + 80, 0, 255)
    combined[text_bool, 0] = combined[text_bool, 0] * 0.5
    # Red tint for figure ink
    combined[fig_bool, 0] = np.clip(combined[fig_bool, 0] + 80, 0, 255)
    combined[fig_bool, 2] = combined[fig_bool, 2] * 0.5

    ax.imshow(combined.astype(np.uint8))
    legend_handles = [
        mpatches.Patch(facecolor=(0.1, 0.1, 0.9), label="text region"),
        mpatches.Patch(facecolor=(0.9, 0.1, 0.1), label="figure ink"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    ax.set_title("Combined overlay", fontsize=11)
    _off(ax)

    if own_fig:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bgr2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _gray2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def _off(ax):
    ax.axis("off")


# ---------------------------------------------------------------------------
# plot_preprocess
# ---------------------------------------------------------------------------

def plot_preprocess(
    result: PreprocessResult,
    figsize: tuple = (18, 7),
) -> plt.Figure:
    """Visualise the output of ``preprocess_page``.

    Three panels (left → right):

    1. **BGR** — deskewed + cropped colour image, annotated with image
       dimensions, skew angle, and binding side.
    2. **Binary (Sauvola)** — binarised + deskewed image, annotated with
       margin measurements.
    3. **Masked binary** — border/binding regions zeroed out (content area
       only), annotated with crop rect.

    Parameters
    ----------
    result : PreprocessResult
    figsize : (width, height) in inches

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not isinstance(result, PreprocessResult):
        raise TypeError(f"Expected PreprocessResult, got {type(result).__name__}")

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ── Panel 1: BGR ────────────────────────────────────────────────────────
    axes[0].imshow(_bgr2rgb(result.bgr))
    axes[0].set_title("BGR — deskewed & cropped", fontsize=12)
    axes[0].set_xlabel(
        f"{result.image_w} × {result.image_h} px  |  "
        f"skew: {result.deskew_angle:+.2f}°  |  binding: {result.binding_side}",
        fontsize=9,
    )

    # ── Panel 2: Sauvola binary ─────────────────────────────────────────────
    axes[1].imshow(result.binary_desk, cmap="gray")
    axes[1].set_title("Binary (Sauvola, deskewed)", fontsize=12)
    axes[1].set_xlabel(
        f"top margin: {result.top_margin} px  |  "
        f"bottom: {result.bottom_margin} px  |  "
        f"ruler: {result.ruler_height} px",
        fontsize=9,
    )

    # ── Panel 3: Masked binary ──────────────────────────────────────────────
    axes[2].imshow(result.masked, cmap="gray")
    axes[2].set_title("Masked binary (content only)", fontsize=12)
    cx, cy, cw, ch = result.crop_rect
    axes[2].set_xlabel(
        f"crop: x={cx}, y={cy}, w={cw}, h={ch}  |  "
        f"margin: {result.margin_width} px  |  binding: {result.binding_width} px",
        fontsize=9,
    )

    for ax in axes:
        _off(ax)

    fig.suptitle("Preprocess", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_segment_kraken
# ---------------------------------------------------------------------------

def plot_segment_kraken(
    result: SegmentKrakenResult,
    pre: Optional[PreprocessResult] = None,
    figsize: tuple = (20, 10),
) -> plt.Figure:
    """Visualise the output of ``segment_kraken``.

    Layout — 2 rows × 3 columns:

    Row 1 (image-independent):
      [0,0] Text binary (ink inside Kraken text-line polygons)
      [0,1] Figure binary (ink outside all text-line polygons)
      [0,2] Text-line mask (polygon fill from Kraken)

    Row 2 (requires *pre* for BGR; shows placeholder otherwise):
      [1,0] Kraken line-boundary polygons overlaid on the BGR image
      [1,1] Figure bounding boxes overlaid on the BGR image
      [1,2] Combined overlay: text region (blue tint) + figure ink (red tint)

    Parameters
    ----------
    result : SegmentKrakenResult
    pre : PreprocessResult, optional
        Provide for BGR-dependent panels.  If omitted those panels show a
        text placeholder.
    figsize : (width, height) in inches

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not isinstance(result, SegmentKrakenResult):
        raise TypeError(f"Expected SegmentKrakenResult, got {type(result).__name__}")

    has_bgr = pre is not None

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # ── Row 0 — image-independent masks ─────────────────────────────────────

    # [0,0] Text binary
    axes[0, 0].imshow(result.text_binary, cmap="gray")
    axes[0, 0].set_title("Text binary", fontsize=11)
    ink_ratio = result.text_px_kept / max(1, result.text_px_input)
    axes[0, 0].set_xlabel(
        f"lines: {result.n_lines}  |  polygon coverage: {result.text_coverage:.1%}  |  "
        f"ink retained: {ink_ratio:.1%}",
        fontsize=9,
    )

    # [0,1] Figure binary
    axes[0, 1].imshow(result.figure_binary, cmap="gray")
    axes[0, 1].set_title("Figure / illustration binary", fontsize=11)
    axes[0, 1].set_xlabel(
        f"figure components: {result.n_figures}  |  "
        f"figure ink coverage: {result.figure_coverage:.2%}",
        fontsize=9,
    )

    # [0,2] Text-line mask
    axes[0, 2].imshow(result.text_mask, cmap="gray")
    axes[0, 2].set_title("Text-line mask (Kraken polygons)", fontsize=11)
    axes[0, 2].set_xlabel("White = inside a detected text-line polygon", fontsize=9)

    # ── Row 1 — BGR overlays ─────────────────────────────────────────────────

    if has_bgr:
        # [1,0] Kraken line polygons
        plot_kraken_polygons(result, pre, ax=axes[1, 0])

        # [1,1] Figure bounding boxes
        bbox_img = bgr_rgb.copy()
        for x, y, w, h in result.figure_bboxes:
            cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (255, 60, 0), 2)
        axes[1, 1].imshow(bbox_img)
        axes[1, 1].set_title("Figure bounding boxes", fontsize=11)
        axes[1, 1].set_xlabel(
            f"{result.n_figures} figure regions  "
            f"(min area: {200} px²)",
            fontsize=9,
        )

        # [1,2] Combined overlay: text (blue) + figures (red)
        plot_combined_overlay(result, pre, ax=axes[1, 2])

    else:
        for col in range(3):
            axes[1, col].text(
                0.5, 0.5,
                "BGR not provided\n(pass pre=PreprocessResult)",
                ha="center", va="center",
                transform=axes[1, col].transAxes,
                fontsize=10, color="gray",
            )
            axes[1, col].set_title(
                ["Kraken line boundaries", "Figure bounding boxes", "Combined overlay"][col],
                fontsize=11,
            )

    for ax in axes.flat:
        _off(ax)

    fig.suptitle(
        f"Segment (Kraken)  —  {result.n_lines} lines  |  "
        f"{result.n_figures} figures  |  "
        f"figure coverage: {result.figure_coverage:.2%}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_postprocess
# ---------------------------------------------------------------------------

def plot_postprocess(
    result: "PostprocessResult",
    pre: PreprocessResult,
    seg: Optional[SegmentKrakenResult] = None,
    figsize: tuple = (18, 9),
) -> plt.Figure:
    """Visualise the output of ``postprocess``.

    Two panels (left → right):

    1. **Original BGR** — deskewed/cropped image with no annotations, for
       reference.
    2. **Annotated overlay** — same image with:
       - Semi-transparent yellow fill over figure ink pixels (from
         ``seg.figure_binary`` when *seg* is provided, otherwise bounding
         boxes are used as a fallback).
       - Red polygons for each surviving Kraken text-line boundary.
       - Green vertical gutter line when a double-column layout is detected.

    Parameters
    ----------
    result : PostprocessResult
    pre    : PreprocessResult  (provides ``bgr`` image)
    seg    : SegmentKrakenResult, optional
        Provide to use the pixel-accurate ``figure_binary`` mask for the
        yellow tint.  Falls back to bounding-box rectangles when omitted.
    figsize : (width, height) in inches

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not isinstance(result, PostprocessResult):
        raise TypeError(f"Expected PostprocessResult, got {type(result).__name__}")
    if not isinstance(pre, PreprocessResult):
        raise TypeError(f"Expected PreprocessResult, got {type(pre).__name__}")

    bgr_rgb = _bgr2rgb(pre.bgr)
    annotated = bgr_rgb.copy().astype(np.float32)

    # ── Figure ink: semi-transparent yellow tint ──────────────────────────
    if seg is not None and seg.figure_binary is not None and seg.figure_binary.any():
        fig_mask = seg.figure_binary > 0
        fig_overlay = annotated.copy()
        fig_overlay[fig_mask] = [220, 220, 0]   # RGB yellow
        annotated = (0.4 * fig_overlay + 0.6 * annotated).clip(0, 255)
    elif result.figure_bboxes:
        # Fallback: paint bounding-box rectangles
        fig_overlay = annotated.copy()
        for x, y, w, h in result.figure_bboxes:
            fig_overlay[y : y + h, x : x + w] = [220, 220, 0]
        annotated = (0.4 * fig_overlay + 0.6 * annotated).clip(0, 255)

    annotated = annotated.astype(np.uint8)

    # ── Kept text-line polygons (red) ─────────────────────────────────────
    for boundary in result.line_boundaries:
        pts = np.array(boundary, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [pts], isClosed=True, color=(220, 0, 0), thickness=2)

    # ── Gutter line (green) ───────────────────────────────────────────────
    if result.is_double_column and result.gutter_x is not None:
        cv2.line(
            annotated,
            (result.gutter_x, result.gutter_y_min),
            (result.gutter_x, result.gutter_y_max),
            (0, 200, 0),   # RGB green
            thickness=4,
        )

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(bgr_rgb)
    axes[0].set_title("BGR — deskewed & cropped", fontsize=12)
    axes[0].set_xlabel(
        f"{pre.image_w} × {pre.image_h} px  |  "
        f"skew: {pre.deskew_angle:+.2f}°  |  binding: {pre.binding_side}",
        fontsize=9,
    )

    axes[1].imshow(annotated)
    axes[1].set_title("Postprocess overlay", fontsize=12)

    # Build a compact annotation string
    layout = (
        f"double-column  gutter_x={result.gutter_x}"
        if result.is_double_column
        else "single-column"
    )
    axes[1].set_xlabel(
        f"lines: {result.n_lines}  "
        f"(−corner {result.n_lines_removed_corner}, −narrow {result.n_lines_removed_narrow})  |  "
        f"figures: {result.n_figures}  "
        f"(−corner {result.n_figures_removed_corner}, −narrow {result.n_figures_removed_narrow})  |  "
        f"{layout}",
        fontsize=9,
    )

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=(0.86, 0, 0), label="text-line boundary"),
        mpatches.Patch(facecolor=(0.86, 0.86, 0), alpha=0.7, label="figure region"),
    ]
    if result.is_double_column:
        legend_handles.append(
            mpatches.Patch(facecolor=(0, 0.78, 0), label="gutter")
        )
    axes[1].legend(handles=legend_handles, loc="lower right", fontsize=8)

    for ax in axes:
        _off(ax)

    fig.suptitle(
        f"Postprocess  —  {result.n_lines} lines  |  {result.n_figures} figures  |  {layout}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig

