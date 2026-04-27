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

from .stages import PreprocessResult, SegmentKrakenResult


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
        cv2.polylines(poly_img, [pts], isClosed=True, color=(0, 180, 255), thickness=1)

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
# plot_postprocess  (placeholder — to be added once postprocess is populated)
# ---------------------------------------------------------------------------

def plot_postprocess(*args, **kwargs):
    """Placeholder — will visualise postprocess output once that stage is built."""
    raise NotImplementedError(
        "plot_postprocess is not yet implemented; "
        "populate postprocess() first."
    )
