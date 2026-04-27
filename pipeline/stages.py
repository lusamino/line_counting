"""Manuscript image processing pipeline.

preprocess_page  — binarise, deskew, mask borders/binding, crop to content.
segment_kraken   — Kraken neural baseline segmentation + figure extraction.
postprocess      — placeholder; will hold layout/line extraction in future.

Each stage returns a dedicated dataclass that bundles processed images with
structured features for downstream analysis and plotting.

Usage
-----
    from pipeline.stages import preprocess_page, segment_kraken, postprocess

    pre  = preprocess_page("data/exemplars/my_page.jpg")
    seg  = segment_kraken(pre, device="mps")
    post = postprocess(pre, seg)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .preprocessing import preprocess
from .masking import mask_non_text_kraken
from .postprocessing import remove_corner_bboxes, remove_narrow_bboxes


# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------

@dataclass
class PreprocessResult:
    """Output of ``preprocess_page``.

    Images
    ------
    gray        : deskewed grayscale (uint8)
    bgr         : deskewed colour image (uint8, BGR)
    binary      : Sauvola binarisation of the original (uint8, 0/255)
    binary_desk : Sauvola binary after deskewing (uint8, 0/255)
    masked      : binary_desk with page borders and binding zeroed (uint8, 0/255)

    Structured features
    -------------------
    image_h, image_w     : pixel dimensions of the deskewed image
    deskew_angle         : rotation applied in degrees (CCW positive)
    binding_side         : 'left' or 'right'
    margin_width         : pixels masked from the binding edge
    top_margin           : pixels masked at the top (solid black border)
    ruler_height         : pixels masked for scale-bar / ruler
    bottom_margin        : pixels masked at the bottom
    binding_width        : pixels masked for the physical binding shadow
    """

    # Images
    gray: np.ndarray
    bgr: np.ndarray
    binary: np.ndarray
    binary_desk: np.ndarray
    masked: np.ndarray

    # Structured features
    image_h: int
    image_w: int
    deskew_angle: float
    binding_side: str
    margin_width: int
    top_margin: int
    ruler_height: int
    bottom_margin: int
    binding_width: int
    crop_rect: Tuple[int, int, int, int]   # (x, y, w, h) of content in the deskewed image


def preprocess_page(
    image_path,
    sauvola_window: int = 51,
    sauvola_k: float = 0.2,
    deskew_range: float = 5.0,
) -> PreprocessResult:
    """Run preprocessing: binarise, deskew, mask borders, crop to content.

    Parameters
    ----------
    image_path : str or Path
        Path to the input manuscript image (JPEG, PNG, …).
    sauvola_window : int
        Local window size for Sauvola binarisation.
    sauvola_k : float
        Sauvola sensitivity parameter.
    deskew_range : float
        Search range (±degrees) for deskewing.

    Returns
    -------
    PreprocessResult
    """
    prep = preprocess(
        str(image_path),
        sauvola_window=sauvola_window,
        sauvola_k=sauvola_k,
        deskew_range=deskew_range,
    )
    bi = prep["border_info"]
    h, w = prep["gray"].shape

    return PreprocessResult(
        gray=prep["gray"],
        bgr=prep["bgr"],
        binary=prep["binary"],
        binary_desk=prep["binary_desk"],
        masked=prep["masked"],
        image_h=h,
        image_w=w,
        deskew_angle=prep["deskew_angle"],
        binding_side=bi.get("binding_side", ""),
        margin_width=bi.get("margin_width", 0),
        top_margin=bi.get("top_margin", 0),
        ruler_height=bi.get("ruler_height", 0),
        bottom_margin=bi.get("bottom_margin", 0),
        binding_width=bi.get("binding_width", 0),
        crop_rect=bi.get("crop_rect", (0, 0, w, h)),
    )


# ---------------------------------------------------------------------------
# Segment (Kraken) + Figure extraction
# ---------------------------------------------------------------------------

_FIGURE_MIN_AREA_PX: int = 200  # minimum connected-component area to report as a figure


@dataclass
class SegmentKrakenResult:
    """Output of ``segment_kraken``.

    Text masks & features
    ---------------------
    text_binary   : ink pixels inside Kraken text-line polygons (uint8, 0/255)
    text_mask     : polygon fill of all detected text lines (uint8, 0/255)
    non_text_mask : bool array — True = outside all text-line polygons
    n_lines       : number of text lines detected
    text_coverage : fraction of image area covered by text-line polygons
    text_px_kept  : foreground pixels remaining in *text_binary*
    text_px_input : foreground pixels in Stage-1 masked binary (before masking)
    line_boundaries : per-line polygon as list of [[x, y], …] lists

    Figure / illustration features
    -------------------------------
    The page is decomposed into three pixel categories:

    * **Background** — pixels that are white/light in the Stage-1 masked binary
      (i.e. parchment/vellum, ``masked == 0``).
    * **Text** — ink pixels (``masked == 255``) inside a Kraken text-line polygon.
    * **Figures** — ink pixels (``masked == 255``) that lie *outside* all
      Kraken text-line polygons; illustrations, decorated initials, ruling
      lines, etc.

    figure_binary   : ink pixels NOT inside any text-line polygon (uint8, 0/255)
    n_figures       : connected-component count in *figure_binary* with area
                      >= _FIGURE_MIN_AREA_PX pixels
    figure_coverage : figure_binary foreground pixels / total image pixels
    figure_bboxes   : (x, y, w, h) per significant figure component
    """

    # Text masks
    text_binary: np.ndarray
    text_mask: np.ndarray
    non_text_mask: np.ndarray

    # Text features
    n_lines: int
    text_coverage: float
    text_px_kept: int
    text_px_input: int
    line_boundaries: List[List]  # [[x,y], …] per detected line

    # Figure / illustration features
    figure_binary: np.ndarray
    n_figures: int
    figure_coverage: float
    figure_bboxes: List[Tuple[int, int, int, int]]


def segment_kraken(
    pre: PreprocessResult,
    model_path: Optional[str] = None,
    device: str = "cpu",
    dilation_px: int = 8,
) -> SegmentKrakenResult:
    """Run Kraken segmentation and extract figure / illustration regions.

    Parameters
    ----------
    pre : PreprocessResult
        Output from ``preprocess_page``.
    model_path : str, optional
        Path to a Kraken ``.mlmodel`` segmentation file.
        Pass ``None`` to use Kraken's built-in default model.
    device : str
        PyTorch device string, e.g. ``"cpu"``, ``"mps"``, ``"cuda"``.
    dilation_px : int
        Pixels to dilate each text-line polygon outward when building the
        text mask, providing a small safety margin around detected lines.

    Returns
    -------
    SegmentKrakenResult
    """
    masking_res = mask_non_text_kraken(
        pre.bgr,
        pre.masked,
        model_path=model_path,
        device=device,
        dilation_px=dilation_px,
    )

    line_boundaries = [
        c["boundary"]
        for c in masking_res.removed_components
        if c.get("type") == "text_line"
    ]

    text_mask = (~masking_res.illustration_mask).astype(np.uint8) * 255
    text_px_input = int((pre.masked > 0).sum())
    text_px_kept = int((masking_res.text_binary > 0).sum())

    # ── Figure extraction ─────────────────────────────────────────────────────
    # non_text_region: True for every pixel outside all Kraken text-line polygons.
    # Any foreground pixel there (ink in Stage-1 masked binary) is not text:
    # illustrations, decorated initials, ruling marks, etc.
    non_text_region: np.ndarray = masking_res.illustration_mask  # bool

    figure_binary = np.zeros_like(pre.masked)
    figure_binary[non_text_region & (pre.masked > 0)] = 255

    # Connected-component analysis to enumerate figure regions
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        figure_binary, connectivity=8
    )
    figure_bboxes: List[Tuple[int, int, int, int]] = []
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= _FIGURE_MIN_AREA_PX:
            figure_bboxes.append((
                int(stats[lbl, cv2.CC_STAT_LEFT]),
                int(stats[lbl, cv2.CC_STAT_TOP]),
                int(stats[lbl, cv2.CC_STAT_WIDTH]),
                int(stats[lbl, cv2.CC_STAT_HEIGHT]),
            ))

    total_px = max(1, pre.image_h * pre.image_w)
    figure_coverage = float((figure_binary > 0).sum()) / total_px

    return SegmentKrakenResult(
        text_binary=masking_res.text_binary,
        text_mask=text_mask,
        non_text_mask=masking_res.illustration_mask,
        n_lines=len(line_boundaries),
        text_coverage=masking_res.text_coverage,
        text_px_kept=text_px_kept,
        text_px_input=text_px_input,
        line_boundaries=line_boundaries,
        figure_binary=figure_binary,
        n_figures=len(figure_bboxes),
        figure_coverage=figure_coverage,
        figure_bboxes=figure_bboxes,
    )


# ---------------------------------------------------------------------------
# Postprocess  (placeholder — to be populated in future stages)
# ---------------------------------------------------------------------------

@dataclass
class PostprocessResult:
    """Output of ``postprocess``.

    Holds cleaned segmentation data after removing artefactual bounding boxes.

    Filtered data
    -------------
    line_boundaries  : line polygons surviving both filters
    n_lines          : number of surviving lines
    figure_bboxes    : figure boxes surviving both filters
    n_figures        : number of surviving figures

    Removal statistics
    ------------------
    n_lines_removed_corner   : lines dropped by corner filter
    n_lines_removed_narrow   : lines dropped by narrow filter
    n_figures_removed_corner : figures dropped by corner filter
    n_figures_removed_narrow : figures dropped by narrow filter
    """

    # Filtered segmentation data
    line_boundaries: List[List]
    n_lines: int
    figure_bboxes: List[Tuple[int, int, int, int]]
    n_figures: int

    # Removal stats
    n_lines_removed_corner: int
    n_lines_removed_narrow: int
    n_figures_removed_corner: int
    n_figures_removed_narrow: int


def postprocess(
    pre: PreprocessResult,
    seg: SegmentKrakenResult,
    corner_fraction: float = 0.10,
    min_dimension_px: int = 20,
) -> PostprocessResult:
    """Clean up segmentation results by removing artefactual bounding boxes.

    Two sequential filters are applied:

    1. **Corner filter** — removes detections whose centroid falls inside one
       of the two corners adjacent to the binding spine (top and bottom).
       The corner zone is a square of ``corner_fraction * image_width`` ×
       ``corner_fraction * image_height``.

    2. **Narrow filter** — removes detections whose bounding box has a shorter
       side below ``min_dimension_px`` pixels (ruling lines, scan edges, etc.).

    Parameters
    ----------
    pre : PreprocessResult
    seg : SegmentKrakenResult
    corner_fraction : float
        Size of each corner zone as a fraction of image dimensions (default 0.10).
    min_dimension_px : int
        Minimum short-side size in pixels for a bbox to be kept (default 20).

    Returns
    -------
    PostprocessResult
    """
    # 1. Corner filter
    lines, figures, nl_c, nf_c = remove_corner_bboxes(
        seg.line_boundaries,
        seg.figure_bboxes,
        image_h=pre.image_h,
        image_w=pre.image_w,
        binding_side=pre.binding_side,
        corner_fraction=corner_fraction,
    )

    # 2. Narrow filter
    lines, figures, nl_n, nf_n = remove_narrow_bboxes(
        lines,
        figures,
        min_dimension_px=min_dimension_px,
    )

    return PostprocessResult(
        line_boundaries=lines,
        n_lines=len(lines),
        figure_bboxes=figures,
        n_figures=len(figures),
        n_lines_removed_corner=nl_c,
        n_lines_removed_narrow=nl_n,
        n_figures_removed_corner=nf_c,
        n_figures_removed_narrow=nf_n,
    )
