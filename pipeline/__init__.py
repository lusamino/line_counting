"""Medieval manuscript line-counting pipeline."""

from .preprocessing import binarise, deskew, preprocess
from .layout import detect_layout, LayoutResult
from .masking import mask_non_text, mask_non_text_fullrgb, mask_non_text_kraken
from .segmentation import segment_lines, LineResult
from .embeddings import (
    PageEmbedding,
    compute_structural_features,
    compute_visual_embedding,
    compute_embedding,
    anomaly_scores,
    compute_umap,
)
from .stages import (
    PreprocessResult, preprocess_page,
    SegmentKrakenResult, segment_kraken,
    PostprocessResult, postprocess,
)
from .postprocessing import remove_corner_bboxes, remove_narrow_bboxes
from .plots import plot_preprocess, plot_segment_kraken, plot_postprocess, plot_kraken_polygons, plot_combined_overlay

__all__ = [
    "binarise", "deskew", "preprocess",
    "detect_layout", "LayoutResult",
    "mask_non_text",
    "mask_non_text_fullrgb",
    "mask_non_text_kraken",
    "segment_lines", "LineResult",
    "PageEmbedding",
    "compute_structural_features", "compute_visual_embedding",
    "compute_embedding", "anomaly_scores", "compute_umap",
    "PreprocessResult", "preprocess_page",
    "SegmentKrakenResult", "segment_kraken",
    "PostprocessResult", "postprocess",
    "remove_corner_bboxes", "remove_narrow_bboxes",
    "plot_preprocess", "plot_segment_kraken", "plot_postprocess",
    "plot_kraken_polygons", "plot_combined_overlay",
]
