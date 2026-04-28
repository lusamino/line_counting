"""Embeddings for anomaly detection and exploration.

Three complementary embedding types:

  structural   — normalised scalar features derived from the three pipeline
                 stages (PreprocessResult, SegmentKrakenResult,
                 PostprocessResult).  No heavy ML model required.

  vit_rgb      — ViT CLS-token (768-d) from the deskewed BGR/RGB page image.

  vit_mask     — ViT CLS-token (768-d) from a colourised text+figure overlay
                 (blue = text ink, red = figure ink) built from
                 SegmentKrakenResult.

Any combination of these three can be concatenated into a single
``combined_vec`` via the ``use_*`` flags of ``compute_embedding``.

Run standalone:

    python -m pipeline.embeddings             # processes all exemplars
    python -m pipeline.embeddings path/to/img # single image embedding
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Device selection (MPS → CPU)
# ---------------------------------------------------------------------------

def _get_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PageEmbedding:
    """Embedding and anomaly score for a single page.

    Attributes
    ----------
    filename       : source image filename
    structural_vec : scalar feature vector from all pipeline stages;
                     empty array if ``use_structural=False``
    vit_rgb_vec    : ViT CLS token from the RGB page image;
                     empty array if ``use_vit_rgb=False``
    vit_mask_vec   : ViT CLS token from the text+figure overlay image;
                     empty array if ``use_vit_mask=False``
    combined_vec   : concatenation of whichever sub-vectors were requested
    anomaly_score  : set by ``anomaly_scores()`` (higher = more anomalous)
    umap_xy        : 2-D coordinates set by ``compute_umap()``
    """
    filename: str
    structural_vec: np.ndarray   # shape (N_STRUCTURAL,) or (0,)
    vit_rgb_vec: np.ndarray      # shape (768,) or (0,)
    vit_mask_vec: np.ndarray     # shape (768,) or (0,)
    combined_vec: np.ndarray     # concatenation of requested sub-vectors
    anomaly_score: float = 0.0
    umap_xy: Optional[Tuple[float, float]] = None


# ---------------------------------------------------------------------------
# Structural feature extraction
# ---------------------------------------------------------------------------

def compute_structural_features(
    pre,   # PreprocessResult
    seg,   # SegmentKrakenResult
    post,  # PostprocessResult
) -> np.ndarray:
    """Build a normalised structural feature vector from the three pipeline stages.

    Features (24 total)
    -------------------
    From PreprocessResult
    ~~~~~~~~~~~~~~~~~~~~~
    [0]  binding_side = left  (one-hot)
    [1]  binding_side = right (one-hot)
    [2]  deskew_angle / 5.0   (normalised; typical max ±5°)
    [3]  margin_width / image_w
    [4]  top_margin    / image_h
    [5]  ruler_height  / image_h
    [6]  bottom_margin / image_h
    [7]  binding_width / image_w
    [8]  image_h / image_w     (page aspect ratio)

    From SegmentKrakenResult
    ~~~~~~~~~~~~~~~~~~~~~~~~
    [9]  n_lines / image_h             (lines per pixel of page height)
    [10] text_coverage
    [11] text_px_kept / text_px_input  (ink retention after masking)
    [12] n_figures / (image_h * image_w / 1e5)  (figures per 100k px)
    [13] figure_coverage
    [14] figure mean bbox area / (image_h * image_w)  (0 if no figures)
    [15] figure std  bbox area / (image_h * image_w)  (0 if ≤1 figure)

    From PostprocessResult (after corner + narrow filtering + gutter detection)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    [16] n_lines_post / image_h
    [17] n_lines_removed_corner / max(1, seg.n_lines)
    [18] n_lines_removed_narrow / max(1, seg.n_lines)
    [19] n_figures_post / (image_h * image_w / 1e5)
    [20] n_figures_removed_corner / max(1, seg.n_figures)
    [21] n_figures_removed_narrow / max(1, seg.n_figures)
    [22] is_double_column  (0.0 or 1.0)
    [23] gutter_x / image_w  (0.0 if not detected)
    """
    H  = max(1, pre.image_h)
    W  = max(1, pre.image_w)
    HW = max(1, H * W)

    # ── Preprocess features ───────────────────────────────────────────────
    is_left  = float(pre.binding_side == "left")
    is_right = float(pre.binding_side == "right")
    angle    = pre.deskew_angle / 5.0  # normalise to approx [-1, 1]

    pre_feats = [
        is_left, is_right, angle,
        pre.margin_width  / W,
        pre.top_margin    / H,
        pre.ruler_height  / H,
        pre.bottom_margin / H,
        pre.binding_width / W,
        H / W,
    ]

    # ── Segment features ──────────────────────────────────────────────────
    ink_retention = seg.text_px_kept / max(1, seg.text_px_input)

    fig_areas = [w * h for (_, _, w, h) in seg.figure_bboxes]
    fig_mean  = float(np.mean(fig_areas)) / HW if fig_areas else 0.0
    fig_std   = float(np.std(fig_areas))  / HW if len(fig_areas) > 1 else 0.0
    fig_norm  = seg.n_figures / max(1, HW / 1e5)

    seg_feats = [
        seg.n_lines    / H,
        seg.text_coverage,
        ink_retention,
        fig_norm,
        seg.figure_coverage,
        fig_mean,
        fig_std,
    ]

    # ── Postprocess features ──────────────────────────────────────────────
    post_feats = [
        post.n_lines   / H,
        post.n_lines_removed_corner / max(1, seg.n_lines),
        post.n_lines_removed_narrow / max(1, seg.n_lines),
        post.n_figures / max(1, HW / 1e5),
        post.n_figures_removed_corner / max(1, seg.n_figures),
        post.n_figures_removed_narrow / max(1, seg.n_figures),
        float(post.is_double_column),
        post.gutter_x / W if post.gutter_x is not None else 0.0,
    ]

    return np.array(pre_feats + seg_feats + post_feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Text+figure overlay image (used for vit_mask embedding)
# ---------------------------------------------------------------------------

def _build_mask_overlay(pre, seg) -> np.ndarray:
    """Build the text (blue) + figure (red) colourised overlay as RGB uint8.

    Replicates the same colouring used by ``plot_combined_overlay`` but
    returns a numpy array directly (no matplotlib).
    """
    # Grayscale → 3-channel float
    overlay = np.stack([pre.gray, pre.gray, pre.gray], axis=-1).astype(np.float32)

    text_mask = seg.text_mask > 0
    fig_mask  = seg.figure_binary > 0

    # Blue tint on text ink  (boost B channel, suppress R)
    overlay[text_mask, 2] = np.clip(overlay[text_mask, 2] + 80, 0, 255)
    overlay[text_mask, 0] = overlay[text_mask, 0] * 0.5

    # Red tint on figure ink (boost R channel, suppress B)
    overlay[fig_mask, 0] = np.clip(overlay[fig_mask, 0] + 80, 0, 255)
    overlay[fig_mask, 2] = overlay[fig_mask, 2] * 0.5

    return overlay.astype(np.uint8)


# ---------------------------------------------------------------------------
# Visual embedding (ViT CLS token)
# ---------------------------------------------------------------------------

def compute_visual_embedding(image: np.ndarray, bgr: bool = False) -> np.ndarray:
    """Extract a ViT-B/16 CLS-token embedding from any uint8 page image.

    Uses ``vit_base_patch16_224`` from timm.  Runs on MPS if available,
    otherwise CPU.

    Parameters
    ----------
    image : np.ndarray  (uint8)
        Any of:
        * Grayscale or binary ``(H, W)`` — converted to 3-channel.
        * RGB ``(H, W, 3)``
        * BGR ``(H, W, 3)`` — pass ``bgr=True`` to convert to RGB.
    bgr : bool
        Set ``True`` when *image* is in OpenCV BGR channel order.

    Returns
    -------
    np.ndarray  shape (768,) float32
    """
    import cv2
    import torch
    import timm
    from PIL import Image
    from torchvision import transforms

    device = _get_device()

    # ── Normalise to RGB uint8 ────────────────────────────────────────────
    if image.ndim == 2:
        # Grayscale or binary → 3-channel
        rgb = np.stack([image, image, image], axis=-1)
    elif bgr:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = image  # already RGB

    pil_img = Image.fromarray(rgb.astype(np.uint8))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    tensor = transform(pil_img).unsqueeze(0).to(device)

    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    model = model.to(device).eval()

    with torch.no_grad():
        embedding = model(tensor)  # (1, 768)

    return embedding.squeeze(0).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Combined embedding
# ---------------------------------------------------------------------------

def compute_embedding(
    filename: str,
    pre,                         # PreprocessResult
    seg,                         # SegmentKrakenResult
    post,                        # PostprocessResult
    use_structural: bool = True,
    use_vit_rgb: bool = True,
    use_vit_mask: bool = False,
) -> PageEmbedding:
    """Compute a compound embedding for one page.

    Parameters
    ----------
    filename : str
        Source image filename (used for identification only).
    pre : PreprocessResult
    seg : SegmentKrakenResult
    post : PostprocessResult
    use_structural : bool
        Include the 22-d normalised structural feature vector.
    use_vit_rgb : bool
        Include a 768-d ViT embedding of the deskewed BGR page image.
    use_vit_mask : bool
        Include a 768-d ViT embedding of the text (blue) + figure (red)
        colourised overlay image.

    Returns
    -------
    PageEmbedding
        ``combined_vec`` is the concatenation of whichever sub-vectors
        were requested (in the order: structural, vit_rgb, vit_mask).

    Raises
    ------
    ValueError
        If all three ``use_*`` flags are False.
    """
    if not (use_structural or use_vit_rgb or use_vit_mask):
        raise ValueError("At least one of use_structural / use_vit_rgb / use_vit_mask must be True.")

    _empty = np.empty(0, dtype=np.float32)

    struct_vec   = compute_structural_features(pre, seg, post) if use_structural else _empty
    vit_rgb_vec  = compute_visual_embedding(pre.bgr, bgr=True) if use_vit_rgb   else _empty
    vit_mask_vec = compute_visual_embedding(_build_mask_overlay(pre, seg))      if use_vit_mask  else _empty

    parts = [v for v in (struct_vec, vit_rgb_vec, vit_mask_vec) if v.size > 0]
    combined = np.concatenate(parts)

    return PageEmbedding(
        filename=filename,
        structural_vec=struct_vec,
        vit_rgb_vec=vit_rgb_vec,
        vit_mask_vec=vit_mask_vec,
        combined_vec=combined,
    )


# ---------------------------------------------------------------------------
# Anomaly scoring
# ---------------------------------------------------------------------------

def anomaly_scores(embeddings: List[PageEmbedding]) -> List[PageEmbedding]:
    """Fit an IsolationForest on the combined embeddings and assign scores.

    Scores are normalised to [0, 1] where 1 = most anomalous.
    Modifies the PageEmbedding objects in place and returns them.
    """
    from sklearn.ensemble import IsolationForest

    if len(embeddings) < 2:
        for e in embeddings:
            e.anomaly_score = 0.0
        return embeddings

    X = np.stack([e.combined_vec for e in embeddings])
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X)
    raw_scores = clf.score_samples(X)  # higher = more normal
    # Invert and normalise to [0, 1]
    inv = -raw_scores
    score_min, score_max = inv.min(), inv.max()
    if score_max > score_min:
        normalised = (inv - score_min) / (score_max - score_min)
    else:
        normalised = np.zeros_like(inv)

    for emb, score in zip(embeddings, normalised):
        emb.anomaly_score = float(score)

    return embeddings


# ---------------------------------------------------------------------------
# UMAP dimensionality reduction
# ---------------------------------------------------------------------------

def compute_umap(
    embeddings: List[PageEmbedding],
    n_neighbors: int = 10,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> List[PageEmbedding]:
    """Reduce embeddings to 2D with UMAP.

    Stores the result in each PageEmbedding.umap_xy.
    Requires at least 4 samples.
    """
    if len(embeddings) < 4:
        for e in embeddings:
            e.umap_xy = (0.0, 0.0)
        return embeddings

    import umap

    X = np.stack([e.combined_vec for e in embeddings])
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, len(embeddings) - 1),
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(X)
    for emb, (x, y) in zip(embeddings, coords):
        emb.umap_xy = (float(x), float(y))
    return embeddings


# ---------------------------------------------------------------------------
# __main__ test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pipeline.stages import preprocess_page, segment_kraken, postprocess

    _MODEL = (
        "/Users/luissalamanca/Library/Application Support/htrmopo/"
        "97665cf3-f83d-5594-8855-f28d3af9df7a/blla.mlmodel"
    )

    exemplars_dir = Path(__file__).parent.parent / "data" / "exemplars"
    if len(sys.argv) > 1:
        images = [Path(sys.argv[1])]
    else:
        images = sorted(exemplars_dir.glob("*.jpg"))[:5]

    page_embeddings: List[PageEmbedding] = []

    for img_path in images:
        print(f"Processing {img_path.name} ...")
        try:
            pre  = preprocess_page(img_path)
            seg  = segment_kraken(pre, model_path=_MODEL, device="mps")
            post = postprocess(pre, seg)

            emb = compute_embedding(
                filename=img_path.name,
                pre=pre,
                seg=seg,
                post=post,
                use_structural=True,
                use_vit_rgb=True,
                use_vit_mask=False,
            )
            page_embeddings.append(emb)
            print(f"  structural dim : {emb.structural_vec.size}")
            print(f"  vit_rgb dim    : {emb.vit_rgb_vec.size}")
            print(f"  combined dim   : {emb.combined_vec.size}")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if page_embeddings:
        page_embeddings = anomaly_scores(page_embeddings)
        if len(page_embeddings) >= 4:
            page_embeddings = compute_umap(page_embeddings)

        print("\n--- Anomaly scores ---")
        for e in sorted(page_embeddings, key=lambda x: x.anomaly_score, reverse=True):
            umap_str = f"  UMAP={e.umap_xy}" if e.umap_xy else ""
            print(f"  {e.filename}: {e.anomaly_score:.3f}{umap_str}")
