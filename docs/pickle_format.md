# Pickle file format

Each image processed by the pipeline produces a `results/{stem}.pkl` file — a plain Python `dict` serialised with `pickle`.

```
{
  "filename"  : str                  # e.g. "page_001.jpg"
  "path"      : pathlib.Path         # absolute path to the source image
  "pre"       : PreprocessResult     # Stage 1–2 output
  "seg"       : SegmentKrakenResult  # Stage 3 (Kraken) output
  "post"      : PostprocessResult    # Stage 4 (postprocessing) output
  "embedding" : PageEmbedding | None # optional; None if --no-embed was passed
}
```

---

## `PreprocessResult` — binarisation, deskew, border masking

Returned by `pipeline.stages.preprocess_page`.

| Field | Type | Description |
|---|---|---|
| `gray` | `uint8 ndarray` | Deskewed grayscale image |
| `bgr` | `uint8 ndarray` | Deskewed colour image (BGR) |
| `binary` | `uint8 ndarray` | Sauvola binarisation (0/255) |
| `binary_desk` | `uint8 ndarray` | Binarisation after deskew |
| `masked` | `uint8 ndarray` | `binary_desk` with borders/binding zeroed |
| `image_h`, `image_w` | `int` | Pixel dimensions after crop |
| `deskew_angle` | `float` | Rotation applied (degrees, CCW positive) |
| `binding_side` | `str` | `"left"` or `"right"` |
| `margin_width` | `int` | Columns masked from the binding edge (dark shadow) |
| `top_margin` | `int` | Rows masked at the top (solid black border) |
| `ruler_height` | `int` | Rows masked for the scale ruler band |
| `bottom_margin` | `int` | Rows masked at the bottom |
| `binding_width` | `int` | Columns masked for the physical fold |
| `crop_rect` | `(x, y, w, h)` | Content bounding box in the deskewed image |

---

## `SegmentKrakenResult` — Kraken neural segmentation + figure extraction

Returned by `pipeline.stages.segment_kraken`.

Ink pixels on the page are divided into three mutually exclusive categories:

- **Background** — light parchment/vellum (`masked == 0`)
- **Text** — ink pixels inside a Kraken text-line polygon
- **Figures** — ink pixels *outside* all text-line polygons (illustrations, decorated initials, ruling marks, etc.)

| Field | Type | Description |
|---|---|---|
| `text_binary` | `uint8 ndarray` | Ink pixels inside Kraken text-line polygons |
| `text_mask` | `uint8 ndarray` | Polygon fill of all detected lines |
| `non_text_mask` | `bool ndarray` | `True` = outside all text-line polygons |
| `n_lines` | `int` | Number of detected text lines |
| `text_coverage` | `float` | Fraction of image area covered by line polygons |
| `text_px_kept` | `int` | Foreground pixels inside text polygons |
| `text_px_input` | `int` | Total foreground pixels before masking |
| `line_boundaries` | `list[list[[x,y]]]` | Per-line polygon vertex list |
| `figure_binary` | `uint8 ndarray` | Ink pixels outside all text polygons |
| `n_figures` | `int` | Connected components in `figure_binary` with area ≥ 200 px² |
| `figure_coverage` | `float` | `figure_binary` foreground pixels / total image pixels |
| `figure_bboxes` | `list[(x, y, w, h)]` | Bounding box per significant figure component |

---

## `PostprocessResult` — corner/narrow filtering + gutter detection

Returned by `pipeline.stages.postprocess`.

Three sequential steps are applied to the Kraken output:

1. **Corner filter** — removes detections whose centroid falls inside a corner zone adjacent to the binding spine.
2. **Narrow filter** — removes detections whose bounding box is thinner than a minimum pixel threshold.
3. **Gutter detection** — finds a central column-separator gap in double-column pages; line polygons crossing the gutter are split.

| Field | Type | Description |
|---|---|---|
| `line_boundaries` | `list[list[[x,y]]]` | Surviving line polygons after all filters |
| `n_lines` | `int` | Surviving line count |
| `figure_bboxes` | `list[(x, y, w, h)]` | Surviving figure bounding boxes |
| `n_figures` | `int` | Surviving figure count |
| `n_lines_removed_corner` | `int` | Lines dropped by the corner filter |
| `n_lines_removed_narrow` | `int` | Lines dropped by the narrow filter |
| `n_figures_removed_corner` | `int` | Figures dropped by the corner filter |
| `n_figures_removed_narrow` | `int` | Figures dropped by the narrow filter |
| `is_double_column` | `bool` | Whether a double-column layout was detected |
| `gutter_x` | `int \| None` | X-position of the column separator, or `None` |
| `gutter_y_min`, `gutter_y_max` | `int \| None` | Vertical span of the detected gutter |
| `n_lines_split` | `int` | Number of polygons split at the gutter |

---

## `PageEmbedding` — feature vectors for anomaly detection / UMAP

Returned by `pipeline.embeddings.compute_embedding`. Present in the pickle only when the pipeline is run with embedding computation enabled.

| Field | Type | Description |
|---|---|---|
| `filename` | `str` | Source image filename |
| `structural_vec` | `float32 ndarray (24,)` | Normalised scalar features derived from all three pipeline stages (see feature index table below) |
| `vit_rgb_vec` | `float32 ndarray (768,)` | ViT CLS token from the deskewed colour image; empty array `(0,)` if not requested |
| `vit_mask_vec` | `float32 ndarray (768,)` | ViT CLS token from a blue/red text+figure overlay; empty array `(0,)` if not requested |
| `combined_vec` | `float32 ndarray` | Concatenation of whichever sub-vectors were requested |
| `anomaly_score` | `float` | IsolationForest score normalised to [0, 1]; higher = more anomalous |
| `umap_xy` | `(float, float) \| None` | 2-D UMAP projection coordinates |
| `umap_xyz` | `(float, float, float) \| None` | 3-D UMAP projection coordinates |

### Structural feature index (24 dimensions)

| Index | Feature | Source |
|---|---|---|
| 0 | `binding_side == left` (one-hot) | `PreprocessResult` |
| 1 | `binding_side == right` (one-hot) | `PreprocessResult` |
| 2 | `deskew_angle / 5.0` | `PreprocessResult` |
| 3 | `margin_width / image_w` | `PreprocessResult` |
| 4 | `top_margin / image_h` | `PreprocessResult` |
| 5 | `ruler_height / image_h` | `PreprocessResult` |
| 6 | `bottom_margin / image_h` | `PreprocessResult` |
| 7 | `binding_width / image_w` | `PreprocessResult` |
| 8 | `image_h / image_w` (aspect ratio) | `PreprocessResult` |
| 9 | `n_lines / image_h` | `SegmentKrakenResult` |
| 10 | `text_coverage` | `SegmentKrakenResult` |
| 11 | `text_px_kept / text_px_input` (ink retention) | `SegmentKrakenResult` |
| 12 | `n_figures / (image_h × image_w / 1e5)` | `SegmentKrakenResult` |
| 13 | `figure_coverage` | `SegmentKrakenResult` |
| 14 | mean figure bbox area / image area | `SegmentKrakenResult` |
| 15 | std figure bbox area / image area | `SegmentKrakenResult` |
| 16 | `n_lines_post / image_h` | `PostprocessResult` |
| 17 | `n_lines_removed_corner / n_lines_seg` | `PostprocessResult` |
| 18 | `n_lines_removed_narrow / n_lines_seg` | `PostprocessResult` |
| 19 | `n_figures_post / (image_h × image_w / 1e5)` | `PostprocessResult` |
| 20 | `n_figures_removed_corner / n_figures_seg` | `PostprocessResult` |
| 21 | `n_figures_removed_narrow / n_figures_seg` | `PostprocessResult` |
| 22 | `is_double_column` (0.0 or 1.0) | `PostprocessResult` |
| 23 | `gutter_x / image_w` (0.0 if single column) | `PostprocessResult` |
