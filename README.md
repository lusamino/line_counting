# Medieval Manuscript Line Counter

A Python pipeline to count text lines in scanned pages of medieval manuscripts using Kraken neural baseline segmentation, with Streamlit tools for parameter tuning, manual validation, and embedding exploration.

---

## Project structure

```
line_counting/
├── pipeline/
│   ├── __init__.py
│   ├── preprocessing.py   # Stages 1–2: binarisation, deskew, border/binding masking
│   ├── masking.py         # Stage 3:   Kraken non-text masking (figures, illustrations)
│   ├── postprocessing.py  # Stage 4:   corner/narrow filter, gutter detection & line splitting
│   ├── segmentation.py    # (legacy helpers)
│   ├── plots.py           # Shared visualisation utilities
│   └── embeddings.py      # Structural + ViT embeddings, IsolationForest anomaly scoring, UMAP
├── 0_TuningPreprocess.py  # Streamlit: preprocessing parameter calibration
├── 0_TuningPostprocess.py # Streamlit: postprocessing parameter tuning
├── 1_ValidateCounting.py  # Streamlit: human line-count validation
├── 2_VisualizeEmbeddings.py # Streamlit: 3-D UMAP embedding explorer (WIP)
├── run_pipeline.py        # CLI batch runner
├── tune_masking.py        # Standalone colour-masking tuner
├── notebooks/
│   └── explore_embeddings.ipynb
├── docs/
│   └── pickle_format.md   # Pickle file format reference
├── requirements.txt
└── data/
    ├── all_images/        # Full image corpus
    ├── exemplars/         # Labelled subset for tuning
    └── exemplars_complexity.csv
```

---

## Pipeline overview

The pipeline consists of three main stages, each returning a typed dataclass that is stored in a per-image `.pkl` file (see [docs/pickle_format.md](docs/pickle_format.md) for the full schema).

```
Image
  │
  ├─ Stage 1 ─ Sauvola adaptive binarisation
  ├─ Stage 2 ─ Projection-variance deskew · border crop · binding-side detection
  │            · black-margin removal · fold-valley detection
  │              → PreprocessResult
  │
  ├─ Stage 3 ─ Kraken neural baseline segmentation
  │            · text-line polygon extraction · figure / illustration separation
  │              → SegmentKrakenResult
  │
  └─ Stage 4 ─ Corner artefact filter · narrow bbox filter
               · double-column gutter detection · polygon splitting at gutter
                 → PostprocessResult

  (optional)
  └─ Embeddings ─ 24-d structural vector + ViT CLS token (RGB / mask overlay)
                  · IsolationForest anomaly score · UMAP 2-D / 3-D projection
                    → PageEmbedding
```

Each image's results are saved as `{images_dir}/results/{stem}.pkl`, and a side-by-side annotated preview as `{images_dir}/results/{stem}_result{ext}`.

---

## Installation

### Prerequisites

- Python 3.13 (the project `.venv` uses 3.13)
- macOS with Apple Silicon recommended — MPS acceleration is picked up automatically for ViT embeddings; falls back to CPU on other hardware.

### 1 — Create a virtual environment

```bash
cd line_counting
python3.13 -m venv .venv
source .venv/bin/activate
```

### 2 — Install PyTorch

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> On Apple Silicon the standard `cpu` wheel includes MPS support.  
> Verify: `python -c "import torch; print(torch.backends.mps.is_available())"` → `True`.

### 3 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4 — Install Kraken

Kraken is required for Stage 3 segmentation.

```bash
pip install kraken
kraken get 10.5281/zenodo.10592135   # download the default blla model
```

After downloading, Kraken stores the model under your home directory:

```
~/Library/Application Support/htrmopo/97665cf3-f83d-5594-8855-f28d3af9df7a/blla.mlmodel
```

This path is hardcoded as `_DEFAULT_MODEL` at the top of `run_pipeline.py` (line 50). If you download the model on a different machine or to a different location, update that constant or pass the path explicitly with `--model-path`.

---

## Running the pipeline

### CLI batch runner

```bash
# Process all images in a folder
python run_pipeline.py --all --dir data/all_images

# Process a single image
python run_pipeline.py path/to/page.jpg

# Skip embedding computation
python run_pipeline.py --all --dir data/all_images --no-embed
```

Full option reference:

| Flag | Default | Description |
|---|---|---|
| `image` | — | Path to a single image |
| `--all` | off | Process every image in `--dir` |
| `--dir` | `data/exemplars` | Input directory |
| `--output` | `{dir}/results/results.csv` | Output CSV path |
| `--model-path` | Kraken default | Path to a `.mlmodel` segmentation file |
| `--device` | `cpu` | PyTorch device (`cpu`, `mps`, `cuda`) |
| `--corner-fraction` | `0.10` | Corner zone size as fraction of image dimensions |
| `--min-dimension-px` | `70` | Min short-side pixels to keep a detection |
| `--no-embed` | off | Skip embedding / anomaly scoring |
| `--vit-mask` | off | Also compute ViT embedding from the text+figure overlay |
| `--verbose` | off | Print per-page feature details |

Results are written to `{dir}/results/`:
- `results.csv` — one row per page with line counts and pipeline features
- `{stem}.pkl` — full dataclass output for each image (see [docs/pickle_format.md](docs/pickle_format.md))
- `{stem}_result{ext}` — side-by-side annotated preview (original | annotated)

---

## Streamlit tools

All apps are launched with `streamlit run <script>` and open at `http://localhost:8501`.

### `0_TuningPreprocess.py` — Preprocessing calibration

Interactively tune the parameters that control border/binding detection on a single image or all exemplars:

- Sauvola binarisation window and `k`
- `dark_threshold` for black-margin detection
- Binding valley detection (`strip_frac`, `valley_frac`, `smooth_frac`)
- Deskew range

```bash
streamlit run 0_TuningPreprocess.py
```

### `0_TuningPostprocess.py` — Postprocessing parameter tuning

Tune the Stage 4 filters on the full exemplar set; displays before/after overlays and aggregate statistics:

- `corner_fraction` — size of corner artefact exclusion zones
- `min_dimension_px` — minimum detection size
- `min_gutter_fraction` — sensitivity of double-column gutter detection
- `single_col_threshold` — fraction of lines that must span the full width to declare single-column

Saves parameters to a JSON config and re-runs the full Stage 4 over all `.pkl` files in a folder.

```bash
streamlit run 0_TuningPostprocess.py
```

### `1_ValidateCounting.py` — Manual line-count validation

Review pipeline results image by image, correct the line count if needed, and record ground-truth labels:

- Displays the annotated result image alongside pipeline metrics
- Editable line-count field with accept / reject controls
- Persists all labels to `{folder}/results/validation_counting.csv`
- Summary table showing validated vs. unvalidated pages

```bash
streamlit run 1_ValidateCounting.py
```

### `2_VisualizeEmbeddings.py` — 3-D UMAP embedding explorer *(WIP)*

Explore the embedding space of a processed image folder:

- Loads all `.pkl` files from a chosen folder and computes (or re-computes) 3-D UMAP projections
- Interactive 3-D Plotly scatter; clicking a point highlights it and loads the annotated image on the right
- Selectbox stays in sync with plot clicks (bidirectional)
- Anomaly score colour scale (red = anomalous, blue = typical)
- Structural feature table per selected page
- Sidebar controls: embedding type selection (structural / ViT RGB / ViT mask), UMAP `n_neighbors` and `min_dist`, dark/light mode

```bash
streamlit run 2_VisualizeEmbeddings.py
```

---

## Notes

- The ViT model (`vit_base_patch16_224`) is downloaded from HuggingFace/timm on first run (~330 MB) and cached locally.
- Kraken baseline segmentation requires a `.mlmodel` file; the default is downloaded via `kraken get` (see Installation).
- MPS acceleration is used automatically on Apple Silicon; no code changes are needed for CPU-only hardware.
