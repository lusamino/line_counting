# Medieval Manuscript Line Counter

A Python pipeline to count text lines and extract bounding boxes from scanned pages of medieval manuscripts, with a Streamlit interface for manual validation and anomaly inspection.

---

## Project structure

```
line_counting/
├── pipeline/
│   ├── __init__.py
│   ├── preprocessing.py   # Stages 1–2: binarisation, deskew, page extraction
│   ├── layout.py          # Stage 3:   column layout detection
│   ├── masking.py         # Stage 4:   non-text (illustration / damage) masking
│   ├── segmentation.py    # Stages 5–6: line segmentation + bounding boxes
│   └── embeddings.py      # Stage 7:   structural + visual embeddings, anomaly scoring
├── app.py                 # Stage 8:   Streamlit frontend
├── run_pipeline.py        # CLI entry point
├── test_pipeline.ipynb    # Stage-by-stage interactive notebook
├── requirements.txt
└── data/
    ├── exemplars/         # Input images (JPG / PNG / TIF)
    └── exemplars_complexity.csv
```

---

## Installation

### Prerequisites

- Python ≥ 3.10
- macOS with Apple Silicon (M1/M2/M3) recommended — MPS acceleration is used automatically for the ViT embedding stage; the pipeline falls back to CPU on other hardware.

### 1 — Create a virtual environment

```bash
cd ~/Dropbox/My_stuff/05_SDSCresearch/10_SideProjects/00_MedievalCambridge/line_counting
python -m venv .venv
source .venv/bin/activate
```

### 2 — Install PyTorch (MPS-enabled build)

Install PyTorch **before** the rest of the requirements so that the correct Metal-accelerated wheel is selected:

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> On Apple Silicon the standard `cpu` wheel already includes MPS support.  
> Verify with `python -c "import torch; print(torch.backends.mps.is_available())"` — should print `True`.

### 3 — Install all other dependencies

```bash
pip install -r requirements.txt
```

### 4 — (Optional) Install Kraken

Kraken is only needed if you want to use the Kraken segmentation method or if the HPP fallback is triggered.  Install it separately because it pulls in heavy dependencies:

```bash
pip install kraken
```

After installing, download a pre-trained Kraken segmentation model:

```bash
kraken get 10.5281/zenodo.10592135   # blla default model
```

---

## Running the pipeline

### Streamlit app (recommended)

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

### Command-line interface

```bash
# Process a single image
python run_pipeline.py data/exemplars/MS-GG-00001-00001-000-00041_double_simple.jpg

# Process all exemplars, write CSV, skip embeddings
python run_pipeline.py --all --no-embed --output results.csv

# Process all exemplars with Kraken segmentation
python run_pipeline.py --all --method kraken --output results_kraken.csv
```

Full option reference:

| Flag | Default | Description |
|---|---|---|
| `image` | — | Path to a single image |
| `--all` | off | Process all images in `--exemplars-dir` |
| `--exemplars-dir` | `data/exemplars` | Input directory |
| `--output` | `results.csv` | Output CSV path |
| `--method` | `hpp` | Segmentation method: `hpp` or `kraken` |
| `--no-embed` | off | Skip embedding / anomaly scoring stage |
| `--verbose` | off | Print per-line details |

### Output CSV columns

| Column | Description |
|---|---|
| `filename` | Source image filename |
| `layout_type` | `single` / `double` / `mixed` |
| `column` | 0-based column index |
| `line_index` | 0-based line index within the column |
| `x_min`, `y_min`, `x_max`, `y_max` | Bounding box in image pixel coordinates |
| `height` | Bounding box height (pixels) |
| `is_anomalous` | 1 if line height > 2.5 × median |
| `method` | `hpp` or `kraken` |
| `deskew_angle` | Rotation applied during deskew (degrees) |
| `anomaly_score` | Page-level anomaly score in [0, 1] |
| `is_validated` | 1 if manually accepted/rejected in the app |

---

## Testing individual stages

Each module can be run directly for a quick visual sanity-check:

```bash
# Stage 1–2: preprocessing
python -m pipeline.preprocessing data/exemplars/MS-GG-00001-00001-000-00041_double_simple.jpg

# Stage 3: layout detection
python -m pipeline.layout data/exemplars/MS-GG-00001-00001-000-00041_double_simple.jpg

# Stage 4: non-text masking
python -m pipeline.masking data/exemplars/MS-GG-00001-00001-000-00284_double_big_stain.jpg

# Stage 5–6: segmentation
python -m pipeline.segmentation data/exemplars/MS-GG-00001-00001-000-00041_double_simple.jpg

# Stage 7: embeddings (first 5 exemplars by default)
python -m pipeline.embeddings
```

Each script saves a preview PNG to `/tmp/` and opens a matplotlib window.

The interactive notebook `test_pipeline.ipynb` walks through all stages cell by cell.

---

## Pipeline overview

```
Image
  │
  ├─ Stage 1 ─ Sauvola binarisation + projection-variance deskew
  ├─ Stage 2 ─ Border crop (largest contour) · binding-side detection · masking
  ├─ Stage 3 ─ Vertical projection per slab → column layout (single/double/mixed)
  ├─ Stage 4 ─ Connected-component analysis → remove illustrations & damage
  ├─ Stage 5 ─ HPP line segmentation (Kraken fallback if >15% anomalous)
  ├─ Stage 6 ─ Tight bounding boxes + vertical padding
  └─ Stage 7 ─ ViT visual embedding + structural vector → IsoForest + UMAP
```

---

## Notes

- The ViT model (`vit_base_patch16_224`) is downloaded from HuggingFace/timm on first use (~330 MB).  Subsequent runs use the local cache.
- MPS is used automatically when available; no code changes are needed to run on CPU-only hardware.
- Kraken baseline segmentation expects a model file; see the Kraken installation step above.
