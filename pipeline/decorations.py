"""Decoration detection for medieval manuscript pages.

Two public classes:

    ManuscriptPage    — loads a page + its pipeline pickle; provides
                        pre-processing helpers and plotting methods.
    DecorationModel   — wraps a detection backend (medieval_yolo, florence2, …);
                        runs inference and returns post-processed detections.

Typical usage::

    from pipeline.decorations import ManuscriptPage, DecorationModel

    page  = ManuscriptPage("data/exemplars/my_page.jpg")
    model = DecorationModel(backend="medieval_yolo", size="x")

    detections = model.run(page)
    page.plot_raw()
    page.plot_detections(detections)
    print(page.summarize(detections))
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Set

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

CATEGORY_MAP: dict[str, str] = {
    # Florence-2 open-vocabulary labels
    "decorated initial":  "decorated_initial",
    "filigree initial":   "initial_filigree",
    "illuminated letter": "decorated_initial",
    "marginalia":         "marginalia",
    "large miniature":    "large_decoration",
    "decorated border":   "large_decoration",
    # Generic YOLO / COCO labels
    "figure":             "large_decoration",
    "picture":            "large_decoration",
    "illustration":       "large_decoration",
    # Medieval YOLO (SegmOnto) labels — normalised to lowercase, no spaces
    "dropcapitalzone":    "decorated_initial",
    "mainzone":           "other_interesting",
    "graphiczone":        "large_decoration",
    "defaultline":        "other",
    "headingline":        "other_interesting",
    "dropcapitalline":    "other_interesting",
}

CATEGORY_COLORS: dict[str, tuple] = {
    "decorated_initial": (220,  50,  50),   # red   (BGR)
    "initial_filigree":  ( 50, 180,  50),   # green
    "marginalia":        ( 50,  50, 220),   # blue
    "large_decoration":  (200, 140,  30),   # orange
    "other":             (160, 160, 160),   # grey
    "other_interesting": ( 50, 200, 200),   # teal
}

DEFAULT_CONFIDENCE: float = 0.20

DEFAULT_FLORENCE2_PROMPTS: str = (
    "decorated initial . filigree initial . illuminated letter . "
    "marginalia . large miniature . decorated border"
)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _remove_text_lines(bgr: np.ndarray, line_boundaries: list) -> np.ndarray:
    out = bgr.copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bg_val = int(np.percentile(gray, 95))
    for boundary in line_boundaries:
        pts = np.array(boundary, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(out, [pts], color=(bg_val, bg_val, bg_val))
    return out


def _apply_figure_mask(bgr: np.ndarray, figure_binary: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bg_val = int(np.percentile(gray, 95))
    out[figure_binary == 0] = bg_val
    return out


def _should_suppress(det: dict, kept: dict, iou_threshold: float, containment_threshold: float) -> bool:
    """Return True if det should be suppressed given an already-kept box.

    Two independent criteria, either is sufficient:
    - IoU >= iou_threshold          (standard symmetric overlap)
    - smaller box containment >= containment_threshold  (one box almost fully inside the other)
    """
    x1 = max(det["x1"], kept["x1"])
    y1 = max(det["y1"], kept["y1"])
    x2 = min(det["x2"], kept["x2"])
    y2 = min(det["y2"], kept["y2"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return False
    area_det  = det["width"]  * det["height"]
    area_kept = kept["width"] * kept["height"]
    iou = inter / (area_det + area_kept - inter)
    if iou >= iou_threshold:
        return True
    smaller_area = min(area_det, area_kept)
    if inter / smaller_area >= containment_threshold:
        return True
    return False


def _nms(detections: list, iou_threshold: float, containment_threshold: float) -> list:
    """Class-agnostic NMS using IoU and containment suppression."""
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for det in sorted_dets:
        if not any(_should_suppress(det, k, iou_threshold, containment_threshold) for k in kept):
            kept.append(det)
    return kept


# ---------------------------------------------------------------------------
# ManuscriptPage
# ---------------------------------------------------------------------------

class ManuscriptPage:
    """A single manuscript page with its pipeline results.

    Parameters
    ----------
    image_path : str or Path
        Path to the source image processed by the pipeline.
    pkl_path : str or Path, optional
        Path to the matching pickle. Auto-detected as
        ``<image_dir>/results/<stem>.pkl`` when omitted.
    """

    def __init__(self, image_path, pkl_path=None):
        self.image_path = Path(image_path).expanduser().resolve()
        if not self.image_path.is_file():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        pkl = (
            Path(pkl_path)
            if pkl_path is not None
            else self.image_path.parent / "results" / self.image_path.with_suffix(".pkl").name
        )
        if not pkl.is_file():
            raise FileNotFoundError(
                f"Pickle not found: {pkl}\n"
                "Run run_pipeline.py on the image first, or pass pkl_path."
            )

        with open(pkl, "rb") as fh:
            result = pickle.load(fh)

        self._pre  = result["pre"]
        self._seg  = result["seg"]
        self._post = result["post"]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def bgr(self) -> np.ndarray:
        return self._pre.bgr

    @property
    def line_boundaries(self) -> list:
        return self._post.line_boundaries

    @property
    def figure_binary(self) -> Optional[np.ndarray]:
        return self._seg.figure_binary

    @property
    def image_h(self) -> int:
        return self._pre.image_h

    @property
    def image_w(self) -> int:
        return self._pre.image_w

    @property
    def binding_side(self) -> str:
        return self._pre.binding_side

    @property
    def deskew_angle(self) -> float:
        return self._pre.deskew_angle

    @property
    def n_lines(self) -> int:
        return self._post.n_lines

    @property
    def n_figures(self) -> int:
        return self._post.n_figures

    # ── Pre-processing ────────────────────────────────────────────────────────

    def get_model_input(
        self,
        remove_lines: bool = False,
        use_figure_mask: bool = False,
    ) -> np.ndarray:
        """Return the page image ready for a detection model.

        Parameters
        ----------
        remove_lines : bool
            Paint Kraken text-line polygons to the background colour before
            passing to the model, reducing handwriting noise.
        use_figure_mask : bool
            Zero out pixels outside ``figure_binary``.  Suppresses marginalia
            that lie entirely outside Kraken text-line regions.
        """
        img = self._pre.bgr.copy()
        if remove_lines:
            img = _remove_text_lines(img, self.line_boundaries)
        if use_figure_mask and self.figure_binary is not None:
            img = _apply_figure_mask(img, self.figure_binary)
        return img

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot_raw(self, figsize: tuple = (14, 10)) -> None:
        """Show the raw page with text-line polygons and figure-ink overlay."""
        annotated = self._pre.bgr.copy()

        if self.figure_binary is not None and self.figure_binary.any():
            overlay = annotated.copy()
            overlay[self.figure_binary > 0] = (0, 220, 220)   # yellow tint (BGR)
            cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

        for boundary in self.line_boundaries:
            pts = np.array(boundary, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 0, 220), thickness=2)

        if self._post.is_double_column and self._post.gutter_x is not None:
            cv2.line(
                annotated,
                (self._post.gutter_x, self._post.gutter_y_min),
                (self._post.gutter_x, self._post.gutter_y_max),
                (0, 220, 0), thickness=4,
            )

        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.title(f"{self.image_path.name}  |  red=text lines  yellow=figure ink")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def plot_detections(
        self,
        detections: List[dict],
        exclude_categories: Optional[Set[str]] = None,
        figsize: tuple = (14, 10),
        save_dir: Optional[Path] = None,
    ) -> None:
        """Overlay detection bounding boxes on the page image.

        Parameters
        ----------
        detections : list[dict]
            Output of ``DecorationModel.run()``.
        exclude_categories : set[str], optional
            Category names to hide. Defaults to ``{"other", "other_interesting"}``.
        save_dir : Path, optional
            Directory to save the annotated image. Saved as
            ``<save_dir>/<stem>_decorations<suffix>``. Directory is created if
            it does not exist.
        """
        if exclude_categories is None:
            exclude_categories = {"other", "other_interesting"}

        viz = self._pre.bgr.copy()

        for d in detections:
            if d["category"] in exclude_categories:
                continue
            color = CATEGORY_COLORS.get(d["category"], CATEGORY_COLORS["other"])
            cv2.rectangle(viz, (d["x1"], d["y1"]), (d["x2"], d["y2"]), color, thickness=3)

            label_text = f"{d['category']} {d['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(
                viz,
                (d["x1"], d["y1"] - th - 6),
                (d["x1"] + tw + 4, d["y1"]),
                color, thickness=-1,
            )
            cv2.putText(
                viz, label_text, (d["x1"] + 2, d["y1"] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                thickness=1, lineType=cv2.LINE_AA,
            )

        legend_patches = [
            mpatches.Patch(color=tuple(c / 255 for c in rgb[::-1]), label=cat)
            for cat, rgb in CATEGORY_COLORS.items()
            if cat not in exclude_categories
        ]
        n_shown = sum(1 for d in detections if d["category"] not in exclude_categories)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Decorations — {self.image_path.name}  ({n_shown} shown)")
        ax.axis("off")
        ax.legend(handles=legend_patches, loc="upper right", fontsize=9, framealpha=0.8)
        plt.tight_layout()
        plt.show()

        if save_dir is not None:
            out_dir = Path(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (self.image_path.stem + "_decorations" + self.image_path.suffix)
            cv2.imwrite(str(out_path), viz)
            print(f"Saved: {out_path}")

    def plot_summary(
        self,
        detections: List[dict],
        exclude_categories: Optional[Set[str]] = None,
        figsize: tuple = (7, 4),
    ) -> None:
        """Bar chart of detection counts by category."""
        if exclude_categories is None:
            exclude_categories = {"other", "other_interesting"}

        df = pd.DataFrame(detections)
        if df.empty:
            print("No detections.")
            return

        counts = df["category"].value_counts()
        counts = counts[~counts.index.isin(exclude_categories)]
        colors_mpl = [
            tuple(c / 255 for c in CATEGORY_COLORS.get(cat, CATEGORY_COLORS["other"])[::-1])
            for cat in counts.index
        ]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(counts.index, counts.values, color=colors_mpl)
        ax.bar_label(bars)
        ax.set_title(f"Decoration counts — {self.image_path.name}")
        ax.set_ylabel("Count")
        ax.set_xlabel("Category")
        plt.tight_layout()
        plt.show()

    def summarize(self, detections: List[dict]) -> pd.DataFrame:
        """Return a per-category summary DataFrame (count, mean confidence, mean area)."""
        df = pd.DataFrame(detections)
        if df.empty:
            return df
        return (
            df.groupby("category")
              .agg(
                  count=("category", "size"),
                  mean_conf=("confidence", "mean"),
                  mean_area_px=("area_px", "mean"),
              )
              .round(3)
              .sort_values("count", ascending=False)
        )

    def __repr__(self) -> str:
        return (
            f"ManuscriptPage({self.image_path.name!r}, "
            f"{self.image_w}×{self.image_h}px, "
            f"{self.n_lines} lines)"
        )


# ---------------------------------------------------------------------------
# DecorationModel
# ---------------------------------------------------------------------------

class DecorationModel:
    """Detection model wrapper for manuscript decoration detection.

    Parameters
    ----------
    backend : str
        ``'medieval_yolo'`` (default), ``'yolo'``, ``'florence2'``, or
        ``'florence2_medieval'``.
    size : str
        Model size for ``medieval_yolo`` — ``'n'``, ``'s'``, ``'m'``, ``'l'``,
        ``'x'`` (default).  Ignored for other backends.
    confidence_threshold : float
        Detections below this score are dropped (default 0.20).
    prompts : str, optional
        Comma-separated text prompts for Florence-2 open-vocabulary detection.
        Defaults to ``DEFAULT_FLORENCE2_PROMPTS``.
    """

    def __init__(
        self,
        backend: str = "medieval_yolo",
        size: str = "x",
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = 0.5,
        containment_threshold: float = 0.8,
        nms_exclude_categories: Optional[Set[str]] = None,
        prompts: Optional[str] = None,
    ):
        import torch

        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.containment_threshold = containment_threshold
        self.nms_exclude_categories = (
            nms_exclude_categories
            if nms_exclude_categories is not None
            else {"other", "other_interesting"}
        )
        self.prompts = prompts or DEFAULT_FLORENCE2_PROMPTS

        self._device = (
            "mps"  if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Device: {self._device}")

        if backend in ("florence2", "florence2_medieval"):
            self._load_florence2(backend)
        elif backend in ("yolo", "medieval_yolo"):
            self._load_yolo(backend, size)
        else:
            raise ValueError(
                f"Unknown backend {backend!r}. "
                "Choose: 'medieval_yolo', 'yolo', 'florence2', 'florence2_medieval'."
            )

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_florence2(self, backend: str) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_id = (
            "medieval-data/florence2-medieval-bbox-zone-detection"
            if backend == "florence2_medieval"
            else "microsoft/Florence-2-base"
        )
        print(f"Loading Florence-2 from {model_id} …")
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._florence = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, attn_implementation="eager"
        ).to(self._device).eval()

        if backend == "florence2_medieval":
            self._patch_florence2_medieval()

        print("Florence-2 ready.")

    def _patch_florence2_medieval(self) -> None:
        # The fine-tune omits GenerationMixin; required in transformers ≥4.45.
        from transformers import GenerationConfig
        from transformers.generation.utils import GenerationMixin

        lang_cls = type(self._florence.language_model)
        if not issubclass(lang_cls, GenerationMixin):
            lang_cls.__bases__ = lang_cls.__bases__ + (GenerationMixin,)
        self._florence.language_model.generation_config = (
            GenerationConfig.from_model_config(self._florence.language_model.config)
        )

    def _load_yolo(self, backend: str, size: str) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Run: pip install ultralytics")

        if backend == "medieval_yolo":
            from huggingface_hub import hf_hub_download
            filename = f"medieval-yolov11{size}.pt"
            print(f"Downloading {filename} from biglam/medieval-manuscript-yolov11 …")
            print("(cached after first run)")
            ckpt = hf_hub_download(
                repo_id="biglam/medieval-manuscript-yolov11",
                filename=filename,
            )
        else:
            ckpt = "yolov8n.pt"
            print("Loading generic YOLOv8n (COCO) …")

        self._yolo = YOLO(ckpt)
        print(f"YOLO ready.  Classes: {list(self._yolo.names.values())}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def run(
        self,
        page: ManuscriptPage,
        remove_lines: bool = False,
        use_figure_mask: bool = False,
    ) -> List[dict]:
        """Run detection on a page and return post-processed detections.

        Parameters
        ----------
        page : ManuscriptPage
        remove_lines : bool
            Paint text-line polygons to background before inference.
        use_figure_mask : bool
            Restrict detection to figure_binary regions.

        Returns
        -------
        list[dict]
            Each dict has keys: x1, y1, x2, y2, confidence, raw_label,
            category, width, height, area_px.
        """
        img = page.get_model_input(remove_lines=remove_lines, use_figure_mask=use_figure_mask)
        raw = self._infer(img)
        return self._postprocess(raw)

    def _infer(self, img: np.ndarray) -> list:
        if self.backend == "florence2":
            return self._infer_florence2(img, task="<OPEN_VOCABULARY_DETECTION>")
        elif self.backend == "florence2_medieval":
            return self._infer_florence2(img, task="<OD>")
        else:
            return self._infer_yolo(img)

    def _infer_florence2(self, img: np.ndarray, task: str) -> list:
        import torch
        from PIL import Image as PILImage

        img_pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        text = task if task == "<OD>" else task + self.prompts

        inputs = self._processor(
            text=text, images=img_pil, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            generated_ids = self._florence.generate(
                **inputs, max_new_tokens=1024, do_sample=False, use_cache=False
            )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        img_h, img_w = img.shape[:2]
        parsed = self._processor.post_process_generation(
            generated_text, task=task, image_size=(img_w, img_h)
        )
        result = parsed.get(task, {})
        raw = []
        for (x1, y1, x2, y2), label in zip(
            result.get("bboxes", []),
            result.get("labels", []),
        ):
            raw.append((int(x1), int(y1), int(x2), int(y2), 1.0, label.lower().strip()))
        return raw

    def _infer_yolo(self, img: np.ndarray) -> list:
        results = self._yolo(img, conf=self.confidence_threshold, verbose=False)
        raw = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self._yolo.names[cls_id].lower().replace(" ", "")
                raw.append((int(x1), int(y1), int(x2), int(y2), score, label))
        return raw

    def _postprocess(self, raw: list) -> List[dict]:
        out = []
        for x1, y1, x2, y2, score, raw_label in raw:
            if score < self.confidence_threshold:
                continue
            category = CATEGORY_MAP.get(raw_label)
            if category is None:
                for key, cat in CATEGORY_MAP.items():
                    if key in raw_label or raw_label in key:
                        category = cat
                        break
            if category is None:
                category = "other"
            out.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": round(score, 3),
                "raw_label":  raw_label,
                "category":   category,
                "width":      x2 - x1,
                "height":     y2 - y1,
                "area_px":    (x2 - x1) * (y2 - y1),
            })
        relevant   = [d for d in out if d["category"] not in self.nms_exclude_categories]
        irrelevant = [d for d in out if d["category"]     in self.nms_exclude_categories]
        return _nms(relevant, self.iou_threshold, self.containment_threshold) + irrelevant

    def __repr__(self) -> str:
        return f"DecorationModel(backend={self.backend!r}, conf≥{self.confidence_threshold})"
