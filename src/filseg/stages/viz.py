"""viz: qualitative overlays: image, per-annotator GT contours and predicted instances."""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from filseg.competition import rle
from filseg.competition.metric import record_of, stem_of_record
from filseg.imaging import read_gray
from filseg.infra.paths import Paths

ANNOTATOR_COLORS = ["#2a78d6", "#1baf7a", "#eda100"]
PRED_COLOR = "#eb6834"


def _contours(mask: np.ndarray) -> list[np.ndarray]:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c[:, 0, :] for c in cs]


def render_overlay(paths: Paths, stem: str, predictions: Path | None, out: Path) -> None:
    """Draw GT contours (one color per annotator) and predicted instances (orange)."""
    is_train = (paths.train_images / f"{stem}.jpeg").exists()
    img = read_gray((paths.train_images if is_train else paths.test_images) / f"{stem}.jpeg")
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, cmap="gray")
    if is_train and paths.gt_csv.exists():
        gt = pd.read_csv(paths.gt_csv)
        rec = gt["filament_id"].map(record_of)
        gt = gt[rec.map(stem_of_record) == stem]
        for i, (_, grp) in enumerate(gt.groupby(rec[gt.index])):
            for counts in grp["segmentation_rle"]:
                for c in _contours(rle.decode(counts)):
                    ax.plot(c[:, 0], c[:, 1], color=ANNOTATOR_COLORS[i % 3], lw=0.8)
    if predictions is not None:
        pred = pd.read_csv(predictions)
        pred = pred[pred["filament_id"].map(record_of) == stem]
        for counts in pred["segmentation_rle"]:
            for c in _contours(rle.decode(counts)):
                ax.plot(c[:, 0], c[:, 1], color=PRED_COLOR, lw=1.2, ls="--")
    ax.set_title(f"{stem}: GT per annotator (solid), predictions (dashed orange)")
    ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
