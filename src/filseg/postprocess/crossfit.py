"""Honest PQ of a tuned selection rule (crossfit) from per-image PQ counts.

*Per-image PQ counts* are a table indexed by image stem with columns ``tp, fp, fn, sum_iou``,
each summed over the image's annotator records. PQ of any set of images is
``sum(sum_iou) / (sum(tp) + (sum(fp) + sum(fn)) / 2)``, so counts pool exactly like the official
metric.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

COUNT_COLUMNS = ("tp", "fp", "fn", "sum_iou")
_TP, _FP, _FN, _IOU = range(4)


def pq_from_counts(counts: np.ndarray) -> np.ndarray:
    """``(..., 4)`` array of (tp, fp, fn, sum_iou) -> PQ over the last axis (0 when empty)."""
    denom = counts[..., _TP] + 0.5 * (counts[..., _FP] + counts[..., _FN])
    out: np.ndarray = np.divide(
        counts[..., _IOU], denom, out=np.zeros_like(denom, dtype=np.float64), where=denom > 0
    )
    return out


def summary(counts: np.ndarray) -> dict[str, float]:
    """(4,) pooled counts -> ``pq, tp, fp, fn``."""
    return {
        "pq": float(pq_from_counts(counts)),
        "tp": int(counts[_TP]),
        "fp": int(counts[_FP]),
        "fn": int(counts[_FN]),
    }


@dataclass
class FoldChoice:
    fold: int
    row: int  # index of the grid point chosen on the other folds
    train_pq: float  # its PQ on the other folds
    counts: np.ndarray  # (4,) its counts on this fold


def crossfit(fold_counts: np.ndarray) -> list[FoldChoice]:
    """Per fold, pick the grid point with the best PQ on the other folds and score it here.

    Args:
        fold_counts: ``(n_points, n_folds, 4)`` counts of every grid point on every fold.
            Ties go to the lowest index.
    """
    total = fold_counts.sum(axis=1)
    choices = []
    for k in range(fold_counts.shape[1]):
        train_pq = pq_from_counts(total - fold_counts[:, k])
        row = int(np.argmax(train_pq))
        choices.append(FoldChoice(k, row, float(train_pq[row]), fold_counts[row, k]))
    return choices


def counts_frame(stems: Sequence[str], counts: np.ndarray) -> pd.DataFrame:
    """``(n_images, 4)`` counts -> per-image table (index: stem)."""
    df = pd.DataFrame(counts, index=pd.Index(list(stems), name="stem"), columns=COUNT_COLUMNS)
    return df.astype({"tp": int, "fp": int, "fn": int})
