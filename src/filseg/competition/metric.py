"""Panoptic Quality exactly as computed by the organizers' self-evaluation notebook.

Reference: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

Semantics reproduced here (see ``tests/test_pq.py`` for an equivalence test against a literal
port of the official code):

* Ground truth is grouped per **annotator record** (``<annotator>-<stem>``); all predictions of
  ``<stem>`` are compared against every record of that image separately.
* A (GT, pred) pair is a TP iff IoU > 0.5; every such pair counts (unique by construction when
  predictions do not overlap).
* FP = predictions hitting no GT of the record, FN = GT hitting no prediction.
* PQ = sum(IoU of TPs) / (TP + FP/2 + FN/2), pooled over all records.
* Predictions for images without any GT record are ignored.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from filseg.competition import rle

IOU_THRESHOLD = 0.5


def record_of(filament_id: str) -> str:
    """``"<record>_<k>"`` -> ``"<record>"``."""
    return filament_id.split("_", maxsplit=1)[0]


def stem_of_record(record_id: str) -> str:
    """``"010101-20140609195854Bh"`` -> ``"20140609195854Bh"``."""
    return record_id.split("-", maxsplit=1)[1]


@dataclass
class RecordMatch:
    """Overlap statistics between one GT record and the predictions for its image."""

    record_id: str
    iou: np.ndarray  # (n_gt, n_pred)

    @property
    def hits(self) -> np.ndarray:
        hits: np.ndarray = self.iou > IOU_THRESHOLD
        return hits

    @property
    def tp_ious(self) -> np.ndarray:
        vals: np.ndarray = self.iou[self.hits]
        return vals

    @property
    def fp(self) -> int:
        return int((self.hits.sum(axis=0) == 0).sum())

    @property
    def fn(self) -> int:
        return int((self.hits.sum(axis=1) == 0).sum())


@dataclass
class PQResult:
    """Aggregate PQ with the per-record overlap matrices, plus the distributions the judges
    score besides PQ: IoU / Dice of the overlapping pairs and the 1:n / n:1 relations."""

    matches: list[RecordMatch] = field(repr=False)
    tp: int
    fp: int
    fn: int
    sum_iou: float

    @property
    def pq(self) -> float:
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.sum_iou / denom if denom > 0 else 0.0

    @property
    def sq(self) -> float:
        """Segmentation quality: mean IoU of matched pairs."""
        return self.sum_iou / self.tp if self.tp else 0.0

    @property
    def rq(self) -> float:
        """Recognition quality: F1 of detection at IoU > 0.5."""
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / denom if denom > 0 else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "pq": self.pq,
            "sq": self.sq,
            "rq": self.rq,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "records": len(self.matches),
        }

    def pairs(self) -> pd.DataFrame:
        """Every overlapping (GT, prediction) pair, one row per pair and annotator record.

        Columns: ``record_id``, ``gt_idx`` and ``pred_idx`` (indices within the record / image),
        ``iou``, ``dice`` and ``tp`` (IoU > 0.5, the pairs PQ counts). For two masks
        Dice = 2·IoU / (1 + IoU) exactly, so IoU > 0.5 is Dice > 2/3.
        """
        cols = {"record_id": str, "gt_idx": int, "pred_idx": int, "iou": float}
        frames = []
        for m in self.matches:
            g, p = np.nonzero(m.iou > 0)
            frames.append(
                pd.DataFrame(
                    {"record_id": m.record_id, "gt_idx": g, "pred_idx": p, "iou": m.iou[g, p]}
                )
            )
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(cols))
        out = out.astype(cols)
        out["dice"] = 2 * out["iou"] / (1 + out["iou"])
        out["tp"] = out["iou"] > IOU_THRESHOLD
        return out

    def relations(self) -> pd.DataFrame:
        """One-to-many and many-to-one relations, counted per annotator record like PQ.

        Row ``n`` (index ``partners``): column ``gt_filaments`` counts the GT filaments that
        overlap (IoU > 0) exactly ``n`` predictions, column ``predictions`` the predictions that
        overlap exactly ``n`` GT filaments of the record. ``gt_filaments`` at ``n`` >= 2 is
        one-to-many (a filament split into pieces), ``predictions`` at ``n`` >= 2 many-to-one
        (filaments merged), and row 0 holds the missed filaments and the predictions with no
        annotation under them.
        """
        empty = np.zeros(0, np.int64)
        gt_deg = np.concatenate([(m.iou > 0).sum(axis=1) for m in self.matches] or [empty])
        pred_deg = np.concatenate([(m.iou > 0).sum(axis=0) for m in self.matches] or [empty])
        n = int(max(gt_deg.max(initial=0), pred_deg.max(initial=0))) + 1
        return pd.DataFrame(
            {
                "gt_filaments": np.bincount(gt_deg, minlength=n),
                "predictions": np.bincount(pred_deg, minlength=n),
            },
            index=pd.RangeIndex(n, name="partners"),
        )


def evaluate(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> PQResult:
    """Compute PQ of predictions against ground truth.

    Args:
        gt_df: Columns ``filament_id`` (``<annotator>-<stem>_<k>``) and ``segmentation_rle``.
        pred_df: Submission-format frame, ``filament_id`` = ``<stem>_<k>``.

    Returns:
        The aggregated result with per-record overlap matrices.
    """
    gt_records = gt_df["filament_id"].map(record_of)
    pred_stems = pred_df["filament_id"].map(record_of)
    pred_by_stem: dict[str, list[str]] = {
        str(stem): list(group) for stem, group in pred_df.groupby(pred_stems)["segmentation_rle"]
    }

    matches: list[RecordMatch] = []
    tp = fp = fn = 0
    sum_iou = 0.0
    for record_id, gt_rles in gt_df.groupby(gt_records, sort=True)["segmentation_rle"]:
        preds = pred_by_stem.get(stem_of_record(str(record_id)), [])
        m = RecordMatch(record_id=str(record_id), iou=rle.iou_matrix(list(gt_rles), preds))
        matches.append(m)
        tp += int(m.hits.sum())
        fp += m.fp
        fn += m.fn
        sum_iou += float(m.tp_ious.sum())
    return PQResult(matches=matches, tp=tp, fp=fp, fn=fn, sum_iou=sum_iou)


def evaluate_subset(gt_df: pd.DataFrame, pred_df: pd.DataFrame, stems: set[str]) -> PQResult:
    """Evaluate only on the given image stems (e.g. one CV fold)."""
    # astype(bool): an empty frame maps to an object Series, which pandas reads as column labels
    gt_mask = (
        gt_df["filament_id"].map(lambda f: stem_of_record(record_of(f)) in stems).astype(bool)
    )
    pred_mask = pred_df["filament_id"].map(lambda f: record_of(f) in stems).astype(bool)
    return evaluate(gt_df[gt_mask], pred_df[pred_mask])
