"""Grid search of instance-extraction parameters directly on PQ, using out-of-fold maps.

Expensive work (labelling, RLE encoding, IoU against every annotator record) depends only on
``(t_low, close_radius)``; the remaining parameters are filters over the resulting components
(`keep`), so they are evaluated on cached IoU matrices at negligible cost. A new filter is a field
of `InstanceParams` checked in `keep`; adding it to `TuneGrid` makes the search cover it.

Besides the in-sample ranking, `tune` reports a crossfit PQ (`filseg.postprocess.crossfit`): each
fold is scored with the grid point that is best on the other folds. That is the number to compare
selection rules on.
"""

import itertools
import logging
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from filseg.competition import rle
from filseg.competition.metric import IOU_THRESHOLD, record_of, stem_of_record
from filseg.config import InstanceParams, TuneGrid
from filseg.imaging import read_prob
from filseg.postprocess.crossfit import FoldChoice, counts_frame, crossfit, summary
from filseg.postprocess.instances import Component, components, keep

log = logging.getLogger(__name__)

# The parameters that shape the components; every other one only filters them.
COMPONENT_PARAMS = ("t_low", "close_radius")


@dataclass
class ImageCandidates:
    stem: str
    comps: list[Component]
    ious: dict[str, np.ndarray]  # record_id -> (n_gt, n_comp)


def gt_by_stem(gt_df: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """GT table -> {stem: {record_id: [rle, ...]}}."""
    out: dict[str, dict[str, list[str]]] = {}
    records = gt_df["filament_id"].map(record_of)
    for record_id, group in gt_df.groupby(records)["segmentation_rle"]:
        out.setdefault(stem_of_record(str(record_id)), {})[str(record_id)] = list(group)
    return out


def build_candidates(
    stem: str,
    gt: Mapping[str, Sequence[str]],
    *,
    prob_dir: Path,
    t_low: float,
    close_radius: int,
    min_area: int,
    min_peak: float,
) -> ImageCandidates:
    prob = read_prob(prob_dir / f"{stem}.png")
    labels, comps = components(prob, t_low=t_low, close_radius=close_radius)
    comps = [c for c in comps if c.area >= min_area and c.max_prob >= min_peak]
    rles = [rle.encode(labels == c.label) for c in comps]
    return ImageCandidates(
        stem=stem, comps=comps, ious={rid: rle.iou_matrix(g, rles) for rid, g in gt.items()}
    )


@dataclass
class CandidateTable:
    """Candidates of many images, flattened, with what each one earns if kept.

    Components of one image are disjoint, so a GT filament overlaps at most one of them with
    IoU > 0.5. Hence an image's counts for any selection are sums over its kept candidates, and
    ``fn = n_gt - tp``.
    """

    comps: list[Component]
    image: np.ndarray  # (n_cand,) index into the image list
    gain: np.ndarray  # (n_cand, 3): records hit (tp), records missed (fp), sum of hit IoUs
    n_gt: np.ndarray  # (n_images,) GT filaments summed over records

    @classmethod
    def from_images(cls, images: Sequence[ImageCandidates]) -> "CandidateTable":
        comps: list[Component] = []
        image: list[int] = []
        gains: list[np.ndarray] = []
        n_gt = np.zeros(len(images))
        for i, img in enumerate(images):
            gain = np.zeros((len(img.comps), 3))
            for iou in img.ious.values():
                hit = iou > IOU_THRESHOLD
                n_hit = hit.sum(axis=0)
                gain[:, 0] += n_hit
                gain[:, 1] += n_hit == 0
                gain[:, 2] += np.where(hit, iou, 0.0).sum(axis=0)
                n_gt[i] += iou.shape[0]
            comps += img.comps
            image += [i] * len(img.comps)
            gains.append(gain)
        gain = np.concatenate([np.zeros((0, 3)), *gains])
        return cls(comps, np.array(image, dtype=np.intp), gain, n_gt)

    def select(self, params: InstanceParams) -> np.ndarray:
        """Boolean mask of the candidates `keep` accepts."""
        return np.fromiter((keep(c, params) for c in self.comps), bool, len(self.comps))

    def image_counts(self, sel: np.ndarray) -> np.ndarray:
        """(n_images, 4) PQ counts ``tp, fp, fn, sum_iou`` of the selected candidates."""
        n = len(self.n_gt)
        tp, fp, iou = (
            np.bincount(self.image[sel], weights=self.gain[sel, j], minlength=n) for j in range(3)
        )
        return np.stack([tp, fp, self.n_gt - tp, iou], axis=1)


def pq_of(cands: Sequence[ImageCandidates], params: InstanceParams) -> dict[str, float]:
    """PQ (official semantics) of the components passing `params` filters."""
    table = CandidateTable.from_images(cands)
    return summary(table.image_counts(table.select(params)).sum(axis=0))


@dataclass
class TuneResult:
    results: pd.DataFrame  # one row per grid point (InstanceParams fields + pq, tp, fp, fn)
    folds: list[FoldChoice]  # crossfit choice per fold; ``row`` indexes `results`
    crossfit_images: pd.DataFrame  # per-image counts of the crossfit selection (index: stem)
    fold_of: dict[str, int] = field(default_factory=dict)  # stem -> CV fold

    @property
    def crossfit_summary(self) -> dict[str, float]:
        return summary(np.sum([f.counts for f in self.folds], axis=0))

    @property
    def best_row(self) -> int:
        """In-sample best grid point (ties: first in grid order)."""
        return int(self.results["pq"].to_numpy().argmax())

    def best_params(self) -> InstanceParams:
        return self.params(self.best_row)

    def params(self, row: int) -> InstanceParams:
        fields = list(InstanceParams.model_fields)
        return InstanceParams.model_validate(self.results.loc[row, fields].to_dict())

    def report(self) -> dict[str, object]:
        """Crossfit and in-sample numbers for crossfit.json."""
        return {
            "crossfit": self.crossfit_summary,
            "in_sample": self.results.loc[self.best_row, ["pq", "tp", "fp", "fn"]].to_dict(),
            "in_sample_params": self.best_params().model_dump(),
            "folds": [
                {
                    "fold": f.fold,
                    "n_images": sum(v == f.fold for v in self.fold_of.values()),
                    "params": self.params(f.row).model_dump(),
                    "train_pq": f.train_pq,
                    **summary(f.counts),
                }
                for f in self.folds
            ],
        }


def tune(
    prob_dir: Path,
    gt_df: pd.DataFrame,
    grid: TuneGrid,
    fold_of: Mapping[str, int],
    *,
    workers: int = 8,
) -> TuneResult:
    """Evaluate every grid point on all maps in ``prob_dir``, in-sample and crossfit.

    Results rows keep the grid order; fields missing from the grid keep their `InstanceParams`
    default. ``fold_of`` maps each image stem to its CV fold.
    """
    gts = gt_by_stem(gt_df)
    stems = sorted(p.stem for p in prob_dir.glob("*.png") if p.stem in gts)
    log.info("tuning on %d out-of-fold maps", len(stems))
    fold_idx = np.array([fold_of[s] for s in stems], dtype=np.intp)
    n_folds = int(fold_idx.max()) + 1
    filters = {k: v for k, v in grid.model_dump().items() if k not in COMPONENT_PARAMS}
    rows: list[dict[str, float]] = []
    fold_counts: list[np.ndarray] = []
    tables: dict[tuple[float, int], CandidateTable] = {}
    for t_low, close in itertools.product(grid.t_low, grid.close_radius):
        fn = partial(
            build_candidates,
            prob_dir=prob_dir,
            t_low=t_low,
            close_radius=close,
            min_area=min(grid.min_area),
            min_peak=min(grid.t_high),
        )
        with ProcessPoolExecutor(max_workers=workers) as pool:
            cands = list(
                tqdm(
                    pool.map(fn, stems, [gts[s] for s in stems]),
                    total=len(stems),
                    desc=f"t_low={t_low} close={close}",
                )
            )
        table = tables[t_low, close] = CandidateTable.from_images(cands)
        for values in itertools.product(*filters.values()):
            params = InstanceParams(
                t_low=t_low, close_radius=close, **dict(zip(filters, values, strict=True))
            )
            if params.t_high < t_low:
                continue
            per_image = table.image_counts(table.select(params))
            per_fold = np.zeros((n_folds, 4))
            np.add.at(per_fold, fold_idx, per_image)
            fold_counts.append(per_fold)
            rows.append({**params.model_dump(), **summary(per_fold.sum(axis=0))})
    res = TuneResult(
        pd.DataFrame(rows),
        crossfit(np.stack(fold_counts)),
        counts_frame([], np.zeros((0, 4))),
        dict(zip(stems, fold_idx.tolist(), strict=True)),
    )
    per_image = np.zeros((len(stems), 4))
    for c in res.folds:
        p = res.params(c.row)
        table = tables[p.t_low, p.close_radius]
        held_out = fold_idx == c.fold
        per_image[held_out] = table.image_counts(table.select(p))[held_out]
    res.crossfit_images = counts_frame(stems, per_image)
    return res
