"""Learned keep/drop decision over connected components (``scorer.mode: flat``).

Candidates are the components of one threshold (`components` at the tuned ``t_low`` /
``close_radius``) of at least ``min_area`` px. A LightGBM regressor predicts each candidate's
*expected matched IoU*: the sum over the image's annotator records of the IoU with the GT filament
it matches (IoU > 0.5, else 0), divided by the number of records. Keeping a candidate adds that sum
to the PQ numerator and exactly ``n_records / 2`` to the denominator (a TP turns an FN into a TP,
an FP adds an FP), so it pays off when its score exceeds about PQ / 2. The cutoff ``tau`` is chosen
on PQ directly.

Honest numbers come from nested crossfit (`crossfit_scorer`): the model scoring fold k is trained
on the other folds, and fold k's tau is chosen on scores of those folds that are themselves
out-of-fold (inner models trained without fold k and without the scored fold).

Features (`FEATURE_NAMES`) use only the final probability map and the image pixels, so they exist
on OOF maps (one model) and test maps (mean of five) alike: 18 shape / intensity / context
features of the blob (`candidate_features`), plus 4 that describe how the blob grows and merges
with its neighbours at the lower thresholds 0.2 / 0.25 (`join_features`).

The model is a `ScorerModel`, a bag of ``scorer.n_bags`` boosters with seeds ``seed + b`` whose
scores are averaged; ``scorer.monotone`` constrains the score to rise or fall with chosen features.
"""

import logging
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from filseg.competition import rle
from filseg.config import ScorerConfig
from filseg.imaging import read_gray, read_prob
from filseg.postprocess.crossfit import counts_frame, pq_from_counts, summary
from filseg.postprocess.instances import Component, components
from filseg.postprocess.tune import CandidateTable, ImageCandidates
from filseg.preprocess.disk import Disk, locate_disk

if TYPE_CHECKING:
    import lightgbm as lgb

log = logging.getLogger(__name__)

BASE_FEATURES = (
    "area",
    "p_mean",
    "p_max",
    "p_std",
    "p_q10",
    "p_q50",
    "p_q90",
    "ring_p",  # mean p in a thin ring around the blob
    "gray_rel",  # mean intensity inside / median intensity of the disk
    "gray_contrast",  # (ring intensity - inside intensity) / ring intensity; filaments are dark
    "major",  # axis lengths of the moment ellipse (px)
    "minor",
    "elongation",
    "bbox_fill",
    "r_disk",  # centroid distance from the disk centre / disk radius
    "n_cand",  # candidates in the image
    "n_strong",  # candidates in the image with max p >= 0.5
    "nn_dist",  # centroid distance to the nearest other candidate (px)
)
JOIN_FEATURES = (
    "grow_20",  # area of the blob's component at p >= 0.2 / its area
    "grow_25",  # same at 0.25
    "n_join_25",  # other candidates inside the same 0.25 component
    "partner_25",  # area of the largest of them / the blob's area (0 without one)
)
FEATURE_NAMES = BASE_FEATURES + JOIN_FEATURES
_GROW_LEVELS = (0.2, 0.25)
_RING = 7
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * _RING + 1,) * 2)
_NO_NEIGHBOUR = 2048.0


class CandidateSettings(BaseModel):
    t_low: float
    close_radius: int
    min_area: int


class ScorerParams(CandidateSettings):
    """Everything inference needs besides the model; saved as ``scorer/params.json``."""

    tau: float
    features: list[str] = list(FEATURE_NAMES)
    n_bags: int = 1


@dataclass
class Candidates:
    labels: np.ndarray  # int32 label map
    comps: list[Component]  # the components passing the area pre-filter
    features: np.ndarray  # (n, len(FEATURE_NAMES))

    def rles(self, sel: np.ndarray | None = None) -> list[str]:
        idx = range(len(self.comps)) if sel is None else np.flatnonzero(sel)
        return [rle.encode(self.labels == self.comps[i].label) for i in idx]


def _disk_median(gray: np.ndarray, disk: Disk) -> float:
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    inside = (xx - disk.cx) ** 2 + (yy - disk.cy) ** 2 < (0.95 * disk.r) ** 2
    return float(np.median(gray[inside])) if inside.any() else 1.0


def candidate_features(
    prob: np.ndarray, gray: np.ndarray, labels: np.ndarray, comps: Sequence[Component]
) -> np.ndarray:
    """(n_comps, len(BASE_FEATURES)) features of ``comps`` (labels in ``labels``)."""
    n = len(comps)
    out = np.zeros((n, len(BASE_FEATURES)))
    if n == 0:
        return out
    disk = locate_disk(gray)
    disk_med = max(_disk_median(gray, disk), 1.0)
    h, w = prob.shape
    cents = np.zeros((n, 2))
    for i, c in enumerate(comps):
        x, y, bw, bh = c.bbox
        x0, y0 = max(x - _RING, 0), max(y - _RING, 0)
        x1, y1 = min(x + bw + _RING, w), min(y + bh + _RING, h)
        m = labels[y0:y1, x0:x1] == c.label
        m8 = m.astype(np.uint8)
        ring = cv2.dilate(m8, _KERNEL).astype(bool) & ~m
        p = prob[y0:y1, x0:x1]
        g = gray[y0:y1, x0:x1].astype(np.float64)
        pin, gin = p[m], g[m]
        mo = cv2.moments(m8, binaryImage=True)
        cx, cy = x0 + mo["m10"] / mo["m00"], y0 + mo["m01"] / mo["m00"]
        cov = np.array([[mo["mu20"], mo["mu11"]], [mo["mu11"], mo["mu02"]]]) / mo["m00"]
        lo, hi = np.clip(np.linalg.eigvalsh(cov), 0, None)
        major, minor = 4 * np.sqrt(hi), 4 * np.sqrt(lo)
        ring_g = float(g[ring].mean()) if ring.any() else float(gin.mean())
        q10, q50, q90 = np.quantile(pin, [0.1, 0.5, 0.9])
        cents[i] = cx, cy
        out[i, :15] = (
            c.area,
            pin.mean(),
            pin.max(),
            pin.std(),
            q10,
            q50,
            q90,
            float(p[ring].mean()) if ring.any() else 0.0,
            gin.mean() / disk_med,
            (ring_g - gin.mean()) / max(ring_g, 1.0),
            major,
            minor,
            major / max(minor, 1.0),
            c.area / (bw * bh),
            np.hypot(cx - disk.cx, cy - disk.cy) / disk.r,
        )
    d = np.hypot(*(cents[:, None, :] - cents[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    out[:, 15] = n
    out[:, 16] = sum(c.max_prob >= 0.5 for c in comps)
    out[:, 17] = np.minimum(d.min(axis=1), _NO_NEIGHBOUR)
    return out


def join_features(
    prob: np.ndarray, labels: np.ndarray, comps: Sequence[Component], close_radius: int
) -> np.ndarray:
    """(n_comps, len(JOIN_FEATURES)) from the probability map alone: the components at the lower
    thresholds `_GROW_LEVELS` (same closing) that contain each candidate."""
    out = np.zeros((len(comps), len(JOIN_FEATURES)))
    if not comps:
        return out
    h, w = prob.shape
    lows = {t: components(prob, t_low=t, close_radius=close_radius) for t in _GROW_LEVELS}
    # one pixel of each candidate locates the lower-threshold component containing it
    first: list[tuple[int, int]] = []
    for c in comps:
        x, y, bw, bh = c.bbox
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h)
        m = labels[y0:y1, x0:x1] == c.label
        dy, dx = np.unravel_index(np.argmax(m), m.shape)
        first.append((y0 + int(dy), x0 + int(dx)))
    lab25 = np.array([lows[0.25][0][f] for f in first], dtype=np.intp)  # 0: none
    areas = np.array([c.area for c in comps], dtype=np.float64)
    for i, c in enumerate(comps):
        row: list[float] = []
        for t in _GROW_LEVELS:
            lab_t, comps_t = lows[t]
            v = int(lab_t[first[i]])
            row.append((comps_t[v - 1].area if v > 0 else c.area) / c.area)
        same = (lab25 == lab25[i]) & (lab25 > 0)
        same[i] = False
        row += [float(same.sum()), float(areas[same].max() / c.area) if same.any() else 0.0]
        out[i] = row
    return out


def candidates(prob: np.ndarray, gray: np.ndarray, s: CandidateSettings) -> Candidates:
    labels, comps = components(prob, t_low=s.t_low, close_radius=s.close_radius)
    comps = [c for c in comps if c.area >= s.min_area]
    feats = np.hstack(
        [
            candidate_features(prob, gray, labels, comps),
            join_features(prob, labels, comps, s.close_radius),
        ]
    )
    return Candidates(labels, comps, feats)


# --- training data ----------------------------------------------------------------------------


@dataclass
class TrainingSet:
    """Candidates of the OOF maps with features and what each earns (`CandidateTable`)."""

    stems: list[str]
    table: CandidateTable
    X: np.ndarray  # (n_cand, len(FEATURE_NAMES))
    n_records: np.ndarray  # (n_images,)

    @property
    def target(self) -> np.ndarray:
        """Expected matched IoU per record."""
        out: np.ndarray = self.table.gain[:, 2] / self.n_records[self.table.image]
        return out

    @property
    def weight(self) -> np.ndarray:
        out: np.ndarray = self.n_records[self.table.image]
        return out

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.X, columns=list(FEATURE_NAMES))
        df.insert(0, "stem", [self.stems[i] for i in self.table.image])
        df["target"] = self.target
        df["n_records"] = self.weight
        return df


def _training_image(
    stem: str,
    gt: Mapping[str, Sequence[str]],
    *,
    prob_dir: Path,
    image_dir: Path,
    settings: CandidateSettings,
) -> tuple[ImageCandidates, np.ndarray]:
    c = candidates(
        read_prob(prob_dir / f"{stem}.png"), read_gray(image_dir / f"{stem}.jpeg"), settings
    )
    rles = c.rles()
    ious = {rid: rle.iou_matrix(g, rles) for rid, g in gt.items()}
    return ImageCandidates(stem, c.comps, ious), c.features


def build_training_set(
    prob_dir: Path,
    image_dir: Path,
    gts: Mapping[str, Mapping[str, Sequence[str]]],
    settings: CandidateSettings,
    *,
    workers: int = 8,
) -> TrainingSet:
    """Candidates of every map in ``prob_dir`` with GT (``gts``: `tune.gt_by_stem`)."""
    stems = sorted(p.stem for p in prob_dir.glob("*.png") if p.stem in gts)
    fn = partial(_training_image, prob_dir=prob_dir, image_dir=image_dir, settings=settings)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        res = list(
            tqdm(pool.map(fn, stems, [gts[s] for s in stems]), total=len(stems), desc="cands")
        )
    return TrainingSet(
        stems,
        CandidateTable.from_images([r[0] for r in res]),
        np.concatenate([np.zeros((0, len(FEATURE_NAMES))), *(r[1] for r in res)]),
        np.array([len(gts[s]) for s in stems], dtype=np.float64),
    )


# --- model --------------------------------------------------------------------------------------


def fit_booster(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, cfg: ScorerConfig, *, seed: int = 0
) -> "lgb.Booster":
    import lightgbm as lgb

    params: dict[str, object] = {
        "objective": "regression",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_child_samples,
        "bagging_fraction": cfg.subsample,
        "bagging_freq": 1,
        "feature_fraction": cfg.colsample,
        "seed": seed,
        "deterministic": True,
        "verbose": -1,
    }
    if cfg.monotone:
        unknown = set(cfg.monotone) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"monotone constraints on unknown features {sorted(unknown)}")
        params["monotone_constraints"] = [cfg.monotone.get(n, 0) for n in FEATURE_NAMES]
    data = lgb.Dataset(X, y, weight=w, feature_name=list(FEATURE_NAMES))
    return lgb.train(params, data, num_boost_round=cfg.n_estimators)


def _bag_file(path: Path, bag: int) -> Path:
    """File of one booster: ``path`` itself for bag 0, else ``<stem>_b<bag>_h0.txt`` next to it
    (the layout of the released scorer files)."""
    return path if bag == 0 else path.with_name(f"{path.stem}_b{bag}_h0.txt")


@dataclass
class ScorerModel:
    """Bagged boosters; the score is the mean of their predictions."""

    bags: list["lgb.Booster"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        out: np.ndarray = np.mean(
            [np.asarray(b.predict(X), dtype=np.float64) for b in self.bags], axis=0
        )
        return out

    def feature_importance(self) -> np.ndarray:
        """Total gain, summed over bags."""
        out: np.ndarray = np.sum([b.feature_importance("gain") for b in self.bags], axis=0)
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        for i, booster in enumerate(self.bags):
            booster.save_model(str(_bag_file(path, i)))

    @classmethod
    def load(cls, path: Path, *, n_bags: int) -> "ScorerModel":
        import lightgbm as lgb

        return cls([lgb.Booster(model_file=str(_bag_file(path, b))) for b in range(n_bags)])


def fit_model(
    X: np.ndarray, gain: np.ndarray, n_records: np.ndarray, cfg: ScorerConfig, *, seed: int = 0
) -> ScorerModel:
    """``cfg.n_bags`` boosters (seeds ``seed + b``) on candidates with `CandidateTable` ``gain``
    rows and their image's record count ``n_records``: target = summed matched IoU / records,
    weight = records."""
    y = gain[:, 2] / n_records
    return ScorerModel(
        [fit_booster(X, y, n_records, cfg, seed=seed + b) for b in range(cfg.n_bags)]
    )


def best_tau(
    ts: TrainingSet, scores: np.ndarray, images: np.ndarray, grid: Sequence[float]
) -> tuple[float, float]:
    """Cutoff in ``grid`` with the best pooled PQ over ``images`` (bool mask); ties: lowest."""
    pqs = [
        float(pq_from_counts(ts.table.image_counts(scores >= t)[images].sum(axis=0))) for t in grid
    ]
    i = int(np.argmax(pqs))
    return float(grid[i]), pqs[i]


@dataclass
class ScorerFold:
    fold: int
    tau: float  # chosen on inner out-of-fold scores of the other folds
    train_pq: float  # its PQ there
    counts: np.ndarray  # (4,) counts on this fold


@dataclass
class ScorerCrossfit:
    folds: list[ScorerFold]
    models: list[ScorerModel]  # outer model per fold (trained on the other folds)
    scores: np.ndarray  # (n_cand,) outer out-of-fold scores
    tau: float  # cutoff for the final model: best on all outer out-of-fold scores
    crossfit_images: pd.DataFrame  # per-image counts of the crossfit selection
    mse: float  # record-weighted squared error of ``scores`` against the target

    @property
    def summary(self) -> dict[str, float]:
        return summary(np.sum([f.counts for f in self.folds], axis=0))


def crossfit_scorer(
    ts: TrainingSet, fold_idx: np.ndarray, cfg: ScorerConfig, *, seed: int = 0
) -> ScorerCrossfit:
    """Nested crossfit (see module docstring). ``fold_idx``: CV fold of each image."""
    row_fold = fold_idx[ts.table.image]
    y, w = ts.target, ts.weight
    folds = sorted(set(fold_idx.tolist()))
    if len(folds) < 3:
        raise ValueError(f"nested crossfit needs at least 3 folds, got {len(folds)}")

    def oof(train: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, ScorerModel]:
        m = fit_model(ts.X[train], ts.table.gain[train], w[train], cfg, seed=seed)
        return m.predict(ts.X[score]), m

    outer = np.full(len(y), np.nan)
    per_image = np.zeros((len(ts.stems), 4))
    choices: list[ScorerFold] = []
    models: list[ScorerModel] = []
    for k in tqdm(folds, desc="crossfit"):
        inner = np.full(len(y), np.nan)
        for j in folds:
            if j != k:
                inner[row_fold == j], _ = oof((row_fold != k) & (row_fold != j), row_fold == j)
        tau, train_pq = best_tau(ts, inner, fold_idx != k, cfg.tau_grid)
        outer[row_fold == k], b = oof(row_fold != k, row_fold == k)
        held = fold_idx == k
        per_image[held] = ts.table.image_counts(outer >= tau)[held]
        choices.append(ScorerFold(k, tau, train_pq, per_image[held].sum(axis=0)))
        models.append(b)
        log.info("fold %d: tau %.2f (other folds %.4f) -> %.4f", k, tau, train_pq,
                 summary(choices[-1].counts)["pq"])  # fmt: skip
    tau, _ = best_tau(ts, outer, np.ones(len(ts.stems), bool), cfg.tau_grid)
    mse = float(np.sum(w * (outer - y) ** 2) / np.sum(w))
    return ScorerCrossfit(choices, models, outer, tau, counts_frame(ts.stems, per_image), mse)


# --- inference ----------------------------------------------------------------------------------


class Scorer:
    """Trained model + `ScorerParams`: probability map and image -> disjoint instances."""

    def __init__(self, model: ScorerModel, params: ScorerParams) -> None:
        if tuple(params.features) != FEATURE_NAMES:
            raise ValueError("scorer was trained on a different feature list")
        if len(model.bags) != params.n_bags:
            raise ValueError("model does not match params (n_bags)")
        self.model = model
        self.params = params

    @classmethod
    def load(cls, model: Path, params: Path) -> "Scorer":
        p = ScorerParams.model_validate_json(params.read_text())
        return cls(ScorerModel.load(model, n_bags=p.n_bags), p)

    def save(self, model: Path, params: Path) -> None:
        self.model.save(model)
        params.write_text(self.params.model_dump_json(indent=2))

    def extract(self, prob: np.ndarray, gray: np.ndarray) -> list[str]:
        """RLEs of the candidates scoring >= tau; they are disjoint (components of one map)."""
        c = candidates(prob, gray, self.params)
        return c.rles(self.model.predict(c.features) >= self.params.tau)
