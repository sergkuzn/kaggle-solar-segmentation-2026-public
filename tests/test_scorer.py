"""The learned scorer (`filseg.postprocess.scorer`) on synthetic maps."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from conftest import blob
from filseg.competition import rle
from filseg.config import ExperimentConfig, ScorerConfig, ScorerMode
from filseg.postprocess.crossfit import pq_from_counts
from filseg.postprocess.scorer import (
    FEATURE_NAMES,
    CandidateSettings,
    Scorer,
    ScorerParams,
    build_training_set,
    candidates,
    crossfit_scorer,
    fit_booster,
    fit_model,
)
from filseg.postprocess.tune import gt_by_stem

SETTINGS = CandidateSettings(t_low=0.3, close_radius=0, min_area=10)
SMALL = ScorerConfig(
    n_estimators=20, num_leaves=4, min_child_samples=2, subsample=1.0, colsample=1.0
)


def _toy(tmp_path: Path, n: int) -> tuple[list[str], pd.DataFrame]:
    """Maps + dark-blob images: a GT filament (strong) and a junk blob (weak, no GT) each."""
    stems, rows = [], []
    for i in range(n):
        stem = f"2015010{i % 10}0000{i:02d}Bh"
        prob = np.zeros((2048, 2048), np.float32)
        gray = np.full((2048, 2048), 120, np.uint8)
        good = blob(600, 500 + 30 * i, 70, 20)
        prob[good] = 0.8
        gray[good] = 70
        prob[blob(1400, 1400, 20 + i, 15)] = 0.4 + 0.02 * i
        cv2.imwrite(str(tmp_path / f"{stem}.png"), (prob * 255).astype(np.uint8))
        cv2.imwrite(str(tmp_path / f"{stem}.jpeg"), gray)
        rows += [(f"0101{a:02d}-{stem}_0", rle.encode(good)) for a in (1, 2)]
        stems.append(stem)
    return stems, pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def _col(f: np.ndarray, name: str) -> np.ndarray:
    out: np.ndarray = f[:, FEATURE_NAMES.index(name)]
    return out


def test_mode_defaults_off() -> None:
    assert ExperimentConfig(name="x").scorer.mode == ScorerMode.OFF


def test_features_and_targets(tmp_path: Path) -> None:
    _, gt = _toy(tmp_path, 4)
    ts = build_training_set(tmp_path, tmp_path, gt_by_stem(gt), SETTINGS, workers=1)
    assert ts.X.shape == (8, len(FEATURE_NAMES))
    assert np.isfinite(ts.X).all()
    good = _col(ts.X, "p_max") > 0.7
    assert (ts.target[good] > 0.9).all() and (ts.target[~good] == 0).all()
    assert (ts.weight == 2).all()
    assert (_col(ts.X, "gray_contrast")[good] > 0.3).all(), "filaments are darker than their ring"


def test_join_features() -> None:
    s = SETTINGS.model_copy(update={"t_low": 0.35})
    gray = np.full((2048, 2048), 120, np.uint8)
    prob = np.zeros((2048, 2048), np.float32)
    prob[100:130, 100:140] = 0.4  # 1200 px
    prob[100:130, 200:240] = 0.4
    prob[110:120, 140:200] = 0.3  # bridge, 600 px: joins the two above at 0.25 / 0.2
    prob[500:530, 500:540] = 0.4  # isolated
    f = candidates(prob, gray, s).features
    assert f.shape == (3, len(FEATURE_NAMES))
    joined = np.flatnonzero(_col(f, "n_join_25") == 1)
    alone = np.flatnonzero(_col(f, "n_join_25") == 0)
    assert len(joined) == 2 and len(alone) == 1
    for i in joined:
        assert _col(f, "partner_25")[i] == pytest.approx(1.0)
        assert _col(f, "grow_25")[i] == pytest.approx(3000 / 1200)
        assert _col(f, "grow_20")[i] >= _col(f, "grow_25")[i]
    assert _col(f, "partner_25")[alone[0]] == 0.0
    assert _col(f, "grow_25")[alone[0]] == 1.0


def test_crossfit_scorer_keeps_filaments_and_drops_junk(tmp_path: Path) -> None:
    stems, gt = _toy(tmp_path, 9)
    ts = build_training_set(tmp_path, tmp_path, gt_by_stem(gt), SETTINGS, workers=1)
    fold_idx = np.array([i % 3 for i in range(len(stems))])
    cf = crossfit_scorer(ts, fold_idx, SMALL)
    pooled = cf.crossfit_images.to_numpy(np.float64).sum(axis=0)
    assert pq_from_counts(pooled) == pytest.approx(cf.summary["pq"])
    assert cf.summary["pq"] > 0.9
    assert cf.summary["fp"] == 0
    assert not np.isnan(cf.scores).any()


def test_single_bag_is_one_booster(tmp_path: Path) -> None:
    _, gt = _toy(tmp_path, 6)
    ts = build_training_set(tmp_path, tmp_path, gt_by_stem(gt), SETTINGS, workers=1)
    m = fit_model(ts.X, ts.table.gain, ts.weight, SMALL, seed=3)
    b = fit_booster(ts.X, ts.target, ts.weight, SMALL, seed=3)
    np.testing.assert_array_equal(m.predict(ts.X), b.predict(ts.X))


def test_bagged_scorer_roundtrip_and_disjoint_output(tmp_path: Path) -> None:
    stems, gt = _toy(tmp_path, 9)
    ts = build_training_set(tmp_path, tmp_path, gt_by_stem(gt), SETTINGS, workers=1)
    cfg = SMALL.model_copy(update={"n_bags": 3, "monotone": {"p_mean": 1}})
    cf = crossfit_scorer(ts, np.array([i % 3 for i in range(len(stems))]), cfg)
    y, w = ts.target, ts.weight
    assert cf.mse < np.average((y - np.average(y, weights=w)) ** 2, weights=w)
    m = fit_model(ts.X, ts.table.gain, ts.weight, cfg)
    params = ScorerParams(**SETTINGS.model_dump(), tau=cf.tau, n_bags=3)
    out = tmp_path / "scorer"
    Scorer(m, params).save(out / "model.txt", out / "params.json")
    assert sorted(p.name for p in out.glob("*.txt")) == [
        "model.txt",
        "model_b1_h0.txt",
        "model_b2_h0.txt",
    ]
    scorer = Scorer.load(out / "model.txt", out / "params.json")
    np.testing.assert_allclose(scorer.model.predict(ts.X), m.predict(ts.X))
    with pytest.raises(ValueError, match="n_bags"):
        Scorer(m, params.model_copy(update={"n_bags": 2}))

    prob = cv2.imread(str(tmp_path / f"{stems[0]}.png"), cv2.IMREAD_GRAYSCALE) / 255.0
    gray = cv2.imread(str(tmp_path / f"{stems[0]}.jpeg"), cv2.IMREAD_GRAYSCALE)
    assert len(candidates(prob, gray, SETTINGS).comps) == 2
    kept = scorer.extract(prob, gray)
    assert len(kept) == 1
    masks = [rle.decode(r) for r in kept]
    assert sum(m.sum() for m in masks) == np.logical_or.reduce(masks).sum()


def test_params_of_the_released_scorer_load() -> None:
    """The released params.json was written by the full research code: its extra keys (the
    options of other scorer variants, all at the values this code implements) are ignored."""
    released = {
        "t_low": 0.35,
        "close_radius": 2,
        "min_area": 20,
        "min_peak": 0.0,
        "mode": "flat",
        "levels": [],
        "dup_ratio": 1.0,
        "feature_groups": ["base", "profile_join"],
        "tau": 0.23,
        "features": list(FEATURE_NAMES),
        "target": "iou",
        "n_bags": 5,
    }
    p = ScorerParams.model_validate_json(json.dumps(released))
    assert (p.t_low, p.close_radius, p.min_area, p.tau, p.n_bags) == (0.35, 2, 20, 0.23, 5)
    assert tuple(p.features) == FEATURE_NAMES


def test_monotone_rejects_unknown_feature(tmp_path: Path) -> None:
    _, gt = _toy(tmp_path, 3)
    ts = build_training_set(tmp_path, tmp_path, gt_by_stem(gt), SETTINGS, workers=1)
    cfg = SMALL.model_copy(update={"monotone": {"nope": 1}})
    with pytest.raises(ValueError, match="unknown features"):
        fit_model(ts.X, ts.table.gain, ts.weight, cfg)
