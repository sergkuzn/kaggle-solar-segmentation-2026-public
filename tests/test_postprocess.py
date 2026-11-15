from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from conftest import blob
from filseg.competition import rle
from filseg.competition.metric import evaluate
from filseg.competition.submission import build_submission
from filseg.config import InstanceParams, TuneGrid
from filseg.postprocess.crossfit import crossfit, pq_from_counts
from filseg.postprocess.instances import extract_instances
from filseg.postprocess.tune import COMPONENT_PARAMS, build_candidates, gt_by_stem, pq_of, tune


def test_hysteresis_requires_peak() -> None:
    prob = np.zeros((2048, 2048), np.float32)
    prob[blob(400, 400, 60, 20)] = 0.4  # weak everywhere -> dropped
    prob[blob(1200, 1200, 60, 20)] = 0.4
    prob[blob(1200, 1200, 10, 5)] = 0.9  # strong core -> whole component kept
    out = extract_instances(
        prob, InstanceParams(t_low=0.3, t_high=0.6, close_radius=0, min_area=10)
    )
    assert len(out) == 1
    assert rle.area(out[0]) == int(blob(1200, 1200, 60, 20).sum())


def test_tune_fast_path_matches_evaluate(tmp_path: Path) -> None:
    stem = "20150101000000Bh"
    prob = np.zeros((2048, 2048), np.float32)
    gt_masks = [blob(500, 500, 70, 20), blob(1300, 800, 40, 90)]
    prob[blob(505, 495, 70, 22)] = 0.8
    prob[blob(1300, 800, 40, 60)] = 0.55
    prob[blob(1700, 1700, 15, 15)] = 0.9  # false positive
    cv2.imwrite(str(tmp_path / f"{stem}.png"), (prob * 255).astype(np.uint8))
    gt = pd.DataFrame(
        [
            (f"0101{a:02d}-{stem}_{k}", rle.encode(m))
            for a in (1, 2)
            for k, m in enumerate(gt_masks)
        ],
        columns=["filament_id", "segmentation_rle"],
    )
    for params in [
        InstanceParams(t_low=0.3, t_high=0.5, close_radius=0, min_area=10),
        InstanceParams(t_low=0.3, t_high=0.6, close_radius=2, min_area=10),
    ]:
        cands = build_candidates(
            stem,
            gt_by_stem(gt)[stem],
            prob_dir=tmp_path,
            t_low=params.t_low,
            close_radius=params.close_radius,
            min_area=0,
            min_peak=0.0,
        )
        quantized = (prob * 255).astype(np.uint8).astype(np.float32) / 255
        full = evaluate(gt, build_submission([(stem, extract_instances(quantized, params))]))
        assert pq_of([cands], params)["pq"] == pytest.approx(full.pq)


def test_tune_grid_searches_instance_params() -> None:
    # `tune` builds InstanceParams from the grid's fields and needs the component-shaping ones.
    grid, params = set(TuneGrid.model_fields), set(InstanceParams.model_fields)
    assert set(COMPONENT_PARAMS) <= grid <= params


def _write_toy_maps(tmp_path: Path, stems: list[str]) -> pd.DataFrame:
    """One map per stem with a good, a weak and a spurious blob; GT from two annotators."""
    rows = []
    for i, stem in enumerate(stems):
        prob = np.zeros((2048, 2048), np.float32)
        good, weak = blob(500, 500 + 40 * i, 70, 20), blob(1300, 800, 40, 90)
        prob[good] = 0.8
        prob[blob(1300, 800, 40, 50 + 10 * i)] = 0.45 + 0.1 * (i % 2)
        prob[blob(1700, 1700, 15, 15)] = 0.6 + 0.1 * i  # false positive
        cv2.imwrite(str(tmp_path / f"{stem}.png"), (prob * 255).astype(np.uint8))
        rows += [
            (f"0101{a:02d}-{stem}_{k}", rle.encode(m))
            for a in (1, 2)
            for k, m in enumerate([good, weak])
        ]
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def test_tune_crossfit_scores_each_fold_with_params_from_the_others(tmp_path: Path) -> None:
    stems = [f"2015010{i}000000Bh" for i in range(4)]
    gt = _write_toy_maps(tmp_path, stems)
    grid = TuneGrid(
        t_low=[0.3, 0.5],
        close_radius=[0],
        t_high=[0.5, 0.65, 0.75],
        min_area=[10],
        min_score=[0.0],
    )
    fold_of = {s: i % 2 for i, s in enumerate(stems)}
    res = tune(tmp_path, gt, grid, fold_of, workers=1)
    # in-sample rows agree with the per-image reference implementation
    for row in range(len(res.results)):
        params = res.params(row)
        cands = [
            build_candidates(
                s,
                gt_by_stem(gt)[s],
                prob_dir=tmp_path,
                t_low=params.t_low,
                close_radius=params.close_radius,
                min_area=0,
                min_peak=0.0,
            )
            for s in stems
        ]
        assert res.results.loc[row, "pq"] == pytest.approx(pq_of(cands, params)["pq"])
    # crossfit: per-image counts pool to the reported number, and each fold's choice is the best
    # grid point on the other fold
    pooled = res.crossfit_images.to_numpy(np.float64).sum(axis=0)
    assert pq_from_counts(pooled) == pytest.approx(res.crossfit_summary["pq"])
    assert res.crossfit_summary["pq"] <= res.results["pq"].max() + 1e-12
    for f in res.folds:
        other = [s for s in stems if fold_of[s] != f.fold]
        pqs = []
        for row in range(len(res.results)):
            p = res.params(row)
            cands = [
                build_candidates(
                    s,
                    gt_by_stem(gt)[s],
                    prob_dir=tmp_path,
                    t_low=p.t_low,
                    close_radius=p.close_radius,
                    min_area=0,
                    min_peak=0.0,
                )
                for s in other
            ]
            pqs.append(pq_of(cands, p)["pq"])
        assert f.train_pq == pytest.approx(max(pqs))
        assert f.row == int(np.argmax(pqs))


def test_crossfit_picks_best_on_other_folds() -> None:
    # 2 grid points x 2 folds; point 0 wins fold 0 only, point 1 wins fold 1 only
    c = np.zeros((2, 2, 4))
    c[0, 0] = c[1, 1] = [10, 0, 0, 9]  # PQ 0.9
    c[0, 1] = c[1, 0] = [10, 0, 0, 5]  # PQ 0.5
    choices = crossfit(c)
    assert [ch.row for ch in choices] == [1, 0]  # fold 0 is scored by the winner of fold 1
    assert [pq_from_counts(ch.counts) for ch in choices] == pytest.approx([0.5, 0.5])
