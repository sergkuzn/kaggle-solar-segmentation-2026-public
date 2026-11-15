"""fit-scorer: learned keep/drop scorer on the OOF maps (`filseg.postprocess.scorer`).

Writes ``scorer/``: the final model (all images) with its tau (``params.json``, ``model*.txt``),
the crossfit models, the training candidates, the nested-crossfit PQ (crossfit.json) and its
per-image counts (crossfit_images.csv). Candidates use the tuned ``t_low`` / ``close_radius``
from post_params.json, so ``tune`` runs first. ``predict`` uses the scorer when
``scorer.mode: flat``.
"""

import logging

import numpy as np
import pandas as pd

from filseg.config import ExperimentConfig
from filseg.infra.paths import Paths
from filseg.postprocess.crossfit import summary
from filseg.postprocess.scorer import (
    FEATURE_NAMES,
    CandidateSettings,
    Scorer,
    ScorerParams,
    build_training_set,
    crossfit_scorer,
    fit_model,
)
from filseg.postprocess.tune import gt_by_stem
from filseg.stages.common import console, post_params, write_json

log = logging.getLogger(__name__)


def run(cfg: ExperimentConfig, paths: Paths, *, workers: int) -> None:
    exp = paths.experiment(cfg.name)
    out = exp.scorer
    base = post_params(cfg, paths)
    settings = CandidateSettings(
        t_low=base.t_low, close_radius=base.close_radius, min_area=cfg.scorer.min_area
    )
    folds = pd.read_csv(paths.folds_csv)
    fold_of = dict(zip(folds["stem"], folds["fold"].astype(int), strict=True))
    gts = gt_by_stem(pd.read_csv(paths.gt_csv))
    # candidates are built in worker processes before any LightGBM call: its OpenMP threads make
    # forking unsafe afterwards
    ts = build_training_set(exp.oof, paths.train_images, gts, settings, workers=workers)
    fold_idx = np.array([fold_of[s] for s in ts.stems], dtype=np.intp)
    log.info("%d candidates on %d images", len(ts.X), len(ts.stems))
    cf = crossfit_scorer(ts, fold_idx, cfg.scorer, seed=cfg.seed)
    final = fit_model(ts.X, ts.table.gain, ts.weight, cfg.scorer, seed=cfg.seed)
    params = ScorerParams(**settings.model_dump(), tau=cf.tau, n_bags=cfg.scorer.n_bags)
    Scorer(final, params).save(out.model, out.params)
    for f, m in zip(cf.folds, cf.models, strict=True):
        m.save(out.fold_model(f.fold))

    kept = cf.scores >= cf.tau
    cands = ts.frame()
    cands["fold"] = fold_idx[ts.table.image]
    cands["oof_score"] = cf.scores
    cands["kept"] = kept
    cands.to_csv(out.candidates, index=False)
    cf.crossfit_images.to_csv(out.crossfit_images)
    importance = final.feature_importance().round(1).tolist()
    write_json(
        out.crossfit,
        {
            "crossfit": cf.summary,
            "oof_mse": cf.mse,
            "tau": cf.tau,
            "n_candidates": len(ts.X),
            "n_images": len(ts.stems),
            "kept_per_image_oof": float(kept.sum() / len(ts.stems)),
            "folds": [
                {"fold": f.fold, "tau": f.tau, "train_pq": f.train_pq, **summary(f.counts)}
                for f in cf.folds
            ],
            "importance_gain": dict(
                sorted(zip(FEATURE_NAMES, importance, strict=True), key=lambda kv: -kv[1])
            ),
        },
    )
    for f in cf.folds:
        console.print(
            f"fold {f.fold}: tau {f.tau:.2f}  other folds {f.train_pq:.4f}"
            f" -> this fold {summary(f.counts)['pq']:.4f}"
        )
    console.print(
        f"scorer crossfit PQ {cf.summary['pq']:.4f}  final tau {cf.tau:.2f}"
        f"  OOF score MSE {cf.mse:.5f}"
    )
