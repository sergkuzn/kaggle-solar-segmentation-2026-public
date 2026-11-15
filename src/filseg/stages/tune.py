"""tune: grid search of instance extraction on the OOF maps -> post_params.json, plus the crossfit
PQ of the search (crossfit.json) and its per-image counts (crossfit_images.csv).

The scorer (``fit-scorer``) takes its candidate threshold ``t_low`` and ``close_radius`` from
post_params.json.
"""

import logging

import pandas as pd

from filseg.config import ExperimentConfig
from filseg.infra.paths import Paths
from filseg.postprocess.crossfit import summary
from filseg.postprocess.tune import tune
from filseg.stages.common import console, write_json

log = logging.getLogger(__name__)


def run(cfg: ExperimentConfig, paths: Paths, *, workers: int) -> None:
    exp = paths.experiment(cfg.name)
    folds = pd.read_csv(paths.folds_csv)
    fold_of = dict(zip(folds["stem"], folds["fold"].astype(int), strict=True))
    res = tune(exp.oof, pd.read_csv(paths.gt_csv), cfg.tune, fold_of, workers=workers)
    results = res.results.sort_values("pq", ascending=False, kind="stable")
    results.to_csv(exp.tune_results, index=False)
    best = res.best_params()
    exp.post_params.write_text(best.model_dump_json(indent=2))
    write_json(exp.crossfit, res.report())
    res.crossfit_images.to_csv(exp.crossfit_images)
    console.print(results.head(10).to_string())
    for f in res.folds:
        console.print(
            f"fold {f.fold}: {res.params(f.row)}  other folds {f.train_pq:.4f}"
            f" -> this fold {summary(f.counts)['pq']:.4f}"
        )
    log.info("in-sample OOF PQ %.4f with %s", results.iloc[0]["pq"], best)
    log.info("crossfit OOF PQ %.4f", res.crossfit_summary["pq"])
