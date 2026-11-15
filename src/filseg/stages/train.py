"""train: fold models, their histories, OOF probability maps and CV metrics."""

import logging
from collections.abc import Sequence

import pandas as pd
import torch

from filseg.config import ExperimentConfig, dump_config
from filseg.infra.paths import Paths
from filseg.model.predictor import Predictor, pick_device, write_prob_maps
from filseg.stages.common import write_json
from filseg.training.fit import FoldResult, fit_fold

log = logging.getLogger(__name__)


def train_fold(cfg: ExperimentConfig, paths: Paths, fold: int, device: torch.device) -> FoldResult:
    """Fit one fold (history -> CSV), then write its out-of-fold probability maps."""
    exp = paths.experiment(cfg.name)
    exp.fold(fold).mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(paths.folds_csv)
    train_stems = folds.loc[folds.fold != fold, "stem"].tolist()
    val_stems = folds.loc[folds.fold == fold, "stem"].tolist()[: cfg.train.val_limit]
    log.info("fold %d: %d train / %d val images", fold, len(train_stems), len(val_stems))

    history: list[dict[str, float]] = []

    def on_epoch(row: dict[str, float]) -> None:
        history.append(row)
        pd.DataFrame(history).to_csv(exp.history(fold), index=False)

    result = fit_fold(
        cfg,
        paths,
        fold,
        train_stems=train_stems,
        val_stems=val_stems,
        gt_df=pd.read_csv(paths.gt_csv),
        device=device,
        on_epoch=on_epoch,
    )
    log.info("fold %d best val PQ %.4f -> writing OOF maps", fold, result.best_pq)
    predictor = Predictor.from_experiment(cfg, device, [fold], paths)
    write_prob_maps(predictor, paths.train_images, val_stems, exp.oof)
    return result


def run(cfg: ExperimentConfig, paths: Paths, *, folds: Sequence[int] | None) -> None:
    exp = paths.experiment(cfg.name)
    dump_config(cfg, exp.config)
    device = pick_device()
    log.info("device: %s", device)
    folds = list(folds or cfg.train_folds)
    results = [train_fold(cfg, paths, f, device) for f in folds]
    metrics = {f"fold{f}_best_val_pq": r.best_pq for f, r in zip(folds, results, strict=True)} | {
        "mean_best_val_pq": sum(r.best_pq for r in results) / len(results)
    }
    write_json(exp.cv_metrics, metrics)
