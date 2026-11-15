"""write-maps: infer the OOF and/or test probability maps of a trained experiment.

OOF maps come from each image's own fold model (folds.csv), test maps from the mean of all fold
models, both with the config's ``infer`` settings (TTA). ``train`` writes the OOF maps of each
fold itself (the last step of `train.train_fold`); this stage re-infers them for existing
checkpoints, e.g. with another TTA or after ``fetch-weights``. Existing maps are overwritten.
"""

import logging
from enum import StrEnum

import pandas as pd

from filseg.config import ExperimentConfig
from filseg.infra.paths import Paths
from filseg.model.predictor import Predictor, pick_device, write_prob_maps
from filseg.stages.common import image_stems

log = logging.getLogger(__name__)


class Split(StrEnum):
    OOF = "oof"
    TEST = "test"
    BOTH = "both"


def run(cfg: ExperimentConfig, paths: Paths, *, split: Split) -> None:
    exp = paths.experiment(cfg.name)
    folds = cfg.train_folds
    missing = [f for f in folds if not exp.model(f).exists()]
    if missing:
        raise FileNotFoundError(f"no checkpoint for folds {missing} in {exp.folds}")
    device = pick_device()
    if split in (Split.OOF, Split.BOTH):
        fold_of = pd.read_csv(paths.folds_csv)
        for f in folds:
            stems = fold_of.loc[fold_of.fold == f, "stem"].tolist()[: cfg.train.val_limit]
            log.info("fold %d: %d OOF maps (tta=%s)", f, len(stems), cfg.infer.tta)
            predictor = Predictor.from_experiment(cfg, device, [f], paths)
            write_prob_maps(predictor, paths.train_images, stems, exp.oof)
    if split in (Split.TEST, Split.BOTH):
        stems = image_stems(paths.test_images)
        log.info("%d test maps, %d models (tta=%s)", len(stems), len(folds), cfg.infer.tta)
        predictor = Predictor.from_experiment(cfg, device, folds, paths)
        write_prob_maps(predictor, paths.test_images, stems, exp.test_probs)
