"""download, extract and prepare: the competition data and the training targets."""

import logging

from filseg.competition.coco import load_annotations
from filseg.competition.download import download_archive, extract_archive
from filseg.data.folds import make_folds
from filseg.data.targets import prepare_targets
from filseg.infra.paths import Paths

log = logging.getLogger(__name__)


def download(paths: Paths, *, force: bool) -> None:
    download_archive(paths, force=force)


def extract(paths: Paths) -> None:
    extract_archive(paths)


def prepare(
    paths: Paths, *, n_folds: int, seed: int, spine_thickness: int, workers: int, force: bool
) -> None:
    prepare_targets(paths, spine_thickness=spine_thickness, workers=workers, force=force)
    if force or not paths.folds_csv.exists():
        folds = make_folds(load_annotations(paths.train_json), n_folds=n_folds, seed=seed)
        folds.to_csv(paths.folds_csv, index=False)
        log.info("folds: %s", folds.fold.value_counts().sort_index().to_dict())
