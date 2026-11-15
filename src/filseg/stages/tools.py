"""evaluate and validate: standalone checks of submission-format CSVs."""

from pathlib import Path

import pandas as pd

from filseg.competition.metric import evaluate, evaluate_subset
from filseg.competition.submission import validate_submission
from filseg.infra.paths import Paths
from filseg.stages.common import console, image_stems, print_pq


def evaluate_csv(paths: Paths, predictions: Path, *, fold: int | None) -> None:
    gt, pred = pd.read_csv(paths.gt_csv), pd.read_csv(predictions)
    if fold is None:
        print_pq(predictions.name, evaluate(gt, pred))
    else:
        folds = pd.read_csv(paths.folds_csv)
        stems = set(folds.loc[folds.fold == fold, "stem"])
        print_pq(f"{predictions.name} fold {fold}", evaluate_subset(gt, pred, stems))


def validate(paths: Paths, submission: Path) -> None:
    report = validate_submission(pd.read_csv(submission), image_stems(paths.test_images))
    console.print(report.model_dump(exclude={"missing_images"}))
    console.print("[green]OK[/green]")
