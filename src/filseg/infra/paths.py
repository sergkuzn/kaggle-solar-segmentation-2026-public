"""Filesystem layout of data and artifacts.

Everything heavy lives outside git: ``data/`` (raw competition files + prepared targets) and
``outputs/`` (checkpoints, probability maps, submissions). Both roots can be relocated with the
``FILSEG_DATA_DIR`` / ``FILSEG_OUTPUT_DIR`` environment variables, e.g. onto a GPU box's fast
scratch disk.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

COMPETITION = "filament-segmentation-2026"
DATASET_DIRNAME = "MAGFiLO_1.0_Kaggle_2026"
TRAIN_JSON_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ScorerPaths:
    """Outputs of ``fit-scorer``: ``params.json`` (tau and candidate settings), the final
    LightGBM models trained on all images (``model.txt`` for bag 0, ``model_b<bag>_h0.txt`` for
    the other bags), the crossfit models ``fold<k>[_b<bag>_h0].txt``, ``candidates.csv``,
    ``crossfit.json`` and ``crossfit_images.csv`` (`filseg.postprocess.scorer.ScorerModel`)."""

    dir: Path

    @property
    def model(self) -> Path:
        return self.dir / "model.txt"

    @property
    def params(self) -> Path:
        return self.dir / "params.json"

    def fold_model(self, k: int) -> Path:
        """Crossfit model that scored fold ``k`` (trained on the other folds)."""
        return self.dir / f"fold{k}.txt"

    @property
    def candidates(self) -> Path:
        """Training candidates: stem, features, target, n_records, fold, OOF score."""
        return self.dir / "candidates.csv"

    @property
    def crossfit(self) -> Path:
        return self.dir / "crossfit.json"

    @property
    def crossfit_images(self) -> Path:
        """Per-image PQ counts of the scorer's crossfit selection."""
        return self.dir / "crossfit_images.csv"


class ExperimentPaths(BaseModel):
    """Artifacts of one experiment in ``outputs/<name>/``.

    Each stage owns its own entries here, and they never overlap.
    """

    root: Path

    # train
    @property
    def config(self) -> Path:
        """Resolved config (defaults filled in) the models were trained with."""
        return self.root / "config.yaml"

    @property
    def folds(self) -> Path:
        return self.root / "folds"

    def fold(self, k: int) -> Path:
        return self.folds / f"fold{k}"

    def checkpoint(self, k: int, epoch: int, *, last: bool = False) -> Path:
        """``model_<epoch>.pt`` (best val PQ) or ``model_<epoch>_last.pt`` (final epoch)."""
        return self.fold(k) / f"model_{epoch}{'_last' if last else ''}.pt"

    def checkpoints(self, k: int) -> list[Path]:
        return sorted(self.fold(k).glob("model_*.pt"))

    def model(self, k: int) -> Path:
        """Best-validation-PQ checkpoint of fold ``k``, the one inference uses.

        Training keeps two: ``model_<best>.pt`` and ``model_<epochs>_last.pt``, or only the latter
        when the last epoch was the best. Returns ``model.pt`` when the fold has no checkpoint.
        """
        ckpts = self.checkpoints(k)
        best = [p for p in ckpts if not p.stem.endswith("_last")]
        return (best or ckpts or [self.fold(k) / "model.pt"])[0]

    def history(self, k: int) -> Path:
        return self.fold(k) / "history.csv"

    @property
    def oof(self) -> Path:
        """Out-of-fold probability maps (uint8 PNG), one per train image."""
        return self.root / "oof"

    @property
    def cv_metrics(self) -> Path:
        """Best validation PQ per fold."""
        return self.root / "cv_metrics.json"

    # write-maps
    @property
    def test_probs(self) -> Path:
        """Test probability maps (uint8 PNG): the fold ensemble's mean, input of ``predict``."""
        return self.root / "test_probs"

    # tune
    @property
    def tune_results(self) -> Path:
        return self.root / "tune_results.csv"

    @property
    def post_params(self) -> Path:
        return self.root / "post_params.json"

    @property
    def crossfit(self) -> Path:
        """Crossfit PQ of the tune grid: per fold, the grid point chosen on the other folds."""
        return self.root / "crossfit.json"

    @property
    def crossfit_images(self) -> Path:
        """Per-image PQ counts of the crossfit selection."""
        return self.root / "crossfit_images.csv"

    # blend
    @property
    def blend(self) -> Path:
        """``filseg blend`` record: member experiments and the map directories averaged."""
        return self.root / "blend.json"

    # fit-scorer
    @property
    def scorer(self) -> ScorerPaths:
        return ScorerPaths(self.root / "scorer")

    # predict
    @property
    def submission(self) -> Path:
        return self.root / "submission.csv"

    @property
    def submission_report(self) -> Path:
        return self.root / "submission_report.json"


class Paths(BaseModel):
    """Resolved locations of all pipeline inputs and outputs."""

    data_root: Path
    output_root: Path

    @property
    def raw(self) -> Path:
        return self.data_root / "raw"

    @property
    def archive(self) -> Path:
        """Competition zip as downloaded from Kaggle."""
        return self.raw / f"{COMPETITION}.zip"

    @property
    def dataset(self) -> Path:
        return self.raw / DATASET_DIRNAME

    @property
    def train_images(self) -> Path:
        return self.dataset / "train" / "train_images"

    @property
    def test_images(self) -> Path:
        return self.dataset / "test" / "test_images"

    @property
    def train_json(self) -> Path:
        return self.dataset / "train" / TRAIN_JSON_NAME

    @property
    def processed(self) -> Path:
        return self.data_root / "processed"

    @property
    def targets(self) -> Path:
        """Per-image soft filament / spine targets (uint8 PNG, 2048x2048)."""
        return self.processed / "targets"

    @property
    def input_cache(self) -> Path:
        """Preprocessed model inputs (float16 .npy), one sub-folder per channel/crop setup."""
        return self.processed / "input_cache"

    @property
    def gt_csv(self) -> Path:
        """Ground truth in the official evaluation format (one row per annotator filament)."""
        return self.processed / "gt.csv"

    @property
    def folds_csv(self) -> Path:
        return self.processed / "folds.csv"

    def experiment(self, name: str) -> ExperimentPaths:
        return ExperimentPaths(root=self.output_root / name)


def get_paths() -> Paths:
    """Build `Paths` from environment variables, defaulting to folders in the repo root."""
    data_root = Path(os.environ.get("FILSEG_DATA_DIR", REPO_ROOT / "data"))
    output_root = Path(os.environ.get("FILSEG_OUTPUT_DIR", REPO_ROOT / "outputs"))
    return Paths(data_root=data_root, output_root=output_root)
