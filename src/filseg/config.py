"""Experiment configuration: every schema and enum a YAML config validates against.

A leaf module (no ``filseg`` imports): code everywhere else reads its settings from here. Unknown
keys are errors, so a misspelt key never falls back to a default silently.
"""

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# No annotated filament lies further than 903 px from the image centre, so a centred 1888 crop
# (59 * 32, divisible by every encoder stride) keeps all labels with margin and avoids tiling.
DEFAULT_CROP = 1888


class Channel(StrEnum):
    """Input channels a model can be fed (stacked in the given order, see
    `filseg.preprocess.channels`)."""

    RAW = "raw"  # disk-standardized intensity
    FLAT = "flat"  # intensity minus a disk-aware 48 px Gaussian background (rim stays dark)


class CacheMode(StrEnum):
    OFF = "off"  # always compute on the fly
    READ = "read"  # use cached inputs when present, never write
    READ_WRITE = "read_write"  # use cached inputs, compute and store missing ones


class TTA(StrEnum):
    NONE = "none"
    FLIP = "flip"  # identity + horizontal flip
    D4 = "d4"  # all 8 dihedral transforms


class Section(BaseModel):
    """Base of every config schema: an unknown key (a typo) is an error, not silently dropped."""

    model_config = ConfigDict(extra="forbid")


class DataConfig(Section):
    channels: list[Channel] = [Channel.RAW, Channel.FLAT]
    crop: int = Field(DEFAULT_CROP, description="centred full-res crop fed at inference")
    train_crop: int = Field(768, description="random training window inside the crop")
    fg_prob: float = Field(0.7, ge=0, le=1, description="P(window centred on a filament pixel)")
    cache: CacheMode = Field(CacheMode.OFF, description="preprocessed-input disk cache")


class ModelConfig(Section):
    arch: str = "Unet"  # any segmentation_models_pytorch architecture name
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    spine_head: bool = True  # auxiliary output channel supervised by annotated spines


class AugmentConfig(Section):
    """Training augmentation of input / target windows (see `filseg.training.augment`)."""

    dihedral: bool = True  # random rotation by k * 90 degrees + horizontal flip
    gain: tuple[float, float] = (0.85, 1.15)  # per-channel input gain range
    offset_std: float = 0.1  # per-channel input offset (inputs are ~unit scale)


class TrainConfig(Section):
    epochs: int = 40
    steps_per_epoch: int = 200
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    spine_weight: float = 0.3  # loss = BCE(filament) + spine_weight * BCE(spine)
    grad_clip: float = 1.0
    amp: bool = True  # bf16 autocast on CUDA
    num_workers: int = 8
    val_every: int = 2
    val_limit: int | None = None  # debug only: validate / write OOF on the first N val images
    augment: AugmentConfig = AugmentConfig()


class InferConfig(Section):
    tta: TTA = TTA.FLIP


class InstanceParams(Section):
    """Connected-component extraction parameters (tuned on out-of-fold predictions)."""

    t_low: float = Field(0.35, ge=0, le=1, description="region-growing threshold")
    t_high: float = Field(0.5, ge=0, le=1, description="component must contain p >= t_high")
    close_radius: int = Field(2, ge=0, description="binary closing radius before labelling (px)")
    min_area: int = Field(150, ge=0, description="minimum component area (px at 2048^2)")
    min_score: float = Field(0.0, ge=0, le=1, description="minimum mean probability")


class TuneGrid(Section):
    """Values of each `InstanceParams` field searched by ``filseg tune``."""

    t_low: list[float] = [0.25, 0.35, 0.45]
    close_radius: list[int] = [0, 2, 4]
    t_high: list[float] = [0.4, 0.5, 0.6, 0.7]
    min_area: list[int] = [50, 150, 300, 500]
    min_score: list[float] = [0.0, 0.3, 0.4, 0.5]


class ScorerMode(StrEnum):
    """How `predict` selects instances among the connected components (`filseg.postprocess`)."""

    OFF = "off"  # the fixed `keep` rule with the tuned `InstanceParams`
    FLAT = "flat"  # learned scorer over the components at the tuned t_low / close_radius


class ScorerConfig(Section):
    """Learned keep/drop scorer (``filseg fit-scorer``, `filseg.postprocess.scorer`).

    Candidates are the components at the tuned ``t_low`` / ``close_radius`` of at least
    ``min_area`` px; a bag of gradient-boosted models predicts each one's expected matched IoU,
    and the ones scoring >= tau (chosen by nested crossfit) are kept.
    """

    mode: ScorerMode = ScorerMode.OFF
    min_area: int = Field(20, ge=0, description="candidate pre-filter: minimum area (px)")
    tau_grid: list[float] = Field(
        [round(0.05 + 0.01 * i, 2) for i in range(46)], description="cutoffs searched (0.05-0.5)"
    )
    n_estimators: int = 400
    learning_rate: float = 0.03
    num_leaves: int = 15
    min_child_samples: int = 40
    subsample: float = Field(0.8, gt=0, le=1)
    colsample: float = Field(0.8, gt=0, le=1)
    n_bags: int = Field(
        1, ge=1, description="models with seeds seed, seed+1, ...; their scores are averaged"
    )
    monotone: dict[str, int] = Field(
        {}, description="feature -> +1 / -1: the score never falls / rises with it"
    )

    @field_validator("monotone")
    @classmethod
    def _monotone_signs(cls, v: dict[str, int]) -> dict[str, int]:
        if any(s not in (-1, 0, 1) for s in v.values()):
            raise ValueError("monotone constraints are -1, 0 or +1")
        return v


class ExperimentConfig(Section):
    name: str
    seed: int = 42
    n_folds: int = 5
    folds: list[int] | None = None  # subset of folds to train; None = all
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    infer: InferConfig = InferConfig()
    post: InstanceParams = InstanceParams()  # used until `tune` writes post_params.json
    tune: TuneGrid = TuneGrid()
    scorer: ScorerConfig = ScorerConfig()
    # Experiments whose maps `filseg blend` averages into this one's oof/ and test_probs/; empty
    # for a trained experiment. tune / fit-scorer / predict then work on the blend unchanged.
    blend: list[str] = []

    @property
    def train_folds(self) -> list[int]:
        return self.folds if self.folds is not None else list(range(self.n_folds))


def apply_overrides(raw: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Set ``section.key=value`` entries (``value`` parsed as YAML) in a raw config dict.

    Used by ``filseg --set`` to run a variant of a config without writing a new YAML file. The
    result is validated like a file, so a misspelt key is an error. ``name`` cannot be
    overridden: it decides the output directory and must equal the config's file stem.
    """
    out = dict(raw)
    for item in overrides:
        key, sep, value = item.partition("=")
        parts = key.strip().split(".")
        if not sep or not all(parts):
            raise ValueError(f"override {item!r} is not of the form section.key=value")
        if parts == ["name"]:
            raise ValueError("`name` cannot be overridden: it must equal the config file stem")
        node = out
        for part in parts[:-1]:
            child = node.get(part)
            child = dict(child) if isinstance(child, dict) else {}
            node[part] = child
            node = child
        node[parts[-1]] = yaml.safe_load(value)
    return out


def load_config(path: Path, overrides: Sequence[str] = ()) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text())
    return ExperimentConfig.model_validate(apply_overrides(raw, overrides) if overrides else raw)


def dump_config(cfg: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
