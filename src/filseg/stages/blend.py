"""blend: average the saved maps of several trained experiments into a new experiment.

For a config with ``blend: [a, b, ...]``, the OOF maps (``oof/``) and the test maps
(``test_probs/``) of the members are written to the blend's output dir as their pixel-wise mean.
OOF maps of a blend are still out-of-fold: every member's OOF map of an image comes from a model
that never saw it (all members use folds.csv). Afterwards ``tune``, ``fit-scorer`` and
``predict`` run on the blend as on any experiment.

``split=test`` blends only the test maps and needs no member OOF maps: the inference-only path
after ``fetch-weights`` (members' ``write-maps --split test``, then ``blend --split test``, then
``predict``). It leaves an existing ``blend.json`` (the record of the full blend) in place.
"""

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from enum import StrEnum
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

from filseg.config import ExperimentConfig
from filseg.imaging import read_prob, write_prob
from filseg.infra.paths import ExperimentPaths, Paths
from filseg.stages.common import write_json

log = logging.getLogger(__name__)

# map directories of an experiment, by name (the blend writes the same name)
_DIRS: dict[str, Callable[[ExperimentPaths], Path]] = {
    "oof": lambda e: e.oof,
    "test_probs": lambda e: e.test_probs,
}


class Split(StrEnum):
    ALL = "all"
    TEST = "test"


def _blend_map(name: str, *, srcs: Sequence[Path], dst: Path) -> None:
    write_prob(dst / name, np.mean([read_prob(s / name) for s in srcs], axis=0))


def run(
    cfg: ExperimentConfig, paths: Paths, *, workers: int = 8, split: Split = Split.ALL
) -> None:
    if len(cfg.blend) < 2:
        raise ValueError(f"{cfg.name}: `blend` must list at least two experiments")
    if cfg.name in cfg.blend:
        raise ValueError(f"{cfg.name} cannot blend itself")
    exp = paths.experiment(cfg.name)
    members = [paths.experiment(m) for m in cfg.blend]
    for name, m in zip(cfg.blend, members, strict=True):
        if split is Split.ALL and not m.oof.is_dir():
            raise FileNotFoundError(f"member {name} has no OOF maps in {m.oof}")
    written: dict[str, int] = {}
    for key, get in _DIRS.items():
        if split is Split.TEST and key == "oof":
            continue
        srcs = [get(m) for m in members]
        if not all(s.is_dir() for s in srcs):
            continue
        names = sorted(p.name for p in srcs[0].glob("*.png"))
        for s in srcs[1:]:
            other = {p.name for p in s.glob("*.png")}
            if missing := set(names) ^ other:
                raise FileNotFoundError(f"{key}: members differ in {len(missing)} maps ({s})")
        dst = get(exp)
        dst.mkdir(parents=True, exist_ok=True)
        fn = partial(_blend_map, srcs=srcs, dst=dst)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(tqdm(pool.map(fn, names, chunksize=8), total=len(names), desc=key))
        written[key] = len(names)
    if split is Split.ALL or not exp.blend.exists():
        write_json(exp.blend, {"members": cfg.blend, "maps": written})
    log.info("blended %s -> %s: %s", cfg.blend, exp.root, written)
