"""`filseg blend`: the plain mean of the members' maps; ``split=test`` for inference only."""

import json
from pathlib import Path

import numpy as np
import pytest

from filseg.config import ExperimentConfig
from filseg.imaging import read_prob, write_prob
from filseg.infra.paths import Paths
from filseg.stages import blend

CFG = ExperimentConfig.model_validate({"name": "bl", "blend": ["a", "b"]})


def _members(tmp_path: Path) -> Paths:
    paths = Paths(data_root=tmp_path / "data", output_root=tmp_path / "out")
    rng = np.random.default_rng(0)
    for name in CFG.blend:
        exp = paths.experiment(name)
        for d in (exp.oof, exp.test_probs):
            d.mkdir(parents=True)
            for i in range(3):
                write_prob(d / f"im{i}.png", rng.integers(0, 256, (16, 16)) / 255.0)
    return paths


def _read(paths: Paths, exp: str, sub: str = "oof", name: str = "im0.png") -> np.ndarray:
    return read_prob(getattr(paths.experiment(exp), sub) / name)


def test_blend_is_plain_mean(tmp_path: Path) -> None:
    paths = _members(tmp_path)
    blend.run(CFG, paths, workers=1)
    for sub in ("oof", "test_probs"):
        expect = (_read(paths, "a", sub) + _read(paths, "b", sub)) / 2
        np.testing.assert_allclose(_read(paths, "bl", sub), expect, atol=0.5 / 255 + 1e-6)
    rec = json.loads(paths.experiment("bl").blend.read_text())
    assert rec == {"members": ["a", "b"], "maps": {"oof": 3, "test_probs": 3}}


def test_members_must_have_the_same_maps(tmp_path: Path) -> None:
    paths = _members(tmp_path)
    (paths.experiment("b").oof / "im2.png").unlink()
    with pytest.raises(FileNotFoundError, match="members differ"):
        blend.run(CFG, paths, workers=1)


def test_split_test_needs_no_oof(tmp_path: Path) -> None:
    paths = _members(tmp_path)
    for name in CFG.blend:  # a fresh clone after fetch-weights + write-maps: test maps only
        exp = paths.experiment(name)
        for p in exp.oof.glob("*.png"):
            p.unlink()
        exp.oof.rmdir()
    bl = paths.experiment("bl")
    bl.root.mkdir(parents=True)
    bl.blend.write_text('{"kept": true}')
    with pytest.raises(FileNotFoundError, match="no OOF maps"):
        blend.run(CFG, paths, workers=1)
    blend.run(CFG, paths, workers=1, split=blend.Split.TEST)
    mean = (_read(paths, "a", "test_probs") + _read(paths, "b", "test_probs")) / 2
    np.testing.assert_allclose(_read(paths, "bl", "test_probs"), mean, atol=0.5 / 255 + 1e-6)
    assert not bl.oof.exists()
    assert json.loads(bl.blend.read_text()) == {"kept": True}  # the full blend's record stays
