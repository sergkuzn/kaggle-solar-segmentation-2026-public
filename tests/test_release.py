"""`filseg.infra.release`: pack -> manifest -> fetch round trip over a file:// URL."""

from pathlib import Path

import pytest

from filseg.config import ExperimentConfig
from filseg.infra import release
from filseg.infra.paths import Paths


def _outputs(tmp_path: Path) -> tuple[ExperimentConfig, Paths]:
    paths = Paths(data_root=tmp_path / "data", output_root=tmp_path / "out")
    for name in ("a", "b"):
        exp = paths.experiment(name)
        for k in range(2):
            exp.checkpoint(k, 7).parent.mkdir(parents=True)
            exp.checkpoint(k, 7).write_bytes(f"{name}{k}best".encode())
            exp.checkpoint(k, 9, last=True).write_bytes(b"last")  # not shipped
        exp.config.write_text("name: x\n")
    bl = paths.experiment("bl")
    scorer = bl.scorer
    scorer.dir.mkdir(parents=True)
    for f in ("model.txt", "model_b1_h0.txt", "fold0.txt", "params.json"):
        (scorer.dir / f).write_text(f)
    bl.post_params.write_text("{}")
    cfg = ExperimentConfig.model_validate(
        {"name": "bl", "n_folds": 2, "blend": ["a", "b"], "scorer": {"mode": "flat"}}
    )
    return cfg, paths


def test_pack_and_fetch_round_trip(tmp_path: Path) -> None:
    cfg, paths = _outputs(tmp_path)
    manifest = release.pack(cfg, paths, tmp_path / "dist")
    shipped = {a.path for a in manifest.files}
    assert "a/folds/fold0/model_7.pt" in shipped
    assert "bl/scorer/model_b1_h0.txt" in shipped
    assert not any("_last" in p or "fold0.txt" in p for p in shipped)
    assert all("/" not in a.asset for a in manifest.files)

    dest = Paths(data_root=tmp_path / "data", output_root=tmp_path / "fresh")
    got = release.fetch((tmp_path / "dist").as_uri(), dest)
    assert len(got) == len(manifest.files)
    assert dest.experiment("b").model(1).read_bytes() == b"b1best"
    # second run: everything up to date, nothing re-downloaded
    (tmp_path / "dist" / "a--folds--fold0--model_7.pt").unlink()
    release.fetch((tmp_path / "dist").as_uri(), dest)


def test_fetch_rejects_corrupt_asset(tmp_path: Path) -> None:
    cfg, paths = _outputs(tmp_path)
    release.pack(cfg, paths, tmp_path / "dist")
    (tmp_path / "dist" / "a--folds--fold0--model_7.pt").write_bytes(b"tampered")
    dest = Paths(data_root=tmp_path / "data", output_root=tmp_path / "fresh")
    with pytest.raises(OSError, match="sha256"):
        release.fetch((tmp_path / "dist").as_uri(), dest)
    assert not list((tmp_path / "fresh").rglob("*.part"))


def test_pack_requires_every_fold(tmp_path: Path) -> None:
    cfg, paths = _outputs(tmp_path)
    for p in paths.experiment("b").fold(1).glob("*.pt"):
        p.unlink()
    with pytest.raises(FileNotFoundError, match="fold 1"):
        release.pack(cfg, paths, tmp_path / "dist")
