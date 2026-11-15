"""The configs and the output layout in `filseg.infra.paths` stay consistent."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from filseg.config import ExperimentConfig, load_config
from filseg.infra.paths import REPO_ROOT, ExperimentPaths

CONFIGS = sorted((REPO_ROOT / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_name_matches_file(path: Path) -> None:
    # The CLI writes outputs/<cfg.name>/; scripts address experiments by file stem.
    assert load_config(path).name == path.stem


def test_config_rejects_unknown_keys() -> None:
    # A typo must not fall back to the default without a word (load_config above relies on it).
    with pytest.raises(ValidationError, match=r"train\.lerning_rate"):
        ExperimentConfig.model_validate({"name": "e", "train": {"lerning_rate": 1e-3}})


def test_fold_artifacts_live_under_folds_output() -> None:
    exp = ExperimentPaths(root=Path("outputs/e"))
    assert exp.model(3).is_relative_to(exp.folds)
    assert exp.history(3).is_relative_to(exp.folds)
    assert exp.checkpoint(3, 12).is_relative_to(exp.folds)


def test_model_resolves_best_checkpoint(tmp_path: Path) -> None:
    exp = ExperimentPaths(root=tmp_path / "e")
    assert exp.model(0).name == "model.pt"  # nothing trained yet
    exp.fold(0).mkdir(parents=True)
    exp.checkpoint(0, 40, last=True).touch()
    assert exp.model(0).name == "model_40_last.pt"  # last epoch was the best
    exp.checkpoint(0, 12).touch()
    assert exp.model(0).name == "model_12.pt"


def test_overrides_set_nested_fields_and_validate() -> None:
    path = REPO_ROOT / "configs" / "v4__unet_r34_v2_bce_x2.yaml"
    assert load_config(path, []) == load_config(path)
    cfg = load_config(path, ["scorer.n_bags=3", "scorer.monotone={p_max: 1}", "post.t_low=0.3"])
    assert (cfg.scorer.n_bags, cfg.scorer.monotone, cfg.post.t_low) == (3, {"p_max": 1}, 0.3)
    assert cfg.scorer.n_estimators == 200  # untouched siblings keep the file's values
    with pytest.raises(ValidationError):
        load_config(path, ["scorer.n_bagz=3"])
    for bad in ("name=other", "scorer.n_bags", "scorer..n_bags=1"):
        with pytest.raises(ValueError):
            load_config(path, [bad])


def test_final_pipeline_configs() -> None:
    final = load_config(REPO_ROOT / "configs" / "v4__unet_r34_v2_bce_x2.yaml")
    members = [load_config(REPO_ROOT / "configs" / f"{m}.yaml") for m in final.blend]
    assert [m.seed for m in members] == [42, 43]
    # the members differ only in their name and seed
    dumps = [m.model_dump(exclude={"name", "seed"}) for m in members]
    assert dumps[0] == dumps[1]
