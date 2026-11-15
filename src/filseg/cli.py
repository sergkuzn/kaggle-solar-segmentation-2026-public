"""Command-line entry point: ``filseg <stage> ...``.

Stages of one trained model (each is idempotent and writes to data/ or outputs/):

    download -> extract -> prepare -> train -> write-maps --split test -> tune -> fit-scorer
             -> predict

A blend of trained experiments (config ``blend: [...]``) averages its members' maps instead of
training: ``blend`` -> tune -> fit-scorer -> predict. With released weights (``fetch-weights``)
inference is: each member's ``write-maps --split test`` -> ``blend --split test`` -> predict.

This module only parses arguments; each command runs a module of `filseg.stages`, imported
lazily so the commands that need no network don't import torch and ``--help`` stays fast.
"""

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.logging import RichHandler

from filseg.config import ExperimentConfig, load_config
from filseg.infra.paths import get_paths
from filseg.stages.common import console

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
log = logging.getLogger("filseg")

ConfigArg = Annotated[Path, typer.Argument(exists=True, help="experiment YAML in configs/")]


# `--set` overrides of the global callback, applied by `_config` to every command's config
_overrides: list[str] = []


def _config(path: Path) -> ExperimentConfig:
    return load_config(path, _overrides)


@app.callback()
def _setup(
    verbose: bool = False,
    set_: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            help="config override section.key=value (YAML value), repeatable; e.g. "
            "--set train.val_limit=4 write-maps cfg.yaml",
        ),
    ] = None,
) -> None:
    _overrides[:] = set_ or []
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )


@app.command()
def download(force: bool = False) -> None:
    """Download the competition zip (Kaggle CLI auth)."""
    from filseg.stages import data

    data.download(get_paths(), force=force)


@app.command()
def extract() -> None:
    """Extract the competition zip into data/raw (replaces any previous extraction)."""
    from filseg.stages import data

    data.extract(get_paths())


@app.command()
def prepare(
    n_folds: int = 5,
    seed: int = 42,
    spine_thickness: int = 5,
    workers: int = 8,
    force: bool = False,
) -> None:
    """Build soft targets, the official-format GT table and grouped CV folds."""
    from filseg.stages import data

    data.prepare(
        get_paths(),
        n_folds=n_folds,
        seed=seed,
        spine_thickness=spine_thickness,
        workers=workers,
        force=force,
    )


@app.command()
def train(
    config: ConfigArg,
    fold: Annotated[
        list[int] | None, typer.Option(help="fold(s) to train; default from config")
    ] = None,
) -> None:
    """Train fold models (GPU strongly recommended); writes OOF probability maps."""
    from filseg.stages import train as stage

    stage.run(_config(config), get_paths(), folds=fold)


@app.command()
def write_maps(
    config: ConfigArg,
    split: Annotated[str, typer.Option(help="'oof', 'test' or 'both'")] = "both",
) -> None:
    """Infer OOF / test probability maps from the fold checkpoints (config's TTA)."""
    from filseg.stages import write_maps as stage

    stage.run(_config(config), get_paths(), split=stage.Split(split))


@app.command()
def blend(
    config: ConfigArg,
    workers: int = 8,
    split: Annotated[
        str, typer.Option(help="'all' or 'test' (test maps only, no member OOF maps needed)")
    ] = "all",
) -> None:
    """Average the saved maps of the experiments in ``blend`` into this experiment's outputs."""
    from filseg.stages import blend as stage

    stage.run(_config(config), get_paths(), workers=workers, split=stage.Split(split))


@app.command()
def tune(config: ConfigArg, workers: int = 8) -> None:
    """Grid-search instance extraction on OOF maps; writes post_params.json."""
    from filseg.stages import tune as stage

    stage.run(_config(config), get_paths(), workers=workers)


@app.command()
def fit_scorer(config: ConfigArg, workers: int = 8) -> None:
    """Fit the learned instance scorer on OOF maps (nested crossfit); writes scorer/."""
    from filseg.stages import fit_scorer as stage

    stage.run(_config(config), get_paths(), workers=workers)


@app.command()
def predict(config: ConfigArg) -> None:
    """Post-process the saved test maps; writes and validates submission.csv."""
    from filseg.stages import predict as stage

    stage.run(_config(config), get_paths())


@app.command(name="evaluate")
def evaluate_cmd(
    predictions: Annotated[Path, typer.Argument(exists=True)],
    fold: Annotated[int | None, typer.Option(help="restrict to one CV fold")] = None,
) -> None:
    """Score a submission-format CSV of *train* predictions against the GT."""
    from filseg.stages import tools

    tools.evaluate_csv(get_paths(), predictions, fold=fold)


@app.command()
def validate(submission: Annotated[Path, typer.Argument(exists=True)]) -> None:
    """Check a submission CSV locally (format, decodability, non-overlap)."""
    from filseg.stages import tools

    tools.validate(get_paths(), submission)


@app.command()
def release_pack(
    config: ConfigArg,
    out: Annotated[Path, typer.Option(help="directory to upload as release assets")] = Path(
        "dist/weights"
    ),
) -> None:
    """Collect the checkpoints and scorer files inference needs, plus manifest.json."""
    from filseg.infra import release

    release.pack(_config(config), get_paths(), out)


@app.command()
def fetch_weights(
    repo: Annotated[
        str | None,
        typer.Option(envvar="FILSEG_WEIGHTS_REPO", help="GitHub repo owner/name"),
    ] = None,
    tag: str = "weights-v1",
    base_url: Annotated[
        str | None, typer.Option(help="assets base URL instead of a GitHub release")
    ] = None,
    force: bool = False,
) -> None:
    """Download released weights into outputs/ (SHA-256 checked)."""
    from filseg.infra import release

    if base_url is None:
        if repo is None:
            raise typer.BadParameter(
                "give --repo owner/name (or FILSEG_WEIGHTS_REPO) or --base-url"
            )
        base_url = release.release_url(repo, tag)
    release.fetch(base_url, get_paths(), force=force)


@app.command()
def viz(
    stem: str,
    predictions: Annotated[Path | None, typer.Option(exists=True)] = None,
    out: Path | None = None,
) -> None:
    """Render image + GT (train) + predictions overlay to a PNG for inspection/report."""
    from filseg.stages.viz import render_overlay

    paths = get_paths()
    target = out or paths.experiment("viz").root / f"{stem}.png"
    render_overlay(paths, stem, predictions, target)
    log.info("wrote %s", target)


if __name__ == "__main__":
    app()
