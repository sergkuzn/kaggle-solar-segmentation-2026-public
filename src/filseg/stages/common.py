"""Helpers shared by the stages: console output, metric files, image listing."""

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from filseg.competition.metric import PQResult
from filseg.config import ExperimentConfig, InstanceParams
from filseg.infra.paths import Paths

console = Console()
log = logging.getLogger(__name__)


def image_stems(directory: Path) -> list[str]:
    return sorted(p.stem for p in directory.glob("*.jp*g"))


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    log.info("wrote %s", path)


def print_pq(title: str, res: PQResult) -> None:
    table = Table(title=title)
    for col in ("PQ", "SQ", "RQ", "TP", "FP", "FN", "records"):
        table.add_column(col, justify="right")
    s = res.summary()
    table.add_row(
        f"{s['pq']:.4f}",
        f"{s['sq']:.4f}",
        f"{s['rq']:.4f}",
        *(str(int(s[k])) for k in ("tp", "fp", "fn", "records")),
    )
    console.print(table)


def post_params(cfg: ExperimentConfig, paths: Paths) -> InstanceParams:
    """Tuned instance-extraction parameters, or the config's ``post`` until `tune` has run."""
    tuned = paths.experiment(cfg.name).post_params
    if tuned.exists():
        return InstanceParams.model_validate_json(tuned.read_text())
    log.warning("no tuned post_params.json, using config defaults")
    return cfg.post
