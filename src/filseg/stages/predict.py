"""predict: test probability maps -> validated submission.csv.

Reads the maps in ``test_probs/`` (written by ``write-maps --split test``, or by ``blend`` for a
blend) and extracts the instances with the scorer (``scorer.mode: flat``) or the tuned fixed rule
(``scorer.mode: off``). No network runs here, so a new post-processing costs minutes.
"""

import logging
from collections.abc import Callable

import numpy as np
from tqdm import tqdm

from filseg.competition.submission import build_submission, validate_submission
from filseg.config import ExperimentConfig, ScorerMode
from filseg.imaging import read_gray, read_prob
from filseg.infra.paths import Paths
from filseg.postprocess.instances import extract_instances
from filseg.postprocess.scorer import Scorer
from filseg.stages.common import console, image_stems, post_params, write_json

log = logging.getLogger(__name__)


def run(cfg: ExperimentConfig, paths: Paths) -> None:
    exp = paths.experiment(cfg.name)
    stems = image_stems(paths.test_images)
    missing = [s for s in stems if not (exp.test_probs / f"{s}.png").exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} test maps missing in {exp.test_probs} (run write-maps / blend)"
        )
    extract: Callable[[str, np.ndarray], list[str]]
    if cfg.scorer.mode == ScorerMode.FLAT:
        scorer = Scorer.load(exp.scorer.model, exp.scorer.params)
        extract = lambda s, prob: scorer.extract(  # noqa: E731
            prob, read_gray(paths.test_images / f"{s}.jpeg")
        )
    else:
        params = post_params(cfg, paths)
        extract = lambda _, prob: extract_instances(prob, params)  # noqa: E731
    preds = [
        (s, extract(s, read_prob(exp.test_probs / f"{s}.png")))
        for s in tqdm(stems, desc=f"test ({cfg.scorer.mode})")
    ]
    sub = build_submission(preds)
    report = validate_submission(sub, stems)
    sub.to_csv(exp.submission, index=False)
    write_json(exp.submission_report, report.model_dump())
    console.print(report.model_dump(exclude={"missing_images"}))
    log.info("wrote %s", exp.submission)
