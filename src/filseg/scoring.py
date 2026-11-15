"""Score probability maps: official PQ after the fixed instance-extraction rule.

Used by validation during training (`training.fit.validate`) to pick each fold's best epoch.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from filseg.competition.metric import PQResult, evaluate_subset
from filseg.competition.submission import build_submission
from filseg.config import InstanceParams
from filseg.postprocess.instances import extract_instances


def score_maps(
    maps: Iterable[tuple[str, np.ndarray]], gt_df: pd.DataFrame, params: InstanceParams
) -> PQResult:
    """(stem, (2048, 2048) probability map) pairs -> PQ of the images given.

    Maps are consumed one at a time, so a generator keeps memory flat.
    """
    preds = [(stem, extract_instances(prob, params)) for stem, prob in maps]
    return evaluate_subset(gt_df, build_submission(preds), {s for s, _ in preds})
