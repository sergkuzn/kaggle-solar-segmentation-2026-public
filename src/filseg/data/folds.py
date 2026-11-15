"""Grouped, stratified K-fold split by *physical image*.

42% of training images carry 2-3 independent annotator records. All records of one image must
land in the same fold, otherwise validation sees images the model was trained on.
Folds are stratified by (station, number of annotators) so every fold mixes instruments and
annotation density similarly.
"""

import random

import pandas as pd

from filseg.competition.coco import AnnotationSet, station_of


def make_folds(annotations: AnnotationSet, *, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Assign each image stem to a fold.

    Returns:
        DataFrame with columns ``stem, fold, station, n_records`` sorted by stem.
    """
    rows = [
        {"stem": stem, "station": station_of(stem), "n_records": len(recs)}
        for stem, recs in annotations.by_stem.items()
    ]
    df = pd.DataFrame(rows)
    rng = random.Random(seed)
    df["fold"] = -1
    offset = 0
    for _, group in df.groupby(["station", "n_records"], sort=True):
        idx = list(group.index)
        rng.shuffle(idx)
        for i, row_idx in enumerate(idx):
            # continue the round-robin across strata so fold sizes stay balanced
            df.loc[row_idx, "fold"] = (offset + i) % n_folds
        offset += len(idx)
    return df.sort_values("stem").reset_index(drop=True)
