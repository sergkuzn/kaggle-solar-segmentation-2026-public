"""Building and validating the competition CSV."""

import logging
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel

from filseg.competition import rle
from filseg.competition.metric import record_of
from filseg.infra.exceptions import SubmissionValidationError

log = logging.getLogger(__name__)

COLUMNS = ["filament_id", "segmentation_rle"]


def build_submission(predictions: Iterable[tuple[str, Sequence[str]]]) -> pd.DataFrame:
    """(stem, [rle counts, ...]) pairs -> submission frame with ids ``<stem>_<k>``."""
    rows = [
        (f"{stem}_{k}", counts)
        for stem, rles in predictions
        for k, counts in enumerate(rles, start=1)
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


class ValidationReport(BaseModel):
    rows: int
    images_with_predictions: int
    images_expected: int
    missing_images: list[str]
    mean_instances_per_image: float


def validate_submission(df: pd.DataFrame, expected_stems: Sequence[str]) -> ValidationReport:
    """Check format rules locally so no daily submission slot is wasted.

    Raises:
        SubmissionValidationError: On wrong columns, duplicate ids, unknown stems, empty or
            undecodable masks, or overlapping masks within an image.
    """
    if list(df.columns) != COLUMNS:
        raise SubmissionValidationError(f"columns must be {COLUMNS}, got {list(df.columns)}")
    if df["filament_id"].duplicated().any():
        raise SubmissionValidationError("duplicate filament_id values")
    stems = df["filament_id"].map(record_of)
    unknown = set(stems) - set(expected_stems)
    if unknown:
        raise SubmissionValidationError(
            f"{len(unknown)} unknown image ids, e.g. {sorted(unknown)[:3]}"
        )

    for stem, group in df.groupby(stems):
        acc: np.ndarray | None = None
        for fid, counts in zip(group["filament_id"], group["segmentation_rle"], strict=True):
            try:
                mask = rle.decode(str(counts))
            except Exception as exc:
                raise SubmissionValidationError(f"{fid}: undecodable RLE") from exc
            if not mask.any():
                raise SubmissionValidationError(f"{fid}: empty mask")
            acc = mask.astype(np.uint8) if acc is None else acc + mask
        if acc is not None and acc.max() > 1:
            raise SubmissionValidationError(f"{stem}: overlapping masks")

    missing = sorted(set(expected_stems) - set(stems))
    if missing:
        log.warning("%d test images have no prediction", len(missing))
    n_img = stems.nunique()
    return ValidationReport(
        rows=len(df),
        images_with_predictions=n_img,
        images_expected=len(expected_stems),
        missing_images=missing,
        mean_instances_per_image=len(df) / max(n_img, 1),
    )
