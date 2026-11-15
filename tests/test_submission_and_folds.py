import numpy as np
import pytest

from conftest import blob
from filseg.competition import rle
from filseg.competition.coco import AnnotationSet, CocoAnnotation, Record
from filseg.competition.submission import build_submission, validate_submission
from filseg.data.folds import make_folds
from filseg.infra.exceptions import SubmissionValidationError


def test_rle_roundtrip() -> None:
    m = blob(700, 300, 90, 15)
    assert (rle.decode(rle.encode(m)) == m).all()


def test_polygon_rasterization_area() -> None:
    square = [100.0, 100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 200.0]
    assert abs(rle.area(rle.polygon_to_counts(square)) - 100 * 100) < 250


def test_validate_accepts_disjoint() -> None:
    sub = build_submission(
        [("S1Bh", [rle.encode(blob(500, 500, 50, 20)), rle.encode(blob(900, 900, 50, 20))])]
    )
    report = validate_submission(sub, ["S1Bh", "S2Bh"])
    assert report.rows == 2 and report.missing_images == ["S2Bh"]


@pytest.mark.parametrize(
    ("masks", "match"),
    [
        ([blob(500, 500, 50, 20), blob(520, 500, 50, 20)], "overlapping"),
        ([np.zeros((2048, 2048), bool)], "empty"),
    ],
)
def test_validate_rejects(masks: list[np.ndarray], match: str) -> None:
    sub = build_submission([("S1Bh", [rle.encode(m) for m in masks])])
    with pytest.raises(SubmissionValidationError, match=match):
        validate_submission(sub, ["S1Bh"])


def test_folds_group_all_records_of_an_image() -> None:
    ann = CocoAnnotation(
        id="a",
        image_id="x",
        category_id=1,
        segmentation=[[0, 0, 1, 1]],
        spine=[],
        bbox=[0, 0, 1, 1],
        area=1,
    )
    records = tuple(
        Record(record_id=f"0101{a:02d}-{s}", stem=s, annotations=(ann,))
        for i, s in enumerate(
            f"201501{d:02d}000000{st}" for d in range(1, 21) for st in ("Bh", "Lh")
        )
        for a in range(1 + i % 3)
    )
    folds = make_folds(AnnotationSet(records=records), n_folds=5, seed=1)
    assert folds.stem.is_unique, "one row (one fold) per physical image"
    sizes = folds.fold.value_counts()
    assert sizes.max() - sizes.min() <= 1
