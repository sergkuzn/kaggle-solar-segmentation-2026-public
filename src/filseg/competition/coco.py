"""Parsing of the MAGFiLO COCO annotation file into per-image, per-annotator records."""

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from pydantic import BaseModel

IMAGE_SIZE = 2048


class CocoImage(BaseModel):
    id: str  # "<annotator batch>-<stem>", e.g. "010101-20140609195854Bh"
    file_name: str
    width: int
    height: int
    date_captured: str | None = None
    url: str | None = None


class CocoAnnotation(BaseModel):
    id: str
    image_id: str
    category_id: int
    segmentation: list[list[float]]  # always exactly one closed polygon
    spine: list[float]  # flat polyline [x0, y0, x1, y1, ...]
    bbox: list[float]
    area: float


class CocoFile(BaseModel):
    images: list[CocoImage]
    annotations: list[CocoAnnotation]


def stem_of(file_name: str) -> str:
    """Return the image stem (``YYYYMMDDHHMMSSII``) of a file name."""
    return Path(file_name).stem


def station_of(stem: str) -> str:
    """Return the two-letter GONG station code (e.g. ``Bh`` = Big Bear) of an image stem."""
    return stem[-2:]


@dataclass(frozen=True)
class Record:
    """One annotator's complete set of filaments for one physical image."""

    record_id: str
    stem: str
    annotations: tuple[CocoAnnotation, ...]


@dataclass(frozen=True)
class AnnotationSet:
    """All training records grouped by physical image."""

    records: tuple[Record, ...]

    @cached_property
    def by_stem(self) -> dict[str, list[Record]]:
        grouped: dict[str, list[Record]] = defaultdict(list)
        for rec in self.records:
            grouped[rec.stem].append(rec)
        return dict(sorted(grouped.items()))

    @property
    def stems(self) -> list[str]:
        return list(self.by_stem)


def load_annotations(path: Path) -> AnnotationSet:
    """Load the competition training JSON.

    Args:
        path: Path to ``MAGFiLO_1.0_Annotations_kaggle2026_train.json``.

    Returns:
        Records (one per annotator x image), each holding its filament annotations.
    """
    coco = CocoFile.model_validate_json(path.read_bytes())
    anns: dict[str, list[CocoAnnotation]] = defaultdict(list)
    for ann in coco.annotations:
        anns[ann.image_id].append(ann)
    records = tuple(
        Record(record_id=img.id, stem=stem_of(img.file_name), annotations=tuple(anns[img.id]))
        for img in sorted(coco.images, key=lambda i: i.id)
    )
    return AnnotationSet(records=records)
