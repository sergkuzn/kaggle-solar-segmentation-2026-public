"""Training targets derived from the annotations.

* **Soft filament target**: per-pixel fraction of an image's annotators that marked it as a
  filament (0, 1/3, 1/2, 2/3, 1). This learns the annotator *consensus*, which is what PQ
  rewards when every annotator record is scored separately.
* **Soft spine target**: same averaging over rasterized spine polylines (auxiliary head).
* **GT table**: one row per annotated filament in the official evaluation format.
"""

import logging
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from filseg.competition import rle
from filseg.competition.coco import IMAGE_SIZE, AnnotationSet, Record, load_annotations
from filseg.infra.paths import Paths

log = logging.getLogger(__name__)

SOFT_SUFFIX = "_soft.png"
SPINE_SUFFIX = "_spine.png"


def record_union_mask(record: Record) -> np.ndarray:
    """Union of all filament polygons of one annotator record, as a bool mask."""
    out = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    for ann in record.annotations:
        out |= rle.decode(rle.polygon_to_counts(ann.segmentation[0]))
    return out


def record_spine_mask(record: Record, *, thickness: int) -> np.ndarray:
    out = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for ann in record.annotations:
        pts = np.round(np.asarray(ann.spine, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(out, [pts], isClosed=False, color=1, thickness=thickness)
    return out.astype(bool)


def soft_targets(
    records: Sequence[Record], *, spine_thickness: int
) -> tuple[np.ndarray, np.ndarray]:
    """Annotator-averaged filament and spine targets as uint8 (0..255)."""
    soft = np.mean([record_union_mask(r) for r in records], axis=0)
    spine = np.mean([record_spine_mask(r, thickness=spine_thickness) for r in records], axis=0)
    return (soft * 255).round().astype(np.uint8), (spine * 255).round().astype(np.uint8)


def gt_rows(record: Record) -> list[tuple[str, str]]:
    return [
        (f"{record.record_id}_{k}", rle.polygon_to_counts(ann.segmentation[0]))
        for k, ann in enumerate(record.annotations, start=1)
    ]


def _write_targets(args: tuple[list[Record], Path, int]) -> str:
    records, out_dir, thickness = args
    stem = records[0].stem
    soft, spine = soft_targets(records, spine_thickness=thickness)
    cv2.imwrite(str(out_dir / f"{stem}{SOFT_SUFFIX}"), soft)
    cv2.imwrite(str(out_dir / f"{stem}{SPINE_SUFFIX}"), spine)
    return stem


def build_gt_table(annotations: AnnotationSet) -> pd.DataFrame:
    rows = [row for rec in annotations.records for row in gt_rows(rec)]
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def prepare_targets(
    paths: Paths, *, spine_thickness: int = 5, workers: int = 8, force: bool = False
) -> None:
    """Write soft/spine target PNGs for every training image and the GT table."""
    annotations = load_annotations(paths.train_json)
    paths.targets.mkdir(parents=True, exist_ok=True)

    if force or not paths.gt_csv.exists():
        build_gt_table(annotations).to_csv(paths.gt_csv, index=False)
        log.info("wrote %s", paths.gt_csv)

    todo = [
        (recs, paths.targets, spine_thickness)
        for stem, recs in annotations.by_stem.items()
        if force or not (paths.targets / f"{stem}{SPINE_SUFFIX}").exists()
    ]
    if not todo:
        log.info("targets already prepared (%d images)", len(annotations.by_stem))
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in tqdm(pool.map(_write_targets, todo), total=len(todo), desc="targets"):
            pass


def load_soft_target(paths: Paths, stem: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (filament, spine) soft targets in [0, 1] as float32 (2048x2048)."""
    soft = cv2.imread(str(paths.targets / f"{stem}{SOFT_SUFFIX}"), cv2.IMREAD_GRAYSCALE)
    spine = cv2.imread(str(paths.targets / f"{stem}{SPINE_SUFFIX}"), cv2.IMREAD_GRAYSCALE)
    if soft is None or spine is None:
        raise FileNotFoundError(f"targets for {stem} missing in {paths.targets}; run `prepare`")
    return soft.astype(np.float32) / 255.0, spine.astype(np.float32) / 255.0
