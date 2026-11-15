"""COCO RLE helpers matching the competition's submission format (counts string only)."""

from collections.abc import Sequence
from typing import TypedDict, cast

import numpy as np
from pycocotools import mask as mask_util

from filseg.competition.coco import IMAGE_SIZE


class EncodedRle(TypedDict):
    """Compressed COCO RLE (structurally identical to pycocotools' ``_EncodedRLE``)."""

    size: list[int]
    counts: str | bytes


def _counts_str(rle: EncodedRle) -> str:
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else counts


def encode(mask: np.ndarray) -> str:
    """Encode a binary (H, W) mask into a compressed COCO RLE counts string."""
    return _counts_str(mask_util.encode(np.asfortranarray(mask.astype(np.uint8))))


def to_rle_dict(counts: str, size: int = IMAGE_SIZE) -> EncodedRle:
    return {"size": [size, size], "counts": counts}


def decode(counts: str, size: int = IMAGE_SIZE) -> np.ndarray:
    """Decode a counts string into a boolean (size, size) mask."""
    mask: np.ndarray = mask_util.decode(to_rle_dict(counts, size)).astype(bool)
    return mask


def area(counts: str, size: int = IMAGE_SIZE) -> int:
    return int(mask_util.area(to_rle_dict(counts, size)))


def polygon_to_counts(polygon: Sequence[float], size: int = IMAGE_SIZE) -> str:
    """Rasterize one flat COCO polygon ``[x0, y0, x1, y1, ...]`` into RLE counts."""
    # The stub types polygons as list[int], but pycocotools accepts float coordinates.
    rles = mask_util.frPyObjects([cast(list[int], list(polygon))], size, size)
    return _counts_str(mask_util.merge(rles))


def iou_matrix(gt: Sequence[str], pred: Sequence[str], size: int = IMAGE_SIZE) -> np.ndarray:
    """Pairwise IoU between GT and predicted RLE counts strings.

    Returns:
        Array of shape (len(gt), len(pred)); rows are GT, columns predictions.
    """
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)), dtype=np.float64)
    g = [to_rle_dict(c, size) for c in gt]
    p = [to_rle_dict(c, size) for c in pred]
    # pycocotools: iou(dt, gt, iscrowd) -> (n_dt, n_gt)
    ious = mask_util.iou(p, g, [0] * len(g))
    return np.asarray(ious, dtype=np.float64).reshape(len(p), len(g)).T
