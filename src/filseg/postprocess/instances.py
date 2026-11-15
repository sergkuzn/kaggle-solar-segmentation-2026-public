"""Turning probability maps into disjoint filament instances (connected components).

PQ-relevant facts that shape this module:

* every submitted instance adds 1/2 to the PQ denominator per annotator record, TP or not, so
  low-confidence components should be dropped rather than "tried";
* IoU > 0.5 is a hard cliff, so fragmentation (one filament -> several pieces) is expensive;
  closing + hysteresis keep weakly connected filament bodies together;
* predicted masks of one image must not overlap, or the submission errors.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from filseg.competition import rle
from filseg.config import InstanceParams


@dataclass
class Component:
    """One candidate instance with features usable by a downstream keep/drop scorer."""

    label: int
    area: int
    max_prob: float
    mean_prob: float
    bbox: tuple[int, int, int, int]  # x, y, w, h


def components(
    prob: np.ndarray, *, t_low: float, close_radius: int
) -> tuple[np.ndarray, list[Component]]:
    """Label connected components of ``prob >= t_low`` after an optional binary closing.

    Returns:
        (label map int32, components). Components are disjoint by construction.
    """
    binary: np.ndarray = (prob >= t_low).astype(np.uint8)
    if close_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1,) * 2)
        binary = np.asarray(cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # Fortran order makes per-instance masks (labels == k) directly RLE-encodable without a copy.
    labels = np.asfortranarray(labels, dtype=np.int32)
    if n <= 1:
        return labels, []
    # per-component stats over foreground pixels only (filaments cover ~1% of the frame)
    fg = labels > 0
    lab: npt.NDArray[np.intp] = labels[fg].astype(np.intp)
    p = prob[fg]
    maxp = np.zeros(n, dtype=np.float64)
    np.maximum.at(maxp, lab, p)
    meanp = np.bincount(lab, weights=p, minlength=n) / np.maximum(np.bincount(lab, minlength=n), 1)
    idx = np.arange(1, n)
    comps = [
        Component(
            label=int(i),
            area=int(stats[i, cv2.CC_STAT_AREA]),
            max_prob=float(maxp[i]),
            mean_prob=float(meanp[i]),
            bbox=(
                int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]),
                int(stats[i, cv2.CC_STAT_HEIGHT]),
            ),
        )
        for i in idx
    ]
    return labels, comps


def keep(comp: Component, params: InstanceParams) -> bool:
    return (
        comp.area >= params.min_area
        and comp.max_prob >= params.t_high
        and comp.mean_prob >= params.min_score
    )


def kept_components(
    prob: np.ndarray, params: InstanceParams
) -> tuple[np.ndarray, list[Component]]:
    """(label map, components that pass `keep`), in `extract_instances` order."""
    labels, comps = components(prob, t_low=params.t_low, close_radius=params.close_radius)
    return labels, [c for c in comps if keep(c, params)]


def encode_components(labels: np.ndarray, comps: Sequence[Component]) -> list[str]:
    return [rle.encode(labels == c.label) for c in comps]


def extract_instances(prob: np.ndarray, params: InstanceParams) -> list[str]:
    """Probability map (H, W) in [0, 1] -> list of RLE counts of disjoint instances."""
    return encode_components(*kept_components(prob, params))
