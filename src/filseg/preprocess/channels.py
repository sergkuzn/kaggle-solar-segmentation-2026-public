"""Model input channels: one function per `Channel`, each standardized over the solar disk.

A channel's output must never change once a model has been trained on it
(`tests/test_imaging.py` pins every channel to golden values).
"""

from collections.abc import Callable, Sequence

import cv2
import numpy as np

from filseg.config import Channel
from filseg.preprocess.disk import Disk, circle_mask, locate_disk


def _standardize(x: np.ndarray, disk: np.ndarray) -> np.ndarray:
    vals = x[disk][::7]  # subsample: percentiles of ~400k pixels are plenty
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo, med, hi = np.percentile(vals, [1, 50, 99])
    scale = max(float(hi - lo), 1e-3) / 4.0
    out: np.ndarray = np.clip((x - med) / scale, -5, 5).astype(np.float32)
    out[~disk] = 0
    return out


def flat_field(img: np.ndarray, disk: np.ndarray, *, sigma: float = 48.0) -> np.ndarray:
    """Subtract a disk-aware Gaussian background (normalized convolution) from the image.

    The background is smooth, so it is estimated at 1/4 resolution for speed. It cannot follow
    the steep limb darkening of the last ~60 px, so the rim stays dark.
    """
    h, w = img.shape
    small = (w // 4, h // 4)
    f = cv2.resize(img.astype(np.float32) * disk, small, interpolation=cv2.INTER_AREA)
    m = cv2.resize(disk.astype(np.float32), small, interpolation=cv2.INTER_AREA)
    s = sigma / 4
    bg = cv2.GaussianBlur(f, (0, 0), s) / np.maximum(cv2.GaussianBlur(m, (0, 0), s), 1e-3)
    full_bg: np.ndarray = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    flat: np.ndarray = img.astype(np.float32) - full_bg
    return flat


# --- channels: (image, disk centre, disk mask) -> standardized channel ---


def _raw(img: np.ndarray, centre: Disk, disk: np.ndarray) -> np.ndarray:
    return _standardize(img.astype(np.float32), disk)


def _flat(img: np.ndarray, centre: Disk, disk: np.ndarray) -> np.ndarray:
    return _standardize(flat_field(img, disk), disk)


_CHANNELS: dict[Channel, Callable[[np.ndarray, Disk, np.ndarray], np.ndarray]] = {
    Channel.RAW: _raw,
    Channel.FLAT: _flat,
}


def build_channels(img: np.ndarray, channels: Sequence[Channel]) -> np.ndarray:
    """Stack the requested input channels of a uint8 grayscale frame into (C, H, W) float32."""
    centre = locate_disk(img)
    disk = circle_mask(img.shape, centre)
    return np.stack([_CHANNELS[ch](img, centre, disk) for ch in channels])
