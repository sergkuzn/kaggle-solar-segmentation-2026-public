"""Image I/O, centred-crop geometry and the probability-map file format."""

from pathlib import Path

import cv2
import numpy as np

from filseg.competition.coco import IMAGE_SIZE


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    """Crop the central ``size x size`` window of the last two axes."""
    h, w = arr.shape[-2:]
    y0, x0 = (h - size) // 2, (w - size) // 2
    return arr[..., y0 : y0 + size, x0 : x0 + size]


def paste_center(arr: np.ndarray, full: int = IMAGE_SIZE) -> np.ndarray:
    """Inverse of `center_crop` for a 2-D map (zeros outside the crop)."""
    size = arr.shape[-1]
    out = np.zeros((full, full), dtype=arr.dtype)
    y0 = (full - size) // 2
    out[y0 : y0 + size, y0 : y0 + size] = arr
    return out


def write_prob(path: Path, prob: np.ndarray) -> None:
    """Save a [0, 1] probability map as a uint8 PNG (~50-200 KB; OOF and test maps)."""
    cv2.imwrite(str(path), (prob * 255).round().astype(np.uint8))


def read_prob(path: Path) -> np.ndarray:
    """Load a map saved by `write_prob` as float32 in [0, 1]."""
    return read_gray(path).astype(np.float32) / 255.0
