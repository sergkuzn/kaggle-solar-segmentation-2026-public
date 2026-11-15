"""Training-time augmentation of (C, H, W) input / target window pairs."""

import numpy as np

from filseg.config import AugmentConfig


def dihedral(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """One of the 8 dihedral transforms, applied identically to input and target.

    The Sun has no preferred orientation in these crops.
    """
    k = int(rng.integers(4))
    x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(1, 2))
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, :, ::-1]
    return x, y


def intensity_jitter(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    gain: tuple[float, float],
    offset_std: float,
) -> np.ndarray:
    """Mild per-channel gain/offset jitter (inputs are disk-standardized, so ~unit scale)."""
    g = rng.uniform(*gain, size=(x.shape[0], 1, 1))
    b = rng.normal(0, offset_std, size=(x.shape[0], 1, 1))
    out: np.ndarray = x * g + b
    return out


def augment_pair(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the augmentations enabled in ``cfg``; the target only moves with the input."""
    if cfg.dihedral:
        x, y = dihedral(x, y, rng)
    x = intensity_jitter(x, rng, gain=cfg.gain, offset_std=cfg.offset_std)
    return x, y
