"""Cached image -> model-input preprocessing.

`Preprocessor` turns a JPEG into the (C, crop, crop) model input. With caching enabled the
result is stored as ``<cache>/<channels>_crop<N>_v<VERSION>/<stem>.npy`` (float16, ~7 MB for
two channels) and later memory-mapped, so a training window only reads the pixels it needs.
"""

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from filseg.config import CacheMode, Channel, DataConfig
from filseg.imaging import center_crop, read_gray
from filseg.infra.paths import Paths
from filseg.preprocess.channels import build_channels

# Cache-format version. Channel outputs are frozen per `Channel` name (a new transform is a new
# channel, which gets its own cache directory), so bump this only for a change common to all
# channels, e.g. the disk geometry or the standardization.
PREPROC_VERSION = 2
# Inputs are clipped to [-5, 5], where float16 is accurate to ~0.004. Outputs are always
# quantized, so cached and on-the-fly inputs are bit-identical.
INPUT_DTYPE = np.float16


class Preprocessor:
    """Image -> (C, crop, crop) `INPUT_DTYPE` model input, optionally backed by a disk cache.

    Cached inputs come back as read-only memory maps: slice first, then cast to float32.
    Safe to share across DataLoader workers (cache writes are atomic renames).
    """

    def __init__(
        self,
        channels: Sequence[Channel],
        crop: int,
        *,
        cache_root: Path | None = None,
        mode: CacheMode = CacheMode.OFF,
    ) -> None:
        self.channels = list(channels)
        self.crop = crop
        self.mode = mode if cache_root is not None else CacheMode.OFF
        tag = "-".join(self.channels)
        self.cache_dir = (
            cache_root / f"{tag}_crop{crop}_v{PREPROC_VERSION}" if cache_root is not None else None
        )

    @classmethod
    def from_config(cls, cfg: DataConfig, paths: Paths | None = None) -> "Preprocessor":
        """``paths`` locates the input cache; without it nothing is cached."""
        cache_root = paths.input_cache if paths is not None else None
        return cls(cfg.channels, cfg.crop, cache_root=cache_root, mode=cfg.cache)

    def from_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess an in-memory grayscale image (never cached)."""
        return center_crop(build_channels(img, self.channels), self.crop).astype(INPUT_DTYPE)

    def cache_path(self, stem: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / f"{stem}.npy"

    def __call__(self, image_path: Path) -> np.ndarray:
        """Preprocess ``image_path``, consulting / filling the cache according to ``mode``."""
        cached = self.cache_path(image_path.stem) if self.mode is not CacheMode.OFF else None
        if cached is not None and cached.exists():
            arr: np.ndarray = np.load(cached, mmap_mode="r")
            return arr
        x = self.from_image(read_gray(image_path))
        if cached is not None and self.mode is CacheMode.READ_WRITE:
            _atomic_save(cached, x)
        return x

    def warm(self, image_path: Path) -> None:
        """Make sure ``image_path`` is in the cache (for parallel precomputation)."""
        cached = self.cache_path(image_path.stem)
        if cached is None:
            raise ValueError("no cache directory configured")
        if not cached.exists():
            _atomic_save(cached, self.from_image(read_gray(image_path)))


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    # Write-then-rename: concurrent workers may race on one stem, but never see a partial file.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
