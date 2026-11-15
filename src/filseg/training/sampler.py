"""Random training windows from full-resolution images with soft multi-annotator targets."""

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from filseg.config import AugmentConfig, DataConfig
from filseg.data.targets import load_soft_target
from filseg.imaging import center_crop
from filseg.infra.paths import Paths
from filseg.preprocess import Preprocessor
from filseg.training.augment import augment_pair


class FilamentCrops(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Samples ``samples_per_epoch`` random windows, cycling through the training images.

    Sample ``index`` uses image ``index % len(stems)``, so with a shuffling DataLoader every image
    appears equally often (±1) per epoch; window position and augmentation stay random.

    Windows are centred on a filament pixel with probability ``fg_prob`` (filaments cover ~0.3%
    of the disk, so uniform sampling would mostly show background). ``augment=None`` disables
    augmentation (see `filseg.training.augment`).
    """

    def __init__(
        self,
        paths: Paths,
        stems: Sequence[str],
        cfg: DataConfig,
        *,
        samples_per_epoch: int,
        augment: AugmentConfig | None,
    ) -> None:
        self.paths = paths
        self.stems = list(stems)
        self.cfg = cfg
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.pre = Preprocessor.from_config(cfg, paths)
        self._rng: np.random.Generator | None = None

    @property
    def rng(self) -> np.random.Generator:
        # Created lazily inside each DataLoader worker: torch seeds workers differently, and a
        # persistent generator keeps advancing across epochs.
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
        return self._rng

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _window(self, soft: np.ndarray, rng: np.random.Generator, win: int) -> tuple[int, int]:
        size = soft.shape[0]
        if rng.random() < self.cfg.fg_prob:
            ys, xs = np.nonzero(soft > 0)
            if len(ys):
                j = rng.integers(len(ys))
                cy, cx = int(ys[j]), int(xs[j])
                y0 = int(
                    np.clip(cy - win // 2 + rng.integers(-win // 4, win // 4 + 1), 0, size - win)
                )
                x0 = int(
                    np.clip(cx - win // 2 + rng.integers(-win // 4, win // 4 + 1), 0, size - win)
                )
                return y0, x0
        return int(rng.integers(0, size - win + 1)), int(rng.integers(0, size - win + 1))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = self.rng
        stem = self.stems[index % len(self.stems)]
        x = self.pre(self.paths.train_images / f"{stem}.jpeg")  # float16, maybe a memmap
        soft, spine = load_soft_target(self.paths, stem)
        y = center_crop(np.stack([soft, spine]), self.cfg.crop)

        win = self.cfg.train_crop
        y0, x0 = self._window(y[0], rng, win)
        x = x[:, y0 : y0 + win, x0 : x0 + win].astype(np.float32)
        y = y[:, y0 : y0 + win, x0 : x0 + win]
        if self.augment is not None:
            x, y = augment_pair(x, y, rng, self.augment)
        return (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)),
        )
