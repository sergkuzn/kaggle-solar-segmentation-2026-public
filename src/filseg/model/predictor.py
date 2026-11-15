"""Full-disk probability prediction with fold ensembling and test-time augmentation."""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from filseg.config import TTA, ExperimentConfig
from filseg.imaging import paste_center, write_prob
from filseg.infra.paths import Paths
from filseg.model.checkpoint import load_checkpoint
from filseg.preprocess import Preprocessor

log = logging.getLogger(__name__)

Transform = tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]


def _tta_transforms(tta: TTA) -> list[Transform]:
    """(forward, inverse) pairs acting on the last two dims of NCHW tensors."""
    ident: Transform = (lambda t: t, lambda t: t)
    hflip: Transform = (lambda t: t.flip(-1), lambda t: t.flip(-1))
    if tta is TTA.NONE:
        return [ident]
    if tta is TTA.FLIP:
        return [ident, hflip]
    out: list[Transform] = []
    for k in range(4):
        for flip in (False, True):

            def fwd(t: torch.Tensor, k: int = k, flip: bool = flip) -> torch.Tensor:
                t = torch.rot90(t, k, dims=(-2, -1))
                return t.flip(-1) if flip else t

            def inv(t: torch.Tensor, k: int = k, flip: bool = flip) -> torch.Tensor:
                t = t.flip(-1) if flip else t
                return torch.rot90(t, -k, dims=(-2, -1))

            out.append((fwd, inv))
    return out


class Predictor:
    """Averages sigmoid filament probabilities over models and TTA views."""

    def __init__(
        self,
        models: Sequence[nn.Module],
        pre: Preprocessor,
        device: torch.device,
        *,
        tta: TTA = TTA.NONE,
        amp: bool = False,
    ):
        self.models = list(models)
        self.pre = pre
        self.device = device
        self.transforms = _tta_transforms(tta)
        self.amp = amp and device.type == "cuda"

    @classmethod
    def from_experiment(
        cls,
        cfg: ExperimentConfig,
        device: torch.device,
        folds: Sequence[int],
        paths: Paths,
    ) -> "Predictor":
        """Ensemble of the best checkpoints of ``folds``, with the config's inputs and TTA."""
        exp = paths.experiment(cfg.name)
        models = [
            load_checkpoint(
                exp.model(f),
                cfg.model,
                in_channels=len(cfg.data.channels),
                device=device,
            )
            for f in folds
        ]
        pre = Preprocessor.from_config(cfg.data, paths)
        return cls(models, pre, device, tta=cfg.infer.tta, amp=cfg.train.amp)

    def prob(self, img: np.ndarray) -> np.ndarray:
        """Return the (2048, 2048) float32 filament probability of an in-memory image."""
        return self._run(self.pre.from_image(img))

    def prob_file(self, image_path: Path) -> np.ndarray:
        """Like `prob`, but goes through the preprocessing cache (if configured)."""
        return self._run(self.pre(image_path))

    @torch.inference_mode()
    def _run(self, x: np.ndarray) -> np.ndarray:
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None].to(self.device)
        acc = torch.zeros(xt.shape[-2:], device=self.device)
        for model in self.models:
            for fwd, inv in self.transforms:
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                    logits = model(fwd(xt))
                acc += torch.sigmoid(inv(logits.float())[0, 0])  # channel 0: filament
        n = len(self.models) * len(self.transforms)
        return paste_center((acc / n).cpu().numpy())


def write_prob_maps(
    predictor: Predictor, image_dir: Path, stems: Sequence[str], out: Path
) -> None:
    """Save the probability maps of ``stems`` to ``out`` (`write_prob`)."""
    out.mkdir(parents=True, exist_ok=True)
    for stem in tqdm(stems, desc=f"prob maps -> {out.name}"):
        write_prob(out / f"{stem}.png", predictor.prob_file(image_dir / f"{stem}.jpeg"))


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
