"""Per-fold training loop with PQ-based model selection."""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from filseg.config import ExperimentConfig
from filseg.infra.paths import Paths
from filseg.model.arch import build_model
from filseg.model.checkpoint import save_checkpoint
from filseg.model.predictor import Predictor
from filseg.preprocess import Preprocessor
from filseg.scoring import score_maps
from filseg.training.losses import compute_loss
from filseg.training.optim import build_optimizer
from filseg.training.sampler import FilamentCrops

log = logging.getLogger(__name__)

EpochCallback = Callable[[dict[str, float]], None]


def validate(
    model: nn.Module,
    cfg: ExperimentConfig,
    paths: Paths,
    stems: Sequence[str],
    gt_df: pd.DataFrame,
    device: torch.device,
) -> dict[str, float]:
    """Full-disk inference on the validation fold (no TTA) -> official-semantics PQ summary,
    with the config's ``post`` rule."""
    model.eval()
    pre = Preprocessor.from_config(cfg.data, paths)
    predictor = Predictor([model], pre, device, amp=cfg.train.amp)
    maps = ((s, predictor.prob_file(paths.train_images / f"{s}.jpeg")) for s in stems)
    res = score_maps(maps, gt_df, cfg.post)
    model.train()
    return res.summary()


@dataclass(frozen=True)
class FoldResult:
    best_pq: float
    best_epoch: int
    best_metrics: dict[str, float]  # full validation summary (pq, sq, rq, tp, ...) of best_epoch


def fit_fold(
    cfg: ExperimentConfig,
    paths: Paths,
    fold: int,
    *,
    train_stems: Sequence[str],
    val_stems: Sequence[str],
    gt_df: pd.DataFrame,
    device: torch.device,
    on_epoch: EpochCallback | None = None,
) -> FoldResult:
    """Train one fold; keep the best-validation-PQ and the final weights (``exp.checkpoint``).

    The best epoch is saved as ``model_<epoch>.pt``, the final one as ``model_<epochs>_last.pt``;
    when the final epoch is also the best, only ``model_<epochs>_last.pt`` is kept.

    ``on_epoch`` receives each epoch's row (loss parts, lr, time, and ``val_*`` on validation
    epochs) for logging; it does not influence training.
    """
    torch.manual_seed(cfg.seed + fold)
    np.random.seed(cfg.seed + fold)
    exp = paths.experiment(cfg.name)

    ds = FilamentCrops(
        paths,
        train_stems,
        cfg.data,
        samples_per_epoch=cfg.train.steps_per_epoch * cfg.train.batch_size,
        augment=cfg.train.augment,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.train.num_workers > 0,
        drop_last=True,
    )
    model = build_model(cfg.model, in_channels=len(cfg.data.channels)).to(device)
    opt, sched = build_optimizer(model, cfg.train)
    amp = device.type == "cuda" and cfg.train.amp

    for stale in exp.checkpoints(fold):  # from an earlier run in the same directory
        stale.unlink()
    best_pq, best_epoch, best_ckpt = -1.0, 0, None
    best_metrics: dict[str, float] = {}
    for epoch in range(cfg.train.epochs):
        t0, losses, parts = time.time(), [], []
        model.train()
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(x)
            loss, loss_parts = compute_loss(out.float(), y, cfg)
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            sched.step()
            losses.append(loss.item())
            parts.append(loss_parts)
        row: dict[str, float] = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            **{f"loss_{k}": float(np.mean([p[k] for p in parts])) for k in parts[0]},
            "lr": float(sched.get_last_lr()[0]),
            "sec": time.time() - t0,
        }

        last = epoch + 1 == cfg.train.epochs
        if (epoch + 1) % cfg.train.val_every == 0 or last:
            metrics = validate(model, cfg, paths, val_stems, gt_df, device)
            row |= {f"val_{k}": v for k, v in metrics.items()}
            if metrics["pq"] > best_pq:
                best_pq, best_epoch, best_metrics = metrics["pq"], epoch + 1, metrics
                if best_ckpt is not None:
                    best_ckpt.unlink()
                best_ckpt = exp.checkpoint(fold, epoch + 1)
                save_checkpoint(
                    model, best_ckpt, meta={"fold": fold, "epoch": epoch + 1, "val_pq": best_pq}
                )
        log.info("fold %d %s", fold, {k: round(v, 4) for k, v in row.items()})
        if on_epoch is not None:
            on_epoch(row)

    # The last epoch is always validated, so row["val_pq"] exists here.
    save_checkpoint(
        model,
        exp.checkpoint(fold, cfg.train.epochs, last=True),
        meta={"fold": fold, "epoch": cfg.train.epochs, "val_pq": row["val_pq"]},
    )
    if best_epoch == cfg.train.epochs and best_ckpt is not None:
        best_ckpt.unlink()  # same weights as the _last checkpoint
    return FoldResult(best_pq=best_pq, best_epoch=best_epoch, best_metrics=best_metrics)
