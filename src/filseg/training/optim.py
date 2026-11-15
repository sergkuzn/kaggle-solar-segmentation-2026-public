"""Optimizer and learning-rate schedule."""

import math
from collections.abc import Callable

import torch
from torch import nn

from filseg.config import TrainConfig


def _lr_lambda(cfg: TrainConfig) -> Callable[[int], float]:
    """Linear warmup then cosine decay, as a multiplier of the base learning rate."""
    total = cfg.epochs * cfg.steps_per_epoch
    warm = cfg.warmup_epochs * cfg.steps_per_epoch

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))

    return f


def build_optimizer(
    model: nn.Module, cfg: TrainConfig
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """AdamW with a per-step warmup + cosine schedule (call ``sched.step()`` every batch)."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(cfg))
