"""Training loss."""

import torch
import torch.nn.functional as F

from filseg.config import ExperimentConfig


def compute_loss(
    out: torch.Tensor, y: torch.Tensor, cfg: ExperimentConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    """BCE on the soft filament target (channel 0) plus ``train.spine_weight`` x BCE on the soft
    spine target (channel 1, with ``model.spine_head``).

    BCE is a proper scoring rule, minimised at p = t, so on the annotator-consensus target the
    maps stay calibrated: a pixel half the annotators marked gets p ~ 0.5.
    """
    fil, fil_t = out[:, 0], y[:, 0]
    loss = F.binary_cross_entropy_with_logits(fil, fil_t)
    parts = {"bce": loss.item()}
    if cfg.model.spine_head:
        spine = F.binary_cross_entropy_with_logits(out[:, 1], y[:, 1])
        loss = loss + cfg.train.spine_weight * spine
        parts["spine"] = spine.item()
    return loss, parts
