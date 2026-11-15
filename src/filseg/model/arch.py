"""Model architecture."""

import segmentation_models_pytorch as smp
from torch import nn

from filseg.config import ModelConfig


def build_model(cfg: ModelConfig, *, in_channels: int, pretrained: bool = True) -> nn.Module:
    """Create an encoder-decoder segmentation net.

    Output channel 0 is the filament logit, channel 1 (if ``spine_head``) the spine logit.
    """
    model: nn.Module = smp.create_model(
        cfg.arch,
        encoder_name=cfg.encoder,
        encoder_weights=cfg.encoder_weights if pretrained else None,
        in_channels=in_channels,
        classes=2 if cfg.spine_head else 1,
    )
    return model
