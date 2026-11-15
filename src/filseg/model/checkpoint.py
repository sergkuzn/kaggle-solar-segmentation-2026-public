"""Checkpoint I/O."""

from pathlib import Path

import torch
from torch import nn

from filseg.config import ModelConfig
from filseg.model.arch import build_model


def save_checkpoint(model: nn.Module, path: Path, *, meta: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(
    path: Path, cfg: ModelConfig, *, in_channels: int, device: torch.device
) -> nn.Module:
    model = build_model(cfg, in_channels=in_channels, pretrained=False)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval()
