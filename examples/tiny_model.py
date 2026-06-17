"""Minimal Conv2d → GELU → Conv2d → residual add model for testing the exporter."""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyConvModel(nn.Module):
    """
    Two depthwise-preserving convolutions with a GELU between them and a residual
    add at the end.  Input and output channels are the same so the skip connection
    is a bare addition (no projection).

        x  →  Conv2d(C, H, 3×3)  →  GELU  →  Conv2d(H, C, 3×3)  →  + x  →  out
    """

    def __init__(self, channels: int = 3, hidden: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=3, padding=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return out + x   # residual add — requires in_channels == out_channels


def create_model() -> TinyConvModel:
    return TinyConvModel()


def create_example_input() -> tuple[torch.Tensor, ...]:
    # (batch=1, C=3, H=32, W=32) — static shape, float32, CPU
    return (torch.randn(1, 3, 32, 32),)
