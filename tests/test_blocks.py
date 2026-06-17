"""
Common architectural blocks: SE, encoder stage, simple U-Net, multi-output head.
These are the building blocks most likely to appear in real Niko/Nika models.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from conftest import run_pipeline, assert_artifacts_exist, assert_weight_sizes_match, assert_ops_present


# ── Squeeze-and-Excitation ────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Channel attention via global avg pool + two FC layers + sigmoid gate."""
    def __init__(self, channels: int = 16, reduction: int = 4):
        super().__init__()
        mid = channels // reduction
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(mid, channels)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x).flatten(1)
        s = self.gate(self.fc2(self.act(self.fc1(s))))
        return x * s[:, :, None, None]


def test_se_block(tmp_path):
    result = run_pipeline(SEBlock(channels=16), (torch.randn(1, 16, 8, 8),), tmp_path)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
    assert result["manifest"]["outputs"][0]["shape"] == [1, 16, 8, 8]


def test_se_block_weight_count(tmp_path):
    result = run_pipeline(SEBlock(channels=16, reduction=4), (torch.randn(1, 16, 8, 8),), tmp_path)
    # fc1.weight, fc1.bias, fc2.weight, fc2.bias → 4 tensors
    assert len(result["weight_entries"]) == 4


# ── Encoder stage ────────────────────────────────────────────────────────────

class EncoderStage(nn.Module):
    """Two conv blocks + MaxPool downsample — one level of a typical encoder."""
    def __init__(self, in_ch: int = 3, out_ch: int = 32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.block(x))


def test_encoder_stage(tmp_path):
    result = run_pipeline(EncoderStage(3, 32), (torch.randn(1, 3, 32, 32),), tmp_path)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
    assert result["manifest"]["outputs"][0]["shape"] == [1, 32, 16, 16]


# ── Minimal U-Net ─────────────────────────────────────────────────────────────

class TinyUNet(nn.Module):
    """
    Two-level U-Net: encode → bottleneck → decode with skip concat.
    Smallest realistic test of the full encoder-decoder pattern.
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc = nn.Conv2d(1, 8, 3, padding=1)
        self.down = nn.MaxPool2d(2)
        # Bottleneck
        self.mid = nn.Conv2d(8, 16, 3, padding=1)
        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec = nn.Conv2d(16 + 8, 8, 3, padding=1)  # 16 + skip
        # Head
        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        skip = torch.relu(self.enc(x))
        x = torch.relu(self.mid(self.down(skip)))
        x = torch.relu(self.dec(torch.cat([self.up(x), skip], dim=1)))
        return self.head(x)


def test_tiny_unet(tmp_path):
    result = run_pipeline(TinyUNet(), (torch.randn(1, 1, 32, 32),), tmp_path)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
    assert_ops_present(result["manifest"], "aten.cat.default", "aten.convolution.default")
    # output same spatial size as input
    assert result["manifest"]["outputs"][0]["shape"] == [1, 1, 32, 32]


# ── Multi-output head ────────────────────────────────────────────────────────

class MultiHeadOutput(nn.Module):
    """
    Shared backbone + two output heads (e.g. classification + regression).
    Tests that the exporter handles tuple outputs.
    """
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.cls_head = nn.Linear(16, 10)
        self.reg_head = nn.Linear(16, 4)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.cls_head(feat), self.reg_head(feat)


def test_multi_head_output(tmp_path):
    result = run_pipeline(MultiHeadOutput(), (torch.randn(1, 3, 32, 32),), tmp_path)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
    outputs = result["manifest"]["outputs"]
    assert len(outputs) == 2
    shapes = {tuple(o["shape"]) for o in outputs}
    assert (1, 10) in shapes
    assert (1, 4) in shapes


# ── Gated / multiplicative features ─────────────────────────────────────────

class GatedConv(nn.Module):
    """Conv with sigmoid gate — common in image restoration and generation models."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 16, 3, padding=1)  # doubled channels: feat + gate

    def forward(self, x):
        out = self.conv(x)
        feat, gate = out.chunk(2, dim=1)
        return feat * torch.sigmoid(gate)


def test_gated_conv(tmp_path):
    result = run_pipeline(GatedConv(), (torch.randn(1, 8, 16, 16),), tmp_path)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
    assert result["manifest"]["outputs"][0]["shape"] == [1, 8, 16, 16]
