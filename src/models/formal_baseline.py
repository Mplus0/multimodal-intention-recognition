"""Compact baseline model for formal multimodal feature tensors."""

from __future__ import annotations

import torch
import torch.nn as nn


INTENT_NAMES = {
    0: "menu",
    1: "select",
    2: "magnify",
    3: "narrow",
    4: "brush",
    5: "cancel",
}


class VideoEncoder(nn.Module):
    """Small shared CNN for sampled video-frame tensors."""

    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(96, output_dim), nn.ReLU(inplace=True))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = frames.shape
        x = frames.reshape(batch_size * time_steps, channels, height, width)
        x = self.proj(self.net(x))
        return x.reshape(batch_size, time_steps, -1).mean(dim=1)


class FormalFeatureBaseline(nn.Module):
    """Baseline over HoloLens, fisheye, IMU, and scene tensors."""

    def __init__(
        self,
        num_classes: int = 6,
        hidden_dim: int = 128,
        dropout: float = 0.25,
        imu_steps: int = 10,
    ) -> None:
        super().__init__()
        self.hololens_encoder = VideoEncoder(hidden_dim)
        self.fisheye_encoder = VideoEncoder(hidden_dim)
        self.imu_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(imu_steps * 12, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.scene_encoder = nn.Sequential(nn.Linear(2, 32), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        hololens = self.hololens_encoder(features["hololens_frames"])
        fisheye = self.fisheye_encoder(features["fisheye_frames"])
        imu = self.imu_encoder(features["imu"])
        scene = self.scene_encoder(features["scene"])
        return self.classifier(torch.cat([hololens, fisheye, imu, scene], dim=1))

