"""Formal five-modality baseline model for Member A.

The model keeps the teacher baseline's clear modality structure: each formal
modality has its own projection, temporal modalities receive time embeddings,
scene is represented as one token, and all tokens are fused by attention.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.data.dataset import MultimodalIntentDataset  # noqa: E402
from src.data.features import MODALITY_KEYS, get_feature_dims, get_target_timesteps  # noqa: E402
from src.utils.paths import load_config  # noqa: E402


DEFAULT_NUM_INTENT_CLASSES = 6
DEFAULT_NUM_SCENE_CLASSES = 2
DEFAULT_NUM_JOINT_CLASSES = 12


class FormalMultimodalBaseline(nn.Module):
    """A compact attention baseline over imu, gesture, audio, text, and scene."""

    def __init__(
        self,
        num_intent_classes: int = DEFAULT_NUM_INTENT_CLASSES,
        num_scene_classes: int | None = DEFAULT_NUM_SCENE_CLASSES,
        num_joint_classes: int | None = DEFAULT_NUM_JOINT_CLASSES,
        feature_dims: dict[str, int] | None = None,
        target_timesteps: int = 10,
        model_dim: int = 128,
        num_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
        ff_multiplier: int = 4,
    ):
        super().__init__()
        self.feature_dims = feature_dims or get_feature_dims()
        self.target_timesteps = int(target_timesteps)
        self.model_dim = int(model_dim)

        self.projections = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.Linear(int(self.feature_dims[key]), model_dim),
                    nn.LayerNorm(model_dim),
                )
                for key in MODALITY_KEYS
            }
        )
        self.time_embedding = nn.Parameter(torch.randn(1, self.target_timesteps, model_dim) * 0.02)
        self.modality_embedding = nn.Parameter(torch.randn(len(MODALITY_KEYS), 1, model_dim) * 0.02)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fusion_norm = nn.LayerNorm(model_dim)

        self.intent_classifier = self._make_classifier(model_dim, num_intent_classes, dropout)
        self.scene_classifier = (
            self._make_classifier(model_dim, int(num_scene_classes), dropout)
            if num_scene_classes is not None
            else None
        )
        self.joint_classifier = (
            self._make_classifier(model_dim, int(num_joint_classes), dropout)
            if num_joint_classes is not None
            else None
        )

    @staticmethod
    def _make_classifier(model_dim: int, num_classes: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes),
        )

    def _temporal_tokens(self, features: torch.Tensor, modality_index: int, key: str) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"{key} must have shape (batch, time, dim), got {tuple(features.shape)}.")
        if features.shape[1] != self.target_timesteps:
            raise ValueError(
                f"{key} must have {self.target_timesteps} timesteps, got {features.shape[1]}."
            )
        tokens = self.projections[key](features.float())
        return tokens + self.time_embedding + self.modality_embedding[modality_index]

    def _scene_token(self, features: torch.Tensor, modality_index: int) -> torch.Tensor:
        if features.ndim == 3 and features.shape[1] == 1:
            features = features[:, 0, :]
        if features.ndim != 2:
            raise ValueError(f"scene must have shape (batch, dim), got {tuple(features.shape)}.")
        token = self.projections["scene"](features.float()).unsqueeze(1)
        return token + self.modality_embedding[modality_index]

    def _tokens_from_features(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [key for key in MODALITY_KEYS if key not in features]
        if missing:
            raise KeyError(f"Missing formal modality features: {missing}")
        tokens = [
            self._temporal_tokens(features["imu"], 0, "imu"),
            self._temporal_tokens(features["gesture"], 1, "gesture"),
            self._temporal_tokens(features["audio"], 2, "audio"),
            self._temporal_tokens(features["text"], 3, "text"),
            self._scene_token(features["scene"], 4),
        ]
        return self.input_dropout(torch.cat(tokens, dim=1))

    def forward(
        self,
        batch_or_features: dict[str, Any],
        modality_mask: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass.

        Args:
            batch_or_features: Either a Dataset batch with a ``features`` field
                or the feature dictionary itself.
            modality_mask: Optional mask from the Dataset. Missing modalities
                are already zero-filled, so the current baseline only keeps this
                argument for interface compatibility.
        """
        features = batch_or_features.get("features", batch_or_features)
        tokens = self._tokens_from_features(features)
        encoded = self.encoder(tokens)
        fused = self.fusion_norm(encoded.mean(dim=1))

        outputs = {
            "intent_logits": self.intent_classifier(fused),
            "fused": fused,
        }
        if self.scene_classifier is not None:
            outputs["scene_logits"] = self.scene_classifier(fused)
        if self.joint_classifier is not None:
            outputs["joint_logits"] = self.joint_classifier(fused)
        return outputs


def build_formal_baseline_from_config(
    config: dict[str, Any] | None = None,
    **overrides,
) -> FormalMultimodalBaseline:
    """Construct the formal baseline using configured modality dimensions."""
    active_config = config or load_config()
    model_config = active_config.get("model", {}).get("formal_baseline", {})
    kwargs = {
        "feature_dims": get_feature_dims(active_config),
        "target_timesteps": get_target_timesteps(active_config),
        "model_dim": int(model_config.get("model_dim", 128)),
        "num_heads": int(model_config.get("num_heads", 4)),
        "depth": int(model_config.get("depth", 2)),
        "dropout": float(model_config.get("dropout", 0.1)),
        "ff_multiplier": int(model_config.get("ff_multiplier", 4)),
        "num_intent_classes": int(model_config.get("num_intent_classes", DEFAULT_NUM_INTENT_CLASSES)),
        "num_scene_classes": int(model_config.get("num_scene_classes", DEFAULT_NUM_SCENE_CLASSES)),
        "num_joint_classes": int(model_config.get("num_joint_classes", DEFAULT_NUM_JOINT_CLASSES)),
    }
    kwargs.update(overrides)
    return FormalMultimodalBaseline(**kwargs)


def _make_smoke_dataset(config: dict[str, Any]) -> MultimodalIntentDataset:
    dims = get_feature_dims(config)
    target_steps = get_target_timesteps(config)
    record = {
        "sample_id": "smoke_sample",
        "user": "user_A",
        "split": "train",
        "intent_label": 1,
        "scene_label": 0,
        "joint_label": "office_select",
        "features": {
            "imu": np.zeros((target_steps, dims["imu"]), dtype=np.float32),
            "gesture": np.zeros((target_steps, dims["gesture"]), dtype=np.float32),
            "audio": np.zeros((target_steps, dims["audio"]), dtype=np.float32),
            "text": np.zeros((target_steps, dims["text"]), dtype=np.float32),
            "scene": np.zeros((dims["scene"],), dtype=np.float32),
        },
    }
    return MultimodalIntentDataset([record], config=config)


def smoke_test(config_path: str = "configs/default.yaml") -> None:
    config = load_config(config_path)
    dataset = _make_smoke_dataset(config)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    model = build_formal_baseline_from_config(config)
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
    print("[smoke_test]")
    print(f"  feature_keys: {sorted(batch['features'].keys())}")
    print(f"  intent_logits: {tuple(outputs['intent_logits'].shape)}")
    if "scene_logits" in outputs:
        print(f"  scene_logits: {tuple(outputs['scene_logits'].shape)}")
    if "joint_logits" in outputs:
        print(f"  joint_logits: {tuple(outputs['joint_logits'].shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal five-modality baseline smoke test.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to project YAML config.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one synthetic Dataset batch forward pass.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        smoke_test(args.config)
        return 0
    print("Use --smoke-test to run a one-batch forward check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
