"""Reliability-gated improved model for robust five-modality fusion.

This module is intentionally independent from ``formal_baseline.py``. It keeps
the same Dataset batch contract while adding modality reliability gates and
explicit ``modality_mask`` handling for missing-modality robustness.
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
from src.models.formal_baseline import (  # noqa: E402
    DEFAULT_NUM_INTENT_CLASSES,
    DEFAULT_NUM_JOINT_CLASSES,
    DEFAULT_NUM_SCENE_CLASSES,
)
from src.utils.paths import load_config  # noqa: E402


class ReliabilityGatedMultimodalModel(nn.Module):
    """Transformer fusion model with per-modality reliability gates.

    The model follows the formal baseline input shapes:

    - ``imu``, ``gesture``, ``audio``, ``text``: ``(batch, time, dim)``
    - ``scene``: ``(batch, dim)``

    Unlike the baseline, this model converts ``modality_mask`` into token-level
    weights and uses reliability-gated weighted pooling after the encoder.
    """

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
        use_reliability_gate: bool = True,
        use_modality_mask: bool = True,
        gate_bias_init: float = 2.0,
    ):
        super().__init__()
        self.feature_dims = feature_dims or get_feature_dims()
        self.target_timesteps = int(target_timesteps)
        self.model_dim = int(model_dim)
        self.use_reliability_gate = bool(use_reliability_gate)
        self.use_modality_mask = bool(use_modality_mask)

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

        self.reliability_gates = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.LayerNorm(model_dim),
                    nn.Linear(model_dim, max(1, model_dim // 2)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(max(1, model_dim // 2), 1),
                    nn.Sigmoid(),
                )
                for key in MODALITY_KEYS
            }
        )
        self._init_gate_bias(gate_bias_init)

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

    def _init_gate_bias(self, gate_bias_init: float) -> None:
        """Initialize gates to trust present modalities at the start."""
        for gate in self.reliability_gates.values():
            final_linear = gate[-2]
            if isinstance(final_linear, nn.Linear):
                nn.init.constant_(final_linear.bias, float(gate_bias_init))

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

    def _tokens_by_modality(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = [key for key in MODALITY_KEYS if key not in features]
        if missing:
            raise KeyError(f"Missing formal modality features: {missing}")
        return {
            "imu": self._temporal_tokens(features["imu"], 0, "imu"),
            "gesture": self._temporal_tokens(features["gesture"], 1, "gesture"),
            "audio": self._temporal_tokens(features["audio"], 2, "audio"),
            "text": self._temporal_tokens(features["text"], 3, "text"),
            "scene": self._scene_token(features["scene"], 4),
        }

    def _modality_mask(
        self,
        batch_or_features: dict[str, Any],
        features: dict[str, torch.Tensor],
        modality_mask: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        if modality_mask is None and "features" in batch_or_features:
            candidate = batch_or_features.get("modality_mask")
            if isinstance(candidate, dict):
                modality_mask = candidate

        first_feature = features[MODALITY_KEYS[0]]
        batch_size = int(first_feature.shape[0])
        device = first_feature.device
        masks: dict[str, torch.Tensor] = {}
        for key in MODALITY_KEYS:
            value = modality_mask.get(key) if isinstance(modality_mask, dict) else None
            if value is None or not self.use_modality_mask:
                masks[key] = torch.ones(batch_size, dtype=torch.bool, device=device)
            else:
                masks[key] = value.to(device=device, dtype=torch.bool).view(batch_size)
        return masks

    def _reliability_scores(
        self,
        tokens_by_modality: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        scores: dict[str, torch.Tensor] = {}
        for key, tokens in tokens_by_modality.items():
            summary = tokens.mean(dim=1)
            if self.use_reliability_gate:
                score = self.reliability_gates[key](summary)
            else:
                score = torch.ones(summary.shape[0], 1, dtype=summary.dtype, device=summary.device)
            scores[key] = score * masks[key].float().unsqueeze(1)
        return scores

    def _token_weights(
        self,
        tokens_by_modality: dict[str, torch.Tensor],
        scores: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        weights = []
        for key in MODALITY_KEYS:
            token_count = int(tokens_by_modality[key].shape[1])
            weights.append(scores[key].unsqueeze(1).expand(-1, token_count, -1))
        return torch.cat(weights, dim=1)

    def _padding_mask(
        self,
        tokens_by_modality: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        token_masks = []
        for key in MODALITY_KEYS:
            token_count = int(tokens_by_modality[key].shape[1])
            token_masks.append(masks[key].unsqueeze(1).expand(-1, token_count))
        keep_mask = torch.cat(token_masks, dim=1)
        if bool((~keep_mask).all(dim=1).any()):
            keep_mask = keep_mask.clone()
            keep_mask[(~keep_mask).all(dim=1), :] = True
        return ~keep_mask

    def forward(
        self,
        batch_or_features: dict[str, Any],
        modality_mask: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Run a forward pass with reliability-gated fusion."""
        features = batch_or_features.get("features", batch_or_features)
        tokens_by_modality = self._tokens_by_modality(features)
        masks = self._modality_mask(batch_or_features, features, modality_mask)
        scores = self._reliability_scores(tokens_by_modality, masks)

        tokens = self.input_dropout(torch.cat([tokens_by_modality[key] for key in MODALITY_KEYS], dim=1))
        token_weights = self._token_weights(tokens_by_modality, scores)
        padding_mask = self._padding_mask(tokens_by_modality, masks) if self.use_modality_mask else None
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        fused = (encoded * token_weights).sum(dim=1) / token_weights.sum(dim=1).clamp_min(1e-6)
        fused = self.fusion_norm(fused)

        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "intent_logits": self.intent_classifier(fused),
            "fused": fused,
            "reliability_scores": scores,
        }
        if self.scene_classifier is not None:
            outputs["scene_logits"] = self.scene_classifier(fused)
        if self.joint_classifier is not None:
            outputs["joint_logits"] = self.joint_classifier(fused)
        return outputs


def build_improved_model_from_config(
    base_config: dict[str, Any] | None = None,
    improved_config: dict[str, Any] | None = None,
    **overrides,
) -> ReliabilityGatedMultimodalModel:
    """Construct the improved model from project and improved-model configs."""
    active_base_config = base_config or load_config()
    active_improved_config = improved_config or {}
    baseline_config = active_base_config.get("model", {}).get("formal_baseline", {})
    improved_model_config = active_improved_config.get("model", {}).get("improved", {})

    def config_value(key: str, default: Any) -> Any:
        return improved_model_config.get(key, baseline_config.get(key, default))

    kwargs = {
        "feature_dims": get_feature_dims(active_base_config),
        "target_timesteps": get_target_timesteps(active_base_config),
        "model_dim": int(config_value("model_dim", 128)),
        "num_heads": int(config_value("num_heads", 4)),
        "depth": int(config_value("depth", 2)),
        "dropout": float(config_value("dropout", 0.1)),
        "ff_multiplier": int(config_value("ff_multiplier", 4)),
        "use_reliability_gate": bool(config_value("use_reliability_gate", True)),
        "use_modality_mask": bool(config_value("use_modality_mask", True)),
        "gate_bias_init": float(config_value("gate_bias_init", 2.0)),
        "num_intent_classes": int(config_value("num_intent_classes", DEFAULT_NUM_INTENT_CLASSES)),
        "num_scene_classes": int(config_value("num_scene_classes", DEFAULT_NUM_SCENE_CLASSES)),
        "num_joint_classes": int(config_value("num_joint_classes", DEFAULT_NUM_JOINT_CLASSES)),
    }
    kwargs.update(overrides)
    return ReliabilityGatedMultimodalModel(**kwargs)


def _make_smoke_dataset(config: dict[str, Any]) -> MultimodalIntentDataset:
    dims = get_feature_dims(config)
    target_steps = get_target_timesteps(config)
    record = {
        "sample_id": "improved_smoke_sample",
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
    """Run one synthetic Dataset batch through the improved model."""
    config = load_config(config_path)
    dataset = _make_smoke_dataset(config)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    model = build_improved_model_from_config(config)
    model.eval()
    with torch.no_grad():
        outputs = model(batch)

    print("[improved_model_smoke_test]")
    print(f"  feature_keys: {sorted(batch['features'].keys())}")
    print(f"  intent_logits: {tuple(outputs['intent_logits'].shape)}")
    print(f"  fused: {tuple(outputs['fused'].shape)}")
    reliability_scores = outputs["reliability_scores"]
    if isinstance(reliability_scores, dict):
        for key in MODALITY_KEYS:
            print(f"  reliability_{key}: {tuple(reliability_scores[key].shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reliability-gated improved model smoke test.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to base project YAML config.")
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
