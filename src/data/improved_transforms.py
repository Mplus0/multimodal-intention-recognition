"""Training-only transforms for the improved robust multimodal model."""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.data.transforms import DEFAULT_MODALITIES, validate_modalities


def _stable_seed(base_seed: int, *parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return (int(base_seed) + int(digest[:8], 16)) % (2**31)


def _clone_sample(sample: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(sample)
    cloned["features"] = dict(sample.get("features", {}))
    if "modality_mask" in sample:
        cloned["modality_mask"] = dict(sample["modality_mask"])
    return cloned


class RandomModalityDropoutTransform:
    """Randomly zero one or more modalities during training.

    This transform is intended only for improved-model training. It follows the
    same missing-modality representation as ``MissingModalityTransform``:
    dropped modalities are zero-filled and marked as ``modality_mask=False``.
    """

    def __init__(
        self,
        drop_prob: float = 0.3,
        max_drop_modalities: int = 2,
        modalities: list[str] | tuple[str, ...] | None = None,
        seed: int = 42,
        min_keep_modalities: int = 1,
    ):
        if drop_prob < 0 or drop_prob > 1:
            raise ValueError(f"drop_prob must be between 0 and 1, got {drop_prob}.")
        self.modalities = tuple(modalities or DEFAULT_MODALITIES)
        validate_modalities(list(self.modalities))
        if not self.modalities:
            raise ValueError("At least one modality is required.")

        self.drop_prob = float(drop_prob)
        self.max_drop_modalities = max(0, int(max_drop_modalities))
        self.seed = int(seed)
        self.min_keep_modalities = max(1, int(min_keep_modalities))

        max_allowed_drop = max(0, len(self.modalities) - self.min_keep_modalities)
        self.max_drop_modalities = min(self.max_drop_modalities, max_allowed_drop)

    def _sample_dropped_modalities(self, sample: dict[str, Any]) -> tuple[str, ...]:
        if self.drop_prob <= 0 or self.max_drop_modalities <= 0:
            return ()

        rng = random.Random(
            _stable_seed(
                self.seed,
                sample.get("sample_id", ""),
                sample.get("split", ""),
                "random_modality_dropout",
            )
        )
        if rng.random() >= self.drop_prob:
            return ()

        drop_count = rng.randint(1, self.max_drop_modalities)
        return tuple(rng.sample(list(self.modalities), k=drop_count))

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        transformed = _clone_sample(sample)
        features = transformed.get("features", {})
        modality_mask = transformed.setdefault("modality_mask", {})
        dropped_modalities = self._sample_dropped_modalities(sample)

        for modality in dropped_modalities:
            if modality not in features:
                raise KeyError(f"Sample features missing modality: {modality}")
            feature = features[modality]
            if not torch.is_tensor(feature):
                feature = torch.as_tensor(feature)
            features[modality] = torch.zeros_like(feature)
            modality_mask[modality] = torch.tensor(False, dtype=torch.bool)

        transformed["dropped_modalities"] = list(dropped_modalities)
        return transformed


def build_random_modality_dropout_from_config(
    improved_config: dict[str, Any] | None = None,
    *,
    seed: int = 42,
) -> RandomModalityDropoutTransform | None:
    """Build a training transform from ``configs/improved_model.yaml`` style config."""
    config = improved_config or {}
    dropout_config = config.get("training", {}).get("modality_dropout", {})
    if not isinstance(dropout_config, dict):
        raise ValueError("training.modality_dropout must be a mapping when provided.")
    if not bool(dropout_config.get("enabled", True)):
        return None

    modalities = dropout_config.get("modalities", DEFAULT_MODALITIES)
    if not isinstance(modalities, list):
        raise ValueError("training.modality_dropout.modalities must be a list when provided.")

    return RandomModalityDropoutTransform(
        drop_prob=float(dropout_config.get("drop_prob", 0.3)),
        max_drop_modalities=int(dropout_config.get("max_drop_modalities", 2)),
        modalities=modalities,
        seed=int(dropout_config.get("seed", seed)),
        min_keep_modalities=int(dropout_config.get("min_keep_modalities", 1)),
    )


def _make_smoke_sample() -> dict[str, Any]:
    return {
        "sample_id": "dropout_smoke",
        "split": "train",
        "features": {
            "imu": torch.ones(10, 12),
            "gesture": torch.ones(10, 768),
            "audio": torch.ones(10, 39),
            "text": torch.ones(10, 384),
            "scene": torch.ones(768),
        },
        "modality_mask": {
            key: torch.tensor(True, dtype=torch.bool)
            for key in DEFAULT_MODALITIES
        },
    }


def smoke_test() -> None:
    """Run a deterministic transform check without loading real data."""
    transform = RandomModalityDropoutTransform(drop_prob=1.0, max_drop_modalities=2, seed=42)
    sample = transform(_make_smoke_sample())
    print("[random_modality_dropout_smoke_test]")
    print(f"  dropped_modalities: {sample['dropped_modalities']}")
    for key in DEFAULT_MODALITIES:
        mask_value = bool(sample["modality_mask"][key].item())
        feature_sum = float(sample["features"][key].float().sum().item())
        print(f"  {key}: mask={mask_value} feature_sum={feature_sum:.1f}")


if __name__ == "__main__":
    smoke_test()
