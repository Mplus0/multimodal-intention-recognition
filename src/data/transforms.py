"""Experiment-matrix and sample transforms for baseline perturbation runs."""

from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Any

import torch


DEFAULT_MODALITIES = ["imu", "gesture", "audio", "text", "scene"]


def validate_modalities(modalities: list[str] | tuple[str, ...]) -> None:
    """Validate modality names against the project baseline modality list."""
    unknown_modalities = [name for name in modalities if name not in DEFAULT_MODALITIES]
    if unknown_modalities:
        allowed = ", ".join(DEFAULT_MODALITIES)
        unknown = ", ".join(unknown_modalities)
        raise ValueError(f"Unknown modalities: {unknown}. Allowed modalities: {allowed}.")


def _validate_noise_ratios(noise_ratios: list[float] | tuple[float, ...]) -> None:
    for ratio in noise_ratios:
        if not isinstance(ratio, (int, float)):
            raise TypeError(f"Noise ratio must be numeric, got {type(ratio).__name__}.")
        if ratio < 0 or ratio > 1:
            raise ValueError(f"Noise ratio must be between 0 and 1, got {ratio}.")


def build_noise_experiment_matrix(
    modalities: list[str],
    noise_ratios: list[float],
) -> list[dict]:
    """Return single-target-modality noise experiment settings."""
    validate_modalities(modalities)
    _validate_noise_ratios(noise_ratios)
    return [
        {"target_modality": modality, "noise_ratio": float(noise_ratio)}
        for modality in modalities
        for noise_ratio in noise_ratios
    ]


def build_missing_modality_matrix(
    modalities: list[str],
    drop_one: bool = True,
    drop_two: bool = True,
) -> list[dict]:
    """Return single- and double-missing-modality experiment settings."""
    validate_modalities(modalities)
    matrix: list[dict] = []
    if drop_one:
        matrix.extend({"missing_modalities": [modality]} for modality in modalities)
    if drop_two:
        matrix.extend(
            {"missing_modalities": list(pair)}
            for pair in combinations(modalities, 2)
        )
    return matrix


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


class ModalNoiseTransform:
    """Add deterministic feature-level Gaussian noise to one formal modality.

    The course requirement describes noise on raw single-modality data. This
    transform implements the baseline integration point at Dataset feature
    level while keeping sample shapes and keys unchanged. If a later raw-data
    perturbation hook is added, the same experiment matrix can be reused there.
    """

    def __init__(
        self,
        target_modality: str,
        noise_ratio: float,
        seed: int = 42,
        noise_std_scale: float = 0.1,
    ):
        validate_modalities([target_modality])
        _validate_noise_ratios([noise_ratio])
        self.target_modality = target_modality
        self.noise_ratio = float(noise_ratio)
        self.seed = int(seed)
        self.noise_std_scale = float(noise_std_scale)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        transformed = _clone_sample(sample)
        features = transformed.get("features", {})
        if self.target_modality not in features:
            raise KeyError(f"Sample features missing modality: {self.target_modality}")

        feature = features[self.target_modality]
        if not torch.is_tensor(feature):
            feature = torch.as_tensor(feature)

        if self.noise_ratio <= 0:
            features[self.target_modality] = feature.clone()
            return transformed

        generator = torch.Generator(device=feature.device)
        generator.manual_seed(
            _stable_seed(
                self.seed,
                sample.get("sample_id", ""),
                self.target_modality,
                self.noise_ratio,
            )
        )
        random_values = torch.rand(
            feature.shape,
            generator=generator,
            device=feature.device,
            dtype=torch.float32,
        )
        mask = random_values < self.noise_ratio
        std = feature.float().std(unbiased=False).clamp_min(1e-6)
        noise = torch.randn(
            feature.shape,
            generator=generator,
            device=feature.device,
            dtype=torch.float32,
        ) * std * self.noise_std_scale
        features[self.target_modality] = torch.where(mask, feature.float() + noise, feature.float()).to(feature.dtype)
        return transformed


class MissingModalityTransform:
    """Represent missing modalities by zero-fill plus ``modality_mask=False``."""

    def __init__(self, missing_modalities: list[str] | tuple[str, ...]):
        validate_modalities(list(missing_modalities))
        self.missing_modalities = tuple(missing_modalities)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        transformed = _clone_sample(sample)
        features = transformed.get("features", {})
        modality_mask = transformed.setdefault("modality_mask", {})

        for modality in self.missing_modalities:
            if modality not in features:
                raise KeyError(f"Sample features missing modality: {modality}")
            feature = features[modality]
            if not torch.is_tensor(feature):
                feature = torch.as_tensor(feature)
            features[modality] = torch.zeros_like(feature)
            modality_mask[modality] = torch.tensor(False, dtype=torch.bool)
        return transformed


class ComposeTransforms:
    """Apply multiple sample transforms in order."""

    def __init__(self, transforms):
        self.transforms = [transform for transform in transforms if transform is not None]

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        result = sample
        for transform in self.transforms:
            result = transform(result)
        return result


def apply_modal_noise(sample, target_modality: str, noise_ratio: float):
    """Apply feature-level modal noise to one sample."""
    return ModalNoiseTransform(target_modality, noise_ratio)(sample)


def apply_missing_modalities(sample, missing_modalities: list[str]):
    """Apply zero-fill missing-modality handling to one sample."""
    return MissingModalityTransform(missing_modalities)(sample)
