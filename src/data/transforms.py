"""Experiment-matrix helpers for baseline data perturbation plans.

This module intentionally does not import or assume any Dataset/DataLoader
implementation. Real sample transforms must wait until Member A finalizes the
data structure and train/test API.
"""

from __future__ import annotations

from itertools import combinations


DEFAULT_MODALITIES = ["imu", "gesture", "audio", "text", "scene"]

_MEMBER_A_TODO = (
    "TODO(member A): connect to the finalized train/test API before running "
    "this experiment."
)


def validate_modalities(modalities: list[str]) -> None:
    """Validate modality names against the project baseline modality list."""
    unknown_modalities = [name for name in modalities if name not in DEFAULT_MODALITIES]
    if unknown_modalities:
        allowed = ", ".join(DEFAULT_MODALITIES)
        unknown = ", ".join(unknown_modalities)
        raise ValueError(f"Unknown modalities: {unknown}. Allowed modalities: {allowed}.")


def _validate_noise_ratios(noise_ratios: list[float]) -> None:
    for ratio in noise_ratios:
        if not isinstance(ratio, (int, float)):
            raise TypeError(f"Noise ratio must be numeric, got {type(ratio).__name__}.")
        if ratio < 0 or ratio > 1:
            raise ValueError(f"Noise ratio must be between 0 and 1, got {ratio}.")


def build_noise_experiment_matrix(
    modalities: list[str],
    noise_ratios: list[float],
) -> list[dict]:
    """Return single-target-modality noise experiment settings.

    The returned matrix is a plan only; it does not apply noise to any data.
    """
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


def apply_modal_noise(sample, target_modality: str, noise_ratio: float):
    """Placeholder for real modal-noise injection."""
    validate_modalities([target_modality])
    _validate_noise_ratios([noise_ratio])
    # TODO(member A): decide whether noise should be applied to raw data or extracted features.
    raise NotImplementedError(_MEMBER_A_TODO)


def apply_missing_modalities(sample, missing_modalities: list[str]):
    """Placeholder for real missing-modality handling."""
    validate_modalities(missing_modalities)
    # TODO(member A): decide whether missing modality means zero-fill, mask, or removing input fields.
    raise NotImplementedError(_MEMBER_A_TODO)
