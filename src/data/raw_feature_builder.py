"""Build missing clean source features from raw multimodal data.

This module is the first-stage dispatcher for the Member A end-to-end pipeline.
It does not change the Dataset contract and it does not implement noise,
missing-modality, or improved-model logic.  Its only job is to ensure that a
sample has clean five-modality source ``.npy`` feature files when raw data and
the required local models are available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.data.raw_extractors import (
    build_audio_feature,
    build_gesture_feature,
    build_imu_feature,
    build_scene_feature,
    build_text_feature,
)
from src.data.raw_extractors.common import RawFeatureExtractionError, sample_id_from


class RawFeatureBuildError(RuntimeError):
    """Raised when source feature files cannot be ensured for one sample."""


MODALITY_KEYS: tuple[str, ...] = ("imu", "gesture", "audio", "text", "scene")
BUILD_ORDER: tuple[str, ...] = ("gesture", "audio", "imu", "text", "scene")
FEATURE_SUBDIRS: dict[str, str] = {
    "imu": "imu_features",
    "gesture": "strong_gesture_features",
    "audio": "audio_features",
    "text": "text_features",
    "scene": "scene_features",
}
FEATURE_PREFIXES: dict[str, str] = {
    "imu": "imu_features",
    "gesture": "strong_gesture_features",
    "audio": "audio_features",
    "text": "text_features",
    "scene": "scene_features",
}
GESTURE_METADATA_PREFIX = "metadata_strong_gesture"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_config_if_needed(config: dict[str, Any] | None) -> dict[str, Any]:
    if config is not None:
        return config
    from src.utils.paths import load_config

    return load_config()


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return _project_root() / path


def _configured_bool(config: dict[str, Any], section: str, key: str, default: bool) -> bool:
    value = config.get(section, {}).get(key, default)
    return bool(value)


def _feature_cache_root(config: dict[str, Any]) -> Path:
    value = config.get("cache", {}).get("feature_cache")
    if not value:
        raise RawFeatureBuildError("cache.feature_cache is not configured.")
    return _resolve_path(str(value))


def _target_source_feature_path(sample: dict[str, Any], modality: str, config: dict[str, Any]) -> Path:
    sample_id = sample_id_from(sample)
    root = _feature_cache_root(config)
    subdir = FEATURE_SUBDIRS[modality]
    prefix = FEATURE_PREFIXES[modality]
    return root / subdir / f"{prefix}_{sample_id}.npy"


def _target_gesture_metadata_path(sample: dict[str, Any], config: dict[str, Any]) -> Path:
    sample_id = sample_id_from(sample)
    return (
        _feature_cache_root(config)
        / FEATURE_SUBDIRS["gesture"]
        / f"{GESTURE_METADATA_PREFIX}_{sample_id}.npy"
    )


def resolve_source_feature_paths(
    sample: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Return expected source feature paths for all formal modalities.

    Existing ``sample["feature_paths"]`` values are respected.  Missing values
    fall back to the standard ``cache.feature_cache`` subdirectories.
    """
    config = _load_config_if_needed(config)

    feature_paths = sample.get("feature_paths")
    resolved: dict[str, Path] = {}
    for modality in MODALITY_KEYS:
        configured = feature_paths.get(modality) if isinstance(feature_paths, dict) else None
        resolved[modality] = (
            _resolve_path(str(configured))
            if configured
            else _target_source_feature_path(sample, modality, config)
        )
    return resolved


def resolve_gesture_metadata_path(
    sample: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> Path:
    """Return the expected gesture metadata path for a sample."""
    config = _load_config_if_needed(config)

    feature_paths = sample.get("feature_paths")
    if isinstance(feature_paths, dict):
        for key in ("gesture_metadata", "metadata", "gesture_meta"):
            value = feature_paths.get(key)
            if value:
                return _resolve_path(str(value))
    return _target_gesture_metadata_path(sample, config)


def missing_source_features(
    sample: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Return source feature paths that are currently absent on disk."""
    paths = resolve_source_feature_paths(sample, config)
    return {modality: path for modality, path in paths.items() if not path.exists()}


def _call_extractor(
    modality: str,
    builder: Callable[..., Path],
    sample: dict[str, Any],
    config: dict[str, Any],
    output_path: Path,
    gesture_metadata_path: Path,
) -> Path:
    try:
        if modality == "gesture":
            return builder(sample, config, output_path, metadata_path=gesture_metadata_path)
        return builder(sample, config, output_path, gesture_metadata_path)
    except RawFeatureExtractionError as error:
        raise RawFeatureBuildError(
            f"Failed to build source feature for modality '{modality}' "
            f"of sample '{sample_id_from(sample)}': {error}"
        ) from error
    except Exception as error:
        raise RawFeatureBuildError(
            f"Unexpected error while building source feature for modality '{modality}' "
            f"of sample '{sample_id_from(sample)}': {error}"
        ) from error


def build_missing_source_features(
    sample: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Build any missing clean source features for one sample.

    Existing source feature files are reused unless
    ``feature_builder.rebuild_source_features`` is true in config.
    """
    config = _load_config_if_needed(config)

    if not _configured_bool(config, "feature_builder", "enabled", True):
        raise RawFeatureBuildError(
            "Raw feature builder is disabled by feature_builder.enabled=false."
        )

    source_paths = resolve_source_feature_paths(sample, config)
    gesture_metadata_path = resolve_gesture_metadata_path(sample, config)
    rebuild = _configured_bool(config, "feature_builder", "rebuild_source_features", False)
    skip_existing = _configured_bool(config, "feature_builder", "skip_existing", True)

    builders: dict[str, Callable[..., Path]] = {
        "gesture": build_gesture_feature,
        "audio": build_audio_feature,
        "imu": build_imu_feature,
        "text": build_text_feature,
        "scene": build_scene_feature,
    }

    for modality in BUILD_ORDER:
        output_path = source_paths[modality]
        if output_path.exists() and skip_existing and not rebuild:
            continue
        _call_extractor(
            modality,
            builders[modality],
            sample,
            config,
            output_path,
            gesture_metadata_path,
        )

    missing = {key: path for key, path in source_paths.items() if not path.exists()}
    if missing:
        details = "; ".join(f"{key}: {path}" for key, path in missing.items())
        raise RawFeatureBuildError(
            f"Raw feature builder finished but source features are still missing: {details}"
        )

    return source_paths


def ensure_sample_source_features(
    sample: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Ensure one sample has all five clean source feature files.

    Returns:
        A mapping from formal modality key to existing source feature path.
    """
    config = _load_config_if_needed(config)

    source_paths = resolve_source_feature_paths(sample, config)
    if all(path.exists() for path in source_paths.values()):
        return source_paths
    return build_missing_source_features(sample, config)


__all__ = [
    "BUILD_ORDER",
    "FEATURE_PREFIXES",
    "FEATURE_SUBDIRS",
    "MODALITY_KEYS",
    "RawFeatureBuildError",
    "build_missing_source_features",
    "ensure_sample_source_features",
    "missing_source_features",
    "resolve_gesture_metadata_path",
    "resolve_source_feature_paths",
]
