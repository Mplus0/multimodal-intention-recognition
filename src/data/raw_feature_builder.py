"""Build missing source feature files from raw Member A sample metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.paths import get_path, resolve_path


MODALITY_KEYS: tuple[str, ...] = ("imu", "gesture", "audio", "text", "scene")


class RawFeatureBuildError(RuntimeError):
    """Raised when a raw-data source feature cannot be generated."""


def expected_feature_paths(sample: dict[str, Any], config: dict[str, Any]) -> dict[str, Path]:
    """Return the unified source-feature cache paths for one sample."""
    sample_id = str(sample.get("sample_id") or Path(str(sample.get("video_name", ""))).stem)
    if not sample_id:
        raise RawFeatureBuildError("Sample metadata must include sample_id or video_name.")

    feature_root = get_path("cache", "feature_cache", config=config)
    return {
        "imu": feature_root / "imu_features" / f"imu_features_{sample_id}.npy",
        "gesture": feature_root / "strong_gesture_features" / f"strong_gesture_features_{sample_id}.npy",
        "audio": feature_root / "audio_features" / f"audio_features_{sample_id}.npy",
        "text": feature_root / "text_features" / f"text_features_{sample_id}.npy",
        "scene": feature_root / "scene_features" / f"scene_features_{sample_id}.npy",
    }


def feature_status(paths: dict[str, Path]) -> dict[str, bool]:
    """Return whether each formal modality source-feature file exists."""
    return {key: Path(path).exists() for key, path in paths.items()}


def _raw_path(sample: dict[str, Any], key: str) -> Path:
    raw_paths = sample.get("raw_paths", {})
    value = raw_paths.get(key) if isinstance(raw_paths, dict) else None
    if not value:
        raise RawFeatureBuildError(f"Missing raw path for modality {key}.")
    path = resolve_path(str(value))
    if not path.exists():
        raise RawFeatureBuildError(f"Raw path for modality {key} does not exist: {path}")
    return path


def _output_paths(sample: dict[str, Any], config: dict[str, Any]) -> dict[str, Path]:
    paths = expected_feature_paths(sample, config)
    configured = sample.get("feature_paths", {})
    if isinstance(configured, dict):
        for key in MODALITY_KEYS:
            if configured.get(key):
                paths[key] = resolve_path(str(configured[key]))
    return paths


def ensure_source_features(
    sample: dict[str, Any],
    config: dict[str, Any],
    rebuild: bool = False,
) -> dict[str, Path]:
    """Generate missing five-modality source features from raw files.

    Cache files are reused by default. Missing audio/text dependencies or model
    resources raise clear errors; this function never fabricates official
    zero-filled source features.
    """
    from src.modules.feature_extraction.adapters import (
        build_audio_feature,
        build_imu_feature,
        build_scene_feature,
        build_text_feature,
        build_visual_sequence_feature,
    )

    paths = _output_paths(sample, config)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    sample_id = str(sample.get("sample_id") or Path(str(sample.get("video_name", ""))).stem)
    intent_label = sample.get("intent_label")
    scene_label = sample.get("scene_label")
    intent_value = int(intent_label) if intent_label is not None else None
    scene_value = int(scene_label) if scene_label is not None else None

    if rebuild or not paths["gesture"].exists():
        build_visual_sequence_feature(
            _raw_path(sample, "gesture"),
            paths["gesture"],
            config,
            sample_id=sample_id,
            intent_label=intent_value,
            scene_label=scene_value,
        )
    if rebuild or not paths["scene"].exists():
        build_scene_feature(_raw_path(sample, "scene"), paths["scene"], config)
    if rebuild or not paths["imu"].exists():
        build_imu_feature(_raw_path(sample, "imu"), paths["imu"], config)
    if rebuild or not paths["audio"].exists():
        build_audio_feature(_raw_path(sample, "audio"), paths["audio"], config)
    if rebuild or not paths["text"].exists():
        build_text_feature(_raw_path(sample, "text"), paths["text"], config)

    sample["feature_paths"] = {key: str(path) for key, path in paths.items()}
    sample["feature_status"] = feature_status(paths)
    return paths
