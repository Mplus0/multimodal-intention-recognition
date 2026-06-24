"""Five-modality feature loading helpers for the Member A pipeline.

This module migrates reusable preprocessing ideas from the teacher baseline
without keeping its hard-coded local paths. It works from sample metadata and
project-relative configuration. Cache files may speed repeated checks, but
source feature paths are still required when a cached sample is unavailable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils.paths import get_path, load_config, resolve_path


MODALITY_KEYS: tuple[str, ...] = ("imu", "gesture", "audio", "text", "scene")
DEFAULT_FEATURE_DIMS: dict[str, int] = {
    "imu": 12,
    "gesture": 768,
    "audio": 39,
    "text": 384,
    "scene": 768,
}
DEFAULT_TARGET_TIMESTEPS = 10
UNKNOWN_LABELS = {"未知", "unknown", "Unknown"}


@dataclass
class FeatureBundle:
    """Feature payload for one metadata sample."""

    sample_id: str
    features: dict[str, np.ndarray]
    intent_labels: np.ndarray | None = None
    scene_labels: np.ndarray | None = None
    approx_timestamps: np.ndarray | None = None
    source_paths: dict[str, Path] | None = None


class FeatureBuildError(RuntimeError):
    """Raised when sample metadata is insufficient for feature construction."""


def get_modality_keys(config: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Return formal modality keys and validate they match the project contract."""
    if config is None:
        config = load_config()
    keys = tuple(config.get("modalities", {}).get("keys", MODALITY_KEYS))
    if keys != MODALITY_KEYS:
        raise ValueError(
            "Formal modality keys must be exactly "
            f"{MODALITY_KEYS}, got {keys}."
        )
    return keys


def get_feature_dims(config: dict[str, Any] | None = None) -> dict[str, int]:
    """Return configured feature dimensions with teacher-baseline defaults."""
    if config is None:
        config = load_config()
    configured = config.get("modalities", {}).get("feature_dims", {})
    dims = dict(DEFAULT_FEATURE_DIMS)
    for key in MODALITY_KEYS:
        if key in configured:
            dims[key] = int(configured[key])
    return dims


def get_target_timesteps(config: dict[str, Any] | None = None) -> int:
    """Return configured target sequence length."""
    if config is None:
        config = load_config()
    return int(config.get("modalities", {}).get("target_timesteps", DEFAULT_TARGET_TIMESTEPS))


def normalize_sequence_length(
    sequence: np.ndarray,
    target_steps: int,
    feat_dim: int,
    long_mode: str = "truncate",
) -> np.ndarray:
    """Pad or trim one 2D sequence to a fixed number of timesteps."""
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D sequence, got shape {array.shape}.")
    if array.shape[1] != feat_dim:
        raise ValueError(f"Expected feature dim {feat_dim}, got shape {array.shape}.")

    current_steps = array.shape[0]
    if current_steps == target_steps:
        return array

    if current_steps > target_steps:
        if long_mode == "even":
            indices = np.linspace(0, current_steps - 1, target_steps, dtype=int)
            return array[indices]
        return array[:target_steps]

    pad = np.zeros((target_steps - current_steps, feat_dim), dtype=np.float32)
    return np.vstack((array, pad))


def normalize_dense_modality(
    features: np.ndarray,
    target_steps: int,
    feat_dim: int,
    long_mode: str = "truncate",
) -> np.ndarray:
    """Normalize a batch of dense sequence features to ``(N, T, D)``."""
    array = np.asarray(features, dtype=np.float32)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError(f"Expected 3D dense modality features, got {array.shape}.")
    return np.stack(
        [
            normalize_sequence_length(sample, target_steps, feat_dim, long_mode=long_mode)
            for sample in array
        ]
    ).astype(np.float32)


def normalize_audio_modality(
    audio_samples: np.ndarray,
    target_steps: int = DEFAULT_TARGET_TIMESTEPS,
    feat_dim: int = DEFAULT_FEATURE_DIMS["audio"],
) -> np.ndarray:
    """Normalize teacher-style audio feature payloads to ``(N, T, D)``."""
    normalized = []
    for sample in audio_samples:
        feature = sample["feature"] if isinstance(sample, dict) else sample
        normalized.append(
            normalize_sequence_length(
                np.asarray(feature, dtype=np.float32),
                target_steps,
                feat_dim,
                long_mode="even",
            )
        )
    return np.stack(normalized).astype(np.float32)


def fit_scalers(
    train_features: dict[str, np.ndarray],
    feature_dims: dict[str, int] | None = None,
) -> dict[str, StandardScaler]:
    """Fit one ``StandardScaler`` per formal modality on training features."""
    dims = feature_dims or DEFAULT_FEATURE_DIMS
    scalers: dict[str, StandardScaler] = {}
    for key in MODALITY_KEYS:
        if key not in train_features:
            raise KeyError(f"Missing modality in train_features: {key}")
        scaler = StandardScaler()
        scaler.fit(train_features[key].reshape(-1, int(dims[key])))
        scalers[key] = scaler
    return scalers


def apply_scalers(
    features: dict[str, np.ndarray],
    scalers: dict[str, StandardScaler],
    feature_dims: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    """Apply fitted per-modality scalers without changing feature shapes."""
    dims = feature_dims or DEFAULT_FEATURE_DIMS
    transformed: dict[str, np.ndarray] = {}
    for key in MODALITY_KEYS:
        if key not in features:
            raise KeyError(f"Missing modality in features: {key}")
        flat = features[key].reshape(-1, int(dims[key]))
        transformed[key] = scalers[key].transform(flat).reshape(features[key].shape).astype(np.float32)
    return transformed


class FeatureCache:
    """Small NPZ cache for complete five-modality sample features."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, sample_id: str) -> Path:
        digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npz"

    def load(self, sample_id: str) -> FeatureBundle | None:
        path = self._path(sample_id)
        if not path.exists():
            return None
        payload = np.load(path, allow_pickle=True)
        features = {key: payload[key].astype(np.float32) for key in MODALITY_KEYS}
        intent_labels = payload["intent_labels"] if "intent_labels" in payload.files else None
        scene_labels = payload["scene_labels"] if "scene_labels" in payload.files else None
        approx_timestamps = payload["approx_timestamps"] if "approx_timestamps" in payload.files else None
        return FeatureBundle(
            sample_id=sample_id,
            features=features,
            intent_labels=intent_labels,
            scene_labels=scene_labels,
            approx_timestamps=approx_timestamps,
        )

    def save(self, bundle: FeatureBundle) -> Path:
        path = self._path(bundle.sample_id)
        arrays: dict[str, np.ndarray] = {key: value for key, value in bundle.features.items()}
        if bundle.intent_labels is not None:
            arrays["intent_labels"] = bundle.intent_labels
        if bundle.scene_labels is not None:
            arrays["scene_labels"] = bundle.scene_labels
        if bundle.approx_timestamps is not None:
            arrays["approx_timestamps"] = bundle.approx_timestamps
        np.savez_compressed(path, **arrays)
        return path


class SceneFeatureCache:
    """Cache scene feature vectors by sample/timestamp.

    The cache never invents scene features. Callers must pass a builder when a
    cached value does not already exist.
    """

    def __init__(self, cache_dir: Path, scene_dim: int = DEFAULT_FEATURE_DIMS["scene"]):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.scene_dim = int(scene_dim)

    def _path(self, sample_id: str, timestamp_value: str) -> Path:
        key = f"{sample_id}|{timestamp_value}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def get_or_build(
        self,
        sample_id: str,
        timestamp_value: str,
        builder,
    ) -> np.ndarray:
        path = self._path(sample_id, timestamp_value)
        if path.exists():
            return np.load(path).astype(np.float32)
        feature = np.asarray(builder(timestamp_value), dtype=np.float32)
        if feature.shape != (self.scene_dim,):
            raise ValueError(
                f"Scene builder must return shape ({self.scene_dim},), got {feature.shape}."
            )
        np.save(path, feature)
        return feature


def get_default_feature_cache(config: dict[str, Any] | None = None) -> FeatureCache:
    """Create the default complete-sample feature cache from config."""
    if config is None:
        config = load_config()
    return FeatureCache(get_path("cache", "feature_cache", config=config))


def get_default_scene_cache(config: dict[str, Any] | None = None) -> SceneFeatureCache:
    """Create the default scene cache from config."""
    if config is None:
        config = load_config()
    dims = get_feature_dims(config)
    return SceneFeatureCache(get_path("cache", "scene_cache", config=config), dims["scene"])


def _load_npy_payload(path: Path) -> Any:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.shape == () and payload.dtype == object:
        return payload.item()
    return payload


def _extract_features(payload: Any) -> np.ndarray:
    if isinstance(payload, dict):
        if "features" in payload:
            return np.asarray(payload["features"])
        if "feature" in payload:
            return np.asarray(payload["feature"])
    return np.asarray(payload)


def _extract_optional_array(payload: Any, key: str) -> np.ndarray | None:
    if isinstance(payload, dict) and key in payload:
        return np.asarray(payload[key])
    return None


def _metadata_path(sample: dict[str, Any], modality: str) -> Path | None:
    feature_paths = sample.get("feature_paths", {})
    candidates = [
        feature_paths.get(modality) if isinstance(feature_paths, dict) else None,
        sample.get(f"{modality}_feature_path"),
    ]
    for value in candidates:
        if value:
            return resolve_path(str(value))
    return None


def _sample_id(sample: dict[str, Any]) -> str:
    for key in ("sample_id", "video_name", "id"):
        if sample.get(key):
            return str(sample[key])
    raise FeatureBuildError("Sample metadata must include sample_id, video_name, or id.")


def _normalize_scene_features(scene_features: np.ndarray, scene_dim: int) -> np.ndarray:
    array = np.asarray(scene_features, dtype=np.float32)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2 or array.shape[1] != scene_dim:
        raise ValueError(f"Expected scene features with shape (N, {scene_dim}), got {array.shape}.")
    return array.astype(np.float32)


def _valid_label_mask(labels: np.ndarray | None, sample_count: int) -> np.ndarray:
    if labels is None:
        return np.ones(sample_count, dtype=bool)
    labels = labels[:sample_count]
    return np.array([str(label) not in UNKNOWN_LABELS for label in labels], dtype=bool)


def load_or_build_sample_features(
    sample: dict[str, Any],
    config: dict[str, Any] | None = None,
    cache: FeatureCache | None = None,
    use_cache: bool = True,
    rebuild_cache: bool = False,
) -> FeatureBundle:
    """Load one sample's five formal modality features from metadata.

    The current stage expects metadata to provide source feature files, usually
    generated by teacher extraction scripts or later raw-data builders. If a
    complete feature cache is absent, missing source paths produce a clear
    error instead of fabricated arrays.
    """
    if config is None:
        config = load_config()
    get_modality_keys(config)
    dims = get_feature_dims(config)
    target_steps = get_target_timesteps(config)
    sample_id = _sample_id(sample)
    cache = cache or get_default_feature_cache(config)

    if use_cache and not rebuild_cache:
        cached = cache.load(sample_id)
        if cached is not None:
            return cached

    source_paths: dict[str, Path] = {}
    missing_sources: list[str] = []
    for key in MODALITY_KEYS:
        path = _metadata_path(sample, key)
        if path is None or not path.exists():
            missing_sources.append(f"{key}: {path if path is not None else 'not provided'}")
        else:
            source_paths[key] = path

    if missing_sources:
        joined = "; ".join(missing_sources)
        raise FileNotFoundError(
            "Cannot build five-modality features because source feature files are missing. "
            "Cache is only an accelerator and cannot be the only input. "
            f"Missing sources: {joined}"
        )

    payloads = {key: _load_npy_payload(path) for key, path in source_paths.items()}
    raw_features = {key: _extract_features(payload) for key, payload in payloads.items()}

    labels = _extract_optional_array(payloads["gesture"], "labels")
    approx_timestamps = _extract_optional_array(payloads["gesture"], "approx_timestamps")
    lengths = [len(raw_features[key]) for key in MODALITY_KEYS]
    if labels is not None:
        lengths.append(len(labels))
    if approx_timestamps is not None:
        lengths.append(len(approx_timestamps))
    sample_count = min(lengths)
    if sample_count <= 0:
        raise FeatureBuildError(f"No valid feature rows found for sample {sample_id}.")

    mask = _valid_label_mask(labels, sample_count)
    if not mask.any():
        raise FeatureBuildError(f"All labels are filtered as unknown for sample {sample_id}.")

    features = {
        "imu": normalize_dense_modality(raw_features["imu"][:sample_count], target_steps, dims["imu"]),
        "gesture": normalize_dense_modality(raw_features["gesture"][:sample_count], target_steps, dims["gesture"]),
        "audio": normalize_audio_modality(raw_features["audio"][:sample_count], target_steps, dims["audio"]),
        "text": normalize_dense_modality(raw_features["text"][:sample_count], target_steps, dims["text"]),
        "scene": _normalize_scene_features(raw_features["scene"][:sample_count], dims["scene"]),
    }
    features = {key: value[mask] for key, value in features.items()}

    intent_labels = labels[:sample_count][mask].astype(np.int64) if labels is not None else None
    timestamps = approx_timestamps[:sample_count][mask] if approx_timestamps is not None else None
    scene_labels = sample.get("scene_label")
    scene_label_array = None
    if scene_labels is not None:
        scene_label_array = np.full(mask.sum(), int(scene_labels), dtype=np.int64)

    bundle = FeatureBundle(
        sample_id=sample_id,
        features=features,
        intent_labels=intent_labels,
        scene_labels=scene_label_array,
        approx_timestamps=timestamps,
        source_paths=source_paths,
    )
    if use_cache:
        cache.save(bundle)
    return bundle


def aggregate_feature_bundles(bundles: Iterable[FeatureBundle]) -> dict[str, np.ndarray]:
    """Concatenate multiple sample bundles into one feature dictionary."""
    grouped: dict[str, list[np.ndarray]] = {key: [] for key in MODALITY_KEYS}
    for bundle in bundles:
        for key in MODALITY_KEYS:
            grouped[key].append(bundle.features[key])
    if not any(grouped[key] for key in MODALITY_KEYS):
        raise FeatureBuildError("No feature bundles were provided.")
    return {key: np.concatenate(values, axis=0).astype(np.float32) for key, values in grouped.items()}


def describe_feature_shapes(features: dict[str, np.ndarray]) -> dict[str, tuple[int, ...]]:
    """Return shapes for all formal modalities in a stable key order."""
    return {key: tuple(features[key].shape) for key in MODALITY_KEYS}
