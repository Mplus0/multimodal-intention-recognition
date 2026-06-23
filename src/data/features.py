"""Feature loading and caching helpers for multimodal samples."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.utils.paths import get_path, load_paths_config


SCENE_NAME_TO_ID = {"office": 0, "museum": 1}
IMU_NUMERIC_COLUMNS = ("pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "rot_w")


@dataclass(frozen=True)
class FeatureConfig:
    """Controls raw feature extraction used by the formal Dataset."""

    num_video_frames: int = 8
    image_size: int = 112
    imu_steps: int = 10
    cache_version: str = "formal_v1"
    use_cache: bool = True
    rebuild_cache: bool = False


_IMU_TABLE_CACHE: dict[Path, pd.DataFrame] = {}


def get_feature_cache_dir(config_path: Path | None = None) -> Path:
    """Return the configured feature-cache directory."""
    config = load_paths_config(config_path)
    return get_path("cache", "feature_cache", config=config)


def _safe_cache_stem(value: str) -> str:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:10]
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return f"{cleaned[:80]}_{digest}"


def _video_cache_path(
    cache_dir: Path,
    sample_id: str,
    stream_name: str,
    video_path: Path,
    feature_config: FeatureConfig,
) -> Path:
    stem = _safe_cache_stem(f"{sample_id}_{stream_name}_{video_path.name}")
    return (
        cache_dir
        / "video_frames"
        / feature_config.cache_version
        / f"frames{feature_config.num_video_frames}_size{feature_config.image_size}"
        / stream_name
        / f"{stem}.pt"
    )


def _read_frame_at(cap: cv2.VideoCapture, frame_index: int, image_size: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)


def sample_video_frames(
    video_path: Path,
    num_frames: int,
    image_size: int,
) -> torch.Tensor:
    """Uniformly sample video frames as a float tensor shaped ``T,C,H,W``."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise ValueError(f"Video has no readable frames: {video_path}")

    frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=np.int64)
    frames = [_read_frame_at(cap, int(index), image_size) for index in frame_indices]
    cap.release()

    array = np.stack(frames).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
    return tensor


def load_or_build_video_frames(
    sample_id: str,
    stream_name: str,
    video_path: Path,
    cache_dir: Path,
    feature_config: FeatureConfig,
) -> torch.Tensor:
    """Load cached video frames or build them from a raw video."""
    cache_path = _video_cache_path(cache_dir, sample_id, stream_name, video_path, feature_config)
    if feature_config.use_cache and cache_path.exists() and not feature_config.rebuild_cache:
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    tensor = sample_video_frames(
        video_path=video_path,
        num_frames=feature_config.num_video_frames,
        image_size=feature_config.image_size,
    )
    if feature_config.use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, cache_path)
    return tensor


def _load_imu_table(imu_path: Path) -> pd.DataFrame:
    resolved = imu_path.resolve()
    if resolved not in _IMU_TABLE_CACHE:
        table = pd.read_csv(resolved)
        missing = [column for column in IMU_NUMERIC_COLUMNS if column not in table.columns]
        if missing:
            raise ValueError(f"IMU CSV missing columns: {missing}")
        _IMU_TABLE_CACHE[resolved] = table
    return _IMU_TABLE_CACHE[resolved]


def build_imu_sequence(imu_path: Path, steps: int) -> torch.Tensor:
    """Build a fixed ``steps x 12`` IMU tensor from the available CSV.

    The current raw CSV is not timestamp-aligned with individual videos, so this
    function provides a stable project-level IMU signal. Precise per-sample
    alignment can replace this function when reliable sample timestamps are
    available.
    """
    table = _load_imu_table(imu_path)
    numeric = table.loc[:, IMU_NUMERIC_COLUMNS].to_numpy(dtype=np.float32)
    if len(numeric) == 0:
        return torch.zeros((steps, 12), dtype=torch.float32)

    mean = numeric.mean(axis=0, keepdims=True)
    std = numeric.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    normalized = (numeric - mean) / std

    indices = np.linspace(0, len(normalized) - 1, steps, dtype=np.int64)
    sampled = normalized[indices]
    pos = sampled[:, :3]
    rot = sampled[:, 3:7]
    pos_delta = np.vstack([np.zeros((1, 3), dtype=np.float32), np.diff(pos, axis=0)])
    pos_norm = np.linalg.norm(pos, axis=1, keepdims=True)
    rot_norm = np.linalg.norm(rot, axis=1, keepdims=True)
    features = np.concatenate([sampled, pos_delta, pos_norm, rot_norm], axis=1)
    return torch.from_numpy(features.astype(np.float32))


def build_scene_tensor(scene_name: str) -> torch.Tensor:
    """Return a two-value one-hot scene tensor in office/museum order."""
    if scene_name not in SCENE_NAME_TO_ID:
        raise ValueError(f"Unknown scene name: {scene_name}")
    tensor = torch.zeros(len(SCENE_NAME_TO_ID), dtype=torch.float32)
    tensor[SCENE_NAME_TO_ID[scene_name]] = 1.0
    return tensor

