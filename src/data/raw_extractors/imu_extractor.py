"""IMU raw-data adapter.

Source reference:
    ``src/modules/feature_extraction/imu.py``
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.raw_extractors.common import (
    RawFeatureExtractionError,
    ensure_parent,
    optional_import,
    raw_path_for,
    require_file,
)


TARGET_FRAME_NUM = 10
WINDOW_SEC = 0.75
IMU_COLUMNS = [
    "pos_x",
    "pos_y",
    "pos_z",
    "roll",
    "pitch",
    "yaw",
    "vel_x",
    "vel_y",
    "vel_z",
    "acc_x",
    "acc_y",
    "acc_z",
]


def _load_and_preprocess_imu(imu_csv_path: Path):
    np = optional_import("numpy")
    pd = optional_import("pandas")
    scipy_transform = __import__("scipy.spatial.transform", fromlist=["Rotation"])
    Rotation = scipy_transform.Rotation

    if not imu_csv_path.exists():
        raise RawFeatureExtractionError(f"IMU CSV does not exist: {imu_csv_path}")
    df = pd.read_csv(imu_csv_path).dropna().sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp_sec"] = df["timestamp"].astype(np.int64) / 1e9

    rotation = Rotation.from_quat(df[["rot_x", "rot_y", "rot_z", "rot_w"]].values)
    df[["roll", "pitch", "yaw"]] = rotation.as_euler("xyz", degrees=True)
    dt = df["timestamp_sec"].diff().fillna(0.01).clip(lower=0.001)
    for axis in ["x", "y", "z"]:
        df[f"vel_{axis}"] = df[f"pos_{axis}"].diff() / dt
        df[f"acc_{axis}"] = df[f"vel_{axis}"].diff() / dt
    return df.fillna(0)


def _sample_imu_segment(segment_df, target_frame_num: int = TARGET_FRAME_NUM):
    np = optional_import("numpy")
    if segment_df.empty:
        return np.zeros((target_frame_num, 12), dtype=np.float32)
    features = segment_df[IMU_COLUMNS].values
    original_count = len(features)
    if original_count < 2:
        row = features[0] if original_count == 1 else np.zeros(12)
        return np.tile(row, (target_frame_num, 1)).astype(np.float32)

    new_index = np.linspace(0, original_count - 1, target_frame_num)
    sampled = np.zeros((target_frame_num, 12), dtype=np.float32)
    for column_index in range(12):
        sampled[:, column_index] = np.interp(new_index, np.arange(original_count), features[:, column_index])
    return sampled


def _extract_imu_for_metadata(df_imu, metadata_path: Path) -> dict[str, Any]:
    np = optional_import("numpy")
    payload = np.load(metadata_path, allow_pickle=True).item()
    approx_timestamps = payload["approx_timestamps"]
    labels = payload["labels"]

    valid_features, valid_timestamps, valid_labels, imu_ids = [], [], [], []
    for index, timestamp_value in enumerate(approx_timestamps):
        mid_dt = datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00"))
        mid_abs = mid_dt.timestamp()
        start, end = mid_abs - WINDOW_SEC, mid_abs + WINDOW_SEC
        segment_df = df_imu[(df_imu["timestamp_sec"] >= start) & (df_imu["timestamp_sec"] <= end)]
        valid_features.append(_sample_imu_segment(segment_df, TARGET_FRAME_NUM))
        valid_timestamps.append(timestamp_value)
        valid_labels.append(labels[index])
        imu_ids.append(index)

    features_array = np.array(valid_features, dtype=np.float32)
    if len(features_array) > 0:
        mean = np.mean(features_array, axis=(0, 1), keepdims=True)
        std = np.std(features_array, axis=(0, 1), keepdims=True) + 1e-8
        features_array = (features_array - mean) / std
    return {
        "imu_id": np.array(imu_ids),
        "approx_timestamps": np.array(valid_timestamps),
        "features": features_array.astype(np.float32),
        "labels": np.array(valid_labels),
    }


def build_imu_feature(
    sample: dict[str, Any],
    config: dict[str, Any],
    output_path: Path,
    gesture_metadata_path: Path,
) -> Path:
    """Build IMU source features from ``imu.csv`` and gesture timestamps."""
    del config
    imu_path = raw_path_for(sample, "imu")
    require_file(gesture_metadata_path, "gesture metadata for IMU alignment")
    ensure_parent(output_path)
    df_imu = _load_and_preprocess_imu(imu_path)
    result = _extract_imu_for_metadata(df_imu, gesture_metadata_path)
    if len(result["features"]) == 0:
        raise RawFeatureExtractionError(f"No valid IMU segments were extracted from: {imu_path}")
    np = optional_import("numpy")
    np.save(output_path, result)
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "source_metadata": str(gesture_metadata_path),
                "window_total_sec": WINDOW_SEC * 2,
                "frames_per_segment": TARGET_FRAME_NUM,
                "feature_channels": IMU_COLUMNS,
                "total_segments": int(len(result["imu_id"])),
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    return output_path
