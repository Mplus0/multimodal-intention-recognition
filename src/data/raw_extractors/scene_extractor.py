"""Scene raw-data adapter.

Source reference:
    ``src/modules/real_scene_utils.py``
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.data.raw_extractors.common import (
    RawFeatureExtractionError,
    config_path,
    ensure_parent,
    optional_import,
    raw_path_for,
    require_dir,
    require_file,
)


SCENE_FEAT_DIM = 768


def _parse_utc_timestamp(timestamp_value: str) -> datetime:
    return datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00")).replace(tzinfo=None)


def _avi_start_utc(avi_path: Path) -> datetime:
    parts = avi_path.name.split("_")
    if len(parts) < 3:
        raise RawFeatureExtractionError(f"Cannot parse fisheye video start time from: {avi_path.name}")
    time_part = f"{parts[1]}_{parts[2].split('.')[0]}"
    return datetime.strptime(time_part, "%Y%m%d_%H%M%S%f") - timedelta(hours=8)


def _read_scene_frame(video_path: Path, timestamp_value: str):
    cv2 = optional_import("cv2", "opencv-python-headless")
    Image = __import__("PIL.Image", fromlist=["Image"]).Image
    utc_target = _parse_utc_timestamp(timestamp_value)
    offset_ms = (utc_target - _avi_start_utc(video_path)).total_seconds() * 1000.0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RawFeatureExtractionError(f"Cannot open fisheye video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise RawFeatureExtractionError(f"Invalid fisheye video metadata: {video_path}")

    duration_ms = frame_count / fps * 1000.0
    if offset_ms < 0 or offset_ms > duration_ms:
        cap.release()
        return None

    frame_index = int(round(offset_ms * fps / 1000.0))
    frame_index = min(max(frame_index, 0), frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


class _SceneBackbone:
    def __init__(self, model_dir: Path):
        torch = optional_import("torch")
        transformers = optional_import("transformers")
        self.torch = torch
        self.processor = transformers.ViTImageProcessor.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        self.model = transformers.ViTModel.from_pretrained(
            str(model_dir),
            local_files_only=True,
            add_pooling_layer=False,
        )
        self.model.eval()
        self.model.to("cpu")

    def encode(self, image) -> Any:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()


def build_scene_feature(
    sample: dict[str, Any],
    config: dict[str, Any],
    output_path: Path,
    gesture_metadata_path: Path,
) -> Path:
    """Build scene source features from a fisheye video and gesture timestamps."""
    np = optional_import("numpy")
    video_path = raw_path_for(sample, "scene")
    require_file(gesture_metadata_path, "gesture metadata for scene timestamps")
    model_dir = config_path(config, "local_models", "vit_model", description="ViT scene model directory")
    require_dir(model_dir, "ViT scene model directory")
    ensure_parent(output_path)

    payload = np.load(gesture_metadata_path, allow_pickle=True).item()
    approx_timestamps = payload["approx_timestamps"]
    backbone = _SceneBackbone(model_dir)
    features = []
    failed_count = 0
    for timestamp_value in approx_timestamps:
        image = _read_scene_frame(video_path, str(timestamp_value))
        if image is None:
            failed_count += 1
            features.append(np.zeros(SCENE_FEAT_DIM, dtype=np.float32))
        else:
            feature = backbone.encode(image).astype(np.float32)
            if feature.shape != (SCENE_FEAT_DIM,):
                raise RawFeatureExtractionError(
                    f"Scene model returned shape {feature.shape}, expected ({SCENE_FEAT_DIM},)."
                )
            features.append(feature)

    if not features:
        raise RawFeatureExtractionError(f"No scene timestamps were available in: {gesture_metadata_path}")
    stacked = np.stack(features).astype(np.float32)
    np.save(output_path, {"features": stacked, "metadata": {"failed_scene_frames": failed_count}})
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "source_metadata": str(gesture_metadata_path),
                "source_video": str(video_path),
                "sample_count": int(len(approx_timestamps)),
                "failed_scene_frames": int(failed_count),
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    return output_path
