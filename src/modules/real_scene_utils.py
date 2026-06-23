# 截取视频中的真实场景，为训练模型提供场景模态，并缓存场景编码后的特征

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel

os.environ.setdefault("HF_HOME", r"E:\smart AR\.hf_cache")
os.environ.setdefault("HF_HUB_CACHE", r"E:\smart AR\.hf_cache\hub")
os.environ.setdefault("TRANSFORMERS_CACHE", r"E:\smart AR\.hf_cache\transformers")
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

ROOT_DIR = Path(r"E:\smart AR")
DATASET_VIDEO_DIR = ROOT_DIR / "dataset"
LOCAL_VIT_PATH = ROOT_DIR / "鱼眼完整模型" / "vit-base-patch16-224"
REAL_SCENE_CACHE_DIR = ROOT_DIR / "dataset" / "scene_cache_real_vit"
SCENE_FEAT_DIM = 768

AVI_TO_MP4_MAP = {
    "Video_20260306_152340690.avi": "interaction_20260306_072344.mp4",
    "Video_20260227_202553335.avi": "interaction_20260227_122606.mp4",
    "Video_20260227_202953348.avi": "interaction_20260227_122952.mp4",
    "Video_20260227_203348219.avi": "interaction_20260227_123354.mp4",
    "Video_20260227_204553897.avi": "interaction_20260227_124559.mp4",
    "Video_20260227_203753817.avi": "interaction_20260227_123745.mp4",
    "Video_20260131_200029359.avi": "interaction_20260131_120024.mp4",
    "Video_20260227_213001434.avi": "interaction_20260227_132951.mp4",
    "Video_20260227_213404452.avi": "interaction_20260227_133408.mp4",
    "Video_20260131_194205407.avi": "interaction_20260131_114156.mp4",
    "Video_20260131_195202906.avi": "interaction_20260131_115150.mp4",
    "Video_20260131_194854095.avi": "interaction_20260131_114852.mp4",
    "Video_20260301_153037623.avi": "interaction_20260301_073041.mp4",
    "Video_20260301_144803454.avi": "interaction_20260301_064753.mp4",
    "Video_20260306_152721366.avi": "interaction_20260306_072721.mp4",
    "Video_20260301_151942635.avi": "interaction_20260301_071948.mp4",
    "Video_20260131_201556629.avi": "interaction_20260131_121548.mp4",
    "Video_20260301_153434856.avi": "interaction_20260301_073435.mp4",
    "Video_20260301_152459131.avi": "interaction_20260301_072503.mp4",
    "Video_20260131_151559270.avi": "interaction_20260131_071552.mp4",
    "Video_20260131_152410916.avi": "interaction_20260131_072412.mp4",
    "Video_20260131_164304016.avi": "interaction_20260131_084300.mp4",
    "Video_20260131_164745532.avi": "interaction_20260131_084732.mp4",
    "Video_20260131_165208524.avi": "interaction_20260131_085207.mp4",
    "Video_20260131_165614756.avi": "interaction_20260131_085611.mp4",
    "Video_20260131_170142792.avi": "interaction_20260131_090139.mp4",
    "Video_20260131_145524524.avi": "interaction_20260131_065459.mp4",
    "Video_20260131_150734369.avi": "interaction_20260131_070722.mp4",
    "Video_20260131_170539636.avi": "interaction_20260131_090541.mp4",
    "Video_20260131_170919896.avi": "interaction_20260131_090917.mp4",
    "Video_20260131_171253889.avi": "interaction_20260131_091249.mp4",
    "Video_20260131_171648040.avi": "interaction_20260131_091657.mp4",
    "Video_20260306_162401599.avi": "interaction_20260306_082346.mp4",
    "Video_20260306_163105571.avi": "interaction_20260306_083107.mp4",
    "Video_20260306_163434878.avi": "interaction_20260306_083434.mp4",
    "Video_20260306_164407883.avi": "interaction_20260306_084406.mp4",
    "Video_20260306_164902044.avi": "interaction_20260306_084853.mp4",
    "Video_20260306_165839689.avi": "interaction_20260306_085830.mp4",
    "Video_20260306_170449073.avi": "interaction_20260306_090441.mp4",
}
MP4_TO_AVI_MAP = {mp4_name: avi_name for avi_name, mp4_name in AVI_TO_MP4_MAP.items()}

_PROCESSOR: Optional[ViTImageProcessor] = None
_MODEL: Optional[ViTModel] = None


def get_scene_backbone() -> Tuple[ViTImageProcessor, ViTModel]:
    global _PROCESSOR, _MODEL
    if _PROCESSOR is not None and _MODEL is not None:
        return _PROCESSOR, _MODEL
    _PROCESSOR = ViTImageProcessor.from_pretrained(str(LOCAL_VIT_PATH), local_files_only=True)
    _MODEL = ViTModel.from_pretrained(str(LOCAL_VIT_PATH), local_files_only=True, add_pooling_layer=False)
    _MODEL.eval()
    _MODEL.to("cpu")
    return _PROCESSOR, _MODEL


@torch.no_grad()
def encode_scene_pil_image(image: Image.Image) -> np.ndarray:
    processor, model = get_scene_backbone()
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.squeeze(0).cpu().numpy().astype(np.float32)


def parse_utc_timestamp(timestamp_value: str) -> datetime:
    return datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00")).replace(tzinfo=None)


def avi_start_utc_from_name(avi_name: str) -> datetime:
    parts = avi_name.split("_")
    time_part = f"{parts[1]}_{parts[2].split('.')[0]}"
    return datetime.strptime(time_part, "%Y%m%d_%H%M%S%f") - timedelta(hours=8)


def resolve_avi_path(video_name: str) -> Path:
    avi_name = MP4_TO_AVI_MAP[video_name]
    return DATASET_VIDEO_DIR / avi_name


def read_real_scene_frame(video_name: str, timestamp_value: str) -> Optional[Image.Image]:
    try:
        avi_path = resolve_avi_path(video_name)
        utc_target = parse_utc_timestamp(timestamp_value)
        avi_utc_start = avi_start_utc_from_name(avi_path.name)
        offset_ms = (utc_target - avi_utc_start).total_seconds() * 1000.0
    except Exception:
        return None

    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        cap.release()
        return None

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


class RealSceneFeatureCache:
    def __init__(self, cache_dir: Path = REAL_SCENE_CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, np.ndarray] = {}

    def _cache_key(self, video_name: str, timestamp_value: str) -> str:
        return f"{video_name}|{timestamp_value}"

    def _cache_path(self, video_name: str, timestamp_value: str) -> Path:
        key = hashlib.md5(self._cache_key(video_name, timestamp_value).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.npy"

    def get(self, video_name: str, timestamp_value: str) -> np.ndarray:
        cache_key = self._cache_key(video_name, timestamp_value)
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        cache_path = self._cache_path(video_name, timestamp_value)
        if cache_path.exists():
            feature = np.load(cache_path).astype(np.float32)
            self.memory_cache[cache_key] = feature
            return feature

        image = read_real_scene_frame(video_name, timestamp_value)
        feature = encode_scene_pil_image(image) if image is not None else np.zeros(SCENE_FEAT_DIM, dtype=np.float32)
        np.save(cache_path, feature)
        self.memory_cache[cache_key] = feature
        return feature


def load_real_scene_features(
    video_name: str,
    approx_timestamps: np.ndarray,
    cache: RealSceneFeatureCache,
) -> Tuple[np.ndarray, Dict[str, object]]:
    features = []
    failed_count = 0
    for timestamp_value in approx_timestamps.tolist():
        feature = cache.get(video_name, str(timestamp_value))
        if not np.any(feature):
            failed_count += 1
        features.append(feature)

    stacked = np.stack(features).astype(np.float32) if features else np.zeros((0, SCENE_FEAT_DIM), dtype=np.float32)
    record = {
        "scene_source": "real",
        "avi_path": str(resolve_avi_path(video_name)),
        "sample_count": int(len(approx_timestamps)),
        "failed_scene_frames": int(failed_count),
        "example_timestamps": [str(value) for value in approx_timestamps[:5].tolist()],
    }
    return stacked, record
