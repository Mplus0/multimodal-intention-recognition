# 本基线模型使用了 Perceiver-IO 架构，输入包括 IMU、手势、音频、文本和场景五个模态。
# 场景模态使用真实场景，并以单个 scene token 形式接入融合模型。
# 训练完成后，模型在测试集上进行评估，并生成分类报告、混淆矩阵和场景选择记录。

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import random
from collections import Counter
from glob import glob
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("HF_HOME", r"E:\smart AR\.hf_cache")
os.environ.setdefault("HF_HUB_CACHE", r"E:\smart AR\.hf_cache\hub")
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)

from transformers import ViTImageProcessor, ViTModel
from real_scene_utils import REAL_SCENE_CACHE_DIR, RealSceneFeatureCache, load_real_scene_features


# ============================================================
# 1. Config
# ============================================================
ROOT_DIR = Path(r"E:\smart AR")
PROCESSED_DATA_DIR = ROOT_DIR / "AR_Data_Process3.0" / "data"
MODEL_OUTPUT_DIR = Path(
    os.getenv(
        "SMART_AR_MODEL_OUTPUT_DIR",
        str(ROOT_DIR / "Baseline_Model" / "intentionReg" / "baseline_real_scene_perceiver_io"),
    )
)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR = ROOT_DIR / ".hf_cache"
HF_TRANSFORMERS_CACHE_DIR = HF_CACHE_DIR / "transformers"
HF_HUB_CACHE_DIR = HF_CACHE_DIR / "hub"
HF_TRANSFORMERS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE_DIR))

STRONG_GESTURE_DIR = PROCESSED_DATA_DIR / "strong_gesture_features"
AUDIO_FEAT_DIR = PROCESSED_DATA_DIR / "audio_features"
TEXT_FEAT_DIR = PROCESSED_DATA_DIR / "text_features"
IMU_FEAT_DIR = PROCESSED_DATA_DIR / "imu_features"
LOCAL_VIT_PATH = ROOT_DIR / "鱼眼完整模型" / "vit-base-patch16-224"
SCENE_DIRS = {
    "museum": ROOT_DIR / "鱼眼完整模型" / "museum",
    "office": ROOT_DIR / "鱼眼完整模型" / "office",
}
SCENE_CACHE_DIR = REAL_SCENE_CACHE_DIR

RANDOM_SEED = 42
VAL_SPLIT = 0.2
TARGET_TIMESTEPS = 10
NUM_MODALITIES = 5
GESTURE_FEAT_DIM = 768
IMU_FEAT_DIM = 12
AUDIO_FEAT_DIM = 39
TEXT_FEAT_DIM = 384
SCENE_FEAT_DIM = 768

MODEL_DIM = int(os.getenv("BASELINE_SCENE_MODEL_DIM", "128"))
NUM_LATENTS = int(os.getenv("BASELINE_SCENE_NUM_LATENTS", "32"))
NUM_HEADS = int(os.getenv("BASELINE_SCENE_NUM_HEADS", "4"))
DEPTH = int(os.getenv("BASELINE_SCENE_DEPTH", "2"))
DROPOUT = float(os.getenv("BASELINE_SCENE_DROPOUT", "0.1"))
FF_MULTIPLIER = int(os.getenv("BASELINE_SCENE_FF_MULTIPLIER", "4"))

BATCH_SIZE = int(os.getenv("BASELINE_SCENE_BATCH_SIZE", "64"))
EPOCHS = int(os.getenv("BASELINE_SCENE_EPOCHS", "100"))
PATIENCE = int(os.getenv("BASELINE_SCENE_PATIENCE", "10"))
LEARNING_RATE = float(os.getenv("BASELINE_SCENE_LR", "1e-3"))
WEIGHT_DECAY = float(os.getenv("BASELINE_SCENE_WEIGHT_DECAY", "1e-4"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SKIP_TEST_EVAL = os.getenv("SMART_AR_SKIP_TEST_EVAL", "0") == "1"
UNKNOWN_LABELS = {"\u672a\u77e5", "unknown", "Unknown"}
MODALITY_KEYS = ("imu", "gesture", "audio", "text", "scene")
MODALITY_DISPLAY_NAMES = {
    "imu": "IMU",
    "gesture": "Gesture",
    "audio": "Audio",
    "text": "Text",
    "scene": "Scene",
}

INTENT_NAMES = {
    0: "menu",
    1: "select",
    2: "magnify",
    3: "narrow",
    4: "brush",
    5: "cancel",
}
SCENE_NAME_TO_ID = {"office": 0, "museum": 1}
SCENE_ID_TO_NAME = {index: name for name, index in SCENE_NAME_TO_ID.items()}

VIDEO_LABELS = {
    "interaction_20260306_072344.mp4": 0,
    "interaction_20260227_122606.mp4": 1,
    "interaction_20260227_122952.mp4": 2,
    "interaction_20260227_123354.mp4": 3,
    "interaction_20260227_124559.mp4": 4,
    "interaction_20260227_123745.mp4": 5,
    "interaction_20260131_120024.mp4": 0,
    "interaction_20260227_132951.mp4": 1,
    "interaction_20260227_133408.mp4": 2,
    "interaction_20260131_114156.mp4": 3,
    "interaction_20260131_115150.mp4": 4,
    "interaction_20260131_114852.mp4": 5,
    "interaction_20260301_073041.mp4": 0,
    "interaction_20260301_064753.mp4": 1,
    "interaction_20260306_072721.mp4": 2,
    "interaction_20260301_071948.mp4": 3,
    "interaction_20260131_121548.mp4": 3,
    "interaction_20260301_073435.mp4": 4,
    "interaction_20260301_072503.mp4": 5,
    "interaction_20260131_071552.mp4": 0,
    "interaction_20260131_072412.mp4": 1,
    "interaction_20260131_084300.mp4": 1,
    "interaction_20260131_085611.mp4": 2,
    "interaction_20260131_090139.mp4": 3,
    "interaction_20260131_085207.mp4": 4,
    "interaction_20260131_084732.mp4": 5,
    "interaction_20260131_090917.mp4": 0,
    "interaction_20260131_090541.mp4": 1,
    "interaction_20260131_065459.mp4": 2,
    "interaction_20260131_070722.mp4": 3,
    "interaction_20260131_091657.mp4": 4,
    "interaction_20260131_091249.mp4": 5,
    "interaction_20260306_082346.mp4": 2,
    "interaction_20260306_083107.mp4": 3,
    "interaction_20260306_083434.mp4": 1,
    "interaction_20260306_084406.mp4": 0,
    "interaction_20260306_084853.mp4": 5,
    "interaction_20260306_085830.mp4": 4,
    "interaction_20260306_090441.mp4": 1,
}

TEST_VIDEO_NAMES = [
    "interaction_20260306_072344.mp4",
    "interaction_20260227_122606.mp4",
    "interaction_20260227_122952.mp4",
    "interaction_20260227_123354.mp4",
    "interaction_20260227_124559.mp4",
    "interaction_20260227_123745.mp4",
    "interaction_20260306_082346.mp4",
    "interaction_20260306_083107.mp4",
    "interaction_20260306_083434.mp4",
    "interaction_20260306_084406.mp4",
    "interaction_20260306_084853.mp4",
    "interaction_20260306_085830.mp4",
    "interaction_20260306_090441.mp4",
]
TRAIN_VIDEO_NAMES = [
    video_name for video_name in VIDEO_LABELS if video_name not in TEST_VIDEO_NAMES
]
ALL_CLASSES = np.array(sorted(set(VIDEO_LABELS.values())), dtype=np.int64)

OFFICE_VIDEO_NAMES = {
    "interaction_20260306_072344.mp4",
    "interaction_20260227_122606.mp4",
    "interaction_20260227_122952.mp4",
    "interaction_20260227_123354.mp4",
    "interaction_20260227_124559.mp4",
    "interaction_20260227_123745.mp4",
    "interaction_20260131_120024.mp4",
    "interaction_20260227_132951.mp4",
    "interaction_20260227_133408.mp4",
    "interaction_20260131_114156.mp4",
    "interaction_20260131_115150.mp4",
    "interaction_20260131_114852.mp4",
    "interaction_20260301_073041.mp4",
    "interaction_20260301_064753.mp4",
    "interaction_20260306_072721.mp4",
    "interaction_20260301_071948.mp4",
    "interaction_20260131_121548.mp4",
    "interaction_20260301_073435.mp4",
    "interaction_20260301_072503.mp4",
}
MUSEUM_VIDEO_NAMES = set(VIDEO_LABELS) - OFFICE_VIDEO_NAMES
SCENE_BY_VIDEO = {video_name: "office" for video_name in OFFICE_VIDEO_NAMES}
SCENE_BY_VIDEO.update({video_name: "museum" for video_name in MUSEUM_VIDEO_NAMES})
ALL_JOINT_CLASS_NAMES = np.array(
    [
        f"{scene_name}_{INTENT_NAMES[intent_id]}"
        for scene_name in ("office", "museum")
        for intent_id in sorted(INTENT_NAMES)
    ],
    dtype=object,
)


def collect_scene_images(scene_dir: Path) -> List[Path]:
    patterns = ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG")
    image_paths: List[Path] = []
    for pattern in patterns:
        image_paths.extend(Path(path) for path in glob(str(scene_dir / pattern)))
    return sorted(set(image_paths))


SCENE_IMAGE_PATHS = {
    scene_name: collect_scene_images(scene_dir) for scene_name, scene_dir in SCENE_DIRS.items()
}

missing_scene_assignments = sorted(set(VIDEO_LABELS) - set(SCENE_BY_VIDEO))
if missing_scene_assignments:
    raise RuntimeError(f"Missing scene assignments for videos: {missing_scene_assignments}")
for scene_name, image_paths in SCENE_IMAGE_PATHS.items():
    if not image_paths:
        raise RuntimeError(f"No scene images found under: {SCENE_DIRS[scene_name]}")


# ============================================================
# 2. Data loading and alignment
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_label_name(label_value: int) -> str:
    return INTENT_NAMES.get(int(label_value), str(label_value))


def summarize_labels(labels: np.ndarray) -> Dict[str, int]:
    counts = Counter(int(label) for label in labels.tolist())
    return {format_label_name(label): int(count) for label, count in sorted(counts.items())}


def summarize_scene_counts(scene_names: Iterable[str]) -> Dict[str, int]:
    counts = Counter(scene_names)
    return {scene_name: int(count) for scene_name, count in sorted(counts.items())}


def make_joint_label(scene_name: str, intent_value: int) -> str:
    return f"{scene_name}_{format_label_name(intent_value)}"


def build_joint_labels(intent_labels: np.ndarray, scene_targets: np.ndarray) -> np.ndarray:
    return np.array(
        [
            make_joint_label(SCENE_ID_TO_NAME[int(scene_id)], int(intent_value))
            for intent_value, scene_id in zip(intent_labels.tolist(), scene_targets.tolist())
        ],
        dtype=object,
    )


def summarize_joint_labels(labels: np.ndarray) -> Dict[str, int]:
    counts = Counter(str(label) for label in labels.tolist())
    return {label: int(count) for label, count in sorted(counts.items())}


def split_joint_label(label_name: str) -> Tuple[str, str]:
    scene_name, intent_name = label_name.split("_", 1)
    return scene_name, intent_name


_SCENE_PROCESSOR: Optional[ViTImageProcessor] = None
_SCENE_MODEL: Optional[ViTModel] = None


# Load local VIT model
def get_scene_backbone() -> Tuple[ViTImageProcessor, ViTModel]:
    global _SCENE_PROCESSOR, _SCENE_MODEL

    if _SCENE_PROCESSOR is not None and _SCENE_MODEL is not None:
        return _SCENE_PROCESSOR, _SCENE_MODEL

    if not LOCAL_VIT_PATH.exists():
        raise FileNotFoundError(f"Local ViT path does not exist: {LOCAL_VIT_PATH}")

    print(f"[scene] load local ViT from {LOCAL_VIT_PATH}")
    _SCENE_PROCESSOR = ViTImageProcessor.from_pretrained(
        str(LOCAL_VIT_PATH),
        local_files_only=True,
    )
    _SCENE_MODEL = ViTModel.from_pretrained(
        str(LOCAL_VIT_PATH),
        local_files_only=True,
        add_pooling_layer=False,
    )
    _SCENE_MODEL.eval()
    _SCENE_MODEL.to("cpu")
    return _SCENE_PROCESSOR, _SCENE_MODEL

# get the scene feature
@torch.no_grad()
def encode_scene_image(image_path: Path) -> np.ndarray:
    processor, model = get_scene_backbone()
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
        return embedding.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception as exc:
        print(f"[scene] failed to encode {image_path}: {exc}")
        return np.zeros(SCENE_FEAT_DIM, dtype=np.float32)


class SceneFeatureCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, np.ndarray] = {}

    def _cache_path(self, image_path: Path) -> Path:
        key = hashlib.md5(str(image_path).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.npy"

    def get(self, image_path: Path) -> np.ndarray:
        cache_key = str(image_path)
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        cache_path = self._cache_path(image_path)
        if cache_path.exists():
            try:
                feature = np.load(cache_path).astype(np.float32)
                self.memory_cache[cache_key] = feature
                return feature
            except Exception as exc:
                print(f"[scene] cache load failed for {cache_path}: {exc}")

        feature = encode_scene_image(image_path)
        np.save(cache_path, feature)
        self.memory_cache[cache_key] = feature
        return feature


def infer_scene_type(video_name: str) -> str:
    if video_name not in SCENE_BY_VIDEO:
        raise KeyError(f"Unknown scene type for video: {video_name}")
    return SCENE_BY_VIDEO[video_name]


def choose_scene_image(video_name: str) -> Tuple[str, Path]:
    scene_type = infer_scene_type(video_name)
    candidates = SCENE_IMAGE_PATHS[scene_type]
    digest = hashlib.md5(video_name.encode("utf-8")).hexdigest()
    index = int(digest, 16) % len(candidates)
    return scene_type, candidates[index]


def normalize_sequence_length(
    sequence: np.ndarray,
    target_steps: int,
    feat_dim: int,
    long_mode: str = "truncate",
) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D sequence, got shape {array.shape}")
    if array.shape[1] != feat_dim:
        raise ValueError(
            f"Expected feature dim {feat_dim}, got shape {array.shape}"
        )

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
) -> np.ndarray:
    return np.stack(
        [
            normalize_sequence_length(sample, target_steps, feat_dim, long_mode="truncate")
            for sample in np.asarray(features)  # type: ignore[arg-type]
        ]
    ).astype(np.float32)


def normalize_audio_modality(audio_samples: np.ndarray) -> np.ndarray:
    normalized = []
    for sample in audio_samples:
        feature = sample["feature"] if isinstance(sample, dict) else sample
        normalized.append(
            normalize_sequence_length(
                feature,
                TARGET_TIMESTEPS,
                AUDIO_FEAT_DIM,
                long_mode="even",
            )
        )
    return np.stack(normalized).astype(np.float32)


def get_feature_paths(video_name: str) -> Dict[str, Path]:
    name_no_ext = Path(video_name).stem
    return {
        "gesture": STRONG_GESTURE_DIR / f"strong_gesture_features_{name_no_ext}.npy",
        "imu": IMU_FEAT_DIR / f"imu_features_{name_no_ext}.npy",
        "audio": AUDIO_FEAT_DIR / f"audio_features_{name_no_ext}.npy",
        "text": TEXT_FEAT_DIR / f"text_features_{name_no_ext}.npy",
    }


def load_aligned_video(
    video_name: str,
    scene_cache: RealSceneFeatureCache,
) -> Optional[Dict[str, object]]:
    paths = get_feature_paths(video_name)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        print(f"[skip] missing feature files for {video_name}: {missing}")
        return None

    gesture_data = np.load(paths["gesture"], allow_pickle=True).item()
    imu_data = np.load(paths["imu"], allow_pickle=True).item()
    audio_data = np.load(paths["audio"], allow_pickle=True)
    text_data = np.load(paths["text"], allow_pickle=True).item()

    lengths = [
        len(gesture_data["labels"]),
        len(gesture_data["features"]),
        len(gesture_data["approx_timestamps"]),
        len(imu_data["features"]),
        len(audio_data),
        len(text_data["features"]),
    ]
    min_len = min(lengths)
    if min_len <= 0:
        print(f"[skip] no valid aligned samples for {video_name}")
        return None

    labels = np.asarray(gesture_data["labels"][:min_len], dtype=object)
    approx_timestamps = np.asarray(gesture_data["approx_timestamps"][:min_len], dtype=object)
    valid_mask = np.array([str(label) not in UNKNOWN_LABELS for label in labels], dtype=bool)
    if not valid_mask.any():
        print(f"[skip] all labels are filtered as unknown for {video_name}")
        return None

    imu_feat = normalize_dense_modality(
        np.asarray(imu_data["features"][:min_len]),
        TARGET_TIMESTEPS,
        IMU_FEAT_DIM,
    )
    gesture_feat = normalize_dense_modality(
        np.asarray(gesture_data["features"][:min_len]),
        TARGET_TIMESTEPS,
        GESTURE_FEAT_DIM,
    )
    audio_feat = normalize_audio_modality(audio_data[:min_len])
    text_feat = normalize_dense_modality(
        np.asarray(text_data["features"][:min_len]),
        TARGET_TIMESTEPS,
        TEXT_FEAT_DIM,
    )

    scene_type = infer_scene_type(video_name)
    scene_targets = np.full(min_len, SCENE_NAME_TO_ID[scene_type], dtype=np.int64)

    labels = labels[valid_mask].astype(np.int64)
    approx_timestamps = approx_timestamps[valid_mask]
    scene_feat, scene_record = load_real_scene_features(video_name, approx_timestamps, scene_cache)
    return {
        "imu": imu_feat[valid_mask],
        "gesture": gesture_feat[valid_mask],
        "audio": audio_feat[valid_mask],
        "text": text_feat[valid_mask],
        "scene": scene_feat,
        "labels": labels,
        "scene_targets": scene_targets[valid_mask],
        "scene_type": scene_type,
        "scene_record": scene_record,
    }


def load_multimodal_data(
    video_names: Iterable[str],
    scene_cache: RealSceneFeatureCache,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, Dict[str, object]]]:
    aggregated = {"imu": [], "gesture": [], "audio": [], "text": [], "scene": []}
    labels_list: List[np.ndarray] = []
    scene_target_list: List[np.ndarray] = []
    first_loaded = False
    scene_selection: Dict[str, Dict[str, object]] = {}
    sample_scene_names: List[str] = []

    for video_name in video_names:
        payload = load_aligned_video(video_name, scene_cache)
        if payload is None:
            continue

        for key in aggregated:
            aggregated[key].append(payload[key])  # type: ignore[arg-type]
        labels_list.append(payload["labels"])  # type: ignore[arg-type]
        scene_target_list.append(payload["scene_targets"])  # type: ignore[arg-type]

        scene_type = str(payload["scene_type"])
        scene_record = dict(payload["scene_record"])  # type: ignore[arg-type]
        scene_record["scene_type"] = scene_type
        scene_selection[video_name] = scene_record
        sample_scene_names.extend([scene_type] * len(payload["labels"]))  # type: ignore[arg-type]

        if not first_loaded:
            print(f"[sanity] first aligned video: {video_name}")
            print(f"  IMU     {payload['imu'].shape}")  # type: ignore[index]
            print(f"  Gesture {payload['gesture'].shape}")  # type: ignore[index]
            print(f"  Audio   {payload['audio'].shape}")  # type: ignore[index]
            print(f"  Text    {payload['text'].shape}")  # type: ignore[index]
            print(f"  Scene   {payload['scene'].shape}")  # type: ignore[index]
            print(f"  SceneType  {scene_type}")
            print(f"  SceneAVI   {scene_record['avi_path']}")
            print(f"  ExampleTS  {scene_record['example_timestamps']}")
            first_loaded = True

    if not labels_list:
        raise RuntimeError("No aligned samples were loaded from the processed feature files.")

    features = {
        key: np.concatenate(value_list, axis=0).astype(np.float32)
        for key, value_list in aggregated.items()
    }
    labels = np.concatenate(labels_list, axis=0).astype(np.int64)
    scene_targets = np.concatenate(scene_target_list, axis=0).astype(np.int64)

    print("[summary] loaded multimodal dataset with scene")
    print(f"  IMU     {features['imu'].shape}")
    print(f"  Gesture {features['gesture'].shape}")
    print(f"  Audio   {features['audio'].shape}")
    print(f"  Text    {features['text'].shape}")
    print(f"  Scene   {features['scene'].shape}")
    print(f"  Labels  {labels.shape}")
    print(f"  Dist    {summarize_labels(labels)}")
    print(f"  SceneDist {summarize_scene_counts(sample_scene_names)}")
    return features, labels, scene_targets, scene_selection


# ============================================================
# 3. Preprocessing
# ============================================================
def can_stratify(labels: np.ndarray) -> bool:
    counts = Counter(labels.tolist())
    return len(counts) > 1 and min(counts.values()) >= 2


def split_train_val(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    scene_targets: np.ndarray,
    joint_labels: np.ndarray,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    indices = np.arange(len(labels))
    stratify = joint_labels if can_stratify(joint_labels) else None
    train_idx, val_idx = train_test_split(
        indices,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )

    train_features = {key: value[train_idx] for key, value in features.items()}
    val_features = {key: value[val_idx] for key, value in features.items()}
    return (
        train_features,
        val_features,
        labels[train_idx],
        labels[val_idx],
        scene_targets[train_idx],
        scene_targets[val_idx],
        joint_labels[train_idx],
        joint_labels[val_idx],
    )


def fit_scalers(train_features: Dict[str, np.ndarray]) -> Dict[str, StandardScaler]:
    feat_dims = {
        "imu": IMU_FEAT_DIM,
        "gesture": GESTURE_FEAT_DIM,
        "audio": AUDIO_FEAT_DIM,
        "text": TEXT_FEAT_DIM,
        "scene": SCENE_FEAT_DIM,
    }
    scalers = {}
    for key, feat_dim in feat_dims.items():
        scaler = StandardScaler()
        scaler.fit(train_features[key].reshape(-1, feat_dim))
        scalers[key] = scaler
    return scalers


def apply_scalers(
    features: Dict[str, np.ndarray],
    scalers: Dict[str, StandardScaler],
) -> Dict[str, np.ndarray]:
    feat_dims = {
        "imu": IMU_FEAT_DIM,
        "gesture": GESTURE_FEAT_DIM,
        "audio": AUDIO_FEAT_DIM,
        "text": TEXT_FEAT_DIM,
        "scene": SCENE_FEAT_DIM,
    }
    transformed = {}
    for key, feat_dim in feat_dims.items():
        flat = features[key].reshape(-1, feat_dim)
        scaled = scalers[key].transform(flat).reshape(features[key].shape)
        transformed[key] = scaled.astype(np.float32)
    return transformed


# ============================================================
# 4. Dataset / DataLoader and model
# ============================================================
class MultimodalSceneDataset(Dataset):
    def __init__(self, features: Dict[str, np.ndarray], labels: np.ndarray, scene_targets: np.ndarray):
        self.imu = torch.from_numpy(features["imu"].astype(np.float32))
        self.gesture = torch.from_numpy(features["gesture"].astype(np.float32))
        self.audio = torch.from_numpy(features["audio"].astype(np.float32))
        self.text = torch.from_numpy(features["text"].astype(np.float32))
        self.scene = torch.from_numpy(features["scene"].astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.scene_targets = torch.from_numpy(scene_targets.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.imu[index],
            self.gesture[index],
            self.audio[index],
            self.text[index],
            self.scene[index],
            self.labels[index],
            self.scene_targets[index],
        )


class FeedForward(nn.Module):
    def __init__(self, dim: int, multiplier: int = FF_MULTIPLIER, dropout: float = DROPOUT):
        super().__init__()
        hidden_dim = dim * multiplier
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            kdim=context_dim,
            vdim=context_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = FeedForward(query_dim, dropout=dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        x = query + attn_out
        x = x + self.ff(x)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.norm(x),
            self.norm(x),
            self.norm(x),
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ff(x)
        return x


class PerceiverEncoder(nn.Module):
    def __init__(self, latent_dim: int, num_latents: int, depth: int, num_heads: int, dropout: float):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim) * 0.02)
        self.cross_blocks = nn.ModuleList(
            [CrossAttentionBlock(latent_dim, latent_dim, num_heads, dropout) for _ in range(depth)]
        )
        self.self_blocks = nn.ModuleList(
            [SelfAttentionBlock(latent_dim, num_heads, dropout) for _ in range(depth)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        latents = self.latents.expand(batch_size, -1, -1)
        for cross_block, self_block in zip(self.cross_blocks, self.self_blocks):
            latents = cross_block(latents, tokens)
            latents = self_block(latents)
        return latents


class PerceiverIOSceneBaseline(nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_dim: int = MODEL_DIM,
        num_latents: int = NUM_LATENTS,
        depth: int = DEPTH,
        num_heads: int = NUM_HEADS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.imu_proj = nn.Sequential(nn.Linear(IMU_FEAT_DIM, model_dim), nn.LayerNorm(model_dim))
        self.gesture_proj = nn.Sequential(nn.Linear(GESTURE_FEAT_DIM, model_dim), nn.LayerNorm(model_dim))
        self.audio_proj = nn.Sequential(nn.Linear(AUDIO_FEAT_DIM, model_dim), nn.LayerNorm(model_dim))
        self.text_proj = nn.Sequential(nn.Linear(TEXT_FEAT_DIM, model_dim), nn.LayerNorm(model_dim))
        self.scene_proj = nn.Sequential(nn.Linear(SCENE_FEAT_DIM, model_dim), nn.LayerNorm(model_dim))

        self.time_embedding = nn.Parameter(torch.randn(1, TARGET_TIMESTEPS, model_dim) * 0.02)
        self.modality_embedding = nn.Parameter(torch.randn(NUM_MODALITIES, 1, model_dim) * 0.02)
        self.input_dropout = nn.Dropout(dropout)

        self.encoder = PerceiverEncoder(
            latent_dim=model_dim,
            num_latents=num_latents,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.output_query = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        self.decoder = CrossAttentionBlock(model_dim, model_dim, num_heads, dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes),
        )

    def _add_embeddings(self, tokens: torch.Tensor, modality_index: int) -> torch.Tensor:
        return tokens + self.time_embedding + self.modality_embedding[modality_index]

    def _add_single_token_embedding(self, token: torch.Tensor, modality_index: int) -> torch.Tensor:
        return token + self.modality_embedding[modality_index]

    def forward(
        self,
        imu: torch.Tensor,
        gesture: torch.Tensor,
        audio: torch.Tensor,
        text: torch.Tensor,
        scene: torch.Tensor,
    ) -> torch.Tensor:
        imu_tokens = self._add_embeddings(self.imu_proj(imu), 0)
        gesture_tokens = self._add_embeddings(self.gesture_proj(gesture), 1)
        audio_tokens = self._add_embeddings(self.audio_proj(audio), 2)
        text_tokens = self._add_embeddings(self.text_proj(text), 3)
        scene_token = self._add_single_token_embedding(self.scene_proj(scene).unsqueeze(1), 4)

        tokens = torch.cat([imu_tokens, gesture_tokens, audio_tokens, text_tokens, scene_token], dim=1)
        tokens = self.input_dropout(tokens)

        latents = self.encoder(tokens)
        query = self.output_query.expand(tokens.shape[0], -1, -1)
        decoded = self.decoder(query, latents)
        fused = decoded.squeeze(1)
        logits = self.classifier(fused)
        return logits


def make_loader(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    scene_targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = MultimodalSceneDataset(features, labels, scene_targets)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# 5. Train / validation / test
# ============================================================
def encode_labels(labels: np.ndarray) -> Tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    encoder.fit(ALL_JOINT_CLASS_NAMES)
    return encoder.transform(labels), encoder


def build_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    weights = np.ones(num_classes, dtype=np.float32)
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_labels,
        y=labels,
    )
    for label_value, weight in zip(unique_labels, class_weights):
        weights[int(label_value)] = float(weight)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_imu, batch_gesture, batch_audio, batch_text, batch_scene, batch_y, batch_scene_y in loader:
        batch_imu = batch_imu.to(DEVICE)
        batch_gesture = batch_gesture.to(DEVICE)
        batch_audio = batch_audio.to(DEVICE)
        batch_text = batch_text.to(DEVICE)
        batch_scene = batch_scene.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(batch_imu, batch_gesture, batch_audio, batch_text, batch_scene)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_imu.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == batch_y).sum().item()
        total_samples += batch_imu.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch_imu, batch_gesture, batch_audio, batch_text, batch_scene, batch_y, batch_scene_y in loader:
        batch_imu = batch_imu.to(DEVICE)
        batch_gesture = batch_gesture.to(DEVICE)
        batch_audio = batch_audio.to(DEVICE)
        batch_text = batch_text.to(DEVICE)
        batch_scene = batch_scene.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        logits = model(batch_imu, batch_gesture, batch_audio, batch_text, batch_scene)
        loss = criterion(logits, batch_y)

        total_loss += loss.item() * batch_imu.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == batch_y).sum().item()
        total_samples += batch_imu.size(0)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch_y.cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    y_true = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    return avg_loss, avg_acc, y_true, y_pred


def save_loss_curve(train_losses: List[float], val_losses: List[float], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train_loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Baseline Scene Perceiver-IO Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> np.ndarray:
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Baseline Scene Perceiver-IO Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    plt.ylabel("true label")
    plt.xlabel("predicted label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return matrix


def subset_to_name(modalities: Iterable[str]) -> str:
    modality_set = set(modalities)
    ordered = [modality for modality in MODALITY_KEYS if modality in modality_set]
    return "+".join(ordered) if ordered else "none"


def mask_features_for_modalities(
    features: Dict[str, np.ndarray],
    active_modalities: Iterable[str],
) -> Dict[str, np.ndarray]:
    active_set = set(active_modalities)
    masked = {}
    for key, value in features.items():
        if key in active_set:
            masked[key] = value
        else:
            masked[key] = np.zeros_like(value, dtype=np.float32)
    return masked


def evaluate_feature_subset(
    model: nn.Module,
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    scene_targets: np.ndarray,
    criterion: nn.Module,
    label_encoder: LabelEncoder,
) -> Dict[str, float]:
    loader = make_loader(features, labels, scene_targets, BATCH_SIZE, shuffle=False)
    loss, joint_acc, y_true, y_pred = evaluate(model, loader, criterion)

    joint_true_names = label_encoder.inverse_transform(y_true)
    joint_pred_names = label_encoder.inverse_transform(y_pred)
    scene_true = np.array([split_joint_label(label_name)[0] for label_name in joint_true_names], dtype=object)
    scene_pred = np.array([split_joint_label(label_name)[0] for label_name in joint_pred_names], dtype=object)
    intent_true = np.array([split_joint_label(label_name)[1] for label_name in joint_true_names], dtype=object)
    intent_pred = np.array([split_joint_label(label_name)[1] for label_name in joint_pred_names], dtype=object)

    return {
        "loss": float(loss),
        "joint_acc": float(joint_acc),
        "intent_acc": float(np.mean(intent_true == intent_pred)),
        "scene_acc": float(np.mean(scene_true == scene_pred)),
    }


def evaluate_modality_subsets(
    model: nn.Module,
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    scene_targets: np.ndarray,
    criterion: nn.Module,
    label_encoder: LabelEncoder,
) -> Dict[str, Dict[str, object]]:
    subset_metrics: Dict[str, Dict[str, object]] = {}
    total_subsets = 2 ** len(MODALITY_KEYS)
    subset_index = 0

    for subset_size in range(len(MODALITY_KEYS) + 1):
        for subset in combinations(MODALITY_KEYS, subset_size):
            subset_index += 1
            subset_name = subset_to_name(subset)
            print(f"[contribution] evaluate subset {subset_index:02d}/{total_subsets:02d}: {subset_name}")
            masked_features = mask_features_for_modalities(features, subset)
            metrics = evaluate_feature_subset(
                model,
                masked_features,
                labels,
                scene_targets,
                criterion,
                label_encoder,
            )
            subset_metrics[subset_name] = {
                "active_modalities": list(subset),
                **metrics,
            }

    return subset_metrics


def compute_shapley_contributions(
    subset_metrics: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    metric_names = ("joint_acc", "intent_acc", "scene_acc")
    subset_value_map = {
        frozenset(payload["active_modalities"]): payload
        for payload in subset_metrics.values()
    }
    num_modalities = len(MODALITY_KEYS)
    factorial_total = math.factorial(num_modalities)
    full_set = frozenset(MODALITY_KEYS)
    empty_set = frozenset()

    shapley: Dict[str, Dict[str, Dict[str, float]]] = {}
    for modality in MODALITY_KEYS:
        others = [candidate for candidate in MODALITY_KEYS if candidate != modality]
        shapley[modality] = {}

        for metric_name in metric_names:
            value = 0.0
            for subset_size in range(len(others) + 1):
                for subset in combinations(others, subset_size):
                    subset_set = frozenset(subset)
                    with_modality = subset_set | {modality}
                    weight = (
                        math.factorial(subset_size)
                        * math.factorial(num_modalities - subset_size - 1)
                        / factorial_total
                    )
                    delta = float(subset_value_map[with_modality][metric_name]) - float(
                        subset_value_map[subset_set][metric_name]
                    )
                    value += weight * delta

            full_gain = float(subset_value_map[full_set][metric_name]) - float(
                subset_value_map[empty_set][metric_name]
            )
            share = value / full_gain if not np.isclose(full_gain, 0.0) else 0.0
            shapley[modality][metric_name] = {
                "value": float(value),
                "share_of_full_gain": float(share),
            }

    return shapley


def compute_leave_one_out_contributions(
    subset_metrics: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[str, float]]:
    metric_names = ("joint_acc", "intent_acc", "scene_acc", "loss")
    subset_value_map = {
        frozenset(payload["active_modalities"]): payload
        for payload in subset_metrics.values()
    }
    full_set = frozenset(MODALITY_KEYS)

    contributions: Dict[str, Dict[str, float]] = {}
    for modality in MODALITY_KEYS:
        masked_set = full_set - {modality}
        contributions[modality] = {}
        for metric_name in metric_names:
            full_value = float(subset_value_map[full_set][metric_name])
            masked_value = float(subset_value_map[masked_set][metric_name])
            if metric_name == "loss":
                contributions[modality]["loss_increase"] = float(masked_value - full_value)
            else:
                contributions[modality][f"{metric_name}_drop"] = float(full_value - masked_value)
    return contributions


def build_modality_contribution_report(
    subset_metrics: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    empty_subset_name = subset_to_name(())
    full_subset_name = subset_to_name(MODALITY_KEYS)

    return {
        "modalities": [
            {"name": modality, "display_name": MODALITY_DISPLAY_NAMES[modality]}
            for modality in MODALITY_KEYS
        ],
        "empty_subset_metrics": subset_metrics[empty_subset_name],
        "full_subset_metrics": subset_metrics[full_subset_name],
        "shapley": compute_shapley_contributions(subset_metrics),
        "leave_one_out": compute_leave_one_out_contributions(subset_metrics),
    }


def save_modality_contribution_bar_plots(
    contribution_report: Dict[str, object],
    output_dir: Path,
) -> Dict[str, str]:
    shapley = contribution_report["shapley"]  # type: ignore[index]
    metric_names = ("joint_acc", "intent_acc", "scene_acc")
    metric_titles = {
        "joint_acc": "Joint Accuracy Contribution",
        "intent_acc": "Intent Accuracy Contribution",
        "scene_acc": "Scene Accuracy Contribution",
    }
    output_names = {
        "joint_acc": "modality_contribution_joint_bar.png",
        "intent_acc": "modality_contribution_intent_bar.png",
        "scene_acc": "modality_contribution_scene_bar.png",
    }
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
    saved_paths: Dict[str, str] = {}

    for metric_name in metric_names:
        values = [
            float(shapley[modality][metric_name]["value"]) * 100.0  # type: ignore[index]
            for modality in MODALITY_KEYS
        ]
        labels = [MODALITY_DISPLAY_NAMES[modality] for modality in MODALITY_KEYS]

        fig, axis = plt.subplots(figsize=(8, 5))
        bars = axis.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
        axis.set_title(metric_titles[metric_name])
        axis.set_ylabel("Shapley Contribution (pp)")
        axis.grid(True, axis="y", alpha=0.3)
        axis.tick_params(axis="x", rotation=20)

        value_min = min(values)
        value_max = max(values)
        if np.isclose(value_min, value_max):
            pad = 0.5 if np.isclose(value_max, 0.0) else abs(value_max) * 0.2
            axis.set_ylim(value_min - pad, value_max + pad)
        else:
            lower = min(0.0, value_min) - max(0.2, abs(value_min) * 0.15)
            upper = max(0.0, value_max) + max(0.2, abs(value_max) * 0.15)
            axis.set_ylim(lower, upper)

        for bar, value in zip(bars, values):
            vertical_align = "bottom" if value >= 0 else "top"
            offset = 0.12 if value >= 0 else -0.12
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:.2f}",
                ha="center",
                va=vertical_align,
                fontsize=10,
            )

        output_path = output_dir / output_names[metric_name]
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths[metric_name] = str(output_path)

    return saved_paths


# ============================================================
# 6. Main
# ============================================================
def main() -> None:
    set_seed(RANDOM_SEED)
    scene_cache = RealSceneFeatureCache(SCENE_CACHE_DIR)

    print(f"[device] {DEVICE}")
    print(
        f"[config] epochs={EPOCHS}, batch_size={BATCH_SIZE}, patience={PATIENCE}, "
        f"model_dim={MODEL_DIM}, num_latents={NUM_LATENTS}, depth={DEPTH}, heads={NUM_HEADS}"
    )
    print(f"[scene] museum_images={len(SCENE_IMAGE_PATHS['museum'])} office_images={len(SCENE_IMAGE_PATHS['office'])}")

    print("[step] load train split with real scene")
    train_raw_features, train_raw_labels, train_raw_scene_targets, train_scene_selection = load_multimodal_data(
        TRAIN_VIDEO_NAMES,
        scene_cache,
    )
    test_raw_features = None
    test_raw_labels = None
    test_raw_scene_targets = None
    test_scene_selection: Dict[str, Dict[str, object]] = {}

    train_joint_labels_raw = build_joint_labels(train_raw_labels, train_raw_scene_targets)
    if SKIP_TEST_EVAL:
        test_joint_labels_raw = np.array([], dtype=object)
        print("[step] skip test split for train-only timing")
    else:
        print("[step] load test split with real scene")
        test_raw_features, test_raw_labels, test_raw_scene_targets, test_scene_selection = load_multimodal_data(
            TEST_VIDEO_NAMES,
            scene_cache,
        )
        test_joint_labels_raw = build_joint_labels(test_raw_labels, test_raw_scene_targets)

    (
        train_features_raw,
        val_features_raw,
        y_train_raw,
        y_val_raw,
        y_train_scene_raw,
        y_val_scene_raw,
        y_train_joint_raw,
        y_val_joint_raw,
    ) = split_train_val(
        train_raw_features,
        train_raw_labels,
        train_raw_scene_targets,
        train_joint_labels_raw,
    )

    print("[split]")
    print(f"  train {len(y_train_joint_raw)} -> {summarize_joint_labels(y_train_joint_raw)}")
    print(f"  val   {len(y_val_joint_raw)} -> {summarize_joint_labels(y_val_joint_raw)}")
    if SKIP_TEST_EVAL:
        print("  test  skipped")
    else:
        print(f"  test  {len(test_joint_labels_raw)} -> {summarize_joint_labels(test_joint_labels_raw)}")

    print("[step] fit scalers on training split only")
    scalers = fit_scalers(train_features_raw)
    train_features_scaled = apply_scalers(train_features_raw, scalers)
    val_features_scaled = apply_scalers(val_features_raw, scalers)
    test_features_scaled = apply_scalers(test_raw_features, scalers) if not SKIP_TEST_EVAL else None
    print("[feature]")
    print(f"  train IMU     {train_features_scaled['imu'].shape}")
    print(f"  train Gesture {train_features_scaled['gesture'].shape}")
    print(f"  train Audio   {train_features_scaled['audio'].shape}")
    print(f"  train Text    {train_features_scaled['text'].shape}")
    print(f"  train Scene   {train_features_scaled['scene'].shape}")

    y_train, label_encoder = encode_labels(y_train_joint_raw)
    y_val = label_encoder.transform(y_val_joint_raw)
    joint_class_names = label_encoder.classes_.tolist()
    scene_class_names = [SCENE_ID_TO_NAME[index] for index in range(len(SCENE_ID_TO_NAME))]
    intent_class_names = [INTENT_NAMES[index] for index in sorted(INTENT_NAMES)]

    print("[labels]")
    print(f"  joint_classes {joint_class_names}")
    print(f"  scene_names {scene_class_names}")
    print(f"  intent_names {intent_class_names}")

    train_loader = make_loader(train_features_scaled, y_train, y_train_scene_raw, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(val_features_scaled, y_val, y_val_scene_raw, BATCH_SIZE, shuffle=False)
    if SKIP_TEST_EVAL:
        y_test = None
        test_loader = None
    else:
        y_test = label_encoder.transform(test_joint_labels_raw)
        test_loader = make_loader(test_features_scaled, y_test, test_raw_scene_targets, BATCH_SIZE, shuffle=False)

    batch_imu, batch_gesture, batch_audio, batch_text, batch_scene, batch_y, batch_scene_y = next(iter(train_loader))
    print(
        "[sanity] first batch "
        f"imu={tuple(batch_imu.shape)} "
        f"gesture={tuple(batch_gesture.shape)} "
        f"audio={tuple(batch_audio.shape)} "
        f"text={tuple(batch_text.shape)} "
        f"scene={tuple(batch_scene.shape)} "
        f"y={tuple(batch_y.shape)} "
        f"scene_y={tuple(batch_scene_y.shape)}"
    )

    model = PerceiverIOSceneBaseline(
        num_classes=len(label_encoder.classes_),
        model_dim=MODEL_DIM,
        num_latents=NUM_LATENTS,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    ).to(DEVICE)
    with torch.no_grad():
        sanity_logits = model(
            batch_imu.to(DEVICE),
            batch_gesture.to(DEVICE),
            batch_audio.to(DEVICE),
            batch_text.to(DEVICE),
            batch_scene.to(DEVICE),
        )
    print(f"[sanity] joint_logits shape {tuple(sanity_logits.shape)}")
    class_weights = build_class_weights(y_train, len(label_encoder.classes_))
    print(
        "[class_weights]",
        {joint_class_names[i]: float(class_weights[i].item()) for i in range(len(joint_class_names))},
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    checkpoint_path = MODEL_OUTPUT_DIR / "baseline_real_scene_perceiver_io.pt"
    scalers_path = MODEL_OUTPUT_DIR / "scalers.pkl"
    label_encoder_path = MODEL_OUTPUT_DIR / "label_encoder.pkl"
    report_path = MODEL_OUTPUT_DIR / "classification_report.txt"
    intent_report_path = MODEL_OUTPUT_DIR / "intent_classification_report.txt"
    scene_report_path = MODEL_OUTPUT_DIR / "scene_classification_report.txt"
    metrics_path = MODEL_OUTPUT_DIR / "metrics.json"
    loss_curve_path = MODEL_OUTPUT_DIR / "loss_curve.png"
    cm_path = MODEL_OUTPUT_DIR / "confusion_matrix.png"
    intent_cm_path = MODEL_OUTPUT_DIR / "intent_confusion_matrix.png"
    scene_cm_path = MODEL_OUTPUT_DIR / "scene_confusion_matrix.png"
    scene_selection_path = MODEL_OUTPUT_DIR / "scene_selection.json"
    modality_contribution_path = MODEL_OUTPUT_DIR / "modality_contribution.json"
    modality_subset_metrics_path = MODEL_OUTPUT_DIR / "modality_subset_metrics.json"

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []

    print("[step] start training")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )
        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"epoch {epoch:03d}/{EPOCHS:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        improved = (val_acc > best_val_acc) or (
            np.isclose(val_acc, best_val_acc) and val_loss < best_val_loss
        )
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": len(label_encoder.classes_),
                    "model_dim": MODEL_DIM,
                    "num_latents": NUM_LATENTS,
                    "depth": DEPTH,
                    "num_heads": NUM_HEADS,
                    "dropout": DROPOUT,
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[early_stop] no validation improvement for {PATIENCE} epochs")
                break

    if not checkpoint_path.exists():
        raise RuntimeError("Training finished without saving a checkpoint.")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        f"[best] epoch={checkpoint['best_epoch']} "
        f"val_acc={checkpoint['best_val_acc']:.4f} "
        f"val_loss={checkpoint['best_val_loss']:.4f}"
    )

    val_loss, val_acc, y_val_true, y_val_pred = evaluate(
        model,
        val_loader,
        criterion,
    )
    save_loss_curve(train_losses, val_losses, loss_curve_path)

    if SKIP_TEST_EVAL:
        with open(scalers_path, "wb") as file:
            pickle.dump(scalers, file)
        with open(label_encoder_path, "wb") as file:
            pickle.dump(label_encoder, file)
        with open(scene_selection_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "train_videos": train_scene_selection,
                    "test_videos": {},
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

        metrics = {
            "config": {
                "random_seed": RANDOM_SEED,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "patience": PATIENCE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "model_dim": MODEL_DIM,
                "num_latents": NUM_LATENTS,
                "depth": DEPTH,
                "num_heads": NUM_HEADS,
                "dropout": DROPOUT,
                "device": str(DEVICE),
                "train_scene_source": "real",
                "skip_test_eval": True,
                "scene_vit_path": str(LOCAL_VIT_PATH),
                "scene_cache_dir": str(SCENE_CACHE_DIR),
                "real_scene_cache_dir": str(SCENE_CACHE_DIR),
            },
            "splits": {
                "train_samples": int(len(y_train)),
                "val_samples": int(len(y_val)),
                "test_samples": 0,
                "train_joint_distribution": summarize_joint_labels(y_train_joint_raw),
                "val_joint_distribution": summarize_joint_labels(y_val_joint_raw),
                "train_intent_distribution": summarize_labels(y_train_raw),
                "val_intent_distribution": summarize_labels(y_val_raw),
                "train_scene_distribution": summarize_scene_counts([SCENE_ID_TO_NAME[int(x)] for x in y_train_scene_raw.tolist()]),
                "val_scene_distribution": summarize_scene_counts([SCENE_ID_TO_NAME[int(x)] for x in y_val_scene_raw.tolist()]),
            },
            "scene_images": {
                "museum_count": len(SCENE_IMAGE_PATHS["museum"]),
                "office_count": len(SCENE_IMAGE_PATHS["office"]),
            },
            "best_checkpoint": {
                "epoch": int(checkpoint["best_epoch"]),
                "val_acc": float(checkpoint["best_val_acc"]),
                "val_loss": float(checkpoint["best_val_loss"]),
            },
            "final_metrics": {
                "val_loss": float(val_loss),
                "val_joint_acc": float(val_acc),
            },
            "class_names": joint_class_names,
            "intent_class_names": intent_class_names,
            "scene_class_names": scene_class_names,
            "class_weights": {
                joint_class_names[i]: float(class_weights[i].item()) for i in range(len(joint_class_names))
            },
            "curves": {
                "train_loss": [float(value) for value in train_losses],
                "val_loss": [float(value) for value in val_losses],
                "train_acc": [float(value) for value in train_accs],
                "val_acc": [float(value) for value in val_accs],
            },
        }
        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, ensure_ascii=False)

        print("[saved]")
        print(f"  checkpoint      {checkpoint_path}")
        print(f"  scalers         {scalers_path}")
        print(f"  label_encoder   {label_encoder_path}")
        print(f"  metrics         {metrics_path}")
        print(f"  loss_curve      {loss_curve_path}")
        print(f"  scene_selection {scene_selection_path}")
        return

    test_loss, test_acc, y_test_true, y_test_pred = evaluate(
        model,
        test_loader,
        criterion,
    )

    report = classification_report(
        y_test_true,
        y_test_pred,
        labels=np.arange(len(joint_class_names)),
        target_names=joint_class_names,
        zero_division=0,
        digits=4,
    )
    print("[test]")
    print(report)

    joint_true_names = label_encoder.inverse_transform(y_test_true)
    joint_pred_names = label_encoder.inverse_transform(y_test_pred)

    y_test_scene_true = np.array([split_joint_label(label_name)[0] for label_name in joint_true_names], dtype=object)
    y_test_scene_pred = np.array([split_joint_label(label_name)[0] for label_name in joint_pred_names], dtype=object)
    y_test_intent_true = np.array([split_joint_label(label_name)[1] for label_name in joint_true_names], dtype=object)
    y_test_intent_pred = np.array([split_joint_label(label_name)[1] for label_name in joint_pred_names], dtype=object)

    intent_report = classification_report(
        y_test_intent_true,
        y_test_intent_pred,
        labels=intent_class_names,
        target_names=intent_class_names,
        zero_division=0,
        digits=4,
    )
    print("[intent_test]")
    print(intent_report)

    scene_report = classification_report(
        y_test_scene_true,
        y_test_scene_pred,
        labels=scene_class_names,
        target_names=scene_class_names,
        zero_division=0,
        digits=4,
    )
    print("[scene_test]")
    print(scene_report)

    scene_name_to_idx = {name: index for index, name in enumerate(scene_class_names)}
    intent_name_to_idx = {name: index for index, name in enumerate(intent_class_names)}
    y_test_scene_true_idx = np.array([scene_name_to_idx[name] for name in y_test_scene_true], dtype=np.int64)
    y_test_scene_pred_idx = np.array([scene_name_to_idx[name] for name in y_test_scene_pred], dtype=np.int64)
    y_test_intent_true_idx = np.array([intent_name_to_idx[name] for name in y_test_intent_true], dtype=np.int64)
    y_test_intent_pred_idx = np.array([intent_name_to_idx[name] for name in y_test_intent_pred], dtype=np.int64)

    cm = save_confusion_matrix_plot(y_test_true, y_test_pred, joint_class_names, cm_path)
    intent_cm = save_confusion_matrix_plot(
        y_test_intent_true_idx,
        y_test_intent_pred_idx,
        intent_class_names,
        intent_cm_path,
    )
    scene_cm = save_confusion_matrix_plot(
        y_test_scene_true_idx,
        y_test_scene_pred_idx,
        scene_class_names,
        scene_cm_path,
    )
    print("[step] modality contribution analysis on test split")
    modality_subset_metrics = evaluate_modality_subsets(
        model,
        test_features_scaled,
        y_test,
        test_raw_scene_targets,
        criterion,
        label_encoder,
    )
    modality_contribution = build_modality_contribution_report(modality_subset_metrics)
    modality_contribution_plot_paths = save_modality_contribution_bar_plots(
        modality_contribution,
        MODEL_OUTPUT_DIR,
    )

    with open(scalers_path, "wb") as file:
        pickle.dump(scalers, file)
    with open(label_encoder_path, "wb") as file:
        pickle.dump(label_encoder, file)
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)
    with open(intent_report_path, "w", encoding="utf-8") as file:
        file.write(intent_report)
    with open(scene_report_path, "w", encoding="utf-8") as file:
        file.write(scene_report)
    with open(scene_selection_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "train_videos": train_scene_selection,
                "test_videos": test_scene_selection,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    with open(modality_subset_metrics_path, "w", encoding="utf-8") as file:
        json.dump(modality_subset_metrics, file, indent=2, ensure_ascii=False)
    with open(modality_contribution_path, "w", encoding="utf-8") as file:
        json.dump(modality_contribution, file, indent=2, ensure_ascii=False)

    metrics = {
        "config": {
            "random_seed": RANDOM_SEED,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "model_dim": MODEL_DIM,
            "num_latents": NUM_LATENTS,
            "depth": DEPTH,
            "num_heads": NUM_HEADS,
            "dropout": DROPOUT,
            "device": str(DEVICE),
            "train_scene_source": "real",
            "test_scene_source": "real",
            "scene_vit_path": str(LOCAL_VIT_PATH),
            "scene_cache_dir": str(SCENE_CACHE_DIR),
            "real_scene_cache_dir": str(SCENE_CACHE_DIR),
            "modality_contribution_method": "shapley_values_on_masked_test_subsets",
        },
        "splits": {
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "test_samples": int(len(y_test)),
            "train_joint_distribution": summarize_joint_labels(y_train_joint_raw),
            "val_joint_distribution": summarize_joint_labels(y_val_joint_raw),
            "test_joint_distribution": summarize_joint_labels(test_joint_labels_raw),
            "train_intent_distribution": summarize_labels(y_train_raw),
            "val_intent_distribution": summarize_labels(y_val_raw),
            "test_intent_distribution": summarize_labels(test_raw_labels),
            "train_scene_distribution": summarize_scene_counts([SCENE_ID_TO_NAME[int(x)] for x in y_train_scene_raw.tolist()]),
            "val_scene_distribution": summarize_scene_counts([SCENE_ID_TO_NAME[int(x)] for x in y_val_scene_raw.tolist()]),
            "test_scene_distribution": summarize_scene_counts([SCENE_ID_TO_NAME[int(x)] for x in test_raw_scene_targets.tolist()]),
        },
        "scene_images": {
            "museum_count": len(SCENE_IMAGE_PATHS["museum"]),
            "office_count": len(SCENE_IMAGE_PATHS["office"]),
        },
        "best_checkpoint": {
            "epoch": int(checkpoint["best_epoch"]),
            "val_acc": float(checkpoint["best_val_acc"]),
            "val_loss": float(checkpoint["best_val_loss"]),
        },
        "final_metrics": {
            "val_loss": float(val_loss),
            "val_joint_acc": float(val_acc),
            "test_loss": float(test_loss),
            "test_joint_acc": float(test_acc),
            "test_scene_acc": float(np.mean(y_test_scene_true == y_test_scene_pred)),
            "test_intent_acc": float(np.mean(y_test_intent_true == y_test_intent_pred)),
        },
        "class_names": joint_class_names,
        "intent_class_names": intent_class_names,
        "scene_class_names": scene_class_names,
        "class_weights": {
            joint_class_names[i]: float(class_weights[i].item()) for i in range(len(joint_class_names))
        },
        "curves": {
            "train_loss": [float(value) for value in train_losses],
            "val_loss": [float(value) for value in val_losses],
            "train_acc": [float(value) for value in train_accs],
            "val_acc": [float(value) for value in val_accs],
        },
        "joint_confusion_matrix": cm.tolist(),
        "intent_confusion_matrix": intent_cm.tolist(),
        "scene_confusion_matrix": scene_cm.tolist(),
        "modality_contribution": modality_contribution,
    }
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print("[saved]")
    print(f"  checkpoint      {checkpoint_path}")
    print(f"  scalers         {scalers_path}")
    print(f"  label_encoder   {label_encoder_path}")
    print(f"  report          {report_path}")
    print(f"  intent_report   {intent_report_path}")
    print(f"  scene_report    {scene_report_path}")
    print(f"  metrics         {metrics_path}")
    print(f"  loss_curve      {loss_curve_path}")
    print(f"  confusion_mat   {cm_path}")
    print(f"  intent_conf_mat {intent_cm_path}")
    print(f"  scene_conf_mat  {scene_cm_path}")
    print(f"  scene_selection {scene_selection_path}")
    print(f"  modality_subset {modality_subset_metrics_path}")
    print(f"  modality_contrib {modality_contribution_path}")
    print(f"  modality_plot_joint  {modality_contribution_plot_paths['joint_acc']}")
    print(f"  modality_plot_intent {modality_contribution_plot_paths['intent_acc']}")
    print(f"  modality_plot_scene  {modality_contribution_plot_paths['scene_acc']}")


if __name__ == "__main__":
    main()
