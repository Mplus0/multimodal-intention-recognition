"""Dataset interface for formal five-modality intention recognition samples."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.data.features import (  # noqa: E402
    MODALITY_KEYS,
    FeatureBundle,
    get_feature_dims,
    get_target_timesteps,
    load_or_build_sample_features,
)
from src.utils.paths import load_config  # noqa: E402


def _zero_feature(modality: str, feature_dims: dict[str, int], target_steps: int) -> np.ndarray:
    dim = int(feature_dims[modality])
    if modality == "scene":
        return np.zeros((dim,), dtype=np.float32)
    return np.zeros((target_steps, dim), dtype=np.float32)


def _normalize_single_feature(
    modality: str,
    value: Any,
    feature_dims: dict[str, int],
    target_steps: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    expected_dim = int(feature_dims[modality])
    if modality == "scene":
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        if array.shape != (expected_dim,):
            raise ValueError(f"Expected scene feature shape ({expected_dim},), got {array.shape}.")
        return array

    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (target_steps, expected_dim):
        raise ValueError(
            f"Expected {modality} feature shape ({target_steps}, {expected_dim}), got {array.shape}."
        )
    return array


def _as_int_label(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def make_dataset_record(
    *,
    features: dict[str, Any] | None,
    sample_id: str,
    user: str,
    split: str,
    intent_label: Any = None,
    scene_label: Any = None,
    joint_label: Any = None,
    feature_dims: dict[str, int] | None = None,
    target_steps: int = 10,
) -> dict[str, Any]:
    """Create one item-level dataset record with stable modality keys."""
    dims = feature_dims or get_feature_dims()
    input_features = features or {}
    normalized_features: dict[str, np.ndarray] = {}
    modality_mask: dict[str, bool] = {}
    for modality in MODALITY_KEYS:
        if modality in input_features and input_features[modality] is not None:
            normalized_features[modality] = _normalize_single_feature(
                modality,
                input_features[modality],
                dims,
                target_steps,
            )
            modality_mask[modality] = True
        else:
            normalized_features[modality] = _zero_feature(modality, dims, target_steps)
            modality_mask[modality] = False

    return {
        "features": normalized_features,
        "intent_label": _as_int_label(intent_label),
        "scene_label": _as_int_label(scene_label),
        "joint_label": str(joint_label) if joint_label is not None else "unknown_unknown",
        "sample_id": str(sample_id),
        "user": str(user),
        "split": str(split),
        "modality_mask": modality_mask,
    }


class MultimodalIntentDataset(Dataset):
    """PyTorch Dataset returning the standard Member A sample dictionary."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        transform=None,
        config: dict[str, Any] | None = None,
    ):
        self.config = config or load_config()
        self.feature_dims = get_feature_dims(self.config)
        self.target_steps = get_target_timesteps(self.config)
        self.records = [
            make_dataset_record(
                features=record.get("features"),
                sample_id=record.get("sample_id", index),
                user=record.get("user", "unknown"),
                split=record.get("split", "unknown"),
                intent_label=record.get("intent_label"),
                scene_label=record.get("scene_label"),
                joint_label=record.get("joint_label"),
                feature_dims=self.feature_dims,
                target_steps=self.target_steps,
            )
            for index, record in enumerate(records)
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample = {
            "features": {
                key: torch.from_numpy(record["features"][key].astype(np.float32))
                for key in MODALITY_KEYS
            },
            "intent_label": torch.tensor(record["intent_label"], dtype=torch.long),
            "scene_label": torch.tensor(record["scene_label"], dtype=torch.long),
            "joint_label": record["joint_label"],
            "sample_id": record["sample_id"],
            "user": record["user"],
            "split": record["split"],
            "modality_mask": {
                key: torch.tensor(bool(record["modality_mask"][key]), dtype=torch.bool)
                for key in MODALITY_KEYS
            },
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @classmethod
    def from_feature_bundles(
        cls,
        bundles: Sequence[FeatureBundle],
        metadata: Sequence[dict[str, Any]] | None = None,
        transform=None,
        config: dict[str, Any] | None = None,
    ) -> "MultimodalIntentDataset":
        """Expand video/sample-level feature bundles into item-level records."""
        records: list[dict[str, Any]] = []
        metadata_by_id = {
            str(item.get("sample_id", item.get("video_name", index))): item
            for index, item in enumerate(metadata or [])
        }
        for bundle in bundles:
            meta = metadata_by_id.get(bundle.sample_id, {})
            first_feature = bundle.features[MODALITY_KEYS[0]]
            item_count = int(first_feature.shape[0]) if first_feature.ndim >= 2 else 1
            for item_index in range(item_count):
                item_features = {
                    key: (
                        value[item_index]
                        if value.ndim > (1 if key == "scene" else 2)
                        else value
                    )
                    for key, value in bundle.features.items()
                }
                intent_label = (
                    bundle.intent_labels[item_index]
                    if bundle.intent_labels is not None and len(bundle.intent_labels) > item_index
                    else meta.get("intent_label")
                )
                scene_label = (
                    bundle.scene_labels[item_index]
                    if bundle.scene_labels is not None and len(bundle.scene_labels) > item_index
                    else meta.get("scene_label")
                )
                records.append(
                    {
                        "features": item_features,
                        "intent_label": intent_label,
                        "scene_label": scene_label,
                        "joint_label": meta.get("joint_label"),
                        "sample_id": f"{bundle.sample_id}#{item_index}",
                        "user": meta.get("user", "unknown"),
                        "split": meta.get("split", "unknown"),
                    }
                )
        return cls(records, transform=transform, config=config)

    @classmethod
    def from_metadata_samples(
        cls,
        samples: Sequence[dict[str, Any]],
        transform=None,
        config: dict[str, Any] | None = None,
        use_cache: bool = True,
        rebuild_cache: bool = False,
        build_missing_source_features: bool = True,
        rebuild_source_features: bool = False,
    ) -> "MultimodalIntentDataset":
        """Load features from sample metadata and return an item-level Dataset."""
        active_config = config or load_config()
        bundles = [
            load_or_build_sample_features(
                sample,
                config=active_config,
                use_cache=use_cache,
                rebuild_cache=rebuild_cache,
                build_missing_source_features=build_missing_source_features,
                rebuild_source_features=rebuild_source_features,
            )
            for sample in samples
        ]
        return cls.from_feature_bundles(
            bundles,
            metadata=samples,
            transform=transform,
            config=active_config,
        )
