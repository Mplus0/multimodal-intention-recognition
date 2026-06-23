"""Dataset and DataLoader utilities for the sample index.

The first version keeps data loading intentionally lightweight: it validates
indexed file paths and returns labels plus metadata. Feature extraction can be
plugged in later without changing the train/test split contract.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import (
    FeatureConfig,
    build_imu_sequence,
    build_scene_tensor,
    get_feature_cache_dir,
    load_or_build_video_frames,
)
from src.utils.paths import get_project_root, get_path, load_paths_config
from src.utils.seed import DEFAULT_SEED, make_generator, seed_worker


REQUIRED_COLUMNS = (
    "sample_id",
    "split",
    "user",
    "source_person",
    "scene",
    "intent_label",
    "intent_name",
    "hololens_video",
    "fisheye_video",
    "hololens_path",
    "fisheye_path",
    "imu_path",
)


@dataclass(frozen=True)
class IndexedSample:
    """One row from ``data/processed/sample_index.csv`` with resolved paths."""

    sample_id: str
    split: str
    user: str
    source_person: str
    scene: str
    intent_label: int
    intent_name: str
    hololens_video: str
    fisheye_video: str
    hololens_path: Path
    fisheye_path: Path
    imu_path: Path


def _resolve_index_path(config_path: Path | None = None, sample_index_path: Path | None = None) -> Path:
    """Return the sample-index CSV path from an explicit path or config."""
    if sample_index_path is not None:
        path = Path(sample_index_path)
        return path if path.is_absolute() else get_project_root() / path

    config = load_paths_config(config_path)
    try:
        return get_path("data", "sample_index", config=config)
    except KeyError:
        return get_path("data", "processed_dir", config=config) / "sample_index.csv"


def _resolve_project_path(path_value: str | Path, root: Path) -> Path:
    """Resolve a path from the index against the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def read_sample_index(
    config_path: Path | None = None,
    sample_index_path: Path | None = None,
) -> list[IndexedSample]:
    """Read and validate the generated sample index.

    Args:
        config_path: Optional YAML config path.
        sample_index_path: Optional direct CSV path. Relative paths are
            resolved from the repository root.

    Raises:
        FileNotFoundError: If the sample index does not exist.
        ValueError: If required columns are missing or labels are invalid.
    """
    index_path = _resolve_index_path(config_path, sample_index_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"Sample index not found: {index_path}. "
            "Run `python src/data/build_samples.py --config configs/default.yaml` first."
        )

    root = get_project_root()
    samples: list[IndexedSample] = []
    with index_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing_columns:
            raise ValueError(f"Sample index missing columns: {missing_columns}")

        for row_number, row in enumerate(reader, start=2):
            try:
                intent_label = int(row["intent_label"])
            except ValueError as error:
                raise ValueError(f"Invalid intent_label at row {row_number}: {row['intent_label']}") from error

            samples.append(
                IndexedSample(
                    sample_id=row["sample_id"],
                    split=row["split"],
                    user=row["user"],
                    source_person=row["source_person"],
                    scene=row["scene"],
                    intent_label=intent_label,
                    intent_name=row["intent_name"],
                    hololens_video=row["hololens_video"],
                    fisheye_video=row["fisheye_video"],
                    hololens_path=_resolve_project_path(row["hololens_path"], root),
                    fisheye_path=_resolve_project_path(row["fisheye_path"], root),
                    imu_path=_resolve_project_path(row["imu_path"], root),
                )
            )

    return samples


class MultimodalIntentDataset(Dataset[dict[str, Any]]):
    """Lightweight Dataset backed by the generated sample index.

    The returned dictionary is deliberately stable for future work: metadata
    and raw paths are available now, while optional transforms can later add
    decoded frames, IMU windows, cached features, or fused tensors.
    """

    def __init__(
        self,
        split: str,
        config_path: Path | None = None,
        sample_index_path: Path | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        validate_files: bool = True,
        load_features: bool = False,
        feature_config: FeatureConfig | None = None,
    ) -> None:
        self.split = split
        self.transform = transform
        self.config_path = config_path
        self.load_features = load_features
        self.feature_config = feature_config or FeatureConfig()
        self.feature_cache_dir = get_feature_cache_dir(config_path)
        all_samples = read_sample_index(config_path=config_path, sample_index_path=sample_index_path)
        self.samples = [sample for sample in all_samples if sample.split == split]

        if not self.samples:
            available = sorted({sample.split for sample in all_samples})
            raise ValueError(f"No samples found for split={split!r}. Available splits: {available}")

        if validate_files:
            self.validate_files()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        item: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "split": sample.split,
            "user": sample.user,
            "source_person": sample.source_person,
            "scene": sample.scene,
            "intent_label": torch.tensor(sample.intent_label, dtype=torch.long),
            "intent_name": sample.intent_name,
            "hololens_video": sample.hololens_video,
            "fisheye_video": sample.fisheye_video,
            "paths": {
                "hololens": str(sample.hololens_path),
                "fisheye": str(sample.fisheye_path),
                "imu": str(sample.imu_path),
            },
        }

        if self.load_features:
            item["features"] = {
                "hololens_frames": load_or_build_video_frames(
                    sample_id=sample.sample_id,
                    stream_name="hololens",
                    video_path=sample.hololens_path,
                    cache_dir=self.feature_cache_dir,
                    feature_config=self.feature_config,
                ),
                "fisheye_frames": load_or_build_video_frames(
                    sample_id=sample.sample_id,
                    stream_name="fisheye",
                    video_path=sample.fisheye_path,
                    cache_dir=self.feature_cache_dir,
                    feature_config=self.feature_config,
                ),
                "imu": build_imu_sequence(sample.imu_path, self.feature_config.imu_steps),
                "scene": build_scene_tensor(sample.scene),
            }
            item["feature_status"] = {
                "video": "uniform_frame_sampling",
                "imu": "global_unaligned_sequence",
                "scene": "one_hot",
            }

        if self.transform is not None:
            item = self.transform(item)

        return item

    def validate_files(self) -> None:
        """Raise if any indexed raw file is missing."""
        missing: list[str] = []
        for sample in self.samples:
            for path in (sample.hololens_path, sample.fisheye_path, sample.imu_path):
                if not path.exists():
                    missing.append(f"{sample.sample_id}: {path}")

        if missing:
            preview = "\n".join(missing[:10])
            suffix = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
            raise FileNotFoundError(f"Missing indexed files:\n{preview}{suffix}")

    def labels(self) -> list[int]:
        """Return intent labels in dataset order."""
        return [sample.intent_label for sample in self.samples]


def build_dataloaders(
    config_path: Path | None = None,
    sample_index_path: Path | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
    seed: int = DEFAULT_SEED,
    validate_files: bool = True,
    load_features: bool = False,
    feature_config: FeatureConfig | None = None,
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]]]:
    """Build train and test DataLoaders for the lightweight Dataset."""
    train_dataset = MultimodalIntentDataset(
        split="train",
        config_path=config_path,
        sample_index_path=sample_index_path,
        validate_files=validate_files,
        load_features=load_features,
        feature_config=feature_config,
    )
    test_dataset = MultimodalIntentDataset(
        split="test",
        config_path=config_path,
        sample_index_path=sample_index_path,
        validate_files=validate_files,
        load_features=load_features,
        feature_config=feature_config,
    )

    pin_memory = bool(torch.cuda.is_available())
    generator = make_generator(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    return train_loader, test_loader


def count_labels(labels: Iterable[int]) -> dict[int, int]:
    """Return sorted label counts for quick diagnostics."""
    counts: dict[int, int] = {}
    for label in labels:
        counts[int(label)] = counts.get(int(label), 0) + 1
    return dict(sorted(counts.items()))
