"""Smoke test for the lightweight Dataset/DataLoader layer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MultimodalIntentDataset, build_dataloaders, count_labels
from src.data.features import FeatureConfig
from src.utils.logger import setup_experiment_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check sample-index Dataset and DataLoaders.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config. Defaults to configs/default.yaml.",
    )
    parser.add_argument(
        "--sample-index",
        type=Path,
        default=None,
        help="Optional explicit path to sample_index.csv.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--load-features", action="store_true", help="Build cached tensors from raw video/IMU files.")
    parser.add_argument("--num-video-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--imu-steps", type=int, default=10)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--no-validate-files",
        action="store_true",
        help="Skip checking whether indexed raw files exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger, log_path = setup_experiment_logger("dataset_check", file_mode="w")
    validate_files = not args.no_validate_files
    feature_config = FeatureConfig(
        num_video_frames=args.num_video_frames,
        image_size=args.image_size,
        imu_steps=args.imu_steps,
        rebuild_cache=args.rebuild_cache,
    )

    train_dataset = MultimodalIntentDataset(
        split="train",
        config_path=args.config,
        sample_index_path=args.sample_index,
        validate_files=validate_files,
        load_features=args.load_features,
        feature_config=feature_config,
    )
    test_dataset = MultimodalIntentDataset(
        split="test",
        config_path=args.config,
        sample_index_path=args.sample_index,
        validate_files=validate_files,
        load_features=args.load_features,
        feature_config=feature_config,
    )
    train_loader, test_loader = build_dataloaders(
        config_path=args.config,
        sample_index_path=args.sample_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validate_files=validate_files,
        load_features=args.load_features,
        feature_config=feature_config,
    )

    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    logger.info("Dataset check passed.")
    logger.info("Train samples: %s", len(train_dataset))
    logger.info("Test samples: %s", len(test_dataset))
    logger.info("Train label counts: %s", count_labels(train_dataset.labels()))
    logger.info("Test label counts: %s", count_labels(test_dataset.labels()))
    logger.info("Train batch keys: %s", sorted(train_batch.keys()))
    logger.info("Test batch keys: %s", sorted(test_batch.keys()))
    logger.info("Train batch size: %s", len(train_batch["sample_id"]))
    logger.info("Test batch size: %s", len(test_batch["sample_id"]))
    logger.info("Example sample_id: %s", train_batch["sample_id"][0])
    logger.info("Example label tensor shape: %s", tuple(train_batch["intent_label"].shape))
    if args.load_features:
        logger.info(
            "Feature shapes: hololens=%s fisheye=%s imu=%s scene=%s",
            tuple(train_batch["features"]["hololens_frames"].shape),
            tuple(train_batch["features"]["fisheye_frames"].shape),
            tuple(train_batch["features"]["imu"].shape),
            tuple(train_batch["features"]["scene"].shape),
        )
        logger.info("Feature status: %s", train_batch["feature_status"])
    logger.info("Log file: %s", log_path)

    print("Dataset check passed.")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Train label counts: {count_labels(train_dataset.labels())}")
    print(f"Test label counts: {count_labels(test_dataset.labels())}")
    print(f"Train batch keys: {sorted(train_batch.keys())}")
    print(f"Example sample_id: {train_batch['sample_id'][0]}")
    print(f"Example label tensor shape: {tuple(train_batch['intent_label'].shape)}")
    if args.load_features:
        print(
            "Feature shapes: "
            f"hololens={tuple(train_batch['features']['hololens_frames'].shape)} "
            f"fisheye={tuple(train_batch['features']['fisheye_frames'].shape)} "
            f"imu={tuple(train_batch['features']['imu'].shape)} "
            f"scene={tuple(train_batch['features']['scene'].shape)}"
        )
        print(f"Feature status: {train_batch['feature_status']}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
