"""Precompute cached feature tensors for indexed samples."""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MultimodalIntentDataset
from src.data.features import FeatureConfig, get_feature_cache_dir
from src.utils.logger import setup_experiment_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached multimodal feature tensors.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--sample-index", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--num-video-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--imu-steps", type=int, default=10)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger, log_path = setup_experiment_logger("feature_build", file_mode="w")
    feature_config = FeatureConfig(
        num_video_frames=args.num_video_frames,
        image_size=args.image_size,
        imu_steps=args.imu_steps,
        rebuild_cache=args.rebuild_cache,
    )

    start_time = time.time()
    total_samples = 0
    last_shapes: dict[str, tuple[int, ...]] = {}
    cache_dir = get_feature_cache_dir(args.config)

    for split in args.splits:
        dataset = MultimodalIntentDataset(
            split=split,
            config_path=args.config,
            sample_index_path=args.sample_index,
            validate_files=True,
            load_features=True,
            feature_config=feature_config,
        )
        logger.info("Building features for split=%s samples=%s", split, len(dataset))
        for index in range(len(dataset)):
            item = dataset[index]
            features = item["features"]
            last_shapes = {key: tuple(value.shape) for key, value in features.items()}
            total_samples += 1
            if (index + 1) % 5 == 0 or index + 1 == len(dataset):
                logger.info("split=%s cached %s/%s", split, index + 1, len(dataset))

    elapsed = time.time() - start_time
    logger.info("Feature build complete. total_samples=%s elapsed_sec=%.2f", total_samples, elapsed)
    logger.info("Feature shapes per sample: %s", last_shapes)
    logger.info("Cache dir: %s", cache_dir)
    logger.info("Log file: %s", log_path)

    print("Feature build complete.")
    print(f"Total samples: {total_samples}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Feature shapes per sample: {last_shapes}")
    print(f"Cache dir: {cache_dir}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()

