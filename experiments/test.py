"""Evaluate a trained clean baseline checkpoint on the user_C test split."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.data.build_samples import build_sample_index, summarize_samples  # noqa: E402
from src.data.dataset import MultimodalIntentDataset  # noqa: E402
from src.data.features import get_feature_dims, get_target_timesteps  # noqa: E402
from src.models.formal_baseline import build_formal_baseline_from_config  # noqa: E402
from src.training.engine import (  # noqa: E402
    INTENT_CLASS_NAMES,
    evaluate,
    save_confusion_matrix,
    save_metrics_csv,
    save_predictions,
    save_summary_json,
)
from src.utils.logger import setup_logger  # noqa: E402
from src.utils.paths import ensure_runtime_dirs, get_path, load_config, setup_huggingface_env  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


EXPERIMENT_NAME = "clean_baseline_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Member A clean baseline.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to results/checkpoints/best.pt.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit real test metadata samples.")
    parser.add_argument("--smoke-test", action="store_true", help="Use tiny synthetic samples instead of raw data.")
    return parser.parse_args()


def _output_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        "metrics": get_path("outputs", "metrics_dir", config=config) / "clean_baseline_test_metrics.csv",
        "summary": get_path("outputs", "metrics_dir", config=config) / "clean_baseline_test_summary.json",
        "predictions": get_path("outputs", "prediction_dir", config=config) / "clean_baseline_predictions.csv",
        "log": get_path("outputs", "logs_dir", config=config) / "clean_baseline.log",
        "confusion_matrix": get_path("outputs", "figure_dir", config=config) / "confusion_matrix.png",
    }


def _synthetic_records(config: dict[str, Any], count: int, seed: int) -> list[dict[str, Any]]:
    dims = get_feature_dims(config)
    target_steps = get_target_timesteps(config)
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for index in range(count):
        intent_label = index % len(INTENT_CLASS_NAMES)
        scene_label = index % 2
        scene_name = "office" if scene_label == 0 else "museum"
        intent_name = INTENT_CLASS_NAMES[intent_label]
        records.append(
            {
                "sample_id": f"smoke_test_{index:03d}",
                "user": "user_C",
                "split": "test",
                "intent_label": intent_label,
                "scene_label": scene_label,
                "joint_label": f"{scene_name}_{intent_name}",
                "features": {
                    "imu": rng.normal(0, 0.1, (target_steps, dims["imu"])).astype(np.float32),
                    "gesture": rng.normal(0, 0.1, (target_steps, dims["gesture"])).astype(np.float32),
                    "audio": rng.normal(0, 0.1, (target_steps, dims["audio"])).astype(np.float32),
                    "text": rng.normal(0, 0.1, (target_steps, dims["text"])).astype(np.float32),
                    "scene": rng.normal(0, 0.1, (dims["scene"],)).astype(np.float32),
                },
            }
        )
    return records


def _make_test_loader(args: argparse.Namespace, config: dict[str, Any], seed: int):
    batch_size = args.batch_size or int(config.get("training", {}).get("batch_size", 64))
    num_workers = int(config.get("training", {}).get("num_workers", 0))
    samples, missing = build_sample_index(config)
    if args.smoke_test:
        dataset = MultimodalIntentDataset(_synthetic_records(config, 4, seed + 100), config=config)
    else:
        test_samples = [sample for sample in samples if sample.get("split") == "test"]
        if args.max_samples is not None:
            test_samples = test_samples[: args.max_samples]
        if not test_samples:
            missing_text = "\n".join(f"- {item}" for item in missing[:20])
            raise RuntimeError(f"No user_C test samples are available. Missing items:\n{missing_text}")
        dataset = MultimodalIntentDataset.from_metadata_samples(test_samples, config=config)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, samples, missing


def _labels_from_rows(rows: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    y_true = [int(row["intent_true"]) for row in rows if int(row["intent_true"]) >= 0]
    y_pred = [int(row["intent_pred"]) for row in rows if int(row["intent_true"]) >= 0]
    return y_true, y_pred


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_runtime_dirs(config)
    setup_huggingface_env(config)
    paths = _output_paths(config)
    logger = setup_logger(EXPERIMENT_NAME, log_file=paths["log"], file_mode="a")

    seed = int(config.get("training", {}).get("seed", 42))
    set_seed(seed)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else get_path("outputs", "checkpoint_dir", config=config) / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loader, indexed_samples, missing = _make_test_loader(args, config, seed)
    logger.info("sample_index=%s", summarize_samples(indexed_samples))
    if missing:
        logger.warning("missing_items_count=%s first_items=%s", len(missing), missing[:8])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_formal_baseline_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    metrics, rows = evaluate(model, loader, criterion, device)
    elapsed = time.time() - start_time
    sample_count = max(1, len(rows))
    metrics.update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "model_type": "FormalMultimodalBaseline",
            "training_time": "",
            "avg_test_time_per_sample": round(elapsed / sample_count, 6),
            "status": "smoke_test" if args.smoke_test else "completed",
            "notes": "synthetic smoke data; not an official result" if args.smoke_test else "",
        }
    )

    save_metrics_csv(metrics, paths["metrics"])
    save_predictions(rows, paths["predictions"])
    save_summary_json(
        {
            "metrics": metrics,
            "checkpoint": str(checkpoint_path),
            "smoke_test": bool(args.smoke_test),
            "missing_items_count": len(missing),
            "output_paths": {key: str(value) for key, value in paths.items()},
        },
        paths["summary"],
    )
    y_true, y_pred = _labels_from_rows(rows)
    save_confusion_matrix(y_true, y_pred, paths["confusion_matrix"])

    logger.info("test_metrics=%s", metrics)
    print("[test_done]")
    for key in ("metrics", "summary", "predictions", "log", "confusion_matrix"):
        print(f"  {key}: {paths[key]}")
    print(f"  metrics: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
