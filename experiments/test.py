"""Standard test entry for checkpoints produced by ``experiments/train.py``."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import build_dataloaders  # noqa: E402
from src.data.features import FeatureConfig  # noqa: E402
from src.models.formal_baseline import FormalFeatureBaseline, INTENT_NAMES  # noqa: E402
from src.training.engine import evaluate, write_predictions  # noqa: E402
from src.utils.logger import setup_experiment_logger  # noqa: E402
from src.utils.paths import ensure_runtime_dirs, get_path, load_paths_config  # noqa: E402
from src.utils.seed import DEFAULT_SEED, set_seed  # noqa: E402


def _default_checkpoint(config: dict[str, Any]) -> Path:
    return get_path("outputs", "result_dir", config=config) / "checkpoints" / "best.pt"


def _write_report_csv(path: Path, report: dict[str, Any], loss: float, accuracy: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"metric": "test_loss", "value": f"{loss:.6f}"},
        {"metric": "test_acc", "value": f"{accuracy:.6f}"},
    ]
    for label_name in [INTENT_NAMES[index] for index in sorted(INTENT_NAMES)]:
        label_report = report.get(label_name, {})
        if isinstance(label_report, dict):
            for metric_name in ("precision", "recall", "f1-score", "support"):
                rows.append(
                    {
                        "metric": f"{label_name}_{metric_name}",
                        "value": str(label_report.get(metric_name, "")),
                    }
                )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a clean baseline checkpoint.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_paths_config(args.config)
    ensure_runtime_dirs(config)
    set_seed(args.seed)

    checkpoint_path = args.checkpoint or _default_checkpoint(config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run `python experiments/train.py --config configs/default.yaml` first."
        )

    logger, log_path = setup_experiment_logger("test_clean", file_mode="w")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_kwargs = checkpoint.get("model_kwargs", {})
    feature_values = checkpoint.get("feature_config", {})
    feature_config = FeatureConfig(
        num_video_frames=int(feature_values.get("num_video_frames", 8)),
        image_size=int(feature_values.get("image_size", 112)),
        imu_steps=int(feature_values.get("imu_steps", 10)),
    )

    _, test_loader = build_dataloaders(
        config_path=args.config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        load_features=True,
        feature_config=feature_config,
    )

    model = FormalFeatureBaseline(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, rows, report = evaluate(model, test_loader, criterion, device)

    metrics_dir = get_path("outputs", "metrics_dir", config=config)
    prediction_dir = get_path("outputs", "prediction_dir", config=config)
    metrics_json = metrics_dir / "clean_baseline_test_metrics.json"
    metrics_csv = metrics_dir / "clean_baseline_test_metrics.csv"
    predictions_path = prediction_dir / "clean_baseline_predictions.csv"

    payload = {
        "checkpoint": str(checkpoint_path),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "classification_report": report,
        "feature_config": feature_values,
    }
    metrics_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report_csv(metrics_csv, report, test_loss, test_acc)
    write_predictions(predictions_path, rows)

    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Test loss=%.4f acc=%.4f", test_loss, test_acc)
    logger.info("Metrics JSON: %s", metrics_json)
    logger.info("Metrics CSV: %s", metrics_csv)
    logger.info("Predictions: %s", predictions_path)
    logger.info("Log: %s", log_path)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Metrics JSON: {metrics_json}")
    print(f"Metrics CSV: {metrics_csv}")
    print(f"Predictions: {predictions_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
