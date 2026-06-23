"""Standard clean-data training entry for the end-to-end pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
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
from src.training.engine import evaluate, train_one_epoch  # noqa: E402
from src.utils.logger import setup_experiment_logger  # noqa: E402
from src.utils.paths import ensure_runtime_dirs, get_path, load_paths_config  # noqa: E402
from src.utils.seed import DEFAULT_SEED, set_seed  # noqa: E402


def _checkpoint_dir(config: dict[str, Any]) -> Path:
    return get_path("outputs", "result_dir", config=config) / "checkpoints"


def _write_history_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "train_acc", "test_loss", "test_acc"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _save_checkpoint(
    path: Path,
    model: FormalFeatureBaseline,
    epoch: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_kwargs = {
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "imu_steps": args.imu_steps,
        "num_classes": len(INTENT_NAMES),
    }
    feature_config = {
        "num_video_frames": args.num_video_frames,
        "image_size": args.image_size,
        "imu_steps": args.imu_steps,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
            "feature_config": feature_config,
            "epoch": epoch,
            "metrics": metrics,
            "intent_names": INTENT_NAMES,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train clean baseline from raw-data-derived features.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--num-video-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--imu-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_paths_config(args.config)
    ensure_runtime_dirs(config)
    set_seed(args.seed)

    logger, log_path = setup_experiment_logger("train_clean", file_mode="w")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    feature_config = FeatureConfig(
        num_video_frames=args.num_video_frames,
        image_size=args.image_size,
        imu_steps=args.imu_steps,
    )
    train_loader, test_loader = build_dataloaders(
        config_path=args.config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        load_features=True,
        feature_config=feature_config,
    )

    model = FormalFeatureBaseline(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        imu_steps=args.imu_steps,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_dir = _checkpoint_dir(config)
    metrics_dir = get_path("outputs", "metrics_dir", config=config)
    best_path = checkpoint_dir / "best.pt"
    final_path = checkpoint_dir / "final.pt"
    history_path = metrics_dir / "clean_baseline_metrics.csv"
    summary_path = metrics_dir / "clean_baseline_summary.json"
    scalers_path = checkpoint_dir / "scalers.pkl"
    label_encoder_path = checkpoint_dir / "label_encoder.pkl"

    history: list[dict[str, float]] = []
    best_metric: dict[str, Any] | None = None
    best_acc = -1.0
    best_epoch = 0

    logger.info("Training clean baseline on %s", device)
    logger.info("train_batches=%s test_batches=%s", len(train_loader), len(test_loader))
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
        }
        history.append(row)
        logger.info(
            "epoch=%03d train_loss=%.4f train_acc=%.4f test_loss=%.4f test_acc=%.4f",
            epoch,
            train_loss,
            train_acc,
            test_loss,
            test_acc,
        )

        if test_acc > best_acc:
            best_acc = float(test_acc)
            best_epoch = epoch
            best_metric = dict(row)
            _save_checkpoint(best_path, model, epoch, {"best": best_metric, "history": history}, args)

    final_loss, final_acc, _, final_report = evaluate(model, test_loader, criterion, device)
    final_metrics = {
        "best_epoch": best_epoch,
        "best_test_acc": best_acc,
        "best_metric": best_metric,
        "final_test_loss": float(final_loss),
        "final_test_acc": float(final_acc),
        "history": history,
        "classification_report": final_report,
    }
    _save_checkpoint(final_path, model, args.epochs, final_metrics, args)
    _write_history_csv(history_path, history)
    summary_path.write_text(json.dumps(final_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with scalers_path.open("wb") as file:
        pickle.dump({"note": "No sklearn scalers are used in the current raw-tensor baseline."}, file)
    with label_encoder_path.open("wb") as file:
        pickle.dump({"intent_names": INTENT_NAMES}, file)

    logger.info("Best checkpoint: %s", best_path)
    logger.info("Final checkpoint: %s", final_path)
    logger.info("Metrics CSV: %s", history_path)
    logger.info("Summary JSON: %s", summary_path)
    logger.info("Log: %s", log_path)

    print(f"Best test acc: {best_acc:.4f} at epoch {best_epoch}")
    print(f"Final test acc: {final_acc:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Final checkpoint: {final_path}")
    print(f"Metrics CSV: {history_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Scalers placeholder: {scalers_path}")
    print(f"Label encoder: {label_encoder_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
