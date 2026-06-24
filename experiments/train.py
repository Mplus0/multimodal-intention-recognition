"""Train the clean five-modality baseline for Member A."""

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

from src.data.build_features import dry_run as feature_dry_run  # noqa: E402
from src.data.build_samples import build_sample_index, summarize_samples  # noqa: E402
from src.data.dataset import MultimodalIntentDataset  # noqa: E402
from src.data.features import get_feature_dims, get_target_timesteps  # noqa: E402
from src.models.formal_baseline import build_formal_baseline_from_config  # noqa: E402
from src.training.engine import (  # noqa: E402
    INTENT_CLASS_NAMES,
    checkpoint_payload,
    evaluate,
    save_confusion_matrix,
    save_loss_curve,
    save_metrics_csv,
    save_predictions,
    save_summary_json,
    train_one_epoch,
)
from src.utils.logger import setup_logger  # noqa: E402
from src.utils.paths import ensure_runtime_dirs, get_path, load_config, setup_huggingface_env  # noqa: E402
from src.utils.seed import make_generator, seed_worker, set_seed  # noqa: E402


EXPERIMENT_NAME = "clean_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Member A clean baseline.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit real metadata samples for quick checks.")
    parser.add_argument("--smoke-test", action="store_true", help="Use tiny synthetic samples instead of raw data.")
    return parser.parse_args()


def _output_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        "best": get_path("outputs", "checkpoint_dir", config=config) / "best.pt",
        "final": get_path("outputs", "checkpoint_dir", config=config) / "final.pt",
        "metrics": get_path("outputs", "metrics_dir", config=config) / "clean_baseline_metrics.csv",
        "summary": get_path("outputs", "metrics_dir", config=config) / "clean_baseline_summary.json",
        "predictions": get_path("outputs", "prediction_dir", config=config) / "clean_baseline_predictions.csv",
        "log": get_path("outputs", "logs_dir", config=config) / "clean_baseline.log",
        "loss_curve": get_path("outputs", "figure_dir", config=config) / "loss_curve.png",
        "confusion_matrix": get_path("outputs", "figure_dir", config=config) / "confusion_matrix.png",
    }


def _synthetic_records(config: dict[str, Any], split: str, count: int, seed: int) -> list[dict[str, Any]]:
    dims = get_feature_dims(config)
    target_steps = get_target_timesteps(config)
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    user = "user_C" if split == "test" else "user_A"
    for index in range(count):
        intent_label = index % len(INTENT_CLASS_NAMES)
        scene_label = index % 2
        scene_name = "office" if scene_label == 0 else "museum"
        intent_name = INTENT_CLASS_NAMES[intent_label]
        records.append(
            {
                "sample_id": f"smoke_{split}_{index:03d}",
                "user": user,
                "split": split,
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


def _split_real_samples(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_samples = [sample for sample in samples if sample.get("split") == "train"]
    if len(train_samples) < 2:
        return train_samples, train_samples
    val_count = max(1, int(round(len(train_samples) * 0.2)))
    return train_samples[:-val_count], train_samples[-val_count:]


def _make_loaders(args: argparse.Namespace, config: dict[str, Any], seed: int):
    batch_size = args.batch_size or int(config.get("training", {}).get("batch_size", 64))
    num_workers = int(config.get("training", {}).get("num_workers", 0))

    samples, missing = build_sample_index(config)
    if args.smoke_test:
        train_records = _synthetic_records(config, "train", 8, seed)
        val_records = _synthetic_records(config, "val", 4, seed + 1)
        test_records = _synthetic_records(config, "test", 4, seed + 2)
        train_dataset = MultimodalIntentDataset(train_records, config=config)
        val_dataset = MultimodalIntentDataset(val_records, config=config)
        test_dataset = MultimodalIntentDataset(test_records, config=config)
    else:
        if args.max_samples is not None:
            samples = samples[: args.max_samples]
        train_samples, val_samples = _split_real_samples(samples)
        if not train_samples or not val_samples:
            missing_text = "\n".join(f"- {item}" for item in missing[:20])
            raise RuntimeError(f"No train/validation samples are available. Missing items:\n{missing_text}")
        train_dataset = MultimodalIntentDataset.from_metadata_samples(train_samples, config=config)
        val_dataset = MultimodalIntentDataset.from_metadata_samples(val_samples, config=config)
        test_dataset = val_dataset

    generator = make_generator(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    eval_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, eval_loader, test_loader, samples, missing


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
    logger = setup_logger(EXPERIMENT_NAME, log_file=paths["log"], file_mode="w")

    seed = int(config.get("training", {}).get("seed", 42))
    seed_state = set_seed(seed)
    logger.info("seed=%s cuda_available=%s", seed_state.seed, seed_state.cuda_available)

    feature_dry_run(config)
    train_loader, val_loader, test_loader, indexed_samples, missing = _make_loaders(args, config, seed)
    logger.info("sample_index=%s", summarize_samples(indexed_samples))
    if missing:
        logger.warning("missing_items_count=%s first_items=%s", len(missing), missing[:8])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_formal_baseline_from_config(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_macro_f1 = -1.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _ = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(float(train_metrics["loss"]))
        history["val_loss"].append(float(val_metrics["loss"]))
        logger.info(
            "epoch=%s train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["accuracy"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
        )
        if val_metrics["macro_f1"] >= best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            torch.save(checkpoint_payload(model, epoch, val_metrics, config), paths["best"])

    final_metrics, prediction_rows = evaluate(model, test_loader, criterion, device)
    elapsed = time.time() - start_time
    sample_count = max(1, len(prediction_rows))
    final_metrics.update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "model_type": "FormalMultimodalBaseline",
            "training_time": round(elapsed, 4),
            "avg_test_time_per_sample": round(elapsed / sample_count, 6),
            "status": "smoke_test" if args.smoke_test else "completed",
            "notes": "synthetic smoke data; not an official result" if args.smoke_test else "",
        }
    )

    torch.save(checkpoint_payload(model, args.epochs, final_metrics, config), paths["final"])
    save_metrics_csv(final_metrics, paths["metrics"])
    save_predictions(prediction_rows, paths["predictions"])
    save_summary_json(
        {
            "metrics": final_metrics,
            "epochs": args.epochs,
            "smoke_test": bool(args.smoke_test),
            "missing_items_count": len(missing),
            "output_paths": {key: str(value) for key, value in paths.items()},
        },
        paths["summary"],
    )
    y_true, y_pred = _labels_from_rows(prediction_rows)
    save_loss_curve(history, paths["loss_curve"])
    save_confusion_matrix(y_true, y_pred, paths["confusion_matrix"])

    print("[train_done]")
    for key in ("best", "final", "metrics", "summary", "predictions", "log", "loss_curve", "confusion_matrix"):
        print(f"  {key}: {paths[key]}")
    print(f"  metrics: {final_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
