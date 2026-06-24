"""Reusable baseline experiment runner for clean, noise, and missing-modality runs."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.build_samples import build_sample_index, summarize_samples
from src.data.dataset import MultimodalIntentDataset
from src.data.features import get_feature_dims, get_target_timesteps
from src.models.formal_baseline import build_formal_baseline_from_config
from src.training.engine import (
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
from src.utils.logger import setup_logger
from src.utils.paths import ensure_runtime_dirs, get_path, setup_huggingface_env
from src.utils.seed import make_generator, seed_worker, set_seed


def _safe_name(value: str) -> str:
    return (
        str(value)
        .replace(" ", "_")
        .replace("+", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def ratio_suffix(noise_ratio: float) -> str:
    """Return a stable percent suffix such as ``20`` for ``0.2``."""
    return str(int(round(float(noise_ratio) * 100)))


def output_paths(config: dict[str, Any], output_prefix: str) -> dict[str, Path]:
    """Build all per-run output paths from ``configs/default.yaml``."""
    prefix = _safe_name(output_prefix)
    return {
        "best": get_path("outputs", "checkpoint_dir", config=config) / f"{prefix}_best.pt",
        "final": get_path("outputs", "checkpoint_dir", config=config) / f"{prefix}_final.pt",
        "metrics": get_path("outputs", "metrics_dir", config=config) / f"{prefix}_metrics.csv",
        "summary": get_path("outputs", "metrics_dir", config=config) / f"{prefix}_summary.json",
        "predictions": get_path("outputs", "prediction_dir", config=config) / f"{prefix}_predictions.csv",
        "log": get_path("outputs", "logs_dir", config=config) / f"{prefix}.log",
        "loss_curve": get_path("outputs", "figure_dir", config=config) / f"{prefix}_loss_curve.png",
        "confusion_matrix": get_path("outputs", "figure_dir", config=config) / f"{prefix}_confusion_matrix.png",
    }


def _synthetic_records(config: dict[str, Any], split: str, count: int, seed: int) -> list[dict[str, Any]]:
    dims = get_feature_dims(config)
    target_steps = get_target_timesteps(config)
    rng = np.random.default_rng(seed)
    user = "user_C" if split == "test" else "user_A"
    records: list[dict[str, Any]] = []
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


def _split_train_val(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_samples = [sample for sample in samples if sample.get("split") == "train"]
    if len(train_samples) < 2:
        return train_samples, train_samples
    val_count = max(1, int(round(len(train_samples) * 0.2)))
    return train_samples[:-val_count], train_samples[-val_count:]


def _limit_samples(samples: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return samples
    return samples[: max(0, int(limit))]


def _make_loaders(
    *,
    config: dict[str, Any],
    seed: int,
    batch_size: int,
    num_workers: int,
    train_transform=None,
    val_transform=None,
    test_transform=None,
    max_samples: int | None = None,
    smoke_test: bool = False,
) -> tuple[Any, Any, Any, list[dict[str, Any]], list[str]]:
    samples, missing = build_sample_index(config)
    if smoke_test:
        train_dataset = MultimodalIntentDataset(
            _synthetic_records(config, "train", 8, seed),
            transform=train_transform,
            config=config,
        )
        val_dataset = MultimodalIntentDataset(
            _synthetic_records(config, "val", 4, seed + 1),
            transform=val_transform,
            config=config,
        )
        test_dataset = MultimodalIntentDataset(
            _synthetic_records(config, "test", 4, seed + 2),
            transform=test_transform,
            config=config,
        )
    else:
        if max_samples is not None:
            samples = _limit_samples(samples, max_samples)
        train_samples, val_samples = _split_train_val(samples)
        test_samples = [sample for sample in samples if sample.get("split") == "test"]
        if not train_samples or not val_samples or not test_samples:
            missing_text = "\n".join(f"- {item}" for item in missing[:20])
            raise RuntimeError(
                "Train/validation/test samples are not all available. "
                f"Indexed summary: {summarize_samples(samples)}\n"
                f"Missing items:\n{missing_text}"
            )
        train_dataset = MultimodalIntentDataset.from_metadata_samples(
            train_samples,
            transform=train_transform,
            config=config,
        )
        val_dataset = MultimodalIntentDataset.from_metadata_samples(
            val_samples,
            transform=val_transform,
            config=config,
        )
        test_dataset = MultimodalIntentDataset.from_metadata_samples(
            test_samples,
            transform=test_transform,
            config=config,
        )

    generator = make_generator(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, samples, missing


def _labels_from_rows(rows: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    y_true = [int(row["intent_true"]) for row in rows if int(row["intent_true"]) >= 0]
    y_pred = [int(row["intent_pred"]) for row in rows if int(row["intent_true"]) >= 0]
    return y_true, y_pred


def run_single_experiment(
    *,
    base_config: dict[str, Any],
    experiment_name: str,
    model_type: str = "FormalMultimodalBaseline",
    output_prefix: str,
    train_transform=None,
    val_transform=None,
    test_transform=None,
    epochs: int = 5,
    batch_size: int | None = None,
    lr: float = 1e-3,
    max_samples: int | None = None,
    smoke_test: bool = False,
    metric_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train and test the formal baseline once and save all standard outputs."""
    ensure_runtime_dirs(base_config)
    setup_huggingface_env(base_config)
    paths = output_paths(base_config, output_prefix)
    logger = setup_logger(experiment_name, log_file=paths["log"], file_mode="w")

    seed = int(base_config.get("training", {}).get("seed", 42))
    seed_state = set_seed(seed)
    logger.info("seed=%s cuda_available=%s", seed_state.seed, seed_state.cuda_available)

    active_batch_size = batch_size or int(base_config.get("training", {}).get("batch_size", 64))
    num_workers = int(base_config.get("training", {}).get("num_workers", 0))
    if val_transform is None:
        val_transform = train_transform
    if test_transform is None:
        test_transform = train_transform

    train_loader, val_loader, test_loader, indexed_samples, missing = _make_loaders(
        config=base_config,
        seed=seed,
        batch_size=active_batch_size,
        num_workers=num_workers,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        max_samples=max_samples,
        smoke_test=smoke_test,
    )
    logger.info("sample_index=%s", summarize_samples(indexed_samples))
    if missing:
        logger.warning("missing_items_count=%s first_items=%s", len(missing), missing[:8])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_formal_baseline_from_config(base_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_macro_f1 = -1.0
    train_start = time.time()

    for epoch in range(1, int(epochs) + 1):
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
            torch.save(checkpoint_payload(model, epoch, val_metrics, base_config), paths["best"])

    training_time = time.time() - train_start
    test_start = time.time()
    final_metrics, prediction_rows = evaluate(model, test_loader, criterion, device)
    test_time = time.time() - test_start
    sample_count = max(1, len(prediction_rows))
    final_metrics.update(
        {
            "experiment_name": experiment_name,
            "model_type": model_type,
            "training_time": round(training_time, 4),
            "avg_test_time_per_sample": round(test_time / sample_count, 6),
            "status": "smoke_test" if smoke_test else "completed",
            "notes": "synthetic smoke data; not an official result" if smoke_test else "",
        }
    )
    if metric_overrides:
        final_metrics.update(metric_overrides)

    torch.save(checkpoint_payload(model, int(epochs), final_metrics, base_config), paths["final"])
    save_metrics_csv(final_metrics, paths["metrics"])
    save_predictions(prediction_rows, paths["predictions"])
    save_summary_json(
        {
            "metrics": final_metrics,
            "epochs": int(epochs),
            "smoke_test": bool(smoke_test),
            "missing_items_count": len(missing),
            "output_paths": {key: str(value) for key, value in paths.items()},
        },
        paths["summary"],
    )
    y_true, y_pred = _labels_from_rows(prediction_rows)
    save_loss_curve(history, paths["loss_curve"])
    save_confusion_matrix(y_true, y_pred, paths["confusion_matrix"])
    logger.info("final_metrics=%s", final_metrics)
    return final_metrics


def save_metrics_table(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Save a multi-run metrics table without inventing missing fields."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "experiment_name",
        "model_type",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "loss",
        "scene_accuracy",
        "joint_accuracy",
        "training_time",
        "avg_test_time_per_sample",
        "target_modality",
        "noise_ratio",
        "missing_modalities",
        "status",
        "notes",
    ]
    fieldnames = list(preferred)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
