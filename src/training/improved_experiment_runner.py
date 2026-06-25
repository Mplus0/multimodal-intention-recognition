"""Reusable experiment runner for reliability-gated improved-model runs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.data.improved_transforms import build_random_modality_dropout_from_config
from src.data.transforms import (
    ComposeTransforms,
    DEFAULT_MODALITIES,
    MissingModalityTransform,
    ModalNoiseTransform,
    build_missing_modality_matrix,
    build_noise_experiment_matrix,
    validate_modalities,
)
from src.models.improved_model import build_improved_model_from_config
from src.training.engine import (
    checkpoint_payload,
    evaluate,
    save_confusion_matrix,
    save_loss_curve,
    save_metrics_csv,
    save_predictions,
    save_summary_json,
    train_one_epoch,
)
from src.training.experiment_runner import (
    _labels_from_rows,
    _make_loaders,
    output_paths,
    ratio_suffix,
    save_metrics_table,
)
from src.utils.logger import setup_logger
from src.utils.paths import ensure_runtime_dirs, get_path, setup_huggingface_env
from src.utils.seed import set_seed


def _effective_training_args(
    improved_config: dict[str, Any],
    *,
    epochs: int | None = None,
    lr: float | None = None,
) -> tuple[int, float]:
    training_config = improved_config.get("training", {})
    active_epochs = int(epochs if epochs is not None else training_config.get("epochs", 5))
    active_lr = float(lr if lr is not None else training_config.get("lr", 1e-3))
    return active_epochs, active_lr


def _with_model_variant(
    improved_config: dict[str, Any],
    *,
    use_reliability_gate: bool | None = None,
    use_modality_dropout: bool | None = None,
) -> dict[str, Any]:
    config = {
        key: value.copy() if isinstance(value, dict) else value
        for key, value in improved_config.items()
    }
    model_config = dict(config.get("model", {}).get("improved", {}))
    training_config = dict(config.get("training", {}))
    dropout_config = dict(training_config.get("modality_dropout", {}))

    if use_reliability_gate is not None:
        model_config["use_reliability_gate"] = bool(use_reliability_gate)
    if use_modality_dropout is not None:
        dropout_config["enabled"] = bool(use_modality_dropout)

    config["model"] = dict(config.get("model", {}))
    config["model"]["improved"] = model_config
    training_config["modality_dropout"] = dropout_config
    config["training"] = training_config
    return config


def _training_dropout_transform(improved_config: dict[str, Any], seed: int):
    return build_random_modality_dropout_from_config(improved_config, seed=seed)


def run_improved_single_experiment(
    *,
    base_config: dict[str, Any],
    improved_config: dict[str, Any],
    experiment_name: str,
    output_prefix: str,
    train_transform=None,
    val_transform=None,
    test_transform=None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    max_samples: int | None = None,
    smoke_test: bool = False,
    metric_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train and test the improved model once and save standard outputs."""
    ensure_runtime_dirs(base_config)
    setup_huggingface_env(base_config)
    paths = output_paths(base_config, output_prefix)
    logger = setup_logger(experiment_name, log_file=paths["log"], file_mode="w")

    seed = int(improved_config.get("experiment", {}).get("seed", base_config.get("training", {}).get("seed", 42)))
    seed_state = set_seed(seed)
    logger.info("seed=%s cuda_available=%s", seed_state.seed, seed_state.cuda_available)

    active_epochs, active_lr = _effective_training_args(improved_config, epochs=epochs, lr=lr)
    active_batch_size = batch_size or int(base_config.get("training", {}).get("batch_size", 64))
    num_workers = int(base_config.get("training", {}).get("num_workers", 0))

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
    logger.info("indexed_sample_count=%s missing_items_count=%s", len(indexed_samples), len(missing))
    if missing:
        logger.warning("first_missing_items=%s", missing[:8])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_improved_model_from_config(base_config, improved_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=active_lr)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_macro_f1 = -1.0
    train_start = time.time()

    for epoch in range(1, active_epochs + 1):
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

    model_config = improved_config.get("model", {}).get("improved", {})
    dropout_config = improved_config.get("training", {}).get("modality_dropout", {})
    final_metrics.update(
        {
            "experiment_name": experiment_name,
            "model_type": "ReliabilityGatedMultimodalModel",
            "training_time": round(training_time, 4),
            "avg_test_time_per_sample": round(test_time / sample_count, 6),
            "status": "smoke_test" if smoke_test else "completed",
            "notes": "synthetic smoke data; not an official result" if smoke_test else "",
            "use_reliability_gate": bool(model_config.get("use_reliability_gate", True)),
            "use_modality_dropout": bool(dropout_config.get("enabled", True)),
        }
    )
    if metric_overrides:
        final_metrics.update(metric_overrides)

    torch.save(checkpoint_payload(model, active_epochs, final_metrics, base_config), paths["final"])
    save_metrics_csv(final_metrics, paths["metrics"])
    save_predictions(prediction_rows, paths["predictions"])
    save_summary_json(
        {
            "metrics": final_metrics,
            "epochs": active_epochs,
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


def run_improved_clean(
    *,
    base_config: dict[str, Any],
    improved_config: dict[str, Any],
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    max_samples: int | None = None,
    smoke_test: bool = False,
) -> dict[str, Any]:
    seed = int(improved_config.get("experiment", {}).get("seed", base_config.get("training", {}).get("seed", 42)))
    train_dropout = _training_dropout_transform(improved_config, seed)
    return run_improved_single_experiment(
        base_config=base_config,
        improved_config=improved_config,
        experiment_name="improved_clean",
        output_prefix="improved_clean",
        train_transform=train_dropout,
        val_transform=None,
        test_transform=None,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_samples=max_samples,
        smoke_test=smoke_test,
        metric_overrides={
            "target_modality": "none",
            "noise_ratio": "none",
            "missing_modalities": "none",
            "ablation_variant": "none",
        },
    )


def run_improved_noise(
    *,
    base_config: dict[str, Any],
    improved_config: dict[str, Any],
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    max_samples: int | None = None,
    smoke_test: bool = False,
) -> list[dict[str, Any]]:
    seed = int(improved_config.get("experiment", {}).get("seed", base_config.get("training", {}).get("seed", 42)))
    noise_config = improved_config.get("noise", {})
    modalities = noise_config.get("modalities", list(DEFAULT_MODALITIES))
    ratios = noise_config.get("ratios", [0.2, 0.4, 0.6])
    matrix = build_noise_experiment_matrix(modalities, ratios)

    rows: list[dict[str, Any]] = []
    for item in matrix:
        target_modality = item["target_modality"]
        noise_ratio = float(item["noise_ratio"])
        suffix = ratio_suffix(noise_ratio)
        run_name = f"improved_noise_{target_modality}_{suffix}"
        train_dropout = _training_dropout_transform(improved_config, seed)
        noise_transform = ModalNoiseTransform(target_modality=target_modality, noise_ratio=noise_ratio, seed=seed)
        metrics = run_improved_single_experiment(
            base_config=base_config,
            improved_config=improved_config,
            experiment_name=run_name,
            output_prefix=run_name,
            train_transform=ComposeTransforms([noise_transform, train_dropout]),
            val_transform=noise_transform,
            test_transform=noise_transform,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_samples=max_samples,
            smoke_test=smoke_test,
            metric_overrides={
                "target_modality": target_modality,
                "noise_ratio": noise_ratio,
                "missing_modalities": "none",
                "ablation_variant": "none",
            },
        )
        rows.append(metrics)

    metrics_path = get_path("outputs", "metrics_dir", config=base_config) / "improved_noise_metrics.csv"
    save_metrics_table(rows, metrics_path)
    return rows


def _missing_matrix_from_config(improved_config: dict[str, Any]) -> list[dict[str, Any]]:
    missing_config = improved_config.get("missing_modality", {})
    single_groups = missing_config.get("single")
    pair_groups = missing_config.get("pairs")
    if single_groups is None and pair_groups is None:
        return build_missing_modality_matrix(["imu", "gesture", "audio", "text", "scene"])

    matrix: list[dict[str, Any]] = []
    for groups in (single_groups or [], pair_groups or []):
        for group in groups:
            if not isinstance(group, list):
                raise ValueError("Each missing modality group must be a list.")
            validate_modalities(group)
            matrix.append({"missing_modalities": list(group)})
    return matrix


def run_improved_missing(
    *,
    base_config: dict[str, Any],
    improved_config: dict[str, Any],
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    max_samples: int | None = None,
    smoke_test: bool = False,
) -> list[dict[str, Any]]:
    seed = int(improved_config.get("experiment", {}).get("seed", base_config.get("training", {}).get("seed", 42)))
    rows: list[dict[str, Any]] = []
    for item in _missing_matrix_from_config(improved_config):
        missing_modalities = list(item["missing_modalities"])
        run_name = f"improved_missing_{'_'.join(missing_modalities)}"
        train_dropout = _training_dropout_transform(improved_config, seed)
        missing_transform = MissingModalityTransform(missing_modalities)
        metrics = run_improved_single_experiment(
            base_config=base_config,
            improved_config=improved_config,
            experiment_name=run_name,
            output_prefix=run_name,
            train_transform=ComposeTransforms([missing_transform, train_dropout]),
            val_transform=missing_transform,
            test_transform=missing_transform,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_samples=max_samples,
            smoke_test=smoke_test,
            metric_overrides={
                "target_modality": "none",
                "noise_ratio": "none",
                "missing_modalities": "+".join(missing_modalities),
                "ablation_variant": "none",
            },
        )
        rows.append(metrics)

    metrics_path = get_path("outputs", "metrics_dir", config=base_config) / "improved_missing_modality_metrics.csv"
    save_metrics_table(rows, metrics_path)
    return rows


def run_improved_ablation(
    *,
    base_config: dict[str, Any],
    improved_config: dict[str, Any],
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    max_samples: int | None = None,
    smoke_test: bool = False,
) -> list[dict[str, Any]]:
    variants = improved_config.get("ablation", {}).get("variants", [])
    if not isinstance(variants, list):
        raise ValueError("ablation.variants must be a list.")

    rows: list[dict[str, Any]] = []
    for variant in variants:
        name = str(variant.get("name", "improved_ablation"))
        variant_config = _with_model_variant(
            improved_config,
            use_reliability_gate=bool(variant.get("use_reliability_gate", True)),
            use_modality_dropout=bool(variant.get("use_modality_dropout", True)),
        )
        seed = int(variant_config.get("experiment", {}).get("seed", base_config.get("training", {}).get("seed", 42)))
        train_dropout = _training_dropout_transform(variant_config, seed)
        metrics = run_improved_single_experiment(
            base_config=base_config,
            improved_config=variant_config,
            experiment_name=name,
            output_prefix=name,
            train_transform=train_dropout,
            val_transform=None,
            test_transform=None,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_samples=max_samples,
            smoke_test=smoke_test,
            metric_overrides={
                "target_modality": "none",
                "noise_ratio": "none",
                "missing_modalities": "none",
                "ablation_variant": name,
            },
        )
        rows.append(metrics)

    metrics_path = get_path("outputs", "metrics_dir", config=base_config) / "improved_ablation_metrics.csv"
    save_metrics_table(rows, metrics_path)
    return rows


def save_combined_improved_metrics(base_config: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    metrics_path = get_path("outputs", "metrics_dir", config=base_config) / "improved_model_metrics.csv"
    save_metrics_table(rows, metrics_path)
    return metrics_path
