"""Training and evaluation engine for the clean formal baseline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


INTENT_CLASS_NAMES = ["menu", "select", "magnify", "narrow", "brush", "cancel"]


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor fields in a Dataset batch to the target device."""
    moved = dict(batch)
    moved["features"] = {key: value.to(device) for key, value in batch["features"].items()}
    for key in ("intent_label", "scene_label"):
        if key in batch and torch.is_tensor(batch[key]):
            moved[key] = batch[key].to(device)
    if "modality_mask" in batch:
        moved["modality_mask"] = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch["modality_mask"].items()
        }
    return moved


def _valid_label_mask(labels: torch.Tensor) -> torch.Tensor:
    return labels >= 0


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = batch["intent_label"]
        valid_mask = _valid_label_mask(labels)
        if not bool(valid_mask.any()):
            continue

        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        logits = outputs["intent_logits"]
        loss = criterion(logits[valid_mask], labels[valid_mask])
        loss.backward()
        optimizer.step()

        batch_count = int(valid_mask.sum().item())
        total_loss += float(loss.item()) * batch_count
        total_correct += int((logits[valid_mask].argmax(dim=1) == labels[valid_mask]).sum().item())
        total_count += batch_count

    if total_count == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {"loss": total_loss / total_count, "accuracy": total_correct / total_count}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    scene_true: list[int] = []
    scene_pred: list[int] = []
    rows: list[dict[str, Any]] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = batch["intent_label"]
        valid_mask = _valid_label_mask(labels)
        outputs = model(batch)
        logits = outputs["intent_logits"]
        preds = logits.argmax(dim=1)

        if bool(valid_mask.any()):
            loss = criterion(logits[valid_mask], labels[valid_mask])
            count = int(valid_mask.sum().item())
            total_loss += float(loss.item()) * count
            y_true.extend(labels[valid_mask].cpu().numpy().astype(int).tolist())
            y_pred.extend(preds[valid_mask].cpu().numpy().astype(int).tolist())

        if "scene_logits" in outputs and "scene_label" in batch:
            scene_labels = batch["scene_label"]
            scene_mask = _valid_label_mask(scene_labels)
            if bool(scene_mask.any()):
                scene_true.extend(scene_labels[scene_mask].cpu().numpy().astype(int).tolist())
                scene_pred.extend(outputs["scene_logits"].argmax(dim=1)[scene_mask].cpu().numpy().astype(int).tolist())

        sample_ids = batch.get("sample_id", [""] * int(logits.shape[0]))
        users = batch.get("user", [""] * int(logits.shape[0]))
        splits = batch.get("split", [""] * int(logits.shape[0]))
        joint_labels = batch.get("joint_label", [""] * int(logits.shape[0]))
        for index in range(int(logits.shape[0])):
            true_value = int(labels[index].detach().cpu().item())
            pred_value = int(preds[index].detach().cpu().item())
            rows.append(
                {
                    "sample_id": sample_ids[index],
                    "user": users[index],
                    "split": splits[index],
                    "intent_true": true_value,
                    "intent_pred": pred_value,
                    "intent_true_name": _intent_name(true_value),
                    "intent_pred_name": _intent_name(pred_value),
                    "intent_correct": int(true_value == pred_value),
                    "joint_label": joint_labels[index],
                }
            )

    metrics = compute_metrics(y_true, y_pred)
    if y_true:
        metrics["loss"] = total_loss / len(y_true)
    else:
        metrics["loss"] = 0.0
    if scene_true:
        metrics["scene_accuracy"] = float(accuracy_score(scene_true, scene_pred))
    else:
        metrics["scene_accuracy"] = 0.0
    metrics["joint_accuracy"] = 0.0
    return metrics, rows


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _intent_name(label: int) -> str:
    if 0 <= label < len(INTENT_CLASS_NAMES):
        return INTENT_CLASS_NAMES[label]
    return "unknown"


def save_predictions(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "user",
        "split",
        "intent_true",
        "intent_pred",
        "intent_true_name",
        "intent_pred_name",
        "intent_correct",
        "joint_label",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_metrics_csv(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
        "status",
        "notes",
    ]
    row = {key: metrics.get(key, "") for key in fieldnames}
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def save_summary_json(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def save_loss_curve(history: dict[str, list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    if history.get("train_loss"):
        plt.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="train_loss")
    if history.get("val_loss"):
        plt.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_confusion_matrix(y_true: list[int], y_pred: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(INTENT_CLASS_NAMES)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels) if y_true else np.zeros((len(labels), len(labels)), dtype=int)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(labels, INTENT_CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(labels, INTENT_CLASS_NAMES)
    plt.xlabel("predicted")
    plt.ylabel("true")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def checkpoint_payload(
    model: nn.Module,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "config": config,
    }
