"""Training and evaluation helpers for multimodal intent models."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.models.formal_baseline import INTENT_NAMES


def move_features(features: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a feature dictionary to the requested device."""
    return {key: value.to(device, non_blocking=True) for key, value in features.items()}


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one supervised training epoch."""
    model.train()
    total_loss = 0.0
    all_targets: list[int] = []
    all_preds: list[int] = []
    for batch in loader:
        targets = batch["intent_label"].to(device)
        logits = model(move_features(batch["features"], device))
        loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item()) * len(targets)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return total_loss / max(len(all_targets), 1), accuracy_score(all_targets, all_preds)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[dict[str, Any]], dict[str, Any]]:
    """Evaluate a model and return loss, accuracy, prediction rows, and report."""
    model.eval()
    total_loss = 0.0
    all_targets: list[int] = []
    all_preds: list[int] = []
    rows: list[dict[str, Any]] = []

    for batch in loader:
        targets = batch["intent_label"].to(device)
        logits = model(move_features(batch["features"], device))
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)

        total_loss += float(loss.item()) * len(targets)
        target_list = targets.cpu().tolist()
        pred_list = preds.cpu().tolist()
        all_targets.extend(target_list)
        all_preds.extend(pred_list)

        for sample_id, true_label, pred_label in zip(batch["sample_id"], target_list, pred_list):
            rows.append(
                {
                    "sample_id": sample_id,
                    "true_label": true_label,
                    "true_name": INTENT_NAMES[int(true_label)],
                    "pred_label": pred_label,
                    "pred_name": INTENT_NAMES[int(pred_label)],
                }
            )

    label_ids = sorted(INTENT_NAMES)
    report = classification_report(
        all_targets,
        all_preds,
        labels=label_ids,
        target_names=[INTENT_NAMES[index] for index in label_ids],
        zero_division=0,
        output_dict=True,
    )
    report["confusion_matrix"] = confusion_matrix(all_targets, all_preds, labels=label_ids).tolist()
    return total_loss / max(len(all_targets), 1), accuracy_score(all_targets, all_preds), rows, report


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write prediction rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["sample_id", "true_label", "true_name", "pred_label", "pred_name"])
        writer.writeheader()
        writer.writerows(rows)

