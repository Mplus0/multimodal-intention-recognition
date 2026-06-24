"""Evaluation field definitions for baseline experiment scaffolds.

Real metric computation must wait until Member A exposes stable prediction
outputs from the test pipeline.
"""

from __future__ import annotations


METRIC_FIELDS = [
    "experiment_name",
    "model_type",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "training_time",
    "avg_test_time_per_sample",
    "target_modality",
    "noise_ratio",
    "missing_modalities",
    "status",
    "notes",
]

_CORE_TBD_FIELDS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "training_time",
    "avg_test_time_per_sample",
]

_MEMBER_A_TODO = (
    "TODO(member A): connect to the finalized train/test API before running "
    "this experiment."
)


def build_empty_metric_row(
    experiment_name: str,
    model_type: str = "baseline",
    **kwargs,
) -> dict:
    """Build a pending metric row without inventing experimental results."""
    row = {field: "" for field in METRIC_FIELDS}
    row["experiment_name"] = experiment_name
    row["model_type"] = model_type
    for field in _CORE_TBD_FIELDS:
        row[field] = "TBD"
    row["status"] = "pending"
    row["notes"] = (
        "Pending scaffold only; Member A train/test/Dataset/DataLoader/model "
        "interfaces are not finalized."
    )

    for key, value in kwargs.items():
        if key not in row:
            row[key] = value
        else:
            row[key] = value
    return row


def evaluate_predictions(y_true, y_pred) -> dict:
    """Placeholder for final metric computation."""
    # TODO(member A): connect this function after test.py returns stable prediction outputs.
    raise NotImplementedError(_MEMBER_A_TODO)
