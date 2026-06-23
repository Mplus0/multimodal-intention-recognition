"""Scaffold runner for modal-noise baseline experiments."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.transforms import build_noise_experiment_matrix
from src.training.evaluate import METRIC_FIELDS, build_empty_metric_row


MEMBER_A_TODO = (
    "TODO(member A): connect to the finalized train/test API before running "
    "this experiment."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare modal-noise baseline pending outputs.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the modal-noise YAML config, e.g. configs/noise.yaml.",
    )
    return parser.parse_args()


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = _resolve_project_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    import yaml

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def ensure_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    outputs = config.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("Config key 'outputs' must be a mapping when provided.")

    defaults = {
        "metrics": "results/metrics",
        "logs": "results/logs",
        "predictions": "results/predictions",
        "figures": "figures",
    }
    dirs: dict[str, Path] = {}
    for key, default_path in defaults.items():
        path = _resolve_project_path(outputs.get(key, default_path))
        path.mkdir(parents=True, exist_ok=True)
        dirs[key] = path
    return dirs


def save_metric_template(rows: list[dict], output_path: Path) -> None:
    fieldnames = list(METRIC_FIELDS)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_and_test_placeholder() -> None:
    # TODO(member A): replace this placeholder after train.py/test.py expose stable APIs.
    raise NotImplementedError(MEMBER_A_TODO)


def _ratio_suffix(noise_ratio: float) -> str:
    return str(int(round(noise_ratio * 100)))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    from src.utils.logger import setup_logger
    from src.utils.seed import DEFAULT_SEED, set_seed

    experiment_config = config.get("experiment", {})
    if not isinstance(experiment_config, dict):
        raise ValueError("Config key 'experiment' must be a mapping when provided.")

    experiment_name = str(experiment_config.get("name", "modal_noise_baseline"))
    model_type = str(experiment_config.get("model_type", "baseline"))
    seed = int(experiment_config.get("seed", DEFAULT_SEED))

    modalities = config.get("modalities")
    if not isinstance(modalities, list):
        raise ValueError("Config key 'modalities' must be a list.")
    noise = config.get("noise", {})
    if not isinstance(noise, dict):
        raise ValueError("Config key 'noise' must be a mapping.")
    noise_ratios = noise.get("ratios")
    if not isinstance(noise_ratios, list):
        raise ValueError("Config key 'noise.ratios' must be a list.")

    matrix = build_noise_experiment_matrix(modalities, noise_ratios)
    output_dirs = ensure_output_dirs(config)
    logger = setup_logger(
        experiment_name,
        log_file=output_dirs["logs"] / f"{experiment_name}.log",
        file_mode="w",
    )
    seed_state = set_seed(seed)
    logger.info("Prepared modal-noise baseline scaffold. seed_state=%s", seed_state)
    logger.info("Noise experiment count: %d", len(matrix))

    rows = []
    for item in matrix:
        target_modality = item["target_modality"]
        noise_ratio = item["noise_ratio"]
        run_name = f"{experiment_name}_{target_modality}_{_ratio_suffix(noise_ratio)}"
        rows.append(
            build_empty_metric_row(
                experiment_name=run_name,
                model_type=model_type,
                target_modality=target_modality,
                noise_ratio=noise_ratio,
                missing_modalities="none",
            )
        )

    metrics_path = output_dirs["metrics"] / "noise_baseline_metrics.csv"
    save_metric_template(rows, metrics_path)
    logger.info("Saved pending metric template: %s", metrics_path)

    train_and_test_placeholder()


if __name__ == "__main__":
    main()
