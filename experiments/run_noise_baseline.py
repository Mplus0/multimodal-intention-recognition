"""Run modal-noise baseline experiments with the formal five-modality baseline."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.transforms import ModalNoiseTransform, build_noise_experiment_matrix
from src.training.experiment_runner import ratio_suffix, run_single_experiment, save_metrics_table
from src.utils.logger import setup_logger
from src.utils.paths import ensure_runtime_dirs, get_path, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modal-noise baseline experiments.")
    parser.add_argument("--config", default="configs/noise.yaml", help="Modal-noise YAML config.")
    parser.add_argument("--base-config", default="configs/default.yaml", help="Base project YAML config.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per run.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit indexed samples for quick server checks.")
    parser.add_argument("--smoke-test", action="store_true", help="Use synthetic samples; not official results.")
    return parser.parse_args()


def _config_seed(exp_config: dict[str, Any], base_config: dict[str, Any]) -> int:
    experiment = exp_config.get("experiment", {})
    if isinstance(experiment, dict) and "seed" in experiment:
        return int(experiment["seed"])
    return int(base_config.get("training", {}).get("seed", 42))


def _with_seed(base_config: dict[str, Any], seed: int) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config.setdefault("training", {})["seed"] = int(seed)
    return config


def main() -> int:
    args = parse_args()
    base_config = load_config(args.base_config)
    exp_config = load_config(args.config)
    seed = _config_seed(exp_config, base_config)
    active_config = _with_seed(base_config, seed)
    ensure_runtime_dirs(active_config)

    experiment_config = exp_config.get("experiment", {})
    if not isinstance(experiment_config, dict):
        raise ValueError("Config key 'experiment' must be a mapping when provided.")
    experiment_name = str(experiment_config.get("name", "modal_noise_baseline"))
    model_type = str(experiment_config.get("model_type", "FormalMultimodalBaseline"))

    modalities = exp_config.get("modalities")
    if not isinstance(modalities, list):
        raise ValueError("Config key 'modalities' must be a list.")
    noise = exp_config.get("noise", {})
    if not isinstance(noise, dict):
        raise ValueError("Config key 'noise' must be a mapping.")
    noise_ratios = noise.get("ratios")
    if not isinstance(noise_ratios, list):
        raise ValueError("Config key 'noise.ratios' must be a list.")

    matrix = build_noise_experiment_matrix(modalities, noise_ratios)
    logger = setup_logger(
        experiment_name,
        log_file=get_path("outputs", "logs_dir", config=active_config) / f"{experiment_name}.log",
        file_mode="w",
    )
    logger.info("Noise experiment count: %d", len(matrix))

    rows: list[dict[str, Any]] = []
    for item in matrix:
        target_modality = item["target_modality"]
        noise_ratio = float(item["noise_ratio"])
        suffix = ratio_suffix(noise_ratio)
        run_name = f"{experiment_name}_{target_modality}_{suffix}"
        transform = ModalNoiseTransform(
            target_modality=target_modality,
            noise_ratio=noise_ratio,
            seed=seed,
        )
        logger.info("Running %s", run_name)
        metrics = run_single_experiment(
            base_config=active_config,
            experiment_name=run_name,
            model_type=model_type,
            output_prefix=run_name,
            train_transform=transform,
            val_transform=transform,
            test_transform=transform,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_samples=args.max_samples,
            smoke_test=args.smoke_test,
            metric_overrides={
                "target_modality": target_modality,
                "noise_ratio": noise_ratio,
                "missing_modalities": "none",
            },
        )
        rows.append(metrics)

    metrics_path = get_path("outputs", "metrics_dir", config=active_config) / "noise_baseline_metrics.csv"
    save_metrics_table(rows, metrics_path)
    logger.info("Saved aggregate metrics: %s", metrics_path)
    print("[noise_baseline_done]")
    print(f"  metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
