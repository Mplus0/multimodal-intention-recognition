"""Run the clean formal five-modality baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.experiment_runner import run_single_experiment
from src.utils.paths import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clean baseline training and testing.")
    parser.add_argument("--config", default="configs/default.yaml", help="Base project YAML config.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit indexed samples for quick server checks.")
    parser.add_argument("--smoke-test", action="store_true", help="Use synthetic samples; not an official result.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    metrics = run_single_experiment(
        base_config=config,
        experiment_name="clean_baseline",
        model_type="FormalMultimodalBaseline",
        output_prefix="clean_baseline",
        train_transform=None,
        val_transform=None,
        test_transform=None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        smoke_test=args.smoke_test,
        metric_overrides={
            "target_modality": "none",
            "noise_ratio": "none",
            "missing_modalities": "none",
        },
    )
    print("[clean_baseline_done]")
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
