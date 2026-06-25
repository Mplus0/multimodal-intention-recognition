"""Run reliability-gated improved-model experiments."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.improved_experiment_runner import (  # noqa: E402
    run_improved_ablation,
    run_improved_clean,
    run_improved_missing,
    run_improved_noise,
    save_combined_improved_metrics,
)
from src.utils.paths import ensure_runtime_dirs, load_config  # noqa: E402


VALID_MODES = ("all", "clean", "noise", "missing", "ablation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run improved reliability-gated model experiments.")
    parser.add_argument("--config", default="configs/improved_model.yaml", help="Improved-model YAML config.")
    parser.add_argument("--base-config", default="configs/default.yaml", help="Base project YAML config.")
    parser.add_argument("--mode", choices=VALID_MODES, default=None, help="Experiment subset to run.")
    parser.add_argument("--epochs", type=int, default=None, help="Override improved config epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override base config batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override improved config learning rate.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit indexed samples for quick server checks.")
    parser.add_argument("--smoke-test", action="store_true", help="Use synthetic samples; not official results.")
    return parser.parse_args()


def _with_seed(base_config: dict[str, Any], improved_config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    seed = int(improved_config.get("experiment", {}).get("seed", config.get("training", {}).get("seed", 42)))
    config.setdefault("training", {})["seed"] = seed
    return config


def _mode_from_config(args: argparse.Namespace, improved_config: dict[str, Any]) -> str:
    if args.mode:
        return args.mode
    return str(improved_config.get("evaluation", {}).get("mode", "all"))


def main() -> int:
    args = parse_args()
    base_config = load_config(args.base_config)
    improved_config = load_config(args.config)
    active_base_config = _with_seed(base_config, improved_config)
    ensure_runtime_dirs(active_base_config)

    mode = _mode_from_config(args, improved_config)
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode}. Expected one of: {', '.join(VALID_MODES)}")

    rows: list[dict[str, Any]] = []
    if mode in ("all", "clean") and bool(improved_config.get("evaluation", {}).get("run_clean", True)):
        rows.append(
            run_improved_clean(
                base_config=active_base_config,
                improved_config=improved_config,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_samples=args.max_samples,
                smoke_test=args.smoke_test,
            )
        )

    if mode in ("all", "noise") and bool(improved_config.get("evaluation", {}).get("run_noise", True)):
        rows.extend(
            run_improved_noise(
                base_config=active_base_config,
                improved_config=improved_config,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_samples=args.max_samples,
                smoke_test=args.smoke_test,
            )
        )

    if mode in ("all", "missing") and bool(improved_config.get("evaluation", {}).get("run_missing", True)):
        rows.extend(
            run_improved_missing(
                base_config=active_base_config,
                improved_config=improved_config,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_samples=args.max_samples,
                smoke_test=args.smoke_test,
            )
        )

    if mode in ("all", "ablation") and bool(improved_config.get("evaluation", {}).get("run_ablation", True)):
        rows.extend(
            run_improved_ablation(
                base_config=active_base_config,
                improved_config=improved_config,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_samples=args.max_samples,
                smoke_test=args.smoke_test,
            )
        )

    combined_path = save_combined_improved_metrics(active_base_config, rows)
    print("[improved_model_done]")
    print(f"  mode: {mode}")
    print(f"  run_count: {len(rows)}")
    print(f"  metrics: {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
