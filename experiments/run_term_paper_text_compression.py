"""Run the individual term-paper method with isolated output directories."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_improved_model import main as run_improved_main  # noqa: E402
from src.utils.paths import load_config  # noqa: E402


def _write_isolated_configs(
    base_config_path: str,
    method_config_path: str,
    target_dir: Path,
) -> Path:
    """Create an effective base config whose generated files stay under results/term_paper."""
    import yaml

    config = copy.deepcopy(load_config(base_config_path))
    method_config = load_config(method_config_path)
    config["outputs"] = {
        "model_dir": "results/term_paper/models",
        "result_dir": "results/term_paper",
        "metrics_dir": "results/term_paper/metrics",
        "logs_dir": "results/term_paper/logs",
        "prediction_dir": "results/term_paper/predictions",
        "checkpoint_dir": "results/term_paper/checkpoints",
        "figure_dir": "results/term_paper/figures",
        "report_dir": "results/term_paper/report",
    }
    config.setdefault("report", {})["screenshots_dir"] = "results/term_paper/report/screenshots"
    target_dir.mkdir(parents=True, exist_ok=True)
    base_target = target_dir / "effective_base.yaml"
    method_target = target_dir / "effective_method.yaml"
    with base_target.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    with method_target.open("w", encoding="utf-8") as file:
        yaml.safe_dump(method_config, file, allow_unicode=True, sort_keys=False)
    return base_target


def main() -> int:
    args = sys.argv[1:]
    base_config = "configs/default.yaml"
    if "--base-config" in args:
        index = args.index("--base-config")
        base_config = args[index + 1]
        del args[index : index + 2]
    if "--config" in args:
        raise ValueError("This entry uses the fixed personal config; do not pass --config.")
    isolated_config = _write_isolated_configs(
        base_config,
        "configs/term_paper_text_compression.yaml",
        REPO_ROOT / "results" / "term_paper" / "configs",
    )
    sys.argv = [
        sys.argv[0],
        "--config",
        "configs/term_paper_text_compression.yaml",
        "--base-config",
        str(isolated_config),
        *args,
    ]
    return run_improved_main()


if __name__ == "__main__":
    raise SystemExit(main())
