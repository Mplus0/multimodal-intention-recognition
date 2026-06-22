"""Lightweight path helpers for local project collaboration.

This module reads path conventions from ``configs/default.yaml`` and resolves
all relative paths from the repository root. It is intentionally standalone and
does not import model, training, testing, or feature extraction code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return the repository root detected from this file location."""
    return Path(__file__).resolve().parents[2]


def load_paths_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load the YAML path configuration.

    Args:
        config_path: Optional path to a YAML config file. When omitted, the
            default path is ``configs/default.yaml`` under the repository root.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML content is not a mapping.
    """
    if config_path is None:
        config_path = get_project_root() / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Path config file not found: {config_path}. "
            "Please create configs/default.yaml first."
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Path config must be a YAML mapping: {config_path}")

    return config


def resolve_path(path_value: str | Path) -> Path:
    """Resolve a config path value against the repository root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return get_project_root() / path


def _get_nested_value(config: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Read a nested value from a dictionary using a sequence of keys."""
    if not keys:
        raise KeyError("At least one key is required.")

    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            dotted_key = ".".join(keys)
            raise KeyError(f"Path key not found in config: {dotted_key}")
        current = current[key]
    return current


def get_path(*keys: str, config: dict[str, Any] | None = None) -> Path:
    """Return a resolved path from nested YAML keys.

    Examples:
        get_path("data", "raw_dir")
        get_path("local_models", "sentence_model")
        get_path("cache", "huggingface")
        get_path("outputs", "metrics_dir")
    """
    if config is None:
        config = load_paths_config()

    path_value = _get_nested_value(config, tuple(keys))
    if not isinstance(path_value, (str, Path)):
        dotted_key = ".".join(keys)
        raise TypeError(f"Config value is not a path string: {dotted_key}")

    return resolve_path(path_value)


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return the same path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


_RUNTIME_DIR_KEYS: tuple[tuple[str, ...], ...] = (
    ("data", "processed_dir"),
    ("cache", "root"),
    ("cache", "huggingface"),
    ("cache", "hf_hub"),
    ("cache", "transformers"),
    ("cache", "scene_cache"),
    ("cache", "feature_cache"),
    ("outputs", "result_dir"),
    ("outputs", "metrics_dir"),
    ("outputs", "logs_dir"),
    ("outputs", "prediction_dir"),
    ("outputs", "figure_dir"),
    ("outputs", "model_dir"),
    ("outputs", "report_dir"),
    ("report", "screenshots_dir"),
)


def ensure_runtime_dirs(config: dict[str, Any] | None = None) -> None:
    """Create safe generated-output directories from the config.

    This function does not create raw dataset directories automatically.
    """
    if config is None:
        config = load_paths_config()

    for keys in _RUNTIME_DIR_KEYS:
        try:
            ensure_dir(get_path(*keys, config=config))
        except KeyError:
            # Missing optional runtime path keys are skipped.
            continue


def setup_huggingface_env(config: dict[str, Any] | None = None) -> None:
    """Set Hugging Face cache environment variables if they are not set."""
    if config is None:
        config = load_paths_config()

    env_to_keys = {
        "HF_HOME": ("cache", "huggingface"),
        "HF_HUB_CACHE": ("cache", "hf_hub"),
        "TRANSFORMERS_CACHE": ("cache", "transformers"),
    }

    for env_name, keys in env_to_keys.items():
        try:
            os.environ.setdefault(env_name, str(get_path(*keys, config=config)))
        except KeyError:
            continue


_VALIDATION_KEYS: tuple[tuple[str, ...], ...] = (
    ("data", "raw_dir"),
    ("data", "processed_dir"),
    ("data", "imu_csv"),
    ("data", "users", "user_a"),
    ("data", "users", "user_b"),
    ("data", "users", "user_c"),
    ("local_models", "root"),
    ("local_models", "sentence_model"),
    ("local_models", "clip_teacher_model"),
    ("local_models", "vit_model"),
    ("cache", "root"),
    ("cache", "huggingface"),
    ("cache", "hf_hub"),
    ("cache", "transformers"),
    ("cache", "scene_cache"),
    ("cache", "feature_cache"),
    ("outputs", "model_dir"),
    ("outputs", "result_dir"),
    ("outputs", "metrics_dir"),
    ("outputs", "logs_dir"),
    ("outputs", "prediction_dir"),
    ("outputs", "figure_dir"),
    ("outputs", "report_dir"),
    ("report", "screenshots_dir"),
    ("report", "references_file"),
)


def _dotted(keys: tuple[str, ...]) -> str:
    return ".".join(keys)


def validate_paths(config: dict[str, Any] | None = None) -> dict[str, bool]:
    """Check important configured paths without failing on missing local data."""
    if config is None:
        config = load_paths_config()

    results: dict[str, bool] = {}
    for keys in _VALIDATION_KEYS:
        name = _dotted(keys)
        try:
            results[name] = get_path(*keys, config=config).exists()
        except (KeyError, TypeError):
            results[name] = False
    return results


def print_path_report(config: dict[str, Any] | None = None) -> None:
    """Print a readable report for important project paths."""
    if config is None:
        config = load_paths_config()

    validation = validate_paths(config)
    for keys in _VALIDATION_KEYS:
        name = _dotted(keys)
        try:
            path = get_path(*keys, config=config)
        except (KeyError, TypeError) as error:
            print(f"[WARN] {name} -> {error}")
            continue

        status = "OK" if validation.get(name, False) else "WARN"
        print(f"[{status}] {name} -> {path}")


if __name__ == "__main__":
    paths_config = load_paths_config()
    setup_huggingface_env(paths_config)
    ensure_runtime_dirs(paths_config)
    print_path_report(paths_config)
