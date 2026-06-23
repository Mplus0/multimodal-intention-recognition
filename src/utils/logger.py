"""Logging helpers for project scripts.

The helpers in this module keep experiment logging consistent without tying
training, testing, or feature extraction code to a specific experiment runner.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TextIO

from src.utils.paths import get_path, load_paths_config


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _coerce_level(level: int | str) -> int:
    """Convert a logging level name or integer into a logging level integer."""
    if isinstance(level, int):
        return level

    normalized = level.upper()
    if not hasattr(logging, normalized):
        raise ValueError(f"Unknown logging level: {level}")

    value = getattr(logging, normalized)
    if not isinstance(value, int):
        raise ValueError(f"Invalid logging level: {level}")
    return value


def _reset_handlers(logger: logging.Logger) -> None:
    """Remove existing handlers so repeated setup calls do not duplicate logs."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def slugify_log_name(name: str) -> str:
    """Return a filesystem-friendly log name stem."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    slug = slug.strip("._-")
    return slug or "run"


def get_log_path(log_name: str, suffix: str = ".log") -> Path:
    """Return a log path under the configured logs directory.

    Args:
        log_name: Human-readable experiment or script name.
        suffix: File suffix to use. The default is ``.log``.
    """
    config = load_paths_config()
    logs_dir = get_path("outputs", "logs_dir", config=config)
    return logs_dir / f"{slugify_log_name(log_name)}{suffix}"


def setup_logger(
    name: str,
    log_file: str | Path | None = None,
    level: int | str = logging.INFO,
    console: bool = True,
    file_mode: str = "a",
    stream: TextIO | None = None,
) -> logging.Logger:
    """Create or reconfigure a project logger.

    Args:
        name: Logger name, usually ``__name__`` or an experiment name.
        log_file: Optional file path for persistent logs. Parent directories are
            created automatically. When omitted, only console logging is added.
        level: Logging level such as ``"INFO"`` or ``logging.INFO``.
        console: Whether to also log to the console.
        file_mode: File open mode for ``FileHandler``.
        stream: Optional console stream, mainly useful for tests.
    """
    resolved_level = _coerce_level(level)
    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)
    logger.propagate = False
    _reset_handlers(logger)

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    if console:
        console_handler = logging.StreamHandler(stream)
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode=file_mode, encoding="utf-8")
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    return logger


def setup_experiment_logger(
    experiment_name: str,
    level: int | str = logging.INFO,
    console: bool = True,
    file_mode: str = "a",
) -> tuple[logging.Logger, Path]:
    """Create a logger whose file is saved under ``results/logs``."""
    log_path = get_log_path(experiment_name)
    logger = setup_logger(
        experiment_name,
        log_file=log_path,
        level=level,
        console=console,
        file_mode=file_mode,
    )
    return logger, log_path


if __name__ == "__main__":
    demo_logger, demo_path = setup_experiment_logger("logger_demo", file_mode="w")
    demo_logger.info("Logger is ready. log_file=%s", demo_path)
