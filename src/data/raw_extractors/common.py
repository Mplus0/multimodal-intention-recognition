"""Shared helpers for raw feature extractor adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

class RawFeatureExtractionError(RuntimeError):
    """Raised when raw data cannot be converted into a source feature file."""


def project_root() -> Path:
    """Return the repository root without importing YAML-based path helpers."""
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a project-relative path value."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root() / path


def sample_id_from(sample: dict[str, Any]) -> str:
    """Return a stable sample id from metadata."""
    for key in ("sample_id", "video_name", "id"):
        value = sample.get(key)
        if value:
            return str(value)
    raise RawFeatureExtractionError("Sample metadata must include sample_id, video_name, or id.")


def raw_path_for(sample: dict[str, Any], modality: str) -> Path:
    """Return and validate a raw source path for one formal modality."""
    raw_paths = sample.get("raw_paths")
    if not isinstance(raw_paths, dict):
        raise RawFeatureExtractionError(
            f"Sample {sample_id_from(sample)} does not contain a raw_paths mapping."
        )

    value = raw_paths.get(modality)
    if not value:
        raise RawFeatureExtractionError(
            f"Sample {sample_id_from(sample)} is missing raw path for modality '{modality}'."
        )

    path = resolve_project_path(str(value))
    if not path.exists():
        raise RawFeatureExtractionError(
            f"Raw path for modality '{modality}' does not exist: {path}"
        )
    return path


def require_file(path: Path, description: str) -> Path:
    """Validate that a required file exists."""
    if not path.exists():
        raise RawFeatureExtractionError(f"Required {description} does not exist: {path}")
    if not path.is_file():
        raise RawFeatureExtractionError(f"Required {description} is not a file: {path}")
    return path


def require_dir(path: Path, description: str) -> Path:
    """Validate that a required directory exists."""
    if not path.exists():
        raise RawFeatureExtractionError(f"Required {description} does not exist: {path}")
    if not path.is_dir():
        raise RawFeatureExtractionError(f"Required {description} is not a directory: {path}")
    return path


def ensure_parent(path: Path) -> Path:
    """Create the parent directory for an output feature file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def optional_import(module_name: str, package_hint: str | None = None):
    """Import an optional dependency or raise a clear extraction error."""
    try:
        return __import__(module_name)
    except ImportError as error:
        package = package_hint or module_name
        raise RawFeatureExtractionError(
            f"Missing Python dependency '{module_name}'. Install package '{package}' "
            "before running raw feature extraction."
        ) from error


def config_path(config: dict[str, Any], *keys: str, description: str) -> Path:
    """Read a project-relative path from nested config keys."""
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            dotted = ".".join(keys)
            raise RawFeatureExtractionError(f"{description} is not configured: {dotted}")
        current = current[key]
    return resolve_project_path(str(current))


def extractor_not_implemented(modality: str, source_reference: str) -> None:
    """Raise a clear first-stage error for adapters that are not wired yet."""
    raise RawFeatureExtractionError(
        f"Raw extractor for modality '{modality}' is not implemented yet. "
        f"Use teacher reference script '{source_reference}' as the implementation source "
        "in the next stage, or provide an existing source .npy feature file/cache."
    )
