"""Build or inspect five-modality features from sample metadata.

This command is intentionally lightweight for Member A stage 2. It can run a
dry-run path/config check without raw data, and it can build actual feature
arrays when a sample metadata JSON file provides source feature paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.data.features import (  # noqa: E402
    MODALITY_KEYS,
    aggregate_feature_bundles,
    describe_feature_shapes,
    get_feature_dims,
    get_modality_keys,
    get_target_timesteps,
    load_or_build_sample_features,
)
from src.data.raw_feature_builder import (  # noqa: E402
    RawFeatureBuildError,
    ensure_sample_source_features,
)
from src.utils.paths import (  # noqa: E402
    ensure_runtime_dirs,
    get_path,
    load_config,
    setup_huggingface_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or dry-run five-modality features.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to project YAML config.",
    )
    parser.add_argument(
        "--metadata-json",
        help=(
            "Optional sample metadata JSON. It may be a list of samples or a "
            "mapping with a 'samples' list. Each sample should provide sample_id "
            "and feature_paths for imu, gesture, audio, text, scene."
        ),
    )
    parser.add_argument("--limit", type=int, default=1, help="Maximum samples to inspect.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Ignore existing feature cache.")
    parser.add_argument("--no-cache", action="store_true", help="Do not read or write complete feature cache.")
    parser.add_argument(
        "--build-missing",
        action="store_true",
        help="Build missing source .npy features from raw_paths before loading feature arrays.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print config/path checks when no metadata is given.")
    return parser.parse_args()


def _load_samples(metadata_path: str | Path) -> list[dict[str, Any]]:
    path = Path(metadata_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        samples = payload.get("samples")
    else:
        samples = payload

    if not isinstance(samples, list):
        raise ValueError("Sample metadata JSON must be a list or contain a 'samples' list.")
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Sample metadata item {index} is not a mapping.")
    return samples


def _print_expected_shapes(config: dict[str, Any]) -> None:
    dims = get_feature_dims(config)
    target_steps = get_target_timesteps(config)
    print("[expected_shapes]")
    for key in MODALITY_KEYS:
        if key == "scene":
            print(f"  {key}: (N, {dims[key]})")
        else:
            print(f"  {key}: (N, {target_steps}, {dims[key]})")


def _print_source_path_status(config: dict[str, Any]) -> None:
    checks = (
        ("data.raw_dir", ("data", "raw_dir")),
        ("data.imu_csv", ("data", "imu_csv")),
        ("data.users.user_a", ("data", "users", "user_a")),
        ("data.users.user_b", ("data", "users", "user_b")),
        ("data.users.user_c", ("data", "users", "user_c")),
        ("local_models.sentence_model", ("local_models", "sentence_model")),
        ("local_models.vit_model", ("local_models", "vit_model")),
        ("cache.feature_cache", ("cache", "feature_cache")),
        ("cache.scene_cache", ("cache", "scene_cache")),
    )
    print("[path_status]")
    for name, keys in checks:
        path = get_path(*keys, config=config)
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {name} -> {path}")


def dry_run(config: dict[str, Any]) -> None:
    get_modality_keys(config)
    ensure_runtime_dirs(config)
    setup_huggingface_env(config)
    print("[modalities]", ", ".join(MODALITY_KEYS))
    print(f"[target_timesteps] {get_target_timesteps(config)}")
    _print_expected_shapes(config)
    _print_source_path_status(config)
    print(
        "[dry_run] No sample metadata was provided, so no actual feature arrays "
        "were built. Provide --metadata-json to print real output shapes."
    )


def build_from_metadata(args: argparse.Namespace, config: dict[str, Any]) -> int:
    samples = _load_samples(args.metadata_json)
    if not samples:
        print("[ERROR] Sample metadata JSON contains no samples.")
        return 2

    selected = samples[: max(args.limit, 1)]
    bundles = []
    for sample in selected:
        if args.build_missing:
            source_paths = ensure_sample_source_features(sample, config)
            sample = dict(sample)
            sample["feature_paths"] = {key: str(path) for key, path in source_paths.items()}
            sample["feature_exists"] = {key: path.exists() for key, path in source_paths.items()}
        bundles.append(
            load_or_build_sample_features(
                sample,
                config=config,
                use_cache=not args.no_cache,
                rebuild_cache=args.rebuild_cache,
            )
        )

    features = aggregate_feature_bundles(bundles)
    print("[actual_shapes]")
    for key, shape in describe_feature_shapes(features).items():
        print(f"  {key}: {shape}")
    return 0


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.metadata_json:
        try:
            return build_from_metadata(args, config)
        except (FileNotFoundError, ValueError, RuntimeError, KeyError, RawFeatureBuildError) as error:
            print(f"[ERROR] {error}")
            return 2

    dry_run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
