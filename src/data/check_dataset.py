"""Lightweight checks for the Member A sample index and Dataset interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.data.build_samples import build_sample_index, summarize_samples  # noqa: E402
from src.data.dataset import MultimodalIntentDataset  # noqa: E402
from src.data.features import MODALITY_KEYS, describe_feature_shapes, get_feature_dims, get_target_timesteps  # noqa: E402
from src.utils.paths import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check formal five-modality Dataset inputs.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to project YAML config.")
    parser.add_argument("--metadata-json", help="Optional existing sample metadata JSON.")
    parser.add_argument("--limit", type=int, default=1, help="Number of metadata samples to load for feature check.")
    parser.add_argument("--no-cache", action="store_true", help="Do not use feature cache when loading metadata.")
    return parser.parse_args()


def _load_metadata(path_value: str | Path) -> list[dict[str, Any]]:
    path = Path(path_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError("metadata-json must be a list or a mapping with a 'samples' list.")
    return samples


def _print_label_distribution(samples: list[dict[str, Any]]) -> None:
    labels = [
        str(sample.get("intent_name", sample.get("intent_label", "unknown")))
        for sample in samples
    ]
    print("[label_distribution]", dict(Counter(labels)))


def _print_expected_shapes(config: dict[str, Any]) -> None:
    dims = get_feature_dims(config)
    target_steps = get_target_timesteps(config)
    print("[expected_feature_shapes]")
    for key in MODALITY_KEYS:
        if key == "scene":
            print(f"  {key}: ({dims[key]},)")
        else:
            print(f"  {key}: ({target_steps}, {dims[key]})")


def _print_first_sample_keys(sample: dict[str, Any]) -> None:
    print("[first_metadata_sample]")
    print(f"  keys: {sorted(sample.keys())}")
    print(f"  sample_id: {sample.get('sample_id')}")
    print(f"  user: {sample.get('user')}")
    print(f"  split: {sample.get('split')}")
    print(f"  formal_raw_path_keys: {sorted((sample.get('raw_paths') or {}).keys())}")
    print(f"  formal_feature_path_keys: {sorted((sample.get('feature_paths') or {}).keys())}")


def _check_dataset_features(
    samples: list[dict[str, Any]],
    config: dict[str, Any],
    limit: int,
    use_cache: bool,
) -> None:
    selected = samples[: max(limit, 1)]
    try:
        dataset = MultimodalIntentDataset.from_metadata_samples(
            selected,
            config=config,
            use_cache=use_cache,
        )
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
        print(f"[feature_check_error] {error}")
        print("[feature_check_note] Actual feature shapes require sample metadata with existing formal feature_paths.")
        _print_expected_shapes(config)
        return

    if len(dataset) == 0:
        print("[feature_check_error] Dataset is empty after feature loading.")
        return

    first = dataset[0]
    print("[first_dataset_sample]")
    print(f"  keys: {sorted(first.keys())}")
    print(f"  feature_keys: {sorted(first['features'].keys())}")
    print(f"  sample_id: {first['sample_id']}")
    print(f"  user: {first['user']}")
    print(f"  split: {first['split']}")
    shapes = {key: tuple(value.shape) for key, value in first["features"].items()}
    print("[first_feature_shapes]", shapes)
    print("[modality_mask]", {key: bool(value.item()) for key, value in first["modality_mask"].items()})


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.metadata_json:
        samples = _load_metadata(args.metadata_json)
        missing: list[str] = []
    else:
        samples, missing = build_sample_index(config)

    summary = summarize_samples(samples)
    print("[sample_summary]")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    _print_label_distribution(samples)

    if missing:
        print("[missing_items]")
        for item in missing:
            print(f"  - {item}")

    if not samples:
        print("[dataset_check] No samples available. Copy raw data into configured data/raw user directories first.")
        return 0

    _print_first_sample_keys(samples[0])
    _check_dataset_features(samples, config, args.limit, use_cache=not args.no_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
