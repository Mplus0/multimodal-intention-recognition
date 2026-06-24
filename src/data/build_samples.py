"""Build sample metadata for the formal five-modality pipeline.

The sample index is based on configured user directories. It keeps the formal
modality keys fixed as ``imu``, ``gesture``, ``audio``, ``text``, and ``scene``.
Raw video folders are treated only as sources, not as experiment modality names.
"""

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

from src.data.features import MODALITY_KEYS  # noqa: E402
from src.utils.paths import get_path, load_config, resolve_path  # noqa: E402


INTENT_NAMES = {
    0: "menu",
    1: "select",
    2: "magnify",
    3: "narrow",
    4: "brush",
    5: "cancel",
}
SCENE_NAME_TO_ID = {"office": 0, "museum": 1}
SCENE_ID_TO_NAME = {value: key for key, value in SCENE_NAME_TO_ID.items()}

TEACHER_VIDEO_LABELS = {
    "interaction_20260306_072344.mp4": 0,
    "interaction_20260227_122606.mp4": 1,
    "interaction_20260227_122952.mp4": 2,
    "interaction_20260227_123354.mp4": 3,
    "interaction_20260227_124559.mp4": 4,
    "interaction_20260227_123745.mp4": 5,
    "interaction_20260131_120024.mp4": 0,
    "interaction_20260227_132951.mp4": 1,
    "interaction_20260227_133408.mp4": 2,
    "interaction_20260131_114156.mp4": 3,
    "interaction_20260131_115150.mp4": 4,
    "interaction_20260131_114852.mp4": 5,
    "interaction_20260301_073041.mp4": 0,
    "interaction_20260301_064753.mp4": 1,
    "interaction_20260306_072721.mp4": 2,
    "interaction_20260301_071948.mp4": 3,
    "interaction_20260131_121548.mp4": 3,
    "interaction_20260301_073435.mp4": 4,
    "interaction_20260301_072503.mp4": 5,
    "interaction_20260131_071552.mp4": 0,
    "interaction_20260131_072412.mp4": 1,
    "interaction_20260131_084300.mp4": 1,
    "interaction_20260131_085611.mp4": 2,
    "interaction_20260131_090139.mp4": 3,
    "interaction_20260131_085207.mp4": 4,
    "interaction_20260131_084732.mp4": 5,
    "interaction_20260131_090917.mp4": 0,
    "interaction_20260131_090541.mp4": 1,
    "interaction_20260131_065459.mp4": 2,
    "interaction_20260131_070722.mp4": 3,
    "interaction_20260131_091657.mp4": 4,
    "interaction_20260131_091249.mp4": 5,
    "interaction_20260306_082346.mp4": 2,
    "interaction_20260306_083107.mp4": 3,
    "interaction_20260306_083434.mp4": 1,
    "interaction_20260306_084406.mp4": 0,
    "interaction_20260306_084853.mp4": 5,
    "interaction_20260306_085830.mp4": 4,
    "interaction_20260306_090441.mp4": 1,
}

OFFICE_VIDEO_NAMES = {
    "interaction_20260306_072344.mp4",
    "interaction_20260227_122606.mp4",
    "interaction_20260227_122952.mp4",
    "interaction_20260227_123354.mp4",
    "interaction_20260227_124559.mp4",
    "interaction_20260227_123745.mp4",
    "interaction_20260131_120024.mp4",
    "interaction_20260227_132951.mp4",
    "interaction_20260227_133408.mp4",
    "interaction_20260131_114156.mp4",
    "interaction_20260131_115150.mp4",
    "interaction_20260131_114852.mp4",
    "interaction_20260301_073041.mp4",
    "interaction_20260301_064753.mp4",
    "interaction_20260306_072721.mp4",
    "interaction_20260301_071948.mp4",
    "interaction_20260131_121548.mp4",
    "interaction_20260301_073435.mp4",
    "interaction_20260301_072503.mp4",
}

AVI_TO_MP4_MAP = {
    "Video_20260306_152340690.avi": "interaction_20260306_072344.mp4",
    "Video_20260227_202553335.avi": "interaction_20260227_122606.mp4",
    "Video_20260227_202953348.avi": "interaction_20260227_122952.mp4",
    "Video_20260227_203348219.avi": "interaction_20260227_123354.mp4",
    "Video_20260227_204553897.avi": "interaction_20260227_124559.mp4",
    "Video_20260227_203753817.avi": "interaction_20260227_123745.mp4",
    "Video_20260131_200029359.avi": "interaction_20260131_120024.mp4",
    "Video_20260227_213001434.avi": "interaction_20260227_132951.mp4",
    "Video_20260227_213404452.avi": "interaction_20260227_133408.mp4",
    "Video_20260131_194205407.avi": "interaction_20260131_114156.mp4",
    "Video_20260131_195202906.avi": "interaction_20260131_115150.mp4",
    "Video_20260131_194854095.avi": "interaction_20260131_114852.mp4",
    "Video_20260301_153037623.avi": "interaction_20260301_073041.mp4",
    "Video_20260301_144803454.avi": "interaction_20260301_064753.mp4",
    "Video_20260306_152721366.avi": "interaction_20260306_072721.mp4",
    "Video_20260301_151942635.avi": "interaction_20260301_071948.mp4",
    "Video_20260131_201556629.avi": "interaction_20260131_121548.mp4",
    "Video_20260301_153434856.avi": "interaction_20260301_073435.mp4",
    "Video_20260301_152459131.avi": "interaction_20260301_072503.mp4",
    "Video_20260131_151559270.avi": "interaction_20260131_071552.mp4",
    "Video_20260131_152410916.avi": "interaction_20260131_072412.mp4",
    "Video_20260131_164304016.avi": "interaction_20260131_084300.mp4",
    "Video_20260131_164745532.avi": "interaction_20260131_084732.mp4",
    "Video_20260131_165208524.avi": "interaction_20260131_085207.mp4",
    "Video_20260131_165614756.avi": "interaction_20260131_085611.mp4",
    "Video_20260131_170142792.avi": "interaction_20260131_090139.mp4",
    "Video_20260131_145524524.avi": "interaction_20260131_065459.mp4",
    "Video_20260131_150734369.avi": "interaction_20260131_070722.mp4",
    "Video_20260131_170539636.avi": "interaction_20260131_090541.mp4",
    "Video_20260131_170919896.avi": "interaction_20260131_090917.mp4",
    "Video_20260131_171253889.avi": "interaction_20260131_091249.mp4",
    "Video_20260131_171648040.avi": "interaction_20260131_091657.mp4",
    "Video_20260306_162401599.avi": "interaction_20260306_082346.mp4",
    "Video_20260306_163105571.avi": "interaction_20260306_083107.mp4",
    "Video_20260306_163434878.avi": "interaction_20260306_083434.mp4",
    "Video_20260306_164407883.avi": "interaction_20260306_084406.mp4",
    "Video_20260306_164902044.avi": "interaction_20260306_084853.mp4",
    "Video_20260306_165839689.avi": "interaction_20260306_085830.mp4",
    "Video_20260306_170449073.avi": "interaction_20260306_090441.mp4",
}
MP4_TO_AVI_MAP = {mp4: avi for avi, mp4 in AVI_TO_MP4_MAP.items()}


def _config_user_key(user_name: str) -> str:
    return user_name.lower()


def _iter_video_files(directory: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not directory.exists():
        return []
    files = []
    for path in directory.iterdir():
        if path.name.startswith("._") or path.name == ".DS_Store":
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            files.append(path)
    return sorted(files)


def _split_for_user(user_name: str, config: dict[str, Any]) -> str:
    split_config = config.get("training", {}).get("split", {})
    train_users = set(split_config.get("train_users", ["user_A", "user_B"]))
    test_users = set(split_config.get("test_users", ["user_C"]))
    if user_name in train_users:
        return "train"
    if user_name in test_users:
        return "test"
    return "unknown"


def _scene_for_video(video_name: str) -> tuple[int | None, str | None]:
    if video_name in OFFICE_VIDEO_NAMES:
        return SCENE_NAME_TO_ID["office"], "office"
    if video_name in TEACHER_VIDEO_LABELS:
        return SCENE_NAME_TO_ID["museum"], "museum"
    return None, None


def _feature_paths_for_video(video_path: Path, config: dict[str, Any]) -> dict[str, str]:
    """Return existing formal-modality feature paths for a video if present."""
    feature_root = get_path("cache", "feature_cache", config=config)
    stem = video_path.stem
    candidates = {
        "imu": feature_root / "imu_features" / f"imu_features_{stem}.npy",
        "gesture": feature_root / "strong_gesture_features" / f"strong_gesture_features_{stem}.npy",
        "audio": feature_root / "audio_features" / f"audio_features_{stem}.npy",
        "text": feature_root / "text_features" / f"text_features_{stem}.npy",
        "scene": feature_root / "scene_features" / f"scene_features_{stem}.npy",
    }
    return {key: str(path) for key, path in candidates.items() if path.exists()}


def build_sample_index(config: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """Build sample metadata for configured users.

    Returns:
        A tuple of ``(samples, missing_items)``. Missing raw data is reported
        instead of raising so check commands can be run before data is copied.
    """
    if config is None:
        config = load_config()

    user_paths = config.get("data", {}).get("users", {})
    subdirs = config.get("data", {}).get("subdirs", {})
    audio_subdir = subdirs.get("hololens", "HoloLens")
    visual_subdir = subdirs.get("fisheye", "fisheye")
    imu_csv = get_path("data", "imu_csv", config=config)

    samples: list[dict[str, Any]] = []
    missing: list[str] = []
    if not imu_csv.exists():
        missing.append(f"data.imu_csv missing: {imu_csv}")

    for user_name in ("user_A", "user_B", "user_C"):
        user_key = _config_user_key(user_name)
        raw_user_path = user_paths.get(user_key)
        if raw_user_path is None:
            missing.append(f"data.users.{user_key} is not configured")
            continue

        user_dir = resolve_path(raw_user_path)
        if not user_dir.exists():
            missing.append(f"{user_name} directory missing: {user_dir}")
            continue

        audio_dir = user_dir / audio_subdir
        visual_dir = user_dir / visual_subdir
        if not audio_dir.exists():
            missing.append(f"{user_name} audio source directory missing: {audio_dir}")
        if not visual_dir.exists():
            missing.append(f"{user_name} visual source directory missing: {visual_dir}")

        visual_by_name = {path.name: path for path in _iter_video_files(visual_dir, (".avi", ".mp4"))}
        for audio_path in _iter_video_files(audio_dir, (".mp4",)):
            visual_name = MP4_TO_AVI_MAP.get(audio_path.name)
            visual_path = visual_by_name.get(visual_name) if visual_name else None
            intent_label = TEACHER_VIDEO_LABELS.get(audio_path.name)
            intent_name = INTENT_NAMES.get(intent_label) if intent_label is not None else None
            scene_label, scene_name = _scene_for_video(audio_path.name)
            joint_label = f"{scene_name}_{intent_name}" if scene_name and intent_name else None

            raw_paths = {
                "imu": str(imu_csv),
                "gesture": str(visual_path) if visual_path else None,
                "audio": str(audio_path),
                "text": str(audio_path),
                "scene": str(visual_path) if visual_path else None,
            }
            samples.append(
                {
                    "sample_id": audio_path.stem,
                    "video_name": audio_path.name,
                    "user": user_name,
                    "split": _split_for_user(user_name, config),
                    "raw_paths": raw_paths,
                    "feature_paths": _feature_paths_for_video(audio_path, config),
                    "intent_label": intent_label,
                    "intent_name": intent_name,
                    "scene_label": scene_label,
                    "scene_name": scene_name,
                    "joint_label": joint_label,
                }
            )

            if visual_path is None:
                missing.append(f"{audio_path.name} visual source not found under {visual_dir}")
            if intent_label is None:
                missing.append(f"{audio_path.name} intent label not found in teacher mapping")
            if scene_label is None:
                missing.append(f"{audio_path.name} scene label not found in teacher mapping")

    return samples, sorted(set(missing))


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Return compact counts for sample index inspection."""
    return {
        "sample_count": len(samples),
        "split_counts": dict(Counter(str(sample.get("split")) for sample in samples)),
        "user_counts": dict(Counter(str(sample.get("user")) for sample in samples)),
        "intent_counts": dict(Counter(str(sample.get("intent_name")) for sample in samples)),
    }


def save_sample_index(samples: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Save sample metadata as JSON."""
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump({"samples": samples}, file, ensure_ascii=False, indent=2)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build formal five-modality sample metadata.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to project YAML config.")
    parser.add_argument(
        "--output",
        default="data/processed/sample_index.json",
        help="Where to save sample metadata JSON.",
    )
    parser.add_argument("--no-save", action="store_true", help="Only print checks; do not write JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    samples, missing = build_sample_index(config)
    summary = summarize_samples(samples)
    print("[sample_summary]")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if missing:
        print("[missing_items]")
        for item in missing:
            print(f"  - {item}")

    if samples and not args.no_save:
        output_path = save_sample_index(samples, args.output)
        print(f"[saved] {output_path}")
    elif not samples:
        print("[info] No samples were indexed. Check missing_items above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
