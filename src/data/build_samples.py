"""Build and validate the project sample index.

This script turns the teacher-provided video lists, intent labels, and
fisheye-to-HoloLens mapping into a tabular sample index. The index is the
first reusable input for the end-to-end train/test pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import ensure_runtime_dirs, get_path, load_paths_config

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

INTENT_NAMES = {
    0: "menu",
    1: "select",
    2: "magnify",
    3: "narrow",
    4: "brush",
    5: "cancel",
}

VIDEO_LABELS = {
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

TEST_VIDEO_NAMES = [
    "interaction_20260306_072344.mp4",
    "interaction_20260227_122606.mp4",
    "interaction_20260227_122952.mp4",
    "interaction_20260227_123354.mp4",
    "interaction_20260227_124559.mp4",
    "interaction_20260227_123745.mp4",
    "interaction_20260306_082346.mp4",
    "interaction_20260306_083107.mp4",
    "interaction_20260306_083434.mp4",
    "interaction_20260306_084406.mp4",
    "interaction_20260306_084853.mp4",
    "interaction_20260306_085830.mp4",
    "interaction_20260306_090441.mp4",
]

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

SCENE_BY_VIDEO = {video_name: "office" for video_name in OFFICE_VIDEO_NAMES}
SCENE_BY_VIDEO.update({video_name: "museum" for video_name in set(VIDEO_LABELS) - OFFICE_VIDEO_NAMES})

USER_BY_SOURCE = {
    "Luo": "user_A",
    "Gu": "user_B",
    "Bian": "user_C",
}

LUO_VIDEO_NAMES = {
    "interaction_20260131_120024.mp4",
    "interaction_20260227_132951.mp4",
    "interaction_20260227_133408.mp4",
    "interaction_20260131_114156.mp4",
    "interaction_20260131_115150.mp4",
    "interaction_20260131_114852.mp4",
    "interaction_20260131_071552.mp4",
    "interaction_20260131_072412.mp4",
    "interaction_20260131_084300.mp4",
    "interaction_20260131_084732.mp4",
    "interaction_20260131_085207.mp4",
    "interaction_20260131_085611.mp4",
    "interaction_20260131_090139.mp4",
}

GU_VIDEO_NAMES = {
    "interaction_20260301_073041.mp4",
    "interaction_20260301_064753.mp4",
    "interaction_20260306_072721.mp4",
    "interaction_20260301_071948.mp4",
    "interaction_20260131_121548.mp4",
    "interaction_20260301_073435.mp4",
    "interaction_20260301_072503.mp4",
    "interaction_20260131_065459.mp4",
    "interaction_20260131_070722.mp4",
    "interaction_20260131_090541.mp4",
    "interaction_20260131_090917.mp4",
    "interaction_20260131_091249.mp4",
    "interaction_20260131_091657.mp4",
}

BIAN_VIDEO_NAMES = set(TEST_VIDEO_NAMES)

SOURCE_BY_VIDEO = {video_name: "Luo" for video_name in LUO_VIDEO_NAMES}
SOURCE_BY_VIDEO.update({video_name: "Gu" for video_name in GU_VIDEO_NAMES})
SOURCE_BY_VIDEO.update({video_name: "Bian" for video_name in BIAN_VIDEO_NAMES})


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    split: str
    user: str
    source_person: str
    scene: str
    intent_label: int
    intent_name: str
    hololens_video: str
    fisheye_video: str
    hololens_path: str
    fisheye_path: str
    imu_path: str


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()


def _list_file_names(path: Path, pattern: str) -> set[str]:
    if not path.exists():
        return set()
    return {file_path.name for file_path in path.glob(pattern) if file_path.is_file()}


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _count_by(records: list[SampleRecord], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(getattr(record, key))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def build_sample_index(config_path: Path | None = None) -> tuple[list[SampleRecord], dict[str, object]]:
    config = load_paths_config(config_path)
    ensure_runtime_dirs(config)

    project_root = get_path("project", "root", config=config)
    raw_dir = get_path("data", "raw_dir", config=config)
    imu_path = get_path("data", "imu_csv", config=config)
    hololens_dir = raw_dir / config["data"]["subdirs"]["hololens"]
    fisheye_dir = raw_dir / config["data"]["subdirs"]["fisheye"]

    available_mp4 = _list_file_names(hololens_dir, "interaction_*.mp4")
    available_avi = _list_file_names(fisheye_dir, "*.avi")
    mp4_to_avi = {mp4_name: avi_name for avi_name, mp4_name in AVI_TO_MP4_MAP.items()}

    records: list[SampleRecord] = []
    missing_mp4: list[str] = []
    missing_avi: list[str] = []
    missing_user: list[str] = []
    missing_scene: list[str] = []
    missing_mapping: list[str] = []

    for video_name in sorted(VIDEO_LABELS):
        if video_name not in available_mp4:
            missing_mp4.append(video_name)
            continue
        if video_name not in mp4_to_avi:
            missing_mapping.append(video_name)
            continue

        avi_name = mp4_to_avi[video_name]
        if avi_name not in available_avi:
            missing_avi.append(avi_name)
            continue

        if video_name not in SOURCE_BY_VIDEO:
            missing_user.append(video_name)
            continue
        if video_name not in SCENE_BY_VIDEO:
            missing_scene.append(video_name)
            continue

        source_person = SOURCE_BY_VIDEO[video_name]
        intent_label = int(VIDEO_LABELS[video_name])
        split = "test" if video_name in TEST_VIDEO_NAMES else "train"
        scene = SCENE_BY_VIDEO[video_name]
        sample_id = f"{split}_{source_person.lower()}_{Path(video_name).stem.replace('interaction_', '')}"

        records.append(
            SampleRecord(
                sample_id=sample_id,
                split=split,
                user=USER_BY_SOURCE[source_person],
                source_person=source_person,
                scene=scene,
                intent_label=intent_label,
                intent_name=INTENT_NAMES[intent_label],
                hololens_video=video_name,
                fisheye_video=avi_name,
                hololens_path=_relative_to_project(hololens_dir / video_name, project_root),
                fisheye_path=_relative_to_project(fisheye_dir / avi_name, project_root),
                imu_path=_relative_to_project(imu_path, project_root),
            )
        )

    mapped_mp4 = set(mp4_to_avi)
    labeled_mp4 = set(VIDEO_LABELS)
    used_mp4 = {record.hololens_video for record in records}
    used_avi = {record.fisheye_video for record in records}

    stats: dict[str, object] = {
        "total_samples": len(records),
        "train_samples": sum(record.split == "train" for record in records),
        "test_samples": sum(record.split == "test" for record in records),
        "available_hololens_mp4": len(available_mp4),
        "available_fisheye_avi": len(available_avi),
        "labeled_videos": len(labeled_mp4),
        "mapped_pairs": len(AVI_TO_MP4_MAP),
        "split_counts": _count_by(records, "split"),
        "user_counts": _count_by(records, "user"),
        "scene_counts": _count_by(records, "scene"),
        "intent_counts": _count_by(records, "intent_name"),
        "missing_mp4": missing_mp4,
        "missing_avi": missing_avi,
        "missing_user": missing_user,
        "missing_scene": missing_scene,
        "missing_mapping": missing_mapping,
        "extra_mp4_files": sorted(available_mp4 - labeled_mp4),
        "mapped_but_unlabeled_mp4": sorted(mapped_mp4 - labeled_mp4),
        "labeled_but_unmapped_mp4": sorted(labeled_mp4 - mapped_mp4),
        "unused_avi_files": sorted(available_avi - used_avi),
        "unused_labeled_mp4": sorted(labeled_mp4 - used_mp4),
        "imu_exists": imu_path.exists(),
        "hololens_dir_exists": hololens_dir.exists(),
        "fisheye_dir_exists": fisheye_dir.exists(),
    }
    return records, stats


def save_outputs(
    records: list[SampleRecord],
    stats: dict[str, object],
    config_path: Path | None = None,
) -> dict[str, Path]:
    config = load_paths_config(config_path)
    processed_dir = get_path("data", "processed_dir", config=config)
    metrics_dir = get_path("outputs", "metrics_dir", config=config)
    logs_dir = get_path("outputs", "logs_dir", config=config)

    index_path = processed_dir / "sample_index.csv"
    stats_path = metrics_dir / "sample_statistics.csv"
    log_path = logs_dir / "sample_build_log.txt"

    sample_rows = [asdict(record) for record in records]
    _write_csv(index_path, sample_rows, list(SampleRecord.__dataclass_fields__))

    stat_rows: list[dict[str, str]] = []
    for key, value in stats.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            rendered = str(value)
        stat_rows.append({"metric": key, "value": rendered})
    _write_csv(stats_path, stat_rows, ["metric", "value"])

    log_lines = [
        "Sample index build report",
        "=========================",
        f"Total usable samples: {stats['total_samples']}",
        f"Train samples: {stats['train_samples']}",
        f"Test samples: {stats['test_samples']}",
        f"Available HoloLens mp4: {stats['available_hololens_mp4']}",
        f"Available fisheye avi: {stats['available_fisheye_avi']}",
        f"Labeled videos: {stats['labeled_videos']}",
        f"Mapped pairs: {stats['mapped_pairs']}",
        f"Split counts: {json.dumps(stats['split_counts'], ensure_ascii=False, sort_keys=True)}",
        f"User counts: {json.dumps(stats['user_counts'], ensure_ascii=False, sort_keys=True)}",
        f"Scene counts: {json.dumps(stats['scene_counts'], ensure_ascii=False, sort_keys=True)}",
        f"Intent counts: {json.dumps(stats['intent_counts'], ensure_ascii=False, sort_keys=True)}",
        "",
        "Data issues and notes:",
    ]
    for key in (
        "missing_mp4",
        "missing_avi",
        "missing_user",
        "missing_scene",
        "missing_mapping",
        "extra_mp4_files",
        "mapped_but_unlabeled_mp4",
        "labeled_but_unmapped_mp4",
        "unused_avi_files",
        "unused_labeled_mp4",
    ):
        log_lines.append(f"- {key}: {json.dumps(stats[key], ensure_ascii=False, sort_keys=True)}")
    log_lines.extend(
        [
            "",
            f"Sample index: {index_path}",
            f"Statistics: {stats_path}",
        ]
    )
    _write_log(log_path, log_lines)

    return {"index": index_path, "statistics": stats_path, "log": log_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the multimodal sample index.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config. Defaults to configs/default.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, stats = build_sample_index(args.config)
    output_paths = save_outputs(records, stats, args.config)

    print(f"Built {stats['total_samples']} usable samples.")
    print(f"Train: {stats['train_samples']} | Test: {stats['test_samples']}")
    print(f"Sample index: {output_paths['index']}")
    print(f"Statistics: {output_paths['statistics']}")
    print(f"Log: {output_paths['log']}")
    if stats["extra_mp4_files"] or stats["unused_avi_files"]:
        print("Warnings were recorded in the sample build log.")


if __name__ == "__main__":
    main()
