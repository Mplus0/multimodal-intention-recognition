"""Audio raw-data adapter.

Source reference:
    ``src/modules/feature_extraction/mfcc.py``
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.raw_extractors.common import (
    RawFeatureExtractionError,
    ensure_parent,
    optional_import,
    raw_path_for,
    require_file,
)


AUDIO_WINDOW_SEC = 0.75


def _extract_audio_from_mp4(mp4_path: Path, wav_path: Path) -> None:
    moviepy = optional_import("moviepy", "moviepy")
    try:
        VideoFileClip = moviepy.editor.VideoFileClip
    except AttributeError:
        from moviepy.editor import VideoFileClip

    clip = VideoFileClip(str(mp4_path))
    try:
        if clip.audio is None:
            raise RawFeatureExtractionError(f"Video has no audio track: {mp4_path}")
        clip.audio.write_audiofile(
            str(wav_path),
            fps=16000,
            nbytes=2,
            codec="pcm_s16le",
            verbose=False,
            logger=None,
        )
    finally:
        clip.close()


def _video_start_dt(video_name: str) -> datetime:
    match_str = video_name.replace("interaction_", "").split(".")[0]
    return datetime.strptime(match_str, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)


def _extract_mfcc_39d(wav_path: Path, metadata_path: Path, video_start_dt: datetime) -> list[dict[str, Any]]:
    np = optional_import("numpy")
    librosa = optional_import("librosa")
    payload = np.load(metadata_path, allow_pickle=True).item()
    approx_ts_list = payload["approx_timestamps"]
    audio, sample_rate = librosa.load(str(wav_path), sr=16000, mono=True)
    results: list[dict[str, Any]] = []

    for index, timestamp_value in enumerate(approx_ts_list):
        mid_dt = datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00"))
        mid_sec = (mid_dt - video_start_dt).total_seconds()
        start = max(0.0, mid_sec - AUDIO_WINDOW_SEC)
        end = mid_sec + AUDIO_WINDOW_SEC
        segment = audio[int(start * sample_rate) : int(end * sample_rate)]
        if len(segment) < sample_rate * 0.1:
            continue

        mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=13)
        steps = mfcc.shape[1]
        if steps < 3:
            combined = np.vstack([mfcc, np.zeros_like(mfcc), np.zeros_like(mfcc)])
        else:
            width = min(9, steps if steps % 2 == 1 else steps - 1)
            combined = np.vstack(
                [
                    mfcc,
                    librosa.feature.delta(mfcc, width=width),
                    librosa.feature.delta(mfcc, order=2, width=width),
                ]
            )
        results.append({"id": index, "timestamp": [float(start), float(end)], "feature": combined.T})
    return results


def build_audio_feature(
    sample: dict[str, Any],
    config: dict[str, Any],
    output_path: Path,
    gesture_metadata_path: Path,
) -> Path:
    """Build audio MFCC source features from a HoloLens video."""
    del config
    audio_path = raw_path_for(sample, "audio")
    require_file(gesture_metadata_path, "gesture metadata for audio alignment")
    ensure_parent(output_path)
    wav_path = output_path.with_suffix(".wav")
    try:
        _extract_audio_from_mp4(audio_path, wav_path)
        results = _extract_mfcc_39d(wav_path, gesture_metadata_path, _video_start_dt(audio_path.name))
        if not results:
            raise RawFeatureExtractionError(f"No valid audio MFCC segments were extracted from: {audio_path}")
        np = optional_import("numpy")
        np.save(output_path, results)
        json_path = output_path.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(
                [
                    {
                        "id": item["id"],
                        "timestamp": item["timestamp"],
                        "feature_shape": list(item["feature"].shape),
                    }
                    for item in results
                ],
                file,
                indent=2,
                ensure_ascii=False,
            )
        return output_path
    finally:
        if wav_path.exists():
            try:
                os.remove(wav_path)
            except OSError:
                pass
