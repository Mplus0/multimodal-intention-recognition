"""Text raw-data adapter.

Source reference:
    ``src/modules/feature_extraction/ASR.py``
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.raw_extractors.common import (
    RawFeatureExtractionError,
    config_path,
    ensure_parent,
    optional_import,
    raw_path_for,
    require_dir,
    require_file,
)


GESTURE_TIMESTEPS = 10
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
            ffmpeg_params=["-ac", "1"],
            verbose=False,
            logger=None,
        )
    finally:
        clip.close()


def _parse_metadata_timestamps(metadata_path: Path, mp4_name: str) -> list[tuple[float, float]]:
    np = optional_import("numpy")
    payload = np.load(metadata_path, allow_pickle=True).item()
    approx_timestamps = payload["approx_timestamps"]
    name_parts = Path(mp4_name).name.split("_")
    video_start_dt = datetime.strptime(name_parts[1] + name_parts[2].split(".")[0], "%Y%m%d%H%M%S")

    ranges: list[tuple[float, float]] = []
    for timestamp_value in approx_timestamps:
        current_dt = datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00")).replace(tzinfo=None)
        mid_sec = (current_dt - video_start_dt).total_seconds()
        ranges.append((max(0.0, mid_sec - AUDIO_WINDOW_SEC), mid_sec + AUDIO_WINDOW_SEC))
    return ranges


def _whisper_model_name(config: dict[str, Any]) -> str:
    return str(config.get("local_models", {}).get("whisper_model", "small"))


def _transcribe_and_embed(
    wav_path: Path,
    ranges: list[tuple[float, float]],
    sentence_model_dir: Path,
    whisper_model_name: str,
):
    np = optional_import("numpy")
    librosa = optional_import("librosa")
    whisper = optional_import("whisper", "openai-whisper")
    sentence_transformers = optional_import("sentence_transformers", "sentence-transformers")
    pypinyin = optional_import("pypinyin")
    SentenceTransformer = sentence_transformers.SentenceTransformer

    audio, sample_rate = librosa.load(str(wav_path), sr=16000, mono=True)
    whisper_model = whisper.load_model(whisper_model_name)
    sentence_model = SentenceTransformer(str(sentence_model_dir))
    segment_meta: list[dict[str, Any]] = []
    embeddings = []

    for index, (start, end) in enumerate(ranges):
        segment = audio[int(start * sample_rate) : int(end * sample_rate)]
        text = ""
        if len(segment) >= sample_rate * 0.1:
            try:
                result = whisper_model.transcribe(segment, language="zh", fp16=False)
                text = result["text"].strip()
            except Exception:
                text = ""

        pinyin_text = ""
        if text:
            pinyin_list = pypinyin.pinyin(text, style=pypinyin.Style.TONE)
            pinyin_text = " ".join(item[0] for item in pinyin_list)

        segment_meta.append(
            {
                "id": index,
                "abs_timestamp": [float(start), float(end)],
                "text": text,
                "pinyin": pinyin_text,
            }
        )
        embeddings.append(sentence_model.encode(pinyin_text if pinyin_text else "", normalize_embeddings=True))

    embeddings_np = np.array(embeddings, dtype=np.float32)
    features = np.tile(embeddings_np[:, np.newaxis, :], (1, GESTURE_TIMESTEPS, 1))
    return features.astype(np.float32), segment_meta


def build_text_feature(
    sample: dict[str, Any],
    config: dict[str, Any],
    output_path: Path,
    gesture_metadata_path: Path,
) -> Path:
    """Build text embedding source features from HoloLens audio/video."""
    audio_path = raw_path_for(sample, "text")
    require_file(gesture_metadata_path, "gesture metadata for text alignment")
    sentence_model_dir = config_path(
        config,
        "local_models",
        "sentence_model",
        description="sentence model directory",
    )
    require_dir(sentence_model_dir, "sentence model directory")
    ensure_parent(output_path)
    wav_path = output_path.parent / f"temp_asr_{output_path.stem}.wav"
    try:
        _extract_audio_from_mp4(audio_path, wav_path)
        ranges = _parse_metadata_timestamps(gesture_metadata_path, audio_path.name)
        if not ranges:
            raise RawFeatureExtractionError(f"No text time ranges were parsed from: {gesture_metadata_path}")
        features, metadata = _transcribe_and_embed(
            wav_path,
            ranges,
            sentence_model_dir,
            _whisper_model_name(config),
        )
        if len(features) == 0:
            raise RawFeatureExtractionError(f"No text embeddings were extracted from: {audio_path}")
        np = optional_import("numpy")
        np.save(output_path, {"features": features, "metadata": metadata})
        with output_path.with_suffix(".json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)
        return output_path
    finally:
        if wav_path.exists():
            try:
                os.remove(wav_path)
            except OSError:
                pass
