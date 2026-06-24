"""Gesture raw-data adapter.

Source references:
    ``src/modules/feature_extraction/get_timestamp.py``
    ``src/modules/feature_extraction/strong_gesture2.0.py``
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.data.raw_extractors.common import (
    RawFeatureExtractionError,
    config_path,
    ensure_parent,
    optional_import,
    raw_path_for,
)


SEQ_LEN = 10
HALF_WINDOW_MS = 750


def _intent_label(sample: dict[str, Any]) -> int:
    label = sample.get("intent_label")
    if label is None:
        raise RawFeatureExtractionError(
            f"Sample {sample.get('sample_id', '<unknown>')} has no intent_label for timestamp labels."
        )
    return int(label)


def _read_wave(path: Path):
    np = optional_import("numpy")
    with contextlib.closing(wave.open(str(path), "rb")) as wave_file:
        sample_rate = wave_file.getframerate()
        audio = np.frombuffer(wave_file.readframes(wave_file.getnframes()), dtype=np.int16)
        if wave_file.getnchannels() == 2:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
        return audio, sample_rate


def _vad_segments(audio_path: Path) -> list[tuple[float, float]]:
    np = optional_import("numpy")
    webrtcvad = optional_import("webrtcvad")
    audio, sample_rate = _read_wave(audio_path)
    vad = webrtcvad.Vad(2)

    audio_float = audio.astype(np.float32)
    q70_rms = np.percentile(np.abs(audio_float), 70)
    frame_ms = 30
    frame_len = int(sample_rate * (frame_ms / 1000.0))
    frames = [audio[i : i + frame_len] for i in range(0, len(audio) - frame_len, frame_len)]

    segments: list[tuple[float, float]] = []
    start = None
    for index, frame in enumerate(frames):
        is_speech = vad.is_speech(frame.tobytes(), sample_rate)
        frame_rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
        valid_speech = is_speech and frame_rms > q70_rms * 0.6
        timestamp = index * (frame_ms / 1000.0)
        if valid_speech and start is None:
            start = timestamp
        elif not valid_speech and start is not None:
            duration = timestamp - start
            if 0.3 <= duration <= 5.0:
                segments.append((round(max(0.0, start - 0.4), 3), round(timestamp + 0.4, 3)))
            start = None
    return segments


def _mp4_start_utc(video_name: str) -> datetime:
    match = re.search(r"interaction_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", video_name)
    if not match:
        raise RawFeatureExtractionError(f"Cannot parse HoloLens video start time from: {video_name}")
    parts = [int(part) for part in match.groups()]
    return datetime(*parts, tzinfo=timezone.utc)


def _iso_timestamps(video_name: str, segments: list[tuple[float, float]]) -> list[str]:
    start_dt = _mp4_start_utc(video_name)
    timestamps: list[str] = []
    for start, end in segments:
        mid_dt = start_dt + timedelta(seconds=(start + end) / 2)
        timestamps.append(mid_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")
    return timestamps


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
            codec="pcm_s16le",
            verbose=False,
            logger=None,
        )
    finally:
        clip.close()


def _avi_start_utc(avi_path: Path) -> datetime:
    parts = avi_path.name.split("_")
    if len(parts) < 3:
        raise RawFeatureExtractionError(f"Cannot parse fisheye video start time from: {avi_path.name}")
    time_part = f"{parts[1]}_{parts[2].split('.')[0]}"
    return datetime.strptime(time_part, "%Y%m%d_%H%M%S%f") - timedelta(hours=8)


def _timestamp_to_offset_ms(avi_path: Path, timestamp_value: str) -> float | None:
    cv2 = optional_import("cv2", "opencv-python-headless")
    utc_target = datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00")).replace(tzinfo=None)
    offset_ms = (utc_target - _avi_start_utc(avi_path)).total_seconds() * 1000.0
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        raise RawFeatureExtractionError(f"Cannot open fisheye video: {avi_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0 or frame_count <= 0:
        raise RawFeatureExtractionError(f"Invalid fisheye video metadata: {avi_path}")
    duration_ms = frame_count / fps * 1000.0
    return offset_ms if 0 <= offset_ms <= duration_ms else None


class _GestureBackbone:
    def __init__(self, model_path: Path):
        torch = optional_import("torch")
        transformers = optional_import("transformers")
        mediapipe = optional_import("mediapipe")

        self.np = optional_import("numpy")
        self.cv2 = optional_import("cv2", "opencv-python-headless")
        self.Image = __import__("PIL.Image", fromlist=["Image"]).Image
        self.torch = torch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hands = mediapipe.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.3,
        )
        self.processor = transformers.CLIPImageProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        self.model = transformers.CLIPVisionModel.from_pretrained(
            str(model_path),
            local_files_only=True,
        ).to(self.device).eval()

    def crop_hand(self, image):
        image_np = self.np.array(image)
        height, width = image_np.shape[:2]
        image_rgb = self.cv2.cvtColor(image_np, self.cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            xs = [int(lm.x * width) for hand in results.multi_hand_landmarks for lm in hand.landmark]
            ys = [int(lm.y * height) for hand in results.multi_hand_landmarks for lm in hand.landmark]
            x1, y1, x2, y2 = max(0, min(xs)), max(0, min(ys)), min(width, max(xs)), min(height, max(ys))
            crop_w, crop_h = x2 - x1, y2 - y1
            pad = 0.4
            x1, y1 = max(0, int(x1 - crop_w * pad)), max(0, int(y1 - crop_h * pad))
            x2, y2 = min(width, int(x2 + crop_w * pad)), min(height, int(y2 + crop_h * pad))
            return image.crop((x1, y1, x2, y2)).resize((224, 224), self.Image.LANCZOS)
        return image.resize((224, 224), self.Image.LANCZOS)

    def extract_sequence(self, video_path: Path, center_ms: float):
        start_ms = center_ms - HALF_WINDOW_MS
        end_ms = center_ms + HALF_WINDOW_MS
        cap = self.cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RawFeatureExtractionError(f"Cannot open fisheye video: {video_path}")
        fps = cap.get(self.cv2.CAP_PROP_FPS)
        frame_count = cap.get(self.cv2.CAP_PROP_FRAME_COUNT)
        total_ms = frame_count / fps * 1000.0 if fps > 0 else 0.0
        if start_ms < 0 or end_ms > total_ms:
            cap.release()
            return None

        features = []
        for msec in self.np.linspace(start_ms, end_ms, SEQ_LEN):
            cap.set(self.cv2.CAP_PROP_POS_MSEC, float(msec))
            ok, frame = cap.read()
            if not ok:
                break
            image = self.Image.fromarray(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))
            hand_image = self.crop_hand(image)
            inputs = self.processor(images=hand_image.convert("RGB"), return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                outputs = self.model(**inputs)
            features.append(outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy())
        cap.release()
        return self.np.array(features, dtype=self.np.float32) if len(features) == SEQ_LEN else None


def _build_timestamps(sample: dict[str, Any], tmp_wav_path: Path) -> list[str]:
    np = optional_import("numpy")
    audio_path = raw_path_for(sample, "audio")
    _extract_audio_from_mp4(audio_path, tmp_wav_path)
    segments = _vad_segments(tmp_wav_path)
    timestamps = _iso_timestamps(audio_path.name, segments)
    if not timestamps:
        raise RawFeatureExtractionError(f"No VAD timestamps were detected for audio source: {audio_path}")
    return timestamps


def build_gesture_feature(
    sample: dict[str, Any],
    config: dict[str, Any],
    output_path: Path,
    metadata_path: Path | None = None,
) -> Path:
    """Build gesture source features from raw HoloLens/fisheye videos."""
    np = optional_import("numpy")
    visual_path = raw_path_for(sample, "gesture")
    ensure_parent(output_path)
    metadata_path = metadata_path or output_path.with_name(
        f"metadata_strong_gesture_{output_path.stem.replace('strong_gesture_features_', '')}.npy"
    )
    ensure_parent(metadata_path)
    tmp_wav_path = output_path.parent / f"temp_vad_{output_path.stem}.wav"
    try:
        timestamps = _build_timestamps(sample, tmp_wav_path)
        labels = np.full(len(timestamps), _intent_label(sample), dtype=np.int64)
        model_path = config_path(config, "local_models", "clip_teacher_model", description="CLIP teacher model")
        if not model_path.exists():
            raise RawFeatureExtractionError(f"CLIP teacher model directory does not exist: {model_path}")
        backbone = _GestureBackbone(model_path)

        valid_features, valid_labels, valid_timestamps = [], [], []
        debug_log: dict[str, Any] = {}
        for index, timestamp in enumerate(timestamps):
            offset_ms = _timestamp_to_offset_ms(visual_path, timestamp)
            if offset_ms is None:
                continue
            sequence = backbone.extract_sequence(visual_path, offset_ms)
            if sequence is None:
                continue
            valid_features.append(sequence)
            valid_labels.append(labels[index])
            valid_timestamps.append(timestamp)
            debug_log[str(len(valid_features) - 1)] = {
                "original_idx": index,
                "utc_time": timestamp,
                "msec_center": round(float(offset_ms), 2),
            }

        if not valid_features:
            raise RawFeatureExtractionError(
                f"No valid gesture sequences were extracted from fisheye video: {visual_path}"
            )

        final_payload = {
            "features": np.array(valid_features, dtype=np.float32),
            "labels": np.array(valid_labels, dtype=np.int64),
            "video_names": np.array([sample.get("video_name", raw_path_for(sample, "audio").name)] * len(valid_labels)),
            "approx_timestamps": np.array(valid_timestamps),
        }
        np.save(output_path, final_payload)
        np.save(metadata_path, {key: value for key, value in final_payload.items() if key != "features"})
        debug_path = output_path.with_name(f"debug_strong_gesture_{output_path.stem.replace('strong_gesture_features_', '')}.json")
        with debug_path.open("w", encoding="utf-8") as file:
            json.dump(debug_log, file, indent=2, ensure_ascii=False)
        return output_path
    finally:
        if tmp_wav_path.exists():
            try:
                os.remove(tmp_wav_path)
            except OSError:
                pass
