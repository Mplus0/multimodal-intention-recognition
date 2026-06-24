"""Callable raw-data feature extraction adapters for Member A.

The teacher scripts in this folder are kept as references. This module exposes
path-parameterized functions that can be called from train/test pipelines.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.fftpack import dct
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel


TARGET_SAMPLE_RATE = 16000
TARGET_STEPS = 10


class RawFeatureBuildError(RuntimeError):
    """Raised when a raw-data feature cannot be built honestly."""


def _video_start_from_name(video_name: str) -> datetime | None:
    parts = Path(video_name).stem.split("_")
    if len(parts) < 3:
        return None
    try:
        return datetime.strptime(parts[1] + parts[2][:6], "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _even_indices(length: int, count: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(count, dtype=int)
    return np.linspace(0, length - 1, count, dtype=int)


def _pad_or_trim_2d(array: np.ndarray, steps: int, dim: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got {array.shape}.")
    if array.shape[1] < dim:
        pad = np.zeros((array.shape[0], dim - array.shape[1]), dtype=np.float32)
        array = np.concatenate([array, pad], axis=1)
    if array.shape[1] > dim:
        array = array[:, :dim]
    if array.shape[0] > steps:
        array = array[_even_indices(array.shape[0], steps)]
    if array.shape[0] < steps:
        pad = np.zeros((steps - array.shape[0], dim), dtype=np.float32)
        array = np.concatenate([array, pad], axis=0)
    return array.astype(np.float32)


def _read_video_frames(video_path: Path, steps: int = TARGET_STEPS) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RawFeatureBuildError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[Image.Image] = []
    for index in _even_indices(frame_count, steps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb).convert("RGB"))
    cap.release()
    if not frames:
        raise RawFeatureBuildError(f"No frames decoded from video: {video_path}")
    while len(frames) < steps:
        frames.append(frames[-1])
    return frames[:steps]


class ClipFeatureExtractor:
    """Lazy local CLIP image encoder."""

    def __init__(self, model_dir: Path, device: str | None = None):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise RawFeatureBuildError(f"CLIP model directory missing: {self.model_dir}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = CLIPImageProcessor.from_pretrained(str(self.model_dir), local_files_only=True)
        self.model = CLIPVisionModel.from_pretrained(str(self.model_dir), local_files_only=True).to(self.device).eval()

    @torch.no_grad()
    def encode_frames(self, frames: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)


_CLIP_EXTRACTOR: ClipFeatureExtractor | None = None
_TEXT_MODEL: tuple[Any, Any, torch.device] | None = None
_WHISPER_MODEL: Any | None = None


def _get_clip(config: dict[str, Any]) -> ClipFeatureExtractor:
    global _CLIP_EXTRACTOR
    if _CLIP_EXTRACTOR is None:
        model_path = Path(config["local_models"].get("clip_teacher_model", "data/raw/models/clip_teacher_model"))
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        _CLIP_EXTRACTOR = ClipFeatureExtractor(model_path)
    return _CLIP_EXTRACTOR


def build_visual_sequence_feature(
    fisheye_video: Path,
    output_path: Path,
    config: dict[str, Any],
    *,
    sample_id: str,
    intent_label: int | None,
    scene_label: int | None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = _read_video_frames(fisheye_video, TARGET_STEPS)
    features = _get_clip(config).encode_frames(frames)
    features = _pad_or_trim_2d(features, TARGET_STEPS, 768)
    payload = {
        "features": features[np.newaxis, :, :].astype(np.float32),
        "labels": np.array([intent_label if intent_label is not None else -1], dtype=np.int64),
        "scene_labels": np.array([scene_label if scene_label is not None else -1], dtype=np.int64),
        "approx_timestamps": np.array([sample_id]),
    }
    np.save(output_path, payload)
    return output_path


def build_scene_feature(fisheye_video: Path, output_path: Path, config: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = _read_video_frames(fisheye_video, 5)
    features = _get_clip(config).encode_frames(frames)
    scene = features.mean(axis=0).astype(np.float32)
    np.save(output_path, {"features": scene[np.newaxis, :]})
    return output_path


def build_imu_feature(imu_csv: Path, output_path: Path, config: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(imu_csv)
    numeric = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    if numeric.size == 0:
        raise RawFeatureBuildError(f"No numeric IMU columns found: {imu_csv}")
    sampled = numeric[_even_indices(len(numeric), TARGET_STEPS)]
    sampled = _pad_or_trim_2d(sampled, TARGET_STEPS, 12)
    np.save(output_path, {"features": sampled[np.newaxis, :, :]})
    return output_path


def _decode_audio(mp4_path: Path) -> np.ndarray:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-nostdin",
        "-i",
        str(mp4_path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-hide_banner",
        "-loglevel",
        "error",
        "pipe:1",
    ]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RawFeatureBuildError(proc.stderr.decode("utf-8", errors="ignore") or f"ffmpeg failed: {mp4_path}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RawFeatureBuildError(f"No audio decoded from {mp4_path}")
    return audio


def _mel_filterbank(num_filters: int, n_fft: int, sample_rate: int) -> np.ndarray:
    def hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    low_mel = hz_to_mel(np.array([0.0]))[0]
    high_mel = hz_to_mel(np.array([sample_rate / 2]))[0]
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    fbank = np.zeros((num_filters, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, num_filters + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center > left:
            fbank[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fbank[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fbank


def _mfcc_39(audio: np.ndarray) -> np.ndarray:
    frame_len = int(0.025 * TARGET_SAMPLE_RATE)
    hop = int(0.010 * TARGET_SAMPLE_RATE)
    if audio.size < frame_len:
        audio = np.pad(audio, (0, frame_len - audio.size))
    frame_count = 1 + max(0, (audio.size - frame_len) // hop)
    frames = np.stack([audio[i * hop : i * hop + frame_len] for i in range(frame_count)])
    frames *= np.hamming(frame_len).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames, n=512)) ** 2
    mel = np.maximum(spectrum @ _mel_filterbank(26, 512, TARGET_SAMPLE_RATE).T, 1e-10)
    coeff = dct(np.log(mel), type=2, axis=1, norm="ortho")[:, :13]
    delta = np.gradient(coeff, axis=0)
    delta2 = np.gradient(delta, axis=0)
    return np.concatenate([coeff, delta, delta2], axis=1).astype(np.float32)


def build_audio_feature(hololens_video: Path, output_path: Path, config: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = _decode_audio(hololens_video)
    mfcc = _mfcc_39(audio)
    mfcc = _pad_or_trim_2d(mfcc, TARGET_STEPS, 39)
    np.save(output_path, {"features": mfcc[np.newaxis, :, :]})
    return output_path


def _get_text_encoder(config: dict[str, Any]):
    global _TEXT_MODEL
    if _TEXT_MODEL is None:
        model_path = Path(config["local_models"].get("sentence_model", "data/raw/models/all-MiniLM-L6-v2"))
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        if not model_path.exists():
            raise RawFeatureBuildError(f"Sentence model directory missing: {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(device).eval()
        _TEXT_MODEL = (tokenizer, model, device)
    return _TEXT_MODEL


def _embed_text(text: str, config: dict[str, Any]) -> np.ndarray:
    tokenizer, model, device = _get_text_encoder(config)
    inputs = tokenizer([text or ""], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled[0].detach().cpu().numpy().astype(np.float32)


def _get_whisper(config: dict[str, Any]):
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            import whisper
        except ImportError as error:
            raise RawFeatureBuildError("openai-whisper is required for text feature extraction.") from error
        cache_root = Path(config.get("cache", {}).get("root", "data/processed/cache"))
        if not cache_root.is_absolute():
            cache_root = Path.cwd() / cache_root
        model_name = config.get("raw_feature_builder", {}).get("whisper_model", "small")
        _WHISPER_MODEL = whisper.load_model(model_name, download_root=str(cache_root / "whisper"))
    return _WHISPER_MODEL


def build_text_feature(hololens_video: Path, output_path: Path, config: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = _decode_audio(hololens_video)
    model = _get_whisper(config)
    result = model.transcribe(audio, language="zh", fp16=False)
    text = str(result.get("text", "")).strip()
    embedding = _embed_text(text, config)
    embedding = _pad_or_trim_2d(embedding[np.newaxis, :], 1, 384)[0]
    features = np.tile(embedding[np.newaxis, np.newaxis, :], (1, TARGET_STEPS, 1)).astype(np.float32)
    np.save(output_path, {"features": features, "metadata": [{"text": text}]})
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps({"text": text}, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
