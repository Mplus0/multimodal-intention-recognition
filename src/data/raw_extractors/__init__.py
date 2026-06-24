"""Raw-data extractor adapters for the Member A end-to-end pipeline.

The modules in this package intentionally do not modify the teacher-provided
reference scripts in ``src/modules``.  They provide a stable, config-driven
interface that ``src.data.raw_feature_builder`` can call when source ``.npy``
features are missing.
"""

from __future__ import annotations

from src.data.raw_extractors.audio_extractor import build_audio_feature
from src.data.raw_extractors.gesture_extractor import build_gesture_feature
from src.data.raw_extractors.imu_extractor import build_imu_feature
from src.data.raw_extractors.scene_extractor import build_scene_feature
from src.data.raw_extractors.text_extractor import build_text_feature

__all__ = [
    "build_audio_feature",
    "build_gesture_feature",
    "build_imu_feature",
    "build_scene_feature",
    "build_text_feature",
]
