# Multimodal User Intention Recognition

## 1. Project Overview

This repository is created for the final project of the Machine Learning course.

The project focuses on **multimodal user intention recognition** in augmented reality interaction scenarios. The goal is to refactor and improve the provided multimodal recognition model so that it can recognize user intentions from raw multimodal interaction data.

The course project requires the model to handle:

- Modal noise
- Missing modalities
- Cross-user testing
- End-to-end input from raw data to intention labels

The current repository already contains the reorganized raw dataset, baseline source code, feature extraction scripts, original code documentation, and report/result folders for later experiments.

---

## 2. Project Background

AR glasses and wearable interaction systems usually receive information from multiple modalities, including scene video, gesture video, audio, speech/text, and IMU signals. A single modality can be noisy or unavailable in real use, so this project studies robust multimodal fusion for user intention recognition.

The original teacher-provided code first extracts features from the dataset and then trains a multimodal fusion model on saved feature files. This project will refactor that workflow into a clearer end-to-end pipeline and evaluate the model under clean, noisy, and missing-modality settings.

---

## 3. Project Objectives

1. Use user A and user B as the training set.
2. Use user C as the testing set.
3. Refactor the original feature-based workflow into an end-to-end training and testing workflow.
4. Integrate preprocessing and feature extraction into the training/testing pipeline.
5. Build modal noise baselines at 20%, 40%, and 60% noise levels.
6. Build missing-modality baselines by dropping one or two modalities.
7. Improve the model with new modules or loss terms.
8. Save metrics, logs, figures, screenshots, and references for the final report and defense.

---

## 4. Current Repository Structure

The current repository is organized as follows:

```text
multimodal-intention-recognition/
├── README.md
├── README_CHINESE.md
├── requirements.txt
├── .gitignore
├── 课程项目2026.pdf
│
├── data/
│   ├── raw/
│   │   ├── imu.csv
│   │   ├── user_A/
│   │   │   ├── HoloLens/
│   │   │   └── fisheye/
│   │   ├── user_B/
│   │   │   ├── HoloLens/
│   │   │   └── fisheye/
│   │   ├── user_C/
│   │   │   ├── HoloLens/
│   │   │   └── fisheye/
│   │   └── models/
│   │       ├── all-MiniLM-L6-v2/
│   │       └── clip_teacher_model/
│   └── processed/
│
├── src/
│   ├── models/
│   │   └── baseline_real_scene.py
│   └── modules/
│       ├── real_scene_utils.py
│       └── feature_extraction/
│           ├── ASR.py
│           ├── get_timestamp.py
│           ├── imu.py
│           ├── mfcc.py
│           └── strong_gesture2.0.py
│
├── experiments/
│   └── train_and_test.py
│
├── docs/
│   └── original_code_readme.md
│
├── results/
│   ├── metrics/
│   ├── logs/
│   └── predictions/
│
├── figures/
│   ├── model_structure/
│   ├── curves/
│   └── result_charts/
│
└── report/
    ├── references.md
    └── screenshots/
```

---

## 5. File and Folder Explanation

### `data/raw/`

This folder stores the reorganized raw dataset and local model resources.

- `imu.csv`: the original IMU signal file. It is used as the IMU modality and is aligned with interaction videos by timestamp.
- `user_A/`: raw data for user A. This user belongs to the training set.
- `user_B/`: raw data for user B. This user belongs to the training set.
- `user_C/`: raw data for user C. This user belongs to the testing set.
- `HoloLens/`: HoloLens interaction videos in `.mp4` format. These videos are used mainly for audio and speech-related processing.
- `fisheye/`: fisheye camera videos in `.avi` format. These videos are used mainly for scene and gesture modalities.
- `models/all-MiniLM-L6-v2/`: local sentence embedding model used by text or ASR-related feature extraction.
- `models/clip_teacher_model/`: local vision-language/visual backbone resources from the original dataset package.

Some directories may contain `.DS_Store` or `._*` files created by macOS. These are system metadata files and should not be treated as valid training samples.

### `data/processed/`

This folder is reserved for generated intermediate data, cached features, aligned samples, and other preprocessing outputs. It is currently kept as an output directory and should not be used as the primary source of raw data.

### `src/models/`

This folder stores model definitions.

- `baseline_real_scene.py`: the teacher-provided baseline model code after relocation. It contains the Perceiver-IO based multimodal fusion baseline, intent labels, train/test video lists, modality definitions, training logic, and evaluation utilities. The code still needs further refactoring to become a clean end-to-end implementation.

### `src/modules/`

This folder stores reusable modules used by models and experiments.

- `real_scene_utils.py`: utilities for mapping fisheye `.avi` videos to HoloLens `.mp4` videos, reading real scene frames, extracting scene features with a local ViT model, and caching scene features.
- `feature_extraction/`: original feature extraction scripts moved from the teacher-provided code package.

### `src/modules/feature_extraction/`

This folder contains the modality preprocessing and feature extraction scripts.

- `get_timestamp.py`: extracts or aligns timestamps for interaction data.
- `ASR.py`: performs speech recognition or speech-text feature processing.
- `imu.py`: processes IMU signals and extracts IMU features.
- `mfcc.py`: extracts audio MFCC features.
- `strong_gesture2.0.py`: extracts gesture-related features from video data.

These scripts are the main references for integrating feature extraction into the future end-to-end training and testing pipeline.

### `experiments/`

This folder stores experiment entry scripts.

- `train_and_test.py`: relocated original training/testing script. It currently serves as the main reference implementation. Later work should split it into clearer training and testing entry points, such as `train.py`, `test.py`, and separate experiment scripts for clean, noisy, and missing-modality settings.

### `docs/`

This folder stores project documentation.

- `original_code_readme.md`: the teacher-provided code README. It explains the original code structure, training/testing data split, feature extraction scripts, and submission expectations.

### `results/`

This folder stores experiment outputs.

- `metrics/`: accuracy, F1 score, classification reports, confusion matrices, and comparison tables.
- `logs/`: training and testing logs.
- `predictions/`: model predictions on validation or test samples.

### `figures/`

This folder stores figures for the project report and defense slides.

- `model_structure/`: model architecture diagrams.
- `curves/`: training loss, validation accuracy, and other curves.
- `result_charts/`: charts comparing clean, noisy, missing-modality, baseline, and improved results.

### `report/`

This folder stores report-related materials.

- `references.md`: reference list or citation notes for the final report.
- `screenshots/`: screenshots proving that the end-to-end model runs successfully.

### `课程项目2026.pdf`

This is the official course project requirement document. It defines the project topic, implementation requirements, report requirements, and scoring criteria.

---

## 6. Development Plan

### Stage 1: Dataset and Code Organization

- Keep the raw multimodal data under `data/raw/`.
- Keep user A and user B for training.
- Keep user C for testing.
- Keep the teacher-provided baseline and feature extraction scripts under `src/` and `experiments/`.

### Stage 2: Baseline Refactoring

- Refactor `experiments/train_and_test.py`.
- Move reusable model components into `src/models/`.
- Move preprocessing and feature extraction calls into reusable modules.
- Replace hard-coded original local paths with project-relative paths.

### Stage 3: End-to-End Training and Testing

- Build a training script that starts from raw multimodal data.
- Build a testing script that outputs intention classification results.
- Save model checkpoints, scalers, label encoders, metrics, and logs.

### Stage 4: Modal Noise Experiments

- Add 20%, 40%, and 60% noise to each single modality.
- Train and evaluate the model under each noise setting.
- Save comparison metrics and result charts.

### Stage 5: Missing Modality Experiments

- Drop each single modality and each pair of modalities.
- Train and evaluate the model under missing-modality settings.
- Analyze robustness under incomplete input.

### Stage 6: Model Improvement

- Add improved fusion modules, modality reliability estimation, auxiliary losses, or other robustness-oriented components.
- Compare improved results with baseline results.

### Stage 7: Report and Defense

- Save screenshots of successful end-to-end runs.
- Organize result tables and charts.
- Write the project report with citations and team contribution percentages.
- Prepare the group defense slides.

---

## 7. Team Collaboration

Suggested task division:

- Dataset organization and preprocessing
- Baseline refactoring
- Feature extraction integration
- Noise and missing-modality experiments
- Model improvement
- Result analysis and visualization
- Report writing and defense preparation

The final contribution percentage of each member should be recorded in the project report.

---

## 8. Current Status

Current repository status:

- Project topic and requirement document are available.
- Raw data has been reorganized under `data/raw/`.
- User folders `user_A`, `user_B`, and `user_C` exist with `HoloLens/` and `fisheye/` subfolders.
- IMU data is available at `data/raw/imu.csv`.
- Local model resources are available under `data/raw/models/`.
- Baseline model code is available at `src/models/baseline_real_scene.py`.
- Real scene utilities are available at `src/modules/real_scene_utils.py`.
- Feature extraction scripts are available under `src/modules/feature_extraction/`.
- The original training/testing script is available at `experiments/train_and_test.py`.
- Result, figure, and report folders are prepared for later outputs.

---

## 9. Notes

- The current source code is still close to the teacher-provided version and may contain hard-coded paths from the original environment. These paths should be replaced with project-relative paths during refactoring.
- Large raw data and generated model checkpoints should not be committed to Git unless the course submission explicitly requires them.
- macOS metadata files such as `.DS_Store` and `._*` should be ignored during data loading and experiment execution.
