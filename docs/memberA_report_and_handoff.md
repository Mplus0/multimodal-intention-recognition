# Member A Report Materials and Handoff Notes

Date: 2026-06-23  
Project: `multimodal-intention-recognition`  
Owner: Member A  

## 1. Member A Responsibility Check

According to `docs/PROJECT_WORKFLOW_AND_TEAM_DIVISION.md`, Member A is responsible for end-to-end data flow and data engineering.

Required tasks:

1. Check repository structure and data paths.
2. Handle user A/B/C data reading under `data/raw/`.
3. Integrate multimodal feature extraction interfaces.
4. Write or maintain:
   - `src/data/dataset.py`
   - `src/data/build_samples.py`
   - related feature extraction interfaces
   - train/test entry scripts
5. Ensure the end-to-end flow can run.
6. Save running screenshots and basic logs.

Current completion status: completed as a runnable baseline. The current pipeline is suitable for report evidence and for Member B to continue model/experiment improvement.

## 2. Files Delivered by Member A

Core data pipeline:

```text
src/data/build_samples.py
src/data/dataset.py
src/data/features.py
src/data/check_dataset.py
src/data/build_features.py
```

End-to-end baseline entry:

```text
experiments/train_formal_dataset.py
```

Utility files involved:

```text
src/utils/paths.py
src/utils/logger.py
src/utils/seed.py
configs/default.yaml
```

Generated result files:

```text
data/processed/sample_index.csv
results/metrics/sample_statistics.csv
results/logs/sample_build_log.txt
results/logs/dataset_check.log
results/logs/feature_build.log
results/logs/formal_dataset_train.log
results/metrics/formal_dataset_metrics.json
results/predictions/formal_dataset_predictions.csv
models/formal_dataset_baseline.pt
```

## 3. Confirmed Data Results

Sample index command:

```bash
source /share/home/tm1078571822880000/a893873600/wyr_ml_course_project_work/software/memberA_activate.sh
python src/data/build_samples.py --config configs/default.yaml
```

Confirmed output:

```text
Built 39 usable samples.
Train: 26 | Test: 13
```

Important statistics from `results/metrics/sample_statistics.csv`:

```text
total_samples: 39
train_samples: 26
test_samples: 13
available_hololens_mp4: 42
available_fisheye_avi: 42
labeled_videos: 39
mapped_pairs: 39
user_counts: {"user_A": 13, "user_B": 13, "user_C": 13}
scene_counts: {"museum": 20, "office": 19}
intent_counts: {"brush": 6, "cancel": 6, "magnify": 6, "menu": 6, "narrow": 7, "select": 8}
missing_mp4: []
missing_avi: []
missing_user: []
missing_scene: []
missing_mapping: []
```

Known data notes:

```text
extra_mp4_files:
- interaction_20260131_071918.mp4
- interaction_20260131_092133.mp4
- interaction_20260227_124133.mp4

unused_avi_files:
- Video_20260131_151916937.avi
- Video_20260131_172143627.avi
- Video_20260227_204121788.avi
```

These files are not included because they currently lack complete label/mapping information.

## 4. Confirmed Dataset and Feature Inputs

Dataset check command:

```bash
python src/data/check_dataset.py --config configs/default.yaml --batch-size 2 --load-features
```

Confirmed feature shapes:

```text
hololens_frames: (batch, 8, 3, 112, 112)
fisheye_frames:  (batch, 8, 3, 112, 112)
imu:             (batch, 10, 12)
scene:           (batch, 2)
intent_label:    (batch,)
```

Feature cache command:

```bash
python src/data/build_features.py --config configs/default.yaml
```

Confirmed cache result:

```text
Total samples: 39
Cache files: 78 .pt files
Reason: 39 samples x 2 video streams
Cache directory:
data/processed/cache/feature_cache/video_frames/formal_v1/frames8_size112
```

Important limitation:

The current raw `imu.csv` timestamps do not align directly with the 2026 video filenames. Therefore the current IMU feature is marked as `global_unaligned_sequence`. It is enough for the end-to-end baseline, but the high-score version should replace it with per-sample aligned IMU windows or recovered processed IMU features.

## 5. Confirmed End-to-End Baseline

Run command:

```bash
python experiments/train_formal_dataset.py --config configs/default.yaml --epochs 2 --batch-size 4
```

Confirmed output files:

```text
results/metrics/formal_dataset_metrics.json
results/predictions/formal_dataset_predictions.csv
models/formal_dataset_baseline.pt
results/logs/formal_dataset_train.log
```

Smoke-run metric:

```text
final_test_acc: 0.15384615384615385
final_test_loss: 1.792035487981943
epochs: 2
```

Interpretation for report:

This result proves the end-to-end flow can run from sample index and raw data paths to feature tensors, model training, test prediction, metrics output, and model checkpoint saving. The accuracy is only a smoke-test baseline and should not be presented as the final model performance.

## 6. Screenshots Needed for Report

Recommended screenshots:

1. Environment/GPU screenshot
   - Command:
     ```bash
     source /share/home/tm1078571822880000/a893873600/wyr_ml_course_project_work/software/memberA_activate.sh
     python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
     ```
   - Expected key result:
     ```text
     2.12.1+cu126
     True
     8
     ```

2. Sample index build screenshot
   - Command:
     ```bash
     python src/data/build_samples.py --config configs/default.yaml
     ```
   - Must show:
     ```text
     Built 39 usable samples.
     Train: 26 | Test: 13
     ```

3. Sample statistics screenshot
   - Open or print:
     ```text
     results/metrics/sample_statistics.csv
     ```
   - Must show user split, scene counts, intent counts, and no missing mapped files.

4. Dataset feature check screenshot
   - Command:
     ```bash
     python src/data/check_dataset.py --config configs/default.yaml --batch-size 2 --load-features
     ```
   - Must show feature shapes:
     ```text
     hololens=(2, 8, 3, 112, 112)
     fisheye=(2, 8, 3, 112, 112)
     imu=(2, 10, 12)
     scene=(2, 2)
     ```

5. Feature cache screenshot
   - Command:
     ```bash
     python src/data/build_features.py --config configs/default.yaml
     ```
   - Must show:
     ```text
     Total samples: 39
     Feature shapes per sample: {'hololens_frames': (8, 3, 112, 112), ...}
     ```

6. End-to-end training screenshot
   - Command:
     ```bash
     python experiments/train_formal_dataset.py --config configs/default.yaml --epochs 2 --batch-size 4
     ```
   - Must show output paths:
     ```text
     Metrics: results/metrics/formal_dataset_metrics.json
     Predictions: results/predictions/formal_dataset_predictions.csv
     Model: models/formal_dataset_baseline.pt
     ```

7. VS Code file tree screenshot
   - Show these files in Explorer:
     ```text
     src/data/build_samples.py
     src/data/dataset.py
     src/data/features.py
     src/data/check_dataset.py
     src/data/build_features.py
     experiments/train_formal_dataset.py
     results/metrics/
     results/logs/
     results/predictions/
     ```

## 7. Report Text Draft for Member A Contribution

Member A was responsible for data flow construction and end-to-end refactoring. The work started from checking the remote project environment, CUDA/PyTorch availability, raw data paths, and local model/data resources. A structured sample index was built from the original HoloLens videos, fisheye videos, intent labels, scene labels, user split, and IMU path. The final index contains 39 usable samples, including 26 training samples from users A/B and 13 testing samples from user C.

Based on this index, a reusable `MultimodalIntentDataset` was implemented. The Dataset supports both lightweight metadata loading and formal feature loading. The formal input pipeline samples HoloLens and fisheye frames into fixed-size tensors, builds IMU numeric tensors, encodes scene labels as one-hot vectors, and caches video tensors for efficient reuse. A feature cache was generated for all 39 samples, producing 78 cached video tensor files.

Finally, an end-to-end baseline training script was added to verify the full pipeline. The script reads the formal Dataset, trains a compact multimodal classifier, evaluates on user C, and saves metrics, predictions, logs, and model checkpoints. This verifies that the project can run from data index and raw-data-derived tensors to model training and evaluation. The current baseline is mainly used as a runnable proof of the end-to-end pipeline; further accuracy improvements are assigned to the experimental/model-improvement stage.

## 8. Handoff Notes for Member B

Member B can start from:

```bash
source /share/home/tm1078571822880000/a893873600/wyr_ml_course_project_work/software/memberA_activate.sh
cd /share/home/tm1078571822880000/multimodal-intention-recognition
python src/data/check_dataset.py --config configs/default.yaml --batch-size 2 --load-features
python experiments/train_formal_dataset.py --config configs/default.yaml --epochs 20 --batch-size 4
```

Recommended Member B work:

1. Use `experiments/train_formal_dataset.py` as the clean runnable baseline.
2. Replace the compact CNN with pretrained CLIP/ViT feature caches using local model files under `data/raw/models/`.
3. Add validation split and best-checkpoint selection.
4. Add modality noise and missing-modality transforms.
5. Improve IMU handling by recovering processed IMU feature files or aligning the raw IMU timeline per sample.
6. Produce final experiment tables from `results/metrics/*.json` or `*.csv`.

Do not treat the smoke-run `0.1538` accuracy as final performance. It only proves that the end-to-end pipeline works.

## 9. Current Caveats

1. The current IMU tensor is not per-video aligned because the raw IMU timestamp range does not directly match the 2026 video filenames.
2. The current model is a compact baseline, not the final improved model.
3. The old teammate path `E:\smart AR` appears in legacy code, but the new data pipeline avoids importing those side-effect modules directly.
4. Some generated files may be ignored by Git, especially caches, logs, checkpoints, and predictions. They still exist on the remote machine and can be used for screenshots/report evidence.


## 10. Standard Train/Test Interface Update

The project now matches the expected stage-2 interface with:

```text
experiments/train.py
experiments/test.py
```

Standard handoff commands for Member B:

```bash
source /share/home/tm1078571822880000/a893873600/wyr_ml_course_project_work/software/memberA_activate.sh
cd /share/home/tm1078571822880000/multimodal-intention-recognition
python experiments/train.py --config configs/default.yaml --epochs 20 --batch-size 4
python experiments/test.py --config configs/default.yaml --checkpoint results/checkpoints/best.pt --batch-size 4
```

Architecture cleanup:

```text
src/models/formal_baseline.py     # compact clean-baseline model
src/training/engine.py            # shared train/evaluate/prediction helpers
experiments/train.py              # standard training entry
experiments/test.py               # standard test entry
experiments/train_formal_dataset.py # compatibility wrapper for old notes
experiments/train_and_test.py     # legacy/original reference, kept for comparison
```

Do not delete `experiments/train_and_test.py` before GitHub submission. It is a useful original/legacy reference for Member B and for comparing against the teacher-provided baseline. The standard commands should use `experiments/train.py` and `experiments/test.py`.

