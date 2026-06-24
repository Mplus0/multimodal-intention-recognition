# 成员 A 实验日志

## 2026-06-24 - 阶段 5 Clean Baseline Smoke Test

### 实验目的
验证成员 A 五模态端到端 clean baseline 的训练、checkpoint 保存、测试、指标导出、预测导出和图像导出链路是否可执行。

### 实验设置
- 入口：`experiments/train.py`、`experiments/test.py`
- 配置：`configs/default.yaml`
- 模型：`FormalMultimodalBaseline`
- 模态 key：`imu`、`gesture`、`audio`、`text`、`scene`
- 验证方式：`--smoke-test` 极小合成样本
- 训练轮数：1 epoch
- batch size：2

### 运行命令
```bash
python experiments/train.py --smoke-test --epochs 1 --batch-size 2
python experiments/test.py --smoke-test --checkpoint results/checkpoints/best.pt --batch-size 2
```

### 输出文件
- `results/checkpoints/best.pt`
- `results/checkpoints/final.pt`
- `results/metrics/clean_baseline_metrics.csv`
- `results/metrics/clean_baseline_summary.json`
- `results/metrics/clean_baseline_test_metrics.csv`
- `results/metrics/clean_baseline_test_summary.json`
- `results/predictions/clean_baseline_predictions.csv`
- `results/logs/clean_baseline.log`
- `figures/loss_curve.png`
- `figures/confusion_matrix.png`

### 结果说明
本次 smoke-test 仅用于验证工程链路，不作为正式实验结果。正式指标需要在 `data/raw/user_A`、`data/raw/user_B`、`data/raw/user_C` 和真实五模态特征准备完成后重新运行。

### 已知问题
- 当前仓库缺少正式 raw data，真实 user_C 测试集无法完成端到端验证。
- `joint_accuracy` 目前为补充占位指标，需要后续补充 joint label 到类别 id 的稳定映射。

## 阶段 6 Clean Baseline 最小流程验证

### Date
2026-06-24

### Purpose
在不运行长时间完整训练的前提下，验证成员 A clean baseline 主线可以从 raw data 检查入口启动，能清晰报告缺失数据，并能通过 smoke-test 完成训练、checkpoint 加载、测试预测和结果文件导出。

### Command
```bash
python src/data/check_dataset.py --config configs/default.yaml
python experiments/train.py --config configs/default.yaml --smoke-test --epochs 1 --batch-size 2
python experiments/test.py --config configs/default.yaml --smoke-test --checkpoint results/checkpoints/best.pt --batch-size 2
```

### Dataset Split
- Train: user A + user B
- Test: user C
- 当前真实 raw data 样本数：0

### Settings
- Modalities: imu, gesture, audio, text, scene
- Noise setting: none
- Missing modality: none
- Model: clean baseline / `FormalMultimodalBaseline`
- 验证方式：1 epoch smoke-test，batch size 2

### Main Results
| Metric | Value |
|---|---:|
| Train Smoke Accuracy | 0.25 |
| Train Smoke Macro F1 | 0.10 |
| Train Smoke Weighted F1 | 0.10 |
| Test Smoke Accuracy | 0.25 |
| Test Smoke Macro F1 | 0.10 |
| Test Smoke Weighted F1 | 0.10 |
| Training Time | 0.7756 |
| Avg Test Time / Sample | 0.06062 |

### Output Paths
- Metrics: `results/metrics/clean_baseline_metrics.csv`
- Test Metrics: `results/metrics/clean_baseline_test_metrics.csv`
- Summary: `results/metrics/clean_baseline_summary.json`
- Test Summary: `results/metrics/clean_baseline_test_summary.json`
- Log: `results/logs/clean_baseline.log`
- Predictions: `results/predictions/clean_baseline_predictions.csv`
- Checkpoint: `results/checkpoints/best.pt`
- Final Checkpoint: `results/checkpoints/final.pt`
- Figure: `figures/loss_curve.png`
- Figure: `figures/confusion_matrix.png`

### Brief Analysis
`src/data/check_dataset.py` 能从 `configs/default.yaml` 启动并检查 user A/B/C 划分，但当前仓库缺少正式 raw data，因此真实样本数为 0。`train.py` 和 `test.py` 的 smoke-test 链路已跑通，证明五模态 Dataset、DataLoader、模型、训练引擎、checkpoint、预测和指标输出可以连接起来。当前 smoke-test 使用合成小样本，只能作为工程链路验证，不能作为正式实验性能。

### Problems
- 缺少 `data/raw/imu.csv`。
- 缺少 `data/raw/user_A`。
- 缺少 `data/raw/user_B`。
- 缺少 `data/raw/user_C`。
- 缺少本地模型目录 `data/raw/models/all-MiniLM-L6-v2`。
- 缺少本地模型目录 `data/raw/models/clip_teacher_model/vit-base-patch16-224`。
- `joint_accuracy` 仍为补充占位指标，正式计算需要后续建立 joint label 到类别 id 的映射。

## 阶段 8 Clean Baseline 文档入口整理

### Date
2026-06-24

### Purpose
整理成员 A 已完成 clean baseline 主线的运行说明、输出说明和报告材料入口，方便后续报告撰写和答辩复现。

### Command
```bash
python src/data/check_dataset.py --config configs/default.yaml
python experiments/train.py --config configs/default.yaml --smoke-test --epochs 1 --batch-size 2
python experiments/test.py --config configs/default.yaml --smoke-test --checkpoint results/checkpoints/best.pt --batch-size 2
```

### Dataset Split
- Train: user A + user B
- Test: user C

### Settings
- Modalities: imu, gesture, audio, text, scene
- Noise setting: none
- Missing modality: none
- Model: clean baseline / `FormalMultimodalBaseline`
- 文档状态：正式 clean baseline 指标为 TBD；当前只记录 smoke-test 工程链路已跑通。

### Main Results
| Metric | Value |
|---|---:|
| Formal Clean Accuracy | TBD |
| Formal Clean Macro F1 | TBD |
| Formal Clean Weighted F1 | TBD |
| Smoke Test Status | passed |

### Output Paths
- Metrics: `results/metrics/clean_baseline_metrics.csv`
- Test Metrics: `results/metrics/clean_baseline_test_metrics.csv`
- Summary: `results/metrics/clean_baseline_summary.json`
- Test Summary: `results/metrics/clean_baseline_test_summary.json`
- Log: `results/logs/clean_baseline.log`
- Predictions: `results/predictions/clean_baseline_predictions.csv`
- Checkpoint: `results/checkpoints/best.pt`
- Final Checkpoint: `results/checkpoints/final.pt`
- Figure: `figures/loss_curve.png`
- Figure: `figures/confusion_matrix.png`

### Brief Analysis
本阶段没有重新训练模型，只整理 README 和报告入口说明。README 已明确端到端链路为 `raw data -> preprocessing/features -> model -> intent output`，并说明用户 A/B 为训练集、用户 C 为测试集。正式实验结果未编造，保持 TBD。

### Problems
- 当前仓库缺少正式 raw data 和本地模型资源，正式 clean baseline 指标仍不能生成。
- 当前文档只覆盖成员 A clean baseline 主线，不覆盖成员 B 的 noise、missing 或 improved model 实验实现。
## 2026-06-24 成员 A raw-to-feature 最小真实样本验证

### 目标
验证成员 A clean baseline 不依赖人工提前准备的五模态 `.npy`，能够从 `data/raw/` 中的正式原始文件启动，自动生成或复用五模态源特征缓存，并进入训练入口。

### 数据与划分
- 样本总数：39
- 训练集：`user_A` + `user_B`，共 26 条
- 测试集：`user_C`，共 13 条
- 本次最小验证样本：`interaction_20260131_071552`

### 运行命令
```bash
../software/memberA_miniconda3/bin/python src/data/build_samples.py --config configs/default.yaml --output data/processed/sample_index.json
../software/memberA_miniconda3/bin/python src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing-source-features --limit 1
../software/memberA_miniconda3/bin/python experiments/train.py --config configs/default.yaml --max-samples 1 --epochs 1 --batch-size 1
```

### 五模态特征 shape
| Modality | Shape |
|---|---|
| imu | `(1, 10, 12)` |
| gesture | `(1, 10, 768)` |
| audio | `(1, 10, 39)` |
| text | `(1, 10, 384)` |
| scene | `(1, 768)` |

### 输出文件
- `data/processed/sample_index.json`
- `data/processed/cache/feature_cache/imu_features/imu_features_interaction_20260131_071552.npy`
- `data/processed/cache/feature_cache/strong_gesture_features/strong_gesture_features_interaction_20260131_071552.npy`
- `data/processed/cache/feature_cache/audio_features/audio_features_interaction_20260131_071552.npy`
- `data/processed/cache/feature_cache/text_features/text_features_interaction_20260131_071552.npy`
- `data/processed/cache/feature_cache/scene_features/scene_features_interaction_20260131_071552.npy`
- `results/checkpoints/best.pt`
- `results/checkpoints/final.pt`
- `results/metrics/clean_baseline_metrics.csv`
- `results/metrics/clean_baseline_summary.json`
- `results/predictions/clean_baseline_predictions.csv`
- `results/logs/clean_baseline.log`
- `figures/loss_curve.png`
- `figures/confusion_matrix.png`

### 结果说明
- 本次 `--max-samples 1` 训练只作为工程链路验证，不能作为正式实验结果写入报告。
- 该验证说明代码已经能从 raw data 构建五模态缓存，并复用缓存进入 clean baseline 训练。
- 首次文本特征生成需要 Whisper ASR 模型；本次已准备 `data/processed/cache/whisper/small.pt`。

### 下一步
- 建议先运行：
```bash
../software/memberA_miniconda3/bin/python src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing-source-features --limit 3
```
- 如果 3 个样本稳定，再运行全量预构建或正式训练：
```bash
../software/memberA_miniconda3/bin/python src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing-source-features --all
../software/memberA_miniconda3/bin/python experiments/train.py --config configs/default.yaml
../software/memberA_miniconda3/bin/python experiments/test.py --config configs/default.yaml --checkpoint results/checkpoints/best.pt
```
