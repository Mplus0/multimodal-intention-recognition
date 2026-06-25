# 多模态用户意图识别

机器学习课程项目，2025-2026 学年第二学期。

本仓库实现增强现实交互场景下的多模态用户意图识别流程。项目在老师提供的“先提取特征、保存特征文件、再读取特征训练”的原始流程基础上，重构为项目相对路径下的端到端流程，并补充 clean baseline、模态噪声、模态缺失和鲁棒改进模型实验入口。

当前仓库状态：

- 代码流程和实验入口已经完整实现，可在课程服务器环境运行。
- 仓库不包含大体积原始数据集、本地预训练模型、已训练 checkpoint 或正式结果文件。
- 当前快照尚未完成正式全量训练和评估。
- `results/`、`figures/`、`report/screenshots/` 是为服务器运行后生成结果预留的输出目录。

---

## 1. 课程项目要求

官方任务要求：

1. 用户 A、用户 B 作为训练集。
2. 用户 C 作为测试集。
3. 将原始流程重构为：

```text
raw data -> 数据预处理 / 特征提取 -> 多模态融合模型 -> 意图类别输出
```

4. 分别对每个单模态加入 20%、40%、60% 噪声，构建模态噪声 baseline。
5. 分别丢弃任意一个模态和任意两个模态，构建模态缺失 baseline。
6. 引入新模块或损失项，提升噪声和缺失条件下的分类准确率。
7. 保存报告和答辩需要的日志、指标、预测、图表、截图和参考文献。

---

## 2. 正式模态与数据划分

正式模态 key：

```text
imu
gesture
audio
text
scene
```

特征形状约定：

| 模态 | 形状 | 含义 |
|---|---|---|
| `imu` | `(N, 10, 12)` | IMU 时序特征 |
| `gesture` | `(N, 10, 768)` | 手势视觉时序特征 |
| `audio` | `(N, 10, 39)` | MFCC 音频时序特征 |
| `text` | `(N, 10, 384)` | ASR / 句向量文本时序特征 |
| `scene` | `(N, 768)` | 场景视觉特征 |

数据划分：

| 划分 | 用户 |
|---|---|
| 训练集 | `user_A`, `user_B` |
| 测试集 | `user_C` |

样本索引由 `src/data/build_samples.py` 生成。完整数据集的视频级样本统计预期为：

```text
sample_count: 39
train: 26
test: 13
user_A: 13
user_B: 13
user_C: 13
```

---

## 3. 仓库结构

```text
multimodal-intention-recognition/
├── AGENTS.md
├── README.md
├── README_CHINESE.md
├── requirements.txt
├── 课程项目2026.pdf
├── MEMBER_A_FORMAL_RUN_PLAN.md
├── MODEL_OPTIMIZATION_TASK_PLAN.md
│
├── configs/
│   ├── default.yaml
│   ├── noise.yaml
│   ├── missing_modality.yaml
│   └── improved_model.yaml
│
├── docs/
│   ├── collaboration_log.md
│   ├── original_code_readme.md
│   └── path_setup.md
│
├── experiments/
│   ├── train.py
│   ├── test.py
│   ├── train_and_test.py
│   ├── train_formal_dataset.py
│   ├── run_clean_baseline.py
│   ├── run_noise_baseline.py
│   ├── run_missing_baseline.py
│   └── run_improved_model.py
│
├── src/
│   ├── data/
│   │   ├── build_samples.py
│   │   ├── build_features.py
│   │   ├── dataset.py
│   │   ├── features.py
│   │   ├── raw_feature_builder.py
│   │   ├── transforms.py
│   │   ├── improved_transforms.py
│   │   └── raw_extractors/
│   ├── models/
│   │   ├── baseline_real_scene.py
│   │   ├── formal_baseline.py
│   │   └── improved_model.py
│   ├── modules/
│   │   ├── real_scene_utils.py
│   │   └── feature_extraction/
│   ├── training/
│   │   ├── engine.py
│   │   ├── evaluate.py
│   │   ├── experiment_runner.py
│   │   └── improved_experiment_runner.py
│   └── utils/
│       ├── logger.py
│       ├── paths.py
│       └── seed.py
│
├── results/
│   ├── logs/
│   ├── metrics/
│   └── predictions/
├── figures/
│   ├── curves/
│   ├── model_structure/
│   └── result_charts/
└── report/
    ├── references.md
    └── screenshots/
```

---

## 4. 数据与本地模型放置方式

大文件不提交到仓库。服务器中应按以下结构准备：

```text
data/raw/
├── imu.csv
├── user_A/
│   ├── HoloLens/
│   └── fisheye/
├── user_B/
│   ├── HoloLens/
│   └── fisheye/
├── user_C/
│   ├── HoloLens/
│   └── fisheye/
└── models/
    ├── all-MiniLM-L6-v2/
    └── clip_teacher_model/
        └── vit-base-patch16-224/
```

说明：

- `HoloLens/` 中的 `.mp4` 视频用于音频和文本模态。
- `fisheye/` 中的 `.avi` 视频用于手势和场景模态。
- `imu.csv` 通过时间戳与交互视频对齐。
- CLIP、ViT、sentence model 均使用 `local_files_only=True` 从本地加载。
- 生成的特征缓存放在 `data/processed/cache/` 下。

---

## 5. 主要代码模块

### 数据流程

| 文件 | 作用 |
|---|---|
| `src/data/build_samples.py` | 构建用户 A/B/C 的样本元数据 |
| `src/data/build_features.py` | 检查或构建五模态特征数组 |
| `src/data/raw_feature_builder.py` | 调度缺失源特征的构建 |
| `src/data/raw_extractors/*` | 手势、音频、IMU、文本、场景的 raw data adapter |
| `src/data/features.py` | 加载、缓存、归一化五模态正式特征 |
| `src/data/dataset.py` | PyTorch Dataset 与 `modality_mask` 约定 |
| `src/data/transforms.py` | feature-level 模态噪声和模态缺失 transform |
| `src/data/improved_transforms.py` | improved model 训练阶段随机模态 Dropout |

### 模型

| 文件 | 作用 |
|---|---|
| `src/models/baseline_real_scene.py` | 老师 baseline 迁移代码 / 参考实现 |
| `src/models/formal_baseline.py` | 正式五模态 Transformer baseline |
| `src/models/improved_model.py` | 可靠性门控改进融合模型 |

### 训练与实验

| 文件 | 作用 |
|---|---|
| `src/training/engine.py` | 通用训练、评估、保存工具 |
| `src/training/experiment_runner.py` | clean/noise/missing baseline 通用 runner |
| `src/training/improved_experiment_runner.py` | improved model 通用 runner |
| `experiments/train.py` | clean baseline 训练入口 |
| `experiments/test.py` | clean baseline 测试入口 |
| `experiments/run_clean_baseline.py` | 统一 clean baseline 入口 |
| `experiments/run_noise_baseline.py` | 5 模态 × 3 噪声比例实验 |
| `experiments/run_missing_baseline.py` | 单模态和双模态缺失实验 |
| `experiments/run_improved_model.py` | improved clean/noise/missing/ablation 实验 |

---

## 6. 当前已实现实验组

| 实验组 | 入口 | 状态 |
|---|---|---|
| 端到端特征构建 | `src/data/build_features.py` | 已实现 |
| clean baseline | `experiments/train.py`, `experiments/test.py`, `experiments/run_clean_baseline.py` | 已实现 |
| 模态噪声 baseline | `experiments/run_noise_baseline.py` | 已实现 |
| 模态缺失 baseline | `experiments/run_missing_baseline.py` | 已实现 |
| improved model | `experiments/run_improved_model.py` | 已实现 |
| 消融实验 | `experiments/run_improved_model.py --mode ablation` | 已实现 |

代码已经在服务器完成 smoke test 和短跑验证，但当前仓库快照不包含正式全量训练结果。

---

## 7. 改进模型简介

改进模型名称：

```text
Reliability-Gated Multimodal Fusion with Modality Dropout
```

中文描述：

```text
基于模态可靠性门控与模态 Dropout 的鲁棒多模态融合方法
```

核心思想：

1. 为每个模态学习一个可靠性门控权重。
2. 在融合阶段显式使用 `modality_mask`。
3. 缺失模态对最终融合表示的贡献被置为 0。
4. 训练阶段使用随机模态 Dropout，提高模型对模态缺失的鲁棒性。

关键文件：

```text
src/models/improved_model.py
src/data/improved_transforms.py
src/training/improved_experiment_runner.py
experiments/run_improved_model.py
configs/improved_model.yaml
```

详细设计见：

```text
MODEL_OPTIMIZATION_TASK_PLAN.md
```

---

## 8. 服务器快速运行

进入项目：

```bash
cd /share/home/tm1078571822880000/a1086482920/Mplus0/multimodal-intention-recognition
conda activate machine_learning
export PY=/share/home/tm1078571822880000/a1086482920/Mplus0/software/miniconda3/envs/machine_learning/bin/python
mkdir -p report/screenshots report/terminal_logs
```

检查 Python / CUDA：

```bash
$PY -V
$PY -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

生成样本索引：

```bash
$PY src/data/build_samples.py --config configs/default.yaml --output data/processed/sample_index.json
```

构建或检查五模态特征：

```bash
$PY src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing --limit 1
$PY src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing --limit 3
$PY src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing --limit 100000
```

运行 clean baseline：

```bash
$PY experiments/train.py --config configs/default.yaml
$PY experiments/test.py --config configs/default.yaml --checkpoint results/checkpoints/best.pt
```

运行 baseline 实验组：

```bash
$PY experiments/run_clean_baseline.py --config configs/default.yaml
$PY experiments/run_noise_baseline.py --config configs/noise.yaml --base-config configs/default.yaml
$PY experiments/run_missing_baseline.py --config configs/missing_modality.yaml --base-config configs/default.yaml
```

运行 improved model 检查与实验：

```bash
$PY src/models/improved_model.py --config configs/default.yaml --smoke-test
$PY src/data/improved_transforms.py
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean --smoke-test --epochs 1
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean --max-samples 30 --epochs 1
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode ablation
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode noise
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode missing
```

完整命令说明和截图清单位于：

```text
MEMBER_A_FORMAL_RUN_PLAN.md
```

---

## 9. 输出位置

正式运行结果应写入：

```text
results/metrics/
results/logs/
results/predictions/
results/checkpoints/
figures/
report/screenshots/
report/terminal_logs/
```

全量实验后预期关键指标文件：

```text
results/metrics/clean_baseline_metrics.csv
results/metrics/clean_baseline_test_metrics.csv
results/metrics/noise_baseline_metrics.csv
results/metrics/missing_modality_metrics.csv
results/metrics/improved_clean_metrics.csv
results/metrics/improved_ablation_metrics.csv
results/metrics/improved_noise_metrics.csv
results/metrics/improved_missing_modality_metrics.csv
results/metrics/improved_model_metrics.csv
```

当前仓库快照说明：

- 不包含正式结果 CSV。
- 不包含已训练 checkpoint。
- 不包含正式实验生成的图表。
- 输出目录仅作为服务器运行后的结果保存位置。

---

## 10. 论文 / 报告编写材料

有用文件：

| 文件 | 用途 |
|---|---|
| `课程项目2026.pdf` | 官方要求文档 |
| `docs/original_code_readme.md` | 老师代码和数据划分参考 |
| `docs/path_setup.md` | 数据、模型、缓存、输出路径约定 |
| `docs/collaboration_log.md` | 实现过程和团队协作记录 |
| `MEMBER_A_FORMAL_RUN_PLAN.md` | 全流程服务器运行和截图保存手册 |
| `MODEL_OPTIMIZATION_TASK_PLAN.md` | 改进模型设计和可复用方法描述 |
| `report/references.md` | 参考文献占位文件 |

建议报告结构：

1. Introduction
2. Dataset and task definition
3. End-to-end refactoring
4. Baseline model
5. Modal noise experiments
6. Missing-modality experiments
7. Improved method
8. Experimental results
9. Ablation study
10. Discussion
11. Team contribution
12. References

重要规则：

- 不要将 smoke-test 或 `--max-samples` 结果作为正式指标写入报告。
- 正式指标必须来自完整数据运行后的 `results/metrics/`。

---

## 11. 依赖

依赖见 `requirements.txt`，主要包括：

- `numpy`, `pandas`, `scipy`
- `scikit-learn`
- `torch`, `torchvision`, `torchaudio`
- `opencv-python-headless`
- `transformers`
- `librosa`
- `moviepy`
- `openai-whisper`
- `sentence-transformers`
- `pypinyin`
- `webrtcvad`
- `mediapipe`
- `matplotlib`, `seaborn`

开发规则：

- 不为本机缺失依赖添加临时兼容分支。
- 不只为了本机桌面环境安装依赖。
- 正式运行以课程服务器环境为准。

---

## 12. Git 与大文件说明

不要提交：

- 原始视频数据。
- 属于大数据包的 `imu.csv`。
- 本地预训练模型目录。
- 生成的特征缓存。
- 训练生成的 checkpoint，除非课程最终提交明确要求。
- 大体积实验输出。

建议提交：

- 源代码。
- 配置文件。
- 项目文档。
- 轻量级报告材料。
- 如果提交要求允许，可提交最终筛选后的轻量结果表格或图。

---

## 13. 当前快照总结

当前仓库状态可概括为：

```text
代码完整，服务器可运行，但尚未正式全量训练。
```

已实现：

- 端到端 raw data 特征构建。
- clean baseline。
- 模态噪声 baseline。
- 模态缺失 baseline。
- 可靠性门控 improved model。
- 模态 Dropout 训练增强。
- improved clean/noise/missing/ablation runner。
- 完整服务器运行文档和截图保存清单。

未包含：

- 大体积数据集。
- 本地预训练模型目录。
- 正式训练结果。
- 正式 checkpoint。
- 最终结果图表。

