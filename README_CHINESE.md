# 成员 A Clean Baseline 快速运行说明

本节记录成员 A 已完成的五模态端到端 clean baseline 主线。当前任务只覆盖 clean baseline，不展开成员 B 的 noise、missing 或 improved model 实验实现。

## 数据划分

- 训练集：`user_A` + `user_B`
- 测试集：`user_C`
- 正式模态 key 固定为：`imu`、`gesture`、`audio`、`text`、`scene`
- `HoloLens`、`fisheye` 等目录只作为原始数据来源，不作为正式实验模态名。

## 端到端调用链

```text
data/raw 原始数据
-> src/data/build_samples.py 生成 user/split/sample_id/label 样本索引
-> src/data/features.py 与 src/data/build_features.py 读取或构建五模态特征
-> src/data/dataset.py 输出 MultimodalIntentDataset 样本字典
-> src/models/formal_baseline.py 进行五模态融合
-> src/training/engine.py 训练/评估
-> experiments/train.py 与 experiments/test.py 输出 intent prediction
```

Dataset 输出结构保持为：

```python
sample = {
    "features": {
        "imu": ...,
        "gesture": ...,
        "audio": ...,
        "text": ...,
        "scene": ...,
    },
    "intent_label": ...,
    "scene_label": ...,
    "joint_label": ...,
    "sample_id": ...,
    "user": ...,
    "split": ...,
}
```

缺失模态不会删除 `features` 中的 key，而是 zero-fill，并通过 `modality_mask` 标记。

## 运行命令

正式数据检查：

```bash
python src/data/check_dataset.py --config configs/default.yaml
```

短流程 smoke test，不代表正式实验结果：

```bash
python experiments/train.py --config configs/default.yaml --smoke-test --epochs 1 --batch-size 2
python experiments/test.py --config configs/default.yaml --smoke-test --checkpoint results/checkpoints/best.pt --batch-size 2
```

正式训练和测试应在 `data/raw/imu.csv`、`data/raw/user_A`、`data/raw/user_B`、`data/raw/user_C` 以及本地模型目录准备完成后运行：

```bash
python experiments/train.py --config configs/default.yaml
python experiments/test.py --config configs/default.yaml --checkpoint results/checkpoints/best.pt
```

## 输出文件

- Metrics: `results/metrics/clean_baseline_metrics.csv`
- Test Metrics: `results/metrics/clean_baseline_test_metrics.csv`
- Summary: `results/metrics/clean_baseline_summary.json`
- Test Summary: `results/metrics/clean_baseline_test_summary.json`
- Logs: `results/logs/clean_baseline.log`
- Predictions: `results/predictions/clean_baseline_predictions.csv`
- Best Checkpoint: `results/checkpoints/best.pt`
- Final Checkpoint: `results/checkpoints/final.pt`
- Loss Curve: `figures/loss_curve.png`
- Confusion Matrix: `figures/confusion_matrix.png`

## 当前结果状态

阶段 6 已完成 smoke test：工程链路可运行，checkpoint 可保存和加载，predictions 与 metrics 文件可生成。该 smoke test 使用极小合成样本，只能证明主线连通，不作为正式实验性能。

正式 clean baseline 指标当前为 TBD。原因是仓库中尚未放置正式 raw data 与本地模型资源：

- `data/raw/imu.csv`
- `data/raw/user_A`
- `data/raw/user_B`
- `data/raw/user_C`
- `data/raw/models/all-MiniLM-L6-v2`
- `data/raw/models/clip_teacher_model/vit-base-patch16-224`

---

# 多模态用户意图识别

## 1. 项目概述

本仓库用于机器学习课程期末项目。

本项目聚焦于增强现实交互场景下的 **多模态用户意图识别**。项目目标是在老师提供的多模态识别模型基础上进行重构和改进，使模型能够从原始多模态交互数据中识别用户意图。

课程项目重点关注以下问题：

- 模态噪声
- 模态缺失
- 跨用户测试
- 从原始数据输入到意图类别输出的端到端流程

当前仓库已经包含基线源代码、特征提取脚本、路径规范、项目文档，以及后续实验、结果和报告所需的目录。大体积原始数据集和本地模型资源应按照文档约定放置在服务器或本地机器中，但不应默认提交到仓库。

---

## 2. 项目背景

AR 眼镜和可穿戴交互系统通常会同时接收场景视频、手势视频、音频、语音/文本和 IMU 信号等多种模态信息。实际使用中，单一模态可能受到噪声干扰，也可能出现缺失，因此本项目研究面向用户意图识别的鲁棒多模态融合方法。

老师提供的原始代码流程是先从数据集中提取特征并保存为特征文件，再基于保存好的特征文件训练多模态融合模型。本项目需要将该流程重构为更清晰的端到端训练和测试流程，并在干净数据、模态噪声和模态缺失设置下评估模型表现。

---

## 3. 项目目标

1. 使用用户 A 和用户 B 作为训练集。
2. 使用用户 C 作为测试集。
3. 将原始的基于特征文件的流程重构为端到端训练和测试流程。
4. 将数据预处理和特征提取整合进训练/测试流程。
5. 构建 20%、40% 和 60% 噪声比例的模态噪声基线。
6. 通过丢弃一个或两个模态构建模态缺失基线。
7. 通过新模块或损失项改进模型。
8. 保存最终报告和答辩所需的指标、日志、图表、截图和参考文献材料。

---

## 4. 当前仓库结构

当前仓库结构如下：

```text
multimodal-intention-recognition/
├── README.md
├── README_CHINESE.md
├── requirements.txt
├── AGENTS.md
├── .gitignore
├── 课程项目2026.pdf
│
├── configs/
│   └── default.yaml
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
│   ├── utils/
│   │   ├── paths.py
│   │   ├── logger.py
│   │   └── seed.py
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
│   ├── original_code_readme.md
│   ├── path_setup.md
│   └── collaboration_log.md
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

## 5. 文件和文件夹说明

### `data/raw/`

该目录是原始数据集和本地模型资源的推荐放置位置。

- `imu.csv`：原始 IMU 信号文件，用作 IMU 模态，并通过时间戳与交互视频对齐。
- `user_A/`：用户 A 的原始数据，属于训练集。
- `user_B/`：用户 B 的原始数据，属于训练集。
- `user_C/`：用户 C 的原始数据，属于测试集。
- `HoloLens/`：`.mp4` 格式的 HoloLens 交互视频，主要用于音频和语音相关处理。
- `fisheye/`：`.avi` 格式的鱼眼摄像头视频，主要用于场景模态和手势模态。
- `models/all-MiniLM-L6-v2/`：本地句向量模型资源，用于文本或 ASR 相关特征提取。
- `models/clip_teacher_model/`：原始数据包中提供的本地视觉/视觉语言骨干模型资源。

部分目录中可能包含 `.DS_Store` 或 `._*` 文件，这些是 macOS 系统元数据文件，不应作为有效训练样本读取。

大体积原始数据和预训练模型目录可能不会出现在克隆后的仓库中。正式项目运行时，应按照 `configs/default.yaml` 和 `docs/path_setup.md` 在服务器中准备这些文件。

### `data/processed/`

该目录预留给生成的中间数据、缓存特征、对齐后的样本和其他预处理输出。当前它主要作为输出目录保留，不应作为原始数据来源。

### `src/models/`

该目录存放模型定义。

- `baseline_real_scene.py`：迁移后的老师基线模型代码，包含基于 Perceiver-IO 的多模态融合基线、意图标签、训练/测试视频列表、模态定义、训练逻辑和评估工具。该文件后续仍需要继续重构，整理为更清晰的端到端实现。

### `src/modules/`

该目录存放模型和实验会复用的功能模块。

- `real_scene_utils.py`：用于建立鱼眼 `.avi` 视频和 HoloLens `.mp4` 视频之间的映射，读取真实场景帧，使用本地 ViT 模型提取场景特征，并缓存场景特征。
- `feature_extraction/`：从老师提供的代码包中迁移出来的原始特征提取脚本。

### `src/modules/feature_extraction/`

该目录存放各模态预处理和特征提取脚本。

- `get_timestamp.py`：提取或对齐交互数据时间戳。
- `ASR.py`：执行语音识别或语音文本特征处理。
- `imu.py`：处理 IMU 信号并提取 IMU 特征。
- `mfcc.py`：提取音频 MFCC 特征。
- `strong_gesture2.0.py`：从视频数据中提取手势相关特征。

这些脚本是后续将特征提取整合进端到端训练和测试流程的主要参考。

### `src/utils/`

该目录存放后续训练、测试和实验脚本可复用的轻量级工程工具。

- `paths.py`：读取 `configs/default.yaml`，解析项目相对路径，创建运行时输出目录，设置 Hugging Face 缓存环境变量，并打印路径检查报告。
- `logger.py`：提供统一的项目日志工具，实验日志可保存到配置中的 `results/logs/` 目录。
- `seed.py`：提供随机种子工具，用于统一设置 Python、NumPy 和 PyTorch 随机种子，并提供 DataLoader worker 与 generator 辅助函数。

### `experiments/`

该目录存放实验入口脚本。

- `train_and_test.py`：迁移后的原始训练/测试脚本，目前作为主要参考实现。后续应将它拆分为更清晰的训练和测试入口，例如 `train.py`、`test.py`，并进一步增加干净数据、噪声模态和缺失模态等不同实验设置的脚本。

### `docs/`

该目录存放项目文档。

- `original_code_readme.md`：老师提供的代码说明文档，解释了原始代码结构、训练/测试数据划分、特征提取脚本和课程提交要求。
- `path_setup.md`：说明标准化的数据、模型、缓存、输出、图表和报告路径。
- `collaboration_log.md`：记录重要项目修改，便于团队协作和后续报告整理。

### `configs/`

该目录存放项目配置文件。

- `default.yaml`：定义原始数据、处理中间数据、本地模型、缓存、结果、日志、预测、图表和报告材料的项目相对路径。

### `results/`

该目录用于保存实验输出。

- `metrics/`：准确率、F1 分数、分类报告、混淆矩阵和对比表格。
- `logs/`：训练和测试日志。
- `predictions/`：验证集或测试样本上的模型预测结果。

### `figures/`

该目录用于保存项目报告和答辩 PPT 所需的图片。

- `model_structure/`：模型结构图。
- `curves/`：训练损失、验证准确率等曲线。
- `result_charts/`：干净数据、噪声模态、缺失模态、基线模型和改进模型之间的对比图。

### `report/`

该目录用于保存报告相关材料。

- `references.md`：最终报告参考文献列表或引用记录。
- `screenshots/`：证明端到端模型成功运行的截图。

### `课程项目2026.pdf`

该文件是课程项目官方要求文档，定义了项目主题、实现要求、报告要求和评分标准。

### `AGENTS.md`

该文件记录 Codex 协作开发时需要遵守的项目级规则，包括依赖使用、输出目录、日志记录和更新规范。

---

## 6. 开发计划

### 阶段一：数据和代码整理

- 将原始多模态数据保留在 `data/raw/`。
- 使用用户 A 和用户 B 作为训练集。
- 使用用户 C 作为测试集。
- 将老师提供的基线代码和特征提取脚本保留在 `src/` 和 `experiments/` 中。
- 准备数据、本地模型、缓存、输出、图表和报告材料时，遵循 `configs/default.yaml` 和 `docs/path_setup.md`。

### 阶段二：基线代码重构

- 重构 `experiments/train_and_test.py`。
- 将可复用模型组件整理到 `src/models/`。
- 将预处理和特征提取调用整理为可复用模块。
- 将原始代码中的硬编码本地路径替换为项目相对路径。

### 阶段三：端到端训练和测试

- 构建从原始多模态数据开始的训练脚本。
- 构建输出意图分类结果的测试脚本。
- 保存模型检查点、标准化器、标签编码器、指标和日志。
- 编写新脚本时，使用 `src/utils/logger.py` 统一日志记录，并使用 `src/utils/seed.py` 提高实验可复现性。

### 阶段四：模态噪声实验

- 对每一个单独模态加入 20%、40% 和 60% 噪声。
- 在每一种噪声设置下训练并评估模型。
- 保存对比指标和结果图表。

### 阶段五：模态缺失实验

- 分别丢弃每一个单独模态和每一组双模态。
- 在不同模态缺失设置下训练并评估模型。
- 分析模型在不完整输入条件下的鲁棒性。

### 阶段六：模型改进

- 加入改进的融合模块、模态可靠性估计、辅助损失或其他面向鲁棒性的组件。
- 将改进结果与基线结果进行对比。

### 阶段七：报告和答辩

- 保存端到端流程成功运行的截图。
- 整理结果表格和图表。
- 按要求撰写课程项目报告，包含参考文献和团队贡献比例。
- 准备小组答辩 PPT。

---

## 7. 团队协作

建议分工包括：

- 数据集整理和预处理
- 基线代码重构
- 特征提取整合
- 模态噪声和模态缺失实验
- 模型改进
- 结果分析和可视化
- 报告撰写和答辩准备

每位成员最终的贡献比例应记录在项目报告中。

开发说明：

- 编写代码时应严格基于 `requirements.txt` 中列出的依赖。
- 如果当前本地环境缺少 `requirements.txt` 中的依赖，不应为了本地环境加入兼容性兜底，也不应只为当前本地环境安装依赖。
- 正式项目以服务器环境为运行环境，服务器中应已具备 `requirements.txt` 中的依赖。
- 每次进行有意义的代码或文档更新后，应按照项目规范记录到 `docs/collaboration_log.md`、`docs/experiment_log.md` 或 `docs/method_notes.md` 中。

---

## 8. 当前状态

当前仓库状态：

- 项目主题和课程要求文档已经准备好。
- 标准路径配置文件位于 `configs/default.yaml`。
- 路径设置说明文档位于 `docs/path_setup.md`。
- 原始数据、用户目录、IMU 数据和本地模型资源应按照配置中的 `data/raw/` 结构在服务器或本地机器中准备。
- 基线模型代码位于 `src/models/baseline_real_scene.py`。
- 真实场景工具位于 `src/modules/real_scene_utils.py`。
- 特征提取脚本位于 `src/modules/feature_extraction/`。
- 路径、日志和随机种子工具位于 `src/utils/`。
- 原始训练/测试脚本位于 `experiments/train_and_test.py`。
- 结果、图表和报告目录已经为后续输出准备好。

---

## 9. 说明

- 当前源代码仍接近老师提供的原始版本，可能包含原始环境中的硬编码路径，后续重构时需要替换为项目相对路径。
- 大体积原始数据和训练生成的模型检查点不建议提交到 Git，除非课程提交明确要求。
- `.DS_Store` 和 `._*` 等 macOS 元数据文件应在数据加载和实验执行时忽略。
- `src/utils/logger.py` 和 `src/utils/seed.py` 是共享工具文件，目前尚未接入现有 baseline 训练/测试脚本。
