## 2026-06-22 - 路径配置规范化

### 贡献者
- 姓名：Mplus0
- 角色：文档 / 配置

### 修改文件
- `configs/default.yaml`：创建项目相对路径的统一配置约定文件。
- `docs/path_setup.md`：创建中文路径设置说明，解释本地数据集、模型、缓存、输出、图表和报告材料的放置规则。

### 修改目的
标准化团队协作中的文件夹路径约定，便于成员理解本地数据、模型、缓存、中间文件和实验输出应放置的位置。本次修改不改变现有源码或实验流程。

### 运行方式
本次任务只涉及配置和文档，不需要运行命令。

### 输出文件
- `configs/default.yaml`
- `docs/path_setup.md`

### 当前状态
- 已完成

### 给报告撰写者的说明
本步骤只记录本地数据集、本地模型、缓存文件、处理后数据、实验结果、图表和报告材料的推荐放置位置。

### 遗留问题
- 当前训练和测试代码尚未接入 `configs/default.yaml`。
- 本次任务未运行任何实验。

## 2026-06-22 - 路径工具文件创建

### 贡献者
- 姓名：Mplus0
- 角色：代码 / 文档

### 修改文件
- `src/utils/paths.py`：创建轻量级路径工具文件，用于读取 `configs/default.yaml` 并解析项目相对路径。
- `docs/collaboration_log.md`：追加本次路径工具创建记录。

### 修改目的
为团队协作提供可复用的路径读取、路径解析、运行时目录创建、Hugging Face 缓存环境变量设置和路径报告打印工具。当前工具尚未接入 baseline、训练、测试或特征提取流程。

### 运行方式
```bash
python src/utils/paths.py
```

### 输出文件
- `src/utils/paths.py`
- `docs/collaboration_log.md`

### 当前状态
- 已完成

### 给报告撰写者的说明
本次修改只提供路径配置读取工具，不涉及实验结果、模型结构或训练流程变化。

### 遗留问题
- 运行 `src/utils/paths.py` 前需要确保 `configs/default.yaml` 已存在。
- 本次任务未运行任何实验。

## 2026-06-23 - 成员A环境准备与数据路径确认

### Contributor
- Name: 成员A
- Role: Code / Data Engineering

### Files Changed
- data/raw/imu.csv: 创建软链接，指向课程数据集 imu.csv。
- data/raw/HoloLens: 创建软链接，指向课程数据集 HoloLens 视频目录。
- data/raw/fisheye: 创建软链接，指向课程数据集 fisheye 视频目录。
- data/raw/models: 创建软链接，指向课程数据集本地模型目录。
- /share/home/tm1078571822880000/a893873600/wyr_ml_course_project_work/software/memberA_miniconda3: 安装成员A专属 Miniconda。
- /share/home/tm1078571822880000/a893873600/wyr_ml_course_project_work/software/memberA_activate.sh: 新增成员A环境激活脚本。

### Purpose
为成员A后续负责的数据流梳理、端到端重构和路径调试准备基础环境，并确认真实数据集可以通过仓库 data/raw 路径访问。

### How to Run
- source /share/home/tm1078571822880000/a893873600/wyr_ml_course_project_work/software/memberA_activate.sh
- python src/utils/paths.py

### Output Files
- /share/home/tm1078571822880000/a893873600/wyr_ml_course_project_work/software/memberA_miniconda3
- /share/home/tm1078571822880000/a893873600/wyr_ml_course_project_work/software/memberA_activate.sh
- /share/home/tm1078571822880000/a893873600/wyr_ml_course_project_work/software/memberA_pip_no_torch.log

### Current Status
- Partially completed

### Notes for Report Writer
当前已完成数据软链接、Miniconda 下载与基础非 PyTorch 依赖安装。路径检查显示 imu.csv、HoloLens、fisheye 和本地模型根目录可访问。服务器有 8 张 RTX 4090 GPU。

### Remaining Problems
- PyTorch、torchvision、torchaudio 和 torchmetrics 尚未安装，需要确认 CUDA/PyTorch 版本后再安装。
- configs/default.yaml 中 user_A、user_B、user_C 目录仍为 WARN，因为真实数据当前是 HoloLens/fisheye 平铺目录，尚未按用户拆分。
- local_models.vit_model 仍为 WARN，真实模型目录下暂未发现 clip_teacher_model/vit-base-patch16-224 子目录。

## 2026-06-23 - 成员A PyTorch CUDA 环境完善

### Contributor
- Name: 成员A
- Role: Code / Experiment

### Files Changed
- /share/home/tm1078571822880000/a893873600/wyr_ml_course_project_work/software/memberA_miniconda3: 安装 PyTorch CUDA 相关依赖。
- /share/home/tm1078571822880000/a893873600/wyr_ml_course_project_work/software/memberA_pytorch_install.log: 保存 PyTorch 安装日志。

### Purpose
为后续端到端训练、Dataset/DataLoader、GPU 训练和模型 checkpoint 保存准备 PyTorch 运行环境。

### How to Run
- source /share/home/tm1078571822880000/a893873600/wyr_ml_course_project_work/software/memberA_activate.sh
- python -m pip show torch torchvision torchaudio torchmetrics

### Output Files
- /share/home/tm1078571822880000/a893873600/wyr_ml_course_project_work/software/memberA_pytorch_install.log

### Current Status
- Completed

### Notes for Report Writer
成员A环境已安装 torch 2.12.1+cu126、torchvision 0.27.1+cu126、torchaudio 2.11.0+cu126 和 torchmetrics 1.9.0。验证结果显示 CUDA 12.6 可用，torch.cuda.is_available 为 True，可识别 8 张 NVIDIA GeForce RTX 4090。

### Remaining Problems
- 当前成员A环境基于 Python 3.13.13；后续如遇到旧脚本或第三方库兼容问题，再考虑单独创建 Python 3.10 环境。
- 数据仍需完成用户 A/B/C 样本映射或目录拆分，才能进入稳定端到端训练。

## 2026-06-23 - 成员A样本索引与数据检查脚本

### Contributor
- Name: 成员A
- Role: Code / Data Engineering

### Files Changed
- `src/data/build_samples.py`: 新增样本索引构建脚本，复用现有视频标签、训练测试划分和 fisheye-HoloLens 映射，生成端到端数据流所需的样本表。
- `data/processed/sample_index.csv`: 生成可用样本索引，包含 user、split、scene、intent、HoloLens 视频、fisheye 视频和 IMU 路径。
- `results/metrics/sample_statistics.csv`: 生成样本数量、用户分布、场景分布、意图分布和数据问题统计。
- `results/logs/sample_build_log.txt`: 生成样本构建检查日志，记录额外未标注视频和未使用 fisheye 视频。

### Purpose
为成员A负责的端到端流程打地基，将平铺的原始视频文件整理成可被 Dataset、train.py 和 test.py 复用的结构化样本索引，明确用户 A/B/C、训练/测试划分、场景、意图标签和跨模态视频对应关系。

### How to Run
```bash
source /share/home/tm1078571822880000/a893873600/wyr_ml_course_project_work/software/memberA_activate.sh
python src/data/build_samples.py --config configs/default.yaml
```

### Output Files
- `data/processed/sample_index.csv`
- `results/metrics/sample_statistics.csv`
- `results/logs/sample_build_log.txt`

### Current Status
- Completed

### Notes for Report Writer
本次任务确认了 39 条可用于课程实验的完整样本，其中训练集 26 条，测试集 13 条；用户 A、B、C 各 13 条；场景分布为 office 19 条、museum 20 条。该索引是后续端到端训练和测试的数据入口。

### Remaining Problems
- 原始目录中存在 3 个额外 HoloLens mp4：`interaction_20260131_071918.mp4`、`interaction_20260131_092133.mp4`、`interaction_20260227_124133.mp4`，当前缺少标签和映射，暂不纳入训练/测试。
- 原始目录中存在 3 个未使用 fisheye avi：`Video_20260131_151916937.avi`、`Video_20260131_172143627.avi`、`Video_20260227_204121788.avi`，当前缺少对应标注样本。
- 下一步需要基于 `sample_index.csv` 编写 `src/data/dataset.py`，把索引中的原始路径转换为可训练的多模态特征输入。

## 2026-06-23 Member A - sample index side-effect fix

- Observed: running `src/data/build_samples.py` could create a literal `E:\smart AR/` directory under the Linux project root.
- Cause: the first version imported `AVI_TO_MP4_MAP` from `src/modules/real_scene_utils.py`; that legacy module creates a Windows-style HuggingFace cache directory at import time.
- Fix: inlined the fisheye-to-HoloLens mapping inside `src/data/build_samples.py` and removed the import-time dependency on `real_scene_utils.py`.
- Verified: `python src/data/build_samples.py --config configs/default.yaml` still builds 39 usable samples, with 26 train and 13 test samples.
- Cleanup: removed the generated `E:\smart AR/` directory from the project root.


## 2026-06-23 Member A - lightweight Dataset/DataLoader foundation

- Added `src/data/dataset.py` with `MultimodalIntentDataset`, `read_sample_index`, `build_dataloaders`, and label-count diagnostics.
- Added `src/data/check_dataset.py` as a smoke test for CSV reading, raw-path validation, train/test split loading, and DataLoader batching.
- Current lightweight return fields: sample metadata, intent label tensor, HoloLens path, fisheye path, and IMU path.
- Verified with `python src/data/check_dataset.py --config configs/default.yaml --batch-size 4`.
- Result: train dataset has 26 samples, test dataset has 13 samples, and a batch can be loaded successfully.
- Note: this is the stable data-entry contract for end-to-end work. The formal high-score version still needs feature extraction/caching for HoloLens video, fisheye video, and IMU windows.


## 2026-06-23 Member A - formal Dataset feature inputs

- Added `src/data/features.py` for formal input tensors:
  - HoloLens video: uniform frame sampling, cached as `T,C,H,W` tensors.
  - Fisheye video: uniform frame sampling, cached as `T,C,H,W` tensors.
  - IMU: fixed `10 x 12` numeric tensor from the available IMU CSV.
  - Scene: two-value one-hot tensor for `office` and `museum`.
- Updated `src/data/dataset.py` so `MultimodalIntentDataset(load_features=True)` returns `features` in addition to the lightweight metadata/path contract.
- Updated `src/data/check_dataset.py` with `--load-features`, `--num-video-frames`, `--image-size`, `--imu-steps`, and `--rebuild-cache`.
- Added `src/data/build_features.py` to precompute feature caches before training.
- Verified default feature mode with `python src/data/check_dataset.py --config configs/default.yaml --batch-size 2 --load-features`.
- Verified shapes: HoloLens `(batch, 8, 3, 112, 112)`, fisheye `(batch, 8, 3, 112, 112)`, IMU `(batch, 10, 12)`, scene `(batch, 2)`.
- Precomputed caches for all 39 samples. Cache count is 78 `.pt` files, matching 39 samples times 2 video streams.
- Important limitation: the raw IMU CSV timestamps do not currently align with the 2026 video filenames, so IMU is marked as `global_unaligned_sequence`. For the high-score version, replace this with per-sample aligned IMU windows or recovered processed IMU feature files.


## 2026-06-23 Member A - formal Dataset end-to-end training baseline

- Added `experiments/train_formal_dataset.py` as a new end-to-end baseline that consumes `MultimodalIntentDataset(load_features=True)`.
- Model inputs: HoloLens frame tensors, fisheye frame tensors, IMU tensors, and scene one-hot tensors.
- Model structure: compact CNN encoders for both video streams, MLP encoder for IMU, MLP encoder for scene, and a 6-class intent classifier.
- Outputs:
  - `results/metrics/formal_dataset_metrics.json`
  - `results/predictions/formal_dataset_predictions.csv`
  - `models/formal_dataset_baseline.pt`
  - `results/logs/formal_dataset_train.log`
- Verified with `python experiments/train_formal_dataset.py --config configs/default.yaml --epochs 2 --batch-size 4`.
- Smoke-run result: final test accuracy is `0.1538`. This confirms the full data/model/evaluation pipeline works, but it is not yet a competitive result.
- Next high-score work: improve video features with pretrained ViT/CLIP caches, recover or align per-sample IMU features, add validation-based model selection, and run real hyperparameter tuning.


## 2026-06-23 Member A - standard train/test wrapper alignment

- Added `experiments/train.py` to match the expected stage-2 project interface.
- Added `experiments/test.py` to load a checkpoint and run test-set evaluation independently.
- `experiments/train.py` now performs:
  - read config
  - build DataLoaders from `MultimodalIntentDataset(load_features=True)`
  - use cached/raw-derived features
  - train the clean baseline model
  - save checkpoints, metrics, placeholders, and logs
- `experiments/test.py` now performs:
  - read config
  - load `results/checkpoints/best.pt`
  - rebuild the test DataLoader with checkpoint feature settings
  - output metrics and predictions
- Verified train command:
  ```bash
  python experiments/train.py --config configs/default.yaml --epochs 1 --batch-size 4
  ```
- Verified test command:
  ```bash
  python experiments/test.py --config configs/default.yaml --checkpoint results/checkpoints/best.pt --batch-size 4
  ```
- Standard outputs now include:
  - `results/checkpoints/best.pt`
  - `results/checkpoints/final.pt`
  - `results/checkpoints/scalers.pkl`
  - `results/checkpoints/label_encoder.pkl`
  - `results/metrics/clean_baseline_metrics.csv`
  - `results/metrics/clean_baseline_summary.json`
  - `results/metrics/clean_baseline_test_metrics.csv`
  - `results/metrics/clean_baseline_test_metrics.json`
  - `results/predictions/clean_baseline_predictions.csv`
  - `results/logs/train_clean.log`
  - `results/logs/test_clean.log`
- Smoke-run result: best/test accuracy is still `0.1538`; this is for interface verification, not final performance.
- Note: `scalers.pkl` is currently a placeholder because the raw-tensor baseline does not use sklearn scalers. It is present to match the expected project output layout and can be replaced by Member B if feature standardization is added.


## 2026-06-23 Member A - architecture cleanup before GitHub handoff

- Kept `experiments/train_and_test.py` as a legacy/original reference. It should not be deleted before handoff because Member B may need it for comparison with the original baseline.
- Refactored the new clean baseline to reduce duplicate experiment code:
  - `src/models/formal_baseline.py`: compact multimodal baseline model and intent names.
  - `src/training/engine.py`: shared `train_one_epoch`, `evaluate`, and prediction CSV writer.
- Updated standard entry scripts:
  - `experiments/train.py`: standard clean-data training entry.
  - `experiments/test.py`: standard checkpoint test entry.
- Changed `experiments/train_formal_dataset.py` into a compatibility wrapper that delegates to `experiments/train.py`.
- Added `experiments/README.md` to clarify which experiment scripts are standard, compatibility, or legacy.
- Verified after refactor:
  ```bash
  python -m py_compile src/models/formal_baseline.py src/training/engine.py experiments/train.py experiments/test.py experiments/train_formal_dataset.py
  python experiments/train.py --config configs/default.yaml --epochs 1 --batch-size 4
  python experiments/test.py --config configs/default.yaml --checkpoint results/checkpoints/best.pt --batch-size 4
  ```
- Verification result: train/test both run successfully and generate the expected clean-baseline outputs under `results/checkpoints`, `results/metrics`, `results/predictions`, and `results/logs`.

