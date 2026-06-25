# 课程项目全流程服务器运行与截图保存文档

适用仓库：`multimodal-intention-recognition`  
服务器项目路径：`/share/home/tm1078571822880000/a1086482920/Mplus0/multimodal-intention-recognition`  
Conda 环境：`/share/home/tm1078571822880000/a1086482920/Mplus0/software/miniconda3/envs/machine_learning`  
运行日期：2026-06-25 起  
目标：完整跑通课程项目从环境检查、端到端特征构建、baseline 实验、模态噪声实验、模态缺失实验到 improved model 实验的全流程，并保存报告和答辩需要的终端输出、指标文件和截图证据。

---

## 0. 重要原则

1. 正式报告结果必须来自完整数据集正式运行，不能使用 `--smoke-test`、`--max-samples`、`--limit 1` 或 `--limit 3` 的结果。
2. `--smoke-test` 只用于检查代码逻辑。
3. `--max-samples` 只用于真实数据短跑调试。
4. `--limit 1`、`--limit 3` 只用于端到端特征构建检查。
5. 正式训练结果应以 `results/metrics/*.csv`、`results/logs/*.log`、`results/predictions/*.csv`、`results/checkpoints/*.pt` 和 `figures/*.png` 为准。
6. 报告截图应保存到：

```text
report/screenshots/
```

7. 终端文本记录建议保存到：

```text
report/terminal_logs/
```

如果目录不存在，先创建：

```bash
mkdir -p report/screenshots report/terminal_logs
```

---

## 1. 课程要求与当前代码对应关系

课程项目要求：

| 课程要求 | 当前运行内容 |
|---|---|
| 用户 A、用户 B 作为训练集，用户 C 作为测试集 | `configs/default.yaml` 中配置，`src/data/build_samples.py` 生成样本索引 |
| 端到端流程：raw data -> 特征处理 -> 多模态模型 -> 意图输出 | `src/data/raw_feature_builder.py`、`src/data/raw_extractors/*`、`src/data/features.py`、`src/data/dataset.py`、`experiments/train.py`、`experiments/test.py` |
| clean baseline | `experiments/run_clean_baseline.py` 或 `experiments/train.py` + `experiments/test.py` |
| 单模态噪声 baseline | `experiments/run_noise_baseline.py` |
| 单/双模态缺失 baseline | `experiments/run_missing_baseline.py` |
| 模型优化 / 新模块 | `src/models/improved_model.py`、`src/data/improved_transforms.py`、`experiments/run_improved_model.py` |

五个正式模态固定为：

```text
imu
gesture
audio
text
scene
```

五模态特征形状约定：

```text
imu:     (N, 10, 12)
gesture: (N, 10, 768)
audio:   (N, 10, 39)
text:    (N, 10, 384)
scene:   (N, 768)
```

---

## 2. 推荐保存终端输出的方式

为了后续写报告和排查问题，建议关键命令都使用 `tee` 保存一份终端文本。

示例：

```bash
python xxx.py 2>&1 | tee report/terminal_logs/xxx.log
```

说明：

- `2>&1`：同时保存标准输出和报错输出。
- `tee`：终端显示的同时保存到文件。
- 截图时优先截“命令 + 成功标志 + 关键输出路径 + metrics”。

---

## 3. 进入服务器项目与设置 Python

进入项目目录：

```bash
cd /share/home/tm1078571822880000/a1086482920/Mplus0/multimodal-intention-recognition
```

激活环境：

```bash
conda activate machine_learning
```

设置 Python 变量：

```bash
export PY=/share/home/tm1078571822880000/a1086482920/Mplus0/software/miniconda3/envs/machine_learning/bin/python
```

创建截图和终端日志目录：

```bash
mkdir -p report/screenshots report/terminal_logs
```

需要截图：

- 当前路径。
- Conda 环境名为 `machine_learning`。
- `echo $PY` 输出。

建议保存终端文本：

```bash
pwd | tee report/terminal_logs/00_project_path.log
echo $PY | tee report/terminal_logs/00_python_path.log
```

---

## 4. 代码版本确认

查看当前分支和状态：

```bash
git branch --show-current
git status --short
git log --oneline --decorate -5
```

建议保存：

```bash
{
  git branch --show-current
  git status --short
  git log --oneline --decorate -5
} 2>&1 | tee report/terminal_logs/01_git_status.log
```

需要截图：

- 当前分支。
- 最近 commit。
- 工作区是否有未提交修改。

如果 `git pull` 出现 divergent branches，不要直接 `reset --hard`。先保存状态，再决定使用 `git pull --rebase origin mix` 或其他策略。

---

## 5. 环境与依赖检查

### 5.1 Python / PyTorch / CUDA

运行：

```bash
$PY -V
$PY -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

建议保存：

```bash
{
  $PY -V
  $PY -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
} 2>&1 | tee report/terminal_logs/02_python_torch_cuda.log
```

期望看到：

```text
Python 3.10.x
torch >= 2.x
torch.cuda.is_available() = True
GPU 名称
```

需要截图：

- Python 版本。
- Torch 版本。
- CUDA 可用状态。
- GPU 名称。

### 5.2 关键依赖

运行：

```bash
$PY -c "import numpy, pandas, scipy, sklearn, torch, cv2, transformers, librosa, whisper, sentence_transformers, pypinyin, webrtcvad, mediapipe, yaml; print('deps ok')"
```

建议保存：

```bash
$PY -c "import numpy, pandas, scipy, sklearn, torch, cv2, transformers, librosa, whisper, sentence_transformers, pypinyin, webrtcvad, mediapipe, yaml; print('deps ok')" 2>&1 | tee report/terminal_logs/03_dependencies.log
```

需要截图：

- `deps ok`。

### 5.3 moviepy 与 ffmpeg

运行：

```bash
$PY -c "from moviepy.editor import VideoFileClip; import PIL; print('moviepy ok'); print(PIL.__version__)"
which ffmpeg
ffmpeg -version | head
```

建议保存：

```bash
{
  $PY -c "from moviepy.editor import VideoFileClip; import PIL; print('moviepy ok'); print(PIL.__version__)"
  which ffmpeg
  ffmpeg -version | head
} 2>&1 | tee report/terminal_logs/04_moviepy_ffmpeg.log
```

需要截图：

- `moviepy ok`。
- `ffmpeg` 路径。
- `ffmpeg` 版本。

---

## 6. 数据与本地模型路径确认

### 6.1 检查配置路径

运行：

```bash
grep -n "raw_dir\|imu_csv\|user_a\|user_b\|user_c\|local_models\|sentence_model\|clip_teacher_model\|vit_model" configs/default.yaml
```

建议保存：

```bash
grep -n "raw_dir\|imu_csv\|user_a\|user_b\|user_c\|local_models\|sentence_model\|clip_teacher_model\|vit_model" configs/default.yaml 2>&1 | tee report/terminal_logs/05_default_config_paths.log
```

### 6.2 检查数据目录

运行：

```bash
ls -ld data/raw
ls -ld data/raw/user_A data/raw/user_B data/raw/user_C
ls -ld data/raw/user_A/HoloLens data/raw/user_A/fisheye
ls -ld data/raw/user_B/HoloLens data/raw/user_B/fisheye
ls -ld data/raw/user_C/HoloLens data/raw/user_C/fisheye
ls -l data/raw/imu.csv
```

建议保存：

```bash
{
  ls -ld data/raw
  ls -ld data/raw/user_A data/raw/user_B data/raw/user_C
  ls -ld data/raw/user_A/HoloLens data/raw/user_A/fisheye
  ls -ld data/raw/user_B/HoloLens data/raw/user_B/fisheye
  ls -ld data/raw/user_C/HoloLens data/raw/user_C/fisheye
  ls -l data/raw/imu.csv
} 2>&1 | tee report/terminal_logs/06_raw_data_paths.log
```

需要截图：

- 用户 A/B/C 原始数据目录存在。
- `imu.csv` 存在。

### 6.3 检查本地模型

运行：

```bash
ls -ld \
  data/raw/models/all-MiniLM-L6-v2 \
  data/raw/models/clip_teacher_model \
  data/raw/models/clip_teacher_model/vit-base-patch16-224
```

建议保存：

```bash
ls -ld \
  data/raw/models/all-MiniLM-L6-v2 \
  data/raw/models/clip_teacher_model \
  data/raw/models/clip_teacher_model/vit-base-patch16-224 \
  2>&1 | tee report/terminal_logs/07_local_models_paths.log
```

### 6.4 检查 CLIP / ViT 可加载

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 $PY - <<'PY'
from transformers import CLIPImageProcessor, CLIPVisionModel, ViTImageProcessor, ViTModel

print("try_clip")
CLIPImageProcessor.from_pretrained("data/raw/models/clip_teacher_model", local_files_only=True)
CLIPVisionModel.from_pretrained("data/raw/models/clip_teacher_model", local_files_only=True)
print("clip_ok")

print("try_vit")
ViTImageProcessor.from_pretrained("data/raw/models/clip_teacher_model/vit-base-patch16-224", local_files_only=True)
ViTModel.from_pretrained("data/raw/models/clip_teacher_model/vit-base-patch16-224", local_files_only=True, add_pooling_layer=False)
print("vit_ok")
PY
```

建议保存：

```bash
PYTHONDONTWRITEBYTECODE=1 $PY - <<'PY' 2>&1 | tee report/terminal_logs/08_clip_vit_load.log
from transformers import CLIPImageProcessor, CLIPVisionModel, ViTImageProcessor, ViTModel

print("try_clip")
CLIPImageProcessor.from_pretrained("data/raw/models/clip_teacher_model", local_files_only=True)
CLIPVisionModel.from_pretrained("data/raw/models/clip_teacher_model", local_files_only=True)
print("clip_ok")

print("try_vit")
ViTImageProcessor.from_pretrained("data/raw/models/clip_teacher_model/vit-base-patch16-224", local_files_only=True)
ViTModel.from_pretrained("data/raw/models/clip_teacher_model/vit-base-patch16-224", local_files_only=True, add_pooling_layer=False)
print("vit_ok")
PY
```

需要截图：

- `clip_ok`
- `vit_ok`

说明：

- 加载时出现部分 `UNEXPECTED` 权重提示可以忽略。
- 报告中只需要说明本地模型成功加载。

---

## 7. 生成样本索引

运行：

```bash
$PY src/data/build_samples.py --config configs/default.yaml --output data/processed/sample_index.json
```

建议保存：

```bash
$PY src/data/build_samples.py --config configs/default.yaml --output data/processed/sample_index.json 2>&1 | tee report/terminal_logs/09_build_samples.log
```

期望输出：

```text
[sample_summary]
  sample_count: 39
  split_counts: {'train': 26, 'test': 13}
  user_counts: {'user_A': 13, 'user_B': 13, 'user_C': 13}
[saved] data/processed/sample_index.json
```

需要截图：

- `[sample_summary]`
- `sample_count: 39`
- `train: 26`
- `test: 13`
- `[saved]`

输出文件：

```text
data/processed/sample_index.json
```

---

## 8. 端到端特征构建检查

### 8.1 单样本检查

运行：

```bash
$PY src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing --limit 1
```

建议保存：

```bash
$PY src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing --limit 1 2>&1 | tee report/terminal_logs/10_build_features_limit1.log
```

期望输出：

```text
[actual_shapes]
  imu: (N, 10, 12)
  gesture: (N, 10, 768)
  audio: (N, 10, 39)
  text: (N, 10, 384)
  scene: (N, 768)
```

需要截图：

- 命令。
- `[actual_shapes]`
- 五模态形状。

### 8.2 小规模多样本检查

运行：

```bash
$PY src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing --limit 3
```

建议保存：

```bash
$PY src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing --limit 3 2>&1 | tee report/terminal_logs/11_build_features_limit3.log
```

需要截图：

- 命令。
- 最后的 `[actual_shapes]`。

### 8.3 全量特征构建

运行：

```bash
$PY src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing --limit 100000
```

建议保存：

```bash
$PY src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing --limit 100000 2>&1 | tee report/terminal_logs/12_build_features_full.log
```

需要截图：

- 全量命令开始。
- 结束时 `[actual_shapes]`。
- 如果终端太长，至少截最后成功输出。

### 8.4 检查特征缓存数量

运行：

```bash
find data/processed/cache/feature_cache/imu_features -name 'imu_features_*.npy' | wc -l
find data/processed/cache/feature_cache/strong_gesture_features -name 'strong_gesture_features_*.npy' | wc -l
find data/processed/cache/feature_cache/audio_features -name 'audio_features_*.npy' | wc -l
find data/processed/cache/feature_cache/text_features -name 'text_features_*.npy' | wc -l
find data/processed/cache/feature_cache/scene_features -name 'scene_features_*.npy' | wc -l
```

建议保存：

```bash
{
  find data/processed/cache/feature_cache/imu_features -name 'imu_features_*.npy' | wc -l
  find data/processed/cache/feature_cache/strong_gesture_features -name 'strong_gesture_features_*.npy' | wc -l
  find data/processed/cache/feature_cache/audio_features -name 'audio_features_*.npy' | wc -l
  find data/processed/cache/feature_cache/text_features -name 'text_features_*.npy' | wc -l
  find data/processed/cache/feature_cache/scene_features -name 'scene_features_*.npy' | wc -l
} 2>&1 | tee report/terminal_logs/13_feature_cache_counts.log
```

需要截图：

- 五个模态缓存文件数量。

---

## 9. clean baseline 正式训练与测试

### 9.1 clean baseline 训练

运行：

```bash
$PY experiments/train.py --config configs/default.yaml
```

建议保存：

```bash
$PY experiments/train.py --config configs/default.yaml 2>&1 | tee report/terminal_logs/14_train_clean_baseline.log
```

期望输出：

```text
[train_done]
```

输出文件：

```text
results/checkpoints/best.pt
results/checkpoints/final.pt
results/metrics/clean_baseline_metrics.csv
results/metrics/clean_baseline_summary.json
results/logs/clean_baseline.log
results/predictions/clean_baseline_predictions.csv
figures/loss_curve.png
figures/confusion_matrix.png
```

需要截图：

- 训练命令。
- epoch 日志。
- `[train_done]`
- 输出路径列表。
- metrics 字典。

### 9.2 clean baseline user_C 测试

运行：

```bash
$PY experiments/test.py --config configs/default.yaml --checkpoint results/checkpoints/best.pt
```

建议保存：

```bash
$PY experiments/test.py --config configs/default.yaml --checkpoint results/checkpoints/best.pt 2>&1 | tee report/terminal_logs/15_test_clean_baseline.log
```

期望输出：

```text
[test_done]
```

输出文件：

```text
results/metrics/clean_baseline_test_metrics.csv
results/metrics/clean_baseline_test_summary.json
results/predictions/clean_baseline_predictions.csv
results/logs/clean_baseline.log
figures/confusion_matrix.png
```

检查 metrics：

```bash
cat results/metrics/clean_baseline_test_metrics.csv
```

建议保存：

```bash
cat results/metrics/clean_baseline_test_metrics.csv 2>&1 | tee report/terminal_logs/16_clean_baseline_test_metrics.log
```

需要截图：

- `[test_done]`
- metrics 字典。
- `clean_baseline_test_metrics.csv` 内容。

---

## 10. clean baseline runner 统一入口

如果希望使用统一 runner，也可以运行：

```bash
$PY experiments/run_clean_baseline.py --config configs/default.yaml
```

建议保存：

```bash
$PY experiments/run_clean_baseline.py --config configs/default.yaml 2>&1 | tee report/terminal_logs/17_run_clean_baseline.log
```

期望输出：

```text
[clean_baseline_done]
```

注意：

- `experiments/train.py` + `experiments/test.py` 更适合展示“训练/测试拆分”的端到端主线。
- `experiments/run_clean_baseline.py` 更适合作为和 noise、missing、improved 对齐的统一 runner 入口。

---

## 11. 模态噪声 baseline 正式运行

当前配置：

```text
configs/noise.yaml
```

实验数量：

```text
5 个模态 × 3 个噪声比例 = 15 组
```

正式运行：

```bash
$PY experiments/run_noise_baseline.py --config configs/noise.yaml --base-config configs/default.yaml
```

建议保存：

```bash
$PY experiments/run_noise_baseline.py --config configs/noise.yaml --base-config configs/default.yaml 2>&1 | tee report/terminal_logs/18_run_noise_baseline.log
```

期望输出：

```text
[noise_baseline_done]
  metrics: results/metrics/noise_baseline_metrics.csv
```

输出文件：

```text
results/metrics/noise_baseline_metrics.csv
results/logs/modal_noise_baseline*.log
results/predictions/modal_noise_baseline_*_predictions.csv
results/checkpoints/modal_noise_baseline_*_best.pt
results/checkpoints/modal_noise_baseline_*_final.pt
figures/modal_noise_baseline_*_loss_curve.png
figures/modal_noise_baseline_*_confusion_matrix.png
```

检查结果：

```bash
cat results/metrics/noise_baseline_metrics.csv
```

建议保存：

```bash
cat results/metrics/noise_baseline_metrics.csv 2>&1 | tee report/terminal_logs/19_noise_baseline_metrics.log
```

需要截图：

- 命令开始。
- 每组实验开始或 epoch 输出。
- `[noise_baseline_done]`
- `noise_baseline_metrics.csv` 内容。

---

## 12. 模态缺失 baseline 正式运行

当前配置：

```text
configs/missing_modality.yaml
```

实验数量：

```text
5 个单模态缺失 + 10 个双模态缺失 = 15 组
```

正式运行：

```bash
$PY experiments/run_missing_baseline.py --config configs/missing_modality.yaml --base-config configs/default.yaml
```

建议保存：

```bash
$PY experiments/run_missing_baseline.py --config configs/missing_modality.yaml --base-config configs/default.yaml 2>&1 | tee report/terminal_logs/20_run_missing_baseline.log
```

期望输出：

```text
[missing_modality_baseline_done]
  metrics: results/metrics/missing_modality_metrics.csv
```

输出文件：

```text
results/metrics/missing_modality_metrics.csv
results/logs/missing_modality_baseline*.log
results/predictions/missing_modality_baseline_*_predictions.csv
results/checkpoints/missing_modality_baseline_*_best.pt
results/checkpoints/missing_modality_baseline_*_final.pt
figures/missing_modality_baseline_*_loss_curve.png
figures/missing_modality_baseline_*_confusion_matrix.png
```

检查结果：

```bash
cat results/metrics/missing_modality_metrics.csv
```

建议保存：

```bash
cat results/metrics/missing_modality_metrics.csv 2>&1 | tee report/terminal_logs/21_missing_baseline_metrics.log
```

需要截图：

- 命令开始。
- 每组实验开始或 epoch 输出。
- `[missing_modality_baseline_done]`
- `missing_modality_metrics.csv` 内容。

---

## 13. improved model 运行前检查

新增 improved model 文件：

```text
src/models/improved_model.py
src/data/improved_transforms.py
src/training/improved_experiment_runner.py
experiments/run_improved_model.py
configs/improved_model.yaml
```

### 13.1 improved model forward smoke test

运行：

```bash
$PY src/models/improved_model.py --config configs/default.yaml --smoke-test
```

建议保存：

```bash
$PY src/models/improved_model.py --config configs/default.yaml --smoke-test 2>&1 | tee report/terminal_logs/22_improved_model_smoke.log
```

期望输出：

```text
[improved_model_smoke_test]
  feature_keys: [...]
  intent_logits: (1, 6)
  fused: (1, 128)
  reliability_imu: (1, 1)
  reliability_gesture: (1, 1)
  reliability_audio: (1, 1)
  reliability_text: (1, 1)
  reliability_scene: (1, 1)
```

需要截图：

- `[improved_model_smoke_test]`
- `intent_logits`
- 五个 `reliability_*`

### 13.2 modality dropout smoke test

运行：

```bash
$PY src/data/improved_transforms.py
```

建议保存：

```bash
$PY src/data/improved_transforms.py 2>&1 | tee report/terminal_logs/23_improved_dropout_smoke.log
```

期望输出：

```text
[random_modality_dropout_smoke_test]
  dropped_modalities: ...
  imu: mask=...
  gesture: mask=...
  audio: mask=...
  text: mask=...
  scene: mask=...
```

需要截图：

- `[random_modality_dropout_smoke_test]`
- 至少一个模态 `mask=False` 或 `feature_sum=0.0`

### 13.3 improved runner smoke test

运行：

```bash
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean --smoke-test --epochs 1
```

建议保存：

```bash
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean --smoke-test --epochs 1 2>&1 | tee report/terminal_logs/24_improved_runner_smoke.log
```

期望输出：

```text
[improved_model_done]
  mode: clean
  run_count: 1
  metrics: .../results/metrics/improved_model_metrics.csv
```

注意：

- 该命令使用 synthetic smoke data。
- metrics 中会出现 `status=smoke_test`。
- 不能作为正式实验结果。

需要截图：

- epoch 日志。
- `final_metrics`
- `[improved_model_done]`

---

## 14. improved model 真实数据短跑

### 14.1 不推荐的短跑命令

不要使用：

```bash
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean --max-samples 6 --epochs 1
```

原因：

```text
样本索引顺序是 user_A -> user_B -> user_C。
--max-samples 6 只会截取 user_A 的前 6 个样本，只有 train，没有 user_C test。
```

会导致：

```text
Train/validation/test samples are not all available.
```

### 14.2 推荐真实数据短跑

使用：

```bash
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean --max-samples 30 --epochs 1
```

建议保存：

```bash
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean --max-samples 30 --epochs 1 2>&1 | tee report/terminal_logs/25_improved_clean_short_real.log
```

说明：

```text
--max-samples 30 会覆盖 user_A 13 + user_B 13 + user_C 前 4 个样本。
因此 train / validation / test 都存在。
```

需要截图：

- 命令。
- sample index 日志。
- epoch 日志。
- `[improved_model_done]`

该结果仍然不能写入正式报告指标，只能证明真实数据链路可运行。

---

## 15. improved model 正式运行

### 15.1 clean improved

正式运行：

```bash
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean
```

建议保存：

```bash
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean 2>&1 | tee report/terminal_logs/26_improved_clean_full.log
```

期望输出：

```text
[improved_model_done]
  mode: clean
  run_count: 1
  metrics: results/metrics/improved_model_metrics.csv
```

主要输出：

```text
results/metrics/improved_clean_metrics.csv
results/metrics/improved_model_metrics.csv
results/logs/improved_clean.log
results/predictions/improved_clean_predictions.csv
results/checkpoints/improved_clean_best.pt
results/checkpoints/improved_clean_final.pt
figures/improved_clean_loss_curve.png
figures/improved_clean_confusion_matrix.png
```

### 15.2 ablation improved

正式运行：

```bash
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode ablation
```

建议保存：

```bash
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode ablation 2>&1 | tee report/terminal_logs/27_improved_ablation_full.log
```

消融实验包括：

```text
improved_gate_only
improved_dropout_only
improved_gate_dropout
```

输出：

```text
results/metrics/improved_ablation_metrics.csv
```

检查：

```bash
cat results/metrics/improved_ablation_metrics.csv
```

建议保存：

```bash
cat results/metrics/improved_ablation_metrics.csv 2>&1 | tee report/terminal_logs/28_improved_ablation_metrics.log
```

### 15.3 noise improved

正式运行：

```bash
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode noise
```

建议保存：

```bash
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode noise 2>&1 | tee report/terminal_logs/29_improved_noise_full.log
```

实验数量：

```text
5 个模态 × 3 个噪声比例 = 15 组
```

输出：

```text
results/metrics/improved_noise_metrics.csv
```

检查：

```bash
cat results/metrics/improved_noise_metrics.csv
```

建议保存：

```bash
cat results/metrics/improved_noise_metrics.csv 2>&1 | tee report/terminal_logs/30_improved_noise_metrics.log
```

### 15.4 missing improved

正式运行：

```bash
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode missing
```

建议保存：

```bash
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode missing 2>&1 | tee report/terminal_logs/31_improved_missing_full.log
```

实验数量：

```text
5 个单模态缺失 + 10 个双模态缺失 = 15 组
```

输出：

```text
results/metrics/improved_missing_modality_metrics.csv
```

检查：

```bash
cat results/metrics/improved_missing_modality_metrics.csv
```

建议保存：

```bash
cat results/metrics/improved_missing_modality_metrics.csv 2>&1 | tee report/terminal_logs/32_improved_missing_metrics.log
```

---

## 16. 正式运行推荐总顺序

如果从零开始，推荐顺序如下：

```bash
cd /share/home/tm1078571822880000/a1086482920/Mplus0/multimodal-intention-recognition
conda activate machine_learning
export PY=/share/home/tm1078571822880000/a1086482920/Mplus0/software/miniconda3/envs/machine_learning/bin/python
mkdir -p report/screenshots report/terminal_logs

$PY src/data/build_samples.py --config configs/default.yaml --output data/processed/sample_index.json

$PY src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing --limit 1
$PY src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing --limit 3
$PY src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing --limit 100000

$PY experiments/train.py --config configs/default.yaml
$PY experiments/test.py --config configs/default.yaml --checkpoint results/checkpoints/best.pt

$PY experiments/run_clean_baseline.py --config configs/default.yaml
$PY experiments/run_noise_baseline.py --config configs/noise.yaml --base-config configs/default.yaml
$PY experiments/run_missing_baseline.py --config configs/missing_modality.yaml --base-config configs/default.yaml

$PY src/models/improved_model.py --config configs/default.yaml --smoke-test
$PY src/data/improved_transforms.py
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean --smoke-test --epochs 1
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean --max-samples 30 --epochs 1

$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode ablation
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode noise
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode missing
```

---

## 17. 运行后统一检查输出文件

运行：

```bash
find results/metrics -maxdepth 1 -type f | sort
find results/logs -maxdepth 1 -type f | sort | head -50
find results/predictions -maxdepth 1 -type f | sort | head -50
find results/checkpoints -maxdepth 1 -type f | sort | head -50
find figures -maxdepth 1 -type f | sort | head -50
```

建议保存：

```bash
{
  find results/metrics -maxdepth 1 -type f | sort
  find results/logs -maxdepth 1 -type f | sort | head -50
  find results/predictions -maxdepth 1 -type f | sort | head -50
  find results/checkpoints -maxdepth 1 -type f | sort | head -50
  find figures -maxdepth 1 -type f | sort | head -50
} 2>&1 | tee report/terminal_logs/33_final_output_tree.log
```

需要截图：

- metrics 文件列表。
- logs 文件列表。
- checkpoints 文件列表。
- figures 文件列表。

至少应包含：

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

---

## 18. 汇总查看关键 metrics

运行：

```bash
cat results/metrics/clean_baseline_test_metrics.csv
cat results/metrics/noise_baseline_metrics.csv
cat results/metrics/missing_modality_metrics.csv
cat results/metrics/improved_clean_metrics.csv
cat results/metrics/improved_ablation_metrics.csv
cat results/metrics/improved_noise_metrics.csv
cat results/metrics/improved_missing_modality_metrics.csv
```

建议保存：

```bash
{
  echo "===== clean_baseline_test_metrics.csv ====="
  cat results/metrics/clean_baseline_test_metrics.csv
  echo "===== noise_baseline_metrics.csv ====="
  cat results/metrics/noise_baseline_metrics.csv
  echo "===== missing_modality_metrics.csv ====="
  cat results/metrics/missing_modality_metrics.csv
  echo "===== improved_clean_metrics.csv ====="
  cat results/metrics/improved_clean_metrics.csv
  echo "===== improved_ablation_metrics.csv ====="
  cat results/metrics/improved_ablation_metrics.csv
  echo "===== improved_noise_metrics.csv ====="
  cat results/metrics/improved_noise_metrics.csv
  echo "===== improved_missing_modality_metrics.csv ====="
  cat results/metrics/improved_missing_modality_metrics.csv
} 2>&1 | tee report/terminal_logs/34_all_key_metrics.log
```

需要截图：

- baseline clean 指标。
- noise baseline 表格。
- missing baseline 表格。
- improved clean 指标。
- improved ablation 表格。
- improved noise 表格。
- improved missing 表格。

---

## 19. 推荐截图清单

建议按以下文件名保存到 `report/screenshots/`：

```text
01_project_path_and_conda_env.png
02_python_torch_cuda.png
03_dependencies_ok.png
04_moviepy_ffmpeg_ok.png
05_raw_data_paths.png
06_local_models_paths.png
07_clip_vit_ok.png
08_sample_index_summary.png
09_build_features_limit1_shapes.png
10_build_features_limit3_shapes.png
11_build_features_full_shapes.png
12_feature_cache_counts.png
13_train_clean_baseline_done.png
14_test_clean_baseline_done.png
15_clean_baseline_metrics_csv.png
16_run_clean_baseline_done.png
17_noise_baseline_done.png
18_noise_baseline_metrics_csv.png
19_missing_baseline_done.png
20_missing_baseline_metrics_csv.png
21_improved_model_smoke.png
22_improved_dropout_smoke.png
23_improved_runner_smoke.png
24_improved_clean_short_real.png
25_improved_clean_full_done.png
26_improved_clean_metrics_csv.png
27_improved_ablation_done.png
28_improved_ablation_metrics_csv.png
29_improved_noise_done.png
30_improved_noise_metrics_csv.png
31_improved_missing_done.png
32_improved_missing_metrics_csv.png
33_final_output_tree.png
34_all_key_metrics.png
```

截图要求：

1. 截图中尽量包含当前路径。
2. 截图中包含完整命令。
3. 截图中包含成功标志，例如 `[train_done]`、`[test_done]`、`[noise_baseline_done]`、`[missing_modality_baseline_done]`、`[improved_model_done]`。
4. 指标截图应能看到 CSV 表头和关键数值。
5. 不要只截一小块结果，尽量包含命令和输出。

---

## 20. 报告中不能使用的结果

以下内容不能作为正式实验结果：

```text
--smoke-test 结果
--max-samples 结果
--limit 1 结果
--limit 3 结果
中断或报错的实验输出
手工填写或猜测的 metrics
```

可以在报告或附录中说明：

```text
项目先使用 smoke test 和小规模样本短跑检查代码可运行性，随后使用完整训练集和测试集运行正式实验。报告中的 Accuracy、Macro F1 和 Weighted F1 均来自完整数据集正式运行结果。
```

---

## 21. 可能出现的 warning 与处理

### 21.1 CUDA deterministic warning

可能出现：

```text
Deterministic behavior was enabled ...
CuBLAS ...
```

这通常不影响运行。如果想减少 warning，可在运行前设置：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### 21.2 ViT / CLIP unexpected weights

可能出现部分权重 `UNEXPECTED` 提示，只要最后出现：

```text
clip_ok
vit_ok
```

即可认为本地模型加载成功。

### 21.3 `--max-samples 6` 无 test split

如果出现：

```text
Train/validation/test samples are not all available.
```

检查是否使用了太小的 `--max-samples`。推荐真实数据短跑使用：

```bash
--max-samples 30
```

---

## 22. 最终交付检查清单

提交或写报告前确认：

- [ ] `sample_index.json` 已生成。
- [ ] 全量五模态特征已构建。
- [ ] clean baseline 已训练。
- [ ] user_C clean baseline 已测试。
- [ ] noise baseline 15 组结果已生成。
- [ ] missing baseline 15 组结果已生成。
- [ ] improved clean 已生成。
- [ ] improved ablation 已生成。
- [ ] improved noise 15 组结果已生成。
- [ ] improved missing 15 组结果已生成。
- [ ] 所有关键 metrics CSV 已保存。
- [ ] 所有关键 logs 已保存。
- [ ] checkpoints 已保存。
- [ ] confusion matrix / loss curve 图已保存。
- [ ] `report/screenshots/` 中保存了关键运行截图。
- [ ] `report/terminal_logs/` 中保存了关键终端文本。
- [ ] 报告没有使用 smoke-test 或 max-samples 结果作为正式指标。

