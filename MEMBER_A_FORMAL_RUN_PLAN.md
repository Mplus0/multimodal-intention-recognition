# 成员 A 正式训练与运行方案

适用仓库：`multimodal-intention-recognition`  
服务器项目路径：`/share/home/tm1078571822880000/a1086482920/Mplus0/multimodal-intention-recognition`  
Conda 环境：`/share/home/tm1078571822880000/a1086482920/Mplus0/software/miniconda3/envs/machine_learning`  
日期：2026-06-25

## 1. 当前目标与课程要求对应关系

本方案用于完成成员 A 当前负责的正式运行任务，并为后续成员 B 实验和报告撰写提供可复用输出。

课程项目核心要求包括：

1. 使用用户 A、用户 B 的交互数据作为训练集，使用用户 C 的交互数据作为测试集。
2. 将老师原始“先提取特征、保存特征、再读取特征训练”的代码重构为端到端流程：

```text
raw data -> 数据预处理 / 特征提取 -> 多模态融合模型 -> 意图类别输出
```

3. 构建模态噪声 baseline：分别对每个单模态加入 20%、40%、60% 噪声，训练并在用户 C 测试。
4. 构建模态缺失 baseline：分别丢弃任意一个模态和任意两个模态，训练并在用户 C 测试。
5. 模型改进 / 创新模块：当前代码仓库暂未实现，后续单独添加；本运行方案暂不覆盖该部分。

当前代码已经实现：

- 成员 A clean 端到端主线。
- raw data 自动生成五模态 clean source feature。
- clean baseline 训练与 user_C 测试。
- 成员 B 的 noise baseline 和 missing-modality baseline 运行入口。
- 暂未实现 improved model。

## 2. 当前代码实现链路

### 2.1 成员 A 端到端 clean 主线

正式链路为：

```text
configs/default.yaml
-> src/data/build_samples.py
-> data/processed/sample_index.json
-> src/data/build_features.py --build-missing
-> src/data/raw_feature_builder.py
-> src/data/raw_extractors/*
-> data/processed/cache/feature_cache/*
-> src/data/features.py
-> src/data/dataset.py
-> experiments/train.py
-> experiments/test.py
```

五个正式模态 key 固定为：

```text
imu
gesture
audio
text
scene
```

五模态特征形状约定为：

```text
imu:     (N, 10, 12)
gesture: (N, 10, 768)
audio:   (N, 10, 39)
text:    (N, 10, 384)
scene:   (N, 768)
```

### 2.2 raw extractor 来源

| 模态 | 当前封装文件 | 老师源代码参考 | 输出 |
|---|---|---|---|
| gesture | `src/data/raw_extractors/gesture_extractor.py` | `get_timestamp.py`, `strong_gesture2.0.py` | `strong_gesture_features_*.npy`, `metadata_strong_gesture_*.npy` |
| audio | `src/data/raw_extractors/audio_extractor.py` | `mfcc.py` | `audio_features_*.npy` |
| imu | `src/data/raw_extractors/imu_extractor.py` | `imu.py` | `imu_features_*.npy` |
| text | `src/data/raw_extractors/text_extractor.py` | `ASR.py` | `text_features_*.npy` |
| scene | `src/data/raw_extractors/scene_extractor.py` | `real_scene_utils.py` | `scene_features_*.npy` |

### 2.3 成员 B baseline 主线

噪声和缺失模态实验复用成员 A 的 Dataset、feature cache、baseline model 和 training engine：

```text
experiments/run_clean_baseline.py
experiments/run_noise_baseline.py
experiments/run_missing_baseline.py
src/data/transforms.py
src/training/experiment_runner.py
```

注意：当前噪声 baseline 是 feature-level noise baseline。课程文字描述是 raw data 噪声；若后续需要严格 raw-level noise，需要继续扩展 raw-level hook。

## 3. 正式运行前环境确认

进入项目目录：

```bash
cd /share/home/tm1078571822880000/a1086482920/Mplus0/multimodal-intention-recognition
```

设置 Python 路径变量，后续命令更短：

```bash
export PY=/share/home/tm1078571822880000/a1086482920/Mplus0/software/miniconda3/envs/machine_learning/bin/python
```

确认 Python 环境：

```bash
$PY -V
$PY -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

期望：

```text
Python 3.10.x
torch >= 2.6.0
torch.cuda.is_available() = True
```

确认关键依赖：

```bash
$PY -c "import numpy, pandas, scipy, sklearn, torch, cv2, transformers, librosa, whisper, sentence_transformers, pypinyin, webrtcvad, mediapipe, yaml; print('deps ok')"
```

确认 `moviepy.editor` 可用：

```bash
$PY -c "from moviepy.editor import VideoFileClip; import PIL; print('moviepy ok'); print(PIL.__version__)"
```

确认 ffmpeg：

```bash
which ffmpeg
ffmpeg -version | head
```

如果没有 ffmpeg：

```bash
/share/home/tm1078571822880000/a1086482920/Mplus0/software/miniconda3/bin/conda install -n machine_learning -c conda-forge ffmpeg -y
```

## 4. 正式运行前模型路径确认

确认 `configs/default.yaml` 中模型路径：

```bash
grep -n "local_models\|sentence_model\|clip_teacher_model\|vit_model" configs/default.yaml
```

确认本地模型目录存在：

```bash
ls -ld \
  data/raw/models/all-MiniLM-L6-v2 \
  data/raw/models/clip_teacher_model \
  data/raw/models/clip_teacher_model/vit-base-patch16-224
```

确认 CLIP 和 ViT 模型可加载：

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

说明：

- `CLIPVisionModel` 加载 CLIP 根目录时出现 text 分支权重 `UNEXPECTED` 可以忽略。
- `ViTModel` 加载分类模型时出现 `classifier.weight` / `classifier.bias` 为 `UNEXPECTED` 可以忽略。
- 关键是最后输出：

```text
clip_ok
vit_ok
```

## 5. 正式运行步骤

### 5.1 生成样本索引

运行：

```bash
$PY src/data/build_samples.py --config configs/default.yaml --output data/processed/sample_index.json
```

期望输出：

```text
[sample_summary]
  sample_count: 39
  split_counts: {'train': 26, 'test': 13}
  user_counts: {'user_A': 13, 'user_B': 13, 'user_C': 13}
```

输出文件：

```text
data/processed/sample_index.json
```

需要截图：

- 终端中 `[sample_summary]` 的完整输出。
- 终端中 `[saved] data/processed/sample_index.json` 的输出。

### 5.2 单样本端到端特征构建检查

运行：

```bash
$PY src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing --limit 1
```

这一步会从 raw data 自动构建第一个视频级 sample 的五模态特征。  
如果已经跑过，缓存存在时会复用已有输出。

期望输出形状类似：

```text
[actual_shapes]
  imu: (N, 10, 12)
  gesture: (N, 10, 768)
  audio: (N, 10, 39)
  text: (N, 10, 384)
  scene: (N, 768)
```

注意：`N` 是该视频内有效动作片段数，不一定是 1。

需要截图：

- CLIP / ViT 模型加载成功附近输出。
- `[actual_shapes]` 五模态形状输出。
- 若出现 `UNEXPECTED` 权重提示，截图时不用避开，但报告中不需要重点解释。

### 5.3 小规模多样本特征构建检查

建议运行：

```bash
$PY src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing --limit 3
```

目的：

- 检查不止一个视频样本。
- 提前暴露个别视频音频、时间戳、fisheye 对齐问题。

需要截图：

- 命令开始位置。
- 最后的 `[actual_shapes]` 输出。

### 5.4 全量五模态特征预构建

运行：

```bash
$PY src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing --limit 100000
```

这一步是正式训练前最重要的预处理步骤。它会为全部样本构建或复用 clean 五模态 source feature 与完整 FeatureCache。

主要输出目录：

```text
data/processed/cache/feature_cache/
data/processed/cache/feature_cache/imu_features/
data/processed/cache/feature_cache/strong_gesture_features/
data/processed/cache/feature_cache/audio_features/
data/processed/cache/feature_cache/text_features/
data/processed/cache/feature_cache/scene_features/
```

需要截图：

- 全量命令开始。
- 结束时 `[actual_shapes]` 输出。
- 如终端太长，至少截最后成功输出。

建议运行后检查文件数量：

```bash
find data/processed/cache/feature_cache/imu_features -name 'imu_features_*.npy' | wc -l
find data/processed/cache/feature_cache/strong_gesture_features -name 'strong_gesture_features_*.npy' | wc -l
find data/processed/cache/feature_cache/audio_features -name 'audio_features_*.npy' | wc -l
find data/processed/cache/feature_cache/text_features -name 'text_features_*.npy' | wc -l
find data/processed/cache/feature_cache/scene_features -name 'scene_features_*.npy' | wc -l
```

需要截图：

- 上述五个 `wc -l` 数量。

### 5.5 正式 clean baseline 训练

运行：

```bash
$PY experiments/train.py --config configs/default.yaml
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

期望终端出现：

```text
[train_done]
```

需要截图：

- 训练命令开始。
- 若可见 epoch 日志，截取若干 epoch 输出。
- 最后的 `[train_done]` 输出。
- 输出路径列表。
- metrics 字典。

注意：

- 不要使用 `--smoke-test` 作为正式结果。
- metrics 中 `status` 应为 `completed`，不能是 `smoke_test`。

### 5.6 正式 user_C 测试

运行：

```bash
$PY experiments/test.py --config configs/default.yaml --checkpoint results/checkpoints/best.pt
```

输出文件：

```text
results/metrics/clean_baseline_test_metrics.csv
results/metrics/clean_baseline_test_summary.json
results/predictions/clean_baseline_predictions.csv
results/logs/clean_baseline.log
figures/confusion_matrix.png
```

期望终端出现：

```text
[test_done]
```

需要截图：

- 测试命令开始。
- `[test_done]` 输出。
- metrics 字典。
- 输出路径列表。

运行后检查 metrics：

```bash
cat results/metrics/clean_baseline_test_metrics.csv
```

需要截图：

- `clean_baseline_test_metrics.csv` 的内容。

## 6. 噪声 baseline 正式运行

当前代码提供 5 个模态 × 3 个噪声比例，共 15 组实验。

配置文件：

```text
configs/noise.yaml
```

运行：

```bash
$PY experiments/run_noise_baseline.py --config configs/noise.yaml --base-config configs/default.yaml
```

如果想先短跑检查：

```bash
$PY experiments/run_noise_baseline.py --config configs/noise.yaml --base-config configs/default.yaml --max-samples 6 --epochs 1
```

正式运行不要加 `--smoke-test`。

主要输出：

```text
results/metrics/noise_baseline_metrics.csv
results/logs/modal_noise_baseline*.log
results/predictions/modal_noise_baseline_*_predictions.csv
results/checkpoints/modal_noise_baseline_*_best.pt
results/checkpoints/modal_noise_baseline_*_final.pt
figures/modal_noise_baseline_*_loss_curve.png
figures/modal_noise_baseline_*_confusion_matrix.png
```

需要截图：

- 命令开始。
- 每组实验开始时的日志或终端输出。
- 最后的：

```text
[noise_baseline_done]
```

- 聚合 metrics 路径。
- `cat results/metrics/noise_baseline_metrics.csv | head`。
- 如果终端可读，截图完整 15 行结果表。

## 7. 缺失模态 baseline 正式运行

当前代码提供：

- 5 个单模态缺失组合。
- 10 个双模态缺失组合。
- 共 15 组实验。

配置文件：

```text
configs/missing_modality.yaml
```

运行：

```bash
$PY experiments/run_missing_baseline.py --config configs/missing_modality.yaml --base-config configs/default.yaml
```

如果想先短跑检查：

```bash
$PY experiments/run_missing_baseline.py --config configs/missing_modality.yaml --base-config configs/default.yaml --max-samples 6 --epochs 1
```

正式运行不要加 `--smoke-test`。

主要输出：

```text
results/metrics/missing_modality_metrics.csv
results/logs/missing_modality_baseline*.log
results/predictions/missing_modality_baseline_*_predictions.csv
results/checkpoints/missing_modality_baseline_*_best.pt
results/checkpoints/missing_modality_baseline_*_final.pt
figures/missing_modality_baseline_*_loss_curve.png
figures/missing_modality_baseline_*_confusion_matrix.png
```

需要截图：

- 命令开始。
- 每组实验开始时的日志或终端输出。
- 最后的：

```text
[missing_modality_baseline_done]
```

- 聚合 metrics 路径。
- `cat results/metrics/missing_modality_metrics.csv | head`。
- 如果终端可读，截图完整 15 行结果表。

## 8. 正式运行后必须检查的文件

运行完成后检查：

```bash
find results/metrics -maxdepth 1 -type f | sort
find results/logs -maxdepth 1 -type f | sort | head
find results/predictions -maxdepth 1 -type f | sort | head
find results/checkpoints -maxdepth 1 -type f | sort | head
find figures -maxdepth 1 -type f | sort | head
```

至少应包含：

```text
results/metrics/clean_baseline_metrics.csv
results/metrics/clean_baseline_test_metrics.csv
results/metrics/noise_baseline_metrics.csv
results/metrics/missing_modality_metrics.csv
results/predictions/clean_baseline_predictions.csv
results/checkpoints/best.pt
results/checkpoints/final.pt
figures/loss_curve.png
figures/confusion_matrix.png
```

需要截图：

- 上述 `find` 命令输出。
- metrics 文件列表。
- checkpoints 文件列表。
- figures 文件列表。

## 9. 报告建议保存的截图清单

建议在 `report/screenshots/` 中保存以下截图，文件名可以按顺序命名：

```text
01_environment_python_torch_cuda.png
02_model_paths_clip_vit_ok.png
03_sample_index_summary.png
04_build_features_limit1_shapes.png
05_build_features_full_shapes.png
06_feature_cache_file_counts.png
07_train_clean_baseline_done.png
08_test_user_c_done.png
09_clean_baseline_metrics_csv.png
10_noise_baseline_done.png
11_noise_baseline_metrics_csv.png
12_missing_modality_done.png
13_missing_modality_metrics_csv.png
14_results_file_tree.png
```

截图内容不要只截命令，要尽量包含：

- 当前项目路径。
- 执行命令。
- 成功标志。
- 输出文件路径。
- metrics 或 shapes。

## 10. 不应写入正式报告的内容

以下内容不能作为正式实验结果：

```text
--smoke-test 结果
--max-samples 短跑结果
只跑 --limit 1 或 --limit 3 的特征检查结果
报错中断的实验输出
手工填写或猜测的 metrics
```

可以在报告或附录中说明：

```text
先使用 --limit 1 / --limit 3 检查端到端特征构建链路，随后执行全量特征构建与正式训练测试。
```

## 11. 当前未完成部分

当前代码仓库暂未实现：

```text
experiments/run_improved_model.py
src/models/improved_model.py
configs/improved_model.yaml
```

因此本方案暂不包含 improved model、新模块或新损失项的正式运行命令。

后续添加模型改进后，应补充：

```bash
$PY experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml
```

并新增对应输出：

```text
results/metrics/improved_model_metrics.csv
results/logs/improved_model*.log
results/predictions/improved_model*_predictions.csv
figures/improved_model*_confusion_matrix.png
```

## 12. 推荐正式运行总命令顺序

如果环境已经确认无误，正式推荐顺序如下：

```bash
cd /share/home/tm1078571822880000/a1086482920/Mplus0/multimodal-intention-recognition
export PY=/share/home/tm1078571822880000/a1086482920/Mplus0/software/miniconda3/envs/machine_learning/bin/python

$PY src/data/build_samples.py --config configs/default.yaml --output data/processed/sample_index.json

$PY src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing --limit 1

$PY src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing --limit 3

$PY src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing --limit 100000

$PY experiments/train.py --config configs/default.yaml

$PY experiments/test.py --config configs/default.yaml --checkpoint results/checkpoints/best.pt

$PY experiments/run_noise_baseline.py --config configs/noise.yaml --base-config configs/default.yaml

$PY experiments/run_missing_baseline.py --config configs/missing_modality.yaml --base-config configs/default.yaml
```

全部完成后，再统一检查：

```bash
find results/metrics -maxdepth 1 -type f | sort
find results/predictions -maxdepth 1 -type f | sort | head
find results/checkpoints -maxdepth 1 -type f | sort | head
find figures -maxdepth 1 -type f | sort | head
```

