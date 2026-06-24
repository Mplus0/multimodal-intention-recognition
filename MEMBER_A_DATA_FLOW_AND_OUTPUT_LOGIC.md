# 成员 A 数据读取、处理与输出逻辑说明

本文档整理当前 `repo-main` 中成员 A clean baseline 主线的代码运行逻辑，重点说明数据如何从 `data/raw/` 被读取、如何生成或复用五模态特征缓存、如何进入 Dataset 和模型，以及最终输出保存到哪些目录。

本文档只描述当前代码逻辑，不代表正在运行训练。

## 1. 总体链路

```text
configs/default.yaml
  -> src/utils/paths.py
  -> src/data/build_samples.py
  -> data/processed/sample_index.json
  -> src/data/features.py
     -> src/data/raw_feature_builder.py
        -> src/modules/feature_extraction/adapters.py
        -> data/processed/cache/feature_cache/
     -> complete sample .npz cache
  -> src/data/dataset.py
  -> src/models/formal_baseline.py
  -> src/training/engine.py
  -> experiments/train.py / experiments/test.py
  -> results/ and figures/
```

正式实验模态 key 固定为：

```text
imu, gesture, audio, text, scene
```

`HoloLens`、`fisheye`、`hololens`、`fisheye` 只作为原始数据来源或文件夹名称，不作为正式实验模态名。

## 2. 配置与路径来源

入口配置文件为：

```text
configs/default.yaml
```

路径解析由以下文件负责：

```text
src/utils/paths.py
```

当前代码要求所有路径都通过配置和项目相对路径管理，不应写硬编码绝对路径。主要路径包括：

```text
data/raw/
data/raw/imu.csv
data/raw/user_A/
data/raw/user_B/
data/raw/user_C/
data/raw/models/all-MiniLM-L6-v2
data/raw/models/clip_teacher_model
data/processed/cache/
data/processed/cache/feature_cache/
results/metrics/
results/logs/
results/predictions/
results/checkpoints/
figures/
```

## 3. 原始数据读取逻辑

样本索引由以下文件生成：

```text
src/data/build_samples.py
```

它从配置读取三个用户目录：

```text
data/raw/user_A
data/raw/user_B
data/raw/user_C
```

每个用户目录下按照老师数据结构读取：

```text
HoloLens/*.mp4
fisheye/*.avi
```

其中：

- `HoloLens/*.mp4` 作为 `audio` 和 `text` 的原始来源。
- `fisheye/*.avi` 作为 `gesture` 和 `scene` 的原始来源。
- `data/raw/imu.csv` 作为 `imu` 的原始来源。

`build_samples.py` 内部维护老师给出的：

- `TEACHER_VIDEO_LABELS`：mp4 文件名到意图类别的映射。
- `AVI_TO_MP4_MAP` / `MP4_TO_AVI_MAP`：fisheye 视频和 HoloLens 视频之间的对应关系。
- `OFFICE_VIDEO_NAMES`：区分 `office` / `museum` 场景。

训练测试划分规则：

```text
train = user_A + user_B
test  = user_C
```

## 4. sample_index.json 结构

`build_samples.py` 输出：

```text
data/processed/sample_index.json
```

每条样本包含：

```json
{
  "sample_id": "interaction_xxx",
  "video_name": "interaction_xxx.mp4",
  "user": "user_A",
  "split": "train",
  "raw_paths": {
    "imu": "data/raw/imu.csv",
    "gesture": "data/raw/user_A/fisheye/Video_xxx.avi",
    "audio": "data/raw/user_A/HoloLens/interaction_xxx.mp4",
    "text": "data/raw/user_A/HoloLens/interaction_xxx.mp4",
    "scene": "data/raw/user_A/fisheye/Video_xxx.avi"
  },
  "feature_paths": {
    "imu": "data/processed/cache/feature_cache/imu_features/imu_features_interaction_xxx.npy",
    "gesture": "data/processed/cache/feature_cache/strong_gesture_features/strong_gesture_features_interaction_xxx.npy",
    "audio": "data/processed/cache/feature_cache/audio_features/audio_features_interaction_xxx.npy",
    "text": "data/processed/cache/feature_cache/text_features/text_features_interaction_xxx.npy",
    "scene": "data/processed/cache/feature_cache/scene_features/scene_features_interaction_xxx.npy"
  },
  "feature_status": {
    "imu": false,
    "gesture": false,
    "audio": false,
    "text": false,
    "scene": false
  },
  "intent_label": 0,
  "intent_name": "menu",
  "scene_label": 0,
  "scene_name": "office",
  "joint_label": "office_menu"
}
```

说明：

- `raw_paths` 指向原始文件。
- `feature_paths` 指向五模态源特征缓存的预期位置。
- `feature_status` 表示对应源特征文件当前是否已经存在。
- 即使 `.npy` 尚未生成，`feature_paths` 也会写入，保证成员 A 和成员 B 后续复用同一套缓存路径规则。

## 5. 五模态源特征生成逻辑

缺失源特征由以下文件调度生成：

```text
src/data/raw_feature_builder.py
```

核心函数：

```python
ensure_source_features(sample, config, rebuild=False)
```

它根据 `sample["raw_paths"]` 和 `sample["feature_paths"]` 生成缺失的五模态 `.npy` 文件：

```text
data/processed/cache/feature_cache/
  imu_features/
  strong_gesture_features/
  audio_features/
  text_features/
  scene_features/
```

特征提取算法封装在：

```text
src/modules/feature_extraction/adapters.py
```

### 5.1 IMU

输入：

```text
data/raw/imu.csv
```

处理：

- 读取 CSV。
- 选择数值列。
- 均匀采样到 `target_timesteps=10`。
- 补齐或截断到 `imu=12` 维。

输出：

```text
data/processed/cache/feature_cache/imu_features/imu_features_<sample_id>.npy
```

shape：

```text
(1, 10, 12)
```

### 5.2 Gesture

输入：

```text
data/raw/user_X/fisheye/Video_xxx.avi
```

处理：

- 用 OpenCV 从 fisheye 视频中均匀取帧。
- 使用本地 `clip_teacher_model` 中的 CLIP vision model 编码帧。
- 生成 10 个时间步的视觉序列特征。
- 写入 intent label、scene label 和 approximate timestamp 信息。

输出：

```text
data/processed/cache/feature_cache/strong_gesture_features/strong_gesture_features_<sample_id>.npy
```

shape：

```text
(1, 10, 768)
```

### 5.3 Audio

输入：

```text
data/raw/user_X/HoloLens/interaction_xxx.mp4
```

处理：

- 使用 `imageio-ffmpeg` 解码 mp4 中的音频。
- 转为 16kHz、单声道、float32。
- 提取 13 维 MFCC、一阶差分和二阶差分，组成 39 维音频特征。
- 均匀采样或补齐到 10 个时间步。

输出：

```text
data/processed/cache/feature_cache/audio_features/audio_features_<sample_id>.npy
```

shape：

```text
(1, 10, 39)
```

### 5.4 Text

输入：

```text
data/raw/user_X/HoloLens/interaction_xxx.mp4
```

处理：

- 同样从 HoloLens mp4 解码音频。
- 使用 Whisper ASR 识别中文文本。
- 使用本地 `all-MiniLM-L6-v2` sentence model 对识别文本做 embedding。
- 将 384 维文本向量扩展为 10 个时间步。

输出：

```text
data/processed/cache/feature_cache/text_features/text_features_<sample_id>.npy
data/processed/cache/feature_cache/text_features/text_features_<sample_id>.json
```

shape：

```text
(1, 10, 384)
```

注意：

- 当前逻辑不会用 `unavailable` 零特征冒充正式 text 特征。
- 如果 `openai-whisper`、Whisper 模型文件或 sentence model 缺失，会明确报错。
- Whisper 模型默认缓存位置为：

```text
data/processed/cache/whisper/
```

### 5.5 Scene

输入：

```text
data/raw/user_X/fisheye/Video_xxx.avi
```

处理：

- 从 fisheye 视频中抽取少量帧。
- 使用本地 `clip_teacher_model` 编码。
- 对帧级特征取平均，得到单个 scene token。

输出：

```text
data/processed/cache/feature_cache/scene_features/scene_features_<sample_id>.npy
```

shape：

```text
(1, 768)
```

## 6. features.py 的缓存读取与归一化逻辑

五模态特征读取由以下文件负责：

```text
src/data/features.py
```

核心函数：

```python
load_or_build_sample_features(
    sample,
    config=None,
    cache=None,
    use_cache=True,
    rebuild_cache=False,
    build_missing_source_features=True,
    rebuild_source_features=False,
)
```

读取顺序：

1. 优先读取完整样本 `.npz` 缓存。
2. 如果 `.npz` 不存在，则检查五模态源 `.npy` 是否存在。
3. 如果源 `.npy` 缺失且 `build_missing_source_features=True`，调用 `raw_feature_builder.ensure_source_features(...)` 从 raw data 生成。
4. 读取五模态源 `.npy`。
5. 对序列长度和维度做规范化。
6. 保存完整样本 `.npz` 缓存，加速后续训练和成员 B 实验复用。

完整样本 `.npz` 缓存位置：

```text
data/processed/cache/feature_cache/<md5(sample_id)>.npz
```

正式维度约定：

| Modality | Shape |
|---|---|
| imu | `(N, 10, 12)` |
| gesture | `(N, 10, 768)` |
| audio | `(N, 10, 39)` |
| text | `(N, 10, 384)` |
| scene | `(N, 768)` |

## 7. Dataset 输出逻辑

Dataset 文件：

```text
src/data/dataset.py
```

类名：

```python
MultimodalIntentDataset
```

从 metadata 构建 Dataset：

```python
MultimodalIntentDataset.from_metadata_samples(...)
```

每条 `sample` 输出结构：

```python
sample = {
    "features": {
        "imu": ...,
        "gesture": ...,
        "audio": ...,
        "text": ...,
        "scene": ...
    },
    "intent_label": ...,
    "scene_label": ...,
    "joint_label": ...,
    "sample_id": ...,
    "user": ...,
    "split": ...,
    "modality_mask": ...
}
```

说明：

- `features` 中五个正式模态 key 始终存在。
- `transform=None` 是默认支持的。
- 对于 Dataset 记录层面缺失的 feature，会保留 key，并在 Dataset 内部用 zero-fill 和 `modality_mask=False` 表示。但 raw-to-feature 正式链路不会把 ASR 或音频失败伪造成正式零特征。
- 成员 B 后续 noise/missing 实验可以基于 `features` 和 `modality_mask` 做接口接入。

## 8. build_features.py 的用途

文件：

```text
src/data/build_features.py
```

主要用途：

- `--dry-run`：检查配置路径、模态 key、预期 shape。
- `--metadata-json`：读取 `sample_index.json` 并构建或检查实际特征。
- `--build-missing-source-features`：源 `.npy` 缺失时，从 raw data 自动生成。
- `--rebuild-source-features`：强制重建源 `.npy`。
- `--rebuild-cache`：忽略完整样本 `.npz` 缓存，重新读取源 `.npy`。
- `--all`：处理全部样本。
- `--limit N`：只处理前 N 个样本，适合正式训练前小规模检查。

示例命令，仅作为说明：

```bash
python src/data/build_features.py --config configs/default.yaml --dry-run
python src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing-source-features --limit 3
python src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing-source-features --all
```

## 9. train.py 的数据读取与输出逻辑

训练入口：

```text
experiments/train.py
```

主要流程：

1. 读取 `configs/default.yaml`。
2. 调用 `ensure_runtime_dirs(config)` 创建运行输出目录。
3. 调用 `feature_dry_run(config)` 打印关键路径和预期 shape。
4. 调用 `build_sample_index(config)` 生成内存中的样本索引。
5. 按 `split=train` 选取训练样本，并从训练样本末尾切出一部分作为验证集。
6. 调用 `MultimodalIntentDataset.from_metadata_samples(...)` 读取或构建特征。
7. 构建 `FormalMultimodalBaseline`。
8. 使用 `src/training/engine.py` 训练、验证、评估。
9. 保存 checkpoint、metrics、predictions、summary、log、figures。

训练入口默认会自动构建缺失源特征：

```text
build_missing_source_features=True
rebuild_source_features=False
rebuild_feature_cache=False
```

可选参数：

```text
--no-build-source-features
--rebuild-source-features
--rebuild-feature-cache
--max-samples
--epochs
--batch-size
--lr
--smoke-test
```

`--smoke-test` 会使用合成样本，只用于工程链路测试，不是正式实验。

## 10. test.py 的数据读取与输出逻辑

测试入口：

```text
experiments/test.py
```

主要流程：

1. 读取 `configs/default.yaml`。
2. 默认加载：

```text
results/checkpoints/best.pt
```

3. 调用 `build_sample_index(config)` 获取样本索引。
4. 选取 `split=test` 的 `user_C` 样本。
5. 调用 `MultimodalIntentDataset.from_metadata_samples(...)` 读取或构建特征。
6. 加载 `FormalMultimodalBaseline` 权重。
7. 调用 `evaluate(...)` 生成指标和预测。
8. 保存测试 metrics、summary、predictions 和 confusion matrix。

可选参数：

```text
--checkpoint
--batch-size
--max-samples
--smoke-test
--no-build-source-features
--rebuild-source-features
--rebuild-feature-cache
```

## 11. 输出文件整理

### 11.1 特征缓存

源特征缓存：

```text
data/processed/cache/feature_cache/imu_features/
data/processed/cache/feature_cache/strong_gesture_features/
data/processed/cache/feature_cache/audio_features/
data/processed/cache/feature_cache/text_features/
data/processed/cache/feature_cache/scene_features/
```

完整样本 `.npz` 缓存：

```text
data/processed/cache/feature_cache/<md5(sample_id)>.npz
```

Whisper 模型缓存：

```text
data/processed/cache/whisper/
```

### 11.2 训练输出

```text
results/checkpoints/best.pt
results/checkpoints/final.pt
results/metrics/clean_baseline_metrics.csv
results/metrics/clean_baseline_summary.json
results/predictions/clean_baseline_predictions.csv
results/logs/clean_baseline.log
figures/loss_curve.png
figures/confusion_matrix.png
```

### 11.3 测试输出

```text
results/metrics/clean_baseline_test_metrics.csv
results/metrics/clean_baseline_test_summary.json
results/predictions/clean_baseline_predictions.csv
results/logs/clean_baseline.log
figures/confusion_matrix.png
```

## 12. predictions 与 metrics 字段

当前训练和测试都通过 `src/training/engine.py` 保存指标和预测。

核心指标：

```text
accuracy
macro_f1
weighted_f1
training_time
avg_test_time_per_sample
```

补充指标：

```text
scene_accuracy
joint_accuracy
loss
```

预测 CSV 至少应包含：

```text
sample_id
user
split
intent_true
intent_pred
```

## 13. 错误与缓存策略

### 13.1 不伪造正式特征

音频和文本链路如果缺少依赖或模型，不会静默生成 `unavailable` 零特征。典型错误包括：

```text
CLIP model directory missing
Sentence model directory missing
openai-whisper is required for text feature extraction
No audio decoded from ...
Cannot open video ...
Raw path for modality ... does not exist
```

### 13.2 缓存复用规则

默认规则：

1. 如果完整 `.npz` 存在，优先读 `.npz`。
2. 如果 `.npz` 不存在，但源 `.npy` 存在，读取源 `.npy` 并生成 `.npz`。
3. 如果源 `.npy` 缺失，默认从 raw data 自动生成。
4. 如果加 `--no-build-source-features`，源 `.npy` 缺失会直接报错。
5. 如果加 `--rebuild-source-features`，即使源 `.npy` 已存在也重新提取。
6. 如果加 `--rebuild-feature-cache`，忽略完整 `.npz`，重新从源 `.npy` 组装。

## 14. 正式运行前建议顺序

以下命令只是建议流程，本文档生成时不执行：

```bash
python src/utils/paths.py
python src/data/build_samples.py --config configs/default.yaml --output data/processed/sample_index.json
python src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing-source-features --limit 3
python src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --build-missing-source-features --all
python experiments/train.py --config configs/default.yaml
python experiments/test.py --config configs/default.yaml --checkpoint results/checkpoints/best.pt
```

如果时间紧，也可以跳过全量预构建，直接运行 `train.py`。训练入口会在遇到缺失源特征时按需构建并复用缓存。

## 15. 当前已知限制

- 当前 text 模态依赖 Whisper ASR 和本地 sentence model；如果首次运行时 Whisper 模型尚未准备，需要先完成模型缓存。
- 1 个样本或少量样本的 smoke/quick run 只能证明工程链路，不代表正式实验性能。
- 成员 B 的 noise、missing、improved model 实验不在成员 A 当前主线中展开；成员 B 可复用 `MultimodalIntentDataset`、五模态 key、feature cache、metrics 和 predictions 输出接口。

