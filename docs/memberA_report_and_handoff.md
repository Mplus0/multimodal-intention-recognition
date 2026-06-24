# 成员 A 报告材料与交接说明

日期：2026-06-24
项目：`multimodal-intention-recognition`
负责人：成员 A

## 1. 成员 A 职责

成员 A 负责建立课程项目中的五模态端到端主线，重点是数据工程、Dataset、特征接口、clean baseline 模型和训练/测试入口。

成员 A 当前完成的主线覆盖：

1. 统一项目路径配置和输出目录。
2. 读取 user_A、user_B、user_C 的样本划分。
3. 建立五模态正式 key：`imu`、`gesture`、`audio`、`text`、`scene`。
4. 整理老师源码中合理的五模态特征处理逻辑。
5. 实现可复用的 Dataset 和 transform 接口。
6. 实现 clean baseline 模型、训练入口和测试入口。
7. 统一保存 metrics、logs、predictions、checkpoints 和 figures。
8. 为成员 B 后续 noise、missing、improved model 实验预留接口。

本交接说明只覆盖成员 A clean baseline 主线，不展开成员 B 的实验实现。

## 2. 正式模态与 raw data 来源

正式实验模态 key 固定为：

```text
imu
gesture
audio
text
scene
```

raw data 来源约定：

- `data/raw/imu.csv`：IMU 原始数据来源。
- `data/raw/user_A/HoloLens`：用户 A 的 HoloLens 视频来源，主要用于 audio/text。
- `data/raw/user_A/fisheye`：用户 A 的 fisheye 视频来源，主要用于 gesture/scene。
- `data/raw/user_B/HoloLens`、`data/raw/user_B/fisheye`：用户 B 数据来源。
- `data/raw/user_C/HoloLens`、`data/raw/user_C/fisheye`：用户 C 数据来源。

注意：`HoloLens`、`fisheye` 只作为 raw data 来源目录，不作为正式实验模态名。最终 Dataset 和模型只使用 `imu`、`gesture`、`audio`、`text`、`scene`。

## 3. 数据划分

训练集：

```text
user_A + user_B
```

测试集：

```text
user_C
```

该规则写在 `configs/default.yaml`：

```yaml
training:
  split:
    train_users:
      - user_A
      - user_B
    test_users:
      - user_C
```

成员 B 当前 `repo-mplus0/configs/noise.yaml` 和 `repo-mplus0/configs/missing_modality.yaml` 中的划分也与该规则一致。

## 4. 成员 A 交付文件

路径与配置：

```text
configs/default.yaml
src/utils/paths.py
```

数据与特征：

```text
src/data/build_samples.py
src/data/features.py
src/data/build_features.py
src/data/dataset.py
src/data/check_dataset.py
```

模型与训练：

```text
src/models/formal_baseline.py
src/training/engine.py
experiments/train.py
experiments/test.py
```

文档：

```text
README_CHINESE.md
docs/collaboration_log.md
docs/memberA_report_and_handoff.md
```

保留参考：

```text
experiments/train_and_test.py
```

该文件作为老师原始/旧版训练测试逻辑参考保留，不作为当前成员 A clean baseline 标准入口。

## 5. 端到端调用链

成员 A 当前标准链路如下：

```text
data/raw 原始数据
-> src/data/build_samples.py 构建样本索引
-> src/data/features.py / src/data/build_features.py 读取或构建五模态特征
-> src/data/dataset.py 输出 MultimodalIntentDataset
-> src/models/formal_baseline.py 五模态融合模型
-> src/training/engine.py 训练与评估
-> experiments/train.py / experiments/test.py 输出 intent prediction
```

模型主输出为：

```text
intent_logits
```

可选辅助输出：

```text
scene_logits
joint_logits
```

## 6. Dataset 接口

`MultimodalIntentDataset` 支持：

```python
MultimodalIntentDataset(records, transform=None, config=config)
```

标准 sample 输出：

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
    "modality_mask": ...,
}
```

缺失模态处理：

- 五个模态 key 始终存在。
- 缺失模态使用 zero-fill。
- `modality_mask` 标记该模态是否真实存在。
- 不删除 `features` 中的模态 key，避免成员 B 做 missing modality 实验时出现 `KeyError`。

## 7. 特征缓存规则

完整五模态样本缓存目录：

```text
data/processed/cache/feature_cache
```

scene cache 目录：

```text
data/processed/cache/scene_cache_real_vit
```

缓存读取规则：

1. `MultimodalIntentDataset.from_metadata_samples(...)` 默认 `use_cache=True`、`rebuild_cache=False`。
2. `load_or_build_sample_features(...)` 会先根据 `sample_id` 查找完整五模态缓存。
3. 如果缓存存在，直接返回 `FeatureBundle`，不会重新读取源特征文件，也不会重新提取 raw data。
4. 如果缓存不存在，才读取 sample metadata 中的 `feature_paths` 并构建缓存。
5. 如果缓存和源特征文件都不存在，会抛出清晰错误，不会伪造特征。

因此，成员 A 提取完特征后，成员 B 后续训练测试可以直接复用缓存，前提是成员 B 使用同一套成员 A Dataset 或指向同一个缓存目录。

成员 B 使用时注意：

- 不要使用 `--rebuild-cache`，否则会强制重建。
- 不要使用 `--no-cache`，否则会禁用缓存。
- 如果成员 B 在 `repo-mplus0` 中独立运行，需要让配置指向 `repo-main` 的同一份缓存，或复制缓存目录，否则相对路径会指向 `repo-mplus0/data/processed/cache/...`。

## 8. 运行命令

路径检查：

```bash
python src/utils/paths.py
```

Dataset 检查：

```bash
python src/data/check_dataset.py --config configs/default.yaml
```

特征 dry-run：

```bash
python src/data/build_features.py --config configs/default.yaml --dry-run
```

clean baseline smoke test：

```bash
python experiments/train.py --config configs/default.yaml --smoke-test --epochs 1 --batch-size 2
python experiments/test.py --config configs/default.yaml --smoke-test --checkpoint results/checkpoints/best.pt --batch-size 2
```

正式训练和测试：

```bash
python experiments/train.py --config configs/default.yaml
python experiments/test.py --config configs/default.yaml --checkpoint results/checkpoints/best.pt
```

正式训练和测试需要先准备 raw data 与本地模型资源。

## 9. 输出文件

metrics：

```text
results/metrics/clean_baseline_metrics.csv
results/metrics/clean_baseline_summary.json
results/metrics/clean_baseline_test_metrics.csv
results/metrics/clean_baseline_test_summary.json
```

logs：

```text
results/logs/clean_baseline.log
```

predictions：

```text
results/predictions/clean_baseline_predictions.csv
```

checkpoints：

```text
results/checkpoints/best.pt
results/checkpoints/final.pt
```

figures：

```text
figures/loss_curve.png
figures/confusion_matrix.png
```

## 10. 当前验证状态

已完成：

- Dataset 接口检查。
- 单 batch 模型 forward smoke test。
- `train.py --smoke-test --epochs 1 --batch-size 2`。
- `test.py --smoke-test --checkpoint results/checkpoints/best.pt --batch-size 2`。
- metrics、predictions、logs、checkpoints、figures 输出检查。
- 成员 A / 成员 B 接口兼容性检查。
- README 与报告入口整理。

当前 smoke test 只证明工程链路可运行，不代表正式实验性能。

正式 clean baseline 指标当前为 TBD。

## 11. 当前缺失资源

正式运行需要准备以下资源：

```text
data/raw/imu.csv
data/raw/user_A
data/raw/user_B
data/raw/user_C
data/raw/models/all-MiniLM-L6-v2
data/raw/models/clip_teacher_model/vit-base-patch16-224
```

这些资源缺失是正常的，因为大体积 raw data 和本地模型资源不应默认提交到 Git 仓库。

## 12. 成员 B 可直接复用的接口

成员 B 可以直接复用：

1. `MultimodalIntentDataset(..., transform=None)`。
2. `features` 中固定的五个模态 key。
3. `modality_mask` 和 zero-fill 缺失模态约定。
4. `results/predictions/clean_baseline_predictions.csv` 的字段：
   ```text
   sample_id,user,split,intent_true,intent_pred,intent_true_name,intent_pred_name,intent_correct,joint_label
   ```
5. `results/metrics/*.csv` 的核心字段：
   ```text
   accuracy,macro_f1,weighted_f1,training_time,avg_test_time_per_sample
   ```
6. `results/checkpoints/best.pt` 和 `results/checkpoints/final.pt`。

## 13. 成员 B 后续需要补的部分

以下内容不属于成员 A 当前阶段，不在本文件中实现：

1. `apply_modal_noise(...)`。
2. `apply_missing_modalities(...)`。
3. `run_noise_baseline.py` 的真实训练逻辑。
4. `run_missing_baseline.py` 的真实训练逻辑。
5. improved model 的结构或损失函数。
6. 正式实验表格和报告性能结论。

## 14. 报告撰写建议

报告中可以将成员 A 的贡献概括为：

> 成员 A 负责五模态端到端 clean baseline 主线，完成了项目相对路径配置、样本索引、五模态特征读取和缓存、统一 Dataset、五模态 baseline 模型、训练/测试入口、metrics/predictions/checkpoints/figures 输出，以及与成员 B 后续实验的接口兼容性检查。

报告中不应把 smoke test 指标当作正式模型性能。正式结果需要在 raw data 与本地模型资源准备完成后重新运行生成。
