# 成员 A 交接前任务与命令清单

日期：2026-06-24
适用仓库：`repo-main`
服务器目录：`/share/home/tm1078571822880000/a893873600/wyr_ml_course_project_work/repo-main`

## 1. 阅读依据

本清单根据以下材料整理：

- `MEMBER_A_PRE_HANDOFF_RUN_PLAN.md`
- `课程项目2026.pdf`
- `docs/PROJECT_WORKFLOW_AND_TEAM_DIVISION.md`
- `AGENTS.md`

课程要求的核心点是：

- 用户 A、用户 B 作为训练集。
- 用户 C 作为测试集。
- 将老师源码中“先提取特征再训练”的流程重构为端到端主线：

```text
raw data -> preprocessing/features -> multimodal fusion model -> intent output
```

- 重构后需要支持 clean baseline、模态噪声 baseline、模态缺失 baseline 和改进模型实验。
- 成员 A 当前只负责五模态 clean baseline 端到端主线和交接接口，不展开成员 B 的 noise、missing、improved model 实验实现。

## 2. 成员 A 交接边界

成员 A 交接前必须完成：

1. 确认项目路径和输出目录正确。
2. 确认 raw data、本地模型、样本索引和五模态配置可被代码读取。
3. 生成或验证 `user_A + user_B` 训练、`user_C` 测试的样本索引。
4. 验证五模态 key 固定为 `imu`、`gesture`、`audio`、`text`、`scene`。
5. 跑通 clean baseline 的训练入口 `experiments/train.py`。
6. 跑通 clean baseline 的测试入口 `experiments/test.py`。
7. 保存 metrics、logs、predictions、checkpoints 和 figures。
8. 更新中文实验日志和协作日志。
9. 给成员 B 留下可复用的 Dataset、缓存、预测和指标接口。

成员 A 不需要在交接前完成：

- 模态噪声实验的完整实现和运行。
- 模态缺失实验的完整实现和运行。
- improved model 的实现和对比实验。
- 成员 B 的 runner 脚手架扩展。

## 3. 正式数据与模型放置要求

正式运行前，服务器上应存在：

```text
data/raw/imu.csv
data/raw/user_A
data/raw/user_B
data/raw/user_C
data/raw/models/all-MiniLM-L6-v2
data/raw/models/clip_teacher_model/vit-base-patch16-224
```

正式模态 key 固定为：

```text
imu, gesture, audio, text, scene
```

`HoloLens`、`fisheye` 只作为 raw data 来源目录，不作为正式实验模态名。

## 4. 推荐环境检查命令

进入仓库根目录：

```bash
cd /share/home/tm1078571822880000/a893873600/wyr_ml_course_project_work/repo-main
```

确认 Python：

```bash
which python
python --version
```

如果当前 shell 中没有 `python`，改用：

```bash
which python3
python3 --version
```

或使用课程服务器中的成员 A 环境：

```bash
/share/home/tm1078571822880000/a893873600/wyr_ml_course_project_work/software/memberA_miniconda/bin/python --version
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

如果使用 `python3` 或指定 conda Python，请把下面所有命令里的 `python` 替换为对应解释器。

检查 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 5. 成员 A 正式运行前检查

### 5.1 路径检查

```bash
python src/utils/paths.py
```

检查目标：

- 能读取 `configs/default.yaml`。
- 能创建或确认以下目录：

```text
results/metrics/
results/logs/
results/predictions/
results/checkpoints/
figures/
report/screenshots/
data/processed/cache/
```

- 如果 raw data 或本地模型缺失，应先补齐资源，不应进入正式训练。

### 5.2 Dataset 检查

```bash
python src/data/check_dataset.py --config configs/default.yaml
```

检查目标：

- 能识别 `user_A`、`user_B`、`user_C`。
- 训练集为 `user_A + user_B`。
- 测试集为 `user_C`。
- Dataset 输出中五个模态 key 始终存在。
- 缺失模态不删除 key，而是使用 zero-fill 或 `modality_mask` 标记。

### 5.3 生成正式样本索引

```bash
python src/data/build_samples.py --config configs/default.yaml --output data/processed/sample_index.json
```

交付文件：

```text
data/processed/sample_index.json
```

检查目标：

- 样本数不应为 0。
- train/test 划分符合用户 A/B 训练、用户 C 测试。
- 样本字段应包含 `sample_id`、`user`、`split`、`raw_paths`、`feature_paths`、`intent_label`、`scene_label`、`joint_label`。

### 5.4 五模态特征 dry-run

```bash
python src/data/build_features.py --config configs/default.yaml --dry-run
```

检查目标：

```text
target_timesteps: 10
imu: 12
gesture: 768
audio: 39
text: 384
scene: 768
```

这一步只检查配置和路径，不代表正式训练结果。

### 5.5 可选：抽查或预构建特征缓存

抽查前 5 条样本：

```bash
python src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --limit 5
```

如需预构建全部缓存：

```bash
python src/data/build_features.py --config configs/default.yaml --metadata-json data/processed/sample_index.json --limit 100000
```

缓存交接目录：

```text
data/processed/cache/feature_cache/
data/processed/cache/scene_cache_real_vit/
```

注意：缓存只能作为加速和复用手段，不能成为唯一输入。若缓存和源特征都缺失，代码应输出清晰错误，而不是伪造结果。

## 6. 成员 A 正式 clean baseline 训练

正式训练命令：

```bash
python experiments/train.py --config configs/default.yaml
```

训练目标：

- 使用 `user_A + user_B`。
- 读取或构建五模态特征。
- 训练 `FormalMultimodalBaseline`。
- 保存 checkpoint、metrics、logs、predictions 和 figures。

主要输出：

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

交接判断标准：

- 命令正常结束并输出训练完成信息。
- `results/checkpoints/best.pt` 存在。
- `results/checkpoints/final.pt` 存在。
- metrics 中 `status` 不能是 `smoke_test`。
- 日志中能看到 epoch、loss、accuracy、macro_f1、weighted_f1。

## 7. 成员 A 正式 user_C 测试

正式测试命令：

```bash
python experiments/test.py --config configs/default.yaml --checkpoint results/checkpoints/best.pt
```

测试目标：

- 加载正式训练得到的 `best.pt`。
- 使用 `user_C` 测试集。
- 输出 clean baseline 的正式测试指标和预测结果。

主要输出：

```text
results/metrics/clean_baseline_test_metrics.csv
results/metrics/clean_baseline_test_summary.json
results/predictions/clean_baseline_predictions.csv
results/logs/clean_baseline.log
figures/confusion_matrix.png
```

交接判断标准：

- 命令正常结束并输出测试完成信息。
- 测试结果来自 `user_C`。
- predictions 至少包含：

```text
sample_id
user
split
intent_true
intent_pred
```

- metrics 至少包含：

```text
accuracy
macro_f1
weighted_f1
avg_test_time_per_sample
```

## 8. 可选 smoke test

如果正式 raw data 或本地模型尚未完全准备好，可以只做工程链路 smoke test：

```bash
python experiments/train.py --config configs/default.yaml --smoke-test --epochs 1 --batch-size 2
python experiments/test.py --config configs/default.yaml --smoke-test --checkpoint results/checkpoints/best.pt --batch-size 2
```

注意：

- smoke test 使用合成小样本。
- smoke test 只能证明工程链路能跑通。
- smoke test 指标不能写入正式实验结果表。
- 如果交接时只有 smoke test，应在 `docs/experiment_log.md` 中明确说明正式 raw data 或模型资源缺失。

## 9. 成员 A 不建议作为标准入口运行的文件

| 文件 | 说明 |
|---|---|
| `experiments/train_and_test.py` | 老师原始或旧版训练测试逻辑参考，当前不作为成员 A 标准入口。 |
| `src/models/baseline_real_scene.py` | 老师 baseline 参考代码，保留其合理五模态逻辑，不直接作为交接入口。 |
| `src/modules/feature_extraction/*.py` | 特征提取参考脚本，仅在需要补齐 raw data 到源特征时单独适配。 |
| `experiments/run_noise_baseline.py` | 属于成员 B 噪声实验方向。 |
| `experiments/run_missing_baseline.py` | 属于成员 B 缺失模态实验方向。 |
| `experiments/run_improved_model.py` | 属于成员 B 改进模型方向。 |

## 10. 交接给成员 B 的接口

成员 B 可以直接复用：

```text
configs/default.yaml
data/processed/sample_index.json
data/processed/cache/feature_cache/
src/data/dataset.py
src/data/features.py
src/models/formal_baseline.py
src/training/engine.py
results/checkpoints/best.pt
results/metrics/clean_baseline_test_metrics.csv
results/predictions/clean_baseline_predictions.csv
```

接口约定：

- Dataset 类名：`MultimodalIntentDataset`
- 支持 `transform=None`
- 五模态 key 固定：`imu`、`gesture`、`audio`、`text`、`scene`
- 缺失模态不删除 key
- 使用 `modality_mask` 标记真实存在的模态
- predictions 字段可用于成员 B 后续对比 clean/noise/missing/improved model

## 11. 日志与截图要求

正式运行后需要更新：

```text
docs/collaboration_log.md
docs/experiment_log.md
```

建议保存截图：

```text
report/screenshots/path_check.png
report/screenshots/dataset_check.png
report/screenshots/train_done.png
report/screenshots/test_done.png
```

日志中必须如实记录：

- 运行命令。
- 数据划分。
- 使用模型。
- 输出路径。
- metrics。
- 是否为正式运行或 smoke test。
- 失败原因和缺失资源。

## 12. Git 提交建议

检查改动：

```bash
git status --short
git diff --check
```

建议提交代码和小型文档结果：

```bash
git add configs/default.yaml README_CHINESE.md docs/collaboration_log.md docs/experiment_log.md docs/memberA_report_and_handoff.md MEMBER_A_PRE_HANDOFF_TASKS_AND_COMMANDS.md src/utils/paths.py src/data/features.py src/data/build_features.py src/data/build_samples.py src/data/dataset.py src/data/check_dataset.py src/models/formal_baseline.py src/training/engine.py experiments/train.py experiments/test.py requirements.txt
git commit -m "整理成员A交接前运行任务"
git push origin wyr
```

大体积文件不建议直接提交 GitHub：

```text
data/raw/
data/processed/cache/
results/checkpoints/*.pt
本地预训练模型目录
```

如果团队需要这些文件，应通过服务器路径或网盘交接，并在文档中写清楚位置。

## 13. 最终交接检查清单

- [ ] `python src/utils/paths.py` 通过。
- [ ] `python src/data/check_dataset.py --config configs/default.yaml` 能识别用户 A/B/C。
- [ ] `data/processed/sample_index.json` 已生成。
- [ ] `python src/data/build_features.py --config configs/default.yaml --dry-run` 输出五模态配置正确。
- [ ] 必要时已构建或验证 `data/processed/cache/feature_cache/`。
- [ ] `python experiments/train.py --config configs/default.yaml` 正式完成。
- [ ] `python experiments/test.py --config configs/default.yaml --checkpoint results/checkpoints/best.pt` 正式完成。
- [ ] `results/metrics/` 中存在训练与测试 metrics。
- [ ] `results/logs/clean_baseline.log` 存在。
- [ ] `results/predictions/clean_baseline_predictions.csv` 字段完整。
- [ ] `figures/loss_curve.png` 和 `figures/confusion_matrix.png` 存在。
- [ ] 运行截图已保存到 `report/screenshots/`。
- [ ] `docs/collaboration_log.md` 和 `docs/experiment_log.md` 已更新。
- [ ] 明确说明 smoke test 不是正式实验结果。
- [ ] 成员 B 已知道可复用缓存、Dataset、checkpoint、metrics 和 predictions。
