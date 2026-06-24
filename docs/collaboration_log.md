# 协作日志

## 2026-06-24 - 成员 A 五模态端到端主线整理

### 贡献者
- 姓名：成员 A / Codex
- 角色：代码 / 数据工程 / 文档整理

### 修改文件
- `configs/default.yaml`：统一配置 raw data、processed/cache、本地模型、五模态维度、训练参数和输出目录。
- `src/utils/paths.py`：提供项目相对路径解析、配置读取、运行目录创建、Hugging Face cache 环境变量设置和路径检查函数。
- `src/data/features.py`：实现五模态特征读取、序列长度归一化、音频归一化、scaler 拟合/应用、完整样本特征缓存和 scene cache 接口。
- `src/data/build_features.py`：提供五模态特征 dry-run 和基于 sample metadata 的特征构建检查入口。
- `src/data/build_samples.py`：根据配置生成 user_A、user_B、user_C 的样本索引，训练集为 user_A + user_B，测试集为 user_C。
- `src/data/dataset.py`：实现 `MultimodalIntentDataset`，稳定输出五模态 sample 字典并支持 `transform=None`。
- `src/data/check_dataset.py`：提供 Dataset 检查命令，打印样本统计、缺失项、标签分布和五模态 shape。
- `src/models/formal_baseline.py`：实现清晰的五模态 baseline model，输入固定为 `imu`、`gesture`、`audio`、`text`、`scene`。
- `src/training/engine.py`：实现训练、评估、指标计算、预测保存、checkpoint payload、loss curve 和 confusion matrix 保存。
- `experiments/train.py`：实现 clean baseline 训练入口。
- `experiments/test.py`：实现 checkpoint 测试入口。
- `README_CHINESE.md`：补充成员 A clean baseline 运行说明、端到端链路、输出路径和当前结果状态。
- `docs/collaboration_log.md`：整理为中文协作日志。
- `docs/memberA_report_and_handoff.md`：整理为中文报告与交接说明。

### 目标
建立成员 A 负责的五模态端到端 clean baseline 主线，并为成员 B 后续 noise、missing、improved model 实验预留稳定接口。本阶段不展开成员 B 的实验实现。

### 正式模态约定
正式实验模态 key 固定为：

```text
imu
gesture
audio
text
scene
```

`HoloLens`、`fisheye` 等名称只作为 raw data 来源目录或配置字段，不作为正式实验模态名。

### 数据划分
- 训练集：`user_A` + `user_B`
- 测试集：`user_C`

该划分已写入 `configs/default.yaml`，并与成员 B 的 `configs/noise.yaml`、`configs/missing_modality.yaml` 保持一致。

### 端到端调用链
```text
data/raw 原始数据
-> src/data/build_samples.py 构建样本索引
-> src/data/features.py / src/data/build_features.py 读取或构建五模态特征
-> src/data/dataset.py 输出 MultimodalIntentDataset
-> src/models/formal_baseline.py 五模态融合
-> src/training/engine.py 训练与评估
-> experiments/train.py / experiments/test.py 输出 intent prediction
```

### Dataset 输出接口
`MultimodalIntentDataset` 输出结构如下：

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

缺失模态不会删除 `features` 中的 key，而是使用 zero-fill，并通过 `modality_mask` 标记是否真实存在。

### 缓存规则
- 完整五模态样本缓存目录：`data/processed/cache/feature_cache`
- scene 特征缓存目录：`data/processed/cache/scene_cache_real_vit`
- `MultimodalIntentDataset.from_metadata_samples(...)` 默认 `use_cache=True`、`rebuild_cache=False`。
- 如果完整样本缓存存在，训练/测试会优先读取缓存，不重新提取特征。
- 如果缓存不存在，才会读取 `feature_paths` 中的五模态源特征文件并保存缓存。
- 如果缓存和源特征都不存在，会输出清晰错误，不会编造数据。

因此，成员 A 提取完特征后，成员 B 后续只要使用同一套成员 A Dataset 和同一缓存目录，就可以直接复用缓存，不需要重新提取。

### 运行方式
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

正式训练和测试应在 raw data 与本地模型准备完成后运行：

```bash
python experiments/train.py --config configs/default.yaml
python experiments/test.py --config configs/default.yaml --checkpoint results/checkpoints/best.pt
```

### 输出文件
- `results/metrics/clean_baseline_metrics.csv`
- `results/metrics/clean_baseline_summary.json`
- `results/metrics/clean_baseline_test_metrics.csv`
- `results/metrics/clean_baseline_test_summary.json`
- `results/logs/clean_baseline.log`
- `results/predictions/clean_baseline_predictions.csv`
- `results/checkpoints/best.pt`
- `results/checkpoints/final.pt`
- `figures/loss_curve.png`
- `figures/confusion_matrix.png`

### 当前状态
- 成员 A clean baseline 工程链路已经通过 smoke test。
- smoke test 使用极小合成样本，只证明代码链路、checkpoint、metrics、predictions 和 figures 输出可运行，不代表正式实验性能。
- 正式 clean baseline 指标当前为 TBD，需要在正式 raw data 与本地模型资源就绪后重新运行。

### 已知问题
- 当前仓库不应默认包含大体积 raw data 和本地模型资源。
- 正式运行需要准备：
  - `data/raw/imu.csv`
  - `data/raw/user_A`
  - `data/raw/user_B`
  - `data/raw/user_C`
  - `data/raw/models/all-MiniLM-L6-v2`
  - `data/raw/models/clip_teacher_model/vit-base-patch16-224`
- `joint_accuracy` 当前是补充占位指标，后续如果要正式计算，需要建立稳定的 joint label 到类别 id 映射。
- 成员 B 的 `apply_modal_noise`、`apply_missing_modalities` 和 runner 真实训练逻辑仍由成员 B 后续实现。

### 给报告撰写者的说明
报告中可以引用本日志和 `docs/memberA_report_and_handoff.md` 说明成员 A 的贡献：完成了五模态端到端主线、Dataset 接口、特征缓存接口、clean baseline 训练/测试入口和输出规范。正式实验结果不要使用 smoke test 指标替代，应在正式数据就绪后重新运行生成。
