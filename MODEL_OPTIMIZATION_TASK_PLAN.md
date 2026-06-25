# 模型优化任务详细方案

适用仓库：`multimodal-intention-recognition-mix`  
正式项目目录：`multimodal-intention-recognition-mix`  
方案日期：2026-06-25  
任务目标：在已完成端到端重构、clean baseline、模态噪声 baseline、模态缺失 baseline 的基础上，新增模型优化方案，提高模型在模态噪声和模态缺失条件下的分类准确率。

---

## 1. 当前项目状态确认

当前仓库已经具备以下正式运行主线：

```text
raw data
-> src/data/build_samples.py
-> src/data/raw_feature_builder.py
-> src/data/raw_extractors/*
-> src/data/features.py
-> src/data/dataset.py
-> src/models/formal_baseline.py
-> src/training/experiment_runner.py
-> experiments/run_clean_baseline.py
-> experiments/run_noise_baseline.py
-> experiments/run_missing_baseline.py
```

当前已完成任务：

1. 使用用户 A、用户 B 作为训练集，用户 C 作为测试集。
2. 已经建立从 raw data 到五模态特征再到模型输出的端到端流程。
3. 已经提供 clean baseline 运行入口。
4. 已经提供单模态噪声 baseline 运行入口。
5. 已经提供单模态和双模态缺失 baseline 运行入口。

当前尚未完成任务：

```text
configs/improved_model.yaml
src/models/improved_model.py
experiments/run_improved_model.py
```

也就是说，后续模型优化任务应以“新增文件 + 复用当前稳定主线”为主，不应重写或破坏已经验证过的 baseline 代码。

---

## 2. 当前 baseline 的关键特点

当前正式 baseline 模型位于：

```text
src/models/formal_baseline.py
```

模型输入为五个正式模态：

| 模态 | 输入形状 | 来源 |
|---|---|---|
| `imu` | `(N, 10, 12)` | IMU 时序特征 |
| `gesture` | `(N, 10, 768)` | 手势视觉特征 |
| `audio` | `(N, 10, 39)` | MFCC 音频特征 |
| `text` | `(N, 10, 384)` | ASR 文本嵌入 |
| `scene` | `(N, 768)` | 场景视觉特征 |

当前 baseline 流程：

```text
各模态特征
-> modality-specific projection
-> time embedding / modality embedding
-> TransformerEncoder
-> mean pooling
-> intent classifier
```

当前缺失模态实验的实现位于：

```text
src/data/transforms.py
```

其中 `MissingModalityTransform` 会将缺失模态置零，并把对应的：

```text
modality_mask[modality] = False
```

但是当前 baseline 的 `forward()` 中没有真正利用 `modality_mask`。也就是说，当前模型虽然知道数据集层面标记了某些模态缺失，但模型融合阶段仍然会把零填充后的 token 送入 Transformer 并参与平均池化。

这正是模型优化最自然、最稳妥的切入点。

---

## 3. 优化目标

本次模型优化的目标不是重新设计整个项目，而是在当前稳定代码基础上新增一个鲁棒融合模型，使模型能够：

1. 在模态缺失时自动降低缺失模态的贡献。
2. 在模态噪声较强时自动降低不可靠模态的贡献。
3. 在 clean 数据上尽量保持 baseline 性能。
4. 在 noise baseline 和 missing modality baseline 设置下取得更稳定的 Accuracy 和 Macro F1。
5. 能够在报告中清晰说明“新增模块或损失项”的创新点。

---

## 4. 推荐最终方案

推荐采用：

```text
模态可靠性门控融合
+
模态 Dropout 训练
```

英文名称可写为：

```text
Reliability-Gated Multimodal Fusion with Modality Dropout
```

中文名称可写为：

```text
基于模态可靠性门控与模态 Dropout 的鲁棒多模态融合方法
```

该方案包含两个核心组件：

1. 新模块：`Reliability Gate`
2. 新训练策略：`Random Modality Dropout`

其中 `Reliability Gate` 是报告中的主要创新点，`Random Modality Dropout` 是增强鲁棒性的辅助训练策略。

---

## 5. 方案一：模态可靠性门控融合

### 5.1 设计动机

在真实 AR 交互场景中，不同模态的可靠性会随环境变化：

- 音频可能受环境噪声影响。
- 手势可能因遮挡或光照变化变差。
- 文本可能因 ASR 识别错误而不可靠。
- IMU 可能因设备抖动或时间对齐误差产生异常。
- 场景模态可能因视角变化或视频帧缺失而不稳定。

因此，不应简单地把五个模态平均融合，而应让模型根据输入动态估计每个模态的可靠性。

### 5.2 核心思想

对每个模态生成一个可靠性分数：

```text
r_imu, r_gesture, r_audio, r_text, r_scene
```

其中：

```text
r_m ∈ [0, 1]
```

含义：

- `r_m` 越接近 1，说明模型认为该模态越可靠。
- `r_m` 越接近 0，说明模型认为该模态越不可靠。
- 如果该模态缺失，则通过 `modality_mask` 强制令 `r_m = 0`。

最终融合不再使用简单平均：

```text
fused = mean(encoded_tokens)
```

而改为可靠性加权融合：

```text
fused = weighted_mean(encoded_tokens, reliability_scores, modality_mask)
```

### 5.3 结构示意

```text
imu features      -> imu projection      -> imu tokens      -> imu gate
gesture features  -> gesture projection  -> gesture tokens  -> gesture gate
audio features    -> audio projection    -> audio tokens    -> audio gate
text features     -> text projection     -> text tokens     -> text gate
scene features    -> scene projection    -> scene token     -> scene gate

all tokens + modality weights
-> TransformerEncoder
-> reliability weighted pooling
-> classifier
-> intent logits
```

### 5.4 模态级 summary 表示

对于时序模态：

```text
imu, gesture, audio, text
```

每个模态有 10 个 token，可先做平均池化得到模态 summary：

```text
s_m = mean(tokens_m, dim=time)
```

对于场景模态：

```text
scene
```

它本身就是一个 token：

```text
s_scene = token_scene
```

### 5.5 Reliability Gate 设计

每个模态共享一个小型 MLP 或者每个模态单独一个 MLP。为了实现简单且效果稳定，建议每个模态使用独立 gate：

```python
gate_m = nn.Sequential(
    nn.LayerNorm(model_dim),
    nn.Linear(model_dim, model_dim // 2),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(model_dim // 2, 1),
    nn.Sigmoid(),
)
```

输出：

```text
score_m = gate_m(summary_m)
```

形状：

```text
score_m: (batch, 1)
```

### 5.6 使用 modality_mask

当前 Dataset 已经返回：

```text
batch["modality_mask"]
```

形状近似为：

```text
{
  "imu": BoolTensor(batch),
  "gesture": BoolTensor(batch),
  "audio": BoolTensor(batch),
  "text": BoolTensor(batch),
  "scene": BoolTensor(batch)
}
```

融合时应执行：

```text
score_m = score_m * modality_mask_m
```

如果某个模态缺失：

```text
modality_mask_m = False
score_m = 0
```

这样缺失模态不会继续影响融合结果。

### 5.7 token 权重扩展

由于 `imu/gesture/audio/text` 各有 10 个 token，`scene` 有 1 个 token，因此需要把模态级权重扩展到 token 级：

```text
imu score      -> repeat 10 times
gesture score  -> repeat 10 times
audio score    -> repeat 10 times
text score     -> repeat 10 times
scene score    -> repeat 1 time
```

最终 token 权重长度为：

```text
10 + 10 + 10 + 10 + 1 = 41
```

对应 Transformer 输入 token 数量。

### 5.8 加权池化

Transformer 编码后：

```text
encoded: (batch, 41, model_dim)
weights: (batch, 41, 1)
```

融合：

```text
fused = sum(encoded * weights) / (sum(weights) + eps)
```

其中 `eps` 防止所有模态都被屏蔽时除零。

建议设置：

```text
eps = 1e-6
```

---

## 6. 方案二：模态 Dropout 训练

### 6.1 设计动机

课程要求中包含模态缺失实验：

- 丢弃任意一个模态。
- 丢弃任意两个模态。

如果模型只在完整五模态上训练，测试时突然缺失一个或两个模态，性能通常会明显下降。

因此，训练阶段可以主动随机屏蔽部分模态，使模型提前适应模态缺失情况。

### 6.2 实现位置

建议新增：

```text
src/data/improved_transforms.py
```

新增类：

```text
RandomModalityDropoutTransform
```

该 transform 只用于 improved model 的训练集，不影响现有 baseline。

### 6.3 训练时行为

每个样本以一定概率随机丢弃 1 到 2 个模态：

```text
drop_prob = 0.3
max_drop_modalities = 2
```

示例：

```text
样本 1：保留全部模态
样本 2：丢弃 audio
样本 3：丢弃 imu + text
样本 4：丢弃 scene
```

丢弃方式与当前 `MissingModalityTransform` 保持一致：

```text
features[modality] = zeros_like(features[modality])
modality_mask[modality] = False
```

### 6.4 与现有缺失实验的关系

当前已有：

```text
MissingModalityTransform
```

用于构造固定缺失组合的 baseline 实验。

新增：

```text
RandomModalityDropoutTransform
```

用于 improved model 的训练增强。

两者关系：

| Transform | 使用阶段 | 用途 |
|---|---|---|
| `MissingModalityTransform` | train / val / test | 构造指定缺失实验 |
| `RandomModalityDropoutTransform` | train only | 增强模型鲁棒性 |

---

## 7. 是否需要新增损失项

课程要求允许：

```text
引入新模块或损失项
```

本方案已经包含新模块 `Reliability Gate`，因此不强制新增损失项。

如果时间充足，可以增加一个轻量级 gate 正则项，但建议作为可选项。

### 7.1 可选 gate 正则

为了避免模型把所有 gate 都压到接近 0，可以加入一个轻量正则：

```text
L_total = L_ce + λ * L_gate
```

其中：

```text
L_gate = mean((mean(gates) - target_reliability)^2)
```

建议：

```text
target_reliability = 0.7
λ = 0.01
```

但是从项目稳定性角度看，第一版不建议加入复杂损失。优先实现新模块和训练增强，保证能跑通并能与 baseline 对比。

---

## 8. 建议新增文件与职责

### 8.1 `src/models/improved_model.py`

职责：

1. 定义 `ReliabilityGatedMultimodalModel`。
2. 复用 baseline 的五模态输入约定。
3. 新增 reliability gates。
4. 在 forward 中真正使用 `modality_mask`。
5. 输出字段与 baseline 保持兼容：

```text
intent_logits
scene_logits
joint_logits
fused
reliability_scores
```

输出 `reliability_scores` 有助于后续报告分析和可视化。

### 8.2 `src/data/improved_transforms.py`

职责：

1. 定义 `RandomModalityDropoutTransform`。
2. 可选定义 `BuildImprovedTrainTransform`。
3. 复用或兼容当前 `src/data/transforms.py` 中的缺失模态逻辑。

### 8.3 `configs/improved_model.yaml`

职责：

1. 保存 improved model 的超参数。
2. 保存模态 Dropout 参数。
3. 保存需要运行的实验组合。

建议配置：

```yaml
experiment:
  name: improved_model
  model_type: ReliabilityGatedMultimodalModel
  seed: 42

model:
  improved:
    model_dim: 128
    num_heads: 4
    depth: 2
    dropout: 0.1
    ff_multiplier: 4
    use_reliability_gate: true
    use_modality_mask: true

training:
  epochs: 5
  lr: 0.001
  modality_dropout:
    enabled: true
    drop_prob: 0.3
    max_drop_modalities: 2

evaluation:
  run_clean: true
  run_noise: true
  run_missing: true
  run_ablation: true
```

### 8.4 `experiments/run_improved_model.py`

职责：

1. 读取 `configs/improved_model.yaml` 和 `configs/default.yaml`。
2. 构建 improved model。
3. 运行 clean improved 实验。
4. 运行 noise improved 实验。
5. 运行 missing improved 实验。
6. 运行 ablation 实验。
7. 保存聚合指标表。

---

## 9. 与现有 runner 的复用策略

当前通用 runner 位于：

```text
src/training/experiment_runner.py
```

它目前内部固定使用：

```python
build_formal_baseline_from_config(base_config)
```

因此有两种实现方式。

### 9.1 方式 A：最小改动方式

新增：

```text
src/training/improved_experiment_runner.py
```

将 `experiment_runner.py` 中的逻辑复制一份并做最小必要修改：

```python
from src.models.improved_model import build_improved_model_from_config
```

然后把：

```python
model = build_formal_baseline_from_config(base_config).to(device)
```

替换为：

```python
model = build_improved_model_from_config(base_config, improved_config).to(device)
```

优点：

- 不影响已有 baseline。
- 风险最低。
- 出问题时容易回退。

缺点：

- 会与原 runner 有少量重复代码。

### 9.2 方式 B：更优雅方式

修改 `src/training/experiment_runner.py`，新增可选参数：

```python
model_builder=None
criterion_builder=None
```

然后：

```python
if model_builder is None:
    model = build_formal_baseline_from_config(base_config).to(device)
else:
    model = model_builder(base_config).to(device)
```

优点：

- 复用程度更高。
- 后续扩展更干净。

缺点：

- 会修改已验证 baseline 公共 runner，需要重新确认 clean/noise/missing baseline 不受影响。

### 9.3 推荐

考虑到当前已验证代码可运行，并且任务要求不要修改之前代码，推荐采用方式 A：

```text
新增 improved_experiment_runner.py，不修改 experiment_runner.py
```

---

## 10. improved 实验矩阵

### 10.1 clean improved

目的：

验证新模型在完整五模态 clean 数据上不会明显低于 baseline。

命令：

```bash
python experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean
```

输出：

```text
results/metrics/improved_clean_metrics.csv
results/logs/improved_clean.log
results/predictions/improved_clean_predictions.csv
results/checkpoints/improved_clean_best.pt
figures/improved_clean_loss_curve.png
figures/improved_clean_confusion_matrix.png
```

### 10.2 noise improved

目的：

验证新模型在单模态噪声下是否比 baseline 更稳。

实验组合：

```text
5 个模态 × 3 个噪声比例 = 15 组
```

模态：

```text
imu, gesture, audio, text, scene
```

噪声比例：

```text
20%, 40%, 60%
```

输出：

```text
results/metrics/improved_noise_metrics.csv
```

### 10.3 missing improved

目的：

验证新模型在单模态或双模态缺失下是否比 baseline 更稳。

实验组合：

```text
5 个单模态缺失 + 10 个双模态缺失 = 15 组
```

输出：

```text
results/metrics/improved_missing_modality_metrics.csv
```

### 10.4 ablation study

建议至少做三组：

| 实验组 | Reliability Gate | Modality Dropout | 目的 |
|---|---|---|---|
| `improved_gate_only` | 是 | 否 | 验证门控模块作用 |
| `improved_dropout_only` | 否 | 是 | 验证模态 Dropout 作用 |
| `improved_gate_dropout` | 是 | 是 | 最终方案 |

输出：

```text
results/metrics/improved_ablation_metrics.csv
```

---

## 11. 指标设计

与现有 baseline 保持一致：

| 指标 | 说明 |
|---|---|
| `accuracy` | 意图分类准确率，主指标 |
| `macro_f1` | 多类别平均 F1，适合类别不均衡分析 |
| `weighted_f1` | 按类别样本数加权 F1 |
| `loss` | 测试损失 |
| `scene_accuracy` | 场景分类辅助指标 |
| `training_time` | 训练总时间 |
| `avg_test_time_per_sample` | 平均单样本测试时间 |

建议新增分析字段：

| 字段 | 说明 |
|---|---|
| `target_modality` | 噪声目标模态 |
| `noise_ratio` | 噪声比例 |
| `missing_modalities` | 缺失模态组合 |
| `use_reliability_gate` | 是否启用可靠性门控 |
| `use_modality_dropout` | 是否启用模态 Dropout |

---

## 12. 结果对比表建议

### 12.1 clean 对比

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Baseline | 待填 | 待填 | 待填 |
| Improved | 待填 | 待填 | 待填 |

### 12.2 noise 对比

| Target Modality | Noise Ratio | Baseline Acc | Improved Acc | Δ Acc |
|---|---:|---:|---:|---:|
| imu | 20% | 待填 | 待填 | 待填 |
| imu | 40% | 待填 | 待填 | 待填 |
| imu | 60% | 待填 | 待填 | 待填 |
| gesture | 20% | 待填 | 待填 | 待填 |
| ... | ... | ... | ... | ... |

### 12.3 missing 对比

| Missing Modalities | Baseline Acc | Improved Acc | Δ Acc |
|---|---:|---:|---:|
| imu | 待填 | 待填 | 待填 |
| gesture | 待填 | 待填 | 待填 |
| audio | 待填 | 待填 | 待填 |
| text | 待填 | 待填 | 待填 |
| scene | 待填 | 待填 | 待填 |
| imu + gesture | 待填 | 待填 | 待填 |
| ... | ... | ... | ... |

### 12.4 消融实验

| Model Variant | Accuracy | Macro F1 | 说明 |
|---|---:|---:|---|
| Baseline | 待填 | 待填 | 原始融合 |
| Gate Only | 待填 | 待填 | 只加可靠性门控 |
| Dropout Only | 待填 | 待填 | 只加训练阶段模态 Dropout |
| Gate + Dropout | 待填 | 待填 | 最终模型 |

---

## 13. 运行命令建议

### 13.1 短跑检查

短跑只用于检查代码能否跑通，不作为正式报告结果。

```bash
python experiments/run_improved_model.py \
  --config configs/improved_model.yaml \
  --base-config configs/default.yaml \
  --max-samples 6 \
  --epochs 1
```

### 13.2 正式运行

```bash
python experiments/run_improved_model.py \
  --config configs/improved_model.yaml \
  --base-config configs/default.yaml
```

如果希望分阶段运行：

```bash
python experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean
python experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode noise
python experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode missing
python experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode ablation
```

---

## 14. 输出文件规范

根据 `AGENTS.md`，输出应保存到：

```text
results/metrics/
results/logs/
results/predictions/
results/checkpoints/
figures/
```

建议最终输出：

```text
results/metrics/improved_clean_metrics.csv
results/metrics/improved_noise_metrics.csv
results/metrics/improved_missing_modality_metrics.csv
results/metrics/improved_ablation_metrics.csv

results/logs/improved_model.log
results/logs/improved_clean.log
results/logs/improved_noise_*.log
results/logs/improved_missing_*.log

results/predictions/improved_clean_predictions.csv
results/predictions/improved_noise_*_predictions.csv
results/predictions/improved_missing_*_predictions.csv

results/checkpoints/improved_clean_best.pt
results/checkpoints/improved_noise_*_best.pt
results/checkpoints/improved_missing_*_best.pt

figures/improved_clean_loss_curve.png
figures/improved_clean_confusion_matrix.png
figures/improved_noise_accuracy_comparison.png
figures/improved_missing_accuracy_comparison.png
figures/improved_ablation_comparison.png
```

---

## 15. 实现步骤建议

### 第一步：新增 improved model

新增：

```text
src/models/improved_model.py
```

先实现：

```text
ReliabilityGatedMultimodalModel
build_improved_model_from_config()
```

并提供一个 smoke forward 检查函数，只使用随机张量，不依赖真实数据。

### 第二步：新增训练增强 transform

新增：

```text
src/data/improved_transforms.py
```

实现：

```text
RandomModalityDropoutTransform
```

要求：

1. 不修改原始 sample。
2. 保持五模态 key 完整。
3. 被 drop 的模态置零。
4. 被 drop 的模态 `modality_mask=False`。

### 第三步：新增 improved config

新增：

```text
configs/improved_model.yaml
```

配置模型超参数、训练参数和实验开关。

### 第四步：新增 improved runner

新增：

```text
src/training/improved_experiment_runner.py
experiments/run_improved_model.py
```

优先复用现有代码风格：

- `setup_logger`
- `set_seed`
- `build_sample_index`
- `MultimodalIntentDataset`
- `save_metrics_csv`
- `save_predictions`
- `save_summary_json`
- `save_loss_curve`
- `save_confusion_matrix`

### 第五步：短跑检查

服务器中运行：

```bash
python experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --max-samples 6 --epochs 1
```

只检查：

- 是否能构建 Dataset。
- 是否能 forward。
- 是否能 backward。
- 是否能保存 metrics。
- 是否能保存 checkpoint。

### 第六步：正式 improved 实验

按 clean、noise、missing、ablation 顺序运行。

推荐顺序：

```text
clean -> ablation -> noise -> missing
```

理由：

1. clean 最快确认模型基本性能。
2. ablation 能确认模块是否有效。
3. noise 和 missing 组合较多，运行成本更高。

---

## 16. 风险与应对

| 风险 | 原因 | 应对 |
|---|---|---|
| clean 性能低于 baseline | 门控初期训练不稳定 | gate 权重初始化偏向 1，或者降低 dropout |
| 所有 gate 接近 0 | Sigmoid 输出塌缩 | 加 eps，必要时加入 gate 正则 |
| missing 条件提升不明显 | 训练缺失组合不足 | 提高 `drop_prob` 到 0.4 |
| noise 条件提升不明显 | 当前噪声是 feature-level，扰动较轻或较重 | 调整 `noise_std_scale` 或报告中如实分析 |
| 训练时间变长 | 新增 gate 和多组实验 | 先短跑，再正式跑关键组合 |
| 修改影响 baseline | 共用 runner 被改坏 | 推荐新增 improved runner，不修改 baseline runner |

---

## 17. 报告可复用描述草稿

本项目在五模态 Transformer 融合基线基础上，引入了模态可靠性门控融合模块。该模块分别对 IMU、手势、音频、文本和场景五个模态的隐层表示进行可靠性估计，生成模态级权重，并结合数据集中提供的 `modality_mask` 对缺失模态进行显式抑制。与原始平均池化融合相比，可靠性门控能够在某一模态受到噪声污染或发生缺失时降低其对最终分类结果的影响，从而提升模型在不完整和受干扰多模态输入下的鲁棒性。

此外，本项目在训练阶段加入模态 Dropout 策略，随机屏蔽一个或两个输入模态，使模型在训练过程中主动接触多种模态缺失组合。该策略能够缓解训练阶段与测试阶段的模态完整性分布差异，使模型在课程要求的单模态缺失和双模态缺失实验中具有更好的适应能力。

---

## 18. 最终推荐实现结论

本项目后续模型优化不建议重构已有端到端流程，也不建议修改老师源代码或已验证 baseline 主线。最稳妥的实现方式是：

```text
新增 ReliabilityGatedMultimodalModel
+ 新增 RandomModalityDropoutTransform
+ 新增 run_improved_model.py
+ 复用当前 Dataset / transforms / training utils / output paths
```

最终模型优化任务的最小交付物为：

```text
configs/improved_model.yaml
src/models/improved_model.py
src/data/improved_transforms.py
src/training/improved_experiment_runner.py
experiments/run_improved_model.py
results/metrics/improved_*.csv
docs/method_notes.md
docs/experiment_log.md
```

其中正式实验结果必须来自课程服务器的完整数据运行，不能使用 smoke-test、`--max-samples` 或本机缺数据环境下的结果。
