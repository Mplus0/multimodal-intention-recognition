# 个人期末论文方法与运行指南

## 1. 文档用途

本文档说明个人期末论文采用的方法、研究动机、创新点、代码实现、实验设计、运行方法、输出位置和论文写作素材。

个人方法名称：

> **Repetition-Aware Text Compression with Modality Dropout**

建议英文论文题目：

> **Repetition-Aware Text Compression for Robust Multimodal Intention Recognition in AR Interaction**

本文档中的“预期”“可能”等内容不是实验结论。所有 Accuracy、Macro-F1、Weighted-F1、训练时间和测试时间必须来自正式环境中的真实数据运行，不得估计或补写。

## 2. 项目背景

项目研究 AR 交互场景中的多模态用户意图识别。端到端流程为：

```text
raw multimodal data
-> preprocessing / feature extraction
-> multimodal fusion model
-> intention label output
```

五种输入模态为：

```text
imu, gesture, audio, text, scene
```

数据划分固定为：

- Train：user A + user B
- Test：user C

团队改进模型采用：

1. Reliability-Gated Fusion：学习各模态可靠性权重。
2. Uniform Modality Dropout：训练时随机丢弃完整模态。
3. `modality_mask`：显式标记可用和缺失模态。

个人论文在团队方法上增加重复感知 Text 压缩，并通过消融实验分析该模块与 reliability gate、modality dropout 的关系。

## 3. 已发现的问题

### 3.1 Text 特征存在重复表示

当前特征提取流程得到一个句子级 Text embedding，然后沿时间维重复 10 次：

```text
[D] -> [10, D]
```

因此 10 个时间步通常包含相同或高度相似的语义向量。

### 3.2 团队模型会把重复向量处理成多个 token

原团队模型会：

1. 分别投影 10 个 Text 向量；
2. 为每个向量添加不同的时间位置编码；
3. 将 10 个 Text token 送入 Transformer；
4. 为同一模态的 token 分配相同 reliability score；
5. 在最终融合中累计这 10 个 token 的贡献。

这可能使同一份句子语义被当成多份证据，形成结构性的 Text 模态放大。单纯提高 Text dropout 概率只能在部分训练样本中删除 Text，不能直接消除 Text 存在时的重复 token 贡献。

### 3.3 研究问题

个人论文可以围绕以下问题展开：

1. 重复 Text token 是否造成不平衡的多模态融合？
2. 将重复 Text 序列压缩成一个 token，能否保持 clean performance？
3. Text 压缩能否改善 Text 缺失或 Text 相关双模态缺失时的鲁棒性？
4. Text compression、reliability gate 和 modality dropout 是否具有互补作用？
5. Text 压缩是否减少模型计算量和平均样本推理时间？

## 4. 方法设计

### 4.1 核心思想

保持磁盘缓存和数据接口中的 Text 特征为 `[10, D]`，只在个人模型内部进行压缩：

```text
Cached Text feature [B, 10, D]
-> temporal mean compression
-> compressed Text feature [B, 1, D]
-> Text projection
-> modality embedding
-> reliability-gated multimodal Transformer
```

压缩操作位于投影和时间位置编码之前，因此重复 Text 不会被不同位置编码人为区分。

### 4.2 数学形式

设重复 Text 特征为：

```text
X_text = {x_1, x_2, ..., x_T}, T = 10
```

重复感知压缩定义为：

```text
x_text_compressed = (1 / T) * sum(x_t), t = 1 ... T
```

投影后的单个 Text token 为：

```text
h_text = Projection_text(x_text_compressed) + Embedding_text
```

压缩后的 Text 不添加 temporal positional embedding，因为它不再表示真实的时间序列，而是一个句子级语义 token。

其他模态保持原团队处理方式：

```text
h_m,t = Projection_m(x_m,t) + TimeEmbedding_t + ModalityEmbedding_m
```

### 4.3 与 Modality Dropout 的组合

个人主方法继续使用团队的整模态 uniform dropout：

- dropout 发生时，将被选模态的完整特征置零；
- 对应 `modality_mask` 设为 unavailable；
- Text 被丢弃时丢弃整个 Text 模态，而不是部分重复时间步；
- 最少保留一个可用模态。

两部分作用不同：

- Text compression：减少 Text 存在时的重复证据放大。
- Modality dropout：训练模型在部分模态不可用时使用其他模态补偿。

### 4.4 缺失模态处理

缺失模态协议保持不变：

```text
zero-fill missing feature
+
set modality_mask to unavailable
```

因此个人方法不修改数据集划分、标签、模态名称、缓存文件格式或缺失模态定义。

## 5. 创新点

论文可以将创新点概括为以下三点。

### 创新点一：识别并处理重复 Text token 的结构性偏置

不同于仅调整 dropout 概率，本方法从输入表示与融合结构的对应关系出发，识别“句向量重复 10 次但被当作 10 个 token 融合”的问题，并在模型内部进行重复感知压缩。

### 创新点二：不破坏现有特征缓存和五模态协议

方法不重新提取 Text 特征，也不改变 `[10, D]` 缓存格式，只在 forward 阶段将 Text 压缩为一个 token。这使方法可以直接接入团队代码，并保证数据处理流程和比较协议一致。

### 创新点三：与可靠性门控和整模态 Dropout 联合

Text compression 处理重复证据，reliability gate 估计模态可靠性，modality dropout 提供缺失模态训练增强。论文通过消融实验分别评估三者的贡献，而不是将简单概率搜索作为主要创新。

## 6. 与团队方法的区别

| 方法 | Text token 数 | Reliability Gate | Modality Dropout |
|---|---:|---:|---:|
| Original Baseline | 10 | No | No |
| Group Improved | 10 | Yes | Yes |
| Compression Only | 1 | No | No |
| Gate + Compression | 1 | Yes | No |
| Compression + Dropout | 1 | No | Yes |
| Proposed | 1 | Yes | Yes |

团队方法默认行为没有被修改：未配置 `use_text_compression` 时使用 `false`，仍保留 10-token Text 表示。

## 7. 代码实现位置

### 7.1 模型实现

文件：

```text
src/models/improved_model.py
```

关键配置：

```yaml
model:
  improved:
    use_text_compression: true
```

关键行为：

- `false`：使用团队模型原始 10-token Text 行为。
- `true`：沿时间维求均值并生成一个 Text token。

### 7.2 实验 runner

文件：

```text
src/training/improved_experiment_runner.py
```

结果指标中增加：

```text
use_text_compression
text_token_count
use_reliability_gate
use_modality_dropout
```

### 7.3 个人配置

文件：

```text
configs/term_paper_text_compression.yaml
```

该配置定义：

- 个人主方法；
- 训练参数；
- clean evaluation；
- Text 相关缺失模态实验；
- 四个个人消融组。

### 7.4 独立运行入口

文件：

```text
experiments/run_term_paper_text_compression.py
```

该入口固定加载个人配置，并把全部生成文件重定向到：

```text
results/term_paper/
```

不要使用团队入口 `experiments/run_improved_model.py` 运行个人正式实验，否则输出仍可能进入团队目录。

## 8. 实验设计

### 8.1 必需比较方法

论文最终表格至少应包含：

1. Original baseline。
2. Group improved model。
3. Proposed personal method。
4. 至少两个能够说明模块贡献的个人消融版本。

时间有限时，优先保留：

```text
Compression Only
Gate + Compression
Proposed
```

若时间允许，再运行：

```text
Compression + Dropout
```

### 8.2 评价条件

个人配置当前包含：

```text
clean
missing_text
missing_imu_text
missing_audio_text
```

选择这些条件的理由：

- `clean`：检查压缩是否损害正常识别性能。
- `missing_text`：评估最直接的 Text 缺失鲁棒性。
- `missing_imu_text`：评估 Text 与运动模态同时缺失。
- `missing_audio_text`：评估两个语音/语义相关模态同时缺失。

如果运行时间仍然不足，最低实验范围为：

```text
clean
missing_text
missing_imu_text
```

### 8.3 主要指标

正式论文至少报告：

- Loss
- Accuracy
- Macro-F1
- Weighted-F1
- Total Training Time
- Average Training Time per Processed Sample
- Total Testing Time
- Average Testing Time per Test Sample

当前 runner 已保存 Accuracy、Macro-F1、Weighted-F1、训练时间及平均测试时间。正式运行后应核对输出 schema 是否完整满足期末论文要求；如果平均训练样本时间尚未输出，应在论文整理前补充 runner 统计，不能手工估计。

由于 clean baseline 曾出现 Accuracy 与 Macro-F1 差距较大的现象，论文分析应优先关注 Macro-F1，避免仅凭 Accuracy 得出结论。

### 8.4 可视化

建议论文保留以下图表：

1. Proposed method 的 confusion matrix。
2. Proposed method 的 train/validation loss curve。
3. Baseline、Group Improved、Proposed 在不同缺失条件下的 Macro-F1 柱状图。
4. 消融实验对比表或柱状图。
5. 若后续增加 gate weight 导出，可绘制五模态平均 reliability score 对比图。

所有正式图表必须由保存的真实 metrics 或 predictions 生成。

## 9. 正式环境运行前检查

本机不是正式运行环境。本机没有原始数据、缓存和依赖，因此不要在本机运行训练，不要安装依赖，也不要使用本机输出作为论文结果。

将修改后的完整仓库上传到正式服务器后，确认：

1. 当前工作目录为仓库根目录。
2. `configs/default.yaml` 中的数据和本地模型路径适合服务器。
3. user A、user B、user C 的数据或已构建样本索引存在。
4. feature cache 完整。
5. 服务器环境已具有 `requirements.txt` 中的依赖。
6. 正式比较方法使用相同 seed、epochs、batch size、learning rate 和数据划分。

不要使用以下结果作为正式论文结果：

- `--smoke-test`；
- `--max-samples`；
- 缩短 epochs 的 debug run；
- synthetic data；
- 未完成或报错的运行。

## 10. 如何运行个人代码

以下命令均应在正式服务器的仓库根目录执行。

### 10.1 推荐运行顺序

第一步：运行个人主方法的 clean 实验。

```bash
python experiments/run_term_paper_text_compression.py --mode clean
```

运行完成后检查：

```text
results/term_paper/metrics/improved_clean_metrics.csv
results/term_paper/logs/improved_clean.log
results/term_paper/figures/improved_clean_loss_curve.png
results/term_paper/figures/improved_clean_confusion_matrix.png
```

第二步：运行 Text 相关缺失模态实验。

```bash
python experiments/run_term_paper_text_compression.py --mode missing
```

第三步：运行消融实验。

```bash
python experiments/run_term_paper_text_compression.py --mode ablation
```

### 10.2 一次运行全部个人实验

```bash
python experiments/run_term_paper_text_compression.py
```

默认配置中 `run_noise: false`，因此不会重复运行完整噪声矩阵。

### 10.3 覆盖训练参数

如需使用与团队正式实验一致的 epochs：

```bash
python experiments/run_term_paper_text_compression.py --mode clean --epochs 5
```

同时覆盖 batch size 和 learning rate：

```bash
python experiments/run_term_paper_text_compression.py --mode clean --epochs 5 --batch-size 64 --lr 0.001
```

如果服务器使用另一份有效基础配置：

```bash
python experiments/run_term_paper_text_compression.py --base-config configs/default.yaml --mode clean
```

个人入口固定使用：

```text
configs/term_paper_text_compression.yaml
```

因此不要额外传入 `--config`。

### 10.4 推荐的完整命令序列

```bash
python experiments/run_term_paper_text_compression.py --mode clean --epochs 5 --batch-size 64 --lr 0.001
python experiments/run_term_paper_text_compression.py --mode missing --epochs 5 --batch-size 64 --lr 0.001
python experiments/run_term_paper_text_compression.py --mode ablation --epochs 5 --batch-size 64 --lr 0.001
```

如果团队正式实验使用的 epochs、batch size 或 learning rate 不同，应把以上值改成团队正式设置，使比较公平。

## 11. 个人实验输出目录

所有个人结果写入：

```text
results/term_paper/
├── metrics/
├── logs/
├── predictions/
├── checkpoints/
├── figures/
├── models/
├── report/
└── configs/
    ├── effective_base.yaml
    └── effective_method.yaml
```

各目录用途：

- `metrics/`：单次指标 CSV、汇总 CSV 和 summary JSON。
- `logs/`：训练与测试日志。
- `predictions/`：逐样本预测结果。
- `checkpoints/`：best 和 final checkpoint。
- `figures/`：loss curve 和 confusion matrix。
- `configs/`：本次运行实际使用的基础配置与个人方法配置快照。

团队结果仍保存在原来的：

```text
results/metrics/
results/logs/
results/predictions/
results/checkpoints/
figures/
```

不要把个人结果移动到团队目录，也不要使用同名文件覆盖团队输出。

## 12. 运行完成后的结果检查

每个正式实验应检查：

1. log 中没有异常或中断。
2. metrics 的 `status` 为 `completed`。
3. `smoke_test` 为 `false`。
4. `use_text_compression` 与目标方法一致。
5. Proposed 的 `text_token_count` 为 1。
6. 团队 improved 对照的 Text token count 为 10。
7. train/test split 仍为 user A + B / user C。
8. epochs、batch size、learning rate、seed 和 device 已记录。
9. predictions 行数与测试样本数一致。
10. loss curve 和 confusion matrix 能正常打开。

建议将每次正式命令、Git commit、服务器设备和输出路径补充到：

```text
docs/experiment_log.md
```

## 13. 如何选择最终结果

不要用 test set 选择训练 checkpoint 或调整方法。应使用固定配置和 validation Macro-F1 选择 best checkpoint。

结果分析可按以下顺序：

1. 比较 clean performance，确认压缩没有造成不可接受的下降。
2. 比较 missing-Text Macro-F1，判断非 Text 模态补偿是否改善。
3. 比较 Text 相关双缺失条件，评估更困难场景。
4. 对比 Compression Only、Gate + Compression、Compression + Dropout 和 Proposed，分析模块贡献。
5. 比较平均测试时间，分析 token 数量从 50/41 附近减少后是否带来效率收益。实际总 token 数应以代码和真实配置核对后再写入论文。

如果 Proposed 没有在所有条件下最高，也应如实报告。可以重点分析 robustness/clean trade-off，而不能删除不利结果或只报告最优条件。

## 14. 论文结构建议

论文必须为英文，4 或 5 页且不超过 5 页。建议结构如下。

### Abstract

包含：

1. AR 多模态意图识别背景。
2. 重复 Text token 问题。
3. Repetition-Aware Text Compression 方法。
4. Reliability gate 和 modality dropout 组合。
5. 数据划分和评价条件。
6. 真实运行后的主要发现。

在实验完成前不要填写具体提升数字。

### 1. Introduction

建议逻辑：

1. AR 交互需要融合动作、语音、文本、场景等信息。
2. 真实系统可能出现噪声和模态缺失。
3. 团队方法通过可靠性门控和模态 dropout 提高鲁棒性。
4. 现有 Text 句向量被重复为 10 个 token，可能产生重复证据偏置。
5. 提出模型侧 Text 压缩并通过消融验证。

可将贡献写成：

```text
1. We identify a representation-fusion mismatch caused by repeating a sentence embedding over multiple time steps.
2. We propose a repetition-aware Text compression module that preserves the cached feature protocol while using a single semantic token during fusion.
3. We conduct controlled ablations with reliability-gated fusion and modality dropout under clean and Text-related missing-modality conditions.
```

### 2. Related Work

建议包含三小类：

1. Multimodal intention recognition。
2. Missing-modality robust learning and modality dropout。
3. Reliability-aware or gated multimodal fusion。

如引用外部文献，必须核对原论文内容和引用信息。

### 3. Method

建议包括：

1. Problem formulation。
2. Group reliability-gated baseline。
3. Repetition-aware Text compression。
4. Combination with modality dropout。
5. Training objective。

方法图建议画成：

```text
IMU sequence --------> IMU projection -----------\
Gesture sequence ----> Gesture projection --------\
Audio sequence ------> Audio projection -----------> Reliability-Gated Transformer -> Classifier
Text [10, D] -> Mean -> Text [1, D] -> Projection --/
Scene vector --------> Scene projection ----------/
```

### 4. Experiments and Results

至少说明：

- Dataset split：Train A+B, Test C。
- Five modalities。
- Missing-modality settings。
- Training settings。
- Accuracy、Macro-F1、Weighted-F1。
- Average training/testing time per sample。
- Baseline、Group Improved、Proposed、Ablations。

建议主表：

| Method | Clean Macro-F1 | Missing Text Macro-F1 | Missing IMU+Text Macro-F1 | Missing Audio+Text Macro-F1 |
|---|---:|---:|---:|---:|
| Baseline | 待真实结果 | 待真实结果 | 待真实结果 | 待真实结果 |
| Group Improved | 待真实结果 | 待真实结果 | 待真实结果 | 待真实结果 |
| Proposed | 待真实结果 | 待真实结果 | 待真实结果 | 待真实结果 |

占位文本只能用于规划，提交论文前必须替换为真实结果。

建议消融表：

| Compression | Gate | Dropout | Accuracy | Macro-F1 | Avg Test Time/Sample |
|---:|---:|---:|---:|---:|---:|
| Yes | No | No | 待真实结果 | 待真实结果 | 待真实结果 |
| Yes | Yes | No | 待真实结果 | 待真实结果 | 待真实结果 |
| Yes | No | Yes | 待真实结果 | 待真实结果 | 待真实结果 |
| Yes | Yes | Yes | 待真实结果 | 待真实结果 | 待真实结果 |

### 5. Analysis

分析时重点讨论：

- Text compression 是否保持 clean performance。
- Missing Text 下是否真正改善 Macro-F1。
- Accuracy 与 Macro-F1 是否存在明显差异。
- 哪些类别从压缩中获益，结合 confusion matrix 分析。
- Gate 和 dropout 是否与 compression 互补。
- 单 token Text 是否降低测试时间。
- 方法无法恢复完全缺失的语义信息这一限制。

### 6. Conclusion

概括：

1. 发现的重复表示问题。
2. 提出的模型侧压缩方法。
3. 真实实验支持的结论。
4. 方法限制和后续工作。

## 15. 可复用英文方法描述

### 15.1 方法摘要

> The existing Text pipeline repeats a sentence-level embedding over ten time steps to match a unified multimodal feature shape. However, treating these repeated vectors as independent Transformer tokens may amplify the contribution of Text during multimodal fusion. We introduce repetition-aware Text compression, which averages the repeated Text sequence into a single semantic token before feature projection and positional encoding. The cached feature format and the five-modality data protocol remain unchanged. The compressed Text representation is further combined with reliability-gated fusion and whole-modality dropout to improve robustness to missing modalities.

### 15.2 创新点描述

> Unlike probability-only modality dropout strategies, our approach directly addresses the representation-fusion mismatch introduced by repeated sentence embeddings. The proposed module is lightweight, model-side, and compatible with the original feature cache, enabling a controlled comparison with the group baseline without changing the dataset or feature extraction pipeline.

### 15.3 消融描述

> We evaluate the independent and combined contributions of repetition-aware Text compression, reliability-gated fusion, and modality dropout. All variants use the same user-independent split, training settings, and evaluation protocol. The comparison includes clean data, missing Text, and Text-related two-modality missing conditions.

### 15.4 限制描述

> Text compression reduces duplicated evidence but cannot reconstruct semantic information when the Text modality is entirely unavailable. Its effectiveness under missing Text therefore depends on whether the remaining modalities contain sufficient complementary cues for intention recognition.

## 16. 结果解释边界

以下表述只有真实结果支持时才能写：

- “The proposed method significantly improves ...”
- “The method consistently outperforms ...”
- “Text compression reduces inference time by X%.”
- “The proposed model is more robust under all missing conditions.”

如果结果只在部分条件改善，应写：

> The proposed method improves robustness under selected Text-related missing-modality conditions while showing a trade-off on clean data.

如果 clean performance 保持、缺失性能提升有限，可以写：

> The method provides a more balanced token-level representation without substantially degrading clean-data performance, although the gain under complete Text absence remains limited.

如果方法效果不如团队模型，也不能隐瞒；可以分析重复压缩导致的信息交互减少，或其他模态本身缺乏足够语义信息。

## 17. 当前状态和后续任务

当前已完成：

- 模型侧 Text compression 开关。
- 个人方法配置。
- 消融配置。
- 独立运行入口。
- `results/term_paper/` 输出隔离。
- 指标元数据扩展。
- AST 静态语法检查。

当前未完成：

- 正式服务器运行。
- 个人方法真实 metrics。
- 平均训练时间/样本的完整核对。
- gate weight 导出和可视化。
- 论文 LaTeX 初稿。

建议后续顺序：

1. 正式运行 clean。
2. 检查 clean Macro-F1 和输出完整性。
3. 正式运行 missing。
4. 正式运行 ablation。
5. 汇总真实结果并生成论文对比图。
6. 根据真实证据编写英文论文。

