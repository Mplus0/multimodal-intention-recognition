# 实验记录

本文件由 Mplus0 维护，用于记录 clean baseline、modal noise baseline、missing-modality baseline、改进模型实验和消融实验。代码标识符、文件路径、命令、实验名和指标名保留英文原文。

当前阶段成员 A 的端到端数据流尚未完成，因此本文件先记录实验计划、实验矩阵、统一指标字段和待填写模板。所有实验结果必须在真实运行后从 `results/metrics/`、`results/logs/`、`results/predictions/` 和 `figures/` 中同步，不得凭空填写。

---

## 统一实验设置

### 数据划分
- 训练集：user A + user B
- 测试集：user C

### 模态列表
- imu
- gesture
- audio
- text
- scene

### 指标字段
| 指标 | 说明 |
|---|---|
| Accuracy | 意图分类准确率 |
| Macro F1 | 各类别等权平均 F1 |
| Weighted F1 | 按类别样本数加权的 F1 |
| Training Time | 总训练时间 |
| Avg Test Time / Sample | 平均单样本测试时间 |

### 预期输出字段
| 字段 | 说明 |
|---|---|
| experiment_name | 实验名称 |
| model_type | baseline / improved / ablation variant |
| target_modality | none / imu / gesture / audio / text / scene |
| noise_ratio | none / 0.2 / 0.4 / 0.6 |
| missing_modalities | none / 单模态 / 双模态组合 |
| accuracy | 实验完成后填写 |
| macro_f1 | 实验完成后填写 |
| weighted_f1 | 实验完成后填写 |
| training_time | 实验完成后填写 |
| avg_test_time_per_sample | 实验完成后填写 |
| status | pending / completed / failed |
| notes | 实验备注 |

---

## 待执行实验矩阵

### Clean Baseline
| 实验 | 模型 | 噪声 | 缺失模态 | 状态 |
|---|---|---|---|---|
| clean_baseline | baseline | none | none | Pending |

### Modal Noise Baseline
| 实验 | 加噪模态 | 噪声比例 | 模型 | 状态 |
|---|---|---:|---|---|
| noise_imu_20 | imu | 20% | baseline | Pending |
| noise_imu_40 | imu | 40% | baseline | Pending |
| noise_imu_60 | imu | 60% | baseline | Pending |
| noise_gesture_20 | gesture | 20% | baseline | Pending |
| noise_gesture_40 | gesture | 40% | baseline | Pending |
| noise_gesture_60 | gesture | 60% | baseline | Pending |
| noise_audio_20 | audio | 20% | baseline | Pending |
| noise_audio_40 | audio | 40% | baseline | Pending |
| noise_audio_60 | audio | 60% | baseline | Pending |
| noise_text_20 | text | 20% | baseline | Pending |
| noise_text_40 | text | 40% | baseline | Pending |
| noise_text_60 | text | 60% | baseline | Pending |
| noise_scene_20 | scene | 20% | baseline | Pending |
| noise_scene_40 | scene | 40% | baseline | Pending |
| noise_scene_60 | scene | 60% | baseline | Pending |

### Missing Modality Baseline
| 实验 | 缺失模态 | 模型 | 状态 |
|---|---|---|---|
| missing_imu | imu | baseline | Pending |
| missing_gesture | gesture | baseline | Pending |
| missing_audio | audio | baseline | Pending |
| missing_text | text | baseline | Pending |
| missing_scene | scene | baseline | Pending |
| missing_imu_gesture | imu + gesture | baseline | Pending |
| missing_imu_audio | imu + audio | baseline | Pending |
| missing_imu_text | imu + text | baseline | Pending |
| missing_imu_scene | imu + scene | baseline | Pending |
| missing_gesture_audio | gesture + audio | baseline | Pending |
| missing_gesture_text | gesture + text | baseline | Pending |
| missing_gesture_scene | gesture + scene | baseline | Pending |
| missing_audio_text | audio + text | baseline | Pending |
| missing_audio_scene | audio + scene | baseline | Pending |
| missing_text_scene | text + scene | baseline | Pending |

### 改进模型与消融实验
| 实验 | 模型变体 | 目的 | 状态 |
|---|---|---|---|
| improved_model | improved | 对比 baseline，验证改进模型整体效果 | Pending |
| ablation_modality_dropout | modality dropout only | 验证 modality dropout 对缺失模态鲁棒性的贡献 | Pending |
| ablation_reliability_gate | reliability gate only | 验证 reliability gate 对噪声和缺失条件的贡献 | Pending |
| ablation_dropout_gate | modality dropout + reliability gate | 验证两种改进组合后的效果 | Pending |

---

## 实验：Clean Baseline

### 日期
待填写

### 目的
在无噪声、无模态缺失条件下训练并测试 baseline 模型，获得后续噪声实验、缺失模态实验和改进模型实验的对照结果。

### 命令
```bash
python experiments/run_clean_baseline.py --config configs/clean_baseline.yaml
```

### 数据划分
- 训练集：user A + user B
- 测试集：user C

### 设置
- 模态：imu, gesture, audio, text, scene
- 噪声设置：none
- 缺失模态：none
- 模型：baseline
- 当前状态：Pending

### 主要结果
| 指标 | 数值 |
|---|---:|
| Accuracy | TBD |
| Macro F1 | TBD |
| Weighted F1 | TBD |
| Training Time | TBD |
| Avg Test Time / Sample | TBD |

### 输出路径
- 指标：`results/metrics/clean_baseline_metrics.csv`
- 日志：`results/logs/clean_baseline.log`
- 预测：`results/predictions/clean_baseline_predictions.csv`
- 图表：`figures/`
- checkpoint：TBD

### 简要分析
待真实实验完成后填写。不得在未运行实验前填写结论。

### 问题
- `configs/clean_baseline.yaml` 当前不存在，脚本会给出清晰错误提示，不会强制创建。
- 成员 A 的训练和测试接口尚未稳定。
- 当前未运行该实验。

---

## 实验：Modal Noise Baseline

### 日期
2026-06-23

### 目的
分别对每一个单模态加入 20%、40%、60% 噪声，准备 baseline 模型在不同单模态噪声条件下的 pending 实验矩阵。当前只准备框架，不实现真实加噪逻辑。

### 命令
```bash
python experiments/run_noise_baseline.py --config configs/noise.yaml
```

### 数据划分
- 训练集：user A + user B
- 测试集：user C

### 设置
- 模态：imu, gesture, audio, text, scene
- 噪声设置：每次只对一个模态加噪，比例为 20%、40%、60%
- 缺失模态：none
- 模型：baseline
- 当前状态：Pending

### 主要结果
| 指标 | 数值 |
|---|---:|
| Accuracy | TBD |
| Macro F1 | TBD |
| Weighted F1 | TBD |
| Training Time | TBD |
| Avg Test Time / Sample | TBD |

### 输出路径
- 指标：`results/metrics/noise_baseline_metrics.csv`
- 日志：`results/logs/modal_noise_baseline.log`
- 预测：`results/predictions/noise_baseline_predictions.csv`
- 图表：`figures/`
- checkpoint：TBD

### 简要分析
当前只完成配置和实验框架准备，没有真实实验结果。pending CSV 中每组实验状态为 `pending`，指标为 `TBD`。后续需要等待成员 A 的数据变换和 train/test 接口稳定后，再决定噪声作用于 raw data 还是 extracted features。

### 问题
- 真实加噪逻辑尚未实现。
- 成员 A 的 train/test/Dataset/DataLoader/model 接口尚未稳定。
- 当前未运行真实训练或测试实验。

---

## 实验：Missing Modality Baseline

### 日期
2026-06-23

### 目的
准备单模态缺失和双模态缺失的 baseline pending 实验矩阵，用于后续分析模型在模态缺失条件下的鲁棒性。当前只准备框架，不实现真实模态缺失处理逻辑。

### 命令
```bash
python experiments/run_missing_baseline.py --config configs/missing_modality.yaml
```

### 数据划分
- 训练集：user A + user B
- 测试集：user C

### 设置
- 模态：imu, gesture, audio, text, scene
- 噪声设置：none
- 缺失模态：单模态缺失和双模态缺失
- 模型：baseline
- 当前状态：Pending

### 缺失模态组合
- 单模态：imu, gesture, audio, text, scene
- 双模态：imu + gesture, imu + audio, imu + text, imu + scene, gesture + audio, gesture + text, gesture + scene, audio + text, audio + scene, text + scene

### 主要结果
| 指标 | 数值 |
|---|---:|
| Accuracy | TBD |
| Macro F1 | TBD |
| Weighted F1 | TBD |
| Training Time | TBD |
| Avg Test Time / Sample | TBD |

### 输出路径
- 指标：`results/metrics/missing_modality_metrics.csv`
- 日志：`results/logs/missing_modality_baseline.log`
- 预测：`results/predictions/missing_modality_predictions.csv`
- 图表：`figures/`
- checkpoint：TBD

### 简要分析
当前只完成配置和实验框架准备，没有真实实验结果。pending CSV 中每组实验状态为 `pending`，指标为 `TBD`。后续需要等待成员 A 的数据接口稳定后，再决定 missing modality 是 zero-fill、mask 还是删除输入字段。

### 问题
- 真实 missing modality 处理逻辑尚未实现。
- 成员 A 的 train/test/Dataset/DataLoader/model 接口尚未稳定。
- 当前未运行真实训练或测试实验。

---

## 实验：Improved Model

### 日期
待填写

### 目的
在端到端流程稳定后，引入面向模态噪声和模态缺失的模型改进，并与 baseline 结果进行对比。

### 命令
```bash
python experiments/run_improved_model.py --config configs/default.yaml
```

### 数据划分
- 训练集：user A + user B
- 测试集：user C

### 设置
- 模态：imu, gesture, audio, text, scene
- 噪声设置：待按具体实验填写
- 缺失模态：待按具体实验填写
- 模型：improved
- 当前状态：Pending

### 主要结果
| 指标 | 数值 |
|---|---:|
| Accuracy | TBD |
| Macro F1 | TBD |
| Weighted F1 | TBD |
| Training Time | TBD |
| Avg Test Time / Sample | TBD |

### 输出路径
- 指标：`results/metrics/improved_model_metrics.csv`
- 日志：`results/logs/improved_model.log`
- 预测：`results/predictions/improved_model_predictions.csv`
- 图表：`figures/`
- checkpoint：TBD

### 简要分析
待真实实验完成后填写。需要与 clean baseline、noise baseline 和 missing modality baseline 对比，说明改进是否有效。

### 问题
- 改进模型具体实现尚未完成。
- 当前未运行该实验。

---

## 实验：Ablation Study

### 日期
待填写

### 目的
对改进模型中的关键模块进行消融实验，分析每个模块对最终性能的贡献。

### 命令
```bash
python experiments/run_improved_model.py --config configs/default.yaml
```

### 数据划分
- 训练集：user A + user B
- 测试集：user C

### 设置
- 模态：imu, gesture, audio, text, scene
- 噪声设置：待按具体实验填写
- 缺失模态：待按具体实验填写
- 模型：ablation variants
- 当前状态：Pending

### 主要结果
| 指标 | 数值 |
|---|---:|
| Accuracy | TBD |
| Macro F1 | TBD |
| Weighted F1 | TBD |
| Training Time | TBD |
| Avg Test Time / Sample | TBD |

### 输出路径
- 指标：`results/metrics/ablation_metrics.csv`
- 日志：`results/logs/ablation.log`
- 预测：`results/predictions/ablation_predictions.csv`
- 图表：`figures/`
- checkpoint：TBD

### 简要分析
待真实实验完成后填写。需要说明每个改进模块是否带来稳定收益。

### 问题
- 消融实验配置尚未完成。
- 当前未运行该实验。
## Term Paper Text Compression - First Full Run Protocol Audit

### Date
2026-07-13

### Purpose
审查首次 Repetition-Aware Text Compression 全量运行是否满足个人论文协议。

### Command
```bash
python experiments/run_term_paper_text_compression.py --epochs 5 --batch-size 64 --lr 0.001
```

### Dataset Split
- Train: user A + user B
- Test: user C

### Settings
- Modalities: imu, gesture, audio, text, scene
- Noise setting: none
- Missing modality: none / text / imu+text / audio+text
- Model: repetition-aware Text compression variants

### Brief Analysis
首次运行完成 8 个真实数据实验且输出隔离正常，但测试阶段使用训练结束时的模型，没有重新加载 validation Macro-F1 最佳 checkpoint。该次运行还缺少规范化平均训练时间字段，且消融只覆盖 clean。因此结果可用于诊断，不作为最终论文正式表格；代码修复后需要重跑。

### Problems
- Final test 未加载 best checkpoint。
- 缺少 `avg_training_time_per_sample_sec` 和 `testing_time_total_sec`。
- Ablation 未覆盖 `missing_text`。

### Official Result Status
- Debug only / Superseded by required rerun
