# 实验记录

本文档由 Mplus0 维护，用于记录清洁数据 baseline、模态噪声 baseline、模态缺失 baseline、改进模型实验和消融实验。

当前阶段成员 A 的端到端数据流尚未完成，因此本文件先记录实验计划、实验矩阵、统一指标字段和待填写模板。所有实验结果必须在真实运行后从 `results/metrics/`、`results/logs/`、`results/predictions/` 和 `figures/` 中同步，不得凭空填写。

---

## 统一实验设置

### Dataset Split
- Train: user A + user B
- Test: user C

### Modalities
- imu
- gesture
- audio
- text
- scene

### Metrics
| Metric | Description |
|---|---|
| Accuracy | 意图分类准确率 |
| Macro F1 | 各类别等权平均 F1 |
| Weighted F1 | 按类别样本数加权的 F1 |
| Training Time | 总训练时间 |
| Avg Train Time / Sample | 平均单样本训练时间 |
| Avg Test Time / Sample | 平均单样本测试时间 |

### Expected Output Fields
| Field | Description |
|---|---|
| experiment_name | 实验名称 |
| model_name | baseline / improved / ablation variant |
| train_users | user A + user B |
| test_user | user C |
| noise_modality | none / imu / gesture / audio / text / scene |
| noise_ratio | none / 20% / 40% / 60% |
| missing_modalities | none / 单模态 / 双模态组合 |
| accuracy | 待实验完成后填写 |
| macro_f1 | 待实验完成后填写 |
| weighted_f1 | 待实验完成后填写 |
| training_time | 待实验完成后填写 |
| avg_train_time_per_sample | 待实验完成后填写 |
| avg_test_time_per_sample | 待实验完成后填写 |
| metrics_path | `results/metrics/xxx.csv` |
| log_path | `results/logs/xxx.log` |
| prediction_path | `results/predictions/xxx.csv` |
| figure_path | `figures/result_charts/xxx.png` |
| checkpoint_path | `results/checkpoints/xxx.pt` |
| notes | 实验备注 |

---

## 待执行实验矩阵

### Clean Baseline
| Experiment | Model | Noise | Missing Modality | Status |
|---|---|---|---|---|
| clean_baseline | baseline | none | none | 未运行 |

### Modal Noise Baseline
| Experiment | Noise Modality | Noise Ratio | Model | Status |
|---|---|---:|---|---|
| noise_imu_20 | imu | 20% | baseline | 未运行 |
| noise_imu_40 | imu | 40% | baseline | 未运行 |
| noise_imu_60 | imu | 60% | baseline | 未运行 |
| noise_gesture_20 | gesture | 20% | baseline | 未运行 |
| noise_gesture_40 | gesture | 40% | baseline | 未运行 |
| noise_gesture_60 | gesture | 60% | baseline | 未运行 |
| noise_audio_20 | audio | 20% | baseline | 未运行 |
| noise_audio_40 | audio | 40% | baseline | 未运行 |
| noise_audio_60 | audio | 60% | baseline | 未运行 |
| noise_text_20 | text | 20% | baseline | 未运行 |
| noise_text_40 | text | 40% | baseline | 未运行 |
| noise_text_60 | text | 60% | baseline | 未运行 |
| noise_scene_20 | scene | 20% | baseline | 未运行 |
| noise_scene_40 | scene | 40% | baseline | 未运行 |
| noise_scene_60 | scene | 60% | baseline | 未运行 |

### Missing Modality Baseline
| Experiment | Missing Modality | Model | Status |
|---|---|---|---|
| missing_imu | imu | baseline | 未运行 |
| missing_gesture | gesture | baseline | 未运行 |
| missing_audio | audio | baseline | 未运行 |
| missing_text | text | baseline | 未运行 |
| missing_scene | scene | baseline | 未运行 |
| missing_imu_gesture | imu + gesture | baseline | 未运行 |
| missing_imu_audio | imu + audio | baseline | 未运行 |
| missing_imu_text | imu + text | baseline | 未运行 |
| missing_imu_scene | imu + scene | baseline | 未运行 |
| missing_gesture_audio | gesture + audio | baseline | 未运行 |
| missing_gesture_text | gesture + text | baseline | 未运行 |
| missing_gesture_scene | gesture + scene | baseline | 未运行 |
| missing_audio_text | audio + text | baseline | 未运行 |
| missing_audio_scene | audio + scene | baseline | 未运行 |
| missing_text_scene | text + scene | baseline | 未运行 |

### Improved Model And Ablation
| Experiment | Model Variant | Purpose | Status |
|---|---|---|---|
| improved_model | improved | 对比 baseline，验证改进模型整体效果 | 未运行 |
| ablation_modality_dropout | modality dropout only | 验证模态 Dropout 对缺失模态鲁棒性的贡献 | 未运行 |
| ablation_reliability_gate | reliability gate only | 验证可靠性门控对噪声和缺失条件的贡献 | 未运行 |
| ablation_dropout_gate | modality dropout + reliability gate | 验证两种改进组合后的效果 | 未运行 |

---

## Experiment: Clean Baseline

### Date
待填写

### Purpose
在无噪声、无模态缺失条件下训练并测试 baseline 模型，获得后续噪声实验、缺失模态实验和改进模型实验的对照结果。

### Command
```bash
python experiments/run_clean_baseline.py --config configs/default.yaml
```

### Dataset Split
- Train: user A + user B
- Test: user C

### Settings
- Modalities: imu, gesture, audio, text, scene
- Noise setting: none
- Missing modality: none
- Model: baseline

### Main Results
| Metric | Value |
|---|---:|
| Accuracy | 待填写 |
| Macro F1 | 待填写 |
| Weighted F1 | 待填写 |
| Training Time | 待填写 |
| Avg Train Time / Sample | 待填写 |
| Avg Test Time / Sample | 待填写 |

### Output Paths
- Metrics: `results/metrics/clean_baseline_metrics.csv`
- Log: `results/logs/clean_baseline.log`
- Prediction: `results/predictions/clean_baseline_predictions.csv`
- Figure: `figures/result_charts/clean_confusion_matrix.png`
- Checkpoint: `results/checkpoints/clean_baseline_best.pt`

### Brief Analysis
待实验完成后填写。不得在未运行实验前填写结论。

### Problems
- 端到端训练和测试入口尚未完成。
- 当前未运行该实验。

---

## Experiment: Modal Noise Baseline

### Date
待填写

### Purpose
分别对每个单模态加入 20%、40%、60% 噪声，训练 baseline 模型并在 user C 测试集上评估，分析模型对不同模态噪声的鲁棒性。

### Command
```bash
python experiments/run_noise_baseline.py --config configs/noise.yaml
```

### Dataset Split
- Train: user A + user B
- Test: user C

### Settings
- Modalities: imu, gesture, audio, text, scene
- Noise setting: single modality noise at 20%, 40%, 60%
- Missing modality: none
- Model: baseline

### Main Results
| Metric | Value |
|---|---:|
| Accuracy | 待填写 |
| Macro F1 | 待填写 |
| Weighted F1 | 待填写 |
| Training Time | 待填写 |
| Avg Train Time / Sample | 待填写 |
| Avg Test Time / Sample | 待填写 |

### Output Paths
- Metrics: `results/metrics/noise_baseline_metrics.csv`
- Log: `results/logs/noise_baseline.log`
- Prediction: `results/predictions/noise_baseline_predictions.csv`
- Figure: `figures/result_charts/noise_accuracy_comparison.png`
- Checkpoint: `results/checkpoints/noise_baseline_best.pt`

### Brief Analysis
待实验完成后填写。需要比较不同模态、不同噪声比例下 Accuracy 和 Macro F1 的下降趋势。

### Problems
- `configs/noise.yaml` 尚未完成。
- 噪声注入函数尚未接入端到端流程。
- 当前未运行该实验。

---

## Experiment: Missing Modality Baseline

### Date
待填写

### Purpose
分别丢弃任意一个模态和任意两个模态，训练 baseline 模型并在 user C 测试集上评估，分析模型在模态缺失条件下的鲁棒性。

### Command
```bash
python experiments/run_missing_baseline.py --config configs/missing_modality.yaml
```

### Dataset Split
- Train: user A + user B
- Test: user C

### Settings
- Modalities: imu, gesture, audio, text, scene
- Noise setting: none
- Missing modality: single modality and modality pairs
- Model: baseline

### Main Results
| Metric | Value |
|---|---:|
| Accuracy | 待填写 |
| Macro F1 | 待填写 |
| Weighted F1 | 待填写 |
| Training Time | 待填写 |
| Avg Train Time / Sample | 待填写 |
| Avg Test Time / Sample | 待填写 |

### Output Paths
- Metrics: `results/metrics/missing_modality_metrics.csv`
- Log: `results/logs/missing_modality.log`
- Prediction: `results/predictions/missing_modality_predictions.csv`
- Figure: `figures/result_charts/missing_modality_accuracy_comparison.png`
- Checkpoint: `results/checkpoints/missing_modality_best.pt`

### Brief Analysis
待实验完成后填写。需要分析哪些单模态或双模态缺失对性能影响最大。

### Problems
- `configs/missing_modality.yaml` 尚未完成。
- 缺失模态处理函数尚未接入端到端流程。
- 当前未运行该实验。

---

## Experiment: Improved Model

### Date
待填写

### Purpose
在重构后的端到端流程基础上，引入面向模态噪声和模态缺失的模型改进，并与 baseline 结果进行对比。

### Command
```bash
python experiments/run_improved_model.py --config configs/default.yaml
```

### Dataset Split
- Train: user A + user B
- Test: user C

### Settings
- Modalities: imu, gesture, audio, text, scene
- Noise setting: 待按具体实验填写
- Missing modality: 待按具体实验填写
- Model: improved

### Main Results
| Metric | Value |
|---|---:|
| Accuracy | 待填写 |
| Macro F1 | 待填写 |
| Weighted F1 | 待填写 |
| Training Time | 待填写 |
| Avg Train Time / Sample | 待填写 |
| Avg Test Time / Sample | 待填写 |

### Output Paths
- Metrics: `results/metrics/improved_model_metrics.csv`
- Log: `results/logs/improved_model.log`
- Prediction: `results/predictions/improved_model_predictions.csv`
- Figure: `figures/result_charts/improved_model_comparison.png`
- Checkpoint: `results/checkpoints/improved_model_best.pt`

### Brief Analysis
待实验完成后填写。需要与 clean baseline、noise baseline 和 missing modality baseline 对比，说明改进是否有效。

### Problems
- 改进模型具体实现尚未完成。
- 当前未运行该实验。

---

## Experiment: Ablation Study

### Date
待填写

### Purpose
对改进模型中的关键模块进行消融实验，分析每个模块对最终性能的贡献。

### Command
```bash
python experiments/run_improved_model.py --config configs/default.yaml
```

### Dataset Split
- Train: user A + user B
- Test: user C

### Settings
- Modalities: imu, gesture, audio, text, scene
- Noise setting: 待按具体实验填写
- Missing modality: 待按具体实验填写
- Model: ablation variants

### Main Results
| Metric | Value |
|---|---:|
| Accuracy | 待填写 |
| Macro F1 | 待填写 |
| Weighted F1 | 待填写 |
| Training Time | 待填写 |
| Avg Train Time / Sample | 待填写 |
| Avg Test Time / Sample | 待填写 |

### Output Paths
- Metrics: `results/metrics/ablation_metrics.csv`
- Log: `results/logs/ablation.log`
- Prediction: `results/predictions/ablation_predictions.csv`
- Figure: `figures/result_charts/ablation_comparison.png`
- Checkpoint: `results/checkpoints/ablation_best.pt`

### Brief Analysis
待实验完成后填写。需要说明每个改进模块是否带来稳定收益。

### Problems
- 消融实验配置尚未完成。
- 当前未运行该实验。

---

## 基础实验配置准备记录

### Date
2026-06-23

### Current Status
当前处于配置准备阶段，尚未运行任何 baseline 实验。本节只记录 clean baseline、Modal Noise Baseline 和 Missing Modality Baseline 的配置说明与实验记录模板。所有指标结果均为 TBD，不代表真实实验结果。

---

## Baseline Template: Clean Baseline

### Purpose
记录无噪声、无模态缺失条件下的 baseline 实验设置。该实验用于获得后续 Modal Noise Baseline 和 Missing Modality Baseline 的对照结果。

### Command
```bash
python experiments/run_clean_baseline.py --config configs/default.yaml
```

### Dataset Split
- Train: user A + user B
- Test: user C

### Settings
- Modalities: imu, gesture, audio, text, scene
- Noise setting: none
- Missing modality: none
- Model: baseline
- Current status: 配置准备阶段，尚未运行实验

### Main Results
| Metric | Value |
|---|---:|
| Accuracy | TBD |
| Macro F1 | TBD |
| Weighted F1 | TBD |
| Training Time | TBD |
| Avg Train Time / Sample | TBD |
| Avg Test Time / Sample | TBD |

### Expected Output Paths
- Metrics: `results/metrics/clean_baseline_metrics.csv`
- Log: `results/logs/clean_baseline.log`
- Prediction: `results/predictions/clean_baseline_predictions.csv`
- Figure: `figures/result_charts/clean_confusion_matrix.png`

### Notes
- 本模板只用于记录基础 baseline 实验。
- 当前没有真实指标结果。
- 后续需要等待成员 A 的 `train.py` / `test.py` 接口稳定后，再编写并运行 `experiments/run_clean_baseline.py`。

---

## Baseline Template: Modal Noise Baseline

### Purpose
记录 Modal Noise Baseline 的配置说明。该实验用于评估 baseline 模型在单模态噪声条件下的鲁棒性。

### Config
- Config file: `configs/noise.yaml`

### Command
```bash
python experiments/run_noise_baseline.py --config configs/noise.yaml
```

### Dataset Split
- Train: user A + user B
- Test: user C

### Settings
- Modalities: imu, gesture, audio, text, scene
- Noise setting: 每次只对一个模态加入噪声，其他模态保持 clean
- Noise ratios: 20%, 40%, 60%
- Missing modality: none
- Model: baseline
- Current status: 配置准备阶段，尚未运行实验

### Main Results
| Metric | Value |
|---|---:|
| Accuracy | TBD |
| Macro F1 | TBD |
| Weighted F1 | TBD |
| Training Time | TBD |
| Avg Train Time / Sample | TBD |
| Avg Test Time / Sample | TBD |

### Expected Output Paths
- Metrics: `results/metrics/noise_baseline_metrics.csv`
- Log: `results/logs/noise_baseline.log`
- Prediction: `results/predictions/noise_baseline_predictions.csv`
- Figure: `figures/result_charts/noise_accuracy_comparison.png`

### Notes
- 本模板只用于记录 modal noise baseline。
- 当前没有真实指标结果。
- 后续需要等待成员 A 的 `train.py` / `test.py` 接口稳定后，再编写并运行 `experiments/run_noise_baseline.py`。

---

## Baseline Template: Missing Modality Baseline

### Purpose
记录 Missing Modality Baseline 的配置说明。该实验用于评估 baseline 模型在单模态缺失和双模态缺失条件下的鲁棒性。

### Config
- Config file: `configs/missing_modality.yaml`

### Command
```bash
python experiments/run_missing_baseline.py --config configs/missing_modality.yaml
```

### Dataset Split
- Train: user A + user B
- Test: user C

### Settings
- Modalities: imu, gesture, audio, text, scene
- Noise setting: none
- Missing modality: single modality and modality pairs
- Model: baseline
- Current status: 配置准备阶段，尚未运行实验

### Missing Modality Groups
- Single: imu, gesture, audio, text, scene
- Pairs: imu + gesture, imu + audio, imu + text, imu + scene, gesture + audio, gesture + text, gesture + scene, audio + text, audio + scene, text + scene

### Main Results
| Metric | Value |
|---|---:|
| Accuracy | TBD |
| Macro F1 | TBD |
| Weighted F1 | TBD |
| Training Time | TBD |
| Avg Train Time / Sample | TBD |
| Avg Test Time / Sample | TBD |

### Expected Output Paths
- Metrics: `results/metrics/missing_modality_metrics.csv`
- Log: `results/logs/missing_modality.log`
- Prediction: `results/predictions/missing_modality_predictions.csv`
- Figure: `figures/result_charts/missing_modality_accuracy_comparison.png`

### Notes
- 本模板只用于记录 missing-modality baseline。
- 当前没有真实指标结果。
- 后续需要等待成员 A 的 `train.py` / `test.py` 接口稳定后，再编写并运行 `experiments/run_missing_baseline.py`。
