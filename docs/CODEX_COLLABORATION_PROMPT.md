# Codex 协作提示词与日志更新规范

本文件用于指导 Codex 协助完成机器学习课程项目代码开发，并保证每次修改后都能同步更新日志，方便团队协作和后续报告编写。

---

## 1. 使用目标

使用 Codex 时，不只是让它修改代码，还要让它完成以下协作任务：

1. 修改代码前先阅读仓库结构和相关说明文件。
2. 每次只完成一个明确的小任务，不一次性大范围重构。
3. 修改后给出运行命令。
4. 修改后更新实验日志或协作日志。
5. 记录改动文件、运行结果、输出路径和未解决问题。
6. 保证文稿负责人可以直接根据日志整理报告。

---

## 2. 推荐放入仓库的 Codex 说明文件

建议在仓库根目录新增以下文件之一：

```text
AGENTS.md
```

或：

```text
.codex/project-instructions.md
```

如果 Codex 优先读取 `AGENTS.md`，推荐使用 `AGENTS.md`；如果你习惯使用 `.codex` 文件夹，也可以把相同内容放入 `.codex/project-instructions.md`。

建议目录：

```text
multimodal-intention-recognition/
├── AGENTS.md
└── .codex/
    └── project-instructions.md
```

两者不需要同时维护，选择一种即可。如果同时存在，内容应保持一致。

---

## 3. 可直接复制到 `AGENTS.md` 的内容

```markdown
# Codex Project Instructions

## Project Overview

This repository is for a Machine Learning course project: multimodal user intention recognition in AR interaction scenarios.

The project must refactor the teacher-provided feature-based workflow into an end-to-end workflow:

raw multimodal data -> preprocessing / feature extraction -> multimodal fusion model -> intention label output

The project must evaluate:

1. Clean baseline.
2. Modal noise baseline at 20%, 40%, and 60% noise ratios for each single modality.
3. Missing-modality baseline by dropping one or two modalities.
4. Improved model with a new module or loss term for robustness.

Training data: user A and user B.  
Testing data: user C.

## Important Repository Structure

```text
src/models/baseline_real_scene.py
src/modules/real_scene_utils.py
src/modules/feature_extraction/
experiments/train_and_test.py
results/
figures/
report/
docs/
```

Large datasets and trained model files may not be uploaded to GitHub. Do not assume that all raw data exists in the remote repository.

## Development Rules

1. Do not rewrite the whole project at once.
2. Modify only files related to the current task.
3. Keep the original teacher-provided logic understandable.
4. Prefer project-relative paths instead of hard-coded absolute paths.
5. Do not delete baseline code unless a replacement has already been tested.
6. Keep generated large files out of Git unless explicitly requested.
7. Save metrics under `results/metrics/`.
8. Save logs under `results/logs/`.
9. Save predictions under `results/predictions/`.
10. Save figures under `figures/`.
11. Save report screenshots under `report/screenshots/`.
12. After every meaningful code change, update collaboration logs.

## Required Logs After Each Task

After each task, update at least one of the following files:

```text
docs/collaboration_log.md
docs/experiment_log.md
docs/method_notes.md
```

If the files do not exist, create them.

### collaboration_log.md Format

```markdown
## YYYY-MM-DD - Task Title

### Contributor
- Name: xxx
- Role: Code / Experiment / Report

### Files Changed
- `path/to/file.py`: brief description

### Purpose
Briefly explain why this change was made.

### How to Run
```bash
python xxx.py --config xxx.yaml
```

### Output Files
- `results/metrics/xxx.csv`
- `results/logs/xxx.log`
- `figures/xxx.png`

### Current Status
- Completed / Partially completed / Failed

### Notes for Report Writer
Briefly explain what this result means and where the report writer can find evidence.

### Remaining Problems
- Problem 1
- Problem 2
```

### experiment_log.md Format

```markdown
## Experiment Name

### Date
YYYY-MM-DD

### Purpose
Explain the goal of this experiment.

### Command
```bash
python experiments/xxx.py --config configs/xxx.yaml
```

### Dataset Split
- Train: user A + user B
- Test: user C

### Settings
- Modalities: imu, gesture, audio, text, scene
- Noise setting: none / 20% / 40% / 60%
- Missing modality: none / xxx
- Model: baseline / improved

### Main Results
| Metric | Value |
|---|---:|
| Accuracy | xxx |
| Macro F1 | xxx |
| Weighted F1 | xxx |
| Training Time | xxx |
| Avg Test Time / Sample | xxx |

### Output Paths
- Metrics: `results/metrics/xxx.csv`
- Log: `results/logs/xxx.log`
- Figure: `figures/xxx.png`
- Checkpoint: `results/checkpoints/xxx.pt`

### Brief Analysis
Write 2-4 short sentences. Do not invent results.

### Problems
Record errors, unstable results, or missing outputs.
```

### method_notes.md Format

```markdown
## Method Note: Method Name

### Motivation
Explain why this method is added.

### Implementation
Explain which module or loss term is added.

### Related Files
- `src/models/xxx.py`
- `experiments/xxx.py`

### Expected Effect
Explain whether this method is mainly for modal noise, missing modality, or both.

### Current Result
Summarize actual results after experiments are available.

### Report Description Draft
Write a short paragraph that the report writer can reuse and polish.
```

## Coding Style

- Use clear function names.
- Add only necessary comments.
- Avoid long scripts with unrelated logic mixed together.
- Keep training, testing, data processing, and evaluation separated when possible.
- Use deterministic seeds when possible.
- Print and save key runtime information.

## Preferred Output Format After Codex Finishes

After completing a task, provide:

1. Summary of changes.
2. Files modified.
3. Commands to run.
4. Expected outputs.
5. Log files updated.
6. Known issues.
7. Suggested next task.
```

---

## 4. 每次使用 Codex 的通用提示词模板

可以将下面这段作为每次新任务的开头：

```text
You are working on a Machine Learning course project repository: multimodal-intention-recognition.

Before modifying code, please read README.md, docs/original_code_readme.md, and the relevant source files.

Important requirements:
- Train data: user A and user B.
- Test data: user C.
- Refactor the workflow into an end-to-end pipeline from raw data to intention label output.
- Keep large datasets and model checkpoints out of Git unless explicitly requested.
- Save metrics to results/metrics/.
- Save logs to results/logs/.
- Save predictions to results/predictions/.
- Save figures to figures/.
- Save screenshots for the report to report/screenshots/.
- After code changes, update docs/collaboration_log.md and, if an experiment is run, docs/experiment_log.md.

Please only modify files related to the current task. Do not rewrite unrelated code.

Current task:
[在这里写清楚本次要 Codex 完成的具体任务]

After finishing, please output:
1. Modified files.
2. Main changes.
3. How to run.
4. Expected outputs.
5. Updated log entries.
6. Remaining issues.
```

---

## 5. 具体任务提示词示例

### 5.1 端到端重构任务

```text
Current task:
Refactor the current training/testing workflow. Use experiments/train_and_test.py as the reference implementation. Create clearer entry scripts experiments/train.py and experiments/test.py. Integrate feature extraction calls so that the workflow starts from raw data paths instead of manually loaded saved feature files.

Requirements:
- Keep the original train_and_test.py as a reference and do not delete it.
- Use project-relative paths.
- Create reusable helper functions if necessary.
- Save trained model, scalers, and label encoder under results/checkpoints/.
- Save logs under results/logs/.
- Update docs/collaboration_log.md with the files changed, run command, output paths, and remaining problems.
```

---

### 5.2 模态噪声实验任务

```text
Current task:
Add modal noise baseline experiments.

Requirements:
- Add noise to each single modality separately.
- Noise ratios: 20%, 40%, 60%.
- Modalities: imu, gesture, audio, text, scene.
- Train and test on the same user split: train on user A+B, test on user C.
- Save metrics to results/metrics/noise_baseline_metrics.csv.
- Save logs to results/logs/noise_baseline.log.
- Save comparison figures to figures/result_charts/.
- Update docs/experiment_log.md with experiment settings, command, output paths, and result summary.
```

---

### 5.3 模态缺失实验任务

```text
Current task:
Add missing-modality baseline experiments.

Requirements:
- Drop each single modality.
- Drop each pair of modalities.
- Modalities: imu, gesture, audio, text, scene.
- Train on user A+B and test on user C.
- Save metrics to results/metrics/missing_modality_metrics.csv.
- Save logs to results/logs/missing_modality.log.
- Save result charts to figures/result_charts/.
- Update docs/experiment_log.md.
```

---

### 5.4 改进模型任务

```text
Current task:
Implement an improved model for robustness under modal noise and missing modality.

Preferred method:
Add modality dropout and modality reliability gate to the multimodal fusion model.

Requirements:
- Keep the baseline model available for comparison.
- Add the improved model in a separate file if possible, such as src/models/improved_model.py.
- Add an experiment entry script experiments/run_improved_model.py.
- Save improved model metrics to results/metrics/improved_model_metrics.csv.
- Compare the improved model with clean baseline, noise baseline, and missing-modality baseline.
- Update docs/method_notes.md with motivation, implementation, related files, and current result.
- Update docs/experiment_log.md after running experiments.
```

---

### 5.5 报告辅助任务

```text
Current task:
Update report materials based on existing experiment logs and result files.

Requirements:
- Read docs/experiment_log.md, docs/collaboration_log.md, docs/method_notes.md, and results/metrics/.
- Do not invent experimental results.
- Create or update docs/report_outline.md.
- Summarize current finished experiments.
- List missing experiments or missing figures.
- Provide short report-ready paragraphs for method and result analysis.
- Keep the language suitable for an undergraduate machine learning course report.
```

---

## 6. 建议创建的日志文件初始模板

### 6.1 `docs/collaboration_log.md`

```markdown
# Collaboration Log

This file records code changes, experiment progress, and notes for report writing.

---

## YYYY-MM-DD - Initial Project Setup

### Contributor
- Name: xxx
- Role: Project setup

### Files Changed
- `README.md`: updated project overview and repository structure.

### Purpose
Prepare the repository for team collaboration.

### How to Run
Not applicable.

### Output Files
Not applicable.

### Current Status
Completed.

### Notes for Report Writer
The repository structure has been prepared. Large raw data and model files are stored locally and are not uploaded to GitHub.

### Remaining Problems
- Need to run the teacher-provided baseline code.
- Need to refactor the code into train.py and test.py.
```

---

### 6.2 `docs/experiment_log.md`

```markdown
# Experiment Log

This file records all experiments for the final project report.

---

## Experiment 0: Environment Check

### Date
YYYY-MM-DD

### Purpose
Check whether the environment, dependencies, data paths, and GPU are available.

### Command
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Dataset Split
- Train: user A + user B
- Test: user C

### Settings
- Model: not applicable
- Noise: none
- Missing modality: none

### Main Results
| Metric | Value |
|---|---:|
| CUDA Available | xxx |
| GPU Name | xxx |

### Output Paths
- Log: `results/logs/environment_check.log`

### Brief Analysis
Environment check result needs to be filled after running.

### Problems
None recorded yet.
```

---

### 6.3 `docs/method_notes.md`

```markdown
# Method Notes

This file records model improvement ideas and implementation details.

---

## Method Note: Modality Dropout and Reliability Gate

### Motivation
The course project requires the model to handle modal noise and missing modality. Modality dropout can improve missing-modality robustness, while reliability gates can reduce the influence of noisy modalities.

### Implementation
To be filled after implementation.

### Related Files
- `src/models/improved_model.py`
- `experiments/run_improved_model.py`

### Expected Effect
This method is expected to improve accuracy under both modal noise and missing-modality conditions.

### Current Result
To be filled after experiments.

### Report Description Draft
The improved model introduces modality dropout during training and learns modality reliability weights during fusion. This design encourages the model to rely less on unstable modalities and improves robustness when some modalities are noisy or unavailable.
```

---

## 7. 使用 Codex 时的注意事项

1. 不要让 Codex 一次性完成整个项目。
2. 每次只给一个明确任务，例如“先跑通 baseline”或“只添加噪声实验”。
3. 每次修改后先查看 diff，再运行代码。
4. 如果 Codex 修改了很多无关文件，需要回退或要求它缩小修改范围。
5. 大文件不要提交到 GitHub。
6. 实验结果必须来自真实运行，不要让 Codex 编造指标。
7. 文稿负责人只使用日志和真实结果写报告。
8. 每次实验完成后立刻更新日志，否则后期报告会很难整理。

---

## 8. 推荐团队工作节奏

```text
第 1 步：成员 A 用 Codex 跑通原始代码并记录问题
第 2 步：成员 A 用 Codex 拆分 train.py / test.py
第 3 步：成员 B 用 Codex 添加 clean baseline 运行入口
第 4 步：成员 B 用 Codex 添加 noise baseline
第 5 步：成员 B 用 Codex 添加 missing modality baseline
第 6 步：成员 B 用 Codex 添加 improved model
第 7 步：成员 C 根据日志整理报告和 PPT
第 8 步：全组检查最终代码、结果和答辩材料
```

---

## 9. 最推荐的 Codex 使用方式

每次向 Codex 发任务时，按照这个结构写：

```text
背景：这是机器学习课程项目，目标是多模态用户交互意图识别。
当前状态：说明已经完成了什么。
本次任务：只写一个明确任务。
限制条件：不要改哪些文件、不要做哪些事。
输出要求：要求给出修改文件、运行命令、输出路径，并更新日志。
```

示例：

```text
背景：这是机器学习课程项目，目标是完成多模态用户交互意图识别。仓库中已有老师提供的 baseline 代码和特征提取脚本。

当前状态：README 和目录结构已经整理好，但训练和测试流程还没有完全端到端化。

本次任务：请你只完成 train.py 和 test.py 的初步拆分，让它们能够调用原始代码中的训练和测试逻辑。

限制条件：
1. 不要删除 experiments/train_and_test.py。
2. 不要修改数据集目录结构。
3. 不要提交大文件。
4. 不要实现噪声实验和缺失实验，这些后续再做。

输出要求：
1. 列出修改的文件。
2. 给出运行命令。
3. 说明输出保存在哪里。
4. 更新 docs/collaboration_log.md。
5. 如果代码暂时不能完全跑通，请记录具体原因。
```
