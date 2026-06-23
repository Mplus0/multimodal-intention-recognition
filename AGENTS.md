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
5. Write code strictly against the dependencies listed in `requirements.txt`.
6. If the current local environment is missing a dependency from `requirements.txt`, do not add compatibility fallbacks for that local environment and do not install the dependency. The official project runs on the server environment where these dependencies are already available.
7. Do not delete baseline code unless a replacement has already been tested.
8. Keep generated large files out of Git unless explicitly requested.
9. Save metrics under `results/metrics/`.
10. Save logs under `results/logs/`.
11. Save predictions under `results/predictions/`.
12. Save figures under `figures/`.
13. Save report screenshots under `report/screenshots/`.
14. After every meaningful code change, update collaboration logs.

## Required Logs After Each Task

After each task, update at least one of the following files:

```text
docs/collaboration_log.md
docs/experiment_log.md
docs/method_notes.md
```

If the files do not exist, create them.

### Markdown Log Language

Codex must write Markdown task logs in Chinese by default, including:

- `docs/collaboration_log.md`
- `docs/experiment_log.md`
- `docs/method_notes.md`

Keep code snippets, commands, file paths, metric names, and model names unchanged when English is required for accuracy.

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
