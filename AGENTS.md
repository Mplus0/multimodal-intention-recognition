# Codex Project Instructions

## 1. Project Overview

This repository is for a Machine Learning course project on multimodal user intention recognition in AR interaction scenarios.

The end-to-end workflow is:

```text
raw multimodal data
-> preprocessing / feature extraction
-> multimodal fusion model
-> intention label output
```

The group course project evaluates:

1. Clean baseline.
2. Single-modality noise at 20%, 40%, and 60%.
3. Missing one or two modalities.
4. A robustness-oriented improved model using reliability-gated fusion and modality dropout.

Dataset split:

- Train: user A + user B.
- Test: user C.

Large datasets, cached features, pretrained local models, checkpoints, and other large generated files may be absent from GitHub. Do not assume that all runtime data exists in the remote repository.

## 2. Current Individual Term Paper Task

The current task is an individual term-paper extension of the completed group project.

Selected route:

1. Improve the group project method with a new design.
2. Conduct ablation experiments.
3. Compare the proposed method with:
   - the original baseline;
   - the official group improved model;
   - relevant individual ablations.

Proposed method:

**Repetition-Aware Text Compression with Modality Dropout**

The method keeps cached Text features unchanged, but averages the 10 repeated sentence embeddings into one Text token inside the model before projection, positional encoding, and multimodal fusion. It is combined with whole-modality dropout to reduce duplicated Text evidence and encourage non-Text compensation.

This personal extension must not overwrite, silently alter, or invalidate the official group baseline, group configuration, or group results.

## 3. Fixed Research Scope

The current Text pipeline produces a sentence embedding and repeats it across 10 time steps. Treat this as an implementation fact and part of the method motivation.

For this term paper:

1. Keep the five-modality input protocol unchanged.
2. Keep the Text feature shape unchanged.
3. Do not modify Text feature extraction merely to implement the new method.
4. Keep the cached Text feature at 10 tokens; the personal model is explicitly authorized to compress it to one token inside the model.
5. Drop the whole Text modality, not individual Text time steps.
6. Keep the existing missing-modality representation:
   - zero-fill the dropped modality;
   - set its `modality_mask` to unavailable.
7. Focus changes on model-side Text compression, training augmentation, configuration, experiment metadata, and evaluation.
8. Do not modify dataset splits, labels, modality names, or cached feature formats.

## 4. Important Files

```text
AGENTS.md
requirements.txt

configs/
  default.yaml
  improved_model.yaml
  term_paper_text_aware.yaml

src/data/
  features.py
  transforms.py
  improved_transforms.py
  raw_extractors/text_extractor.py

src/models/
  formal_baseline.py
  improved_model.py

src/training/
  engine.py
  evaluate.py
  experiment_runner.py
  improved_experiment_runner.py

experiments/

results/
  metrics/
  logs/
  predictions/
  checkpoints/
  summaries/
  configs/

figures/

report/
  screenshots/

docs/
  collaboration_log.md
  experiment_log.md
  method_notes.md
```

File responsibilities:

- `src/data/raw_extractors/text_extractor.py`
  - Evidence for the repeated Text-token design.
  - Do not modify for this method unless explicitly requested.

- `src/data/features.py`
  - Maintains the unified feature format.
  - Do not change the five-modality protocol.

- `src/data/improved_transforms.py`
  - Main location for uniform and Text-aware modality dropout.
  - Preserve the original uniform behavior.

- `src/models/formal_baseline.py`
  - Official baseline comparison model.
  - Treat as frozen unless fixing a confirmed bug.

- `src/models/improved_model.py`
  - Official reliability-gated group model.
  - Preserve group behavior.

- `src/training/improved_experiment_runner.py`
  - Reuse the existing workflow.
  - Extend only for strategy selection, timing, metadata, or term-paper ablations.

- `configs/improved_model.yaml`
  - Official group improved-model configuration.
  - Do not silently convert it to the personal method.

- `configs/term_paper_text_aware.yaml`
  - Dedicated personal-method configuration.
  - Create it if missing.

## 5. General Development Rules

1. Do not rewrite the whole project at once.
2. Modify only files related to the current task.
3. Keep teacher-provided and group-project logic understandable.
4. Prefer project-relative paths over hard-coded absolute paths.
5. Use dependencies from `requirements.txt`.
6. If the local environment lacks a listed dependency, do not add compatibility fallbacks or install it unless explicitly requested.
7. The official project runs in the designated server environment.
8. Do not add compatibility logic for abandoned methods.
9. Do not delete baseline or group-method code without explicit approval and testing.
10. Keep generated large files out of Git unless explicitly requested.
11. Use deterministic seeds whenever possible.
12. Print and save key runtime information.
13. Keep data processing, training, validation, testing, evaluation, and plotting separated when practical.
14. Add only necessary comments.
15. Avoid unrelated refactoring.
16. After meaningful implementation work, update the appropriate documentation.
17. Do not write experimental results before a real run completes.

## 6. Group-Method Preservation

Treat these as fixed comparison methods:

1. Original baseline.
2. Reliability gate only.
3. Uniform modality dropout only.
4. Reliability gate + uniform modality dropout.

Rules:

1. Do not change their intended behavior without an explicit request.
2. Do not overwrite their configurations with personal settings.
3. Do not overwrite their outputs with term-paper outputs.
4. Preserve configuration-level strategy selection:
   - `strategy: uniform` reproduces the group method;
   - `strategy: text_aware` activates the personal method.
5. Shared code may be extended, but the uniform strategy must remain reproducible.
6. Apply confirmed fairness-related bug fixes consistently across comparison methods.
7. Record result-changing bug fixes in `docs/collaboration_log.md` and `docs/experiment_log.md`.

## 7. Text-Aware Modality Dropout

This section documents the earlier attempted direction. It is retained for reproducibility, but it is no longer the selected personal-paper method.

Preferred implementation:

```text
src/data/improved_transforms.py
```

Preferred configuration:

```text
configs/term_paper_text_aware.yaml
```

The configuration should support:

```yaml
training:
  modality_dropout:
    enabled: true
    strategy: text_aware
    drop_prob: 0.4
    text_drop_prob: 0.6
    max_drop_modalities: 2
    min_keep_modalities: 1
    modalities:
      - imu
      - gesture
      - audio
      - text
      - scene
    modality_weights:
      imu: 1.0
      gesture: 1.0
      audio: 1.0
      text: 2.0
      scene: 1.0
```

These are suggested starting values, not official results.

### `uniform`

- Reproduce the existing random modality dropout.
- Select eligible modalities uniformly.
- Preserve deterministic seed behavior.

### `text_aware`

- Apply dropout only when the overall dropout event is triggered.
- Give Text a higher dropout priority or probability.
- Drop the entire Text modality.
- Optionally drop additional non-Text modalities within configured limits.
- Never exceed `max_drop_modalities`.
- Always preserve at least `min_keep_modalities`.
- Use only configured modalities.
- Validate all probabilities and weights.
- Use zero-fill and mask updates consistent with missing-modality transforms.
- Record actually dropped modalities when supported by the data contract.
- Remain deterministic for the same seed, sample identity, split, and configuration.

Do not:

- drop only selected repeated Text time steps;
- change Text feature extraction as a shortcut;
- change Text tensor shape;
- remove the uniform strategy;
- silently use Text-aware dropout when no strategy is specified;
- silently fall back from invalid Text-aware configuration;
- use test labels or test performance to make training-time per-sample decisions.

## 8. Configuration Rules

Use separate files:

```text
configs/improved_model.yaml
configs/term_paper_text_aware.yaml
```

Rules:

1. `configs/improved_model.yaml` represents the official group method.
2. `configs/term_paper_text_aware.yaml` represents the personal method.
3. Do not change group defaults merely to run personal experiments.
4. Every official experiment must save the effective merged configuration.
5. Save all settings affecting:
   - model structure;
   - dropout strategy;
   - dropout strength;
   - missing/noise condition;
   - seed;
   - epochs;
   - batch size;
   - learning rate;
   - data split;
   - output naming.
6. Fail clearly on invalid strategy names or values.
7. Do not silently fall back to another strategy.

## 9. Required Comparison Groups

### `baseline`

- Original Transformer baseline.
- No reliability gate.
- No training-time modality dropout.

### `group_uniform`

- Reliability gate enabled.
- Uniform modality dropout enabled.
- Official group improved model.

### `text_aware_main`

- Reliability gate enabled.
- Text-aware modality dropout enabled.
- Proposed personal method.

### `text_aware_dropout_only`

- Reliability gate disabled.
- Text-aware modality dropout enabled.
- Measures the independent contribution of the new dropout strategy.

Additional ablations require a clear research question.

## 10. Ablation Rules

This paper follows the improved-design route, not ordinary hyperparameter optimization.

The central method change is:

```text
uniform modality dropout
-> Text-aware modality dropout
```

Recommended variants:

```text
uniform
text_aware_mild
text_aware_medium
text_aware_strong
text_aware_dropout_only
```

A simple search for the best `text_drop_prob` must not be presented as the sole contribution.

Each variant must:

1. Have a clear strategy difference.
2. Use a dedicated name.
3. Save its effective configuration.
4. Record strategy parameters in metrics.
5. Use the same split and evaluation protocol.
6. Use comparable runtime conditions for timing comparisons.
7. Be based on a completed real experiment.

## 11. Required Evaluation Conditions

Prioritize:

```text
clean
missing_text
missing_imu_text
missing_gesture_text
missing_audio_text
missing_text_scene
```

Do not automatically rerun all group noise and missing-modality matrices unless the paper needs them.

## 12. Metrics and Timing

Every official experiment must report:

- Loss.
- Accuracy.
- Macro-F1.
- Weighted-F1.
- Total training time.
- Average training time per processed training sample.
- Total testing time.
- Average testing time per test sample.

Use:

```text
loss
accuracy
macro_f1
weighted_f1
training_time_total_sec
avg_training_time_per_sample_sec
testing_time_total_sec
avg_testing_time_per_sample_sec
```

Unless sampling is nonstandard:

```text
processed training sample visits
= number of training samples × completed epochs
```

```text
avg_training_time_per_sample_sec
= training_time_total_sec / processed training sample visits
```

Timing rules:

1. Use only completed real-data runs for official timing.
2. Synchronize CUDA before and after timed GPU sections when accurate GPU timing is required.
3. Record device, batch size, epochs, train count, and test count.
4. Do not directly compare incompatible devices or settings.
5. Do not use smoke-test timing as official evidence.

## 13. Required Metrics Metadata

Official metrics rows should include:

```text
experiment_name
method_name
model_type
dropout_strategy
text_aware_variant
drop_prob
text_drop_prob
max_drop_modalities
min_keep_modalities
modality_weights
use_reliability_gate
use_modality_dropout
missing_modalities
target_modality
noise_ratio
seed
epochs
batch_size
learning_rate
device
sample_count_train
processed_train_sample_visits
sample_count_test
loss
accuracy
macro_f1
weighted_f1
training_time_total_sec
avg_training_time_per_sample_sec
testing_time_total_sec
avg_testing_time_per_sample_sec
status
config_snapshot
git_commit
notes
```

Use empty values or `none` where fields do not apply.

## 14. Output Rules

Save under:

```text
results/metrics/
results/logs/
results/predictions/
results/checkpoints/
results/summaries/
results/configs/
figures/
report/screenshots/
```

Examples:

```text
results/metrics/term_paper_baseline_clean.csv
results/metrics/term_paper_group_uniform_missing_text.csv
results/metrics/term_paper_text_aware_medium_missing_text.csv
results/logs/term_paper_text_aware_medium_missing_text.log
results/configs/term_paper_text_aware_medium_missing_text.yaml
figures/term_paper_text_aware_medium_loss_curve.png
figures/term_paper_text_aware_medium_missing_text_confusion_matrix.png
```

Never overwrite group outputs with term-paper outputs.

## 15. Required Visualizations

Where applicable, include:

1. Confusion matrix for the proposed main method.
2. Training and validation loss curves.
3. Baseline vs group uniform vs proposed method comparison.
4. Missing-Text and Text-related missing-modality comparison.
5. Text-aware ablation comparison.

Rules:

- Paper-ready titles, axes, legends, and table names must be in English.
- Build plots from actual saved metrics.
- Do not plot invented values.
- Save figures under `figures/`.
- Record source metrics files in the experiment log.

## 16. Official Result Integrity

1. Never invent, estimate, interpolate, or manually complete experimental metrics.
2. AI may help with code, organization, and analysis, but must not generate experiment results.
3. Results are official only after a completed real-data run.
4. Never use these as official results:
   - smoke tests;
   - synthetic data;
   - `max_samples` runs;
   - shortened debug runs;
   - incomplete runs;
   - manually edited metrics;
   - expected, predicted, or placeholder values.
5. Every official result must be traceable to:
   - execution command;
   - Git commit;
   - effective configuration;
   - seed;
   - dataset split;
   - device;
   - metrics;
   - log;
   - checkpoint when applicable.
6. Record failed or incomplete experiments honestly and leave missing values empty.
7. Do not use the test set for checkpoint selection.
8. Prefer a predefined validation metric such as validation Macro-F1.
9. Keep evaluation protocols consistent across methods.

## 17. Smoke Tests

Smoke tests may verify:

- configuration parsing;
- transform construction;
- deterministic selection;
- tensor shapes;
- modality masks;
- zero-filled features;
- model forward pass;
- output paths;
- metrics schema;
- CLI wiring.

Smoke tests must:

1. Be labeled `smoke_test`.
2. Use separate output names.
3. Never overwrite official results.
4. Never appear in official paper tables.

Text-aware tests should cover:

- `uniform`;
- `text_aware`;
- Text selected and not selected;
- one and two dropped modalities;
- `min_keep_modalities`;
- deterministic repetition;
- seed/sample variation;
- invalid probabilities/weights;
- missing feature keys;
- mask updates;
- zero-filled tensors.

## 18. Documentation Rules

After meaningful work, update at least one appropriate file:

```text
docs/collaboration_log.md
docs/experiment_log.md
docs/method_notes.md
```

Use:

- `collaboration_log.md`
  - code/config changes, bug fixes, refactoring, scripts.

- `experiment_log.md`
  - only actual completed, failed, or partial runs.
  - never fictional results.

- `method_notes.md`
  - motivation, method design, implementation decisions, limitations, paper-ready descriptions.

A code-only task does not require a fake experiment entry.

Language:

- Internal logs may be in Chinese.
- Commands, paths, keys, metric names, classes, functions, and model names remain unchanged.
- Paper-ready figure/table names and method descriptions must be in English.
- Do not add unverified claims to the paper draft.

## 19. Log Templates

### `collaboration_log.md`

```markdown
## YYYY-MM-DD - Task Title

### Contributor
- Name: xxx
- Role: Code / Experiment / Report

### Files Changed
- `path/to/file.py`: description

### Purpose
Why the change was made.

### Implementation Summary
Main implementation decisions.

### How to Validate
```bash
python xxx.py --config xxx.yaml
```

### Expected Validation Outputs
Expected files or checks, without invented metric values.

### Current Status
- Completed / Partially completed / Failed

### Notes for Report Writer
Where to find evidence.

### Remaining Problems
- Problem 1
- Problem 2
```

### `experiment_log.md`

```markdown
## Experiment Name

### Date
YYYY-MM-DD

### Purpose
Experiment goal.

### Method
- Method name: baseline / group_uniform / text_aware_main / text_aware_dropout_only
- Dropout strategy: none / uniform / text_aware
- Text-aware variant: none / mild / medium / strong

### Command
```bash
python experiments/xxx.py --config configs/xxx.yaml
```

### Reproducibility
- Git commit: xxx
- Config snapshot: `results/configs/xxx.yaml`
- Seed: xxx
- Device: xxx

### Dataset Split
- Train: user A + user B
- Test: user C
- Train samples: xxx
- Test samples: xxx

### Settings
- Modalities: imu, gesture, audio, text, scene
- Epochs: xxx
- Batch size: xxx
- Learning rate: xxx
- Noise setting: none / 20% / 40% / 60%
- Missing modality: none / xxx
- `drop_prob`: xxx
- `text_drop_prob`: xxx
- `max_drop_modalities`: xxx
- `min_keep_modalities`: xxx

### Main Results
| Metric | Value |
|---|---:|
| Loss | xxx |
| Accuracy | xxx |
| Macro F1 | xxx |
| Weighted F1 | xxx |
| Total Training Time (s) | xxx |
| Avg Training Time / Sample (s) | xxx |
| Total Testing Time (s) | xxx |
| Avg Testing Time / Sample (s) | xxx |

### Output Paths
- Metrics: `results/metrics/xxx.csv`
- Log: `results/logs/xxx.log`
- Predictions: `results/predictions/xxx.csv`
- Summary: `results/summaries/xxx.json`
- Config snapshot: `results/configs/xxx.yaml`
- Loss curve: `figures/xxx_loss_curve.png`
- Confusion matrix: `figures/xxx_confusion_matrix.png`
- Checkpoint: `results/checkpoints/xxx.pt`

### Brief Analysis
2-4 short sentences based only on actual results.

### Problems
Errors, instability, incomplete outputs, or comparability limits.

### Official Result Status
- Official / Debug only / Smoke test / Failed
```

### `method_notes.md`

```markdown
## Method Note: Method Name

### Motivation
Observed problem and motivation.

### Group Method
Original reliability-gated fusion and uniform modality dropout.

### Proposed Improvement
Text-aware modality dropout strategy.

### Implementation
Transform logic, configuration, and mask behavior.

### Related Files
- `src/data/improved_transforms.py`
- `src/training/improved_experiment_runner.py`
- `configs/term_paper_text_aware.yaml`

### Comparison Methods
- Baseline
- Group uniform method
- Proposed Text-aware method
- Relevant ablations

### Expected Evaluation Focus
Missing-Text and Text-related missing-modality robustness.

### Current Results
Only actual completed experiments.

### Limitations
Evidence-supported trade-offs and boundaries.

### English Paper Description Draft
Reusable English paragraph without unverified numerical claims.
```

## 20. Paper Requirements

The paper must:

1. Use the required Overleaf template.
2. Be written in English.
3. Be 4 or 5 pages and not exceed 5 pages.
4. Include:
   - Abstract;
   - Introduction;
   - Related Work;
   - Methods;
   - Results;
   - Analysis;
   - Conclusion;
   - References.
5. Include at least:
   - quantitative results;
   - visualization results;
   - average training time per sample;
   - average testing time per sample.

Do not claim that Text-aware dropout improves robustness until completed experiments support that statement.

## 21. Preferred Codex Workflow

For implementation tasks:

1. Inspect relevant files.
2. Identify the smallest required change.
3. State which group behavior must remain unchanged.
4. Implement the change.
5. Add focused tests or smoke checks.
6. Run only validation supported by the current environment.
7. Do not run large official experiments unless explicitly requested.
8. Update appropriate documentation.
9. Summarize changes, validation, and remaining issues.

For official experiments:

1. Confirm method and condition from the configuration.
2. Confirm the run is not a smoke or shortened debug run.
3. Save the effective configuration.
4. Record Git commit and runtime environment.
5. Run the experiment.
6. Save required metrics and timing.
7. Save predictions and visualizations.
8. Update `docs/experiment_log.md`.
9. Never replace failed outputs with estimates.

## 22. Preferred Output After Codex Finishes

Provide:

1. Summary of changes.
2. Files modified.
3. Group-method behavior preserved.
4. Validation commands.
5. Validation results.
6. Expected or generated outputs.
7. Documentation updated.
8. Known issues.
9. Suggested next task.

Do not report an official performance conclusion without a completed real-data experiment.
