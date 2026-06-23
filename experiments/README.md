# Experiments Entry Points

Use these standard commands for handoff and report reproduction:

```bash
python experiments/train.py --config configs/default.yaml
python experiments/test.py --config configs/default.yaml --checkpoint results/checkpoints/best.pt
```

## Current Roles

- `train.py`: standard clean-data training entry. It builds the formal Dataset, loads/caches features, trains the baseline, and writes checkpoints, metrics, and logs.
- `test.py`: standard clean-data test entry. It loads a checkpoint and writes test metrics plus predictions.
- `train_formal_dataset.py`: compatibility wrapper for earlier development notes. It now delegates to `train.py`.
- `train_and_test.py`: legacy/original experiment reference. Keep it for comparison with the teacher/teammate baseline, but do not use it as the standard handoff entry.

## Standard Outputs

```text
results/checkpoints/best.pt
results/checkpoints/final.pt
results/checkpoints/scalers.pkl
results/checkpoints/label_encoder.pkl
results/metrics/clean_baseline_metrics.csv
results/metrics/clean_baseline_summary.json
results/metrics/clean_baseline_test_metrics.csv
results/metrics/clean_baseline_test_metrics.json
results/predictions/clean_baseline_predictions.csv
results/logs/train_clean.log
results/logs/test_clean.log
```

`scalers.pkl` is currently a placeholder because the raw-tensor baseline does
not use sklearn scalers. It is kept to match the expected output layout.

