# Strawberry Model Comparison - batch128_lr7e4_nopin

Generated: 2026-07-30 10:45

Configuration values are read from each model's metrics.json when available.

## Configuration

| Model | Backbone | Temporal | Fusion | Seq Len | Epochs | Batch | LR | Weight Decay | Dropout |
|-------|----------|----------|--------|---------|--------|-------|----|--------------|---------|
| **A** | EfficientNet-B0 | GRU | late_env_branch | 10 | 10 | 128 | 0.0007 | 0.0003 | 0.4 |

## Test Performance

| Model | MAE | RMSE | MAPE | R2 | Best Epoch | Best Val MAE | Total Params | Trainable Params |
|-------|-----|------|------|----|------------|--------------|--------------|----------------|
| **A** | 28.15 | 35.08 | 72.0 | 0.8009 | 10 | 3.79 | 4,781,919 | 774,371 |
  - Model A: val MAE 36.33 -> 3.79 -> 3.79

## Best Per Metric

- Best MAE: Model A (28.1506)
- Best RMSE: Model A (35.0830)
- Best MAPE: Model A (71.9542)
- Best R2: Model A (0.8009)

## Files

| Model | Checkpoint | History | Predictions | Metrics |
|-------|------------|---------|-------------|---------|
| **A** | `models\model_A_batch128_lr7e4_nopin\best_model.pth` | `data\model_A_batch128_lr7e4_nopin_outputs\training_history.csv` | `data\model_A_batch128_lr7e4_nopin_outputs\test_predictions.csv` | `data\model_A_batch128_lr7e4_nopin_outputs\metrics.json` |

Best overall by MAE: Model A (28.15 hours).