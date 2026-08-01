# Strawberry Model Comparison - batch128_lr7e4_nopin

Generated: 2026-07-31 04:28

Configuration values are read from each model's metrics.json when available.

## Configuration

| Model | Backbone | Temporal | Fusion | Seq Len | Epochs | Batch | LR | Weight Decay | Dropout |
|-------|----------|----------|--------|---------|--------|-------|----|--------------|---------|
| **A** | EfficientNet-B0 | GRU | late_env_branch | 10 | 10 | 128 | 0.0007 | 0.0003 | 0.4 |
| **B** | MobileNetV2 | LSTM | late_env_branch | 10 | 10 | 128 | 0.0007 | 0.0003 | 0.4 |
| **C** | EfficientNet-B0 | LSTM | late_env_branch | 10 | 10 | 128 | 0.0007 | 0.0003 | 0.4 |
| **D** | MobileNetV2 | GRU | late_env_branch | 10 | 10 | 128 | 0.0007 | 0.0003 | 0.4 |

## Test Performance

| Model | MAE | RMSE | MAPE | R2 | Best Epoch | Best Val MAE | Total Params | Trainable Params |
|-------|-----|------|------|----|------------|--------------|--------------|----------------|
| **A** | 38.11 | 46.60 | 119.1 | 0.2994 | 8 | 18.36 | 4,781,919 | 774,371 |
  - Model A: val MAE 56.70 -> 18.36 -> 23.74
| **B** | 31.73 | 38.94 | 94.3 | 0.5109 | 10 | 18.31 | 3,178,723 | 954,851 |
  - Model B: val MAE 57.62 -> 18.31 -> 18.31
| **C** | 45.85 | 56.58 | 148.4 | -0.0328 | 7 | 18.85 | 4,962,399 | 954,851 |
  - Model C: val MAE 56.25 -> 18.85 -> 21.74
| **D** | 31.96 | 39.18 | 97.2 | 0.5047 | 9 | 18.90 | 2,998,243 | 774,371 |
  - Model D: val MAE 57.64 -> 18.90 -> 19.24

## Best Per Metric

- Best MAE: Model B (31.7296)
- Best RMSE: Model B (38.9384)
- Best MAPE: Model B (94.3454)
- Best R2: Model B (0.5109)

## Files

| Model | Checkpoint | History | Predictions | Metrics |
|-------|------------|---------|-------------|---------|
| **A** | `models\model_A_batch128_lr7e4_nopin\best_model.pth` | `data\model_A_batch128_lr7e4_nopin_outputs\training_history.csv` | `data\model_A_batch128_lr7e4_nopin_outputs\test_predictions.csv` | `data\model_A_batch128_lr7e4_nopin_outputs\metrics.json` |
| **B** | `models\model_B_batch128_lr7e4_nopin\best_model.pth` | `data\model_B_batch128_lr7e4_nopin_outputs\training_history.csv` | `data\model_B_batch128_lr7e4_nopin_outputs\test_predictions.csv` | `data\model_B_batch128_lr7e4_nopin_outputs\metrics.json` |
| **C** | `models\model_C_batch128_lr7e4_nopin\best_model.pth` | `data\model_C_batch128_lr7e4_nopin_outputs\training_history.csv` | `data\model_C_batch128_lr7e4_nopin_outputs\test_predictions.csv` | `data\model_C_batch128_lr7e4_nopin_outputs\metrics.json` |
| **D** | `models\model_D_batch128_lr7e4_nopin\best_model.pth` | `data\model_D_batch128_lr7e4_nopin_outputs\training_history.csv` | `data\model_D_batch128_lr7e4_nopin_outputs\test_predictions.csv` | `data\model_D_batch128_lr7e4_nopin_outputs\metrics.json` |

Best overall by MAE: Model B (31.73 hours).