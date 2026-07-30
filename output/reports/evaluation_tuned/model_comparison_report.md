# Tuned Strawberry Model Comparison

Generated: 2026-07-30 07:09

All four tuned runs used seq_len=10, late_env_branch fusion, last_mean_max temporal pooling, SmoothL1 loss, and freeze_backbone=True.

## Configuration

| Model | Backbone | Temporal | Fusion | Seq Len | Epochs |
|-------|----------|----------|--------|---------|--------|
| **A** | EfficientNet-B0 | GRU | late_env_branch | 10 | 10 |

## Test Performance

| Model | MAE | RMSE | MAPE | R2 | Best Epoch | Best Val MAE | Total Params | Trainable Params |
|-------|-----|------|------|----|------------|--------------|--------------|----------------|
| **A** | 27.26 | 34.17 | 62.7 | 0.8111 | 4 | 4.35 | 4,781,919 | 774,371 |
  - Model A: val MAE 15.20 -> 4.35 -> 8.56

## Best Per Metric

- Best MAE: Model A (27.2562)
- Best RMSE: Model A (34.1737)
- Best MAPE: Model A (62.6603)
- Best R2: Model A (0.8111)

## Files

| Model | Checkpoint | History | Predictions | Metrics |
|-------|------------|---------|-------------|---------|
| **A** | `models\model_A_tuned\best_model.pth` | `data\model_A_tuned_outputs\training_history.csv` | `data\model_A_tuned_outputs\test_predictions.csv` | `data\model_A_tuned_outputs\metrics.json` |

Best overall by MAE: Model A (27.26 hours).