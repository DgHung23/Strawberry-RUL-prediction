# All Models — Comparison Summary

*Comparing 4 trained model(s)*

## Leaderboard (Lower Error = Better)

| Rank | Model | Architecture | MAE (h) | RMSE (h) | R² | Params |
|------|-------|-------------|---------|----------|-----|--------|
| 1st | Model C | EfficientNet-B0 + CBAM + LSTM | 32.0 | 39.4 | 0.755 | 4.94M |
| 2nd | Model A | EfficientNet-B0 + CBAM + GRU | 43.1 | 49.9 | 0.608 | 4.76M |
| 3rd | Model B | MobileNetV2 + CBAM + LSTM | 53.8 | 61.5 | 0.403 | 3.16M |
|   4 | Model D | MobileNetV2 + CBAM + GRU | 54.4 | 65.6 | 0.320 | 2.98M |

## Best By Metric

- **Lowest Average Error (MAE):** Model C (EfficientNet-B0 + CBAM + LSTM) — 32.02
- **Most Consistent Predictions (lowest RMSE):** Model C (EfficientNet-B0 + CBAM + LSTM) — 39.39
- **Best Overall Fit (highest R²):** Model C (EfficientNet-B0 + CBAM + LSTM) — 0.76

## Per-Model Detailed Reports

- [Model A](model_A_detailed_report.md) — EfficientNet-B0 + CBAM + GRU
- [Model B](model_B_detailed_report.md) — MobileNetV2 + CBAM + LSTM
- [Model C](model_C_detailed_report.md) — EfficientNet-B0 + CBAM + LSTM
- [Model D](model_D_detailed_report.md) — MobileNetV2 + CBAM + GRU
