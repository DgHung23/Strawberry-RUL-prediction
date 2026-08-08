# Strawberry RUL model comparison report

- Run root: `C:\Users\admin\Desktop\Strawberry-RUL-prediction\output\runs\strawberry\strawberry_loocv_seq8_image_only`
- Graph directory: `C:\Users\admin\Desktop\Strawberry-RUL-prediction\output\graphs\evaluation\strawberry_loocv_seq8_image_only`
- Report directory: `C:\Users\admin\Desktop\Strawberry-RUL-prediction\output\reports\evaluation\strawberry_loocv_seq8_image_only`
- Completed LOOCV folds: 24 total = A:6, B:6, C:6, D:6
- Training config: seq_len=8, fusion_mode=image_only, temporal_pooling=last_mean_max, env_feature_mode=sensor, epochs=18, patience=5

## Executive summary

Model **D** is best overall by fold-averaged LOOCV MAE: **14.01 +/- 6.88 hours**.
The gap to the next best model (B) is **1.05 hours**.
Best RMSE model: **D**. Best R2 model: **D**. Best SMAPE model: **D**.

## Architecture registry

| Model | Architecture | Backbone | Temporal module |
| --- | --- | --- | --- |
| A | EfficientNet-B0 + CBAM + GRU | EfficientNet-B0 | GRU |
| B | MobileNetV2 + CBAM + LSTM | MobileNetV2 | LSTM |
| C | EfficientNet-B0 + CBAM + LSTM | EfficientNet-B0 | LSTM |
| D | MobileNetV2 + CBAM + GRU | MobileNetV2 | GRU |

## Fold-averaged LOOCV metrics

| rank_by_mae | model_key | folds | test_mae_mean | test_mae_std | test_rmse_mean | test_r2_mean | test_smape_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | D | 6.000 | 14.010 | 6.876 | 16.902 | 0.488 | 39.650 |
| 2.000 | B | 6.000 | 15.056 | 5.434 | 17.932 | 0.453 | 43.074 |
| 3.000 | A | 6.000 | 15.235 | 8.362 | 17.418 | 0.424 | 46.330 |
| 4.000 | C | 6.000 | 15.901 | 6.951 | 19.217 | 0.359 | 44.859 |

Primary ranking uses unweighted LOOCV fold means, so each held-out fruit contributes equally.

## Sample-weighted held-out prediction metrics

| model_key | n_predictions | weighted_mae | weighted_rmse | weighted_r2 | weighted_smape | bias_pred_minus_actual | median_absolute_error | p90_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D | 3894.000 | 14.150 | 18.329 | 0.534 | 39.325 | 1.736 | 10.996 | 33.377 |
| B | 3894.000 | 15.156 | 18.909 | 0.504 | 42.665 | 3.366 | 13.122 | 32.062 |
| A | 3894.000 | 15.465 | 19.193 | 0.489 | 47.012 | 2.719 | 14.846 | 31.562 |
| C | 3894.000 | 16.125 | 20.452 | 0.420 | 44.919 | 1.805 | 14.000 | 34.749 |

These metrics weight fruits with more generated sequences more heavily; use them as diagnostics rather than the main LOOCV ranking.

## Best model by held-out fruit

| test_group | model_key | test_mae | test_rmse | test_r2 | test_smape |
| --- | --- | --- | --- | --- | --- |
| F01 | A | 4.649 | 5.718 | 0.940 | 19.283 |
| F02 | D | 16.789 | 20.213 | 0.553 | 39.334 |
| F03 | B | 22.141 | 23.242 | 0.005 | 59.395 |
| F04 | D | 5.749 | 7.064 | 0.908 | 27.061 |
| F05 | D | 14.490 | 19.184 | 0.597 | 33.780 |
| F06 | D | 17.488 | 21.784 | 0.135 | 53.163 |

## Worst held-out fruit per model

| model_key | test_group | test_mae | test_rmse | test_r2 | test_smape |
| --- | --- | --- | --- | --- | --- |
| A | F03 | 26.377 | 27.136 | -0.356 | 65.632 |
| B | F03 | 22.141 | 23.242 | 0.005 | 59.395 |
| C | F03 | 22.759 | 25.042 | -0.155 | 61.110 |
| D | F03 | 23.342 | 24.832 | -0.136 | 61.553 |

## Generated graphs

- [training_curves_comparison.png](../../../graphs/evaluation/strawberry_loocv_seq8_image_only/training_curves_comparison.png)
- [test_metrics_comparison.png](../../../graphs/evaluation/strawberry_loocv_seq8_image_only/test_metrics_comparison.png)
- [predicted_vs_actual.png](../../../graphs/evaluation/strawberry_loocv_seq8_image_only/predicted_vs_actual.png)
- [residual_distribution.png](../../../graphs/evaluation/strawberry_loocv_seq8_image_only/residual_distribution.png)
- [fold_metrics_heatmap.png](../../../graphs/evaluation/strawberry_loocv_seq8_image_only/fold_metrics_heatmap.png)
- [fold_mae_by_fruit.png](../../../graphs/evaluation/strawberry_loocv_seq8_image_only/fold_mae_by_fruit.png)
- [model_stability_boxplots.png](../../../graphs/evaluation/strawberry_loocv_seq8_image_only/model_stability_boxplots.png)
- [model_params_comparison.png](../../../graphs/evaluation/strawberry_loocv_seq8_image_only/model_params_comparison.png)

## Interpretation notes

- Negative fold R2 means the model underperformed a constant-mean predictor on that held-out fruit.
- Folds with high MAE should be audited for unusual visual degradation patterns, sensor gaps, or lifecycle label noise.
- For this run, select the lowest fold-averaged MAE model unless deployment constraints require a lighter model.
