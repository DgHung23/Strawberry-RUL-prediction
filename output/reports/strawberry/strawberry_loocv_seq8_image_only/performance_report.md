# Strawberry RUL model performance report

- Run root: `output\runs\strawberry\strawberry_loocv_seq8_image_only`
- Evaluation artifacts: `output\reports\strawberry\strawberry_loocv_seq8_image_only`
- Config: seq_len=8, fusion_mode=image_only, temporal_pooling=last_mean_max, epochs=18, patience=5, env_feature_mode=sensor
- Completed LOOCV folds: 24 total = A:6, B:6, C:6, D:6

## Executive summary

Model **D** is the best overall by fold-averaged test MAE: **14.01 +/- 6.88 hours**.
It also has the best RMSE (**16.90 h**), best R+/- (**0.488**), and best SMAPE (**39.65%**).
The MAE gap from the next best model (B) is **1.05 hours**, so D is ahead but the margin over B/A is modest.

## Fold-averaged LOOCV metrics

| rank_by_mae | model_key | folds | test_mae_mean | test_mae_std | test_rmse_mean | test_r2_mean | test_smape_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | D | 6 | 14.010 | 6.876 | 16.902 | 0.488 | 39.650 |
| 2 | B | 6 | 15.056 | 5.434 | 17.932 | 0.453 | 43.074 |
| 3 | A | 6 | 15.235 | 8.362 | 17.418 | 0.424 | 46.330 |
| 4 | C | 6 | 15.901 | 6.951 | 19.217 | 0.359 | 44.859 |

Primary ranking uses unweighted LOOCV fold means, so each held-out fruit contributes equally.

## Sample-weighted held-out prediction metrics

| model_key | n_predictions | weighted_mae | weighted_rmse | weighted_r2 | weighted_smape | bias_pred_minus_actual | median_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D | 3894 | 14.150 | 18.329 | 0.534 | 39.325 | 1.736 | 10.996 |
| B | 3894 | 15.156 | 18.909 | 0.504 | 42.665 | 3.366 | 13.122 |
| A | 3894 | 15.465 | 19.193 | 0.489 | 47.012 | 2.719 | 14.846 |
| C | 3894 | 16.125 | 20.452 | 0.420 | 44.919 | 1.805 | 14.000 |

These metrics weight fruits with more generated sequences more heavily, so they are useful diagnostics but not the main LOOCV ranking.

## Best model by held-out fruit

| test_group | model_key | test_mae | test_rmse | test_r2 |
| --- | --- | --- | --- | --- |
| F01 | A | 4.649 | 5.718 | 0.940 |
| F02 | D | 16.789 | 20.213 | 0.553 |
| F03 | B | 22.141 | 23.242 | 0.005 |
| F04 | D | 5.749 | 7.064 | 0.908 |
| F05 | D | 14.490 | 19.184 | 0.597 |
| F06 | D | 17.488 | 21.784 | 0.135 |

## Worst fold per model

| model_key | test_group | test_mae | test_rmse | test_r2 |
| --- | --- | --- | --- | --- |
| A | F03 | 26.377 | 27.136 | -0.356 |
| B | F03 | 22.141 | 23.242 | 0.005 |
| C | F03 | 22.759 | 25.042 | -0.155 |
| D | F03 | 23.342 | 24.832 | -0.136 |

## Graphs

- [01_model_metric_summary.png](graphs/01_model_metric_summary.png): Model-level MAE/RMSE/R+/-/SMAPE comparison with fold std.
- [02_fold_mae_grouped_bar.png](graphs/02_fold_mae_grouped_bar.png): Per-held-out-fruit MAE grouped by model.
- [03_fold_mae_heatmap.png](graphs/03_fold_mae_heatmap.png): MAE heatmap across LOOCV folds.
- [04_fold_r2_heatmap.png](graphs/04_fold_r2_heatmap.png): R+/- heatmap across LOOCV folds.
- [05_predicted_vs_actual.png](graphs/05_predicted_vs_actual.png): Held-out prediction scatter.
- [06_residual_boxplot.png](graphs/06_residual_boxplot.png): Residual distribution by model.
- [07_training_val_mae_curves.png](graphs/07_training_val_mae_curves.png): Mean validation MAE curve across folds.

## Notes

- `predictions.csv` files are interpreted as held-out fold predictions because each file lives under `model/holdout_Fxx/`; the CSV `split` column is the source split label of that fruit, not the LOOCV role.
- Negative R+/- on a fold means the model underperformed a constant-mean predictor for that held-out fruit.
- For deployment/model selection, prefer Model D from this run unless later reruns show a statistically larger reversal.
