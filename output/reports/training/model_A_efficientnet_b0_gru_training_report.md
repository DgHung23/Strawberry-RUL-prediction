# Model A training report

- Backbone: efficientnet_b0
- Temporal model: GRU
- Sequence length: 8
- Train/validation/test sequences: {'train': 1414, 'validation': 303, 'test': 304}
- Best epoch: 3
- Training seconds: 3.82

## Test metrics

- MAE: 1.1671
- RMSE: 2.0014
- MAPE: 66696820.0000
- R2: 0.0000

- Best checkpoint: models\model_A\model_A_efficientnet_b0_gru_best.pt
- Last checkpoint: models\model_A\model_A_efficientnet_b0_gru_last.pt
- Metrics JSON: output\results\ori\model_A_efficientnet_b0_gru_metrics.json
- Test predictions CSV: output\results\ori\model_A_efficientnet_b0_gru_test_predictions.csv
- Training curve: output\graphs\training\model_A_efficientnet_b0_gru_training_curves.png