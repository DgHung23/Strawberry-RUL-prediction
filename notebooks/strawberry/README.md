# Strawberry RUL Lab

This folder is the strawberry-only experiment lab. It intentionally ignores
`avocado` data and reads the current strawberry split from:

```text
data/03_split/strawberry/{train,val,test}
```

## Notebooks

- `00_lab_overview.ipynb`: project/data/model overview and experiment roadmap.
- `01_data_audit.ipynb`: split counts, RUL distribution, environment ranges, and
  sequence-count checks for several `seq_len` values.
- `02_ml_baseline_fusion_search.ipynb`: fast machine-learning search over
  sequence length and fusion style using image summary features plus temperature
  and humidity.
- `03_deep_fusion_ablation_plan.ipynb`: PyTorch experiment matrix for the four
  large model families A/B/C/D.
- `04_apply_best_to_model_c.ipynb`: apply the strongest lab configuration to
  Model C as the first large-model tuning target.

## Initial Findings

Current reported test metrics from `output/reports/evaluation/model_comparison_report.md`:

| Model | Architecture | Test MAE |
| --- | --- | ---: |
| A | EfficientNet-B0 + CBAM + GRU | 43.14 h |
| B | MobileNetV2 + CBAM + LSTM | 53.77 h |
| C | EfficientNet-B0 + CBAM + LSTM | 32.02 h |
| D | MobileNetV2 + CBAM + GRU | 54.44 h |

The quick ML baseline in this lab found that simple image statistics are highly
predictive. The best validation row from `results/ml_baseline_sweep.csv` was
`seq_len=10 + image_only + HistGradientBoosting`, with about
`val MAE=3.55h`, `test MAE=13.55h`, `test R2=0.933`. The best test row was close:
`seq_len=8 + image_only + HistGradientBoosting`, with about
`test MAE=13.13h`.

The sweep shows that longer windows can help when the model summarizes temporal
dynamics well:

| seq_len | Best image-only test MAE | Best early-fusion test MAE |
| ---: | ---: | ---: |
| 5 | 14.19 h | 14.60 h |
| 8 | 13.13 h | 14.08 h |
| 10 | 13.55 h | 14.25 h |

Environment-only and early-fusion models are useful checks, but they did not
beat image-only HGB in this first sweep. For deep models, keep temperature and
humidity as a real side branch, but avoid letting them dominate the visual
sequence signal.

Recommended first deep-model tune:

- Start from Model C, because it is the strongest current large model.
- Use `seq_len=10` for validation-first selection; use `seq_len=8` as the
  cheaper/test-best fallback if memory or speed is an issue.
- Use image-dominant fusion: process image sequence first, summarize
  environment separately, then fuse before the regression head.
- Add temporal pooling over the LSTM output (`last + mean + max`) instead of
  only using the last timestep.
- Use light augmentation, Huber/SmoothL1 loss, early stopping, and
  `ReduceLROnPlateau`.

## Tuned Model C Result

The applied deep-model run in `src/strawberry/stage4_training/model_C/train_tuned.py`
reached these metrics on the saved best checkpoint from epoch 3:

| Split | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| Val | 5.20 h | 9.13 h | 0.973 |
| Test | 25.73 h | 32.94 h | 0.825 |

That is an improvement over the historical Model C report (`32.02h` test MAE),
though the small ML baseline still wins on test MAE.

Note: there is also a newer raw `data/model_D_outputs/metrics.json` file with a
different split accounting than the historical report, so treat the report table
and raw metrics file as separate runs unless you explicitly harmonize them.

## Run Order

1. Run `01_data_audit.ipynb`.
2. Run `02_ml_baseline_fusion_search.ipynb`.
3. Use the best row from `notebooks/strawberry/results/ml_baseline_sweep.csv`.
4. Run `04_apply_best_to_model_c.ipynb` or the tuned Model C script once the
   desired GPU/runtime budget is available.
