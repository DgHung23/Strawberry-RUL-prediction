# 🅱️  Model B — Detailed Report

**Architecture:** MobileNetV2 + CBAM + LSTM

*Auto-generated report — see `src/stage5_evaluation/generate_model_report.py`*

---

## 1. Executive Summary

> **One-sentence summary:** Model B predicts strawberry shelf life with an average error of **53.8 hours** and achieves **weak** accuracy (R² = 0.403).

| Metric | Value | What This Means |
|--------|-------|-----------------|
| **MAE** | **53.8 hours** | On average, predictions are off by about 54 hours (~2.2 days)
| **RMSE** | **61.5 hours** | Large errors are penalized — worst mistakes average ~62 hours
| **R²** | **0.403** | Model B explains **40.3%** of the variation in fruit shelf life
| **MAPE** | **108.7%** | The average error is about 109% of the true RUL value

### How to Read R²

R² = 0.403 means:

- ⚠️ The model finds some signal but is often inaccurate
- ⚠️ Consider this a baseline — try more data, augmentation, or architecture changes

## 2. How This Model Works

Model B is a hybrid AI that combines three specialized modules:

### Step-by-step pipeline

```text
  Strawberry photo (224x224)
       |
       v
  [1] MobileNetV2 (Vision) — see shapes, colors, textures
       |       MobileNetV2 (Lightweight Vision Module) — a smaller, faster version of the vision network. Best for quick predictions or running on limited hardware. Slightly less detailed than EfficientNet but ~40% fewer computations.
       v
  [2] CBAM (Attention) — focus on spoilage signs
       |       CBAM (Attention Module) — helps the model focus on the most important parts of each image. Instead of treating all pixels equally, CBAM learns to pay extra attention to signs of spoilage: mold spots, color changes, texture shifts. Think of it as the model's 'magnifying glass' over the fruit surface.
       v
  [3] LSTM (Memory) — track changes over 5 frames
       |       LSTM (Time-Series Memory, Enhanced) — similar to GRU but with a more detailed 'memory cell'. Better at capturing subtle long-term changes across frames, but uses slightly more computation.
       v
  [4] Prediction Head — output single number = hours left
```

## 3. Training Progress

The model was trained for **10 epochs** (cycles through the entire training set). Each epoch, it sees 2,614 training sequences (4 fruits × ~653 frames each, minus window overlap). After each epoch, it's tested on 598 validation sequences from Fruit #06 (never seen during training).

| Epoch | Train Loss (h) | Val Loss (h) | Notes |
|-------|---------------|-------------|-------|
| 1 | 43.97 | 31.42 | |
| 2 | 30.89 | 16.14 | |
| 3 | 19.30 | 11.69 | |
| 4 | 10.65 | 16.01 | |
| 5 | 7.44 | 12.99 | |
| 6 | 6.24 | 13.85 | |
| 7 | 5.82 | 21.05 | |
| 8 | 5.63 | 11.13 | ⭐ Best epoch (lowest validation error) |
| 9 | 5.36 | 18.13 | ⚠️ Starting to overfit |
| 10 | 5.11 | 16.70 | ⚠️ Starting to overfit |

### What the loss curves tell us

- **Best model saved at epoch 8** (validation error = 11.1 hours)
- **Final training error:** 5.1 hours
- **Final validation error:** 16.7 hours
- ⚠️ **Overfitting detected:** After epoch 8, the model started memorizing training data rather than learning general patterns. Validation error increased from 11.1h to 16.7h (50% worse).
  - **Fix:** Add dropout, freeze backbone earlier, use data augmentation, or reduce epochs.

## 4. Test Set Performance (Final Evaluation)

The best checkpoint (epoch 8) is evaluated on **672 test sequences** from Fruit #05 — a completely held-out fruit never seen during training or validation.

### Error Distribution

| Percentile | Error (hours) | Interpretation |
|-----------|---------------|----------------|
| **P5** | 5.1h |  |
| **P10** | 13.7h |  |
| **P25** | 31.5h |  |
| **P50** | 50.1h | Half of predictions are this accurate or better |
| **P75** | 79.5h |  |
| **P90** | 95.6h | 90% of predictions are better than this |
| **P95** | 102.7h | Only the worst 5% exceed this |

- **Median error:** 50.1h (half of all predictions)
- **90% of predictions:** error ≤ 95.6h
- **Worst 5% of predictions:** error ≥ 102.7h
- **Mean bias:** +34.2h (model tends to overestimate RUL)

## 5. Performance Across Fruit Lifecycle

Does the model work equally well for fresh strawberries vs nearly-spoiled ones?

| RUL Range | Samples | Avg Error | Std Dev | Reliability |
|-----------|---------|-----------|---------|-------------|
| 0–50h | 370 | 66.5h | 33.2h | ❌ Poor |
| 50–100h | 95 | 45.7h | 10.8h | ❌ Poor |
| 100–150h | 43 | 14.3h | 3.1h | ✅ Good |
| 150–200h | 90 | 32.9h | 8.7h | ❌ Poor |
| 200–250h | 74 | 49.1h | 4.2h | ❌ Poor |
| 250–300h | 0 | — | — | — |

## 6. How to Improve

### Quick wins (low effort, possible gains)

- **Data augmentation:** Randomly flip, rotate, or adjust brightness of images during training to make the model more robust. Add to `dataset.py` transforms.
- **Longer sequences:** Change `seq_len` from 5 to 8 or 10 in `train.py` — gives the temporal model more context per prediction.
- **Learning rate schedule:** Reduce learning rate when validation loss plateaus (add `torch.optim.lr_scheduler.ReduceLROnPlateau`).
- **Early stopping:** Stop training automatically when validation loss stops improving for N epochs, instead of fixed 10 epochs.

### Architecture improvements (higher effort)

- **Larger EfficientNet variant:** B1 or B2 offers richer features at the cost of speed.
- **Deeper RNN:** Increase `num_layers` from 1 to 2 (may need more data).
- **Multi-head attention:** Replace or augment CBAM with transformer-style self-attention.
- **Try GRU instead:** Model B uses LSTM. Model D uses GRU with the same backbone — compare their reports.

## 7. File Inventory

| File | Path | Status |
|------|------|--------|
| Model checkpoint | `models\model_B\best_model.pth` | ✅ Exists |
| Performance metrics | `data\model_B_outputs\metrics.json` | ✅ |
| Epoch-by-epoch loss | `data\model_B_outputs\training_history.csv` | ✅ |
| Test set predictions | `data\model_B_outputs\test_predictions.csv` | ✅ |

---

*Report generated by `src/stage5_evaluation/generate_model_report.py`*  
*Model architecture: MobileNetV2 + CBAM + LSTM*  
*Training data: Fruits F01-F04 (2,614 sequences) | Validation: Fruit F06 (598 sequences) | Test: Fruit F05 (672 sequences)*
