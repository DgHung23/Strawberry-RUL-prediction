# 🅳  Model D — Detailed Report

**Architecture:** MobileNetV2 + CBAM + GRU

*Auto-generated report — see `src/stage5_evaluation/generate_model_report.py`*

---

## 1. Executive Summary

> **One-sentence summary:** Model D predicts strawberry shelf life with an average error of **54.4 hours** and achieves **weak** accuracy (R² = 0.320).

| Metric | Value | What This Means |
|--------|-------|-----------------|
| **MAE** | **54.4 hours** | On average, predictions are off by about 54 hours (~2.3 days)
| **RMSE** | **65.6 hours** | Large errors are penalized — worst mistakes average ~66 hours
| **R²** | **0.320** | Model D explains **32.0%** of the variation in fruit shelf life
| **MAPE** | **125.3%** | The average error is about 125% of the true RUL value

### How to Read R²

R² = 0.320 means:

- ⚠️ The model finds some signal but is often inaccurate
- ⚠️ Consider this a baseline — try more data, augmentation, or architecture changes

## 2. How This Model Works

Model D is a hybrid AI that combines three specialized modules:

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
  [3] GRU (Memory) — track changes over 5 frames
       |       GRU (Time-Series Memory) — tracks how the fruit changes across the 5-frame video sequence. A GRU is a simplified recurrent unit that remembers previous frames to understand the speed of degradation.
       v
  [4] Prediction Head — output single number = hours left
```

## 3. Training Progress

The model was trained for **10 epochs** (cycles through the entire training set). Each epoch, it sees 2,614 training sequences (4 fruits × ~653 frames each, minus window overlap). After each epoch, it's tested on 598 validation sequences from Fruit #06 (never seen during training).

| Epoch | Train Loss (h) | Val Loss (h) | Notes |
|-------|---------------|-------------|-------|
| 1 | 43.31 | 25.53 | |
| 2 | 29.63 | 17.62 | |
| 3 | 18.99 | 15.10 | |
| 4 | 10.41 | 8.74 | ⭐ Best epoch (lowest validation error) |
| 5 | 7.32 | 9.00 | |
| 6 | 6.17 | 11.27 | ⚠️ Starting to overfit |
| 7 | 5.50 | 13.65 | ⚠️ Starting to overfit |
| 8 | 5.28 | 11.06 | ⚠️ Starting to overfit |
| 9 | 5.01 | 12.65 | ⚠️ Starting to overfit |
| 10 | 4.84 | 13.26 | ⚠️ Starting to overfit |

### What the loss curves tell us

- **Best model saved at epoch 4** (validation error = 8.7 hours)
- **Final training error:** 4.8 hours
- **Final validation error:** 13.3 hours
- ⚠️ **Overfitting detected:** After epoch 4, the model started memorizing training data rather than learning general patterns. Validation error increased from 8.7h to 13.3h (52% worse).
  - **Fix:** Add dropout, freeze backbone earlier, use data augmentation, or reduce epochs.

## 4. Test Set Performance (Final Evaluation)

The best checkpoint (epoch 4) is evaluated on **672 test sequences** from Fruit #05 — a completely held-out fruit never seen during training or validation.

### Error Distribution

| Percentile | Error (hours) | Interpretation |
|-----------|---------------|----------------|
| **P5** | 0.0h |  |
| **P10** | 0.0h |  |
| **P25** | 31.3h |  |
| **P50** | 56.8h | Half of predictions are this accurate or better |
| **P75** | 80.7h |  |
| **P90** | 103.7h | 90% of predictions are better than this |
| **P95** | 120.1h | Only the worst 5% exceed this |

- **Median error:** 56.8h (half of all predictions)
- **90% of predictions:** error ≤ 103.7h
- **Worst 5% of predictions:** error ≥ 120.1h
- **Mean bias:** +33.3h (model tends to overestimate RUL)

## 5. Performance Across Fruit Lifecycle

Does the model work equally well for fresh strawberries vs nearly-spoiled ones?

| RUL Range | Samples | Avg Error | Std Dev | Reliability |
|-----------|---------|-----------|---------|-------------|
| 0–50h | 370 | 58.0h | 44.5h | ❌ Poor |
| 50–100h | 95 | 67.8h | 13.7h | ❌ Poor |
| 100–150h | 43 | 36.6h | 3.6h | ❌ Poor |
| 150–200h | 90 | 23.9h | 12.2h | ⚠️ Fair |
| 200–250h | 74 | 66.8h | 10.6h | ❌ Poor |
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
- **Try LSTM instead:** Model D uses GRU. Model B uses LSTM with the same backbone — compare their reports.

## 7. File Inventory

| File | Path | Status |
|------|------|--------|
| Model checkpoint | `models\model_D\best_model.pth` | ✅ Exists |
| Performance metrics | `data\model_D_outputs\metrics.json` | ✅ |
| Epoch-by-epoch loss | `data\model_D_outputs\training_history.csv` | ✅ |
| Test set predictions | `data\model_D_outputs\test_predictions.csv` | ✅ |

---

*Report generated by `src/stage5_evaluation/generate_model_report.py`*  
*Model architecture: MobileNetV2 + CBAM + GRU*  
*Training data: Fruits F01-F04 (2,614 sequences) | Validation: Fruit F06 (598 sequences) | Test: Fruit F05 (672 sequences)*
