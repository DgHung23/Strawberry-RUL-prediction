# 🅰️  Model A — Detailed Report

**Architecture:** EfficientNet-B0 + CBAM + GRU

*Auto-generated report — see `src/stage5_evaluation/generate_model_report.py`*

---

## 1. Executive Summary

> **One-sentence summary:** Model A predicts strawberry shelf life with an average error of **43.1 hours** and achieves **moderate** accuracy (R² = 0.608).

| Metric | Value | What This Means |
|--------|-------|-----------------|
| **MAE** | **43.1 hours** | On average, predictions are off by about 43 hours (~1.8 days)
| **RMSE** | **49.9 hours** | Large errors are penalized — worst mistakes average ~50 hours
| **R²** | **0.608** | Model A explains **60.8%** of the variation in fruit shelf life
| **MAPE** | **99.6%** | The average error is about 100% of the true RUL value

### How to Read R²

R² = 0.608 means:

- ⚠️ The model is usable but not highly precise
- ⚠️ Predictions are better than guessing the average, but with room to improve

## 2. How This Model Works

Model A is a hybrid AI that combines three specialized modules:

### Step-by-step pipeline

```text
  Strawberry photo (224x224)
       |
       v
  [1] EfficientNet-B0 (Vision) — see shapes, colors, textures
       |       EfficientNet-B0 (Vision Module) — a pretrained neural network that has already learned to recognize shapes, textures, and colors from millions of real-world images. It converts each strawberry photo into a list of 1,280 numbers describing what it sees.
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
| 1 | 43.48 | 30.60 | |
| 2 | 29.85 | 19.45 | |
| 3 | 19.52 | 9.68 | |
| 4 | 10.54 | 9.48 | |
| 5 | 7.79 | 10.96 | |
| 6 | 6.44 | 10.91 | |
| 7 | 5.81 | 8.99 | ⭐ Best epoch (lowest validation error) |
| 8 | 5.64 | 18.23 | ⚠️ Starting to overfit |
| 9 | 5.65 | 19.05 | ⚠️ Starting to overfit |
| 10 | 5.39 | 13.84 | ⚠️ Starting to overfit |

### What the loss curves tell us

- **Best model saved at epoch 7** (validation error = 9.0 hours)
- **Final training error:** 5.4 hours
- **Final validation error:** 13.8 hours
- ⚠️ **Overfitting detected:** After epoch 7, the model started memorizing training data rather than learning general patterns. Validation error increased from 9.0h to 13.8h (54% worse).
  - **Fix:** Add dropout, freeze backbone earlier, use data augmentation, or reduce epochs.

## 4. Test Set Performance (Final Evaluation)

The best checkpoint (epoch 7) is evaluated on **672 test sequences** from Fruit #05 — a completely held-out fruit never seen during training or validation.

### Error Distribution

| Percentile | Error (hours) | Interpretation |
|-----------|---------------|----------------|
| **P5** | 0.0h |  |
| **P10** | 10.1h |  |
| **P25** | 24.3h |  |
| **P50** | 42.2h | Half of predictions are this accurate or better |
| **P75** | 59.6h |  |
| **P90** | 78.7h | 90% of predictions are better than this |
| **P95** | 87.5h | Only the worst 5% exceed this |

- **Median error:** 42.2h (half of all predictions)
- **90% of predictions:** error ≤ 78.7h
- **Worst 5% of predictions:** error ≥ 87.5h
- **Mean bias:** +26.6h (model tends to overestimate RUL)

## 5. Performance Across Fruit Lifecycle

Does the model work equally well for fresh strawberries vs nearly-spoiled ones?

| RUL Range | Samples | Avg Error | Std Dev | Reliability |
|-----------|---------|-----------|---------|-------------|
| 0–50h | 370 | 47.4h | 29.9h | ❌ Poor |
| 50–100h | 95 | 51.6h | 11.5h | ❌ Poor |
| 100–150h | 43 | 23.2h | 5.8h | ⚠️ Fair |
| 150–200h | 90 | 23.5h | 9.1h | ⚠️ Fair |
| 200–250h | 74 | 46.6h | 5.3h | ❌ Poor |
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
- **Try LSTM instead:** Model A uses GRU. Model C uses LSTM with the same backbone — compare their reports.

## 7. File Inventory

| File | Path | Status |
|------|------|--------|
| Model checkpoint | `models\model_A\best_model.pth` | ✅ Exists |
| Performance metrics | `data\model_A_outputs\metrics.json` | ✅ |
| Epoch-by-epoch loss | `data\model_A_outputs\training_history.csv` | ✅ |
| Test set predictions | `data\model_A_outputs\test_predictions.csv` | ✅ |

---

*Report generated by `src/stage5_evaluation/generate_model_report.py`*  
*Model architecture: EfficientNet-B0 + CBAM + GRU*  
*Training data: Fruits F01-F04 (2,614 sequences) | Validation: Fruit F06 (598 sequences) | Test: Fruit F05 (672 sequences)*
