# Model C: EfficientNet-B0 + CBAM + LSTM

## Overview

Model C is a hybrid deep learning model for strawberry RUL (Remaining Useful Life) prediction. It combines EfficientNet-B0 for spatial feature extraction, CBAM (Convolutional Block Attention Module) for attention-based feature refinement, and LSTM for temporal sequence modeling.

## Architecture

```
Input: 5-frame image sequence (224×224) + environmental data (temp, humidity)
  │
  ├── EfficientNet-B0 (features)  →  1280-dim feature maps (7×7)
  │
  ├── CBAM Attention
  │     ├── Channel Attention   →  "what" features matter
  │     └── Spatial Attention   →  "where" in the image matters
  │
  ├── Global Average Pooling  →  1280-dim vector
  │
  ├── Concatenate env features (temp/30, humidity/100)  →  1282-dim
  │
  ├── LSTM (hidden=128, layers=1)  →  temporal modeling
  │
  └── Regression Head (128→64→ReLU→Dropout→1)  →  RUL (hours)
```

## Why This Combination

| Component | Choice | Rationale |
|-----------|--------|-----------|
| CNN Backbone | **EfficientNet-B0** | Strong feature extraction with compound-scaled architecture; balances accuracy and efficiency |
| Attention | **CBAM** | Highlights salient visual features (mold, color changes, texture degradation) correlated with RUL |
| Temporal | **LSTM** | Explicit cell state for long-term dependency modeling over time sequences |

## Comparison with Other Models

| Model | CNN | Attention | Temporal |
|-------|-----|-----------|----------|
| Model A | EfficientNet-B0 | CBAM | GRU |
| Model B | MobileNetV2 | CBAM | LSTM |
| **Model C** | **EfficientNet-B0** | **CBAM** | **LSTM** |
| Model D | MobileNetV2 | CBAM | GRU |

Model C uses the most powerful CNN (EfficientNet) with the more expressive temporal model (LSTM), making it potentially the best-performing but also the most computationally expensive variant.

## Directory Structure

```
src/stage4_training/model_C/
├── model.py      # StrawberryRULModelC class definition
├── dataset.py    # StrawberrySequenceDataset (sequence-based data loading)
├── train.py      # Training loop with validation & test evaluation
└── predict.py    # Single-image inference script
```

## How to Train

```bash
cd src/stage4_training/model_C
python train.py
```

**Default hyperparameters:**
- Batch size: 4
- Epochs: 10
- Learning rate: 1e-4 (Adam)
- Sequence length: 5 frames
- Loss: L1Loss (MAE) — interpretable in hours

**Outputs:**
- `models/model_C/best_model.pth` — best checkpoint (by validation loss)
- `data/model_C_outputs/training_history.csv` — epoch-wise train/val loss
- `data/model_C_outputs/test_predictions.csv` — test set predictions
- `data/model_C_outputs/metrics.json` — final test metrics (MAE, RMSE, MAPE, R²)

## How to Predict

```bash
cd src/stage4_training/model_C
python predict.py --image path/to/image.png --temp 22.5 --humidity 60.5
```

Optional: `--checkpoint path/to/custom_checkpoint.pth`

For single-image inference, the image is pseudo-sequenced (seq_len=1) through the model.
