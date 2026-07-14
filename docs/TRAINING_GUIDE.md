# Strawberry RUL Prediction — Model Training Guide

> **Target audience:** Developers continuing this project, AI assistants reading the codebase, new team members onboarding.

This document covers every aspect of model training: architecture design, data flow, how to run training, how to evaluate, and how to compare models. Read this before modifying any file under `src/stage4_training/`.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [What Problem Are We Solving?](#what-problem-are-we-solving)
3. [Architecture Design](#architecture-design)
4. [The Four Model Variants](#the-four-model-variants)
5. [CBAM: Where Attention Lives](#cbam-where-attention-lives)
6. [Data Pipeline](#data-pipeline)
7. [How to Train](#how-to-train)
8. [Training Outputs](#training-outputs)
9. [How to Predict (Single Image)](#how-to-predict-single-image)
10. [How to Compare Models](#how-to-compare-models)
11. [Expected Results](#expected-results)
12. [Directory Map](#directory-map)
13. [Hyperparameter Reference](#hyperparameter-reference)
14. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Train all 4 models (run each in its own terminal):
cd src/stage4_training/model_A && python train.py
cd src/stage4_training/model_B && python train.py
cd src/stage4_training/model_C && python train.py
cd src/stage4_training/model_D && python train.py

# After training, generate comparison charts and report:
python src/stage5_evaluation/compare_models.py

# Predict RUL from a single image:
cd src/stage4_training/model_A
python predict.py --image data/03_split/test/F05/images/xxx.png --temp 22.5 --humidity 60.5
```

---

## What Problem Are We Solving?

**Task:** Predict the **Remaining Useful Life (RUL)** of a strawberry, in hours, from a sequence of images + environmental sensor data.

**Input:**
- A sequence of 5 consecutive images (224x224 RGB) of the same fruit
- Temperature (°C) and humidity (%) at each time step

**Output:**
- A single number: estimated hours until the fruit spoils

**Ground truth formula (from [LABELING_PROTOCOL.md](LABELING_PROTOCOL.md)):**
```
RUL_hours = EOL_timestamp_for_that_fruit - sample_timestamp
```

**Split policy (fruit-ID-safe, no leakage):**

| Split | Fruits | Purpose |
|-------|--------|---------|
| Train | F01, F02, F03, F04 | Model learning |
| Validation | F06 | Early stopping, checkpoint selection |
| Test | F05 | Final held-out evaluation |

---

## Architecture Design

Every model follows the same **5-stage pipeline**. The only differences are which CNN backbone and which RNN temporal model are used.

```
                      INPUT
                        |
    +-------------------+-------------------+
    |                                       |
    |  5-frame image sequence               |  Environmental data
    |  (B, 5, 3, 224, 224)                  |  (B, 5, 2)
    |                                       |
    v                                       |
+-----------------------+                   |
|  CNN Backbone          |                   |
|  (per-frame, shared)   |                   |
|  EfficientNet-B0  OR   |                   |
|  MobileNetV2           |                   |
|                        |                   |
|  Output: (B*5, 1280, 7, 7)               |
+-----------+-----------+                   |
            |                               |
            v                               |
+-----------------------+                   |
|  CBAM Attention        |                   |
|  Channel Attn +        |                   |
|  Spatial Attn          |                   |
|                        |                   |
|  Output: (B*5, 1280, 7, 7)               |
+-----------+-----------+                   |
            |                               |
            v                               |
+-----------------------+                   |
|  Global Average Pool   |                   |
|  Output: (B*5, 1280)   |                   |
+-----------+-----------+                   |
            |                               |
            +-----> Reshape to (B, 5, 1280) |
                        |                   |
                        v                   v
              +---------+---------+---------+
              |  Concatenate per-timestep    |
              |  (B, 5, 1280 + 2) = (B, 5, 1282)
              +-------------------+---------+
                                  |
                                  v
                    +-------------+-------------+
                    |  Temporal Model            |
                    |  GRU  OR  LSTM             |
                    |  hidden_size=128, layers=1  |
                    |                             |
                    |  Output: (B, 5, 128)        |
                    +-------------+---------------+
                                  |
                                  v (take last timestep)
                    +-------------+-------------+
                    |  Regression Head            |
                    |  Linear(128->64) -> ReLU    |
                    |  -> Dropout(0.2)            |
                    |  -> Linear(64->1)           |
                    |                             |
                    |  Output: (B, 1) — RUL hours |
                    +-----------------------------+
```

**Key design decisions:**
- CNN backbone weights are pretrained on ImageNet (transfer learning)
- CBAM adds ~205K trainable parameters and sits on the **feature maps** (before pooling), not on the pooled vector — this lets it learn spatial attention
- Environmental features (temp, humidity) are concatenated AFTER pooling, at the per-timestep vector level
- Only the **last** GRU/LSTM timestep output goes to the regression head
- Loss is L1Loss (MAE) — directly interpretable as "hours of error"

---

## The Four Model Variants

We train all 4 combinations to isolate the effect of each component:

| Model | CNN Backbone | Temporal | Class Name | Source Dir | Checkpoint Dir |
|-------|-------------|----------|------------|-----------|---------------|
| **A** | EfficientNet-B0 | GRU | `StrawberryRULModel` | `src/stage4_training/model_A/` | `models/model_A/` |
| **B** | MobileNetV2 | LSTM | `StrawberryRULModelB` | `src/stage4_training/model_B/` | `models/model_B/` |
| **C** | EfficientNet-B0 | LSTM | `StrawberryRULModelC` | `src/stage4_training/model_C/` | `models/model_C/` |
| **D** | MobileNetV2 | GRU | `StrawberryRULModelD` | `src/stage4_training/model_D/` | `models/model_D/` |

### Why these 4?

```
                    GRU              LSTM
EfficientNet-B0   Model A (4.76M)   Model C (4.94M)   <- stronger CNN
MobileNetV2       Model D (2.98M)   Model B (3.16M)   <- lighter CNN
```

| Comparison | Models to compare | What it tells us |
|------------|-------------------|------------------|
| CNN effect (GRU) | A vs D | Does EfficientNet outperform MobileNet with the same RNN? |
| CNN effect (LSTM) | C vs B | Same question, LSTM variant |
| RNN effect (EfficientNet) | A vs C | GRU vs LSTM with the same CNN |
| RNN effect (MobileNet) | D vs B | Same question, MobileNet variant |
| Lightest vs heaviest | D vs C | Speed/performance trade-off |

---

## CBAM: Where Attention Lives

**File:** [src/shared/cbam.py](../src/shared/cbam.py)

CBAM (Convolutional Block Attention Module, Woo et al. ECCV 2018) sits between the CNN conv features and global pooling. Every model uses the same CBAM module.

```
CNN Feature Maps   (B, 1280, H, W)
      |
      v
ChannelAttention   — "which of the 1280 channels matter?"
  - AvgPool spatial -> MLP -> sigmoid
  - MaxPool spatial -> MLP -> sigmoid
  - Element-wise multiply with input
      |
      v
SpatialAttention   — "where in the (H,W) map matters?"
  - AvgPool across channels -> conv 7x7 -> sigmoid
  - MaxPool across channels  -> conv 7x7 -> sigmoid
  - Element-wise multiply with input
      |
      v
Refined Feature Maps (B, 1280, H, W)
```

**Why CBAM matters for strawberry RUL:**
- Strawberries degrade through visible markers: mold spots, color darkening, texture changes, shrinkage
- These markers are spatially localized (a mold spot on one side) and channel-specific (certain CNN filters detect mold textures)
- CBAM helps the model attend to these signals rather than background or lighting variations

**In code, CBAM is used in each model like this:**
```python
feat_maps = self.cnn_features(images)   # (N, 1280, 7, 7)
attended = self.cbam(feat_maps)          # CBAM refines
pooled = self.cnn_pool(attended).flatten(1)  # -> (N, 1280)
```

---

## Data Pipeline

### Dataset class: `StrawberrySequenceDataset`

**Location:** One copy in each model directory (identical across all 4)

**What it does:**

1. Scans `data/03_split/{train,val,test}/` for `F*/` subdirectories
2. Per fruit directory, reads `labels.csv` sorted by timestamp
3. Creates rolling windows: frames [i, i+1, ..., i+seq_len-1]
4. Target RUL = `rul_hours` of the **last** frame in each window
5. Loads images from `F*/images/`, applies transforms:
   - Resize to 224x224
   - ToTensor
   - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (ImageNet stats)
6. Normalizes env features: temperature/30.0, humidity/100.0
7. Returns: `(images [seq_len, 3, 224, 224], env [seq_len, 2], rul [1])`

### Expected data layout

```
data/03_split/
  train/
    F01/labels.csv + images/*.png
    F02/labels.csv + images/*.png
    F03/labels.csv + images/*.png
    F04/labels.csv + images/*.png
  val/
    F06/labels.csv + images/*.png
  test/
    F05/labels.csv + images/*.png
```

`labels.csv` must have columns: `timestamp`, `image_path`, `temperature_c`, `humidity_pct`, `rul_hours`

### Sequence counts (with seq_len=5 and current data)

| Split | F01 | F02 | F03 | F04 | F05 | F06 | Total |
|-------|-----|-----|-----|-----|-----|-----|-------|
| Train | ~654 | ~653 | ~654 | ~653 | — | — | ~2614 |
| Val | — | — | — | — | — | ~598 | ~598 |
| Test | — | — | — | — | ~672 | — | ~672 |

---

## How to Train

### Prerequisites

```bash
pip install torch torchvision numpy pandas pillow tqdm
```

GPU recommended but not required (CPU works, just slower).

### Training a single model

```bash
cd src/stage4_training/model_A   # or model_B, model_C, model_D
python train.py
```

### What happens during training

1. **Data loading:** Creates `StrawberrySequenceDataset` for train (F01-F04), val (F06), test (F05)
2. **Model init:** Loads pretrained ImageNet backbone, adds CBAM + RNN + regression head
3. **10 epochs:**
   - Train loop: forward pass, MAE loss, backprop, Adam(1e-4) update
   - Validation loop: compute val loss (no gradient)
   - Save checkpoint if val_loss improves
4. **Final test:** Load best checkpoint, evaluate on F05, compute MAE/RMSE/MAPE/R^2

### Hyperparameters

Defined at the top of each `train.py`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `batch_size` | 4 | Small — sequences are memory-heavy |
| `num_epochs` | 10 | Models converge quickly on this dataset |
| `learning_rate` | 1e-4 | Adam optimizer |
| `seq_len` | 5 | 5 consecutive frames per sequence |
| `rnn_hidden_size` | 128 | GRU/LSTM hidden state |
| `num_layers` | 1 | Single RNN layer |
| `dropout` | 0.2 | Applied in RNN and regression head |
| `cbam_reduction_ratio` | 16 | Channel attention bottleneck |
| `cbam_kernel_size` | 7 | Spatial attention conv kernel |
| `freeze_backbone` | False | Set True in model.py to freeze CNN |

### Training time (estimate)

| Model | Hardware | ~Time |
|-------|----------|-------|
| A (EfficientNet + GRU) | CPU | ~5-10 min |
| B (MobileNetV2 + LSTM) | CPU | ~3-6 min |
| C (EfficientNet + LSTM) | CPU | ~5-10 min |
| D (MobileNetV2 + GRU) | CPU | ~3-6 min |

GPU is ~10x faster for all models.

---

## Training Outputs

Each model produces these files:

```
models/model_{X}/
  best_model.pth              # PyTorch state_dict (best val loss)

data/model_{X}_outputs/
  training_history.csv        # epoch, train_loss, val_loss
  test_predictions.csv        # predicted_rul, actual_rul
  metrics.json                # MAE, RMSE, MAPE, R2, sequence counts
```

### training_history.csv

```csv
epoch,train_loss,val_loss
1,43.482,30.603
2,29.851,19.446
...
```

### test_predictions.csv

```csv
predicted_rul,actual_rul
177.86,234.56
179.34,234.31
...
```

### metrics.json

```json
{
  "model": "Model_A_EfficientNet-B0_CBAM_GRU",
  "mae": 43.14,
  "rmse": 49.87,
  "mape": 99.61,
  "r2": 0.6075,
  "train_sequences": 2614,
  "val_sequences": 598,
  "test_sequences": 672
}
```

### Metrics interpretation

| Metric | What it means | Target |
|--------|--------------|--------|
| **MAE** | Average error in hours | Lower is better |
| **RMSE** | Penalizes large errors more | Lower is better |
| **MAPE** | Percentage error (may be high when RUL is small) | Lower is better |
| **R^2** | How much variance is explained (0=mean prediction, 1=perfect) | Higher is better (>0.5 is usable) |

---

## How to Predict (Single Image)

```bash
# From model_A directory:
python predict.py \
  --image data/03_split/test/F05/images/2026-04-01_08-00-06_frame-1_F05.png \
  --temp 22.5 \
  --humidity 60.5

# Optional: use a different checkpoint
python predict.py --image ... --temp 22.5 --humidity 60.5 \
  --checkpoint models/model_A/best_model.pth
```

**How single-image inference works:**
- The image is wrapped into a pseudo-sequence of length 1: `(1, 1, 3, 224, 224)`
- This means the GRU/LSTM only sees one timestep (no temporal context)
- For best accuracy, use sequences of 5 frames from the same fruit

---

## How to Compare Models

After training at least 2 models, run the comparison script:

```bash
python src/stage5_evaluation/compare_models.py
```

**Output (auto-generated):**

```
output/graphs/evaluation/
  training_curves_comparison.png    # Loss curves overlaid
  test_metrics_comparison.png       # MAE/RMSE/R2 bar charts
  predicted_vs_actual.png           # Scatter plots per model
  residual_distribution.png         # Error histograms
  model_params_comparison.png       # Parameter counts

output/reports/evaluation/
  model_comparison_report.md        # Tabulated summary
```

The script automatically discovers which models have been trained and only includes those with data. Models C and D are included when their training completes.

---

## Expected Results

Based on initial runs with the current data (seq_len=5, 10 epochs, Adam, MAE loss):

| Model | MAE (h) | R^2 | Notes |
|-------|---------|-----|-------|
| A (EfficientNet+GRU) | ~43.1 | ~0.61 | Best so far — strong CNN + efficient RNN |
| B (MobileNetV2+LSTM) | ~53.8 | ~0.40 | Lighter CNN, weaker results |
| C (EfficientNet+LSTM) | TBD | TBD | Expected: similar or slightly worse than A |
| D (MobileNetV2+GRU) | TBD | TBD | Expected: fastest, may be close to B |

**Key observation from initial runs:**
- The EfficientNet backbone matters more than the RNN choice
- Both models show overfitting after epoch 3-4 (val loss increases while train loss continues dropping)
- The MAE of 43 hours is promising for a prototype; improvements may come from:
  - Data augmentation
  - Longer sequences
  - Learning rate scheduling
  - More epochs with early stopping
  - Larger EfficientNet variants (B1, B2)

---

## Directory Map

```
Strawberry-RUL-prediction/
|
|-- src/
|   |-- shared/
|   |   |-- cbam.py                    # CBAM module used by ALL models
|   |
|   |-- stage4_training/
|   |   |-- model_A/                   # EfficientNet-B0 + CBAM + GRU
|   |   |   |-- model.py               #   StrawberryRULModel class
|   |   |   |-- dataset.py             #   StrawberrySequenceDataset
|   |   |   |-- train.py               #   Training loop
|   |   |   |-- predict.py             #   Single-image inference
|   |   |
|   |   |-- model_B/                   # MobileNetV2 + CBAM + LSTM
|   |   |   |-- model.py               #   StrawberryRULModelB
|   |   |   |-- (same structure)
|   |   |
|   |   |-- model_C/                   # EfficientNet-B0 + CBAM + LSTM
|   |   |   |-- model.py               #   StrawberryRULModelC
|   |   |   |-- (same structure)
|   |   |
|   |   |-- model_D/                   # MobileNetV2 + CBAM + GRU
|   |   |   |-- model.py               #   StrawberryRULModelD
|   |   |   |-- (same structure)
|   |
|   |-- stage5_evaluation/
|       |-- compare_models.py           # Cross-model comparison script
|
|-- models/
|   |-- model_A/best_model.pth          # Trained checkpoints
|   |-- model_B/best_model.pth
|   |-- model_C/best_model.pth
|   |-- model_D/best_model.pth
|
|-- data/
|   |-- 03_split/                       # Input data (fruit-ID split)
|   |-- model_A_outputs/                # Training outputs per model
|   |   |-- training_history.csv
|   |   |-- test_predictions.csv
|   |   |-- metrics.json
|   |-- model_B_outputs/  (same)
|   |-- model_C_outputs/  (same)
|   |-- model_D_outputs/  (same)
|
|-- output/
|   |-- graphs/evaluation/              # Generated comparison charts
|   |-- reports/evaluation/             # Generated comparison report
|
|-- docs/
|   |-- TRAINING_GUIDE.md               # THIS FILE
|   |-- model_A/README.md
|   |-- model_A/TRAINING_AND_PREDICTION.md
|   |-- model_B/README.md
|   |-- model_B/TRAINING_AND_PREDICTION.md
|   |-- model_C/README.md
|   |-- model_C/TRAINING_AND_PREDICTION.md
|   |-- model_D/README.md
|   |-- model_D/TRAINING_AND_PREDICTION.md
```

---

## Hyperparameter Reference

All tunable hyperparameters and where to change them:

### In `model.py` (per model)

| Parameter | Location | Default |
|-----------|----------|---------|
| `rnn_hidden_size` | `__init__` | 128 |
| `num_layers` | `__init__` | 1 |
| `dropout` | `__init__` | 0.2 |
| `cbam_reduction_ratio` | `__init__` | 16 |
| `cbam_kernel_size` | `__init__` | 7 |
| `freeze_backbone` | `__init__` | False |

### In `train.py` (per model)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `batch_size` | 4 | Increase if GPU memory allows |
| `num_epochs` | 10 | Early stopping not implemented yet |
| `learning_rate` | 1e-4 | Adam optimizer |
| `seq_len` | 5 | Also affects dataset construction |

### In `dataset.py` (per model)

| Parameter | Default | Notes |
|-----------|---------|-------|
| Image resize | 224x224 | Both backbones expect this |
| Normalization mean | ImageNet | [0.485, 0.456, 0.406] |
| Normalization std | ImageNet | [0.229, 0.224, 0.225] |
| Temp normalization | /30.0 | Rough range normalization |
| Humidity normalization | /100.0 | Percentage to [0,1] |

---

## Troubleshooting

### "Checkpoint not found" during predict
Run `train.py` first. The checkpoint is saved to `models/model_X/best_model.pth`.

### CUDA out of memory
Reduce `batch_size` in `train.py` from 4 to 2 or 1.

### Dataset returns 0 sequences
Check that `data/03_split/{train,val,test}/` exist and contain `F*/labels.csv` with the expected columns.

### Model shape mismatch when loading checkpoint
The checkpoint was saved from a different model version. Delete the old checkpoint and retrain.

### Import error: "No module named src.shared.cbam"
Run from the model directory (e.g., `cd src/stage4_training/model_A`), not from the project root. Or add the project root to PYTHONPATH:

```bash
cd Strawberry-RUL-prediction
PYTHONPATH=. python src/stage4_training/model_A/train.py
```

### All models predict similar values (collapsed model)
This happened in the prototype. Check:
- Is the backbone frozen? Try setting `freeze_backbone=False`
- Is the data normalized correctly?
- Are RUL values in reasonable range (0-300 hours)?
- Try reducing learning rate to 1e-5

### Training loss decreases but val loss increases (overfitting)
This is normal for small datasets. Mitigations:
- Add Dropout (already at 0.2 — try 0.4)
- Freeze backbone: `freeze_backbone=True` in model `__init__`
- Reduce `num_epochs` and use the best checkpoint
- Implement early stopping (not yet in code)
