# Model C: Training & Prediction Details

## Data Flow

### Dataset (`StrawberrySequenceDataset`)

1. Scans `data/03_split/{train,val,test}/` for fruit directories (F01-F06)
2. Per fruit, reads `labels.csv` sorted by timestamp
3. Creates rolling windows of `seq_len=5` consecutive frames
4. Target: `rul_hours` of the **last** frame in each window
5. Images: resized to 224×224, normalized with ImageNet stats
   - mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
6. Environmental features: temperature/30.0, humidity/100.0

### Split Policy

- **Train**: F01, F02, F03, F04 (4 fruits)
- **Validation**: F06 (1 fruit)
- **Test**: F05 (1 fruit)

This is a fixed fruit-ID-based split to prevent data leakage across fruits.

## Forward Pass Walk-through

### Step 1: CNN Feature Extraction (per frame)

```
Input: (B, S, 3, 224, 224)
  → Reshape to (B*S, 3, 224, 224)
  → EfficientNet-B0.features(x)  →  (B*S, 1280, 7, 7)
```

### Step 2: CBAM Attention

```
Feature maps (B*S, 1280, 7, 7)
  → Channel Attention (AvgPool + MaxPool → shared MLP → sigmoid)
  → Spatial Attention (AvgPool + MaxPool along channels → conv 7×7 → sigmoid)
  → Refined feature maps (B*S, 1280, 7, 7)
```

CBAM has ~205K trainable parameters (entirely within the attention modules).

### Step 3: Pooling & Fusion

```
Refined maps (B*S, 1280, 7, 7)
  → AdaptiveAvgPool2d(1)  →  (B*S, 1280, 1, 1)
  → Flatten  →  (B*S, 1280)
  → Reshape to (B, S, 1280)
  → Concat with env features (B, S, 2)  →  (B, S, 1282)
```

### Step 4: LSTM Temporal Modeling

```
Fused sequence (B, S, 1282)
  → LSTM(hidden=128, layers=1, batch_first=True)
  → Output (B, S, 128)
  → Take last timestep: (B, 128)
```

LSTM uses internal cell state `c_t` alongside hidden state `h_t`, providing more explicit long-term memory than GRU.

### Step 5: Regression

```
Last timestep (B, 128)
  → Linear(128 → 64) → ReLU → Dropout(0.2)
  → Linear(64 → 1)  →  RUL prediction (B, 1)
```

## Loss Function

**L1Loss (MAE)** is used because:
- Directly interpretable as "hours of error"
- Less sensitive to outliers than MSE
- Appropriate for regression tasks where the error magnitude matters linearly

## Optimizer

**Adam** with learning rate `1e-4`:
- Adaptive learning rates per parameter
- Well-suited for fine-tuning pretrained backbones
- Default betas: (0.9, 0.999)

## Training Loop

```
for epoch in range(10):
    train_one_epoch()
    validate()
    if val_loss improved:
        save_checkpoint()

final_test_evaluation()
save_metrics()
```

The best checkpoint is selected based on **validation loss** (not training loss), preventing overfitting.

## Prediction (Single Image)

For single-image inference:
1. Load checkpoint
2. Preprocess image (resize 224×224, ImageNet normalize)
3. Wrap into pseudo-sequence: `(1, 1, 3, 224, 224)`
4. Normalize temp/humidity: temp/30, humidity/100
5. Forward pass → predicted RUL in hours

**Note:** Single-image inference uses seq_len=1, which means the temporal model has limited context. For best results, use sequences of 5 consecutive frames from the same fruit.
