# Detailed Training and Prediction Process (Model A)

This document provides a deep dive into the inner workings of Model A (EfficientNet + GRU), detailing how data flows from the dataset into the network, and how the model makes predictions.

## 1. Data Preparation and Flow

Before reaching the model, the data undergoes specific formatting to accommodate the temporal (time-series) nature of the problem:

* **Time-Series Windows:** The `StrawberrySequenceDataset` groups data into rolling windows. By default, `seq_len=5`. This means for a given timestamp, the model looks at the current image and the 4 preceding images to understand the rate of decay.
* **Image Preprocessing:** Images are resized to `224x224`, converted to tensors, and normalized using standard ImageNet statistics (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
* **Environmental Features:** Temperature and humidity are roughly normalized (Temperature divided by `30.0`, Humidity divided by `100.0`) to keep feature values within a stable range for the neural network.
* **Target (Label):** The target variable is the Remaining Useful Life (RUL) in hours, corresponding to the *last* frame in the sequence window.

---

## 2. Training Process

When running `train.py`, the training loop executes as follows:

### Architecture Data Flow (Forward Pass)
1. **Input Tensors:** The model receives two main inputs:
   * `images_seq`: Shape `(batch_size, seq_len, 3, 224, 224)`
   * `env_seq`: Shape `(batch_size, seq_len, 2)`
2. **Spatial Feature Extraction (EfficientNet-B0):**
   * The `images_seq` is reshaped into a flat batch of images: `(batch_size * seq_len, 3, 224, 224)`.
   * It is passed through the pre-trained EfficientNet-B0 backbone (with the classifier head removed).
   * **Output:** A feature vector of size 1280 per image. The tensor is reshaped back to sequence format: `(batch_size, seq_len, 1280)`.
3. **Feature Fusion:**
   * The visual features (1280) are concatenated with the environmental features (2) at each time step.
   * **Output:** A combined feature sequence of shape `(batch_size, seq_len, 1282)`.
4. **Temporal Processing (GRU):**
   * The combined sequence is fed into a Gated Recurrent Unit (GRU) with a hidden size of `128`.
   * The GRU processes the sequential dependencies of the strawberry rotting over time.
5. **Prediction (Regression Head):**
   * Only the output of the *last time step* from the GRU `(batch_size, 128)` is utilized, as we are predicting the RUL for the current end-of-window state.
   * This vector passes through a Multi-Layer Perceptron (Linear `128->64` -> `ReLU` -> `Dropout` -> Linear `64->1`).
   * **Output:** A single predicted RUL value per sequence in the batch.

### Loss and Optimization
* **Loss Function:** `L1Loss` (Mean Absolute Error) is used. MAE is highly interpretable because an error of `2.5` means the model's prediction is off by 2.5 hours on average.
* **Optimizer:** Adam optimizer with a learning rate of `1e-4`.
* **Model Checkpointing:** At the end of every epoch, validation loss is calculated. If the validation loss improves, the model weights are saved to `models/model_A/best_model.pth`.

---

## 3. Prediction Process

When running `predict.py` to evaluate a single new image:

1. **Initialization:** The script loads the `best_model.pth` checkpoint into the `StrawberryRULModel` architecture and sets the model to evaluation mode (`model.eval()`).
2. **Pseudo-Sequencing:** Since the model expects a sequence but we are only providing a single image for inference, the script wraps the single preprocessed image into a sequence of length 1: `(1, 1, 3, 224, 224)`.
3. **Environmental Normalization:** The input temperature and humidity are normalized using the same scale as training (`/30.0` and `/100.0`).
4. **Inference:** The data is passed through the network without computing gradients (`torch.no_grad()`). The final output is returned as the predicted RUL in hours.
