# Model A: EfficientNet + GRU

Welcome to the documentation for **Model A**, a hybrid deep learning model designed to predict the Remaining Useful Life (RUL) of strawberries! 🍓

## 🧠 What is Model A?

Model A combines two powerful neural network architectures to understand both the spatial (visual) and temporal (time-based) aspects of a strawberry's decay:

1. **EfficientNet-B0 (Spatial):** This looks at a single image of a strawberry and extracts important visual features (like color changes, shriveling, or mold).
2. **GRU (Temporal):** Since a strawberry rots over time, we feed a sequence of consecutive images (along with temperature and humidity data) into a Gated Recurrent Unit (GRU). This allows the model to understand the *progression* of decay over time.

Finally, a **Regression Head** outputs a single number: the estimated Remaining Useful Life in hours.

## 📂 Directory Structure

Here is how Model A interacts with the project folder:

- **Source Code (`src/stage4_training/model_A/`)**: Contains the model architecture, dataset loader, and training script.
- **Data Input (`data/03_split/`)**: The script reads processed images and CSV labels from the train/val/test splits.
- **Checkpoints (`models/model_A/`)**: When the model improves during training, it saves its weights here as `best_model.pth`.
- **Outputs (`data/model_A_outputs/`)**: Training history and evaluation metrics are saved here.

## 🚀 How to Run Training

Running the model is simple. Open your terminal or command prompt, ensure you are in the root directory of the project, and run:

```bash
python src/stage4_training/model_A/train.py
```

### What happens when you run this?
1. The script will automatically scan your `data/03_split/train` and `val` folders.
2. It groups consecutive images of the same strawberry into windows (default is 5 frames).
3. It starts training! You will see a progress bar indicating the **Train Loss** and **Val Loss**. 
4. The loss metric used is **Mean Absolute Error (MAE)**. This means if your Val Loss is `12.5`, the model is currently miscalculating the Remaining Useful Life by roughly 12.5 hours on average.
5. Once complete, you can find the `best_model.pth` saved and ready for inference!

## ⚙️ Customizing Hyperparameters

If you want to tweak how the model trains (for example, batch size or learning rate), open `src/stage4_training/model_A/train.py` and modify the variables under **Hyperparameters**:

```python
# 2. Hyperparameters
batch_size = 4        # Number of sequences processed at once (lower this if you run out of GPU memory)
num_epochs = 10       # How many times the model sees the entire dataset
learning_rate = 1e-4  # How fast the model updates its weights
seq_len = 5           # How many consecutive frames the model looks at simultaneously
```

Enjoy training! 🚀
