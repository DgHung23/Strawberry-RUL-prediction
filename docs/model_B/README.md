# Model B: MobileNetV2 + LSTM

Welcome to the documentation for **Model B**, the second iteration of our deep learning model designed to predict the Remaining Useful Life (RUL) of strawberries! 🍓

## 🧠 What is Model B?

To allow for architectural comparison, Model B replaces the components from Model A with alternatives focused on efficiency and sequence processing:

1. **MobileNetV2 (Spatial):** This model acts as our visual feature extractor. MobileNet is specifically designed to be lightweight and fast, making it highly suitable for mobile or edge devices while still maintaining strong accuracy. It extracts a 1,280-dimensional feature map from each image.
2. **LSTM (Temporal):** Long Short-Term Memory (LSTM) layers are utilized instead of GRUs. LSTMs are slightly more complex than GRUs and maintain an internal "cell state", which often helps in capturing longer temporal dependencies as the strawberry decays.

Like Model A, a **Regression Head** takes the final temporal state and outputs a single continuous value: the estimated Remaining Useful Life in hours.

## 📂 Directory Structure

Here is how Model B interacts with the project folder:

- **Source Code (`src/stage4_training/model_B/`)**: Contains the model architecture, dataset loader, and training/prediction scripts.
- **Data Input (`data/03_split/`)**: Identical to Model A, reading processed images and CSV labels from the train/val/test splits.
- **Checkpoints (`models/model_B/`)**: The best model weights are saved here as `best_model.pth`.
- **Outputs (`data/model_B_outputs/`)**: Training history and evaluation metrics.

## 🚀 How to Run Training

Running Model B is just as simple. In your terminal, run:

```bash
python src/stage4_training/model_B/train.py
```

### 🔮 How to Predict with Model B

You can predict the RUL of a single strawberry image using the prediction script. Make sure to provide the path to your image, temperature, and humidity:

```bash
python src/stage4_training/model_B/predict.py --image path/to/image.png --temp 22.5 --humidity 60.5
```

## ⚙️ Customizing Hyperparameters

If you want to adjust the training settings, open `src/stage4_training/model_B/train.py` and modify the **Hyperparameters** section:

```python
batch_size = 4        
num_epochs = 10       
learning_rate = 1e-4  
seq_len = 5           
```

Enjoy training your Model B architecture! 🚀
