"""
Training script for Model D: MobileNetV2 + CBAM + GRU

Saves:
  - best_model.pth  → models/model_D/
  - training_history.csv → data/model_D_outputs/
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from dataset import StrawberrySequenceDataset
from model import StrawberryRULModelD


def train():
    # ---- 1. Setup paths ----
    project_root = Path(__file__).resolve().parents[3]
    train_dir = project_root / "data" / "03_split" / "train"
    val_dir = project_root / "data" / "03_split" / "val"
    test_dir = project_root / "data" / "03_split" / "test"

    models_dir = project_root / "models" / "model_D"
    models_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = project_root / "data" / "model_D_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ---- 2. Hyperparameters ----
    batch_size = 4
    num_epochs = 10
    learning_rate = 1e-4
    seq_len = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Hyperparameters: batch_size={batch_size}, epochs={num_epochs}, lr={learning_rate}, seq_len={seq_len}")

    # ---- 3. DataLoaders ----
    print("\nLoading datasets...")
    train_dataset = StrawberrySequenceDataset(train_dir, seq_len=seq_len)
    val_dataset = StrawberrySequenceDataset(val_dir, seq_len=seq_len)
    test_dataset = StrawberrySequenceDataset(test_dir, seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences:   {len(val_dataset)}")
    print(f"Test sequences:  {len(test_dataset)}")

    # ---- 4. Model, Loss, Optimizer ----
    model = StrawberryRULModelD().to(device)
    criterion = nn.L1Loss()  # MAE — interpretable in hours
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel D: MobileNetV2 + CBAM + GRU")
    print(f"  Total params:    {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ---- 5. Training Loop ----
    best_val_loss = float('inf')
    history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")

        for images, envs, ruls in tqdm(train_loader, desc="Training"):
            images = images.to(device)
            envs = envs.to(device)
            ruls = ruls.to(device)

            optimizer.zero_grad()
            outputs = model(images, envs)
            loss = criterion(outputs, ruls)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, envs, ruls in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                envs = envs.to(device)
                ruls = ruls.to(device)

                outputs = model(images, envs)
                loss = criterion(outputs, ruls)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_dataset)

        print(f"Train Loss (MAE): {train_loss:.4f}  |  Val Loss (MAE): {val_loss:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), models_dir / "best_model.pth")
            print(f"  >> Saved new best model (val_loss={val_loss:.4f})")

    # ---- 6. Final Test Evaluation ----
    print(f"\n{'='*50}")
    print("Final Test Evaluation (best model)")
    print(f"{'='*50}")

    model.load_state_dict(torch.load(models_dir / "best_model.pth", map_location=device))
    model.eval()

    test_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, envs, ruls in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            envs = envs.to(device)
            ruls = ruls.to(device)

            outputs = model(images, envs)
            loss = criterion(outputs, ruls)
            test_loss += loss.item() * images.size(0)

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(ruls.cpu().numpy().flatten())

    test_loss /= len(test_dataset)

    # Additional metrics
    preds_tensor = torch.tensor(all_preds)
    targets_tensor = torch.tensor(all_targets)

    rmse = torch.sqrt(torch.mean((preds_tensor - targets_tensor) ** 2)).item()

    # MAPE — avoid division by zero
    nonzero_mask = targets_tensor != 0
    mape = (torch.abs((preds_tensor[nonzero_mask] - targets_tensor[nonzero_mask]) / targets_tensor[nonzero_mask]).mean() * 100).item() if nonzero_mask.sum() > 0 else float('nan')

    # R²
    ss_res = torch.sum((targets_tensor - preds_tensor) ** 2).item()
    ss_tot = torch.sum((targets_tensor - targets_tensor.mean()) ** 2).item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')

    print(f"\nTest Results:")
    print(f"  MAE:   {test_loss:.4f} hours")
    print(f"  RMSE:  {rmse:.4f} hours")
    print(f"  MAPE:  {mape:.2f}%")
    print(f"  R²:    {r2:.4f}")

    # Save test predictions
    test_preds_df = pd.DataFrame({
        "predicted_rul": all_preds,
        "actual_rul": all_targets,
    })
    test_preds_df.to_csv(outputs_dir / "test_predictions.csv", index=False)

    # Save metrics
    import json
    metrics = {
        "model": "Model_D_MobileNetV2_CBAM_GRU",
        "mae": test_loss,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "train_sequences": len(train_dataset),
        "val_sequences": len(val_dataset),
        "test_sequences": len(test_dataset),
    }
    with open(outputs_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(outputs_dir / "training_history.csv", index=False)

    print(f"\nTraining completed. Outputs saved to {outputs_dir}/")
    print(f"  - training_history.csv")
    print(f"  - test_predictions.csv")
    print(f"  - metrics.json")


if __name__ == "__main__":
    train()
