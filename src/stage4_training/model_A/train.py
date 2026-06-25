import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from dataset import StrawberrySequenceDataset
from model import StrawberryRULModel

def train():
    # 1. Setup paths
    project_root = Path(__file__).resolve().parents[3]
    train_dir = project_root / "data" / "03_split" / "train"
    val_dir = project_root / "data" / "03_split" / "val"
    
    models_dir = project_root / "models" / "model_A"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    outputs_dir = project_root / "data" / "model_A_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 2. Hyperparameters
    batch_size = 4
    num_epochs = 10
    learning_rate = 1e-4
    seq_len = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. DataLoaders
    print("Loading datasets...")
    train_dataset = StrawberrySequenceDataset(train_dir, seq_len=seq_len)
    val_dataset = StrawberrySequenceDataset(val_dir, seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train sequences: {len(train_dataset)}, Val sequences: {len(val_dataset)}")

    # 4. Model, Loss, Optimizer
    model = StrawberryRULModel().to(device)
    criterion = nn.L1Loss() # MAE is good for RUL interpretation (hours)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 5. Training Loop
    best_val_loss = float('inf')
    history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
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
        
        # Validation
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
        
        print(f"Train Loss (MAE): {train_loss:.4f} | Val Loss (MAE): {val_loss:.4f}")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), models_dir / "best_model.pth")
            print("Saved new best model.")
            
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(outputs_dir / "training_history.csv", index=False)
    print(f"Training completed. History saved to {outputs_dir / 'training_history.csv'}")

if __name__ == "__main__":
    train()
