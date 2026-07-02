import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class StrawberrySequenceDataset(Dataset):
    def __init__(self, split_dir, seq_len=5, transform=None):
        """
        Args:
            split_dir (str or Path): Path to the split directory (e.g., data/03_split/train).
            seq_len (int): Number of consecutive frames in a sequence.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.split_dir = Path(split_dir)
        self.seq_len = seq_len

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        self.samples = []

        self._prepare_sequences()

    def _prepare_sequences(self):
        fruit_dirs = [d for d in self.split_dir.iterdir() if d.is_dir() and d.name.startswith("F")]

        for fruit_dir in fruit_dirs:
            labels_path = fruit_dir / "labels.csv"
            if not labels_path.exists():
                continue

            df = pd.read_csv(labels_path)
            # Ensure it's sorted by timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values(by="timestamp").reset_index(drop=True)

            # Create sequences of length seq_len
            for i in range(len(df) - self.seq_len + 1):
                window = df.iloc[i : i + self.seq_len]

                # Image paths (fix paths to point to local images/ folder)
                image_paths = [
                    fruit_dir / "images" / Path(p).name for p in window["image_path"].tolist()
                ]

                # Env features
                temps = window["temperature_c"].values
                humidities = window["humidity_pct"].values

                # Target: RUL of the last frame in the sequence
                rul = window.iloc[-1]["rul_hours"]

                self.samples.append({
                    "image_paths": image_paths,
                    "temps": temps,
                    "humidities": humidities,
                    "rul": rul
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        images = []
        for path in sample["image_paths"]:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            images.append(img)

        images_tensor = torch.stack(images) # shape: (seq_len, C, H, W)

        # Normalize env features roughly
        # Temp ~20-30 -> / 30, Humidity ~50-80 -> / 100
        env_features = torch.tensor(
            [[t / 30.0, h / 100.0] for t, h in zip(sample["temps"], sample["humidities"])],
            dtype=torch.float32
        ) # shape: (seq_len, 2)

        rul_tensor = torch.tensor([sample["rul"]], dtype=torch.float32) # shape: (1,)

        return images_tensor, env_features, rul_tensor

if __name__ == "__main__":
    # Test dataset
    project_root = Path(__file__).resolve().parents[3]
    test_dir = project_root / "data" / "03_split" / "train"
    if test_dir.exists():
        ds = StrawberrySequenceDataset(test_dir, seq_len=5)
        print(f"Total sequences in train: {len(ds)}")
        if len(ds) > 0:
            imgs, envs, rul = ds[0]
            print(f"Images shape: {imgs.shape}")
            print(f"Env shape: {envs.shape}")
            print(f"RUL shape: {rul.shape}")
