from pathlib import Path
from datetime import datetime
import json
import random
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm.auto import tqdm


def find_project_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for path in [start, *start.parents]:
        if (path / 'README.md').exists() and (path / 'data').exists():
            return path
    raise FileNotFoundError('Could not find project root containing README.md and data/.')


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)
PROJECT_ROOT = find_project_root()
PROJECT_ROOT

MODEL_ID = 'model_A'
BACKBONE_NAME = 'efficientnet_b0'
USE_PRETRAINED = True
BATCH_SIZE = 32
NUM_WORKERS = 0
IMAGE_SIZE = 224

# Temporary mock path. Replace these when stage3 preprocessing produces final data.
DATA_DIR = PROJECT_ROOT / 'data' / '_mock_data'
LABELS_CSV = DATA_DIR / 'ruf_labels.csv'

FEATURE_DIR = PROJECT_ROOT / 'data' / '04_feature' / MODEL_ID / 'efficientnet'
REPORT_DIR = PROJECT_ROOT / 'output' / 'reports' / 'training'
FEATURE_FILE = FEATURE_DIR / f'{MODEL_ID}_{BACKBONE_NAME}_features.npz'
MANIFEST_FILE = FEATURE_DIR / f'{MODEL_ID}_{BACKBONE_NAME}_manifest.csv'
METADATA_FILE = FEATURE_DIR / f'{MODEL_ID}_{BACKBONE_NAME}_metadata.json'

for folder in [FEATURE_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print(f'PROJECT_ROOT: {PROJECT_ROOT}')
print(f'DATA_DIR: {DATA_DIR}')
print(f'LABELS_CSV: {LABELS_CSV}')
print(f'FEATURE_FILE: {FEATURE_FILE}')

def parse_timestamp(filename):
    stem = Path(str(filename)).stem
    for fmt in ('%d-%m-%Y-%H-%M-%S', '%Y-%m-%d-%H-%M-%S'):
        try:
            return datetime.strptime(stem, fmt)
        except ValueError:
            pass
    return pd.NaT


labels = pd.read_csv(LABELS_CSV)
target_col = next((col for col in ['RUL', 'RUF', 'rul', 'ruf'] if col in labels.columns), None)
if target_col is None:
    raise ValueError(f'Expected one target column from RUL/RUF/rul/ruf, found {list(labels.columns)}')
if 'filename' not in labels.columns:
    raise ValueError(f'Expected filename column, found {list(labels.columns)}')

labels = labels.rename(columns={target_col: 'target'}).copy()
labels['filename'] = labels['filename'].astype(str)
labels['target'] = labels['target'].astype(np.float32)
labels['image_path'] = labels['filename'].map(lambda name: DATA_DIR / name)
labels['timestamp'] = labels['filename'].map(parse_timestamp)

missing = labels[~labels['image_path'].map(lambda path: path.exists())]
if len(missing):
    raise FileNotFoundError(f'{len(missing)} labeled images are missing. First missing file: {missing.iloc[0].image_path}')

labels = labels.sort_values(['timestamp', 'filename']).reset_index(drop=True)
print(f'Rows: {len(labels)}')
print(labels[['filename', 'target', 'timestamp']].head())
labels['target'].describe()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fallback_preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

weights = EfficientNet_B0_Weights.DEFAULT if USE_PRETRAINED else None
try:
    preprocess = weights.transforms() if weights is not None else fallback_preprocess
    backbone = efficientnet_b0(weights=weights)
except Exception as exc:
    print(f'Could not load pretrained weights: {exc}')
    print('Falling back to randomly initialized EfficientNet-B0. Set USE_PRETRAINED=False to silence this path.')
    preprocess = fallback_preprocess
    backbone = efficientnet_b0(weights=None)

feature_extractor = nn.Sequential(backbone.features, backbone.avgpool, nn.Flatten()).to(device)
feature_extractor.eval()
feature_dim = int(backbone.classifier[1].in_features)

print(f'Device: {device}')
print(f'Feature dim: {feature_dim}')

class StrawberryImageDataset(Dataset):
    def __init__(self, frame, transform):
        self.paths = frame['image_path'].tolist()
        self.names = frame['filename'].astype(str).tolist()
        self.targets = frame['target'].astype(np.float32).to_numpy()
        self.timestamps = frame['timestamp'].astype(str).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as image:
            image = image.convert('RGB')
        return self.transform(image), self.names[idx], self.targets[idx], self.timestamps[idx]


dataset = StrawberryImageDataset(labels, preprocess)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

len(dataset), len(loader)

all_features = []
all_targets = []
all_filenames = []
all_timestamps = []
started = time.perf_counter()

with torch.inference_mode():
    for images, names, targets, timestamps in tqdm(loader, desc='Extracting EfficientNet features'):
        images = images.to(device, non_blocking=True)
        batch_features = feature_extractor(images).cpu().numpy().astype(np.float32)
        all_features.append(batch_features)
        all_targets.extend(targets.numpy().astype(np.float32).tolist())
        all_filenames.extend(list(names))
        all_timestamps.extend(list(timestamps))

features = np.concatenate(all_features, axis=0).astype(np.float32)
targets = np.asarray(all_targets, dtype=np.float32)
elapsed_seconds = time.perf_counter() - started

np.savez_compressed(
    FEATURE_FILE,
    features=features,
    targets=targets,
    filenames=np.asarray(all_filenames),
    timestamps=np.asarray(all_timestamps),
)

manifest = labels[['filename', 'target', 'timestamp']].copy()
manifest['feature_row'] = np.arange(len(manifest), dtype=np.int32)
manifest['feature_file'] = str(FEATURE_FILE.relative_to(PROJECT_ROOT))
manifest.to_csv(MANIFEST_FILE, index=False)

metadata = {
    'model_id': MODEL_ID,
    'backbone': BACKBONE_NAME,
    'use_pretrained': USE_PRETRAINED,
    'data_dir': str(DATA_DIR.relative_to(PROJECT_ROOT)),
    'labels_csv': str(LABELS_CSV.relative_to(PROJECT_ROOT)),
    'feature_file': str(FEATURE_FILE.relative_to(PROJECT_ROOT)),
    'manifest_file': str(MANIFEST_FILE.relative_to(PROJECT_ROOT)),
    'num_frames': int(len(features)),
    'feature_dim': int(features.shape[1]),
    'elapsed_seconds': float(elapsed_seconds),
    'device': str(device),
}
METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

print(f'Saved features: {FEATURE_FILE}')
print(f'Saved manifest: {MANIFEST_FILE}')
print(f'Shape: {features.shape}')
print(f'Elapsed seconds: {elapsed_seconds:.2f}')
