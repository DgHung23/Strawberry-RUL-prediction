from pathlib import Path
import json
import os
import random
import runpy
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler



def find_project_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for path in [start, *start.parents]:
        if (path / 'C:\\fluttersrc\\Strawberry-RUL-prediction\\README.md').exists() and (path / 'C:\\fluttersrc\\Strawberry-RUL-prediction\\data').exists():
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
TEMPORAL_MODEL = 'gru'

SEQUENCE_LENGTH = 8
STRIDE = 1
SEQUENCE_ID_COLUMN = None
SPLIT_RATIOS = (0.70, 0.15, 0.15)

BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.20
PATIENCE = 8
GRAD_CLIP_NORM = 1.0
DEVICE_NAME = os.environ.get('MODEL_A_DEVICE', 'cpu').lower()

FEATURE_DIR = PROJECT_ROOT / 'data' / '04_feature' / MODEL_ID / 'efficientnet'
FEATURE_FILE = FEATURE_DIR / f'{MODEL_ID}_{BACKBONE_NAME}_features.npz'
MANIFEST_FILE = FEATURE_DIR / f'{MODEL_ID}_{BACKBONE_NAME}_manifest.csv'

MODEL_DIR = PROJECT_ROOT / 'models' / MODEL_ID
GRAPH_DIR = PROJECT_ROOT / 'output' / 'graphs' / 'training'
REPORT_DIR = PROJECT_ROOT / 'output' / 'reports' / 'training'
RESULT_DIR = PROJECT_ROOT / 'output' / 'results' / 'ori'

BEST_MODEL_FILE = MODEL_DIR / f'{MODEL_ID}_{BACKBONE_NAME}_{TEMPORAL_MODEL}_best.pt'
LAST_MODEL_FILE = MODEL_DIR / f'{MODEL_ID}_{BACKBONE_NAME}_{TEMPORAL_MODEL}_last.pt'
METRICS_FILE = RESULT_DIR / f'{MODEL_ID}_{BACKBONE_NAME}_{TEMPORAL_MODEL}_metrics.json'
PREDICTIONS_FILE = RESULT_DIR / f'{MODEL_ID}_{BACKBONE_NAME}_{TEMPORAL_MODEL}_test_predictions.csv'
TRAINING_CURVE_FILE = GRAPH_DIR / f'{MODEL_ID}_{BACKBONE_NAME}_{TEMPORAL_MODEL}_training_curves.png'
REPORT_FILE = REPORT_DIR / f'{MODEL_ID}_{BACKBONE_NAME}_{TEMPORAL_MODEL}_training_report.md'

for folder in [MODEL_DIR, GRAPH_DIR, REPORT_DIR, RESULT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print(f'FEATURE_FILE: {FEATURE_FILE}')
print(f'MODEL_DIR: {MODEL_DIR}')

if not FEATURE_FILE.exists() or not MANIFEST_FILE.exists():
    extractor_script = PROJECT_ROOT / 'models' / MODEL_ID / 'notebooks' / '01_model_A_efficientnet_feature_extraction.py'
    if extractor_script.exists():
        print(f'Feature cache not found. Running feature extraction: {extractor_script}')
        runpy.run_path(str(extractor_script), run_name='__main__')

if not FEATURE_FILE.exists() or not MANIFEST_FILE.exists():
    raise FileNotFoundError(
        'Feature cache not found after extraction. '
        'Run models/model_A/notebooks/01_model_A_efficientnet_feature_extraction.py first.'
    )

bundle = np.load(FEATURE_FILE, allow_pickle=True)
features = bundle['features'].astype(np.float32)
targets = bundle['targets'].astype(np.float32)
filenames = bundle['filenames'].astype(str)

manifest = pd.read_csv(MANIFEST_FILE)
manifest['feature_row'] = manifest['feature_row'].astype(int)
if 'timestamp' in manifest.columns:
    manifest['timestamp'] = pd.to_datetime(manifest['timestamp'], errors='coerce')
manifest = manifest.sort_values('feature_row').reset_index(drop=True)

if len(manifest) != len(features):
    raise ValueError(f'Manifest rows ({len(manifest)}) do not match features ({len(features)}).')

print(f'Features: {features.shape}')
print(f'Targets: {targets.shape}')
manifest.head()

def build_sequences(features, targets, manifest, sequence_length, stride, sequence_col=None):
    X, y, end_rows, sequence_ids = [], [], [], []

    if sequence_col and sequence_col in manifest.columns:
        grouped = manifest.groupby(sequence_col, sort=False)
    else:
        working = manifest.copy()
        working['_sequence_id'] = 'mock_sequence'
        grouped = working.groupby('_sequence_id', sort=False)

    for sequence_id, group in grouped:
        sort_cols = ['feature_row']
        if 'timestamp' in group.columns and group['timestamp'].notna().any():
            sort_cols = ['timestamp', 'filename']
        group = group.sort_values(sort_cols)
        rows = group['feature_row'].to_numpy(dtype=np.int64)

        if len(rows) < sequence_length:
            continue

        for start in range(0, len(rows) - sequence_length + 1, stride):
            window_rows = rows[start:start + sequence_length]
            X.append(features[window_rows])
            y.append(targets[window_rows[-1]])
            end_rows.append(int(window_rows[-1]))
            sequence_ids.append(str(sequence_id))

    if not X:
        raise ValueError('No sequences were built. Lower SEQUENCE_LENGTH or check sequence grouping.')

    return (
        np.stack(X).astype(np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(end_rows, dtype=np.int64),
        np.asarray(sequence_ids),
    )


X, y, sequence_end_rows, sequence_ids = build_sequences(
    features,
    targets,
    manifest,
    sequence_length=SEQUENCE_LENGTH,
    stride=STRIDE,
    sequence_col=SEQUENCE_ID_COLUMN,
)

print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')
print(f'Target range: {y.min():.4f} to {y.max():.4f}')

def make_time_split(n_samples, ratios):
    if n_samples < 3:
        raise ValueError('Need at least 3 sequences for train/validation/test split.')
    train_end = max(1, int(n_samples * ratios[0]))
    val_end = max(train_end + 1, int(n_samples * (ratios[0] + ratios[1])))
    val_end = min(val_end, n_samples - 1)
    indices = np.arange(n_samples)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


train_idx, val_idx, test_idx = make_time_split(len(X), SPLIT_RATIOS)
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

n_steps = X_train.shape[1]
feature_dim = X_train.shape[2]

feature_scaler = StandardScaler()
target_scaler = StandardScaler()

X_train_scaled = feature_scaler.fit_transform(X_train.reshape(-1, feature_dim)).reshape(X_train.shape).astype(np.float32)
X_val_scaled = feature_scaler.transform(X_val.reshape(-1, feature_dim)).reshape(X_val.shape).astype(np.float32)
X_test_scaled = feature_scaler.transform(X_test.reshape(-1, feature_dim)).reshape(X_test.shape).astype(np.float32)

y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1).astype(np.float32)
y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).reshape(-1).astype(np.float32)
y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).reshape(-1).astype(np.float32)

split_summary = {
    'train': int(len(train_idx)),
    'validation': int(len(val_idx)),
    'test': int(len(test_idx)),
}
split_summary

class FeatureSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_ds = FeatureSequenceDataset(X_train_scaled, y_train_scaled)
val_ds = FeatureSequenceDataset(X_val_scaled, y_val_scaled)
test_ds = FeatureSequenceDataset(X_test_scaled, y_test_scaled)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
train_eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

len(train_loader), len(val_loader), len(test_loader)

class EfficientNetGRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        _, hidden = self.gru(x)
        last_hidden = hidden[-1]
        return self.head(last_hidden)


if DEVICE_NAME == 'cuda' and torch.cuda.is_available():
    device = torch.device('cuda')
elif DEVICE_NAME == 'cuda':
    print('MODEL_A_DEVICE=cuda was requested, but CUDA is not available. Falling back to CPU.')
    device = torch.device('cpu')
else:
    device = torch.device('cpu')
model_config = {
    'input_size': int(feature_dim),
    'hidden_size': HIDDEN_SIZE,
    'num_layers': NUM_LAYERS,
    'dropout': DROPOUT,
    'sequence_length': SEQUENCE_LENGTH,
}
model = EfficientNetGRURegressor(**{k: model_config[k] for k in ['input_size', 'hidden_size', 'num_layers', 'dropout']}).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

print(model)
print(f'Device: {device}')

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        with torch.set_grad_enabled(is_train):
            preds = model(batch_x)
            loss = criterion(preds, batch_y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        total_loss += float(loss.item()) * len(batch_x)
        total_count += len(batch_x)

    return total_loss / max(total_count, 1)


def checkpoint_payload(epoch, best_val_loss):
    return {
        'model_id': MODEL_ID,
        'backbone': BACKBONE_NAME,
        'temporal_model': TEMPORAL_MODEL,
        'epoch': int(epoch),
        'best_val_loss': float(best_val_loss),
        'model_config': model_config,
        'model_state_dict': model.state_dict(),
        'feature_scaler_mean': feature_scaler.mean_.astype(np.float32),
        'feature_scaler_scale': feature_scaler.scale_.astype(np.float32),
        'target_scaler_mean': target_scaler.mean_.astype(np.float32),
        'target_scaler_scale': target_scaler.scale_.astype(np.float32),
    }


history = []
best_val_loss = float('inf')
best_epoch = 0
bad_epochs = 0
started = time.perf_counter()

for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(model, train_loader, criterion, optimizer=optimizer)
    val_loss = run_epoch(model, val_loader, criterion, optimizer=None)
    history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

    improved = val_loss < best_val_loss
    if improved:
        best_val_loss = val_loss
        best_epoch = epoch
        bad_epochs = 0
        torch.save(checkpoint_payload(epoch, best_val_loss), BEST_MODEL_FILE)
    else:
        bad_epochs += 1

    print(f'Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | best_epoch={best_epoch}')
    if bad_epochs >= PATIENCE:
        print(f'Early stopping at epoch {epoch}.')
        break

training_seconds = time.perf_counter() - started
torch.save(checkpoint_payload(history[-1]['epoch'], best_val_loss), LAST_MODEL_FILE)

history_df = pd.DataFrame(history)
print(f'Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}')
print(f'Training seconds: {training_seconds:.2f}')
history_df.tail()

def load_checkpoint(path):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def predict(loader):
    model.eval()
    preds, actuals = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_preds = model(batch_x).cpu().numpy().reshape(-1)
            preds.append(batch_preds)
            actuals.append(batch_y.numpy().reshape(-1))
    elapsed = time.perf_counter() - started
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(-1)
    actuals = target_scaler.inverse_transform(actuals.reshape(-1, 1)).reshape(-1)
    return actuals, preds, elapsed


def regression_metrics(y_true, y_pred):
    denom = np.maximum(np.abs(y_true), 1e-8)
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mape': float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0),
        'r2': float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
    }


checkpoint = load_checkpoint(BEST_MODEL_FILE)
model.load_state_dict(checkpoint['model_state_dict'])

train_true, train_pred, train_infer_seconds = predict(train_eval_loader)
val_true, val_pred, val_infer_seconds = predict(val_loader)
test_true, test_pred, test_infer_seconds = predict(test_loader)

metrics = {
    'model_id': MODEL_ID,
    'cnn_backbone': BACKBONE_NAME,
    'temporal_model': TEMPORAL_MODEL,
    'sequence_length': SEQUENCE_LENGTH,
    'stride': STRIDE,
    'split': split_summary,
    'best_epoch': int(best_epoch),
    'training_seconds': float(training_seconds),
    'inference_seconds': {
        'train': float(train_infer_seconds),
        'validation': float(val_infer_seconds),
        'test': float(test_infer_seconds),
        'test_per_sequence': float(test_infer_seconds / max(len(test_ds), 1)),
    },
    'train': regression_metrics(train_true, train_pred),
    'validation': regression_metrics(val_true, val_pred),
    'test': regression_metrics(test_true, test_pred),
}

METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding='utf-8')

test_rows = sequence_end_rows[test_idx]
predictions = pd.DataFrame({
    'filename': filenames[test_rows],
    'y_true': test_true,
    'y_pred': test_pred,
    'absolute_error': np.abs(test_true - test_pred),
})
predictions.to_csv(PREDICTIONS_FILE, index=False)

print(json.dumps(metrics['test'], indent=2))
print(f'Saved metrics: {METRICS_FILE}')
print(f'Saved predictions: {PREDICTIONS_FILE}')
predictions.head()

plt.figure(figsize=(8, 4.5))
plt.plot(history_df['epoch'], history_df['train_loss'], label='Train loss')
plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation loss')
plt.axvline(best_epoch, color='gray', linestyle='--', linewidth=1, label='Best epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE loss on scaled RUF/RUL')
plt.title('Model A EfficientNet-B0 + GRU training')
plt.legend()
plt.tight_layout()
plt.savefig(TRAINING_CURVE_FILE, dpi=160)
plt.close()

report_lines = [
    '# Model A training report',
    '',
    f'- Backbone: {BACKBONE_NAME}',
    f'- Temporal model: {TEMPORAL_MODEL.upper()}',
    f'- Sequence length: {SEQUENCE_LENGTH}',
    f'- Train/validation/test sequences: {split_summary}',
    f'- Best epoch: {best_epoch}',
    f'- Training seconds: {training_seconds:.2f}',
    '',
    '## Test metrics',
    '',
    f"- MAE: {metrics['test']['mae']:.4f}",
    f"- RMSE: {metrics['test']['rmse']:.4f}",
    f"- MAPE: {metrics['test']['mape']:.4f}",
    f"- R2: {metrics['test']['r2']:.4f}" if metrics['test']['r2'] is not None else '- R2: n/a',
    '',
    f'- Best checkpoint: {BEST_MODEL_FILE.relative_to(PROJECT_ROOT)}',
    f'- Last checkpoint: {LAST_MODEL_FILE.relative_to(PROJECT_ROOT)}',
    f'- Metrics JSON: {METRICS_FILE.relative_to(PROJECT_ROOT)}',
    f'- Test predictions CSV: {PREDICTIONS_FILE.relative_to(PROJECT_ROOT)}',
    f'- Training curve: {TRAINING_CURVE_FILE.relative_to(PROJECT_ROOT)}',
]
REPORT_FILE.write_text('\n'.join(report_lines), encoding='utf-8')

print(f'Saved curve: {TRAINING_CURVE_FILE}')
print(f'Saved report: {REPORT_FILE}')
