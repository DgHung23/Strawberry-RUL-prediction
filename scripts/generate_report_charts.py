"""
Generate diagnostic charts for the RUL Prediction Improvement Report.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from pathlib import Path

# Create output directory
out_dir = Path('output/reports/improvement_analysis')
out_dir.mkdir(parents=True, exist_ok=True)

# --- Load data ---
a_preds = pd.read_csv('data/model_A_outputs/test_predictions.csv')
b_preds = pd.read_csv('data/model_B_outputs/test_predictions.csv')
hist_a = pd.read_csv('data/model_A_outputs/training_history.csv')
hist_b = pd.read_csv('data/model_B_outputs/training_history.csv')

# ==========================================
# FIGURE 1: Predictions vs Actual (Model A + B)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: Predicted vs Actual Scatter
ax = axes[0]
errors_a = a_preds['predicted_rul'] - a_preds['actual_rul']
sc = ax.scatter(a_preds['actual_rul'], a_preds['predicted_rul'],
                c=errors_a, cmap='coolwarm', alpha=0.6, s=25, edgecolors='none', vmin=-120, vmax=120)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Error (hours)', fontsize=11)

max_val = max(a_preds['actual_rul'].max(), a_preds['predicted_rul'].max())
ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, alpha=0.8, label='Perfect Prediction')

ax.axvspan(-5, 10, alpha=0.15, color='red', label='EOL Zone')

ax.annotate('EOL FAILURE ZONE\nActual RUL=0, Predicted ~90h',
            xy=(5, 90), xytext=(80, 180),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

ax.annotate('Under-prediction at\nhigh RUL (~55h off)',
            xy=(234, 178), xytext=(185, 220),
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=2),
            fontsize=10, color='darkorange', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

ax.set_xlabel('Actual RUL (hours)', fontsize=13, fontweight='bold')
ax.set_ylabel('Predicted RUL (hours)', fontsize=13, fontweight='bold')
ax.set_title('Model A (EfficientNet-B0 + GRU): Predicted vs Actual', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 260)
ax.set_ylim(-5, 260)

# Right: Error Distribution by RUL range
ax = axes[1]
a_preds_copy = a_preds.copy()
a_preds_copy['rul_bin'] = pd.cut(a_preds_copy['actual_rul'],
    bins=[0, 24, 72, 120, 168, 260],
    labels=['0-24h\n(EOL)', '24-72h\n(Late)', '72-120h\n(Mid)', '120-168h\n(Early)', '168-260h\n(Fresh)'])
box_data = [(a_preds_copy[a_preds_copy['rul_bin']==b]['predicted_rul'].values -
             a_preds_copy[a_preds_copy['rul_bin']==b]['actual_rul'].values)
            for b in a_preds_copy['rul_bin'].cat.categories]
bp = ax.boxplot(box_data, tick_labels=a_preds_copy['rul_bin'].cat.categories, patch_artist=True)
colors = ['#ff4444', '#ff8800', '#ffcc00', '#88cc00', '#0088ff']
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_ylabel('Prediction Error (hours)', fontsize=13, fontweight='bold')
ax.set_title('Model A: Error Distribution by RUL Range', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.annotate('Worst errors:\n+-50-60h', xy=(4.5, 50), fontsize=10, color='red', fontweight='bold')
ax.annotate('High bias:\n~90h at EOL', xy=(1, 70), fontsize=10, color='darkred', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='pink', alpha=0.7))

plt.tight_layout(pad=2)
fig.savefig(out_dir / '01_pred_vs_actual.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Figure 1 saved')

# ==========================================
# FIGURE 2: Training Curves + Overfitting Analysis
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
ax.plot(hist_a['epoch'], hist_a['train_loss'], 'b-o', label='Model A - Train', markersize=8, linewidth=2)
ax.plot(hist_a['epoch'], hist_a['val_loss'], 'b--s', label='Model A - Val', markersize=8, linewidth=2)
ax.plot(hist_b['epoch'], hist_b['train_loss'], 'r-o', label='Model B - Train', markersize=8, linewidth=2)
ax.plot(hist_b['epoch'], hist_b['val_loss'], 'r--s', label='Model B - Val', markersize=8, linewidth=2)

ax.axvspan(3.5, 10.5, alpha=0.08, color='red')
ax.annotate('OVERFITTING ZONE\nVal loss INCREASES\nwhile train loss drops',
            xy=(6, 12), fontsize=11, color='red', fontweight='bold',
            ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))

ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax.set_ylabel('MAE Loss (hours)', fontsize=13, fontweight='bold')
ax.set_title('Training vs Validation Loss Curves', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[1]
epochs = hist_a['epoch']
gap_a = hist_a['val_loss'] - hist_a['train_loss']
gap_b = hist_b['val_loss'] - hist_b['train_loss']
ax.bar(epochs - 0.15, gap_a, 0.3, label='Model A', color='steelblue', alpha=0.8)
ax.bar(epochs + 0.15, gap_b, 0.3, label='Model B', color='coral', alpha=0.8)
ax.axhline(y=0, color='black', linewidth=1)
ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax.set_ylabel('Generalization Gap (Val - Train) [hours]', fontsize=13, fontweight='bold')
ax.set_title('Overfitting Magnitude: Gap Between Val and Train Loss', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.annotate('Gap grows >13h\nfor Model A\nat epoch 9', xy=(9, 13.4), fontsize=10, color='darkred', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout(pad=2)
fig.savefig(out_dir / '02_training_overfitting.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Figure 2 saved')

# ==========================================
# FIGURE 3: EOL Analysis
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
eol_a = a_preds[a_preds['actual_rul'] == 0].copy().sort_values('predicted_rul').reset_index(drop=True)
eol_b = b_preds[b_preds['actual_rul'] == 0].copy().sort_values('predicted_rul').reset_index(drop=True)

ax.scatter(range(len(eol_a)), eol_a['predicted_rul'], alpha=0.6, s=20, label='Model A', color='steelblue')
ax.scatter(range(len(eol_b)), eol_b['predicted_rul'], alpha=0.6, s=20, label='Model B', color='coral')
ax.axhline(y=1, color='green', linestyle='-', linewidth=2, alpha=0.7, label='Ideal prediction (RUL=0)')
ax.axhspan(-2, 5, alpha=0.1, color='green', label='Acceptable zone (+-5h)')
ax.axhline(y=eol_a['predicted_rul'].mean(), color='steelblue', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Model A mean at EOL: {eol_a["predicted_rul"].mean():.0f}h')
ax.axhline(y=eol_b['predicted_rul'].mean(), color='coral', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Model B mean at EOL: {eol_b["predicted_rul"].mean():.0f}h')
ax.set_xlabel('Sample Index (sorted by predicted RUL)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted RUL (hours)', fontsize=12, fontweight='bold')
ax.set_title('CRITICAL: Model Predictions When Fruit is at EOL (Actual RUL = 0)', fontsize=13, fontweight='bold', color='darkred')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_ylim(-10, 180)

stats_text = (
    f'Model A EOL stats:\n'
    f'  Mean: {eol_a["predicted_rul"].mean():.1f}h\n'
    f'  Min: {eol_a["predicted_rul"].min():.1f}h\n'
    f'  Max: {eol_a["predicted_rul"].max():.1f}h\n'
    f'  Std: {eol_a["predicted_rul"].std():.1f}h\n\n'
    f'Model B EOL stats:\n'
    f'  Mean: {eol_b["predicted_rul"].mean():.1f}h\n'
    f'  Min: {eol_b["predicted_rul"].min():.1f}h\n'
    f'  Max: {eol_b["predicted_rul"].max():.1f}h\n'
    f'  Std: {eol_b["predicted_rul"].std():.1f}h'
)
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment='top',
        horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), family='monospace')

ax = axes[1]
transitions = [0]
for i in range(1, len(a_preds)):
    if a_preds['actual_rul'].iloc[i] > a_preds['actual_rul'].iloc[i-1]:
        transitions.append(i)

colors_fruits = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
for j, start in enumerate(transitions[:4]):
    end = transitions[j+1] if j+1 < len(transitions) else len(a_preds)
    chunk = a_preds.iloc[start:end].copy()
    ax.plot(chunk['actual_rul'], chunk['predicted_rul'], '-', color=colors_fruits[j], alpha=0.5, linewidth=1.5,
            label=f'Fruit {j+1} predicted')

max_val2 = a_preds['actual_rul'].max()
ax.plot([0, max_val2], [0, max_val2], 'k--', linewidth=2, label='Perfect prediction')
ax.axvspan(-5, 5, alpha=0.12, color='red')
ax.annotate('RUL=0 zone\n(Model still predicts\n50-110h!)', xy=(2, 60),
            fontsize=11, color='darkred', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))
ax.set_xlabel('Actual RUL (hours)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted RUL (hours)', fontsize=12, fontweight='bold')
ax.set_title('Predicted RUL Trajectory vs Actual (Model A)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout(pad=2)
fig.savefig(out_dir / '03_eol_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Figure 3 saved')

# ==========================================
# FIGURE 4: Error Analysis + Data Statistics
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

ax = axes[0, 0]
errors_model_a = np.abs(a_preds['predicted_rul'] - a_preds['actual_rul'])
errors_model_b = np.abs(b_preds['predicted_rul'] - b_preds['actual_rul'])
ax.hist(errors_model_a, bins=40, alpha=0.6, label=f'Model A (MAE={errors_model_a.mean():.1f}h)', color='steelblue', density=True)
ax.hist(errors_model_b, bins=40, alpha=0.6, label=f'Model B (MAE={errors_model_b.mean():.1f}h)', color='coral', density=True)
ax.axvline(x=errors_model_a.mean(), color='steelblue', linestyle='--', linewidth=2)
ax.axvline(x=errors_model_b.mean(), color='coral', linestyle='--', linewidth=2)
ax.set_xlabel('Absolute Error (hours)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Distribution of Absolute Prediction Errors', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
errors_a_signed = a_preds['predicted_rul'] - a_preds['actual_rul']
errors_b_signed = b_preds['predicted_rul'] - b_preds['actual_rul']
ax.scatter(a_preds['actual_rul'], errors_a_signed, alpha=0.3, s=15, label='Model A', color='steelblue')
ax.scatter(b_preds['actual_rul'], errors_b_signed, alpha=0.3, s=15, label='Model B', color='coral')

df_a_trend = pd.DataFrame({'actual': a_preds['actual_rul'], 'error': errors_a_signed}).sort_values('actual')
df_a_trend['trend'] = df_a_trend['error'].rolling(50, center=True).mean()
ax.plot(df_a_trend['actual'], df_a_trend['trend'], 'b-', linewidth=3, alpha=0.9, label='Model A trend')
df_b_trend = pd.DataFrame({'actual': b_preds['actual_rul'], 'error': errors_b_signed}).sort_values('actual')
df_b_trend['trend'] = df_b_trend['error'].rolling(50, center=True).mean()
ax.plot(df_b_trend['actual'], df_b_trend['trend'], 'r-', linewidth=3, alpha=0.9, label='Model B trend')
ax.axhline(y=0, color='black', linewidth=1)
ax.set_xlabel('Actual RUL (hours)', fontsize=12)
ax.set_ylabel('Prediction Error (hours)', fontsize=12)
ax.set_title('Error Trend vs Actual RUL (Systematic Bias)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.annotate('Over-prediction', xy=(200, 50), fontsize=10, color='darkred')
ax.annotate('Massive over-prediction\nat EOL (RUL=0)', xy=(30, 70), fontsize=10, color='darkred')

ax = axes[1, 0]
metrics = ['MAE (h)', 'RMSE (h)', 'R-squared']
model_a_vals = [43.14, 49.87, 0.6075]
model_b_vals = [53.77, 61.54, 0.4026]
x = np.arange(len(metrics))
width = 0.35
bars1 = ax.bar(x - width/2, model_a_vals, width, label='Model A', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, model_b_vals, width, label='Model B', color='coral', alpha=0.8)
for bar, val in zip(bars1, model_a_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.2f}', ha='center', fontweight='bold', fontsize=10)
for bar, val in zip(bars2, model_b_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.2f}', ha='center', fontweight='bold', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_title('Model Performance Metrics Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 1]
data_categories = ['Train Fruits', 'Val Fruits', 'Test Fruits', 'Total Fruits', 'Train Sequences']
data_values = [4, 1, 1, 6, 2614]
colors_data = ['#2196F3', '#FF9800', '#F44336', '#4CAF50', '#9C27B0']
bars = ax.barh(data_categories, data_values, color=colors_data, alpha=0.8)
for bar, val in zip(bars, data_values):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, str(val), fontweight='bold', fontsize=11)
ax.set_title('Dataset Statistics (Very Limited Data)', fontsize=13, fontweight='bold', color='darkred')
ax.set_xlim(0, 3500)
ax.grid(True, alpha=0.3, axis='x')
ax.text(0.5, -0.25, 'WARNING: Only 4 fruits for training 4.76M parameters -> EXTREME overfitting risk',
        transform=ax.transAxes, fontsize=11, color='darkred', fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.tight_layout(pad=2.5)
fig.savefig(out_dir / '04_error_data_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Figure 4 saved')

# ==========================================
# FIGURE 5: Architecture Limitations Diagram
# ==========================================
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis('off')

layers = [
    (1.5, 7.0, 2.0, 0.8, 'Input\n5 frames\n(5x3x224x224)', '#E3F2FD'),
    (4.2, 7.0, 2.5, 0.8, 'EfficientNet-B0\n(per-frame, shared)\n-> 1280x7x7', '#BBDEFB'),
    (7.4, 7.0, 2.5, 0.8, 'CBAM\nChannel + Spatial\nAttention', '#90CAF9'),
    (10.6, 7.0, 2.0, 0.8, 'Global AvgPool\n-> 1280-dim', '#64B5F6'),
    (13.3, 7.0, 2.0, 0.8, 'Concat with\nEnv (T, H)\n-> 1282-dim', '#42A5F5'),
    (13.3, 5.3, 2.0, 0.8, 'GRU\n(hidden=128)\n-> 128-dim', '#1E88E5'),
    (13.3, 3.6, 2.0, 0.8, 'Regression\n128->64->1\nRUL hours', '#1565C0'),
]

for x, y, w, h, text, color in layers:
    ax.add_patch(FancyBboxPatch((x, y-h/2), w, h, boxstyle='round,pad=0.1',
                                     facecolor=color, edgecolor='#333', linewidth=1.5, alpha=0.85))
    ax.text(x + w/2, y, text, fontsize=9, ha='center', va='center', fontweight='bold')

for i in range(len(layers)-3):
    x1 = layers[i][0] + layers[i][2]
    y1 = layers[i][1]
    x2 = layers[i+1][0]
    ax.annotate('', xy=(x2, y1), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2))

ax.annotate('', xy=(13.3+1.0, 5.3+0.4), xytext=(13.3+1.0, 7.0-0.4),
            arrowprops=dict(arrowstyle='->', color='#555', lw=2))
ax.annotate('', xy=(13.3+1.0, 3.6+0.4), xytext=(13.3+1.0, 5.3-0.4),
            arrowprops=dict(arrowstyle='->', color='#555', lw=2))

ax.add_patch(FancyBboxPatch((8, 4.8), 2.0, 0.7, boxstyle='round,pad=0.1',
                                 facecolor='#C8E6C9', edgecolor='#333', linewidth=1.5, alpha=0.85))
ax.text(9, 5.15, 'Env Features\n(Temp, Humidity)', fontsize=8, ha='center', va='center', fontweight='bold')
ax.annotate('', xy=(13.3, 6.2), xytext=(10, 5.15),
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5, connectionstyle='arc3,rad=0.2'))

ax.annotate('PROBLEM: Only 2 env features\n(firmness available but unused)', xy=(8, 3.2), fontsize=9, color='red',
            bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.9))
ax.annotate('PROBLEM: Single unidirectional GRU,\nonly uses last timestep', xy=(12, 2.0), fontsize=9, color='red',
            bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.9))
ax.annotate('PROBLEM: Features pooled from\n7x7 grid - may lose spatial\ndetail of small mold spots', xy=(8.5, 5.8), fontsize=9, color='red',
            bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.9))

ax.set_title('Current Architecture + Identified Limitations', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(pad=1)
fig.savefig(out_dir / '05_architecture_limits.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Figure 5 saved')

# ==========================================
# FIGURE 6: Suggested Improvements Overview
# ==========================================
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.7, 'STRAWBERRY RUL PREDICTION — ROOT CAUSE DIAGNOSIS & IMPROVEMENT MAP', fontsize=16, fontweight='bold', ha='center', color='darkred')
ax.text(5, 9.3, 'Current Performance: MAE 43-54h | RMSE 50-62h | Test: 1 fruit (F05) | Data: 6 strawberries', fontsize=10, ha='center', color='gray')

problems = [
    ('1. EOL FAILURE', 'Model predicts ~90h when fruit is\nspoiled (Actual RUL=0).\nRUL values clipped to >=0.', '#F44336'),
    ('2. SEVERE OVERFITTING', 'Val loss INCREASES after epoch 3-4.\n2614 train sequences from 4 fruits.\n4.76M parameters.', '#FF5722'),
    ('3. SYSTEMATIC BIAS', 'Under-predicts RUL for fresh fruit.\nOver-predicts for spoiled fruit.\nOutput range compressed.', '#FF9800'),
    ('4. LIMITED DATA', 'Only 6 fruits total.\nNo augmentation. All share\nSAME EOL timestamp.', '#9C27B0'),
    ('5. TRAINING GAPS', 'No early stopping, no LR schedule,\nno data augmentation. batch=4.\nOnly MAE loss.', '#2196F3'),
    ('6. ARCHITECTURE GAPS', 'Single RNN layer, last-timestep only.\nOnly 2 env features. Crude\nnormalization.', '#009688'),
]

for i, (title, desc, color) in enumerate(problems):
    y = 8.5 - i * 1.25
    ax.add_patch(Rectangle((0.3, y-0.5), 2.8, 1.1, facecolor=color, alpha=0.12, edgecolor=color, linewidth=2))
    ax.text(0.5, y+0.2, title, fontsize=12, fontweight='bold', color=color, va='center')
    ax.text(0.5, y-0.35, desc, fontsize=8.5, color='#333333', va='top', family='monospace')

ax.text(5.2, 8.7, 'IMPROVEMENT DIRECTIONS', fontsize=14, fontweight='bold', color='#2E7D32')

fixes = [
    ('A1', 'Allow negative RUL for post-EOL frames', '#E8F5E9'),
    ('A2', 'Per-fruit EOL timestamps (not shared)', '#E8F5E9'),
    ('A3', 'Huber / MSLE / Quantile loss', '#E8F5E9'),
    ('A4', 'Classification head: spoiled yes/no', '#E8F5E9'),
    ('B1', 'Data augmentation (rotate, color jitter, crop)', '#E3F2FD'),
    ('B2', 'EarlyStopping + ReduceLROnPlateau', '#E3F2FD'),
    ('B3', 'Weight decay / L2 regularization', '#E3F2FD'),
    ('B4', 'Dropout increase: 0.2 -> 0.4-0.5', '#E3F2FD'),
    ('C1', 'Multi-layer BiGRU / BiLSTM', '#FFF3E0'),
    ('C2', 'Attention pooling (not just last timestep)', '#FFF3E0'),
    ('C3', 'Include firmness as 3rd env feature', '#FFF3E0'),
    ('C4', 'Proper z-score normalization', '#FFF3E0'),
    ('D1', 'MixUp / CutMix augmentation', '#F3E5F5'),
    ('D2', 'Ensemble all 4 model variants', '#F3E5F5'),
    ('D3', 'Self-supervised pretraining on fruit', '#F3E5F5'),
    ('D4', 'Larger EfficientNet (B1/B2) or ViT', '#F3E5F5'),
    ('D5', 'Multi-task: RUL + spoilage stage + days', '#F3E5F5'),
]

for j, (code, fix, color) in enumerate(fixes):
    y = 8.2 - j * 0.5
    row = j
    ax.add_patch(Rectangle((3.6, y-0.18), 6.0, 0.4, facecolor=color, alpha=0.5, edgecolor='#aaa', linewidth=0.5))
    ax.text(3.75, y, f'[{code}] {fix}', fontsize=8.8, color='#1B5E20', va='center', family='monospace', fontweight='bold')

ax.text(0.5, -0.1, 'NOTE: These are suggestions for analysis — implementation should be prioritized based on feasibility and expected impact.',
        fontsize=9, color='gray', ha='left', fontstyle='italic')

plt.tight_layout(pad=1)
fig.savefig(out_dir / '06_improvement_map.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Figure 6 saved')

print('\n=== ALL 6 FIGURES GENERATED ===')
print(f'Output directory: {out_dir}')
