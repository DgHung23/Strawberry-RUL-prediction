"""
Strawberry model run comparison.

Use --run-name tuned or --run-name regularized to select the output set.
Graphs and reports are written to output/graphs/evaluation_<run-name>/ and
output/reports/evaluation_<run-name>/.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=FutureWarning)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_SPECS = {
    "A": {
        "label": "EfficientNet-B0 + CBAM + GRU",
        "output_dir": PROJECT_ROOT / "data" / "model_A_tuned_outputs",
        "checkpoint": PROJECT_ROOT / "models" / "model_A_tuned" / "best_model.pth",
        "color": "#1E88E5",
        "marker": "o",
        "backbone": "EfficientNet-B0",
        "temporal": "GRU",
    },
    "B": {
        "label": "MobileNetV2 + CBAM + LSTM",
        "output_dir": PROJECT_ROOT / "data" / "model_B_tuned_outputs",
        "checkpoint": PROJECT_ROOT / "models" / "model_B_tuned" / "best_model.pth",
        "color": "#FB8C00",
        "marker": "s",
        "backbone": "MobileNetV2",
        "temporal": "LSTM",
    },
    "C": {
        "label": "EfficientNet-B0 + CBAM + LSTM",
        "output_dir": PROJECT_ROOT / "data" / "model_C_tuned_outputs",
        "checkpoint": PROJECT_ROOT / "models" / "model_C_tuned" / "best_model.pth",
        "color": "#43A047",
        "marker": "D",
        "backbone": "EfficientNet-B0",
        "temporal": "LSTM",
    },
    "D": {
        "label": "MobileNetV2 + CBAM + GRU",
        "output_dir": PROJECT_ROOT / "data" / "model_D_tuned_outputs",
        "checkpoint": PROJECT_ROOT / "models" / "model_D_tuned" / "best_model.pth",
        "color": "#D81B60",
        "marker": "^",
        "backbone": "MobileNetV2",
        "temporal": "GRU",
    },
}

COMMON_CONFIG = {
    "seq_len": 10,
    "fusion_mode": "late_env_branch",
    "temporal_pooling": "last_mean_max",
    "loss": "smooth_l1",
    "freeze_backbone": True,
    "epochs": 10,
}

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    }
)


def load_training_history(output_dir: Path) -> Optional[pd.DataFrame]:
    path = output_dir / "training_history.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_metrics(output_dir: Path) -> Optional[dict]:
    path = output_dir / "metrics.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(output_dir: Path) -> Optional[pd.DataFrame]:
    path = output_dir / "test_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def specs_for_run(run_name: str) -> Dict[str, dict]:
    specs: Dict[str, dict] = {}
    for key, cfg in MODEL_SPECS.items():
        run_cfg = dict(cfg)
        run_cfg["output_dir"] = PROJECT_ROOT / "data" / f"model_{key}_{run_name}_outputs"
        run_cfg["checkpoint"] = PROJECT_ROOT / "models" / f"model_{key}_{run_name}" / "best_model.pth"
        specs[key] = run_cfg
    return specs


def discover_available_models(model_specs: Dict[str, dict], min_epochs: int = 0) -> Dict[str, dict]:
    available: Dict[str, dict] = {}
    for key, cfg in model_specs.items():
        history = load_training_history(cfg["output_dir"])
        if history is None:
            continue
        if min_epochs and int(history["epoch"].max()) < min_epochs:
            continue
        if not (cfg["output_dir"] / "metrics.json").exists():
            continue
        if not (cfg["output_dir"] / "test_predictions.csv").exists():
            continue
        if not cfg["checkpoint"].exists():
            continue
        available[key] = dict(cfg)
    return available


def history_columns(df: pd.DataFrame) -> Tuple[str, str]:
    if {"train_mae", "val_mae"}.issubset(df.columns):
        return "train_mae", "val_mae"
    return "train_loss", "val_loss"


def plot_training_curves(models: Dict[str, dict], output_dir: Path) -> None:
    if not models:
        print("  [skip] No models with training history found.")
        return

    keys = list(models.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.5))
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        cfg = models[key]
        df = load_training_history(cfg["output_dir"])
        if df is None:
            continue

        train_col, val_col = history_columns(df)
        epochs = df["epoch"].values
        ax.plot(epochs, df[train_col].values, "o-", color=cfg["color"], label=f"Train {train_col}", linewidth=2, markersize=4)
        ax.plot(epochs, df[val_col].values, "s--", color=cfg["color"], alpha=0.55, label=f"Val {val_col}", linewidth=2, markersize=4)

        best_idx = df[val_col].idxmin()
        best_epoch = int(df.loc[best_idx, "epoch"])
        best_val = float(df.loc[best_idx, val_col])
        ax.axvline(best_epoch, color="red", linestyle=":", linewidth=1, alpha=0.7)
        ax.text(
            best_epoch + 0.05,
            best_val,
            f"best {best_epoch}",
            fontsize=8,
            color="red",
            va="bottom",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE (hours)")
        ax.set_title(f"Model {key}", fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Training Curves - Strawberry Model Runs", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = output_dir / "training_curves_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


def plot_test_metrics(models: Dict[str, dict], output_dir: Path) -> None:
    if not models:
        print("  [skip] No metrics.json files found.")
        return

    keys = list(models.keys())
    metrics_data = {key: load_metrics(models[key]["output_dir"]) for key in keys}
    metrics_data = {k: v for k, v in metrics_data.items() if v is not None}
    if not metrics_data:
        print("  [skip] No metrics.json files found.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    x = np.arange(len(keys))
    width = 0.35

    mae_vals = [float(metrics_data[k]["mae"]) for k in keys]
    rmse_vals = [float(metrics_data[k]["rmse"]) for k in keys]
    r2_vals = [float(metrics_data[k]["r2"]) for k in keys]
    labels = [f"Model {k}" for k in keys]

    bars1 = ax1.bar(x - width / 2, mae_vals, width, label="MAE", color="#1E88E5", alpha=0.85)
    bars2 = ax1.bar(x + width / 2, rmse_vals, width, label="RMSE", color="#FB8C00", alpha=0.85)
    ax1.set_ylabel("Error (hours)")
    ax1.set_title("Test Error", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25, axis="y")
    for bar, val in zip(bars1, mae_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, f"{val:.1f}", ha="center", fontsize=8)
    for bar, val in zip(bars2, rmse_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, f"{val:.1f}", ha="center", fontsize=8)

    bars3 = ax2.bar(x, r2_vals, width * 1.2, color=[models[k]["color"] for k in keys], alpha=0.9, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("R2")
    ax2.set_title("Explained Variance", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.grid(True, alpha=0.25, axis="y")
    for bar, val in zip(bars3, r2_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")

    fig.suptitle("Test Metrics Comparison - Strawberry Model Runs", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = output_dir / "test_metrics_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


def plot_predicted_vs_actual(models: Dict[str, dict], output_dir: Path) -> None:
    if not models:
        print("  [skip] No test predictions found.")
        return

    keys = list(models.keys())
    cols = min(2, len(keys))
    rows = (len(keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows))

    if len(keys) == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, key in enumerate(keys):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        cfg = models[key]
        df = load_predictions(cfg["output_dir"])
        if df is None:
            ax.set_title(f"Model {key} - No Data")
            continue

        actual = df["actual_rul"].values
        predicted = df["predicted_rul"].values
        ax.scatter(actual, predicted, alpha=0.4, s=12, color=cfg["color"], edgecolors="none")
        all_vals = np.concatenate([actual, predicted])
        min_val, max_val = float(all_vals.min()), float(all_vals.max())
        margin = (max_val - min_val) * 0.05
        ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], "k--", linewidth=0.8, alpha=0.5)
        corr = np.corrcoef(actual, predicted)[0, 1]
        ax.set_xlabel("Actual RUL (hours)")
        ax.set_ylabel("Predicted RUL (hours)")
        ax.set_title(f"Model {key}", fontweight="bold")
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=10, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.grid(True, alpha=0.25)

    for idx in range(len(keys), rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].set_visible(False)

    fig.suptitle("Predicted vs Actual - Test Set", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = output_dir / "predicted_vs_actual.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


def plot_residual_distribution(models: Dict[str, dict], output_dir: Path) -> None:
    if not models:
        print("  [skip] No test predictions for residual plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = 40
    plotted = False

    for key, cfg in models.items():
        df = load_predictions(cfg["output_dir"])
        if df is None:
            continue
        residuals = df["predicted_rul"].values - df["actual_rul"].values
        ax.hist(residuals, bins=bins, alpha=0.35, color=cfg["color"], label=f"Model {key}", edgecolor="black", linewidth=0.3)
        plotted = True

    if not plotted:
        print("  [skip] No test predictions for residual plot.")
        return

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7, label="Zero error")
    ax.set_xlabel("Residual (predicted - actual) [hours]")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution - Test Set", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")

    stats = []
    for key, cfg in models.items():
        df = load_predictions(cfg["output_dir"])
        if df is None:
            continue
        residuals = df["predicted_rul"].values - df["actual_rul"].values
        stats.append(f"Model {key}: mean={np.mean(residuals):+.1f}h, std={np.std(residuals):.1f}h")
    if stats:
        ax.text(0.02, 0.98, "\n".join(stats), transform=ax.transAxes, fontsize=8, va="top", family="monospace", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    fig.tight_layout()
    out_path = output_dir / "residual_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


def count_checkpoint_params(checkpoint_path: Path) -> Tuple[int, int, int]:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    total = 0
    backbone = 0
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        if name.endswith("running_mean") or name.endswith("running_var") or name.endswith("num_batches_tracked"):
            continue
        numel = int(tensor.numel())
        total += numel
        if name.startswith("cnn_features.") or name.startswith("cnn_pool."):
            backbone += numel
    trainable = total - backbone
    return total, trainable, backbone


def plot_model_params(models: Dict[str, dict], output_dir: Path) -> None:
    if not models:
        print("  [skip] No checkpoints found for param plot.")
        return

    keys = list(models.keys())
    totals = []
    trainables = []
    backbones = []
    labels = []

    for key in keys:
        cfg = models[key]
        total, trainable, backbone = count_checkpoint_params(cfg["checkpoint"])
        totals.append(total)
        trainables.append(trainable)
        backbones.append(backbone)
        labels.append(f"Model {key}\n{cfg['backbone']} + {cfg['temporal']}")

    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars1 = ax.bar(x - width / 2, totals, width, label="Total params", color="#455A64", alpha=0.85)
    bars2 = ax.bar(x + width / 2, trainables, width, label="Trainable params", color="#26A69A", alpha=0.85)
    ax.set_ylabel("Parameters")
    ax.set_title("Model Size Comparison - Strawberry Runs", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.2f}M"))
    for bar, val in zip(bars1, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20000, f"{val/1e6:.2f}M", ha="center", fontsize=8)
    for bar, val in zip(bars2, trainables):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20000, f"{val/1e6:.2f}M", ha="center", fontsize=8)

    fig.tight_layout()
    out_path = output_dir / "model_params_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


def best_metric(models: Dict[str, dict], metric: str, lower_is_better: bool = True) -> Optional[Tuple[str, float]]:
    candidates = []
    for key, cfg in models.items():
        metrics = load_metrics(cfg["output_dir"])
        if metrics is not None and metric in metrics:
            candidates.append((key, float(metrics[metric])))
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[1]) if lower_is_better else max(candidates, key=lambda x: x[1])


def generate_report(models: Dict[str, dict], output_dir: Path, run_name: str) -> None:
    keys = list(models.keys())
    if not keys:
        print("  [skip] No model data for report.")
        return

    lines = []
    lines.append(f"# Strawberry Model Comparison - {run_name}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("Configuration values are read from each model's metrics.json when available.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Model | Backbone | Temporal | Fusion | Seq Len | Epochs | Batch | LR | Weight Decay | Dropout |")
    lines.append("|-------|----------|----------|--------|---------|--------|-------|----|--------------|---------|")
    for key in keys:
        cfg = models[key]
        metrics = load_metrics(cfg["output_dir"]) or {}
        lines.append(
            f"| **{key}** | {cfg['backbone']} | {cfg['temporal']} | "
            f"{metrics.get('fusion_mode', COMMON_CONFIG['fusion_mode'])} | "
            f"{metrics.get('seq_len', COMMON_CONFIG['seq_len'])} | "
            f"{metrics.get('completed_epochs', metrics.get('epochs', COMMON_CONFIG['epochs']))} | "
            f"{metrics.get('batch_size', 'n/a')} | "
            f"{metrics.get('learning_rate', 'n/a')} | "
            f"{metrics.get('weight_decay', 'n/a')} | "
            f"{metrics.get('dropout', 'n/a')} |"
        )
    lines.append("")

    lines.append("## Test Performance")
    lines.append("")
    lines.append("| Model | MAE | RMSE | MAPE | R2 | Best Epoch | Best Val MAE | Total Params | Trainable Params |")
    lines.append("|-------|-----|------|------|----|------------|--------------|--------------|----------------|")
    for key in keys:
        cfg = models[key]
        metrics = load_metrics(cfg["output_dir"]) or {}
        history = load_training_history(cfg["output_dir"])
        total_params, trainable_params, _ = count_checkpoint_params(cfg["checkpoint"])
        best_epoch = metrics.get("best_epoch", "n/a")
        best_val_mae = metrics.get("best_val_mae", metrics.get("val_mae", float("nan")))
        lines.append(
            f"| **{key}** | {metrics.get('mae', float('nan')):.2f} | {metrics.get('rmse', float('nan')):.2f} | "
            f"{metrics.get('mape', float('nan')):.1f} | {metrics.get('r2', float('nan')):.4f} | "
            f"{best_epoch} | {best_val_mae:.2f} | {total_params:,} | {trainable_params:,} |"
        )
        if history is not None and "val_mae" in history.columns:
            start_val = float(history.loc[0, "val_mae"])
            best_val = float(history["val_mae"].min())
            end_val = float(history.loc[len(history) - 1, "val_mae"])
            lines.append(f"  - Model {key}: val MAE {start_val:.2f} -> {best_val:.2f} -> {end_val:.2f}")
    lines.append("")

    lines.append("## Best Per Metric")
    lines.append("")
    for metric, label, lower in [
        ("mae", "MAE", True),
        ("rmse", "RMSE", True),
        ("mape", "MAPE", True),
        ("r2", "R2", False),
    ]:
        winner = best_metric(models, metric, lower)
        if winner is not None:
            lines.append(f"- Best {label}: Model {winner[0]} ({winner[1]:.4f})")
    lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("| Model | Checkpoint | History | Predictions | Metrics |")
    lines.append("|-------|------------|---------|-------------|---------|")
    for key in keys:
        cfg = models[key]
        out = cfg["output_dir"]
        checkpoint = cfg["checkpoint"]
        lines.append(
            f"| **{key}** | `{checkpoint.relative_to(PROJECT_ROOT)}` | "
            f"`{(out / 'training_history.csv').relative_to(PROJECT_ROOT)}` | "
            f"`{(out / 'test_predictions.csv').relative_to(PROJECT_ROOT)}` | "
            f"`{(out / 'metrics.json').relative_to(PROJECT_ROOT)}` |"
        )
    lines.append("")

    winner = best_metric(models, "mae", True)
    if winner is not None:
        lines.append(f"Best overall by MAE: Model {winner[0]} ({winner[1]:.2f} hours).")

    report_path = output_dir / "model_comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] Saved: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare strawberry tuned/regularized model runs.")
    parser.add_argument("--run-name", default="batch128_lr7e4_nopin")
    parser.add_argument("--min-epochs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name.strip().replace(" ", "_")

    print("=" * 60)
    print(f" Strawberry Model Comparison: {run_name}")
    print("=" * 60)

    graphs_dir = PROJECT_ROOT / "output" / "graphs" / f"evaluation_{run_name}"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = PROJECT_ROOT / "output" / "reports" / f"evaluation_{run_name}"
    reports_dir.mkdir(parents=True, exist_ok=True)

    models = discover_available_models(specs_for_run(run_name), min_epochs=args.min_epochs)
    print(f"\nFound {len(models)} model run(s): {', '.join(models.keys()) if models else 'none'}")
    if not models:
        print(f"\nNo {run_name} outputs found. Run the training scripts first.")
        return

    print("\n[1/5] Training curves...")
    plot_training_curves(models, graphs_dir)

    print("\n[2/5] Test metrics...")
    plot_test_metrics(models, graphs_dir)

    print("\n[3/5] Predicted vs actual...")
    plot_predicted_vs_actual(models, graphs_dir)

    print("\n[4/5] Residual distribution...")
    plot_residual_distribution(models, graphs_dir)

    print("\n[5/5] Parameter counts and report...")
    plot_model_params(models, graphs_dir)
    generate_report(models, reports_dir, run_name)

    print("\n" + "=" * 60)
    print(" Done!")
    print(f"   Charts: {graphs_dir}")
    print(f"   Report: {reports_dir / 'model_comparison_report.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
