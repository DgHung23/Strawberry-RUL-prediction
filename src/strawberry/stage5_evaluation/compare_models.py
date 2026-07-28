"""
Model Comparison & Evaluation Script
=====================================
Reads training outputs from all model_*_outputs/ directories,
generates comparison charts and a summary report.

Usage:
    python src/stage5_evaluation/compare_models.py

Output:
    output/graphs/evaluation/
        training_curves_comparison.png   — train/val loss over epochs (all models)
        test_metrics_comparison.png      — bar chart: MAE, RMSE, MAPE, R²
        predicted_vs_actual.png          — scatter: predicted RUL vs actual RUL
        residual_distribution.png        — histogram of prediction errors
        model_params_comparison.png      — bar chart: total & trainable params
    output/reports/evaluation/
        model_comparison_report.md       — tabulated summary report
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIRS = {
    "A": {
        "label": "EfficientNet-B0 + CBAM + GRU",
        "output_dir": PROJECT_ROOT / "data" / "model_A_outputs",
        "color": "#2196F3",  # blue
        "marker": "o",
    },
    "B": {
        "label": "MobileNetV2 + CBAM + LSTM",
        "output_dir": PROJECT_ROOT / "data" / "model_B_outputs",
        "color": "#FF9800",  # orange
        "marker": "s",
    },
    "C": {
        "label": "EfficientNet-B0 + CBAM + LSTM",
        "output_dir": PROJECT_ROOT / "data" / "model_C_outputs",
        "color": "#4CAF50",  # green
        "marker": "D",
    },
    "D": {
        "label": "MobileNetV2 + CBAM + GRU",
        "output_dir": PROJECT_ROOT / "data" / "model_D_outputs",
        "color": "#E91E63",  # pink
        "marker": "^",
    },
}

# Global style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_training_history(output_dir: Path) -> Optional[pd.DataFrame]:
    """Load training_history.csv if it exists."""
    path = output_dir / "training_history.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_metrics(output_dir: Path) -> Optional[dict]:
    """Load metrics.json if it exists."""
    path = output_dir / "metrics.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_test_predictions(output_dir: Path) -> Optional[pd.DataFrame]:
    """Load test_predictions.csv if it exists."""
    path = output_dir / "test_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def discover_available_models() -> Dict[str, dict]:
    """Return only models whose output directory contains training_history.csv."""
    available = {}
    for key, cfg in MODEL_DIRS.items():
        if (cfg["output_dir"] / "training_history.csv").exists():
            available[key] = {**cfg}
    return available


# ---------------------------------------------------------------------------
# Chart 1: Training Curves (loss over epochs)
# ---------------------------------------------------------------------------

def plot_training_curves(models: Dict[str, dict], output_dir: Path):
    """Train + validation loss over epochs, one subplot per model."""
    n = len(models)
    if n == 0:
        print("  [skip] No models with training history found.")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (key, cfg) in zip(axes, models.items()):
        df = load_training_history(cfg["output_dir"])
        if df is None:
            continue
        epochs = df["epoch"].values
        ax.plot(epochs, df["train_loss"].values, "o-", color=cfg["color"],
                label="Train Loss", linewidth=2, markersize=5)
        ax.plot(epochs, df["val_loss"].values, "s--", color=cfg["color"],
                alpha=0.5, label="Val Loss", linewidth=2, markersize=5)

        best_idx = df["val_loss"].idxmin()
        best_epoch = int(df.loc[best_idx, "epoch"])
        best_val = df.loc[best_idx, "val_loss"]
        ax.axvline(x=best_epoch, color="red", linestyle=":", alpha=0.6, linewidth=1)
        ax.annotate(f"Best epoch {best_epoch}\n(val={best_val:.1f})",
                    xy=(best_epoch, best_val), fontsize=8, color="red",
                    xytext=(best_epoch + 0.5, ax.get_ylim()[1] * 0.7) if best_epoch < 8
                    else (best_epoch - 0.5, ax.get_ylim()[1] * 0.7),
                    ha="left")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE Loss (hours)")
        ax.set_title(f"Model {key}", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training Curves — All Models", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = output_dir / "training_curves_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Chart 2: Test Metrics Bar Chart
# ---------------------------------------------------------------------------

def plot_test_metrics(models: Dict[str, dict], output_dir: Path):
    """Grouped bar chart: MAE, RMSE for each model (lower = better) + R² separately."""
    model_keys = sorted(models.keys())
    metrics_data = {}
    for key in model_keys:
        m = load_metrics(models[key]["output_dir"])
        if m is not None:
            metrics_data[key] = m

    if not metrics_data:
        print("  [skip] No metrics.json files found.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    labels = [f"Model {k}" for k in model_keys]
    x = np.arange(len(labels))
    width = 0.35

    mae_vals = [metrics_data[k].get("mae", 0) for k in model_keys]
    rmse_vals = [metrics_data[k].get("rmse", 0) for k in model_keys]
    r2_vals = [metrics_data[k].get("r2", 0) for k in model_keys]

    colors = [models[k]["color"] for k in model_keys]

    # MAE & RMSE
    bars1 = ax1.bar(x - width / 2, mae_vals, width, label="MAE (hours)", color="#2196F3", alpha=0.8)
    bars2 = ax1.bar(x + width / 2, rmse_vals, width, label="RMSE (hours)", color="#FF5722", alpha=0.8)
    ax1.set_ylabel("Error (hours)")
    ax1.set_title("Prediction Error (lower = better)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars1, mae_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", fontsize=8, fontweight="bold")
    for bar, val in zip(bars2, rmse_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", fontsize=8, fontweight="bold")

    # R²
    bar_colors_r2 = colors
    bars3 = ax2.bar(x, r2_vals, width * 1.2, color=bar_colors_r2, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("R² Score")
    ax2.set_title("Coefficient of Determination (higher = better)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars3, r2_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 if val >= 0 else 0.02,
                 f"{val:.3f}", ha="center", fontsize=9, fontweight="bold",
                 color="green" if val > 0.5 else ("orange" if val > 0.2 else "red"))

    fig.suptitle("Test Set Metrics Comparison", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = output_dir / "test_metrics_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Chart 3: Predicted vs Actual RUL
# ---------------------------------------------------------------------------

def plot_predicted_vs_actual(models: Dict[str, dict], output_dir: Path):
    """Scatter plots: predicted vs actual RUL, one subplot per model."""
    n = len(models)
    if n == 0:
        print("  [skip] No test predictions found.")
        return

    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows))
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (key, cfg) in enumerate(sorted(models.items())):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        df = load_test_predictions(cfg["output_dir"])
        if df is None:
            ax.set_title(f"Model {key} — No Data")
            continue

        actual = df["actual_rul"].values
        predicted = df["predicted_rul"].values

        ax.scatter(actual, predicted, alpha=0.4, s=12, color=cfg["color"], edgecolors="none")

        # Identity line
        all_vals = np.concatenate([actual, predicted])
        min_val, max_val = all_vals.min(), all_vals.max()
        margin = (max_val - min_val) * 0.05
        ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin],
                "k--", linewidth=0.8, alpha=0.5, label="Perfect prediction")

        # Correlation annotation
        corr = np.corrcoef(actual, predicted)[0, 1]
        ax.set_xlabel("Actual RUL (hours)")
        ax.set_ylabel("Predicted RUL (hours)")
        ax.set_title(f"Model {key}", fontweight="bold")
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].set_visible(False)

    fig.suptitle("Predicted vs Actual RUL — Test Set", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = output_dir / "predicted_vs_actual.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Chart 4: Residual Distribution
# ---------------------------------------------------------------------------

def plot_residual_distribution(models: Dict[str, dict], output_dir: Path):
    """Histogram of residuals (predicted - actual), overlaid."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    all_residuals = {}

    for key, cfg in sorted(models.items()):
        df = load_test_predictions(cfg["output_dir"])
        if df is None:
            continue
        residuals = df["predicted_rul"].values - df["actual_rul"].values
        all_residuals[key] = residuals

    if not all_residuals:
        print("  [skip] No test predictions for residual plot.")
        return

    # Histogram with KDE-like step
    bins = 40
    for key, residuals in all_residuals.items():
        cfg = models[key]
        ax.hist(residuals, bins=bins, alpha=0.35, color=cfg["color"], label=f"Model {key}",
                edgecolor="black", linewidth=0.3)

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.7, label="Zero error")
    ax.set_xlabel("Residual (predicted − actual) [hours]")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution — Test Set", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate stats
    y_max = ax.get_ylim()[1]
    stat_lines = []
    for key, residuals in all_residuals.items():
        mu = np.mean(residuals)
        sigma = np.std(residuals)
        stat_lines.append(f"Model {key}: μ={mu:+.1f}h, σ={sigma:.1f}h")
    ax.text(0.02, 0.98, "\n".join(stat_lines), transform=ax.transAxes,
            fontsize=8, verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    fig.tight_layout()
    out_path = output_dir / "residual_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Chart 5: Model Parameter Count
# ---------------------------------------------------------------------------

def plot_model_params(output_dir: Path):
    """Bar chart showing parameter counts for all four model architectures."""
    # These are the known counts from model.py shape tests
    param_data = {
        "A\n(EfficientNet+GRU)": {"total": 4762975, "color": "#2196F3"},
        "B\n(MobileNetV2+LSTM)": {"total": 3160035, "color": "#FF9800"},
        "C\n(EfficientNet+LSTM)": {"total": 4943711, "color": "#4CAF50"},
        "D\n(MobileNetV2+GRU)": {"total": 2979299, "color": "#E91E63"},
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = list(param_data.keys())
    values = [param_data[k]["total"] for k in labels]
    colors = [param_data[k]["color"] for k in labels]

    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Total Parameters")
    ax.set_title("Model Size Comparison", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M"))

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40000,
                f"{val/1e6:.2f}M", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    out_path = output_dir / "model_params_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Report: Markdown Summary
# ---------------------------------------------------------------------------

def generate_report(models: Dict[str, dict], output_dir: Path):
    """Write a comprehensive comparison report in Markdown."""
    model_keys = sorted(models.keys())
    if not model_keys:
        print("  [skip] No model data for report.")
        return

    lines = []
    lines.append("# Model Comparison Report")
    lines.append("")
    lines.append(f"**Generated:** Auto-generated by `compare_models.py`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Models Overview")
    lines.append("")
    lines.append("| Model | CNN Backbone | Attention | Temporal |")
    lines.append("|-------|-------------|-----------|----------|")
    arch_map = {
        "A": ("EfficientNet-B0", "CBAM", "GRU"),
        "B": ("MobileNetV2", "CBAM", "LSTM"),
        "C": ("EfficientNet-B0", "CBAM", "LSTM"),
        "D": ("MobileNetV2", "CBAM", "GRU"),
    }
    param_map = {
        "A": "4,762,975", "B": "3,160,035",
        "C": "4,943,711", "D": "2,979,299",
    }
    for key in model_keys:
        arch = arch_map.get(key, ("?", "?", "?"))
        lines.append(f"| **{key}** | {arch[0]} | {arch[1]} | {arch[2]} |")
    lines.append("")

    lines.append("## 2. Test Set Performance")
    lines.append("")
    lines.append("| Model | MAE (h) | RMSE (h) | MAPE (%) | R² | Train Seq | Val Seq | Test Seq |")
    lines.append("|-------|---------|----------|----------|-----|-----------|---------|----------|")

    for key in model_keys:
        m = load_metrics(models[key]["output_dir"])
        if m is None:
            lines.append(f"| **{key}** | — | — | — | — | — | — | — |")
            continue
        mae = m.get("mae", float("nan"))
        rmse = m.get("rmse", float("nan"))
        mape = m.get("mape", float("nan"))
        r2 = m.get("r2", float("nan"))
        train_n = m.get("train_sequences", 0)
        val_n = m.get("val_sequences", 0)
        test_n = m.get("test_sequences", 0)
        lines.append(
            f"| **{key}** | {mae:.2f} | {rmse:.2f} | {mape:.1f} | {r2:.4f} | "
            f"{train_n} | {val_n} | {test_n} |"
        )
    lines.append("")

    # Best per metric
    lines.append("### Best Per Metric")
    lines.append("")
    best = {}
    for metric, label, lower_better in [
        ("mae", "MAE (hours)", True),
        ("rmse", "RMSE (hours)", True),
        ("mape", "MAPE (%)", True),
        ("r2", "R²", False),
    ]:
        candidates = []
        for key in model_keys:
            m = load_metrics(models[key]["output_dir"])
            if m and metric in m:
                candidates.append((key, m[metric]))
        if candidates:
            if lower_better:
                winner = min(candidates, key=lambda x: x[1])
            else:
                winner = max(candidates, key=lambda x: x[1])
            best[metric] = winner
            lines.append(f"- **{label}:** Model {winner[0]} ({winner[1]:.3f})")
    lines.append("")

    # Training dynamics summary
    lines.append("## 3. Training Dynamics")
    lines.append("")
    lines.append("| Model | Best Epoch | Best Val Loss | Final Train Loss | Overfit? |")
    lines.append("|-------|-----------|---------------|------------------|----------|")

    for key in model_keys:
        df = load_training_history(models[key]["output_dir"])
        if df is None:
            lines.append(f"| **{key}** | — | — | — | — |")
            continue
        best_idx = df["val_loss"].idxmin()
        best_epoch = int(df.loc[best_idx, "epoch"])
        best_val = df.loc[best_idx, "val_loss"]
        final_train = df.loc[len(df) - 1, "train_loss"]
        # Heuristic: overfitting if val_loss increased by >20% from best
        final_val = df.loc[len(df) - 1, "val_loss"]
        overfit_ratio = (final_val - best_val) / (abs(best_val) + 1e-8)
        overfit_flag = "⚠️ Yes" if overfit_ratio > 0.2 else "No"
        lines.append(
            f"| **{key}** | {best_epoch} | {best_val:.2f} | {final_train:.2f} | {overfit_flag} |"
        )
    lines.append("")

    # Architecture comparison
    lines.append("## 4. Architecture Insights")
    lines.append("")
    lines.append("### CNN Backbone Effect")
    lines.append("")
    lines.append(
        "Comparing models with the **same temporal model** isolates the CNN effect:"
    )
    lines.append("")
    lines.append(
        "- **GRU pairs:** Model A (EfficientNet) vs Model D (MobileNetV2) → "
        "EfficientNet captures richer spatial features."
    )
    lines.append(
        "- **LSTM pairs:** Model C (EfficientNet) vs Model B (MobileNetV2) → "
        "Same comparison across LSTM variants."
    )
    lines.append("")
    lines.append("### Temporal Model Effect")
    lines.append("")
    lines.append(
        "Comparing models with the **same CNN backbone** isolates the RNN effect:"
    )
    lines.append("")
    lines.append(
        "- **EfficientNet pairs:** Model A (GRU) vs Model C (LSTM) → "
        "GRU often trains faster; LSTM may capture longer dependencies."
    )
    lines.append(
        "- **MobileNetV2 pairs:** Model D (GRU) vs Model B (LSTM) → "
        "GRU simpler but often comparable on short sequences."
    )
    lines.append("")
    lines.append("### CBAM Attention")
    lines.append("")
    lines.append(
        "All four models integrate CBAM (Channel + Spatial Attention) between "
        "the CNN feature extractor and global pooling. This helps the network "
        "focus on salient degradation markers (mold spots, color shifts, "
        "texture changes) rather than background noise. ~205K parameters added."
    )
    lines.append("")

    # File paths
    lines.append("## 5. Output Files")
    lines.append("")
    lines.append("| Model | Checkpoint | History | Predictions | Metrics |")
    lines.append("|-------|-----------|---------|-------------|---------|")
    for key in model_keys:
        cfg = models[key]
        out = cfg["output_dir"]
        checkpoint = PROJECT_ROOT / "models" / f"model_{key}" / "best_model.pth"
        ckpt_str = f"`models/model_{key}/best_model.pth`" + (" ✅" if checkpoint.exists() else " ❌")
        hist_str = f"`{out.relative_to(PROJECT_ROOT)}/training_history.csv`" if (out / "training_history.csv").exists() else "—"
        pred_str = f"`{out.relative_to(PROJECT_ROOT)}/test_predictions.csv`" if (out / "test_predictions.csv").exists() else "—"
        metr_str = f"`{out.relative_to(PROJECT_ROOT)}/metrics.json`" if (out / "metrics.json").exists() else "—"
        lines.append(f"| **{key}** | {ckpt_str} | {hist_str} | {pred_str} | {metr_str} |")
    lines.append("")

    # Write
    report_path = output_dir / "model_comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] Saved: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(" Model Comparison & Evaluation")
    print("=" * 60)

    # Setup output directories
    graphs_dir = PROJECT_ROOT / "output" / "graphs" / "evaluation"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = PROJECT_ROOT / "output" / "reports" / "evaluation"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Discover models with training data
    models = discover_available_models()
    print(f"\nFound {len(models)} model(s) with training data: "
          f"{', '.join(sorted(models.keys()))}")

    if not models:
        print("\nNo trained models found. Run training scripts first:")
        for key in sorted(MODEL_DIRS.keys()):
            print(f"  cd src/stage4_training/model_{key} && python train.py")
        return

    # --- Charts ---
    print("\n[1/6] Generating training curves...")
    plot_training_curves(models, graphs_dir)

    print("\n[2/6] Generating test metrics bar chart...")
    plot_test_metrics(models, graphs_dir)

    print("\n[3/6] Generating predicted vs actual plots...")
    plot_predicted_vs_actual(models, graphs_dir)

    print("\n[4/6] Generating residual distribution...")
    plot_residual_distribution(models, graphs_dir)

    print("\n[5/6] Generating model params comparison...")
    plot_model_params(graphs_dir)

    print("\n[6/6] Generating comparison report...")
    generate_report(models, reports_dir)

    print("\n" + "=" * 60)
    print(" Done! Outputs:")
    print(f"   Charts:  {graphs_dir}/")
    print(f"   Report:  {reports_dir}/model_comparison_report.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
