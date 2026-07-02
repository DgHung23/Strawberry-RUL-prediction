"""
Per-Model Detailed Report Generator
====================================
For each trained model under data/model_*_outputs/, produces:

  1. output/reports/evaluation/model_{X}_detailed_report.md
     A standalone, non-technical-friendly report with
     - Architecture summary (plain language)
     - Training progress over epochs
     - Test set performance breakdown
     - Prediction error analysis (distribution, percentiles)
     - Visual regression analysis (per-sample error curve)
     - RUL range analysis (performance across early/mid/late lifecycle)
     - Recommendations

  2. output/graphs/evaluation/model_{X}_detailed_curves.png
     A multi-panel figure:
     - (a) Train/Val loss curve
     - (b) Predicted vs Actual scatter + identity line
     - (c) Residual (error) histogram
     - (d) Error by RUL bucket (performance across fruit lifecycle)

Usage:
    python src/stage5_evaluation/generate_model_report.py          # all models
    python src/stage5_evaluation/generate_model_report.py A B C    # specific models
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_REGISTRY = {
    "A": {
        "name": "Model A",
        "full_name": "EfficientNet-B0 + CBAM + GRU",
        "output_dir": PROJECT_ROOT / "data" / "model_A_outputs",
        "checkpoint": PROJECT_ROOT / "models" / "model_A" / "best_model.pth",
        "color_primary": "#2196F3",
        "color_secondary": "#1976D2",
        "emoji": "🅰️",
    },
    "B": {
        "name": "Model B",
        "full_name": "MobileNetV2 + CBAM + LSTM",
        "output_dir": PROJECT_ROOT / "data" / "model_B_outputs",
        "checkpoint": PROJECT_ROOT / "models" / "model_B" / "best_model.pth",
        "color_primary": "#FF9800",
        "color_secondary": "#E65100",
        "emoji": "🅱️",
    },
    "C": {
        "name": "Model C",
        "full_name": "EfficientNet-B0 + CBAM + LSTM",
        "output_dir": PROJECT_ROOT / "data" / "model_C_outputs",
        "checkpoint": PROJECT_ROOT / "models" / "model_C" / "best_model.pth",
        "color_primary": "#4CAF50",
        "color_secondary": "#1B5E20",
        "emoji": "©️",
    },
    "D": {
        "name": "Model D",
        "full_name": "MobileNetV2 + CBAM + GRU",
        "output_dir": PROJECT_ROOT / "data" / "model_D_outputs",
        "checkpoint": PROJECT_ROOT / "models" / "model_D" / "best_model.pth",
        "color_primary": "#E91E63",
        "color_secondary": "#880E4F",
        "emoji": "🅳",
    },
}

# Plain-language component glossary for non-tech readers
GLOSSARY = {
    "EfficientNet-B0": (
        "EfficientNet-B0 (Vision Module) — a pretrained neural network that "
        "has already learned to recognize shapes, textures, and colors from "
        "millions of real-world images. It converts each strawberry photo into "
        "a list of 1,280 numbers describing what it sees."
    ),
    "MobileNetV2": (
        "MobileNetV2 (Lightweight Vision Module) — a smaller, faster version of "
        "the vision network. Best for quick predictions or running on limited hardware. "
        "Slightly less detailed than EfficientNet but ~40% fewer computations."
    ),
    "CBAM": (
        "CBAM (Attention Module) — helps the model focus on the most important parts "
        "of each image. Instead of treating all pixels equally, CBAM learns to pay "
        "extra attention to signs of spoilage: mold spots, color changes, texture shifts. "
        "Think of it as the model's 'magnifying glass' over the fruit surface."
    ),
    "GRU": (
        "GRU (Time-Series Memory) — tracks how the fruit changes across the 5-frame "
        "video sequence. A GRU is a simplified recurrent unit that remembers previous "
        "frames to understand the speed of degradation."
    ),
    "LSTM": (
        "LSTM (Time-Series Memory, Enhanced) — similar to GRU but with a more detailed "
        "'memory cell'. Better at capturing subtle long-term changes across frames, but "
        "uses slightly more computation."
    ),
}

# Metric explanations (plain English)
METRIC_EXPLANATIONS = {
    "mae": (
        "**MAE (Mean Absolute Error)** — the average prediction error in hours. "
        "If MAE = 43h, the model is typically off by about 43 hours when predicting "
        "remaining shelf life. Lower is better."
    ),
    "rmse": (
        "**RMSE (Root Mean Squared Error)** — similar to MAE but penalizes large "
        "mistakes more heavily. A model with a few very bad predictions will have "
        "a much higher RMSE. Always >= MAE. Lower is better."
    ),
    "mape": (
        "**MAPE (Mean Absolute Percentage Error)** — the average error expressed as "
        "a percentage of the true RUL. MAPE = 100% means predictions are, on average, "
        "off by the same amount as the true value. Lower is better."
    ),
    "r2": (
        "**R² (R-Squared / Coefficient of Determination)** — measures how well the "
        "model explains the variation in fruit shelf life. R² = 0 means the model is "
        "no better than just guessing the average RUL every time. R² = 1 means perfect "
        "prediction. Typically, R² > 0.5 is usable; R² > 0.7 is good; R² > 0.9 is excellent."
    ),
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})


# ===================================================================
# Data loaders
# ===================================================================

def load_data(model_key: str) -> dict:
    """Load all available data for one model. Returns dict with keys or None values."""
    cfg = MODEL_REGISTRY[model_key]
    out = cfg["output_dir"]

    result = {"key": model_key, "config": cfg, "has_data": False}

    # metrics.json
    metrics_path = out / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            result["metrics"] = json.load(f)
    else:
        result["metrics"] = None

    # training_history.csv
    hist_path = out / "training_history.csv"
    if hist_path.exists():
        result["history"] = pd.read_csv(hist_path)
    else:
        result["history"] = None

    # test_predictions.csv
    preds_path = out / "test_predictions.csv"
    if preds_path.exists():
        result["predictions"] = pd.read_csv(preds_path)
    else:
        result["predictions"] = None

    # Checkpoint
    ckpt = cfg["checkpoint"]
    result["has_checkpoint"] = ckpt.exists()

    result["has_data"] = (
        result["metrics"] is not None
        and result["history"] is not None
        and result["predictions"] is not None
    )

    return result


# ===================================================================
# Detailed chart: 4-panel per model
# ===================================================================

def plot_detailed_curves(data: dict, output_path: Path):
    """Generate a 4-panel detailed figure for one model."""
    cfg = data["config"]
    color = cfg["color_primary"]
    color2 = cfg["color_secondary"]

    history = data["history"]
    preds = data["predictions"]
    metrics = data["metrics"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ((ax_loss, ax_scatter), (ax_resid, ax_bucket)) = axes

    # --- Panel (a): Training & Validation Loss ---
    epochs = history["epoch"].values
    train_loss = history["train_loss"].values
    val_loss = history["val_loss"].values

    ax_loss.plot(epochs, train_loss, "o-", color=color, linewidth=2.2,
                 markersize=6, label="Training Loss (how well model fits training fruit)")
    ax_loss.plot(epochs, val_loss, "s--", color=color2, linewidth=2.2,
                 markersize=6, alpha=0.6, label="Validation Loss (how well model works on unseen fruit)")
    best_epoch_idx = val_loss.argmin()
    ax_loss.axvline(x=epochs[best_epoch_idx], color="red", linestyle=":",
                    linewidth=1.2, alpha=0.7)
    ax_loss.annotate(
        f"Best at Epoch {int(epochs[best_epoch_idx])}\n(val={val_loss[best_epoch_idx]:.1f}h)",
        xy=(epochs[best_epoch_idx], val_loss[best_epoch_idx]),
        xytext=(epochs[best_epoch_idx] + 0.7, val_loss[best_epoch_idx] + 5),
        arrowprops=dict(arrowstyle="->", color="red", alpha=0.6),
        fontsize=9, color="red", fontweight="bold",
    )
    ax_loss.set_xlabel("Epoch (training cycle)")
    ax_loss.set_ylabel("Error (hours) — lower is better")
    ax_loss.set_title("(a) Training Progress", fontweight="bold")
    ax_loss.legend(fontsize=8.5, loc="upper right")
    ax_loss.grid(True, alpha=0.25)

    # --- Panel (b): Predicted vs Actual ---
    actual = preds["actual_rul"].values
    predicted = preds["predicted_rul"].values
    residuals = predicted - actual

    ax_scatter.scatter(actual, predicted, alpha=0.4, s=10,
                       color=color, edgecolors="none", label="Each point = one test prediction")
    all_vals = np.concatenate([actual, predicted])
    min_v, max_v = all_vals.min(), all_vals.max()
    pad = (max_v - min_v) * 0.05
    ax_scatter.plot([min_v - pad, max_v + pad], [min_v - pad, max_v + pad],
                    "k--", linewidth=0.8, alpha=0.5, label="Perfect prediction (y = x)")
    corr = np.corrcoef(actual, predicted)[0, 1]
    mae_val = metrics["mae"]
    r2_val = metrics["r2"]
    textstr = f"Correlation: {corr:.3f}\nMAE: {mae_val:.1f} hours\nR²: {r2_val:.3f}"
    ax_scatter.text(0.05, 0.95, textstr, transform=ax_scatter.transAxes,
                    fontsize=9.5, verticalalignment="top", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray"))
    ax_scatter.set_xlabel("Actual RUL (hours) — how long the fruit REALLY lasted")
    ax_scatter.set_ylabel("Predicted RUL (hours) — the model's guess")
    ax_scatter.set_title("(b) Predicted vs Actual Shelf Life", fontweight="bold")
    ax_scatter.legend(fontsize=8, loc="lower right")
    ax_scatter.grid(True, alpha=0.25)

    # --- Panel (c): Residual (Error) Histogram ---
    ax_resid.hist(residuals, bins=35, color=color, alpha=0.7, edgecolor="black", linewidth=0.3)
    ax_resid.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.7,
                     label="Zero error (perfect guess)")
    ax_resid.axvline(x=np.mean(residuals), color=color2, linestyle="-", linewidth=1.5,
                     label=f"Mean bias = {np.mean(residuals):+.1f}h")
    mu, sigma = np.mean(residuals), np.std(residuals)
    p5, p95 = np.percentile(residuals, [5, 95])
    ax_resid.set_xlabel("Prediction Error (hours)")
    ax_resid.set_ylabel("Number of Predictions")
    ax_resid.set_title("(c) Distribution of Prediction Errors", fontweight="bold")
    stat_text = (f"Mean error (bias): {mu:+.1f}h\n"
                 f"Std deviation: {sigma:.1f}h\n"
                 f"90% of errors in: [{p5:+.0f}h, {p95:+.0f}h]")
    ax_resid.text(0.02, 0.98, stat_text, transform=ax_resid.transAxes,
                  fontsize=8.5, verticalalignment="top", family="monospace",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
    ax_resid.legend(fontsize=8, loc="upper right")
    ax_resid.grid(True, alpha=0.25, axis="y")

    # --- Panel (d): Error by RUL Bucket ---
    buckets = [0, 50, 100, 150, 200, 250, 300]
    bucket_labels = []
    bucket_maes = []
    bucket_stds = []
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        mask = (actual >= lo) & (actual < hi)
        if mask.sum() > 0:
            err = np.abs(residuals[mask])
            bucket_labels.append(f"{lo}-{hi}h\n({mask.sum()} samples)")
            bucket_maes.append(err.mean())
            bucket_stds.append(err.std())
        else:
            bucket_labels.append(f"{lo}-{hi}h\n(0)")
            bucket_maes.append(0)
            bucket_stds.append(0)

    xpos = np.arange(len(bucket_labels))
    bars = ax_bucket.bar(xpos, bucket_maes, color=color, alpha=0.8,
                         edgecolor="black", linewidth=0.5,
                         yerr=bucket_stds, capsize=4, error_kw={"alpha": 0.5})
    ax_bucket.set_xticks(xpos)
    ax_bucket.set_xticklabels(bucket_labels, fontsize=8)
    ax_bucket.set_ylabel("Average Error (hours)")
    ax_bucket.set_xlabel("True RUL Range (hours remaining)")
    ax_bucket.set_title("(d) Performance Across Fruit Lifecycle Stages", fontweight="bold")
    ax_bucket.grid(True, alpha=0.25, axis="y")
    for bar, val in zip(bars, bucket_maes):
        if val > 0:
            ax_bucket.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                           f"{val:.0f}h", ha="center", fontsize=8.5, fontweight="bold")

    # Lifecycle annotation
    ax_bucket.annotate(
        "Early life\n(plenty of time left)",
        xy=(0, ax_bucket.get_ylim()[1] * 0.85), fontsize=8,
        ha="center", color="green",
    )
    ax_bucket.annotate(
        "Near spoilage\n(critical predictions)",
        xy=(len(bucket_labels) - 1, ax_bucket.get_ylim()[1] * 0.85), fontsize=8,
        ha="center", color="red",
    )

    # Sup title
    fig.suptitle(
        f"{cfg['name']}: {cfg['full_name']}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  [OK] Chart saved: {output_path}")


# ===================================================================
# Detailed Markdown Report per model
# ===================================================================

def generate_markdown_report(data: dict, output_path: Path):
    """Write a detailed, non-technical-friendly per-model report."""
    cfg = data["config"]
    metrics = data["metrics"]
    history = data["history"]
    preds = data["predictions"]

    actual = preds["actual_rul"].values
    predicted = preds["predicted_rul"].values
    residuals = predicted - actual
    abs_errors = np.abs(residuals)

    L = []  # line accumulator

    def w(s=""):
        L.append(s)

    # ---- Header ----
    w(f"# {cfg['emoji']}  {cfg['name']} — Detailed Report")
    w("")
    w(f"**Architecture:** {cfg['full_name']}")
    w("")
    w(f"*Auto-generated report — see `src/stage5_evaluation/generate_model_report.py`*")
    w("")
    w("---")
    w("")

    # ---- 1. Executive Summary ----
    w("## 1. Executive Summary")
    w("")
    mae, rmse, mape, r2 = metrics["mae"], metrics["rmse"], metrics["mape"], metrics["r2"]
    verdict = (
        "excellent" if r2 > 0.85 else
        "good" if r2 > 0.7 else
        "moderate" if r2 > 0.5 else
        "weak" if r2 > 0.2 else
        "poor"
    )
    w(f"> **One-sentence summary:** {cfg['name']} predicts strawberry shelf life with "
      f"an average error of **{mae:.1f} hours** and achieves **{verdict}** "
      f"accuracy (R² = {r2:.3f}).")
    w("")
    w("| Metric | Value | What This Means |")
    w("|--------|-------|-----------------|")
    w(f"| **MAE** | **{mae:.1f} hours** | On average, predictions are off by about {mae:.0f} hours "
      f"(~{mae/24:.1f} days)")
    w(f"| **RMSE** | **{rmse:.1f} hours** | Large errors are penalized — worst mistakes average ~{rmse:.0f} hours")
    w(f"| **R²** | **{r2:.3f}** | {cfg['name']} explains **{r2*100:.1f}%** of the variation in fruit shelf life")
    w(f"| **MAPE** | **{mape:.1f}%** | The average error is about {mape:.0f}% of the true RUL value")
    w("")

    # Metric gauge-style interpretation
    w("### How to Read R²")
    w("")
    w(f"R² = {r2:.3f} means:")
    w("")
    if r2 > 0.7:
        w(f"- ✅ The model captures most of the pattern in strawberry degradation")
        w(f"- ✅ Predictions are substantially better than just guessing the average shelf life")
    elif r2 > 0.5:
        w(f"- ⚠️ The model is usable but not highly precise")
        w(f"- ⚠️ Predictions are better than guessing the average, but with room to improve")
    elif r2 > 0.2:
        w(f"- ⚠️ The model finds some signal but is often inaccurate")
        w(f"- ⚠️ Consider this a baseline — try more data, augmentation, or architecture changes")
    else:
        w(f"- ❌ The model struggles to find a reliable pattern")
        w(f"- ❌ Predictions are barely better than random guessing")
    w("")

    # ---- 2. Architecture (plain language) ----
    w("## 2. How This Model Works")
    w("")
    w(f"{cfg['name']} is a hybrid AI that combines three specialized modules:")
    w("")

    # Parse components
    if "EfficientNet" in cfg["full_name"]:
        backbone = "EfficientNet-B0"
    else:
        backbone = "MobileNetV2"
    if "LSTM" in cfg["full_name"]:
        temporal = "LSTM"
    else:
        temporal = "GRU"

    w(f"### Step-by-step pipeline")
    w("")
    w("```text")
    w("  Strawberry photo (224x224)")
    w("       |")
    w(f"       v")
    w(f"  [1] {backbone} (Vision) — see shapes, colors, textures")
    w(f"       |       {GLOSSARY[backbone]}")
    w(f"       v")
    w(f"  [2] CBAM (Attention) — focus on spoilage signs")
    w(f"       |       {GLOSSARY['CBAM']}")
    w(f"       v")
    w(f"  [3] {temporal} (Memory) — track changes over 5 frames")
    w(f"       |       {GLOSSARY[temporal]}")
    w(f"       v")
    w(f"  [4] Prediction Head — output single number = hours left")
    w("```")
    w("")

    # ---- 3. Training Progress ----
    w("## 3. Training Progress")
    w("")
    w(f"The model was trained for **{len(history)} epochs** (cycles through the entire training set). "
      f"Each epoch, it sees 2,614 training sequences (4 fruits × ~653 frames each, minus window overlap). "
      f"After each epoch, it's tested on 598 validation sequences from Fruit #06 (never seen during training).")
    w("")

    best_idx = history["val_loss"].idxmin()
    best_epoch = int(history.loc[best_idx, "epoch"])
    best_val = history.loc[best_idx, "val_loss"]
    final_train = history.loc[len(history) - 1, "train_loss"]
    final_val = history.loc[len(history) - 1, "val_loss"]
    overfit_ratio = (final_val - best_val) / (abs(best_val) + 1e-8)

    w("| Epoch | Train Loss (h) | Val Loss (h) | Notes |")
    w("|-------|---------------|-------------|-------|")
    for _, row in history.iterrows():
        ep = int(row["epoch"])
        tl = row["train_loss"]
        vl = row["val_loss"]
        note = ""
        if ep == best_epoch:
            note = " ⭐ Best epoch (lowest validation error)"
        elif ep > best_epoch and vl > best_val * 1.2:
            note = " ⚠️ Starting to overfit"
        w(f"| {ep} | {tl:.2f} | {vl:.2f} |{note} |")
    w("")

    w("### What the loss curves tell us")
    w("")
    w(f"- **Best model saved at epoch {best_epoch}** (validation error = {best_val:.1f} hours)")
    w(f"- **Final training error:** {final_train:.1f} hours")
    w(f"- **Final validation error:** {final_val:.1f} hours")
    if overfit_ratio > 0.2:
        w(f"- ⚠️ **Overfitting detected:** After epoch {best_epoch}, the model started memorizing "
          f"training data rather than learning general patterns. Validation error increased "
          f"from {best_val:.1f}h to {final_val:.1f}h ({(overfit_ratio*100):.0f}% worse).")
        w(f"  - **Fix:** Add dropout, freeze backbone earlier, use data augmentation, or reduce epochs.")
    else:
        w(f"- ✅ **No significant overfitting detected** — training and validation losses stay aligned.")
    w("")

    # ---- 4. Test Performance ----
    w("## 4. Test Set Performance (Final Evaluation)")
    w("")
    w(f"The best checkpoint (epoch {best_epoch}) is evaluated on **672 test sequences** "
      f"from Fruit #05 — a completely held-out fruit never seen during training or validation.")
    w("")

    # Percentiles
    pcts = [5, 10, 25, 50, 75, 90, 95]
    pct_vals = np.percentile(abs_errors, pcts)

    w("### Error Distribution")
    w("")
    w("| Percentile | Error (hours) | Interpretation |")
    w("|-----------|---------------|----------------|")
    for p, v in zip(pcts, pct_vals):
        interp = (
            "Half of predictions are this accurate or better"
            if p == 50 else
            "90% of predictions are better than this"
            if p == 90 else
            "Only the worst 5% exceed this"
            if p == 95 else
            ""
        )
        w(f"| **P{p}** | {v:.1f}h | {interp} |")
    w("")

    w(f"- **Median error:** {pct_vals[3]:.1f}h (half of all predictions)")
    w(f"- **90% of predictions:** error ≤ {pct_vals[5]:.1f}h")
    w(f"- **Worst 5% of predictions:** error ≥ {pct_vals[6]:.1f}h")
    w(f"- **Mean bias:** {np.mean(residuals):+.1f}h "
      f"({'model tends to overestimate' if np.mean(residuals) > 0 else 'model tends to underestimate'} RUL)")
    w("")

    # ---- 5. Performance by Lifecycle Stage ----
    w("## 5. Performance Across Fruit Lifecycle")
    w("")
    w("Does the model work equally well for fresh strawberries vs nearly-spoiled ones?")
    w("")

    buckets = [0, 50, 100, 150, 200, 250, 300]
    w("| RUL Range | Samples | Avg Error | Std Dev | Reliability |")
    w("|-----------|---------|-----------|---------|-------------|")
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        mask = (actual >= lo) & (actual < hi)
        n = mask.sum()
        if n > 0:
            bucket_mae = abs_errors[mask].mean()
            bucket_std = abs_errors[mask].std()
            reliability = (
                "✅ Good" if bucket_mae < (hi - lo) * 0.3 else
                "⚠️ Fair" if bucket_mae < (hi - lo) * 0.6 else
                "❌ Poor"
            )
            w(f"| {lo}–{hi}h | {n} | {bucket_mae:.1f}h | {bucket_std:.1f}h | {reliability} |")
        else:
            w(f"| {lo}–{hi}h | 0 | — | — | — |")
    w("")

    # ---- 6. Component Comparison (if other models available) ----
    w("## 6. How to Improve")
    w("")
    w("### Quick wins (low effort, possible gains)")
    w("")
    w("- **Data augmentation:** Randomly flip, rotate, or adjust brightness of images during training "
      "to make the model more robust. Add to `dataset.py` transforms.")
    w("- **Longer sequences:** Change `seq_len` from 5 to 8 or 10 in `train.py` — gives the temporal "
      "model more context per prediction.")
    w("- **Learning rate schedule:** Reduce learning rate when validation loss plateaus "
      "(add `torch.optim.lr_scheduler.ReduceLROnPlateau`).")
    w("- **Early stopping:** Stop training automatically when validation loss stops improving "
      "for N epochs, instead of fixed 10 epochs.")
    w("")
    w("### Architecture improvements (higher effort)")
    w("")
    w("- **Larger EfficientNet variant:** B1 or B2 offers richer features at the cost of speed.")
    w("- **Deeper RNN:** Increase `num_layers` from 1 to 2 (may need more data).")
    w("- **Multi-head attention:** Replace or augment CBAM with transformer-style self-attention.")
    if temporal == "GRU":
        w(f"- **Try LSTM instead:** {cfg['name']} uses GRU. Model {'C' if backbone == 'EfficientNet-B0' else 'B'} "
          f"uses LSTM with the same backbone — compare their reports.")
    else:
        w(f"- **Try GRU instead:** {cfg['name']} uses LSTM. Model {'A' if backbone == 'EfficientNet-B0' else 'D'} "
          f"uses GRU with the same backbone — compare their reports.")
    w("")

    # ---- 7. Data & File Inventory ----
    w("## 7. File Inventory")
    w("")
    w("| File | Path | Status |")
    w("|------|------|--------|")
    ckpt_path = cfg["checkpoint"]
    check = "✅ Exists" if data["has_checkpoint"] else "❌ Not found"
    w(f"| Model checkpoint | `{ckpt_path.relative_to(PROJECT_ROOT)}` | {check} |")
    out = cfg["output_dir"]
    for fname, desc in [
        ("metrics.json", "Performance metrics"),
        ("training_history.csv", "Epoch-by-epoch loss"),
        ("test_predictions.csv", "Test set predictions"),
    ]:
        exists = (out / fname).exists()
        path_str = str((out / fname).relative_to(PROJECT_ROOT))
        w(f"| {desc} | `{path_str}` | {'✅' if exists else '❌'} |")
    w("")

    # ---- Footer ----
    w("---")
    w("")
    w(f"*Report generated by `src/stage5_evaluation/generate_model_report.py`*  ")
    w(f"*Model architecture: {cfg['full_name']}*  ")
    w(f"*Training data: Fruits F01-F04 (2,614 sequences) | Validation: Fruit F06 (598 sequences) | "
      f"Test: Fruit F05 (672 sequences)*")
    w("")

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  [OK] Report saved: {output_path}")


# ===================================================================
# Comparison summary (all trained models in one table)
# ===================================================================

def generate_comparison_summary(all_data: Dict[str, dict], output_path: Path):
    """Write a single comparison table across all trained models."""
    trained = {k: d for k, d in all_data.items() if d["has_data"]}

    L = []
    def w(s=""): L.append(s)

    w("# All Models — Comparison Summary")
    w("")
    w(f"*Comparing {len(trained)} trained model(s)*")
    w("")

    # ---- Leaderboard ----
    w("## Leaderboard (Lower Error = Better)")
    w("")
    w("| Rank | Model | Architecture | MAE (h) | RMSE (h) | R² | Params |")
    w("|------|-------|-------------|---------|----------|-----|--------|")
    param_map = {"A": "4.76M", "B": "3.16M", "C": "4.94M", "D": "2.98M"}
    rank_emoji = {1: "1st", 2: "2nd", 3: "3rd"}
    ranked = sorted(trained.items(), key=lambda x: x[1]["metrics"]["mae"])
    for rank, (key, d) in enumerate(ranked, 1):
        m = d["metrics"]
        cfg = d["config"]
        rank_str = rank_emoji.get(rank, f"  {rank}")
        w(f"| {rank_str} | {cfg['name']} | {cfg['full_name']} | "
          f"{m['mae']:.1f} | {m['rmse']:.1f} | {m['r2']:.3f} | {param_map.get(key, '?')} |")
    w("")

    # ---- Per-metric ranking ----
    w("## Best By Metric")
    w("")
    for metric, label, lower_better in [
        ("mae", "Lowest Average Error (MAE)", True),
        ("rmse", "Most Consistent Predictions (lowest RMSE)", True),
        ("r2", "Best Overall Fit (highest R²)", False),
    ]:
        if lower_better:
            best_key = min(trained.items(), key=lambda x: x[1]["metrics"][metric])[0]
        else:
            best_key = max(trained.items(), key=lambda x: x[1]["metrics"][metric])[0]
        best = trained[best_key]
        w(f"- **{label}:** {best['config']['name']} ({best['config']['full_name']}) — "
          f"{best['metrics'][metric]:.2f}")
    w("")

    # ---- Quick links ----
    w("## Per-Model Detailed Reports")
    w("")
    for key in sorted(trained.keys()):
        d = trained[key]
        w(f"- [{d['config']['name']}](model_{key}_detailed_report.md) — {d['config']['full_name']}")
    w("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  [OK] Summary saved: {output_path}")


# ===================================================================
# Main
# ===================================================================

def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(MODEL_REGISTRY.keys())
    targets = [t.upper() for t in targets if t.upper() in MODEL_REGISTRY]

    print("=" * 60)
    print(" Per-Model Detailed Report Generator")
    print("=" * 60)
    print(f"\nModels to process: {', '.join(targets)}")

    # Load data
    all_data = {}
    for key in targets:
        all_data[key] = load_data(key)
        status = "DATA FOUND" if all_data[key]["has_data"] else "NO DATA (skip)"
        print(f"  {MODEL_REGISTRY[key]['name']}: {status}")

    # Generate per-model reports
    graphs_dir = PROJECT_ROOT / "output" / "graphs" / "evaluation"
    reports_dir = PROJECT_ROOT / "output" / "reports" / "evaluation"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    trained_count = 0
    for key, data in all_data.items():
        if not data["has_data"]:
            print(f"\n  Skipping {MODEL_REGISTRY[key]['name']} — no training data found.")
            print(f"  Run: cd src/stage4_training/model_{key} && python train.py")
            continue

        print(f"\n{'-' * 50}")
        print(f" Processing {MODEL_REGISTRY[key]['name']}...")
        print(f"{'-' * 50}")

        # Detailed chart
        chart_path = graphs_dir / f"model_{key}_detailed_curves.png"
        plot_detailed_curves(data, chart_path)

        # Detailed markdown report
        report_path = reports_dir / f"model_{key}_detailed_report.md"
        generate_markdown_report(data, report_path)

        trained_count += 1

    # Comparison summary
    if trained_count >= 2:
        print(f"\n{'-' * 50}")
        print(" Generating comparison summary...")
        print(f"{'-' * 50}")
        summary_path = reports_dir / "ALL_MODELS_comparison_summary.md"
        generate_comparison_summary(all_data, summary_path)

    print("\n" + "=" * 60)
    print(f" Done! {trained_count} model(s) processed.")
    print(f"   Charts:  {graphs_dir}/")
    print(f"   Reports: {reports_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
