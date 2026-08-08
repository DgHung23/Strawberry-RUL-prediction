from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from strawberry.training.models import build_model
except Exception:  # pragma: no cover - optional parameter chart only
    build_model = None

from .common import (
    MODEL_REGISTRY,
    EvaluationData,
    format_hours,
    load_evaluation_data,
    markdown_table,
    model_color,
    model_keys,
    model_label,
)


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


def plot_training_curves(data: EvaluationData) -> Path | None:
    if data.histories.empty:
        print("  [skip] No training history found.")
        return None

    keys = model_keys(data)
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.7), sharey=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        hist = data.histories[data.histories["model_key"] == key]
        color = model_color(key)
        for _, fold_hist in hist.groupby("fold_name"):
            ax.plot(fold_hist["epoch"], fold_hist["train_mae"], color=color, alpha=0.13, linewidth=1)
            ax.plot(fold_hist["epoch"], fold_hist["val_mae"], color="#333333", alpha=0.10, linewidth=1, linestyle="--")

        grouped = hist.groupby("epoch").agg(train_mae=("train_mae", "mean"), val_mae=("val_mae", "mean")).reset_index()
        ax.plot(grouped["epoch"], grouped["train_mae"], "o-", color=color, label="Mean train MAE", linewidth=2.2, markersize=4)
        ax.plot(grouped["epoch"], grouped["val_mae"], "s--", color="#D32F2F", label="Mean val MAE", linewidth=2.0, markersize=4)

        model_folds = data.fold_results[data.fold_results["model_key"] == key]
        best_epoch = int(round(model_folds["best_epoch"].median()))
        ax.axvline(best_epoch, color="#D32F2F", linestyle=":", linewidth=1.2, alpha=0.8)
        ax.text(best_epoch + 0.05, ax.get_ylim()[1] * 0.88, f"median best={best_epoch}", color="#D32F2F", fontsize=8)
        ax.set_title(f"Model {key}\n{model_label(key)}", fontweight="bold", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("MAE (hours, lower is better)")
    fig.suptitle("Training Curves Comparison", fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    out = data.graph_dir / "training_curves_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Saved: {out}")
    return out


def plot_test_metrics(data: EvaluationData) -> Path:
    summary = data.fold_summary.sort_values("model_key")
    keys = summary["model_key"].tolist()
    x = np.arange(len(keys))
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    ax1, ax2, ax3 = axes

    bars_mae = ax1.bar(x - width / 2, summary["test_mae_mean"], width, yerr=summary["test_mae_std"], capsize=4, label="MAE", color="#1E88E5", alpha=0.85)
    bars_rmse = ax1.bar(x + width / 2, summary["test_rmse_mean"], width, yerr=summary["test_rmse_std"], capsize=4, label="RMSE", color="#FB8C00", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Model {k}" for k in keys])
    ax1.set_ylabel("Error (hours)")
    ax1.set_title("Prediction Error (lower is better)", fontweight="bold")
    ax1.grid(True, alpha=0.25, axis="y")
    ax1.legend(fontsize=9)
    for bars in (bars_mae, bars_rmse):
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35, f"{bar.get_height():.1f}", ha="center", fontsize=8)

    bars_r2 = ax2.bar(x, summary["test_r2_mean"], width * 1.25, yerr=summary["test_r2_std"], capsize=4, color=[model_color(k) for k in keys], alpha=0.9, edgecolor="black", linewidth=0.5)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Model {k}" for k in keys])
    ax2.set_ylabel("R2")
    ax2.set_title("Explained Variance (higher is better)", fontweight="bold")
    ax2.grid(True, alpha=0.25, axis="y")
    for bar in bars_r2:
        value = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=8, fontweight="bold")

    bars_smape = ax3.bar(x, summary["test_smape_mean"], width * 1.25, yerr=summary["test_smape_std"], capsize=4, color=[model_color(k) for k in keys], alpha=0.9)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Model {k}" for k in keys])
    ax3.set_ylabel("SMAPE (%)")
    ax3.set_title("Symmetric Percent Error (lower is better)", fontweight="bold")
    ax3.grid(True, alpha=0.25, axis="y")
    for bar in bars_smape:
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{bar.get_height():.1f}%", ha="center", fontsize=8)

    fig.suptitle("LOOCV Test Metrics Comparison", fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    out = data.graph_dir / "test_metrics_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Saved: {out}")
    return out


def plot_predicted_vs_actual(data: EvaluationData, max_points_per_model: int = 2500) -> Path | None:
    if data.predictions.empty:
        print("  [skip] No prediction files found.")
        return None
    keys = model_keys(data)
    cols = 2
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11.5, 5.2 * rows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    ymin = float(np.nanmin([data.predictions["actual_rul_hours"].min(), data.predictions["predicted_rul_hours"].min()]))
    ymax = float(np.nanmax([data.predictions["actual_rul_hours"].max(), data.predictions["predicted_rul_hours"].max()]))
    pad = (ymax - ymin) * 0.05
    ymin -= pad
    ymax += pad

    for ax, key in zip(axes, keys):
        group = data.predictions[data.predictions["model_key"] == key]
        if len(group) > max_points_per_model:
            group = group.sample(max_points_per_model, random_state=42)
        ax.scatter(group["actual_rul_hours"], group["predicted_rul_hours"], s=10, alpha=0.25, color=model_color(key), edgecolor="none")
        ax.plot([ymin, ymax], [ymin, ymax], color="black", linestyle="--", linewidth=1)
        summary = data.fold_summary[data.fold_summary["model_key"] == key].iloc[0]
        ax.set_title(f"Model {key}: MAE={summary.test_mae_mean:.2f}h, R2={summary.test_r2_mean:.3f}", fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(ymin, ymax)
        ax.set_ylim(ymin, ymax)
    for ax in axes[len(keys) :]:
        ax.axis("off")
    for ax in axes[::cols]:
        ax.set_ylabel("Predicted RUL (hours)")
    for ax in axes[-cols:]:
        ax.set_xlabel("Actual RUL (hours)")

    fig.suptitle("Predicted vs Actual RUL - Held-out Fruit Predictions", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = data.graph_dir / "predicted_vs_actual.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Saved: {out}")
    return out


def plot_residual_distribution(data: EvaluationData) -> Path | None:
    if data.predictions.empty:
        print("  [skip] No prediction files found.")
        return None
    keys = model_keys(data)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax_hist, ax_box = axes
    bins = np.linspace(-60, 60, 49)
    for key in keys:
        residual = data.predictions[data.predictions["model_key"] == key]["residual_pred_minus_actual"].dropna()
        ax_hist.hist(residual, bins=bins, alpha=0.38, color=model_color(key), label=f"Model {key}", density=True)
    ax_hist.axvline(0, color="black", linestyle="--", linewidth=1)
    ax_hist.set_xlabel("Residual (predicted - actual, hours)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Residual Distribution", fontweight="bold")
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.25)

    box_data = [data.predictions[data.predictions["model_key"] == key]["residual_pred_minus_actual"].dropna().to_numpy() for key in keys]
    try:
        box = ax_box.boxplot(box_data, tick_labels=[f"Model {k}" for k in keys], patch_artist=True, showfliers=False)
    except TypeError:
        box = ax_box.boxplot(box_data, labels=[f"Model {k}" for k in keys], patch_artist=True, showfliers=False)
    for patch, key in zip(box["boxes"], keys):
        patch.set_facecolor(model_color(key))
        patch.set_alpha(0.75)
    ax_box.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_box.set_ylabel("Residual (hours)")
    ax_box.set_title("Residual Spread (outliers hidden)", fontweight="bold")
    ax_box.grid(True, alpha=0.25, axis="y")

    fig.suptitle("Prediction Error Analysis", fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    out = data.graph_dir / "residual_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Saved: {out}")
    return out


def plot_fold_heatmaps(data: EvaluationData) -> Path:
    fold = data.fold_results
    mae = fold.pivot(index="test_group", columns="model_key", values="test_mae").sort_index()
    r2 = fold.pivot(index="test_group", columns="model_key", values="test_r2").sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1))
    for ax, pivot, title, cmap, fmt in [
        (axes[0], mae, "MAE by held-out fruit (lower is better)", "magma_r", ".1f"),
        (axes[1], r2, "R2 by held-out fruit (higher is better)", "viridis", ".2f"),
    ]:
        values = pivot.to_numpy(dtype=float)
        im = ax.imshow(values, aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"Model {c}" for c in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(title, fontweight="bold")
        midpoint = np.nanmean(values)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", fontsize=8, color="white" if values[i, j] > midpoint else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("LOOCV Fold Heatmaps", fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    out = data.graph_dir / "fold_metrics_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Saved: {out}")
    return out


def plot_fold_mae_by_fruit(data: EvaluationData) -> Path:
    pivot = data.fold_results.pivot(index="test_group", columns="model_key", values="test_mae").sort_index()
    keys = list(pivot.columns)
    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(keys), 1)
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for idx, key in enumerate(keys):
        offset = (idx - (len(keys) - 1) / 2) * width
        ax.bar(x + offset, pivot[key], width=width, color=model_color(key), label=f"Model {key}", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_xlabel("Held-out fruit")
    ax.set_ylabel("MAE (hours)")
    ax.set_title("Per-fruit LOOCV MAE", fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(ncol=len(keys), loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    out = data.graph_dir / "fold_mae_by_fruit.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Saved: {out}")
    return out


def plot_model_stability(data: EvaluationData) -> Path:
    keys = model_keys(data)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    for ax, metric, title, ylabel in [
        (axes[0], "test_mae", "MAE stability", "MAE hours"),
        (axes[1], "test_rmse", "RMSE stability", "RMSE hours"),
        (axes[2], "test_r2", "R2 stability", "R2"),
    ]:
        values = [data.fold_results[data.fold_results["model_key"] == key][metric].dropna().to_numpy() for key in keys]
        try:
            box = ax.boxplot(values, tick_labels=[f"Model {k}" for k in keys], patch_artist=True, showfliers=True)
        except TypeError:
            box = ax.boxplot(values, labels=[f"Model {k}" for k in keys], patch_artist=True, showfliers=True)
        for patch, key in zip(box["boxes"], keys):
            patch.set_facecolor(model_color(key))
            patch.set_alpha(0.72)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, axis="y")
    fig.suptitle("Fold-to-fold Stability", fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    out = data.graph_dir / "model_stability_boxplots.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Saved: {out}")
    return out


def count_model_parameters(data: EvaluationData) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    if build_model is None:
        return pd.DataFrame(rows)
    config = data.config
    for key in model_keys(data):
        try:
            model = build_model(
                key,
                hidden_size=int(config.get("hidden_size", 160)),
                env_hidden_size=int(config.get("env_hidden_size", 64)),
                dropout=float(config.get("dropout", 0.25)),
                backbone_dropout=float(config.get("backbone_dropout", 0.1)),
                temporal_pooling=str(config.get("temporal_pooling", "last_mean_max")),
                fusion_mode=str(config.get("fusion_mode", "image_only")),
                pretrained_backbone=False,
            )
        except Exception:
            continue
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rows.append({"model_key": key, "total_params": int(total), "trainable_params": int(trainable)})
        del model
    return pd.DataFrame(rows)


def plot_model_params(data: EvaluationData) -> Path | None:
    params = count_model_parameters(data)
    if params.empty:
        print("  [skip] Could not instantiate models for parameter count.")
        return None
    params.to_csv(data.report_dir / "model_parameter_counts.csv", index=False)
    params = params.sort_values("model_key")
    keys = params["model_key"].tolist()
    x = np.arange(len(keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - width / 2, params["total_params"] / 1e6, width, label="Total params", color="#607D8B", alpha=0.85)
    ax.bar(x + width / 2, params["trainable_params"] / 1e6, width, label="Trainable params", color=[model_color(k) for k in keys], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Model {k}" for k in keys])
    ax.set_ylabel("Parameters (millions)")
    ax.set_title("Model Size Comparison", fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(fontsize=9)
    for xpos, val in zip(x, params["total_params"] / 1e6):
        ax.text(xpos - width / 2, val + 0.25, f"{val:.1f}M", ha="center", fontsize=8)
    fig.tight_layout()
    out = data.graph_dir / "model_params_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Saved: {out}")
    return out


def write_comparison_report(data: EvaluationData, graph_files: list[Path | None]) -> Path:
    summary = data.fold_summary.copy()
    best = summary.sort_values("test_mae_mean").iloc[0]
    second = summary.sort_values("test_mae_mean").iloc[1] if len(summary) > 1 else best
    fold_best = data.fold_results.loc[data.fold_results.groupby("test_group")["test_mae"].idxmin()].sort_values("test_group")
    worst_by_model = data.fold_results.loc[data.fold_results.groupby("model_key")["test_mae"].idxmax()].sort_values("model_key")

    config = data.config
    lines: list[str] = [
        "# Strawberry RUL model comparison report",
        "",
        f"- Run root: `{data.run_root}`",
        f"- Graph directory: `{data.graph_dir}`",
        f"- Report directory: `{data.report_dir}`",
        f"- Completed LOOCV folds: {len(data.fold_results)} total = "
        + ", ".join(f"{key}:{int((data.fold_results['model_key'] == key).sum())}" for key in model_keys(data)),
    ]
    if config:
        lines.append(
            "- Training config: "
            f"seq_len={config.get('seq_len')}, fusion_mode={config.get('fusion_mode')}, "
            f"temporal_pooling={config.get('temporal_pooling')}, env_feature_mode={config.get('env_feature_mode')}, "
            f"epochs={config.get('epochs')}, patience={config.get('patience')}"
        )
    lines.extend(
        [
            "",
            "## Executive summary",
            "",
            f"Model **{best.model_key}** is best overall by fold-averaged LOOCV MAE: **{best.test_mae_mean:.2f} +/- {best.test_mae_std:.2f} hours**.",
            f"The gap to the next best model ({second.model_key}) is **{second.test_mae_mean - best.test_mae_mean:.2f} hours**.",
            f"Best RMSE model: **{summary.sort_values('test_rmse_mean').iloc[0].model_key}**. Best R2 model: **{summary.sort_values('test_r2_mean', ascending=False).iloc[0].model_key}**. Best SMAPE model: **{summary.sort_values('test_smape_mean').iloc[0].model_key}**.",
            "",
            "## Architecture registry",
            "",
            "| Model | Architecture | Backbone | Temporal module |",
            "| --- | --- | --- | --- |",
        ]
    )
    for key in model_keys(data):
        registry = MODEL_REGISTRY.get(key, {})
        lines.append(f"| {key} | {registry.get('label', key)} | {registry.get('backbone', '')} | {registry.get('temporal', '')} |")
    lines.extend(
        [
            "",
            "## Fold-averaged LOOCV metrics",
            "",
            markdown_table(
                summary,
                [
                    "rank_by_mae",
                    "model_key",
                    "folds",
                    "test_mae_mean",
                    "test_mae_std",
                    "test_rmse_mean",
                    "test_r2_mean",
                    "test_smape_mean",
                ],
                digits=3,
            ),
            "",
            "Primary ranking uses unweighted LOOCV fold means, so each held-out fruit contributes equally.",
        ]
    )
    if not data.sample_weighted.empty:
        lines.extend(
            [
                "",
                "## Sample-weighted held-out prediction metrics",
                "",
                markdown_table(
                    data.sample_weighted,
                    [
                        "model_key",
                        "n_predictions",
                        "weighted_mae",
                        "weighted_rmse",
                        "weighted_r2",
                        "weighted_smape",
                        "bias_pred_minus_actual",
                        "median_absolute_error",
                        "p90_absolute_error",
                    ],
                    digits=3,
                ),
                "",
                "These metrics weight fruits with more generated sequences more heavily; use them as diagnostics rather than the main LOOCV ranking.",
            ]
        )
    lines.extend(
        [
            "",
            "## Best model by held-out fruit",
            "",
            markdown_table(fold_best, ["test_group", "model_key", "test_mae", "test_rmse", "test_r2", "test_smape"], digits=3),
            "",
            "## Worst held-out fruit per model",
            "",
            markdown_table(worst_by_model, ["model_key", "test_group", "test_mae", "test_rmse", "test_r2", "test_smape"], digits=3),
            "",
            "## Generated graphs",
            "",
        ]
    )
    for graph in graph_files:
        if graph and graph.exists():
            rel = graph.relative_to(data.report_dir) if graph.is_relative_to(data.report_dir) else Path("..") / ".." / ".." / "graphs" / "evaluation" / data.run_root.name / graph.name
            lines.append(f"- [{graph.name}]({rel.as_posix()})")
    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- Negative fold R2 means the model underperformed a constant-mean predictor on that held-out fruit.",
            "- Folds with high MAE should be audited for unusual visual degradation patterns, sensor gaps, or lifecycle label noise.",
            "- For this run, select the lowest fold-averaged MAE model unless deployment constraints require a lighter model.",
            "",
        ]
    )

    out = data.report_dir / "model_comparison_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    summary_payload = {
        "run_root": str(data.run_root),
        "best_model_by_mae": best.to_dict(),
        "graph_files": [str(p) for p in graph_files if p],
    }
    (data.report_dir / "model_comparison_report.json").write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
    print(f"  [OK] Saved: {out}")
    return out


def run_comparison(run_root: str | Path | None = None, graph_dir: str | Path | None = None, report_dir: str | Path | None = None, max_points_per_model: int = 2500) -> EvaluationData:
    data = load_evaluation_data(run_root=run_root, graph_dir=graph_dir, report_dir=report_dir)
    graph_files = [
        plot_training_curves(data),
        plot_test_metrics(data),
        plot_predicted_vs_actual(data, max_points_per_model=max_points_per_model),
        plot_residual_distribution(data),
        plot_fold_heatmaps(data),
        plot_fold_mae_by_fruit(data),
        plot_model_stability(data),
        plot_model_params(data),
    ]
    write_comparison_report(data, graph_files)
    print("")
    print("Fold-averaged LOOCV summary:")
    print(data.fold_summary[["rank_by_mae", "model_key", "test_mae_mean", "test_mae_std", "test_rmse_mean", "test_r2_mean", "test_smape_mean"]].to_string(index=False))
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate strawberry LOOCV model comparison graphs and report.")
    parser.add_argument("--run-root", type=Path, default=None, help="Run root containing fold_results.csv. Defaults to latest strawberry run.")
    parser.add_argument("--graph-dir", type=Path, default=None, help="Output graph directory.")
    parser.add_argument("--report-dir", type=Path, default=None, help="Output report directory.")
    parser.add_argument("--max-points-per-model", type=int, default=2500, help="Scatter sampling cap per model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_comparison(args.run_root, args.graph_dir, args.report_dir, args.max_points_per_model)


if __name__ == "__main__":
    main()

