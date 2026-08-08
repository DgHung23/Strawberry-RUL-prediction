from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (
    METRIC_NOTES,
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
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    }
)


def _rul_bucket_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    bins = [-np.inf, 24, 48, 72, np.inf]
    labels = ["0-24h", "24-48h", "48-72h", "72h+"]
    frame = predictions.copy()
    frame["rul_bucket"] = pd.cut(frame["actual_rul_hours"], bins=bins, labels=labels)
    return (
        frame.groupby("rul_bucket", observed=False)
        .agg(
            samples=("absolute_error_hours", "count"),
            mae=("absolute_error_hours", "mean"),
            median_ae=("absolute_error_hours", "median"),
            p90_ae=("absolute_error_hours", lambda s: float(np.percentile(s.dropna(), 90)) if len(s.dropna()) else np.nan),
            bias=("residual_pred_minus_actual", "mean"),
        )
        .reset_index()
    )


def plot_detailed_curves(data: EvaluationData, model_key: str) -> Path | None:
    fold = data.fold_results[data.fold_results["model_key"] == model_key].sort_values("test_group")
    hist = data.histories[data.histories["model_key"] == model_key]
    pred = data.predictions[data.predictions["model_key"] == model_key]
    if fold.empty:
        print(f"  [skip] Model {model_key}: no fold metrics.")
        return None

    color = model_color(model_key)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    ax_train, ax_scatter = axes[0]
    ax_resid, ax_bucket = axes[1]

    if not hist.empty:
        for _, fold_hist in hist.groupby("fold_name"):
            ax_train.plot(fold_hist["epoch"], fold_hist["train_mae"], color=color, alpha=0.12, linewidth=1)
            ax_train.plot(fold_hist["epoch"], fold_hist["val_mae"], color="#D32F2F", alpha=0.10, linewidth=1, linestyle="--")
        grouped = hist.groupby("epoch").agg(train_mae=("train_mae", "mean"), val_mae=("val_mae", "mean")).reset_index()
        ax_train.plot(grouped["epoch"], grouped["train_mae"], "o-", color=color, linewidth=2.2, markersize=4, label="Mean train MAE")
        ax_train.plot(grouped["epoch"], grouped["val_mae"], "s--", color="#D32F2F", linewidth=2.0, markersize=4, label="Mean val MAE")
        best_epoch = int(round(fold["best_epoch"].median()))
        ax_train.axvline(best_epoch, color="#D32F2F", linestyle=":", linewidth=1.2, alpha=0.8)
        ax_train.text(best_epoch + 0.05, ax_train.get_ylim()[1] * 0.88, f"median best={best_epoch}", color="#D32F2F", fontsize=8)
        ax_train.legend(fontsize=8)
    ax_train.set_title("(a) Training progress", fontweight="bold")
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("MAE (hours)")
    ax_train.grid(True, alpha=0.25)

    if not pred.empty:
        sample = pred.sample(2500, random_state=42) if len(pred) > 2500 else pred
        ymin = float(np.nanmin([pred["actual_rul_hours"].min(), pred["predicted_rul_hours"].min()]))
        ymax = float(np.nanmax([pred["actual_rul_hours"].max(), pred["predicted_rul_hours"].max()]))
        pad = (ymax - ymin) * 0.05
        ymin -= pad
        ymax += pad
        ax_scatter.scatter(sample["actual_rul_hours"], sample["predicted_rul_hours"], s=10, alpha=0.25, color=color, edgecolor="none")
        ax_scatter.plot([ymin, ymax], [ymin, ymax], color="black", linestyle="--", linewidth=1)
        ax_scatter.set_xlim(ymin, ymax)
        ax_scatter.set_ylim(ymin, ymax)
    ax_scatter.set_title("(b) Predicted vs actual RUL", fontweight="bold")
    ax_scatter.set_xlabel("Actual RUL (hours)")
    ax_scatter.set_ylabel("Predicted RUL (hours)")
    ax_scatter.grid(True, alpha=0.25)

    if not pred.empty:
        residual = pred["residual_pred_minus_actual"].dropna()
        ax_resid.hist(residual, bins=45, color=color, alpha=0.78, edgecolor="white", linewidth=0.3)
        ax_resid.axvline(0, color="black", linestyle="--", linewidth=1)
        ax_resid.axvline(residual.mean(), color="#D32F2F", linestyle=":", linewidth=1.3, label=f"mean bias={residual.mean():.1f}h")
        ax_resid.legend(fontsize=8)
    ax_resid.set_title("(c) Residual histogram", fontweight="bold")
    ax_resid.set_xlabel("Residual (predicted - actual, hours)")
    ax_resid.set_ylabel("Count")
    ax_resid.grid(True, alpha=0.25)

    bucket = _rul_bucket_frame(pred)
    if not bucket.empty:
        bars = ax_bucket.bar(bucket["rul_bucket"].astype(str), bucket["mae"], color=color, alpha=0.85)
        for bar, value in zip(bars, bucket["mae"]):
            ax_bucket.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{value:.1f}", ha="center", fontsize=8)
    ax_bucket.set_title("(d) Error by RUL range", fontweight="bold")
    ax_bucket.set_xlabel("Actual RUL range")
    ax_bucket.set_ylabel("MAE (hours)")
    ax_bucket.grid(True, alpha=0.25, axis="y")

    summary = data.fold_summary[data.fold_summary["model_key"] == model_key].iloc[0]
    fig.suptitle(
        f"Model {model_key} Detailed Evaluation - {model_label(model_key)}\n"
        f"LOOCV MAE={summary.test_mae_mean:.2f} +/- {summary.test_mae_std:.2f}h, "
        f"RMSE={summary.test_rmse_mean:.2f}h, R2={summary.test_r2_mean:.3f}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out = data.graph_dir / f"model_{model_key}_detailed_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Saved: {out}")
    return out


def write_model_report(data: EvaluationData, model_key: str, graph_path: Path | None) -> Path:
    fold = data.fold_results[data.fold_results["model_key"] == model_key].sort_values("test_group")
    pred = data.predictions[data.predictions["model_key"] == model_key]
    summary = data.fold_summary[data.fold_summary["model_key"] == model_key].iloc[0]
    registry = MODEL_REGISTRY.get(model_key, {})
    best_fold = fold.loc[fold["test_mae"].idxmin()]
    worst_fold = fold.loc[fold["test_mae"].idxmax()]
    bucket = _rul_bucket_frame(pred)

    lines = [
        f"# Model {model_key} detailed evaluation report",
        "",
        f"- Architecture: **{registry.get('label', model_label(model_key))}**",
        f"- Backbone: {registry.get('backbone', '')}",
        f"- Temporal module: {registry.get('temporal', '')}",
        f"- Run root: `{data.run_root}`",
        "",
        "## Headline performance",
        "",
        f"- Fold-averaged MAE: **{summary.test_mae_mean:.2f} +/- {summary.test_mae_std:.2f} hours**",
        f"- Fold-averaged RMSE: **{summary.test_rmse_mean:.2f} hours**",
        f"- Fold-averaged R2: **{summary.test_r2_mean:.3f}**",
        f"- Fold-averaged SMAPE: **{summary.test_smape_mean:.2f}%**",
        f"- Best held-out fruit: **{best_fold.test_group}** ({format_hours(best_fold.test_mae)} MAE)",
        f"- Hardest held-out fruit: **{worst_fold.test_group}** ({format_hours(worst_fold.test_mae)} MAE)",
        "",
        "## Fold metrics",
        "",
        markdown_table(fold, ["test_group", "val_group", "best_epoch", "best_val_mae", "test_mae", "test_rmse", "test_r2", "test_smape"], digits=3),
        "",
    ]
    if not pred.empty:
        abs_err = pred["absolute_error_hours"].dropna()
        residual = pred["residual_pred_minus_actual"].dropna()
        lines.extend(
            [
                "## Prediction error distribution",
                "",
                f"- Samples: {len(pred)} held-out sequence predictions",
                f"- Median absolute error: {abs_err.median():.2f} hours",
                f"- 75th percentile absolute error: {np.percentile(abs_err, 75):.2f} hours",
                f"- 90th percentile absolute error: {np.percentile(abs_err, 90):.2f} hours",
                f"- Mean bias (predicted - actual): {residual.mean():.2f} hours",
                "",
            ]
        )
    if not bucket.empty:
        lines.extend(
            [
                "## RUL range analysis",
                "",
                markdown_table(bucket, ["rul_bucket", "samples", "mae", "median_ae", "p90_ae", "bias"], digits=3),
                "",
            ]
        )
    if graph_path and graph_path.exists():
        rel = Path("..") / ".." / ".." / "graphs" / "evaluation" / data.run_root.name / graph_path.name
        lines.extend(["## Detailed graph", "", f"![Model {model_key} detailed curves]({rel.as_posix()})", ""])
    lines.extend(
        [
            "## Metric glossary",
            "",
            f"- {METRIC_NOTES['mae']}",
            f"- {METRIC_NOTES['rmse']}",
            f"- {METRIC_NOTES['r2']}",
            f"- {METRIC_NOTES['smape']}",
            "",
            "## Recommendation",
            "",
            "Use this model if its fold-averaged MAE/RMSE tradeoff fits deployment needs. Compare it against `model_comparison_report.md` before selecting the production checkpoint.",
            "",
        ]
    )

    out = data.report_dir / f"model_{model_key}_detailed_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "model_key": model_key,
        "summary": summary.to_dict(),
        "best_fold": best_fold.to_dict(),
        "worst_fold": worst_fold.to_dict(),
        "graph": str(graph_path) if graph_path else None,
    }
    (data.report_dir / f"model_{model_key}_detailed_report.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"  [OK] Saved: {out}")
    return out


def generate_reports(
    model_filter: list[str] | None = None,
    run_root: str | Path | None = None,
    graph_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
) -> EvaluationData:
    data = load_evaluation_data(run_root=run_root, graph_dir=graph_dir, report_dir=report_dir)
    keys = model_keys(data)
    if model_filter:
        wanted = {key.upper() for key in model_filter}
        keys = [key for key in keys if key in wanted]
    for key in keys:
        graph = plot_detailed_curves(data, key)
        write_model_report(data, key, graph)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate detailed per-model strawberry LOOCV evaluation reports.")
    parser.add_argument("models", nargs="*", help="Optional model keys to report, e.g. A B C D.")
    parser.add_argument("--run-root", type=Path, default=None, help="Run root containing fold_results.csv. Defaults to latest strawberry run.")
    parser.add_argument("--graph-dir", type=Path, default=None, help="Output graph directory.")
    parser.add_argument("--report-dir", type=Path, default=None, help="Output report directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_reports(args.models, args.run_root, args.graph_dir, args.report_dir)


if __name__ == "__main__":
    main()

