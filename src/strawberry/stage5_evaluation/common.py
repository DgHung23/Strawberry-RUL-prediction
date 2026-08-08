from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_PARENT = PROJECT_ROOT / "output" / "runs" / "strawberry"

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "A": {
        "name": "Model A",
        "label": "EfficientNet-B0 + CBAM + GRU",
        "backbone": "EfficientNet-B0",
        "temporal": "GRU",
        "color": "#1E88E5",
        "marker": "o",
    },
    "B": {
        "name": "Model B",
        "label": "MobileNetV2 + CBAM + LSTM",
        "backbone": "MobileNetV2",
        "temporal": "LSTM",
        "color": "#FB8C00",
        "marker": "s",
    },
    "C": {
        "name": "Model C",
        "label": "EfficientNet-B0 + CBAM + LSTM",
        "backbone": "EfficientNet-B0",
        "temporal": "LSTM",
        "color": "#43A047",
        "marker": "D",
    },
    "D": {
        "name": "Model D",
        "label": "MobileNetV2 + CBAM + GRU",
        "backbone": "MobileNetV2",
        "temporal": "GRU",
        "color": "#D81B60",
        "marker": "^",
    },
}

METRIC_NOTES = {
    "mae": "MAE is the average absolute prediction error in hours. Lower is better.",
    "rmse": "RMSE penalizes large errors more strongly than MAE. Lower is better.",
    "r2": "R2 measures explained variance. 1 is perfect; 0 is no better than predicting the mean.",
    "smape": "SMAPE is percentage-style symmetric error. Lower is better.",
}


@dataclass(frozen=True)
class EvaluationData:
    run_root: Path
    graph_dir: Path
    report_dir: Path
    config: dict
    fold_results: pd.DataFrame
    fold_summary: pd.DataFrame
    predictions: pd.DataFrame
    sample_weighted: pd.DataFrame
    histories: pd.DataFrame


def resolve_run_root(run_root: str | Path | None = None) -> Path:
    if run_root is not None:
        path = Path(run_root)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not (path / "fold_results.csv").exists():
            raise FileNotFoundError(f"Missing fold_results.csv under run root: {path}")
        return path

    candidates = [p for p in DEFAULT_RUN_PARENT.iterdir() if (p / "fold_results.csv").exists()] if DEFAULT_RUN_PARENT.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No strawberry run with fold_results.csv found under {DEFAULT_RUN_PARENT}")
    return max(candidates, key=lambda p: (p / "fold_results.csv").stat().st_mtime)


def default_output_dirs(run_root: Path, graph_dir: str | Path | None = None, report_dir: str | Path | None = None) -> tuple[Path, Path]:
    run_name = run_root.name
    if graph_dir is None:
        graph_path = PROJECT_ROOT / "output" / "graphs" / "evaluation" / run_name
    else:
        graph_path = Path(graph_dir)
        if not graph_path.is_absolute():
            graph_path = PROJECT_ROOT / graph_path

    if report_dir is None:
        report_path = PROJECT_ROOT / "output" / "reports" / "evaluation" / run_name
    else:
        report_path = Path(report_dir)
        if not report_path.is_absolute():
            report_path = PROJECT_ROOT / report_path

    graph_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)
    return graph_path, report_path


def load_config(run_root: Path) -> dict:
    config_path = run_root / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def _numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def load_fold_results(run_root: Path) -> pd.DataFrame:
    path = run_root / "fold_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    fold = pd.read_csv(path)
    fold = _numeric(fold, ["best_epoch", "best_val_mae", "test_mae", "test_rmse", "test_r2", "test_smape"])
    return fold.sort_values(["model_key", "test_group"]).reset_index(drop=True)


def make_fold_summary(fold: pd.DataFrame) -> pd.DataFrame:
    summary = (
        fold.groupby("model_key")
        .agg(
            folds=("fold_name", "count"),
            best_epoch_mean=("best_epoch", "mean"),
            val_mae_mean=("best_val_mae", "mean"),
            val_mae_std=("best_val_mae", "std"),
            test_mae_mean=("test_mae", "mean"),
            test_mae_std=("test_mae", "std"),
            test_mae_median=("test_mae", "median"),
            test_mae_min=("test_mae", "min"),
            test_mae_max=("test_mae", "max"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            test_r2_mean=("test_r2", "mean"),
            test_r2_std=("test_r2", "std"),
            test_smape_mean=("test_smape", "mean"),
            test_smape_std=("test_smape", "std"),
        )
        .reset_index()
    )
    summary["rank_by_mae"] = summary["test_mae_mean"].rank(method="min", ascending=True).astype(int)
    summary["rank_by_rmse"] = summary["test_rmse_mean"].rank(method="min", ascending=True).astype(int)
    summary["rank_by_r2"] = summary["test_r2_mean"].rank(method="min", ascending=False).astype(int)
    return summary.sort_values(["rank_by_mae", "model_key"]).reset_index(drop=True)


def _prediction_path(run_root: Path, row: pd.Series) -> Path:
    raw = row.get("prediction_path", "")
    if isinstance(raw, str) and raw:
        path = Path(raw)
        if path.exists():
            return path
    return run_root / str(row["model_key"]) / str(row["fold_name"]) / "predictions.csv"


def load_predictions(run_root: Path, fold: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, row in fold.iterrows():
        path = _prediction_path(run_root, row)
        if not path.exists():
            continue
        pred = pd.read_csv(path)
        pred["model_key"] = row["model_key"]
        pred["fold_name"] = row["fold_name"]
        pred["test_group"] = row["test_group"]
        pred["val_group"] = row.get("val_group", "")
        pred = _numeric(pred, ["actual_rul_hours", "predicted_rul_hours", "error_hours"])
        pred["residual_pred_minus_actual"] = pred["predicted_rul_hours"] - pred["actual_rul_hours"]
        pred["absolute_error_hours"] = pred["residual_pred_minus_actual"].abs()
        frames.append(pred)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_sample_weighted_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    if predictions.empty:
        return pd.DataFrame(rows)

    for model_key, group in predictions.groupby("model_key"):
        y_true = group["actual_rul_hours"].to_numpy(dtype=float)
        y_pred = group["predicted_rul_hours"].to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        residual = y_pred - y_true
        absolute_error = np.abs(residual)
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        smape = float(np.mean(2.0 * absolute_error / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)) * 100.0)
        rows.append(
            {
                "model_key": model_key,
                "n_predictions": int(mask.sum()),
                "weighted_mae": float(np.mean(absolute_error)),
                "weighted_rmse": float(np.sqrt(np.mean(residual**2))),
                "weighted_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                "weighted_smape": smape,
                "bias_pred_minus_actual": float(np.mean(residual)),
                "median_absolute_error": float(np.median(absolute_error)),
                "p90_absolute_error": float(np.percentile(absolute_error, 90)),
            }
        )
    return pd.DataFrame(rows).sort_values("weighted_mae").reset_index(drop=True)


def _history_path(run_root: Path, row: pd.Series) -> Path:
    raw = row.get("history_path", row.get("best_history_path", ""))
    if isinstance(raw, str) and raw:
        path = Path(raw)
        if path.exists():
            return path
    return run_root / str(row["model_key"]) / str(row["fold_name"]) / "history.csv"


def load_histories(run_root: Path, fold: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, row in fold.iterrows():
        path = _history_path(run_root, row)
        if not path.exists():
            continue
        hist = pd.read_csv(path)
        hist["model_key"] = row["model_key"]
        hist["fold_name"] = row["fold_name"]
        hist["test_group"] = row["test_group"]
        frames.append(hist)
    if not frames:
        return pd.DataFrame()
    hist_all = pd.concat(frames, ignore_index=True)
    return _numeric(hist_all, ["epoch", "train_loss", "train_mae", "train_rmse", "val_loss", "val_mae", "val_rmse", "val_r2", "val_smape", "learning_rate"])


def load_evaluation_data(
    run_root: str | Path | None = None,
    graph_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
) -> EvaluationData:
    root = resolve_run_root(run_root)
    graphs, reports = default_output_dirs(root, graph_dir, report_dir)
    config = load_config(root)
    fold = load_fold_results(root)
    summary = make_fold_summary(fold)
    predictions = load_predictions(root, fold)
    weighted = compute_sample_weighted_metrics(predictions)
    histories = load_histories(root, fold)

    summary.to_csv(reports / "fold_metric_summary.csv", index=False)
    fold.to_csv(reports / "fold_metrics_long.csv", index=False)
    if not predictions.empty:
        predictions.to_csv(reports / "heldout_predictions_all_models.csv", index=False)
    if not weighted.empty:
        weighted.to_csv(reports / "sample_weighted_metrics.csv", index=False)
    if not histories.empty:
        histories.to_csv(reports / "training_history_all_models.csv", index=False)

    return EvaluationData(
        run_root=root,
        graph_dir=graphs,
        report_dir=reports,
        config=config,
        fold_results=fold,
        fold_summary=summary,
        predictions=predictions,
        sample_weighted=weighted,
        histories=histories,
    )


def model_keys(data: EvaluationData) -> list[str]:
    return sorted(data.fold_results["model_key"].astype(str).unique())


def model_color(key: str) -> str:
    return MODEL_REGISTRY.get(key, {}).get("color", "#666666")


def model_label(key: str) -> str:
    return MODEL_REGISTRY.get(key, {}).get("label", f"Model {key}")


def markdown_table(df: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    table = df[columns].copy()
    for column in table.columns:
        if pd.api.types.is_numeric_dtype(table[column]):
            table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    header = "| " + " | ".join(table.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in table.to_numpy()]
    return "\n".join([header, sep] + rows)


def format_hours(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f} h"

