"""Utilities for the strawberry-only notebook lab.

The helpers here deliberately read only:
    data/03_split/strawberry/{train,val,test}

They support fast, repeatable baseline experiments before pushing a tuning
configuration into the larger PyTorch models.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_ROOT = PROJECT_ROOT / "data" / "03_split" / "strawberry"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SPLITS = ("train", "val", "test")

FRAME_FEATURE_COLUMNS = [
    "r_mean",
    "g_mean",
    "b_mean",
    "r_std",
    "g_std",
    "b_std",
    "brightness_mean",
    "brightness_std",
    "sat_mean",
    "val_mean",
    "grad_mean",
    "grad_std",
]


def split_dir(split: str, split_root: Path = SPLIT_ROOT) -> Path:
    path = split_root / split
    if not path.exists():
        raise FileNotFoundError(f"Missing strawberry split directory: {path}")
    return path


def fruit_dirs(split: str, split_root: Path = SPLIT_ROOT) -> list[Path]:
    return sorted(
        p for p in split_dir(split, split_root).iterdir()
        if p.is_dir() and p.name.startswith("F")
    )


def load_split_labels(split: str, split_root: Path = SPLIT_ROOT) -> pd.DataFrame:
    frames = []
    for fruit_dir in fruit_dirs(split, split_root):
        labels_path = fruit_dir / "labels.csv"
        df = pd.read_csv(labels_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["split"] = split
        df["fruit_dir"] = str(fruit_dir)
        df["local_image_path"] = [
            str(fruit_dir / "images" / Path(path).name)
            for path in df["image_path"].astype(str)
        ]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_all_labels(split_root: Path = SPLIT_ROOT) -> pd.DataFrame:
    return pd.concat(
        [load_split_labels(split, split_root) for split in SPLITS],
        ignore_index=True,
    )


def sequence_counts(seq_lens: Iterable[int], split_root: Path = SPLIT_ROOT) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for fruit_dir in fruit_dirs(split, split_root):
            df = pd.read_csv(fruit_dir / "labels.csv")
            for seq_len in seq_lens:
                rows.append(
                    {
                        "split": split,
                        "fruit_id": fruit_dir.name,
                        "seq_len": int(seq_len),
                        "frames": int(len(df)),
                        "sequences": max(int(len(df) - seq_len + 1), 0),
                    }
                )
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def extract_frame_features(image_path: str) -> np.ndarray:
    """Return low-cost image descriptors for one frame.

    This is not meant to replace CNN embeddings. It is a quick signal-finder
    for sequence length, fusion style, and feature dynamics.
    """

    img = Image.open(Path(image_path)).convert("RGB").resize((64, 64))
    arr = np.asarray(img).astype(np.float32)
    red = arr[..., 0]
    green = arr[..., 1]
    blue = arr[..., 2]
    gray = arr.mean(axis=2)
    saturation_proxy = arr.std(axis=2)

    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    grad_mean = float((grad_x.mean() + grad_y.mean()) / 2.0)
    grad_std = float(np.sqrt((grad_x.std() ** 2 + grad_y.std() ** 2) / 2.0))

    return np.array(
        [
            red.mean(),
            green.mean(),
            blue.mean(),
            red.std(),
            green.std(),
            blue.std(),
            gray.mean(),
            gray.std(),
            saturation_proxy.mean(),
            arr.max(axis=2).mean(),
            grad_mean,
            grad_std,
        ],
        dtype=np.float32,
    )


def _slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=np.float32)
    x_centered = x - x.mean()
    denom = float((x_centered ** 2).sum())
    if denom == 0:
        return 0.0
    y = values.astype(np.float32)
    return float((x_centered * (y - y.mean())).sum() / denom)


def _sequence_summary(values: np.ndarray, names: list[str], prefix: str) -> dict[str, float]:
    features = {}
    for idx, name in enumerate(names):
        series = values[:, idx].astype(np.float32)
        features[f"{prefix}{name}_first"] = float(series[0])
        features[f"{prefix}{name}_last"] = float(series[-1])
        features[f"{prefix}{name}_mean"] = float(series.mean())
        features[f"{prefix}{name}_std"] = float(series.std())
        features[f"{prefix}{name}_min"] = float(series.min())
        features[f"{prefix}{name}_max"] = float(series.max())
        features[f"{prefix}{name}_delta"] = float(series[-1] - series[0])
        features[f"{prefix}{name}_slope"] = _slope(series)
    return features


def build_sequence_feature_table(
    seq_len: int,
    split_root: Path = SPLIT_ROOT,
) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for fruit_dir in fruit_dirs(split, split_root):
            df = pd.read_csv(fruit_dir / "labels.csv")
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

            for start in range(len(df) - seq_len + 1):
                window = df.iloc[start : start + seq_len]
                image_paths = [
                    str(fruit_dir / "images" / Path(path).name)
                    for path in window["image_path"].astype(str)
                ]
                image_features = np.stack(
                    [extract_frame_features(path) for path in image_paths],
                    axis=0,
                )
                env_values = window[["temperature_c", "humidity_pct"]].to_numpy(
                    dtype=np.float32
                )

                row = {
                    "split": split,
                    "fruit_id": fruit_dir.name,
                    "seq_len": int(seq_len),
                    "start_index": int(start),
                    "target": float(window.iloc[-1]["rul_hours"]),
                }
                row.update(
                    _sequence_summary(
                        image_features,
                        FRAME_FEATURE_COLUMNS,
                        "img_",
                    )
                )
                row.update(
                    _sequence_summary(
                        env_values,
                        ["temperature_c", "humidity_pct"],
                        "env_",
                    )
                )
                rows.append(row)

    return pd.DataFrame(rows)


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_regressor(name: str, random_state: int = 42):
    if name == "ridge":
        return make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0, random_state=random_state),
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            random_state=random_state,
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
        )
    raise ValueError(f"Unknown regressor: {name}")


def _split_xy(table: pd.DataFrame, columns: list[str]):
    train = table[table["split"] == "train"].reset_index(drop=True)
    val = table[table["split"] == "val"].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)
    return (
        train[columns],
        train["target"],
        val[columns],
        val["target"],
        test[columns],
        test["target"],
    )


def evaluate_feature_model(
    table: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    fusion: str,
    random_state: int = 42,
) -> dict[str, float | str | int]:
    x_train, y_train, x_val, y_val, x_test, y_test = _split_xy(table, feature_columns)
    model = make_regressor(model_name, random_state)
    model.fit(x_train, y_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    val_metrics = regression_metrics(y_val, val_pred)
    test_metrics = regression_metrics(y_test, test_pred)
    return {
        "seq_len": int(table["seq_len"].iloc[0]),
        "fusion": fusion,
        "model": model_name,
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "late_weight_image": np.nan,
    }


def evaluate_late_random_forest(table: pd.DataFrame, random_state: int = 42) -> dict[str, float | str | int]:
    image_columns = [c for c in table.columns if c.startswith("img_")]
    env_columns = [c for c in table.columns if c.startswith("env_")]

    x_train_img, y_train, x_val_img, y_val, x_test_img, y_test = _split_xy(
        table,
        image_columns,
    )
    x_train_env, _, x_val_env, _, x_test_env, _ = _split_xy(table, env_columns)

    image_model = make_regressor("random_forest", random_state)
    env_model = make_regressor("random_forest", random_state)
    image_model.fit(x_train_img, y_train)
    env_model.fit(x_train_env, y_train)

    val_img = image_model.predict(x_val_img)
    val_env = env_model.predict(x_val_env)
    test_img = image_model.predict(x_test_img)
    test_env = env_model.predict(x_test_env)

    best_weight = 1.0
    best_val_mae = float("inf")
    best_test_pred = test_img
    for weight in np.linspace(0.0, 1.0, 21):
        val_pred = weight * val_img + (1.0 - weight) * val_env
        val_mae = mean_absolute_error(y_val, val_pred)
        if val_mae < best_val_mae:
            best_val_mae = float(val_mae)
            best_weight = float(weight)
            best_test_pred = weight * test_img + (1.0 - weight) * test_env

    val_pred = best_weight * val_img + (1.0 - best_weight) * val_env
    val_metrics = regression_metrics(y_val, val_pred)
    test_metrics = regression_metrics(y_test, best_test_pred)
    return {
        "seq_len": int(table["seq_len"].iloc[0]),
        "fusion": "late_weighted_average",
        "model": "random_forest",
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "late_weight_image": best_weight,
    }


def run_ml_baseline_sweep(
    seq_lens: Iterable[int] = (3, 5, 8, 10),
    model_names: Iterable[str] = (
        "ridge",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    ),
    output_dir: Path = RESULTS_DIR,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run a small search over sequence length, feature set, and fusion style."""

    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for seq_len in seq_lens:
        table = build_sequence_feature_table(int(seq_len))
        image_columns = [c for c in table.columns if c.startswith("img_")]
        env_columns = [c for c in table.columns if c.startswith("env_")]
        fusion_sets = {
            "image_only": image_columns,
            "numeric_only": env_columns,
            "early_fusion": image_columns + env_columns,
        }

        for fusion, columns in fusion_sets.items():
            for model_name in model_names:
                rows.append(
                    evaluate_feature_model(
                        table,
                        columns,
                        model_name,
                        fusion,
                        random_state=random_state,
                    )
                )

        rows.append(evaluate_late_random_forest(table, random_state=random_state))

    results = pd.DataFrame(rows).sort_values(["val_mae", "test_mae"]).reset_index(drop=True)
    results.to_csv(output_dir / "ml_baseline_sweep.csv", index=False)
    return results


def current_large_model_metrics(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    report_path = project_root / "output" / "reports" / "evaluation" / "model_comparison_report.md"
    rows = []

    if report_path.exists():
        in_table = False
        for line in report_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("| Model | MAE"):
                in_table = True
                continue
            if not in_table:
                continue
            if not line.strip().startswith("|"):
                if rows:
                    break
                continue
            cells = [cell.strip().strip("*") for cell in line.strip().strip("|").split("|")]
            if len(cells) < 5 or cells[0] == "Model" or cells[1].startswith("-"):
                continue
            rows.append(
                {
                    "model": cells[0],
                    "mae": float(cells[1]),
                    "rmse": float(cells[2]),
                    "mape": float(cells[3]),
                    "r2": float(cells[4]),
                }
            )

    index = {row["model"]: i for i, row in enumerate(rows)}
    for key in ["A", "B", "C", "D"]:
        metrics_path = project_root / "data" / f"model_{key}_outputs" / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = pd.read_json(metrics_path, typ="series").to_dict()
        row = {
            "model": key,
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "mape": metrics.get("mape"),
            "r2": metrics.get("r2"),
        }
        if key in index:
            rows[index[key]] = row
        else:
            rows.append(row)
            index[key] = len(rows) - 1

    tuned_path = project_root / "data" / "model_C_tuned_outputs" / "metrics.json"
    if tuned_path.exists():
        metrics = pd.read_json(tuned_path, typ="series").to_dict()
        row = {
            "model": "C_tuned",
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "mape": metrics.get("mape"),
            "r2": metrics.get("r2"),
        }
        if "C_tuned" in index:
            rows[index["C_tuned"]] = row
        else:
            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = run_ml_baseline_sweep()
    print(result.head(12).to_string(index=False))
