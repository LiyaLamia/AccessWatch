#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False
    LGBMClassifier = None
    LGBMRegressor = None


# Identifier columns shared across the modeling tables
ID_COLS = ["raion_id", "raion_name", "oblast_name", "week_start"]

# For the naive baseline, map each prediction target to the current-week
# source signal that will simply be carried forward.
TARGET_MAP_NAIVE = {
    "y_next_high_intensity": "high_intensity_week",
    "y_next_any_event": "any_event",
    "y_next_event_count": "acled_event_count",
    "y_next_fatalities_sum": "fatalities_sum",
    "y_next_air_drone_strike_count": "air_drone_strike_count",
}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training and comparing
    Model 1, Model 2, and an optional naive baseline.
    """
    p = argparse.ArgumentParser(
        description=(
            "Train Model 1 (multimodal non-ACLED only) and Model 2 (multimodal + lagged ACLED history) "
            "using time-based splits. Supports binary classification and regression targets."
        )
    )
    p.add_argument("--master_csv", required=True, help="Path to master_raion_week_modeling.csv")
    p.add_argument("--model1_csv", required=True, help="Path to model1_non_acled_only.csv")
    p.add_argument("--model2_csv", required=True, help="Path to model2_plus_lagged_acled.csv")
    p.add_argument("--feature_spec_json", required=True, help="Path to model_feature_spec.json from build step")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--target_col", default="y_next_high_intensity", help="Target column to train on")
    p.add_argument(
        "--task_type",
        default="auto",
        choices=["auto", "classification", "regression"],
        help="Auto-detect classification vs regression from target values by default",
    )
    p.add_argument("--valid_weeks", type=int, default=13, help="Number of trailing weeks for validation")
    p.add_argument("--test_weeks", type=int, default=13, help="Number of trailing weeks for final test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_k", type=int, default=10, help="Top-k ranking metric per week for classification")
    p.add_argument("--top_frac", type=float, default=0.20, help="Top fraction ranking metric per week for classification")
    p.add_argument("--skip_naive", action="store_true", help="Skip naive persistence baseline")
    p.add_argument(
        "--force_fallback_tree",
        action="store_true",
        help="Use sklearn HistGradientBoosting even if lightgbm is installed",
    )
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    """
    Create an output directory if it does not already exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """
    Set NumPy random seed for reproducibility.
    """
    np.random.seed(seed)


def load_feature_spec(path: str) -> Dict[str, Any]:
    """
    Load the feature-spec JSON. Some files store the actual spec
    under a top-level 'feature_spec' key, so support both layouts.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "feature_spec" in payload:
        return payload["feature_spec"]
    return payload


def detect_task_type(y: pd.Series, arg_task_type: str) -> str:
    """
    Auto-detect whether the target should be treated as
    classification or regression.
    """
    if arg_task_type != "auto":
        return arg_task_type

    vals = pd.Series(y).dropna().unique().tolist()
    vals = sorted(vals)

    # If all non-missing values are binary, treat it as classification
    if vals and set(vals).issubset({0, 1, 0.0, 1.0}):
        return "classification"
    return "regression"


def assign_time_splits(df: pd.DataFrame, valid_weeks: int, test_weeks: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create train/valid/test splits using trailing weeks.
    The test split is the most recent block, validation is the block before that,
    and everything earlier becomes training.
    """
    out = df.copy()
    out["week_start"] = pd.to_datetime(out["week_start"])
    weeks = sorted(pd.to_datetime(out["week_start"]).dropna().unique())

    if len(weeks) <= (valid_weeks + test_weeks + 4):
        raise ValueError(
            f"Not enough distinct weeks ({len(weeks)}) for valid_weeks={valid_weeks} and test_weeks={test_weeks}."
        )

    test_start = weeks[-test_weeks]
    valid_start = weeks[-(test_weeks + valid_weeks)]

    out["split"] = np.where(
        out["week_start"] >= test_start,
        "test",
        np.where(out["week_start"] >= valid_start, "valid", "train"),
    )

    split_meta = {
        "n_unique_weeks": len(weeks),
        "train_week_start_min": str(weeks[0].date()),
        "train_week_start_max": str((valid_start - pd.Timedelta(days=7)).date()),
        "valid_week_start_min": str(valid_start.date()),
        "valid_week_start_max": str((test_start - pd.Timedelta(days=7)).date()),
        "test_week_start_min": str(test_start.date()),
        "test_week_start_max": str(weeks[-1].date()),
        "n_train_weeks": int(sum(w < valid_start for w in weeks)),
        "n_valid_weeks": int(sum((w >= valid_start) and (w < test_start) for w in weeks)),
        "n_test_weeks": int(sum(w >= test_start for w in weeks)),
    }
    return out, split_meta


def get_feature_sets(feature_spec: Dict[str, Any], model1_df: pd.DataFrame, model2_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Get the feature lists for Model 1 and Model 2 from the feature spec.
    If the spec is missing them, fall back to inferring features directly
    from the table columns.
    """
    m1 = feature_spec.get("model1_non_acled_only_features", [])
    m2 = feature_spec.get("model2_plus_lagged_acled_features", [])

    if not m1:
        m1 = [c for c in model1_df.columns if c not in ID_COLS and not c.startswith("y_next_") and c != "target_week_start"]
    if not m2:
        m2 = [c for c in model2_df.columns if c not in ID_COLS and not c.startswith("y_next_") and c != "target_week_start"]

    m1 = [c for c in m1 if c in model1_df.columns]
    m2 = [c for c in m2 if c in model2_df.columns]
    return sorted(m1), sorted(m2)


def to_numeric_frame(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    """
    Convert all feature columns to numeric, coercing bad values to NaN.
    """
    X = df[list(feature_cols)].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def split_xy(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Dict[str, Any]:
    """
    Prepare X, y, metadata, and boolean masks for train/valid/test.
    Rows with missing target values are dropped here.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    X = to_numeric_frame(df, feature_cols)
    y = pd.to_numeric(df[target_col], errors="coerce")

    mask = y.notna()
    work = df.loc[mask].copy()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    meta = work[[c for c in ID_COLS if c in work.columns] + ["split"]].reset_index(drop=True)

    idx_train = meta["split"] == "train"
    idx_valid = meta["split"] == "valid"
    idx_test = meta["split"] == "test"

    return {
        "X": X,
        "y": y,
        "meta": meta,
        "idx_train": idx_train.to_numpy(),
        "idx_valid": idx_valid.to_numpy(),
        "idx_test": idx_test.to_numpy(),
    }


def build_linear_model(task_type: str, seed: int) -> Pipeline:
    """
    Build the linear baseline model:
    logistic regression for classification, ridge for regression.
    """
    if task_type == "classification":
        estimator = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        )
    else:
        estimator = Ridge(alpha=1.0, random_state=seed)

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("estimator", estimator),
        ]
    )


def build_tree_model(task_type: str, seed: int, force_fallback_tree: bool) -> Any:
    """
    Build the tree-based model. Prefer LightGBM when available,
    unless fallback sklearn boosting is explicitly requested.
    """
    if HAS_LIGHTGBM and not force_fallback_tree:
        if task_type == "classification":
            return LGBMClassifier(
                objective="binary",
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                random_state=seed,
                is_unbalance=True,
                n_jobs=-1,
            )
        return LGBMRegressor(
            objective="regression",
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=seed,
            n_jobs=-1,
        )

    # Fallback tree models from sklearn
    if task_type == "classification":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "estimator",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=400,
                        max_depth=None,
                        min_samples_leaf=20,
                        random_state=seed,
                    ),
                ),
            ]
        )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "estimator",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=400,
                    max_depth=None,
                    min_samples_leaf=20,
                    random_state=seed,
                ),
            ),
        ]
    )


def get_scores(model: Any, X: pd.DataFrame, task_type: str) -> np.ndarray:
    """
    Return model scores in a consistent format:
    probabilities for classification, predictions for regression.
    """
    if task_type == "classification":
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        if hasattr(model, "decision_function"):
            raw = model.decision_function(X)
            return 1.0 / (1.0 + np.exp(-raw))
        pred = model.predict(X)
        return np.asarray(pred, dtype=float)

    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def choose_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Choose the classification threshold that gives the best F1
    on the validation set.
    """
    candidates = np.linspace(0.05, 0.95, 19)
    best_t = 0.5
    best_f1 = -1.0

    for t in candidates:
        y_pred = (y_score >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)

    return best_t


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    top_k: int,
    top_frac: float,
    meta: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Compute standard binary classification metrics and,
    when metadata is available, also compute week-level ranking metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    out: Dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "avg_precision": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(y_pred)),
        "n_rows": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }

    if meta is not None and len(meta) == len(y_true):
        wk = meta.copy()
        wk["y_true"] = y_true
        wk["y_score"] = y_score
        out.update(compute_ranking_metrics(wk, top_k=top_k, top_frac=top_frac))

    return out


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Compute standard regression metrics.
    """
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_target": float(np.mean(y_true)),
        "mean_prediction": float(np.mean(y_pred)),
        "n_rows": int(len(y_true)),
    }


def compute_ranking_metrics(df: pd.DataFrame, top_k: int, top_frac: float) -> Dict[str, Any]:
    """
    Compute week-wise top-k and top-fraction precision/recall
    for classification outputs.
    """
    if df.empty:
        return {}

    total_positives = int(df["y_true"].sum())
    if total_positives == 0:
        return {
            f"top_{top_k}_precision": float("nan"),
            f"top_{top_k}_recall": float("nan"),
            f"top_{int(top_frac * 100)}pct_precision": float("nan"),
            f"top_{int(top_frac * 100)}pct_recall": float("nan"),
        }

    selected_k = []
    selected_frac = []

    for _, grp in df.groupby("week_start", sort=False):
        grp = grp.sort_values("y_score", ascending=False)

        selected_k.append(grp.head(top_k))

        n_frac = max(1, int(math.ceil(len(grp) * top_frac)))
        selected_frac.append(grp.head(n_frac))

    topk_df = pd.concat(selected_k, ignore_index=True)
    topf_df = pd.concat(selected_frac, ignore_index=True)

    return {
        f"top_{top_k}_precision": float(topk_df["y_true"].mean()),
        f"top_{top_k}_recall": float(topk_df["y_true"].sum() / total_positives),
        f"top_{int(top_frac * 100)}pct_precision": float(topf_df["y_true"].mean()),
        f"top_{int(top_frac * 100)}pct_recall": float(topf_df["y_true"].sum() / total_positives),
    }


def get_feature_importance(
    model: Any,
    feature_cols: List[str],
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    task_type: str
) -> pd.DataFrame:
    """
    Extract feature importance in a few different ways depending on the model:
    coefficients for linear models, native feature importance for tree models,
    or permutation importance as a fallback.
    """
    # Pipelines expose the final estimator differently
    estimator = model.named_steps.get("estimator") if hasattr(model, "named_steps") and "estimator" in model.named_steps else model

    if hasattr(estimator, "coef_"):
        coef = np.ravel(estimator.coef_)
        out = pd.DataFrame({
            "feature": feature_cols,
            "importance": coef,
            "abs_importance": np.abs(coef),
        }).sort_values("abs_importance", ascending=False)
        return out

    if hasattr(estimator, "feature_importances_"):
        imp = np.ravel(estimator.feature_importances_)
        out = pd.DataFrame({
            "feature": feature_cols,
            "importance": imp,
            "abs_importance": np.abs(imp),
        }).sort_values("abs_importance", ascending=False)
        return out

    # Final fallback: permutation importance on a validation sample
    if len(X_valid) > 0:
        n_sample = min(3000, len(X_valid))
        sample_idx = np.arange(n_sample)
        Xs = X_valid.iloc[sample_idx]
        ys = y_valid.iloc[sample_idx]
        scoring = "average_precision" if task_type == "classification" else "neg_root_mean_squared_error"

        result = permutation_importance(model, Xs, ys, n_repeats=5, random_state=42, scoring=scoring)
        out = pd.DataFrame({
            "feature": feature_cols,
            "importance": result.importances_mean,
            "abs_importance": np.abs(result.importances_mean),
        }).sort_values("abs_importance", ascending=False)
        return out

    return pd.DataFrame(columns=["feature", "importance", "abs_importance"])


def serialize_value(v: Any) -> Any:
    """
    Convert NumPy/pandas scalar types into plain Python values
    so they can be safely written to JSON.
    """
    if isinstance(v, (np.integer, np.int64, np.int32)):
        return int(v)
    if isinstance(v, (np.floating, np.float64, np.float32)):
        return float(v)
    if isinstance(v, (pd.Timestamp,)):
        return str(v)
    return v


def save_json(obj: Dict[str, Any], path: Path) -> None:
    """
    Save a JSON payload after normalizing values into JSON-safe types.
    """
    clean = json.loads(json.dumps(obj, default=serialize_value))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)


def save_predictions(
    meta: pd.DataFrame,
    y_true: pd.Series,
    y_score: np.ndarray,
    task_type: str,
    threshold: Optional[float],
    path: Path
) -> None:
    """
    Save row-level predictions for one split.
    """
    out = meta.copy()
    out["y_true"] = np.asarray(y_true)

    if task_type == "classification":
        out["y_score"] = y_score
        out["y_pred"] = (y_score >= float(threshold)).astype(int)
    else:
        out["y_pred"] = y_score

    out.to_csv(path, index=False)


def train_one_experiment(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    task_type: str,
    seed: int,
    outdir: Path,
    top_k: int,
    top_frac: float,
    force_fallback_tree: bool,
) -> Dict[str, Any]:
    """
    Train and evaluate one experiment configuration,
    comparing a linear model and a tree-based model.
    """
    ensure_dir(outdir)

    payload = split_xy(df, feature_cols, target_col)
    X = payload["X"]
    y = payload["y"]
    meta = payload["meta"]

    idx_train = payload["idx_train"]
    idx_valid = payload["idx_valid"]
    idx_test = payload["idx_test"]

    X_train, y_train = X.loc[idx_train], y.loc[idx_train]
    X_valid, y_valid = X.loc[idx_valid], y.loc[idx_valid]
    X_test, y_test = X.loc[idx_test], y.loc[idx_test]
    meta_train, meta_valid, meta_test = meta.loc[idx_train], meta.loc[idx_valid], meta.loc[idx_test]

    if len(X_train) == 0 or len(X_valid) == 0 or len(X_test) == 0:
        raise ValueError("Train/valid/test split produced an empty subset. Adjust the date split parameters.")

    models = {
        "linear": build_linear_model(task_type, seed),
        "tree": build_tree_model(task_type, seed, force_fallback_tree),
    }

    results: Dict[str, Any] = {
        "task_type": task_type,
        "target_col": target_col,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "n_test": int(len(X_test)),
    }

    for model_name, model in models.items():
        model_dir = outdir / model_name
        ensure_dir(model_dir)

        fitted = clone(model)
        fitted.fit(X_train, y_train)

        if task_type == "classification":
            train_score = get_scores(fitted, X_train, task_type)
            valid_score = get_scores(fitted, X_valid, task_type)
            test_score = get_scores(fitted, X_test, task_type)

            # Tune threshold using the validation split
            threshold = choose_best_threshold(y_valid.to_numpy(), valid_score)

            metrics = {
                "train": compute_classification_metrics(y_train.to_numpy(), train_score, threshold, top_k, top_frac, meta_train),
                "valid": compute_classification_metrics(y_valid.to_numpy(), valid_score, threshold, top_k, top_frac, meta_valid),
                "test": compute_classification_metrics(y_test.to_numpy(), test_score, threshold, top_k, top_frac, meta_test),
            }

            save_predictions(meta_train, y_train, train_score, task_type, threshold, model_dir / "train_predictions.csv")
            save_predictions(meta_valid, y_valid, valid_score, task_type, threshold, model_dir / "valid_predictions.csv")
            save_predictions(meta_test, y_test, test_score, task_type, threshold, model_dir / "test_predictions.csv")
        else:
            train_pred = get_scores(fitted, X_train, task_type)
            valid_pred = get_scores(fitted, X_valid, task_type)
            test_pred = get_scores(fitted, X_test, task_type)
            threshold = None

            metrics = {
                "train": compute_regression_metrics(y_train.to_numpy(), train_pred),
                "valid": compute_regression_metrics(y_valid.to_numpy(), valid_pred),
                "test": compute_regression_metrics(y_test.to_numpy(), test_pred),
            }

            save_predictions(meta_train, y_train, train_pred, task_type, threshold, model_dir / "train_predictions.csv")
            save_predictions(meta_valid, y_valid, valid_pred, task_type, threshold, model_dir / "valid_predictions.csv")
            save_predictions(meta_test, y_test, test_pred, task_type, threshold, model_dir / "test_predictions.csv")

        fi = get_feature_importance(fitted, feature_cols, X_valid, y_valid, task_type)
        fi.to_csv(model_dir / "feature_importance.csv", index=False)
        save_json(metrics, model_dir / "metrics.json")

        results[model_name] = {
            "metrics": metrics,
            "best_threshold": threshold,
        }

    return results


def run_naive_baseline(
    master_df: pd.DataFrame,
    target_col: str,
    task_type: str,
    top_k: int,
    top_frac: float,
    outdir: Path
) -> Optional[Dict[str, Any]]:
    """
    Run a simple naive persistence baseline using the mapped
    current-week source signal as the prediction.
    """
    if target_col not in TARGET_MAP_NAIVE:
        return None

    source_col = TARGET_MAP_NAIVE[target_col]
    if source_col not in master_df.columns:
        return None

    ensure_dir(outdir)

    work = master_df.copy()
    work["week_start"] = pd.to_datetime(work["week_start"])
    work = work[work[target_col].notna()].copy()

    meta = work[[c for c in ID_COLS if c in work.columns] + ["split"]].copy()
    y_true = pd.to_numeric(work[target_col], errors="coerce")
    y_src = pd.to_numeric(work[source_col], errors="coerce").fillna(0)

    results: Dict[str, Any] = {"source_col": source_col}

    for split_name in ["train", "valid", "test"]:
        mask = work["split"] == split_name

        if task_type == "classification":
            metrics = compute_classification_metrics(
                y_true.loc[mask].to_numpy(),
                y_src.loc[mask].to_numpy(),
                threshold=0.5,
                top_k=top_k,
                top_frac=top_frac,
                meta=meta.loc[mask],
            )
            pred_df = meta.loc[mask].copy()
            pred_df["y_true"] = y_true.loc[mask].to_numpy()
            pred_df["y_score"] = y_src.loc[mask].to_numpy()
            pred_df["y_pred"] = (y_src.loc[mask].to_numpy() >= 0.5).astype(int)
        else:
            metrics = compute_regression_metrics(y_true.loc[mask].to_numpy(), y_src.loc[mask].to_numpy())
            pred_df = meta.loc[mask].copy()
            pred_df["y_true"] = y_true.loc[mask].to_numpy()
            pred_df["y_pred"] = y_src.loc[mask].to_numpy()

        pred_df.to_csv(outdir / f"{split_name}_predictions.csv", index=False)
        results[split_name] = metrics

    save_json(results, outdir / "metrics.json")
    return results


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    master = pd.read_csv(args.master_csv)
    model1 = pd.read_csv(args.model1_csv)
    model2 = pd.read_csv(args.model2_csv)
    feature_spec = load_feature_spec(args.feature_spec_json)

    # Create time-based splits from the master table, then join them into model tables
    master, split_meta = assign_time_splits(master, args.valid_weeks, args.test_weeks)
    model1 = model1.merge(
        master[[c for c in ID_COLS if c in master.columns] + ["split"]],
        on=[c for c in ID_COLS if c in model1.columns],
        how="left"
    )
    model2 = model2.merge(
        master[[c for c in ID_COLS if c in master.columns] + ["split"]],
        on=[c for c in ID_COLS if c in model2.columns],
        how="left"
    )

    feature_cols_model1, feature_cols_model2 = get_feature_sets(feature_spec, model1, model2)

    if args.target_col not in master.columns:
        raise ValueError(f"Target column not found in master table: {args.target_col}")

    task_type = detect_task_type(pd.to_numeric(master[args.target_col], errors="coerce").dropna(), args.task_type)

    summary: Dict[str, Any] = {
        "target_col": args.target_col,
        "task_type": task_type,
        "split_meta": split_meta,
        "lightgbm_available": HAS_LIGHTGBM,
        "force_fallback_tree": bool(args.force_fallback_tree),
        "valid_weeks": int(args.valid_weeks),
        "test_weeks": int(args.test_weeks),
        "feature_counts": {
            "model1": len(feature_cols_model1),
            "model2": len(feature_cols_model2),
        },
    }
    save_json(summary, outdir / "run_summary.json")

    if not args.skip_naive:
        naive = run_naive_baseline(master, args.target_col, task_type, args.top_k, args.top_frac, outdir / "naive_baseline")
    else:
        naive = None

    model1_results = train_one_experiment(
        df=model1,
        feature_cols=feature_cols_model1,
        target_col=args.target_col,
        task_type=task_type,
        seed=args.seed,
        outdir=outdir / "model1_non_acled_only",
        top_k=args.top_k,
        top_frac=args.top_frac,
        force_fallback_tree=args.force_fallback_tree,
    )

    model2_results = train_one_experiment(
        df=model2,
        feature_cols=feature_cols_model2,
        target_col=args.target_col,
        task_type=task_type,
        seed=args.seed,
        outdir=outdir / "model2_plus_lagged_acled",
        top_k=args.top_k,
        top_frac=args.top_frac,
        force_fallback_tree=args.force_fallback_tree,
    )

    # Build a flat comparison table across all experiments/splits
    comparison_rows = []

    if naive is not None:
        for split_name in ["train", "valid", "test"]:
            row = {"experiment": "naive_baseline", "algorithm": "naive", "split": split_name}
            row.update(naive.get(split_name, {}))
            comparison_rows.append(row)

    for exp_name, result in [
        ("model1_non_acled_only", model1_results),
        ("model2_plus_lagged_acled", model2_results),
    ]:
        for alg in ["linear", "tree"]:
            metrics = result[alg]["metrics"]
            for split_name, vals in metrics.items():
                row = {"experiment": exp_name, "algorithm": alg, "split": split_name}
                row.update(vals)
                comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(outdir / "comparison_metrics.csv", index=False)

    final_payload = {
        "summary": summary,
        "naive_baseline": naive,
        "model1_non_acled_only": model1_results,
        "model2_plus_lagged_acled": model2_results,
    }
    save_json(final_payload, outdir / "all_results.json")

    print(f"Saved outputs to: {outdir}")
    print(f"Task type: {task_type}")
    print(f"Target: {args.target_col}")
    print("\nComparison metrics (test rows):")

    if not comparison_df.empty:
        test_view = comparison_df[comparison_df["split"] == "test"].copy()
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(test_view.to_string(index=False))


if __name__ == "__main__":
    main()