#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
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

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


# Core identifiers shared by all weekly modeling tables
ID_COLS = ["raion_id", "raion_name", "oblast_name", "week_start"]

# Friendly names for the forecast horizons used in output folders/files
DISPLAY_LABELS = {1: "next_week", 2: "next_2_weeks", 3: "next_3_weeks", 4: "next_4_weeks_month"}

# Raw current-week context columns that are useful to keep in exports
RAW_CONTEXT_COLS = [
    "high_intensity_week", "any_event", "acled_event_count", "fatalities_sum", "air_drone_strike_count",
    "battles_count", "explosions_remote_count", "violence_against_civilians_count", "strategic_developments_count",
    "protests_riots_count", "events_with_fatalities", "civilian_targeting_count", "precise_geo_event_count",
]

# Main hierarchy tasks: parent event occurrence, severity, and a few count targets
MAIN_TARGETS = {
    "any_event": {"source_col": "any_event", "agg": "max", "task_type": "classification"},
    "high_intensity": {"source_col": "high_intensity_week", "agg": "max", "task_type": "classification"},
    "event_count": {"source_col": "acled_event_count", "agg": "sum", "task_type": "regression"},
    "fatalities_sum": {"source_col": "fatalities_sum", "agg": "sum", "task_type": "regression"},
    "air_drone_strike_count": {"source_col": "air_drone_strike_count", "agg": "sum", "task_type": "regression"},
}

# Subtype tasks are all binary "did this happen in the future window?" targets
SUBTYPE_TARGETS = {
    "battle_any": {"source_col": "battles_count", "agg": "max", "task_type": "classification"},
    "explosions_remote_any": {"source_col": "explosions_remote_count", "agg": "max", "task_type": "classification"},
    "violence_against_civilians_any": {"source_col": "violence_against_civilians_count", "agg": "max", "task_type": "classification"},
    "air_drone_any": {"source_col": "air_drone_strike_count", "agg": "max", "task_type": "classification"},
    "strategic_developments_any": {"source_col": "strategic_developments_count", "agg": "max", "task_type": "classification"},
}

# Fixed subtype ordering so outputs stay consistent across windows/models
SUBTYPE_ORDER = list(SUBTYPE_TARGETS.keys())

# Human-readable subtype names for exported summaries
SUBTYPE_DISPLAY = {
    "battle_any": "battle",
    "explosions_remote_any": "explosions_remote",
    "violence_against_civilians_any": "violence_against_civilians",
    "air_drone_any": "air_drone",
    "strategic_developments_any": "strategic_developments",
}

# Rough prefixes used to separate static vs dynamic features for sequence models
STATIC_PREFIXES = (
    "road_", "rail_", "major_road_", "nearest_border_", "border_crossing", "paved_",
    "place_", "places_", "pop_", "unosat_",
)
DYNAMIC_PREFIXES = ("firms_", "ntl_", "acled_", "fatalities_", "air_drone_", "any_event", "high_intensity")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the hierarchical multiwindow pipeline.
    """
    p = argparse.ArgumentParser(
        description=(
            "Hierarchy-consistent multiwindow forecasting. any_event is the parent task; high_intensity and event-subtype "
            "probabilities are constrained to be <= any_event. Count targets use hurdle-style predictions. "
            "Adds a TCN neural model alongside linear/lightgbm/catboost/GRU."
        )
    )
    p.add_argument("--master_csv", required=True)
    p.add_argument("--model1_csv", required=True)
    p.add_argument("--model2_csv", required=True)
    p.add_argument("--feature_spec_json", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--window_weeks", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--valid_weeks", type=int, default=13)
    p.add_argument("--test_weeks", type=int, default=13)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--top_frac", type=float, default=0.2)
    p.add_argument("--skip_lightgbm", action="store_true")
    p.add_argument("--skip_catboost", action="store_true")
    p.add_argument("--skip_gru", action="store_true")
    p.add_argument("--skip_tcn", action="store_true")
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def choose_device(arg: str):
    """
    Pick the torch device. If torch is unavailable, return None.
    """
    if not HAS_TORCH:
        return None
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the key ID columns before merging or splitting.
    """
    out = df.copy()
    for c in ["raion_id", "raion_name", "oblast_name"]:
        if c in out.columns:
            out[c] = out[c].astype("string")
    if "week_start" in out.columns:
        out["week_start"] = pd.to_datetime(out["week_start"], errors="coerce")
    return out


def load_feature_spec(path: str) -> Dict[str, Any]:
    """
    Load the feature spec JSON. Some files wrap the real payload
    under a top-level 'feature_spec' key.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("feature_spec", payload)


def get_feature_sets(feature_spec: Dict[str, Any], model1_df: pd.DataFrame, model2_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Get Model 1 and Model 2 feature lists from the feature spec.
    Fall back to inferring them from the table columns if needed.
    """
    m1 = feature_spec.get("model1_non_acled_only_features", [])
    m2 = feature_spec.get("model2_plus_lagged_acled_features", [])
    if not m1:
        m1 = [c for c in model1_df.columns if c not in ID_COLS and c != "split" and not c.startswith("y_")]
    if not m2:
        m2 = [c for c in model2_df.columns if c not in ID_COLS and c != "split" and not c.startswith("y_")]
    return sorted([c for c in m1 if c in model1_df.columns]), sorted([c for c in m2 if c in model2_df.columns])


def assign_time_splits(df: pd.DataFrame, valid_weeks: int, test_weeks: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create trailing train/validation/test splits based on week_start.
    """
    out = normalize_ids(df)
    weeks = sorted(pd.to_datetime(out["week_start"]).dropna().unique())
    if len(weeks) <= valid_weeks + test_weeks + 4:
        raise ValueError(f"Not enough distinct weeks ({len(weeks)}) for valid/test split")

    test_start = weeks[-test_weeks]
    valid_start = weeks[-(test_weeks + valid_weeks)]

    out["split"] = np.where(
        out["week_start"] >= test_start,
        "test",
        np.where(out["week_start"] >= valid_start, "valid", "train")
    )

    meta = {
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
    return out, meta


def build_future_window(series: pd.Series, weeks: int, agg: str) -> pd.Series:
    """
    Build a future-looking target over the next N weeks.
    """
    future = pd.concat([series.shift(-i) for i in range(1, weeks + 1)], axis=1)
    incomplete = future.isna().any(axis=1)
    out = future.max(axis=1) if agg == "max" else future.sum(axis=1)
    out[incomplete] = np.nan
    return out


def build_trailing_window(series: pd.Series, weeks: int, agg: str) -> pd.Series:
    """
    Build a trailing window version of the source signal
    for the naive baseline.
    """
    trailing = pd.concat([series.shift(i) for i in range(0, weeks)], axis=1)
    return trailing.max(axis=1, skipna=True) if agg == "max" else trailing.sum(axis=1, skipna=True)


def binary_or_nan(values: pd.Series) -> pd.Series:
    """
    Convert numeric counts to binary 0/1 while preserving missing values.
    """
    vals = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.where(vals.notna(), (vals > 0).astype(float), np.nan), index=values.index)


def make_window_master(master: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """
    Build one horizon-specific master table containing all main targets,
    subtype targets, naive baseline sources, and target-window metadata.
    """
    out = normalize_ids(master).sort_values(["raion_id", "week_start"]).copy()
    g = out.groupby("raion_id", sort=False)

    for name, cfg in MAIN_TARGETS.items():
        src = cfg["source_col"]

        # Actual future target over the next N weeks
        out[f"actual_{name}"] = g[src].transform(
            lambda s: build_future_window(pd.to_numeric(s, errors="coerce"), weeks, cfg["agg"])
        )
        if cfg["task_type"] == "classification":
            out[f"actual_{name}"] = binary_or_nan(out[f"actual_{name}"])

        # Horizon-aligned source for naive persistence
        out[f"naive_source_{name}"] = g[src].transform(
            lambda s: build_trailing_window(pd.to_numeric(s, errors="coerce"), weeks, cfg["agg"])
        )
        if cfg["task_type"] == "classification":
            out[f"naive_source_{name}"] = (
                pd.to_numeric(out[f"naive_source_{name}"], errors="coerce").fillna(0) > 0
            ).astype(float)

    for name, cfg in SUBTYPE_TARGETS.items():
        src = cfg["source_col"]

        out[f"actual_{name}"] = g[src].transform(
            lambda s: build_future_window(pd.to_numeric(s, errors="coerce"), weeks, cfg["agg"])
        )
        out[f"actual_{name}"] = binary_or_nan(out[f"actual_{name}"])

        out[f"naive_source_{name}"] = g[src].transform(
            lambda s: build_trailing_window(pd.to_numeric(s, errors="coerce"), weeks, cfg["agg"])
        )
        out[f"naive_source_{name}"] = (
            pd.to_numeric(out[f"naive_source_{name}"], errors="coerce").fillna(0) > 0
        ).astype(float)

        # Also keep future subtype counts for reporting/analysis
        out[f"actual_{name.replace('_any', '_count')}"] = g[src].transform(
            lambda s: build_future_window(pd.to_numeric(s, errors="coerce"), weeks, "sum")
        )

    # Fatalities-any is treated as an extra binary severity helper target
    out["actual_fatalities_any"] = binary_or_nan(out["actual_fatalities_sum"])
    out["naive_source_fatalities_any"] = (
        pd.to_numeric(out["naive_source_fatalities_sum"], errors="coerce").fillna(0) > 0
    ).astype(float)

    out["target_window_start"] = out["week_start"] + pd.Timedelta(days=7)
    out["target_window_end"] = out["week_start"] + pd.Timedelta(days=7 * weeks + 6)
    out["forecast_window_weeks"] = weeks
    out["forecast_window_label"] = DISPLAY_LABELS.get(weeks, f"next_{weeks}_weeks")

    # Drop rows where the future target window is incomplete
    return out[out["actual_any_event"].notna()].copy()


def attach_context(master_h: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the horizon-specific targets/context from the window master
    back onto one of the feature tables.
    """
    keys = [c for c in ID_COLS if c in model_df.columns and c in master_h.columns]
    model_df = normalize_ids(model_df)

    # If both frames contain the same non-key columns, prefer the horizon-master version
    overlap = [c for c in master_h.columns if c in model_df.columns and c not in keys]
    if overlap:
        model_df = model_df.drop(columns=overlap)

    return model_df.merge(master_h, on=keys, how="left", validate="many_to_one")


def infer_dynamic_static(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    """
    Heuristically split features into dynamic vs static groups
    for the sequence models.
    """
    dynamic, static = [], []
    work = df[["raion_id", *feature_cols]].copy()

    for col in feature_cols:
        lo = col.lower()
        if lo.startswith(DYNAMIC_PREFIXES) or "_lag" in lo or "_roll" in lo:
            dynamic.append(col)
            continue
        if lo.startswith(STATIC_PREFIXES):
            static.append(col)
            continue

        # If a feature is constant within almost every raion, treat it as static
        nunique = work.groupby("raion_id", dropna=False)[col].nunique(dropna=False)
        frac_constant = float((nunique <= 1).mean()) if len(nunique) else 0.0
        (static if frac_constant >= 0.95 else dynamic).append(col)

    if not dynamic:
        dynamic = feature_cols.copy()
        static = []

    return sorted(dynamic), sorted(static)


def to_numeric_frame(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """
    Convert feature columns to numeric form, coercing invalid values to NaN.
    """
    X = df[list(feature_cols)].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


class ConstantClassifier:
    """
    Fallback classifier for degenerate cases where the training labels
    contain only one class.
    """
    def __init__(self, prob: float):
        self.prob = float(prob)

    def predict_proba(self, X):
        p = np.full(len(X), self.prob, dtype=float)
        return np.column_stack([1 - p, p])


class ConstantRegressor:
    """
    Fallback regressor for degenerate cases where the target is constant.
    """
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def build_classifier(family: str, seed: int):
    """
    Build one of the supported tabular classifiers.
    """
    if family == "linear":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("estimator", LogisticRegression(max_iter=2500, solver="liblinear", class_weight="balanced", random_state=seed))
        ])
    if family == "lightgbm":
        return LGBMClassifier(
            objective="binary", n_estimators=500, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            random_state=seed, is_unbalance=True, n_jobs=-1, verbose=-1
        )
    if family == "catboost":
        return CatBoostClassifier(
            loss_function="Logloss", iterations=500, learning_rate=0.03, depth=6,
            eval_metric="AUC", random_seed=seed, verbose=False, allow_writing_files=False
        )
    raise ValueError(family)


def build_regressor(family: str, seed: int):
    """
    Build one of the supported count regressors.
    """
    if family == "linear":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("estimator", Ridge(alpha=1.0, random_state=seed))
        ])
    if family == "lightgbm":
        return LGBMRegressor(
            objective="poisson", n_estimators=600, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            random_state=seed, n_jobs=-1, verbose=-1
        )
    if family == "catboost":
        return CatBoostRegressor(
            loss_function="Poisson", iterations=600, learning_rate=0.03, depth=6,
            random_seed=seed, verbose=False, allow_writing_files=False
        )
    raise ValueError(family)


def fit_binary_model(X: pd.DataFrame, y: pd.Series, family: str, seed: int):
    """
    Fit a binary classifier, or fall back to a constant classifier
    if the labels are single-class.
    """
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2:
        return ConstantClassifier(float(y.iloc[0] if len(y) else 0.0))

    m = build_classifier(family, seed)
    m.fit(X, y)
    return m


def predict_binary(model, X: pd.DataFrame) -> np.ndarray:
    """
    Get binary probabilities from a fitted classifier.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1].astype(float)
    return np.clip(np.asarray(model.predict(X), dtype=float), 0.0, 1.0)


def fit_positive_regressor(X: pd.DataFrame, y: pd.Series, family: str, seed: int):
    """
    Fit a positive-valued regressor on log1p(target), or fall back
    to a constant regressor if the target is effectively constant.
    """
    y_pos = np.log1p(pd.to_numeric(y, errors="coerce").fillna(0).clip(lower=0))
    if len(y_pos) == 0 or float(np.nanstd(y_pos)) < 1e-12:
        return ConstantRegressor(float(np.nanmean(y_pos) if len(y_pos) else 0.0))

    m = build_regressor(family, seed)
    m.fit(X, y_pos)
    return m


def predict_positive_regressor(model, X: pd.DataFrame) -> np.ndarray:
    """
    Predict positive counts by inverting the log1p transform
    and clipping at zero.
    """
    return np.clip(np.expm1(np.asarray(model.predict(X), dtype=float)), 0.0, None)


def sweep_threshold(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Optional[Sequence[float]] = None) -> float:
    """
    Pick the threshold that gives the best F1 on the validation set.
    """
    thresholds = thresholds or np.arange(0.10, 0.91, 0.05)
    if len(np.unique(y_true)) < 2:
        return 0.5

    best_t, best_f = 0.5, -1.0
    for t in thresholds:
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f + 1e-12:
            best_f, best_t = f, float(t)
    return best_t


def ranking_metrics_by_week(meta: pd.DataFrame, scores: np.ndarray, y_true: np.ndarray, top_k: int, top_frac: float) -> Dict[str, float]:
    """
    Compute week-wise ranking metrics for prioritization use cases.
    """
    wk = meta[["week_start"]].copy()
    wk["score"] = scores
    wk["y_true"] = y_true

    p10 = []
    r10 = []
    p20 = []
    r20 = []

    for _, grp in wk.groupby("week_start"):
        grp = grp.sort_values("score", ascending=False)
        positives = max(int(grp["y_true"].sum()), 1)

        topk = grp.head(min(top_k, len(grp)))
        tp = float(topk["y_true"].sum())
        p10.append(tp / max(len(topk), 1))
        r10.append(tp / positives)

        topf = grp.head(max(int(np.ceil(len(grp) * top_frac)), 1))
        tp2 = float(topf["y_true"].sum())
        p20.append(tp2 / max(len(topf), 1))
        r20.append(tp2 / positives)

    return {
        "top_10_precision": float(np.mean(p10)),
        "top_10_recall": float(np.mean(r10)),
        "top_20pct_precision": float(np.mean(p20)),
        "top_20pct_recall": float(np.mean(r20)),
    }


def classification_metrics(meta: pd.DataFrame, y_true: np.ndarray, score: np.ndarray, threshold: float, top_k: int, top_frac: float) -> Dict[str, Any]:
    """
    Compute standard classification metrics plus week-level ranking metrics.
    """
    y_pred = (score >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "avg_precision": float(average_precision_score(y_true, score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(y_pred)),
        "n_rows": int(len(y_true)),
        "n_positive": int(np.sum(y_true)),
    }
    out.update(ranking_metrics_by_week(meta, score, y_true, top_k, top_frac))
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Compute regression metrics for count-like targets.
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "mean_target": float(np.mean(y_true)),
        "mean_prediction": float(np.mean(y_pred)),
        "n_rows": int(len(y_true)),
    }


def multilabel_f1_metrics(ex: pd.DataFrame, thresholds: Dict[str, float]) -> Dict[str, float]:
    """
    Compute macro/micro/weighted F1 across the subtype prediction set.
    """
    per = []
    supports = []
    Ys = []
    Ps = []
    out = {}

    for s in SUBTYPE_ORDER:
        y = ex[f"actual_{s}"].to_numpy(dtype=int)
        p = (ex[f"score_{s}"].to_numpy(dtype=float) >= thresholds[s]).astype(int)
        f = f1_score(y, p, zero_division=0)
        out[f"f1_{s}"] = float(f)
        per.append(f)
        supports.append(max(int(y.sum()), 1))
        Ys.append(y)
        Ps.append(p)

    Y = np.vstack(Ys).T
    P = np.vstack(Ps).T
    out["subtype_macro_f1"] = float(np.mean(per))
    out["subtype_weighted_f1"] = float(np.average(per, weights=supports))
    out["subtype_micro_f1"] = float(f1_score(Y.reshape(-1), P.reshape(-1), zero_division=0))
    return out


def add_common_prediction_fields(ex: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    """
    Add thresholded labels, ranks, and a few convenience summary columns
    to a split-level export table.
    """
    ex = ex.copy()

    ex["threshold_any_event"] = thresholds["any_event"]
    ex["threshold_high_intensity"] = thresholds["high_intensity"]
    ex["pred_any_event"] = (ex["score_any_event"] >= thresholds["any_event"]).astype(int)
    ex["pred_high_intensity"] = (ex["score_high_intensity"] >= thresholds["high_intensity"]).astype(int)
    ex["prediction_rank_within_week_any_event"] = ex.groupby("week_start")["score_any_event"].rank(method="first", ascending=False)
    ex["prediction_rank_within_week_high_intensity"] = ex.groupby("week_start")["score_high_intensity"].rank(method="first", ascending=False)

    for s in SUBTYPE_ORDER:
        ex[f"threshold_{s}"] = thresholds[s]
        ex[f"pred_{s}"] = (ex[f"score_{s}"] >= thresholds[s]).astype(int)
        ex[f"prediction_rank_within_week_{s}"] = ex.groupby("week_start")[f"score_{s}"].rank(method="first", ascending=False)

    # Dominant subtype = subtype with highest predicted/actual score among the subtype set
    ex["dominant_predicted_subtype"] = pd.Series([
        SUBTYPE_DISPLAY[SUBTYPE_ORDER[int(np.argmax([ex.iloc[i][f"score_{s}"] for s in SUBTYPE_ORDER]))]]
        for i in range(len(ex))
    ], dtype="string")

    ex["predicted_subtype_list"] = pd.Series([
        ";".join([SUBTYPE_DISPLAY[s] for s in SUBTYPE_ORDER if int(ex.iloc[i][f"pred_{s}"]) == 1])
        for i in range(len(ex))
    ], dtype="string")

    ex["dominant_actual_subtype"] = pd.Series([
        (
            SUBTYPE_DISPLAY[SUBTYPE_ORDER[int(np.argmax([ex.iloc[i][f"actual_{s}"] for s in SUBTYPE_ORDER]))]]
            if sum(int(ex.iloc[i][f"actual_{s}"]) for s in SUBTYPE_ORDER) > 0 else ""
        )
        for i in range(len(ex))
    ], dtype="string")

    ex["actual_subtype_list"] = pd.Series([
        ";".join([SUBTYPE_DISPLAY[s] for s in SUBTYPE_ORDER if int(ex.iloc[i][f"actual_{s}"]) == 1])
        for i in range(len(ex))
    ], dtype="string")

    # Compatibility aliases used in downstream displays
    ex["score_event_name"] = ex["dominant_predicted_subtype"]
    ex["actual_event_name"] = ex["dominant_actual_subtype"]
    return ex


@dataclass
class HierOutputs:
    """
    Small container for thresholds, exported split tables, and metrics.
    """
    thresholds: Dict[str, float]
    exports: Dict[str, pd.DataFrame]
    metrics: Dict[str, Any]


def train_hier_tabular(df: pd.DataFrame, feature_cols: List[str], family: str, seed: int, top_k: int, top_frac: float) -> HierOutputs:
    """
    Train the hierarchical tabular model family.
    any_event is the parent probability and all child probabilities are constrained below it.
    """
    data = df.reset_index(drop=True).copy()
    X = to_numeric_frame(data, feature_cols)

    idx_train = data["split"] == "train"
    idx_valid = data["split"] == "valid"
    idx_test = data["split"] == "test"

    X_train = X.loc[idx_train]

    # Parent event model
    event_model = fit_binary_model(X_train, data.loc[idx_train, "actual_any_event"], family, seed)

    # Child binary models are trained only on event-positive rows
    train_event_pos = idx_train & (pd.to_numeric(data["actual_any_event"], errors="coerce").fillna(0) > 0)
    X_train_event = X.loc[train_event_pos]

    high_model = fit_binary_model(X_train_event, data.loc[train_event_pos, "actual_high_intensity"], family, seed + 7)
    fatal_any_model = fit_binary_model(X_train_event, data.loc[train_event_pos, "actual_fatalities_any"], family, seed + 8)
    subtype_models = {
        s: fit_binary_model(X_train_event, data.loc[train_event_pos, f"actual_{s}"], family, seed + 20 + i)
        for i, s in enumerate(SUBTYPE_ORDER)
    }

    # Positive-only regressors for hurdle-style count predictions
    pos_event_count = idx_train & (pd.to_numeric(data["actual_event_count"], errors="coerce").fillna(0) > 0)
    pos_fatal = idx_train & (pd.to_numeric(data["actual_fatalities_sum"], errors="coerce").fillna(0) > 0)
    pos_air = idx_train & (pd.to_numeric(data["actual_air_drone_strike_count"], errors="coerce").fillna(0) > 0)

    event_count_reg = fit_positive_regressor(X.loc[pos_event_count], data.loc[pos_event_count, "actual_event_count"], family, seed + 30)
    fatal_reg = fit_positive_regressor(X.loc[pos_fatal], data.loc[pos_fatal, "actual_fatalities_sum"], family, seed + 31)
    air_reg = fit_positive_regressor(X.loc[pos_air], data.loc[pos_air, "actual_air_drone_strike_count"], family, seed + 32)

    exports: Dict[str, pd.DataFrame] = {}
    for split_name, mask in [("train", idx_train), ("valid", idx_valid), ("test", idx_test)]:
        ex = data.loc[
            mask,
            [
                *ID_COLS, "split", "target_window_start", "target_window_end",
                "forecast_window_weeks", "forecast_window_label",
                *[c for c in RAW_CONTEXT_COLS if c in data.columns],
                *[f"actual_{k}" for k in MAIN_TARGETS],
                *[f"actual_{s}" for s in SUBTYPE_ORDER],
                *[f"actual_{s.replace('_any', '_count')}" for s in SUBTYPE_ORDER],
            ]
        ].copy()

        Xs = X.loc[mask]

        # Parent probability
        p_event = predict_binary(event_model, Xs)
        ex["score_any_event"] = p_event

        # Child probabilities are multiplied by parent probability
        ex["score_high_intensity"] = np.clip(p_event * predict_binary(high_model, Xs), 0.0, 1.0)
        ex["score_fatalities_any"] = np.clip(p_event * predict_binary(fatal_any_model, Xs), 0.0, 1.0)
        for s, m in subtype_models.items():
            ex[f"score_{s}"] = np.clip(p_event * predict_binary(m, Xs), 0.0, 1.0)

        # Count heads are also gated by the corresponding parent probability
        ex["pred_event_count"] = np.clip(p_event * predict_positive_regressor(event_count_reg, Xs), 0.0, None)
        ex["pred_fatalities_sum"] = np.clip(
            ex["score_fatalities_any"].to_numpy(dtype=float) * predict_positive_regressor(fatal_reg, Xs),
            0.0, None
        )
        ex["pred_air_drone_strike_count"] = np.clip(
            ex["score_air_drone_any"].to_numpy(dtype=float) * predict_positive_regressor(air_reg, Xs),
            0.0, None
        )

        exports[split_name] = ex.reset_index(drop=True)

    thresholds = {
        "any_event": sweep_threshold(
            exports["valid"]["actual_any_event"].to_numpy(dtype=int),
            exports["valid"]["score_any_event"].to_numpy(dtype=float)
        ),
        "high_intensity": sweep_threshold(
            exports["valid"]["actual_high_intensity"].to_numpy(dtype=int),
            exports["valid"]["score_high_intensity"].to_numpy(dtype=float)
        ),
    }
    for s in SUBTYPE_ORDER:
        thresholds[s] = sweep_threshold(
            exports["valid"][f"actual_{s}"].to_numpy(dtype=int),
            exports["valid"][f"score_{s}"].to_numpy(dtype=float)
        )

    metrics: Dict[str, Any] = {}
    for split_name, ex0 in exports.items():
        ex = add_common_prediction_fields(ex0, thresholds)
        exports[split_name] = ex

        m = {
            "any_event": classification_metrics(
                ex[["week_start"]],
                ex["actual_any_event"].to_numpy(dtype=int),
                ex["score_any_event"].to_numpy(dtype=float),
                thresholds["any_event"], top_k, top_frac
            ),
            "high_intensity": classification_metrics(
                ex[["week_start"]],
                ex["actual_high_intensity"].to_numpy(dtype=int),
                ex["score_high_intensity"].to_numpy(dtype=float),
                thresholds["high_intensity"], top_k, top_frac
            ),
            "event_count": regression_metrics(
                ex["actual_event_count"].to_numpy(dtype=float),
                ex["pred_event_count"].to_numpy(dtype=float)
            ),
            "fatalities_sum": regression_metrics(
                ex["actual_fatalities_sum"].to_numpy(dtype=float),
                ex["pred_fatalities_sum"].to_numpy(dtype=float)
            ),
            "air_drone_strike_count": regression_metrics(
                ex["actual_air_drone_strike_count"].to_numpy(dtype=float),
                ex["pred_air_drone_strike_count"].to_numpy(dtype=float)
            ),
        }

        for s in SUBTYPE_ORDER:
            m[s] = classification_metrics(
                ex[["week_start"]],
                ex[f"actual_{s}"].to_numpy(dtype=int),
                ex[f"score_{s}"].to_numpy(dtype=float),
                thresholds[s], top_k, top_frac
            )

        m["subtype_cumulative"] = multilabel_f1_metrics(ex, thresholds)
        m["hierarchy_diagnostics"] = {
            "frac_high_gt_event": float((ex["score_high_intensity"] > ex["score_any_event"] + 1e-9).mean()),
            "frac_any_subtype_gt_event": float(
                np.mean([(ex[f"score_{s}"] > ex["score_any_event"] + 1e-9).mean() for s in SUBTYPE_ORDER])
            ),
        }
        metrics[split_name] = m

    return HierOutputs(thresholds, exports, metrics)


def build_naive(df: pd.DataFrame, top_k: int, top_frac: float) -> HierOutputs:
    """
    Build the hierarchical naive baseline directly from the horizon-aligned
    trailing-window source columns.
    """
    exports = {}
    thresholds = {"any_event": 0.5, "high_intensity": 0.5, **{s: 0.5 for s in SUBTYPE_ORDER}}

    for split_name in ["train", "valid", "test"]:
        ex = df[df["split"] == split_name].copy().reset_index(drop=True)
        ex["score_any_event"] = pd.to_numeric(ex["naive_source_any_event"], errors="coerce").fillna(0)
        ex["score_high_intensity"] = pd.to_numeric(ex["naive_source_high_intensity"], errors="coerce").fillna(0)
        ex["score_fatalities_any"] = pd.to_numeric(ex["naive_source_fatalities_any"], errors="coerce").fillna(0)
        for s in SUBTYPE_ORDER:
            ex[f"score_{s}"] = pd.to_numeric(ex[f"naive_source_{s}"], errors="coerce").fillna(0)
        ex["pred_event_count"] = pd.to_numeric(ex["naive_source_event_count"], errors="coerce").fillna(0)
        ex["pred_fatalities_sum"] = pd.to_numeric(ex["naive_source_fatalities_sum"], errors="coerce").fillna(0)
        ex["pred_air_drone_strike_count"] = pd.to_numeric(ex["naive_source_air_drone_strike_count"], errors="coerce").fillna(0)
        exports[split_name] = add_common_prediction_fields(ex, thresholds)

    metrics = {}
    for split_name, ex in exports.items():
        m = {
            "any_event": classification_metrics(ex[["week_start"]], ex["actual_any_event"].to_numpy(dtype=int), ex["score_any_event"].to_numpy(dtype=float), 0.5, top_k, top_frac),
            "high_intensity": classification_metrics(ex[["week_start"]], ex["actual_high_intensity"].to_numpy(dtype=int), ex["score_high_intensity"].to_numpy(dtype=float), 0.5, top_k, top_frac),
            "event_count": regression_metrics(ex["actual_event_count"].to_numpy(dtype=float), ex["pred_event_count"].to_numpy(dtype=float)),
            "fatalities_sum": regression_metrics(ex["actual_fatalities_sum"].to_numpy(dtype=float), ex["pred_fatalities_sum"].to_numpy(dtype=float)),
            "air_drone_strike_count": regression_metrics(ex["actual_air_drone_strike_count"].to_numpy(dtype=float), ex["pred_air_drone_strike_count"].to_numpy(dtype=float)),
        }
        for s in SUBTYPE_ORDER:
            m[s] = classification_metrics(ex[["week_start"]], ex[f"actual_{s}"].to_numpy(dtype=int), ex[f"score_{s}"].to_numpy(dtype=float), 0.5, top_k, top_frac)
        m["subtype_cumulative"] = multilabel_f1_metrics(ex, thresholds)
        m["hierarchy_diagnostics"] = {
            "frac_high_gt_event": float((ex["score_high_intensity"] > ex["score_any_event"] + 1e-9).mean()),
            "frac_any_subtype_gt_event": float(np.mean([(ex[f"score_{s}"] > ex["score_any_event"] + 1e-9).mean() for s in SUBTYPE_ORDER])),
        }
        metrics[split_name] = m

    return HierOutputs(thresholds, exports, metrics)


# ---------- Sequence models ----------
if HAS_TORCH:
    class SeqDataset(Dataset):
        """
        Torch dataset for the hierarchical sequence models.
        """
        def __init__(self, seq_x, static_x, targets, meta):
            self.seq_x = torch.tensor(seq_x, dtype=torch.float32)
            self.static_x = torch.tensor(static_x, dtype=torch.float32)
            self.targets = {k: torch.tensor(v, dtype=torch.float32) for k, v in targets.items()}
            self.meta = meta.reset_index(drop=True)

        def __len__(self):
            return len(self.meta)

        def __getitem__(self, idx):
            return self.seq_x[idx], self.static_x[idx], {k: v[idx] for k, v in self.targets.items()}

    class ResidualTCNBlock(nn.Module):
        """
        One residual TCN block with two dilated convolutions.
        """
        def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.2):
            super().__init__()
            pad = (kernel - 1) * dilation
            self.pad = pad
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation), nn.ReLU(), nn.Dropout(dropout),
            )
            self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x):
            y = self.net(x)
            if self.pad > 0:
                y = y[:, :, :-2 * self.pad]
            s = self.skip(x)
            if s.shape[-1] != y.shape[-1]:
                s = s[:, :, -y.shape[-1]:]
            return torch.relu(y + s)

    class HierSeqNet(nn.Module):
        """
        Shared hierarchical neural model with either a GRU or TCN encoder.
        """
        def __init__(self, seq_dim, static_dim, encoder="gru", hidden=96, dropout=0.2):
            super().__init__()
            self.encoder_type = encoder

            if encoder == "gru":
                self.seq_enc = nn.GRU(seq_dim, hidden, num_layers=2, batch_first=True, dropout=dropout, bidirectional=True)
                seq_out = hidden * 2
            else:
                self.tcn = nn.Sequential(
                    ResidualTCNBlock(seq_dim, hidden, dilation=1, dropout=dropout),
                    ResidualTCNBlock(hidden, hidden, dilation=2, dropout=dropout),
                    ResidualTCNBlock(hidden, hidden, dilation=4, dropout=dropout)
                )
                seq_out = hidden

            self.static_mlp = nn.Sequential(
                nn.Linear(static_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.ReLU()
            ) if static_dim > 0 else None

            fusion = seq_out + (hidden if static_dim > 0 else 0)
            self.trunk = nn.Sequential(
                nn.Linear(fusion, hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU()
            )

            # Parent/child heads
            self.event_head = nn.Linear(hidden, 1)
            self.high_cond_head = nn.Linear(hidden, 1)
            self.fatal_any_cond_head = nn.Linear(hidden, 1)
            self.sub_heads = nn.ModuleDict({s: nn.Linear(hidden, 1) for s in SUBTYPE_ORDER})

            # Count heads
            self.event_count_head = nn.Linear(hidden, 1)
            self.fatal_count_head = nn.Linear(hidden, 1)
            self.air_count_head = nn.Linear(hidden, 1)

        def encode(self, seq_x, static_x):
            if self.encoder_type == "gru":
                o, _ = self.seq_enc(seq_x)
                seq_repr = o[:, -1, :]
            else:
                o = self.tcn(seq_x.transpose(1, 2))
                seq_repr = o[:, :, -1]

            if self.static_mlp is not None:
                return self.trunk(torch.cat([seq_repr, self.static_mlp(static_x)], dim=1))
            return self.trunk(seq_repr)

        def forward(self, seq_x, static_x):
            h = self.encode(seq_x, static_x)

            # Parent probability
            p_event = torch.sigmoid(self.event_head(h)).squeeze(1)

            # Child binary probabilities are constrained under the parent
            p_high = p_event * torch.sigmoid(self.high_cond_head(h)).squeeze(1)
            p_fatal_any = p_event * torch.sigmoid(self.fatal_any_cond_head(h)).squeeze(1)

            out = {"any_event": p_event, "high_intensity": p_high, "fatalities_any": p_fatal_any}
            for s, head in self.sub_heads.items():
                out[s] = p_event * torch.sigmoid(head(h)).squeeze(1)

            # Count heads also use hierarchical gating
            out["event_count"] = p_event * torch.nn.functional.softplus(self.event_count_head(h)).squeeze(1)
            out["fatalities_sum"] = p_fatal_any * torch.nn.functional.softplus(self.fatal_count_head(h)).squeeze(1)
            out["air_drone_strike_count"] = out["air_drone_any"] * torch.nn.functional.softplus(self.air_count_head(h)).squeeze(1)
            return out

    def build_seq_arrays(df: pd.DataFrame, dynamic_cols: List[str], static_cols: List[str], seq_len: int):
        """
        Turn the raion-week table into fixed-length sequence examples.
        """
        df = df.sort_values(["raion_id", "week_start"]).copy()
        seq_rows = []
        static_rows = []
        metas = []
        targets = {k: [] for k in ["any_event", "high_intensity", *SUBTYPE_ORDER, "fatalities_any", "event_count", "fatalities_sum", "air_drone_strike_count"]}

        for _, grp in df.groupby("raion_id", sort=False):
            grp = grp.sort_values("week_start").reset_index(drop=True)
            dyn = grp[dynamic_cols].fillna(0).to_numpy(dtype=float)
            stat = grp[static_cols].fillna(0).to_numpy(dtype=float) if static_cols else np.zeros((len(grp), 0), dtype=float)

            for i in range(seq_len - 1, len(grp)):
                if np.isnan(grp.loc[i, "actual_any_event"]):
                    continue

                seq_rows.append(dyn[i - seq_len + 1:i + 1])
                static_rows.append(stat[i])

                meta_cols = [
                    *ID_COLS, "split", "target_window_start", "target_window_end",
                    "forecast_window_weeks", "forecast_window_label",
                    *[c for c in RAW_CONTEXT_COLS if c in grp.columns],
                    *[f"actual_{s.replace('_any', '_count')}" for s in SUBTYPE_ORDER if f"actual_{s.replace('_any', '_count')}" in grp.columns]
                ]
                metas.append(grp.loc[i, meta_cols].to_dict())

                targets["any_event"].append(float(grp.loc[i, "actual_any_event"]))
                targets["high_intensity"].append(float(grp.loc[i, "actual_high_intensity"]))
                for s in SUBTYPE_ORDER:
                    targets[s].append(float(grp.loc[i, f"actual_{s}"]))
                targets["fatalities_any"].append(float(grp.loc[i, "actual_fatalities_any"]))
                targets["event_count"].append(float(grp.loc[i, "actual_event_count"]))
                targets["fatalities_sum"].append(float(grp.loc[i, "actual_fatalities_sum"]))
                targets["air_drone_strike_count"].append(float(grp.loc[i, "actual_air_drone_strike_count"]))

        return (
            np.asarray(seq_rows, dtype=float),
            np.asarray(static_rows, dtype=float),
            {k: np.asarray(v, dtype=float) for k, v in targets.items()},
            pd.DataFrame(metas),
        )

    def transform_seq(seq, static, meta):
        """
        Standardize sequence and static features using only training rows.
        """
        idx_train = meta["split"] == "train"
        n, t, d = seq.shape

        seq_scaler = StandardScaler().fit(seq[idx_train].reshape(-1, d))
        seq_t = seq_scaler.transform(seq.reshape(-1, d)).reshape(n, t, d)

        if static.shape[1] > 0:
            st_scaler = StandardScaler().fit(static[idx_train])
            static_t = st_scaler.transform(static)
        else:
            static_t = static

        return seq_t, static_t

    def weighted_bce(prob, target, pos_weight):
        """
        Weighted BCE on probabilities directly.
        """
        eps = 1e-6
        prob = prob.clamp(eps, 1 - eps)
        return -(pos_weight * target * torch.log(prob) + (1 - target) * torch.log(1 - prob)).mean()

    def multitask_loss(out, batch_tgts, pos_weights):
        """
        Combined multitask loss across binary heads and count heads.
        """
        loss = 0.0
        for h in ["any_event", "high_intensity", *SUBTYPE_ORDER, "fatalities_any"]:
            loss = loss + weighted_bce(out[h], batch_tgts[h], pos_weights.get(h, 1.0))
        for h in ["event_count", "fatalities_sum", "air_drone_strike_count"]:
            loss = loss + 0.5 * torch.nn.functional.huber_loss(
                torch.log1p(out[h].clamp(min=0)),
                torch.log1p(batch_tgts[h].clamp(min=0))
            )
        return loss

    def eval_seq(model, loader, device):
        """
        Run the hierarchical sequence model on one split.
        """
        model.eval()
        losses = []
        preds = {k: [] for k in ["any_event", "high_intensity", *SUBTYPE_ORDER, "fatalities_any", "event_count", "fatalities_sum", "air_drone_strike_count"]}
        trues = {k: [] for k in preds}

        with torch.no_grad():
            for seq_x, static_x, tgts in loader:
                seq_x = seq_x.to(device)
                static_x = static_x.to(device)
                tgts = {k: v.to(device) for k, v in tgts.items()}
                out = model(seq_x, static_x)
                loss = multitask_loss(out, tgts, {k: 1.0 for k in preds})
                losses.append(float(loss.item()))
                for k in preds:
                    preds[k].append(out[k].cpu().numpy())
                    trues[k].append(tgts[k].cpu().numpy())

        payload = {f"pred_{k}": np.concatenate(v) if v else np.array([]) for k, v in preds.items()}
        payload.update({f"true_{k}": np.concatenate(v) if v else np.array([]) for k, v in trues.items()})
        return float(np.mean(losses) if losses else np.nan), payload

    def train_hier_seq(df: pd.DataFrame, feature_cols: List[str], dynamic_cols: List[str], static_cols: List[str], encoder: str, args, device, top_k: int, top_frac: float) -> HierOutputs:
        """
        Train one hierarchical neural sequence model, using either GRU or TCN
        as the sequence encoder.
        """
        data = df.copy()
        for c in feature_cols:
            data[c] = pd.to_numeric(data[c], errors="coerce")

        seq, static, targets, meta = build_seq_arrays(data, dynamic_cols, static_cols, args.seq_len)
        seq_t, static_t = transform_seq(seq, static, meta)

        idx_train = meta["split"] == "train"
        idx_valid = meta["split"] == "valid"
        idx_test = meta["split"] == "test"

        ds_train = SeqDataset(seq_t[idx_train], static_t[idx_train], {k: v[idx_train] for k, v in targets.items()}, meta[idx_train])
        ds_valid = SeqDataset(seq_t[idx_valid], static_t[idx_valid], {k: v[idx_valid] for k, v in targets.items()}, meta[idx_valid])
        ds_test = SeqDataset(seq_t[idx_test], static_t[idx_test], {k: v[idx_test] for k, v in targets.items()}, meta[idx_test])

        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(ds_valid, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

        model = HierSeqNet(
            seq_t.shape[2], static_t.shape[1],
            encoder=encoder, hidden=args.hidden_dim, dropout=args.dropout
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Per-head positive weights for the binary tasks
        pos_weights = {}
        for k, arr in ds_train.targets.items():
            if k in ["event_count", "fatalities_sum", "air_drone_strike_count"]:
                pos_weights[k] = 1.0
                continue
            mean = float(arr.mean().item()) if len(arr) else 0.0
            pos_weights[k] = float((1 - mean) / max(mean, 1e-4))

        best_state = None
        best_val = float("inf")
        bad = 0

        for _ in range(args.epochs):
            model.train()
            for seq_x, static_x, tgts in train_loader:
                seq_x = seq_x.to(device)
                static_x = static_x.to(device)
                tgts = {k: v.to(device) for k, v in tgts.items()}

                opt.zero_grad()
                out = model(seq_x, static_x)
                loss = multitask_loss(out, tgts, pos_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            val_loss, _ = eval_seq(model, valid_loader, device)
            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        payloads = {}
        for split_name, loader, ds in [("train", train_loader, ds_train), ("valid", valid_loader, ds_valid), ("test", test_loader, ds_test)]:
            _, pay = eval_seq(model, loader, device)
            payloads[split_name] = (ds.meta.copy(), pay)

        thresholds = {
            "any_event": sweep_threshold(payloads["valid"][1]["true_any_event"].astype(int), payloads["valid"][1]["pred_any_event"]),
            "high_intensity": sweep_threshold(payloads["valid"][1]["true_high_intensity"].astype(int), payloads["valid"][1]["pred_high_intensity"]),
        }
        for s in SUBTYPE_ORDER:
            thresholds[s] = sweep_threshold(payloads["valid"][1][f"true_{s}"].astype(int), payloads["valid"][1][f"pred_{s}"])

        exports = {}
        metrics = {}
        for split_name, (meta_df, pay) in payloads.items():
            ex = meta_df.copy()
            ex["actual_any_event"] = pay["true_any_event"]
            ex["actual_high_intensity"] = pay["true_high_intensity"]
            ex["actual_event_count"] = pay["true_event_count"]
            ex["actual_fatalities_sum"] = pay["true_fatalities_sum"]
            ex["actual_air_drone_strike_count"] = pay["true_air_drone_strike_count"]
            ex["score_any_event"] = pay["pred_any_event"]
            ex["score_high_intensity"] = pay["pred_high_intensity"]
            ex["score_fatalities_any"] = pay["pred_fatalities_any"]
            ex["pred_event_count"] = pay["pred_event_count"]
            ex["pred_fatalities_sum"] = pay["pred_fatalities_sum"]
            ex["pred_air_drone_strike_count"] = pay["pred_air_drone_strike_count"]
            for s in SUBTYPE_ORDER:
                ex[f"actual_{s}"] = pay[f"true_{s}"]
                ex[f"score_{s}"] = pay[f"pred_{s}"]

            ex = add_common_prediction_fields(ex, thresholds)
            exports[split_name] = ex

            m = {
                "any_event": classification_metrics(ex[["week_start"]], ex["actual_any_event"].to_numpy(dtype=int), ex["score_any_event"].to_numpy(dtype=float), thresholds["any_event"], top_k, top_frac),
                "high_intensity": classification_metrics(ex[["week_start"]], ex["actual_high_intensity"].to_numpy(dtype=int), ex["score_high_intensity"].to_numpy(dtype=float), thresholds["high_intensity"], top_k, top_frac),
                "event_count": regression_metrics(ex["actual_event_count"].to_numpy(dtype=float), ex["pred_event_count"].to_numpy(dtype=float)),
                "fatalities_sum": regression_metrics(ex["actual_fatalities_sum"].to_numpy(dtype=float), ex["pred_fatalities_sum"].to_numpy(dtype=float)),
                "air_drone_strike_count": regression_metrics(ex["actual_air_drone_strike_count"].to_numpy(dtype=float), ex["pred_air_drone_strike_count"].to_numpy(dtype=float)),
            }
            for s in SUBTYPE_ORDER:
                m[s] = classification_metrics(ex[["week_start"]], ex[f"actual_{s}"].to_numpy(dtype=int), ex[f"score_{s}"].to_numpy(dtype=float), thresholds[s], top_k, top_frac)
            m["subtype_cumulative"] = multilabel_f1_metrics(ex, thresholds)
            m["hierarchy_diagnostics"] = {
                "frac_high_gt_event": float((ex["score_high_intensity"] > ex["score_any_event"] + 1e-9).mean()),
                "frac_any_subtype_gt_event": float(np.mean([(ex[f"score_{s}"] > ex["score_any_event"] + 1e-9).mean() for s in SUBTYPE_ORDER])),
            }
            metrics[split_name] = m

        return HierOutputs(thresholds, exports, metrics)

else:
    def train_hier_seq(*args, **kwargs):
        raise RuntimeError("PyTorch is not available")


def write_outputs(window_dir: Path, model_name: str, algo_name: str, outputs: HierOutputs, payload_model: Dict[str, Any], summary_rows: List[Dict[str, Any]]) -> None:
    """
    Write split-level exports plus metrics for one model/algorithm combination.
    """
    for split_name, ex in outputs.exports.items():
        df = ex.copy()
        df["model_experiment"] = model_name
        df["algorithm"] = algo_name
        df.to_csv(window_dir / f"{model_name}__{algo_name}__{split_name}.csv", index=False)

    payload_model[algo_name] = {"thresholds": outputs.thresholds, "metrics": outputs.metrics}

    for split_name, split_metrics in outputs.metrics.items():
        for target_name, stats in split_metrics.items():
            if isinstance(stats, dict):
                row = {
                    "forecast_window_label": window_dir.name,
                    "model_experiment": model_name,
                    "algorithm": algo_name,
                    "split": split_name,
                    "target": target_name
                }
                row.update(stats)
                summary_rows.append(row)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    master = normalize_ids(pd.read_csv(args.master_csv))
    model1 = normalize_ids(pd.read_csv(args.model1_csv))
    model2 = normalize_ids(pd.read_csv(args.model2_csv))
    feature_spec = load_feature_spec(args.feature_spec_json)
    model1_features, model2_features = get_feature_sets(feature_spec, model1, model2)

    master, split_meta = assign_time_splits(master, args.valid_weeks, args.test_weeks)

    combined_rows = []
    combined_payload = {
        "summary": {
            "window_weeks": args.window_weeks,
            "split_meta": split_meta,
            "lightgbm_available": HAS_LIGHTGBM and not args.skip_lightgbm,
            "catboost_available": HAS_CATBOOST and not args.skip_catboost,
            "torch_available": HAS_TORCH and (not args.skip_gru or not args.skip_tcn),
            "feature_counts": {"model1": len(model1_features), "model2": len(model2_features)},
            "design": {
                "hierarchy": "high_intensity/subtypes constrained by any_event",
                "counts": "hurdle-style predictions",
                "new_neural_model": "tcn_hierarchical"
            }
        },
        "windows": {}
    }

    for weeks in args.window_weeks:
        label = DISPLAY_LABELS.get(weeks, f"next_{weeks}_weeks")
        window_dir = outdir / label
        window_dir.mkdir(parents=True, exist_ok=True)

        # Build one horizon-specific master table and attach it to both feature sets
        master_h = make_window_master(master, weeks)
        model1_h = attach_context(master_h, model1)
        model2_h = attach_context(master_h, model2)

        summary_rows_window = []
        payload_window = {"forecast_window_weeks": weeks, "forecast_window_label": label, "models": {}}

        for model_name, df_model, feat_cols in [
            ("model1_non_acled_only", model1_h, model1_features),
            ("model2_plus_lagged_acled", model2_h, model2_features)
        ]:
            dyn, stat = infer_dynamic_static(df_model, feat_cols)
            payload_window["models"][model_name] = {
                "n_features": len(feat_cols),
                "feature_cols": feat_cols,
                "dynamic_cols": dyn,
                "static_cols": stat
            }

            # Naive baseline
            write_outputs(
                window_dir, model_name, "naive_hierarchical",
                build_naive(df_model, args.top_k, args.top_frac),
                payload_window["models"][model_name], summary_rows_window
            )

            # Tabular families
            families = ["linear"]
            if HAS_LIGHTGBM and not args.skip_lightgbm:
                families.append("lightgbm")
            if HAS_CATBOOST and not args.skip_catboost:
                families.append("catboost")

            for fam in families:
                write_outputs(
                    window_dir, model_name, f"{fam}_hierarchical",
                    train_hier_tabular(df_model, feat_cols, fam, args.seed, args.top_k, args.top_frac),
                    payload_window["models"][model_name], summary_rows_window
                )

            # Sequence models
            if HAS_TORCH and not args.skip_gru:
                write_outputs(
                    window_dir, model_name, "gru_hierarchical",
                    train_hier_seq(df_model, feat_cols, dyn, stat, "gru", args, device, args.top_k, args.top_frac),
                    payload_window["models"][model_name], summary_rows_window
                )
            if HAS_TORCH and not args.skip_tcn:
                write_outputs(
                    window_dir, model_name, "tcn_hierarchical",
                    train_hier_seq(df_model, feat_cols, dyn, stat, "tcn", args, device, args.top_k, args.top_frac),
                    payload_window["models"][model_name], summary_rows_window
                )

        cmp = pd.DataFrame(summary_rows_window)
        cmp.to_csv(window_dir / "comparison_metrics_hierarchical.csv", index=False)

        with open(window_dir / "all_results_hierarchical.json", "w", encoding="utf-8") as f:
            json.dump(payload_window, f, indent=2)

        combined_rows.extend(summary_rows_window)
        combined_payload["windows"][label] = {
            "results_dir": str(window_dir),
            "comparison_metrics_csv": str(window_dir / "comparison_metrics_hierarchical.csv"),
            "json_results": str(window_dir / "all_results_hierarchical.json")
        }

        print(f"Saved window outputs to: {window_dir}")

    pd.DataFrame(combined_rows).to_csv(outdir / "comparison_metrics_all_windows_hierarchical.csv", index=False)

    with open(outdir / "all_results_hierarchical_multiwindow_summary.json", "w", encoding="utf-8") as f:
        json.dump(combined_payload, f, indent=2)

    print(f"Saved combined metrics to: {outdir / 'comparison_metrics_all_windows_hierarchical.csv'}")
    print(f"Saved combined summary to: {outdir / 'all_results_hierarchical_multiwindow_summary.json'}")


if __name__ == "__main__":
    main()