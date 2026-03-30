#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False
    CatBoostClassifier = None
    CatBoostRegressor = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None
    nn = None
    Dataset = object
    DataLoader = None


# Common identifier columns shared across the modeling tables
ID_COLS = ["raion_id", "raion_name", "oblast_name", "week_start"]

# Current-week context columns that are useful to carry into the export CSVs
# alongside the next-week targets and predictions.
CURRENT_CONTEXT_COLS = [
    "high_intensity_week",
    "any_event",
    "acled_event_count",
    "fatalities_sum",
    "air_drone_strike_count",
]

# All supported next-week targets
ALL_TARGET_COLS = [
    "y_next_high_intensity",
    "y_next_any_event",
    "y_next_event_count",
    "y_next_fatalities_sum",
    "y_next_air_drone_strike_count",
]

# For the naive baseline, just reuse the current-week version of the same signal
NAIVE_SOURCE_MAP = {
    "y_next_high_intensity": "high_intensity_week",
    "y_next_any_event": "any_event",
    "y_next_event_count": "acled_event_count",
    "y_next_fatalities_sum": "fatalities_sum",
    "y_next_air_drone_strike_count": "air_drone_strike_count",
}

# These targets are treated as binary classification problems
CLASSIFICATION_TARGETS = {"y_next_high_intensity", "y_next_any_event"}

# Used to roughly separate static features from time-varying ones for the GRU
STATIC_PREFIXES = (
    "road_",
    "rail_",
    "pop_",
    "place_",
    "places_",
    "border_",
    "nearest_border",
    "unosat_",
    "area_sqkm",
    "major_road_",
    "paved_road_",
    "rail_to_road_ratio",
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training/exporting multiple model families
    and writing one prediction CSV per target per model.
    """
    p = argparse.ArgumentParser(
        description=(
            "Train/export multiple model families for raion-week next-week prediction. "
            "Writes one CSV per target per model with predictions and actual next-week results."
        )
    )
    p.add_argument("--master_csv", required=True)
    p.add_argument("--model1_csv", required=True)
    p.add_argument("--model2_csv", required=True)
    p.add_argument("--feature_spec_json", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument(
        "--targets",
        nargs="+",
        default=["y_next_high_intensity", "y_next_any_event", "y_next_air_drone_strike_count"],
        help="Targets to train/export",
    )
    p.add_argument("--acled_events_csv", default=None, help="Optional event-level ACLED CSV for drilldown exports")
    p.add_argument("--valid_weeks", type=int, default=13)
    p.add_argument("--test_weeks", type=int, default=13)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--top_frac", type=float, default=0.20)
    p.add_argument("--skip_gru", action="store_true")
    p.add_argument("--skip_lightgbm", action="store_true")
    p.add_argument("--skip_catboost", action="store_true")
    p.add_argument("--gru_seq_len", type=int, default=8)
    p.add_argument("--gru_epochs", type=int, default=12)
    p.add_argument("--gru_batch_size", type=int, default=128)
    p.add_argument("--gru_hidden_dim", type=int, default=96)
    p.add_argument("--gru_num_layers", type=int, default=2)
    p.add_argument("--gru_dropout", type=float, default=0.2)
    p.add_argument("--gru_lr", type=float, default=1e-3)
    p.add_argument("--gru_weight_decay", type=float, default=1e-4)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--drilldown_splits",
        nargs="+",
        default=["test"],
        choices=["train", "valid", "test"],
        help="Which splits to include in exploded event drilldown CSVs",
    )
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
    Pick a torch device. If torch is unavailable, return None.
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


def normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the main ID columns before merging:
    parse week_start as datetime and cast name/id columns to string.
    """
    out = df.copy()
    if "week_start" in out.columns:
        out["week_start"] = pd.to_datetime(out["week_start"], errors="coerce")
    for c in ["raion_id", "raion_name", "oblast_name"]:
        if c in out.columns:
            out[c] = out[c].astype("string")
    return out


def load_feature_spec(path: str) -> Dict[str, Any]:
    """
    Load the feature spec JSON. Some build scripts wrap the actual spec
    under a top-level 'feature_spec' key, so support both formats.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("feature_spec", payload)


def assign_time_splits(df: pd.DataFrame, valid_weeks: int, test_weeks: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create time-based train/valid/test splits using trailing weeks.
    """
    out = normalize_id_columns(df)
    weeks = sorted(pd.to_datetime(out["week_start"]).dropna().unique())
    if len(weeks) <= (valid_weeks + test_weeks + 4):
        raise ValueError(f"Not enough distinct weeks ({len(weeks)}) for the requested split sizes.")

    test_start = weeks[-test_weeks]
    valid_start = weeks[-(test_weeks + valid_weeks)]

    out["split"] = np.where(
        out["week_start"] >= test_start,
        "test",
        np.where(out["week_start"] >= valid_start, "valid", "train"),
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


def get_feature_sets(feature_spec: Dict[str, Any], model1_df: pd.DataFrame, model2_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Get Model 1 and Model 2 feature lists from the feature spec.
    Fall back to inferring them from the table columns if needed.
    """
    m1 = feature_spec.get("model1_non_acled_only_features", [])
    m2 = feature_spec.get("model2_plus_lagged_acled_features", [])
    if not m1:
        m1 = [c for c in model1_df.columns if c not in ID_COLS and not c.startswith("y_next_") and c != "target_week_start"]
    if not m2:
        m2 = [c for c in model2_df.columns if c not in ID_COLS and not c.startswith("y_next_") and c != "target_week_start"]
    return sorted([c for c in m1 if c in model1_df.columns]), sorted([c for c in m2 if c in model2_df.columns])


def attach_master_context(master: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join the canonical split labels plus current/target context columns
    from the master table into a model-specific table.
    """
    keys = [c for c in ID_COLS if c in master.columns and c in model_df.columns]
    ctx_only = [c for c in (CURRENT_CONTEXT_COLS + ALL_TARGET_COLS + ["split"]) if c in master.columns]
    ctx_cols = keys + ctx_only

    out = normalize_id_columns(model_df)

    # Drop any local copies of these columns first so the master version wins
    drop_if_present = [c for c in ctx_only if c in out.columns]
    if drop_if_present:
        out = out.drop(columns=drop_if_present)

    merged = out.merge(master[ctx_cols], on=keys, how="left", validate="many_to_one")
    return merged


def task_type_for_target(target_col: str) -> str:
    """
    Decide whether a target is classification or regression.
    """
    return "classification" if target_col in CLASSIFICATION_TARGETS else "regression"


def to_numeric_frame(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    """
    Convert a feature subset to numeric values, coercing anything invalid to NaN.
    """
    X = df[list(feature_cols)].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def make_base_prediction_frame(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Build the common export frame that all model outputs start from.
    """
    cols = [c for c in ID_COLS if c in df.columns] + [
        "split",
        *[c for c in CURRENT_CONTEXT_COLS if c in df.columns],
        *[c for c in ALL_TARGET_COLS if c in df.columns],
    ]
    out = df[cols].copy()
    out = normalize_id_columns(out)
    out["target_col"] = target_col
    out["actual_target"] = pd.to_numeric(df[target_col], errors="coerce")

    # The target is for the following Monday-Sunday style next-week window
    out["target_window_start"] = out["week_start"] + pd.Timedelta(days=7)
    out["target_window_end"] = out["week_start"] + pd.Timedelta(days=13)
    return out


def ranking_metrics_by_week(meta: pd.DataFrame, scores: np.ndarray, y_true: np.ndarray, top_k: int, top_frac: float) -> Dict[str, float]:
    """
    Compute week-wise ranking metrics, which are often useful
    when the goal is to prioritize the riskiest raions each week.
    """
    if len(meta) == 0:
        return {"top_10_precision": np.nan, "top_10_recall": np.nan, "top_20pct_precision": np.nan, "top_20pct_recall": np.nan}

    wk = meta[["week_start"]].copy()
    wk["score"] = scores
    wk["y_true"] = y_true

    topk_precisions, topk_recalls = [], []
    topf_precisions, topf_recalls = [], []

    for _, grp in wk.groupby("week_start"):
        grp = grp.sort_values("score", ascending=False)
        y = grp["y_true"].to_numpy()
        positives = max(int(y.sum()), 1)

        k = min(top_k, len(grp))
        topk = grp.head(k)
        tp_k = float(topk["y_true"].sum())
        topk_precisions.append(tp_k / max(len(topk), 1))
        topk_recalls.append(tp_k / positives)

        kf = max(int(np.ceil(len(grp) * top_frac)), 1)
        topf = grp.head(kf)
        tp_f = float(topf["y_true"].sum())
        topf_precisions.append(tp_f / max(len(topf), 1))
        topf_recalls.append(tp_f / positives)

    return {
        "top_10_precision": float(np.mean(topk_precisions)),
        "top_10_recall": float(np.mean(topk_recalls)),
        "top_20pct_precision": float(np.mean(topf_precisions)),
        "top_20pct_recall": float(np.mean(topf_recalls)),
    }


def classification_metrics(meta: pd.DataFrame, y_true: np.ndarray, score: np.ndarray, threshold: float, top_k: int, top_frac: float) -> Dict[str, Any]:
    """
    Compute classification metrics for one split.
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
    Compute regression metrics for one split.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "mean_target": float(np.mean(y_true)),
        "mean_prediction": float(np.mean(y_pred)),
        "n_rows": int(len(y_true)),
    }


def build_linear_model(task_type: str, seed: int) -> Pipeline:
    """
    Build the linear baseline model:
    logistic regression for classification, ridge regression for regression.
    """
    estimator = (
        LogisticRegression(max_iter=2500, class_weight="balanced", solver="liblinear", random_state=seed)
        if task_type == "classification"
        else Ridge(alpha=1.0, random_state=seed)
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("estimator", estimator),
    ])


def build_lightgbm_model(task_type: str, target_col: str, seed: int):
    """
    Build a LightGBM model. For count-like regression targets,
    use a Poisson objective.
    """
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
            verbose=-1,
        )

    objective = "poisson" if target_col in {"y_next_event_count", "y_next_fatalities_sum", "y_next_air_drone_strike_count"} else "regression"
    return LGBMRegressor(
        objective=objective,
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def build_catboost_model(task_type: str, target_col: str, seed: int):
    """
    Build a CatBoost model. For count-like regression targets,
    use a Poisson loss when appropriate.
    """
    if task_type == "classification":
        return CatBoostClassifier(
            loss_function="Logloss",
            iterations=500,
            learning_rate=0.03,
            depth=6,
            eval_metric="AUC",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )

    loss = "Poisson" if target_col in {"y_next_event_count", "y_next_fatalities_sum", "y_next_air_drone_strike_count"} else "RMSE"
    return CatBoostRegressor(
        loss_function=loss,
        iterations=500,
        learning_rate=0.03,
        depth=6,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def fit_predict_sklearn(model, task_type: str, X_train, y_train, X_valid, X_test):
    """
    Fit a sklearn-style model and return predictions/scores
    for train, validation, and test splits.
    """
    model.fit(X_train, y_train)
    if task_type == "classification":
        p_train = model.predict_proba(X_train)[:, 1]
        p_valid = model.predict_proba(X_valid)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]
        return p_train, p_valid, p_test

    yhat_train = model.predict(X_train)
    yhat_valid = model.predict(X_valid)
    yhat_test = model.predict(X_test)
    return yhat_train, yhat_valid, yhat_test


def fit_predict_lightgbm(model, task_type: str, X_train, y_train, X_valid, y_valid, X_test):
    """
    Fit a LightGBM model and return predictions/scores.
    Clamp regression outputs at zero since these targets should not be negative.
    """
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    if task_type == "classification":
        return model.predict_proba(X_train)[:, 1], model.predict_proba(X_valid)[:, 1], model.predict_proba(X_test)[:, 1]
    return np.maximum(model.predict(X_train), 0.0), np.maximum(model.predict(X_valid), 0.0), np.maximum(model.predict(X_test), 0.0)


def fit_predict_catboost(model, task_type: str, X_train, y_train, X_valid, y_valid, X_test):
    """
    Fit a CatBoost model and return predictions/scores.
    Clamp regression outputs at zero for the count-like targets.
    """
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
    if task_type == "classification":
        return model.predict_proba(X_train)[:, 1], model.predict_proba(X_valid)[:, 1], model.predict_proba(X_test)[:, 1]
    return np.maximum(model.predict(X_train), 0.0), np.maximum(model.predict(X_valid), 0.0), np.maximum(model.predict(X_test), 0.0)


def choose_best_threshold(y_valid: np.ndarray, score_valid: np.ndarray) -> float:
    """
    Pick the classification threshold that gives the best validation F1.
    """
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.round(np.arange(0.10, 0.91, 0.05), 2):
        f1 = f1_score(y_valid, (score_valid >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def add_prediction_columns(df: pd.DataFrame, task_type: str, pred_train, pred_valid, pred_test, threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Add prediction columns back onto the export frame.
    """
    out = df.copy()
    mask_train = out["split"] == "train"
    mask_valid = out["split"] == "valid"
    mask_test = out["split"] == "test"

    if task_type == "classification":
        out["prediction_score"] = np.nan
        out.loc[mask_train, "prediction_score"] = pred_train
        out.loc[mask_valid, "prediction_score"] = pred_valid
        out.loc[mask_test, "prediction_score"] = pred_test
        out["prediction_threshold"] = threshold
        out["prediction_label"] = (out["prediction_score"] >= threshold).astype("Int64")
        out["prediction_rank_within_week"] = out.groupby("week_start")["prediction_score"].rank(method="first", ascending=False).astype(int)
    else:
        out["prediction_value"] = np.nan
        out.loc[mask_train, "prediction_value"] = pred_train
        out.loc[mask_valid, "prediction_value"] = pred_valid
        out.loc[mask_test, "prediction_value"] = pred_test
        out["prediction_value"] = np.maximum(out["prediction_value"], 0.0)
        out["prediction_rank_within_week"] = out.groupby("week_start")["prediction_value"].rank(method="first", ascending=False).astype(int)

    return out


def save_export_csv(base_df: pd.DataFrame, out_csv: Path) -> None:
    """
    Save one prediction export CSV, formatting dates consistently first.
    """
    out = base_df.copy()
    for c in ["week_start", "target_window_start", "target_window_end"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d")
    out.to_csv(out_csv, index=False)


def save_event_drilldown(pred_df: pd.DataFrame, acled_events: pd.DataFrame, target_col: str, out_csv: Path, splits: List[str]) -> None:
    """
    Create an exploded event-level drilldown file for selected splits.
    This links prediction rows to the actual ACLED events inside the target window.
    """
    sub = pred_df[pred_df["split"].isin(splits)].copy()
    if sub.empty:
        return

    ev = acled_events.copy()
    if target_col == "y_next_air_drone_strike_count":
        ev = ev[ev["sub_event_type"].astype(str).str.lower() == "air/drone strike"]

    ev["event_date"] = pd.to_datetime(ev["event_date"], errors="coerce")

    keep_pred_cols = [
        "raion_id", "raion_name", "oblast_name", "week_start", "target_window_start", "target_window_end", "split",
        "actual_target",
        "prediction_score" if "prediction_score" in sub.columns else "prediction_value",
        "prediction_label" if "prediction_label" in sub.columns else None,
        "prediction_rank_within_week",
    ]
    keep_pred_cols = [c for c in keep_pred_cols if c is not None and c in sub.columns]

    # First merge by raion, then filter down to events inside the target week window
    merged = sub[keep_pred_cols].merge(
        ev,
        on=[c for c in ["raion_id", "raion_name", "oblast_name"] if c in ev.columns and c in sub.columns],
        how="left",
    )

    mask = (
        merged["event_date"].notna()
        & (merged["event_date"] >= pd.to_datetime(merged["target_window_start"]))
        & (merged["event_date"] <= pd.to_datetime(merged["target_window_end"]))
    )
    merged = merged[mask].copy()

    if not merged.empty:
        merged["week_start"] = pd.to_datetime(merged["week_start"]).dt.strftime("%Y-%m-%d")
        merged["target_window_start"] = pd.to_datetime(merged["target_window_start"]).dt.strftime("%Y-%m-%d")
        merged["target_window_end"] = pd.to_datetime(merged["target_window_end"]).dt.strftime("%Y-%m-%d")
        merged["event_date"] = pd.to_datetime(merged["event_date"]).dt.strftime("%Y-%m-%d")
        merged.to_csv(out_csv, index=False)


class SequenceDataset(Dataset):
    """
    Simple torch dataset for the GRU model.
    """
    def __init__(self, sequences: np.ndarray, statics: np.ndarray, targets: np.ndarray, meta: pd.DataFrame) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.statics = torch.tensor(statics, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.meta = meta.reset_index(drop=True)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.statics[idx], self.targets[idx]


class GRUHead(nn.Module):
    """
    Bidirectional GRU over the dynamic sequence features,
    optionally fused with a small MLP over static features.
    """
    def __init__(self, seq_dim: int, static_dim: int, hidden_dim: int, num_layers: int, dropout: float, task_type: str):
        super().__init__()
        self.task_type = task_type

        self.gru = nn.GRU(
            input_size=seq_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.static_mlp = None
        if static_dim > 0:
            self.static_mlp = nn.Sequential(
                nn.Linear(static_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        fusion_in = hidden_dim * 2 + (hidden_dim if static_dim > 0 else 0)
        self.head = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, seq_x: torch.Tensor, static_x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(seq_x)
        seq_repr = out[:, -1, :]

        fused = seq_repr
        if self.static_mlp is not None:
            static_repr = self.static_mlp(static_x)
            fused = torch.cat([seq_repr, static_repr], dim=1)

        return self.head(fused).squeeze(1)


def split_dynamic_static(feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    """
    Heuristically separate feature columns into dynamic vs static groups
    for the GRU pipeline.
    """
    dynamic, static = [], []
    for c in feature_cols:
        if c.startswith(STATIC_PREFIXES):
            static.append(c)
        elif c.startswith("firms_") or c.startswith("ntl_") or c.startswith("acled_") or c.startswith("fatalities_") or c.startswith("air_drone_") or c.startswith("any_event_") or c.startswith("high_intensity_"):
            dynamic.append(c)
        elif "_lag" in c or "_roll" in c:
            dynamic.append(c)
        else:
            static.append(c)
    return dynamic, static


def build_sequence_rows(df: pd.DataFrame, dynamic_cols: List[str], static_cols: List[str], seq_len: int, target_col: str):
    """
    Turn the raion-week table into fixed-length sequence examples for the GRU.
    """
    df = df.sort_values(["raion_id", "week_start"]).copy()
    seq_rows, static_rows, targets, metas = [], [], [], []

    for _, grp in df.groupby("raion_id", sort=False):
        grp = grp.sort_values("week_start").reset_index(drop=True)
        dyn = grp[dynamic_cols].fillna(0).to_numpy(dtype=float)
        stat = grp[static_cols].fillna(0).to_numpy(dtype=float) if static_cols else np.zeros((len(grp), 0), dtype=float)
        tgt = pd.to_numeric(grp[target_col], errors="coerce").to_numpy(dtype=float)

        for i in range(seq_len - 1, len(grp)):
            if np.isnan(tgt[i]):
                continue
            seq_rows.append(dyn[i - seq_len + 1:i + 1])
            static_rows.append(stat[i])
            targets.append(tgt[i])
            metas.append(grp.loc[i, [c for c in ID_COLS if c in grp.columns] + ["split"]].to_dict())

    return np.asarray(seq_rows, dtype=float), np.asarray(static_rows, dtype=float), np.asarray(targets, dtype=float), pd.DataFrame(metas)


def fit_scalers(train_seq: np.ndarray, train_static: np.ndarray):
    """
    Fit sequence/static scalers using only the training portion.
    """
    from sklearn.preprocessing import StandardScaler
    n, t, d = train_seq.shape
    seq_scaler = StandardScaler()
    seq_scaler.fit(train_seq.reshape(n * t, d))

    static_scaler = None
    if train_static.shape[1] > 0:
        static_scaler = StandardScaler()
        static_scaler.fit(train_static)

    return seq_scaler, static_scaler


def transform_sequence_data(seq: np.ndarray, static: np.ndarray, seq_scaler, static_scaler):
    """
    Apply the fitted GRU scalers to sequence and static features.
    """
    n, t, d = seq.shape
    seq_t = seq_scaler.transform(seq.reshape(n * t, d)).reshape(n, t, d)
    static_t = static_scaler.transform(static) if static_scaler is not None and static.shape[1] > 0 else static
    return seq_t, static_t


def run_gru_experiment(df: pd.DataFrame, feature_cols: List[str], target_col: str, task_type: str, args: argparse.Namespace, seed: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train and evaluate the GRU model for one target and one feature set.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available for GRU training.")

    dynamic_cols, static_cols = split_dynamic_static(feature_cols)
    if not dynamic_cols:
        raise ValueError("GRU requires at least one dynamic feature column.")

    seq, static, y, meta = build_sequence_rows(df, dynamic_cols, static_cols, args.gru_seq_len, target_col)
    if len(meta) == 0:
        raise ValueError("No GRU sequence rows created. Check sequence length and available history.")

    idx_train = meta["split"] == "train"
    idx_valid = meta["split"] == "valid"
    idx_test = meta["split"] == "test"
    if idx_train.sum() == 0:
        raise ValueError("No GRU training rows found.")

    seq_scaler, static_scaler = fit_scalers(seq[idx_train], static[idx_train])
    seq_t, static_t = transform_sequence_data(seq, static, seq_scaler, static_scaler)

    ds_train = SequenceDataset(seq_t[idx_train], static_t[idx_train], y[idx_train], meta[idx_train])
    ds_valid = SequenceDataset(seq_t[idx_valid], static_t[idx_valid], y[idx_valid], meta[idx_valid])
    ds_test = SequenceDataset(seq_t[idx_test], static_t[idx_test], y[idx_test], meta[idx_test])

    train_loader = DataLoader(ds_train, batch_size=args.gru_batch_size, shuffle=True)
    valid_loader = DataLoader(ds_valid, batch_size=args.gru_batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=args.gru_batch_size, shuffle=False)

    device = choose_device(args.device)
    model = GRUHead(seq_t.shape[2], static_t.shape[1], args.gru_hidden_dim, args.gru_num_layers, args.gru_dropout, task_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.gru_lr, weight_decay=args.gru_weight_decay)

    if task_type == "classification":
        pos_rate = float(ds_train.targets.mean().item()) if len(ds_train) else 0.5
        pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-6)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.MSELoss()

    def eval_loader(loader: DataLoader):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for seq_x, static_x, target in loader:
                seq_x = seq_x.to(device)
                static_x = static_x.to(device)
                logits = model(seq_x, static_x)
                pred = torch.sigmoid(logits) if task_type == "classification" else logits
                ys.append(target.numpy())
                ps.append(pred.detach().cpu().numpy())
        return np.concatenate(ys) if ys else np.array([]), np.concatenate(ps) if ps else np.array([])

    best_state = None
    best_score = -np.inf

    for _ in range(args.gru_epochs):
        model.train()
        for seq_x, static_x, target in train_loader:
            seq_x = seq_x.to(device)
            static_x = static_x.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits = model(seq_x, static_x)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        yv, pv = eval_loader(valid_loader)
        if task_type == "classification":
            score = average_precision_score(yv, pv) if len(np.unique(yv)) > 1 else -np.inf
        else:
            score = -mean_absolute_error(yv, np.maximum(pv, 0.0))

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    yt, pt = eval_loader(train_loader)
    yv, pv = eval_loader(valid_loader)
    ys, ps = eval_loader(test_loader)

    if task_type == "regression":
        pt = np.maximum(pt, 0.0)
        pv = np.maximum(pv, 0.0)
        ps = np.maximum(ps, 0.0)

    base = make_base_prediction_frame(df, target_col)

    # Only rows with enough prior history get GRU predictions,
    # so re-merge against the eligible sequence metadata.
    seq_meta = pd.concat([
        ds_train.meta.assign(_split_name="train"),
        ds_valid.meta.assign(_split_name="valid"),
        ds_test.meta.assign(_split_name="test"),
    ], ignore_index=True)
    seq_meta["has_gru_prediction"] = True

    base = base.merge(
        seq_meta[[c for c in ID_COLS if c in seq_meta.columns] + ["split", "has_gru_prediction"]],
        on=[c for c in ID_COLS if c in seq_meta.columns] + ["split"],
        how="left"
    )
    base = base[base["has_gru_prediction"] == True].drop(columns=["has_gru_prediction"]).copy()

    pred_df = add_prediction_columns(base, task_type, pt, pv, ps, threshold=None)

    metrics = {"n_sequence_rows": int(len(pred_df)), "seq_len": int(args.gru_seq_len)}
    if task_type == "classification":
        thr = choose_best_threshold(yv, pv)
        pred_df["prediction_threshold"] = thr
        pred_df["prediction_label"] = (pred_df["prediction_score"] >= thr).astype("Int64")
        metrics.update({
            "train": classification_metrics(pred_df[pred_df.split == "train"], yt, pt, thr, args.top_k, args.top_frac),
            "valid": classification_metrics(pred_df[pred_df.split == "valid"], yv, pv, thr, args.top_k, args.top_frac),
            "test": classification_metrics(pred_df[pred_df.split == "test"], ys, ps, thr, args.top_k, args.top_frac),
            "best_threshold": thr,
            "dynamic_cols": dynamic_cols,
            "static_cols": static_cols,
        })
    else:
        metrics.update({
            "train": regression_metrics(yt, pt),
            "valid": regression_metrics(yv, pv),
            "test": regression_metrics(ys, ps),
            "dynamic_cols": dynamic_cols,
            "static_cols": static_cols,
        })

    return pred_df, metrics


def run_standard_experiment(df: pd.DataFrame, feature_cols: List[str], target_col: str, task_type: str, algo_name: str, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train and evaluate one standard non-sequence model
    such as linear, LightGBM, or CatBoost.
    """
    work = df.copy()
    X = to_numeric_frame(work, feature_cols)
    y = pd.to_numeric(work[target_col], errors="coerce")

    mask = y.notna()
    work = work.loc[mask].reset_index(drop=True)
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    idx_train = work["split"] == "train"
    idx_valid = work["split"] == "valid"
    idx_test = work["split"] == "test"

    X_train, y_train = X.loc[idx_train], y.loc[idx_train]
    X_valid, y_valid = X.loc[idx_valid], y.loc[idx_valid]
    X_test, y_test = X.loc[idx_test], y.loc[idx_test]

    if algo_name == "linear":
        model = build_linear_model(task_type, args.seed)
        p_train, p_valid, p_test = fit_predict_sklearn(model, task_type, X_train, y_train, X_valid, X_test)
    elif algo_name == "lightgbm":
        if not HAS_LIGHTGBM or args.skip_lightgbm:
            raise RuntimeError("LightGBM not available or skipped.")
        model = build_lightgbm_model(task_type, target_col, args.seed)
        p_train, p_valid, p_test = fit_predict_lightgbm(model, task_type, X_train, y_train, X_valid, y_valid, X_test)
    elif algo_name == "catboost":
        if not HAS_CATBOOST or args.skip_catboost:
            raise RuntimeError("CatBoost not available or skipped.")
        model = build_catboost_model(task_type, target_col, args.seed)
        p_train, p_valid, p_test = fit_predict_catboost(model, task_type, X_train, y_train, X_valid, y_valid, X_test)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    base = make_base_prediction_frame(work, target_col)
    metrics: Dict[str, Any] = {"n_features": len(feature_cols), "feature_cols": feature_cols}

    if task_type == "classification":
        thr = choose_best_threshold(y_valid.to_numpy(), np.asarray(p_valid))
        pred_df = add_prediction_columns(base, task_type, p_train, p_valid, p_test, threshold=thr)
        metrics["best_threshold"] = thr
        metrics["train"] = classification_metrics(pred_df[pred_df.split == "train"], y_train.to_numpy(), np.asarray(p_train), thr, args.top_k, args.top_frac)
        metrics["valid"] = classification_metrics(pred_df[pred_df.split == "valid"], y_valid.to_numpy(), np.asarray(p_valid), thr, args.top_k, args.top_frac)
        metrics["test"] = classification_metrics(pred_df[pred_df.split == "test"], y_test.to_numpy(), np.asarray(p_test), thr, args.top_k, args.top_frac)
    else:
        pred_df = add_prediction_columns(base, task_type, p_train, p_valid, p_test, threshold=None)
        metrics["train"] = regression_metrics(y_train.to_numpy(), np.asarray(p_train))
        metrics["valid"] = regression_metrics(y_valid.to_numpy(), np.asarray(p_valid))
        metrics["test"] = regression_metrics(y_test.to_numpy(), np.asarray(p_test))

    return pred_df, metrics


def run_naive_baseline(master: pd.DataFrame, target_col: str, task_type: str, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the naive persistence baseline using the mapped current-week signal.
    """
    if target_col not in master.columns:
        raise ValueError(f"Target not found in master: {target_col}")

    source_col = NAIVE_SOURCE_MAP[target_col]
    if source_col not in master.columns:
        raise ValueError(f"Naive source column not found: {source_col}")

    work = master[pd.to_numeric(master[target_col], errors="coerce").notna()].copy().reset_index(drop=True)
    base = make_base_prediction_frame(work, target_col)
    metrics = {"source_col": source_col}

    if task_type == "classification":
        score = pd.to_numeric(work[source_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        base = add_prediction_columns(base, task_type, score[work.split == "train"], score[work.split == "valid"], score[work.split == "test"], threshold=0.5)
        for split in ["train", "valid", "test"]:
            mask = work["split"] == split
            metrics[split] = classification_metrics(base[base.split == split], pd.to_numeric(work.loc[mask, target_col]).to_numpy(dtype=float), score[mask], 0.5, args.top_k, args.top_frac)
    else:
        pred = pd.to_numeric(work[source_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        base = add_prediction_columns(base, task_type, pred[work.split == "train"], pred[work.split == "valid"], pred[work.split == "test"], threshold=None)
        for split in ["train", "valid", "test"]:
            mask = work["split"] == split
            metrics[split] = regression_metrics(pd.to_numeric(work.loc[mask, target_col]).to_numpy(dtype=float), pred[mask])

    return base, metrics


def run_hurdle_experiment(df: pd.DataFrame, feature_cols: List[str], target_col: str, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Two-stage hurdle model for next-week air/drone strike counts:
    first predict whether any strike happens, then predict the count given positive.
    """
    work = df.copy()
    X = to_numeric_frame(work, feature_cols)
    y_count = pd.to_numeric(work[target_col], errors="coerce")

    mask = y_count.notna()
    work = work.loc[mask].reset_index(drop=True)
    X = X.loc[mask].reset_index(drop=True)
    y_count = y_count.loc[mask].reset_index(drop=True)
    y_any = (y_count > 0).astype(int)

    idx_train = work["split"] == "train"
    idx_valid = work["split"] == "valid"
    idx_test = work["split"] == "test"

    X_train, X_valid, X_test = X.loc[idx_train], X.loc[idx_valid], X.loc[idx_test]
    y_any_train, y_any_valid, y_any_test = y_any.loc[idx_train], y_any.loc[idx_valid], y_any.loc[idx_test]
    y_count_train, y_count_valid, y_count_test = y_count.loc[idx_train], y_count.loc[idx_valid], y_count.loc[idx_test]

    if HAS_LIGHTGBM and not args.skip_lightgbm:
        stage1 = build_lightgbm_model("classification", "y_next_any_event", args.seed)
        p1_train, p1_valid, p1_test = fit_predict_lightgbm(stage1, "classification", X_train, y_any_train, X_valid, y_any_valid, X_test)

        stage2 = build_lightgbm_model("regression", target_col, args.seed)
        pos_train = y_count_train > 0
        pos_valid = y_count_valid > 0

        # Only fit the second-stage count model on positive-count rows
        stage2.fit(
            X_train.loc[pos_train],
            y_count_train.loc[pos_train],
            eval_set=[(X_valid.loc[pos_valid], y_count_valid.loc[pos_valid])] if pos_valid.sum() else None
        )
        c2_train = np.maximum(stage2.predict(X_train), 0.0)
        c2_valid = np.maximum(stage2.predict(X_valid), 0.0)
        c2_test = np.maximum(stage2.predict(X_test), 0.0)
    else:
        # Simple fallback hurdle using linear models
        stage1 = build_linear_model("classification", args.seed)
        p1_train, p1_valid, p1_test = fit_predict_sklearn(stage1, "classification", X_train, y_any_train, X_valid, X_test)

        stage2 = build_linear_model("regression", args.seed)
        pos_train = y_count_train > 0
        stage2.fit(X_train.loc[pos_train], y_count_train.loc[pos_train])
        c2_train = np.maximum(stage2.predict(X_train), 0.0)
        c2_valid = np.maximum(stage2.predict(X_valid), 0.0)
        c2_test = np.maximum(stage2.predict(X_test), 0.0)

    # Final hurdle prediction = probability of any event * expected count if positive
    pred_train = p1_train * c2_train
    pred_valid = p1_valid * c2_valid
    pred_test = p1_test * c2_test

    base = make_base_prediction_frame(work, target_col)
    pred_df = add_prediction_columns(base, "regression", pred_train, pred_valid, pred_test, threshold=None)
    pred_df["stage1_any_prob"] = np.nan
    pred_df["stage2_count_if_positive"] = np.nan
    pred_df.loc[pred_df.split == "train", "stage1_any_prob"] = p1_train
    pred_df.loc[pred_df.split == "valid", "stage1_any_prob"] = p1_valid
    pred_df.loc[pred_df.split == "test", "stage1_any_prob"] = p1_test
    pred_df.loc[pred_df.split == "train", "stage2_count_if_positive"] = c2_train
    pred_df.loc[pred_df.split == "valid", "stage2_count_if_positive"] = c2_valid
    pred_df.loc[pred_df.split == "test", "stage2_count_if_positive"] = c2_test

    metrics = {
        "stage1_target": "y_next_air_drone_any",
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "train": regression_metrics(y_count_train.to_numpy(), pred_train),
        "valid": regression_metrics(y_count_valid.to_numpy(), pred_valid),
        "test": regression_metrics(y_count_test.to_numpy(), pred_test),
        "stage1_valid_ap": float(average_precision_score(y_any_valid, p1_valid)) if len(np.unique(y_any_valid)) > 1 else float("nan"),
        "stage1_valid_roc_auc": float(roc_auc_score(y_any_valid, p1_valid)) if len(np.unique(y_any_valid)) > 1 else float("nan"),
    }
    return pred_df, metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(args.master_csv)
    model1 = pd.read_csv(args.model1_csv)
    model2 = pd.read_csv(args.model2_csv)
    feature_spec = load_feature_spec(args.feature_spec_json)

    master, split_meta = assign_time_splits(master, args.valid_weeks, args.test_weeks)
    model1 = attach_master_context(master, model1)
    model2 = attach_master_context(master, model2)
    feature_cols_1, feature_cols_2 = get_feature_sets(feature_spec, model1, model2)

    acled_events = None
    if args.acled_events_csv:
        acled_events = normalize_id_columns(pd.read_csv(args.acled_events_csv))

    summary_rows: List[Dict[str, Any]] = []
    summary_json: Dict[str, Any] = {
        "summary": {
            "targets": args.targets,
            "split_meta": split_meta,
            "lightgbm_available": HAS_LIGHTGBM and not args.skip_lightgbm,
            "catboost_available": HAS_CATBOOST and not args.skip_catboost,
            "torch_available": HAS_TORCH and not args.skip_gru,
            "feature_counts": {"model1": len(feature_cols_1), "model2": len(feature_cols_2)},
        }
    }

    target_to_frames = {
        "model1_non_acled_only": (model1, feature_cols_1),
        "model2_plus_lagged_acled": (model2, feature_cols_2),
    }

    for target_col in args.targets:
        task_type = task_type_for_target(target_col)
        target_dir = outdir / target_col
        target_dir.mkdir(parents=True, exist_ok=True)

        target_result: Dict[str, Any] = {
            "task_type": task_type,
            "target_col": target_col,
            "split_meta": split_meta,
            "lightgbm_available": HAS_LIGHTGBM and not args.skip_lightgbm,
            "catboost_available": HAS_CATBOOST and not args.skip_catboost,
            "torch_available": HAS_TORCH and not args.skip_gru,
        }

        # Naive baseline always comes from the master table
        naive_pred_df, naive_metrics = run_naive_baseline(master, target_col, task_type, args)
        naive_name = "naive_baseline"
        naive_out = target_dir / f"{naive_name}.csv"
        save_export_csv(naive_pred_df, naive_out)
        if acled_events is not None:
            save_event_drilldown(
                naive_pred_df,
                acled_events,
                target_col,
                target_dir / f"{naive_name}__event_drilldown.csv",
                args.drilldown_splits
            )
        target_result[naive_name] = naive_metrics
        test_metrics = naive_metrics.get("test", {})
        summary_rows.append({"target_col": target_col, "experiment": naive_name, "algorithm": "naive", "split": "test", **test_metrics})

        # Run all model families for both feature-set variants
        for exp_name, (df_exp, feat_cols) in target_to_frames.items():
            exp_res: Dict[str, Any] = {"n_features": len(feat_cols), "feature_cols": feat_cols}

            for algo in ["linear", "lightgbm", "catboost"]:
                if algo == "lightgbm" and (not HAS_LIGHTGBM or args.skip_lightgbm):
                    continue
                if algo == "catboost" and (not HAS_CATBOOST or args.skip_catboost):
                    continue

                try:
                    pred_df, metrics = run_standard_experiment(df_exp, feat_cols, target_col, task_type, algo, args)
                except Exception as e:
                    exp_res[algo] = {"error": str(e)}
                    continue

                pred_df["model_experiment"] = exp_name
                pred_df["algorithm"] = algo
                out_csv = target_dir / f"{exp_name}__{algo}.csv"
                save_export_csv(pred_df, out_csv)

                if acled_events is not None:
                    save_event_drilldown(
                        pred_df,
                        acled_events,
                        target_col,
                        target_dir / f"{exp_name}__{algo}__event_drilldown.csv",
                        args.drilldown_splits
                    )

                exp_res[algo] = metrics
                summary_rows.append({"target_col": target_col, "experiment": exp_name, "algorithm": algo, "split": "test", **metrics.get("test", {})})

            if HAS_TORCH and not args.skip_gru:
                try:
                    pred_df, metrics = run_gru_experiment(df_exp, feat_cols, target_col, task_type, args, args.seed)
                    pred_df["model_experiment"] = exp_name
                    pred_df["algorithm"] = "gru"
                    out_csv = target_dir / f"{exp_name}__gru.csv"
                    save_export_csv(pred_df, out_csv)

                    if acled_events is not None:
                        save_event_drilldown(
                            pred_df,
                            acled_events,
                            target_col,
                            target_dir / f"{exp_name}__gru__event_drilldown.csv",
                            args.drilldown_splits
                        )

                    exp_res["gru"] = metrics
                    summary_rows.append({"target_col": target_col, "experiment": exp_name, "algorithm": "gru", "split": "test", **metrics.get("test", {})})
                except Exception as e:
                    exp_res["gru"] = {"error": str(e)}

            # Only run the hurdle model for air/drone strike count
            if target_col == "y_next_air_drone_strike_count":
                try:
                    pred_df, metrics = run_hurdle_experiment(df_exp, feat_cols, target_col, args)
                    pred_df["model_experiment"] = exp_name
                    pred_df["algorithm"] = "hurdle"
                    out_csv = target_dir / f"{exp_name}__hurdle.csv"
                    save_export_csv(pred_df, out_csv)

                    if acled_events is not None:
                        save_event_drilldown(
                            pred_df,
                            acled_events,
                            target_col,
                            target_dir / f"{exp_name}__hurdle__event_drilldown.csv",
                            args.drilldown_splits
                        )

                    exp_res["hurdle"] = metrics
                    summary_rows.append({"target_col": target_col, "experiment": exp_name, "algorithm": "hurdle", "split": "test", **metrics.get("test", {})})
                except Exception as e:
                    exp_res["hurdle"] = {"error": str(e)}

            target_result[exp_name] = exp_res

        with open(target_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(target_result, f, indent=2)

        summary_json[target_col] = target_result

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "comparison_metrics_all_targets.csv", index=False)

    with open(outdir / "all_results_multitarget.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(f"Saved combined metrics to: {outdir / 'comparison_metrics_all_targets.csv'}")
    print(f"Saved combined JSON to: {outdir / 'all_results_multitarget.json'}")
    print("Per-target prediction CSVs are under:", outdir)


if __name__ == "__main__":
    main()