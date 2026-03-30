#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_base_module(path: str):
    """
    Dynamically load the main hierarchical forecasting script so this
    direct week-ahead wrapper can reuse its training utilities.
    """
    base_path = Path(path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base script not found: {base_path}")

    spec = importlib.util.spec_from_file_location("hier_base_direct", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import base module from {base_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for direct week-ahead forecasting.
    Unlike the multiwindow setup, this predicts separate week +1, +2, +3, etc.
    """
    p = argparse.ArgumentParser(
        description=(
            "Direct week-ahead hierarchical forecasting. "
            "Builds separate targets for week +1, +2, +3, ... instead of aggregated future windows."
        )
    )
    p.add_argument("--base_script", default=str(Path(__file__).with_name("19_train_export_hierarchical_multitask_multiwindow_v2.py")))
    p.add_argument("--master_csv", required=True)
    p.add_argument("--model1_csv", required=True)
    p.add_argument("--model2_csv", required=True)
    p.add_argument("--feature_spec_json", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--week_aheads", nargs="+", type=int, default=[1, 2, 3, 4], help="Direct future weeks to predict, e.g. 1 2 3 4.")
    p.add_argument("--valid_weeks", type=int, default=13)
    p.add_argument("--test_weeks", type=int, default=13)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--top_frac", type=float, default=0.2)
    p.add_argument("--skip_naive", action="store_true")
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


def safe_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Safer regression metrics for count-style predictions.
    This guards against NaNs, infinities, and extreme values
    before computing MAE/RMSE/R2.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=0.0)
    y_pred = np.clip(y_pred, 0.0, 1e6)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "mean_target": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "mean_prediction": float(np.mean(y_pred)) if len(y_pred) else float("nan"),
        "max_prediction": float(np.max(y_pred)) if len(y_pred) else float("nan"),
        "n_rows": int(len(y_true)),
    }


def make_direct_week_master(base, master: pd.DataFrame, week_ahead: int) -> pd.DataFrame:
    """
    Build a horizon-specific master table for direct forecasting.
    For example, week_ahead=3 means the target is exactly 3 weeks ahead,
    not an aggregate over weeks 1..3.
    """
    out = base.normalize_ids(master).sort_values(["raion_id", "week_start"]).copy()
    g = out.groupby("raion_id", sort=False)

    # Main targets from the base hierarchy
    for name, cfg in base.MAIN_TARGETS.items():
        src = cfg["source_col"]

        # Direct shift to the requested future week
        shifted = g[src].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(-week_ahead))

        if cfg["task_type"] == "classification":
            out[f"actual_{name}"] = base.binary_or_nan(shifted)

            # Naive baseline for binary tasks: did it happen this week?
            out[f"naive_source_{name}"] = (pd.to_numeric(out[src], errors="coerce").fillna(0) > 0).astype(float)
        else:
            out[f"actual_{name}"] = shifted

            # Naive baseline for regression/count tasks: carry forward current value
            out[f"naive_source_{name}"] = pd.to_numeric(out[src], errors="coerce").fillna(0)

    # Subtype targets, both binary and raw shifted counts
    for name, cfg in base.SUBTYPE_TARGETS.items():
        src = cfg["source_col"]
        shifted = g[src].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(-week_ahead))

        out[f"actual_{name}"] = base.binary_or_nan(shifted)
        out[f"naive_source_{name}"] = (pd.to_numeric(out[src], errors="coerce").fillna(0) > 0).astype(float)

        # Keep the actual count version too for reporting/inspection
        out[f"actual_{name.replace('_any', '_count')}"] = shifted

    # Fatalities-any is derived from the shifted fatalities count
    out["actual_fatalities_any"] = base.binary_or_nan(out["actual_fatalities_sum"])
    out["naive_source_fatalities_any"] = (pd.to_numeric(out["fatalities_sum"], errors="coerce").fillna(0) > 0).astype(float)

    # Direct target window is exactly one week wide
    out["target_window_start"] = out["week_start"] + pd.Timedelta(days=7 * week_ahead)
    out["target_window_end"] = out["target_window_start"] + pd.Timedelta(days=6)

    out["forecast_window_weeks"] = week_ahead
    out["forecast_window_label"] = f"week_plus_{week_ahead}"
    out["forecast_week_ahead"] = week_ahead

    # Drop rows where the future target is unavailable
    return out[out["actual_any_event"].notna()].copy()


def choose_algorithms(base, args) -> List[str]:
    """
    Build the list of algorithms that should actually run
    under the current environment and command-line flags.
    """
    algos = []

    if not args.skip_naive:
        algos.append("naive_hierarchical")

    algos.append("linear_hierarchical")

    if base.HAS_LIGHTGBM and not args.skip_lightgbm:
        algos.append("lightgbm_hierarchical")

    if base.HAS_CATBOOST and not args.skip_catboost:
        algos.append("catboost_hierarchical")

    if base.HAS_TORCH and not args.skip_gru:
        algos.append("gru_hierarchical")

    if base.HAS_TORCH and not args.skip_tcn:
        algos.append("tcn_hierarchical")

    return algos


def summarize_row(
    horizon_label: str,
    week_ahead: int,
    model_name: str,
    algo_name: str,
    split_name: str,
    outputs,
) -> Dict[str, Any]:
    """
    Flatten a split's metrics into one summary row
    so later comparison across horizons/models is easy.
    """
    m = outputs.metrics[split_name]

    row = {
        "forecast_horizon_label": horizon_label,
        "forecast_week_ahead": week_ahead,
        "model_experiment": model_name,
        "algorithm": algo_name,
        "split": split_name,
        "any_event_f1": m["any_event"].get("f1"),
        "high_intensity_f1": m["high_intensity"].get("f1"),
        "any_event_avg_precision": m["any_event"].get("avg_precision"),
        "high_intensity_avg_precision": m["high_intensity"].get("avg_precision"),
        "subtype_macro_f1": m["subtype_cumulative"].get("subtype_macro_f1"),
        "subtype_weighted_f1": m["subtype_cumulative"].get("subtype_weighted_f1"),
        "subtype_micro_f1": m["subtype_cumulative"].get("subtype_micro_f1"),
        "battle_any_f1": m["battle_any"].get("f1"),
        "explosions_remote_any_f1": m["explosions_remote_any"].get("f1"),
        "violence_against_civilians_any_f1": m["violence_against_civilians_any"].get("f1"),
        "air_drone_any_f1": m["air_drone_any"].get("f1"),
        "strategic_developments_any_f1": m["strategic_developments_any"].get("f1"),
        "event_count_mae": m["event_count"].get("mae"),
        "fatalities_sum_mae": m["fatalities_sum"].get("mae"),
        "air_drone_strike_count_mae": m["air_drone_strike_count"].get("mae"),
        "frac_high_gt_event": m.get("hierarchy_diagnostics", {}).get("frac_high_gt_event"),
    }
    return row


def run_algo(base, algo_name: str, df_model: pd.DataFrame, feat_cols: List[str], dyn: List[str], stat: List[str], args, device):
    """
    Dispatch to the correct training/prediction routine
    based on the selected algorithm name.
    """
    if algo_name == "naive_hierarchical":
        return base.build_naive(df_model, args.top_k, args.top_frac)

    if algo_name == "linear_hierarchical":
        return base.train_hier_tabular(df_model, feat_cols, "linear", args.seed, args.top_k, args.top_frac)

    if algo_name == "lightgbm_hierarchical":
        return base.train_hier_tabular(df_model, feat_cols, "lightgbm", args.seed, args.top_k, args.top_frac)

    if algo_name == "catboost_hierarchical":
        return base.train_hier_tabular(df_model, feat_cols, "catboost", args.seed, args.top_k, args.top_frac)

    if algo_name == "gru_hierarchical":
        return base.train_hier_seq(df_model, feat_cols, dyn, stat, "gru", args, device, args.top_k, args.top_frac)

    if algo_name == "tcn_hierarchical":
        return base.train_hier_seq(df_model, feat_cols, dyn, stat, "tcn", args, device, args.top_k, args.top_frac)

    raise ValueError(f"Unsupported algorithm: {algo_name}")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading base trainer...")
    base = load_base_module(args.base_script)

    # Override the base regression metric function with the safer version above
    base.regression_metrics = safe_regression_metrics

    base.set_seed(args.seed)
    device = base.choose_device(args.device)

    print("Loading data...")
    master = base.normalize_ids(pd.read_csv(args.master_csv))
    model1 = base.normalize_ids(pd.read_csv(args.model1_csv))
    model2 = base.normalize_ids(pd.read_csv(args.model2_csv))
    feature_spec = base.load_feature_spec(args.feature_spec_json)
    model1_features, model2_features = base.get_feature_sets(feature_spec, model1, model2)
    master, split_meta = base.assign_time_splits(master, args.valid_weeks, args.test_weeks)

    algorithms = choose_algorithms(base, args)
    print(f"Week-ahead horizons: {args.week_aheads}")
    print(f"Algorithms: {algorithms}")

    combined_rows: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {
        "summary": {
            "forecast_type": "direct_week_ahead",
            "week_aheads": args.week_aheads,
            "split_meta": split_meta,
            "algorithms": algorithms,
            "feature_counts": {"model1": len(model1_features), "model2": len(model2_features)},
        },
        "horizons": {},
    }

    for week_ahead in args.week_aheads:
        label = f"week_plus_{week_ahead}"

        print("\n" + "=" * 88)
        print(f"Starting direct horizon {label}")
        print("=" * 88)

        horizon_dir = outdir / label
        horizon_dir.mkdir(parents=True, exist_ok=True)

        # Build direct week-ahead targets for this horizon
        master_h = make_direct_week_master(base, master, week_ahead)

        # Attach the horizon-specific targets/context back to the feature tables
        model1_h = base.attach_context(master_h, model1)
        model2_h = base.attach_context(master_h, model2)

        summary_rows_h: List[Dict[str, Any]] = []
        payload_h = {
            "forecast_week_ahead": week_ahead,
            "forecast_horizon_label": label,
            "models": {}
        }

        for model_name, df_model, feat_cols in [
            ("model1_non_acled_only", model1_h, model1_features),
            ("model2_plus_lagged_acled", model2_h, model2_features),
        ]:
            dyn, stat = base.infer_dynamic_static(df_model, feat_cols)

            payload_h["models"][model_name] = {
                "n_features": len(feat_cols),
                "dynamic_cols": dyn,
                "static_cols": stat,
            }

            for algo_name in algorithms:
                t0 = time.time()
                print(f"[START] horizon={label} | model={model_name} | algo={algo_name}")

                outputs = run_algo(base, algo_name, df_model, feat_cols, dyn, stat, args, device)

                # Save one CSV per split
                for split_name, ex in outputs.exports.items():
                    df_out = ex.copy()
                    df_out["model_experiment"] = model_name
                    df_out["algorithm"] = algo_name
                    df_out["forecast_type"] = "direct_week_ahead"
                    df_out["forecast_week_ahead"] = week_ahead
                    df_out.to_csv(horizon_dir / f"{model_name}__{algo_name}__{split_name}.csv", index=False)

                # Save metrics to the JSON payload
                payload_h["models"].setdefault(model_name, {})
                payload_h["models"][model_name][algo_name] = {
                    "thresholds": outputs.thresholds,
                    "metrics": outputs.metrics,
                }

                # Build summary rows for all splits
                for split_name in ["train", "valid", "test"]:
                    row = summarize_row(label, week_ahead, model_name, algo_name, split_name, outputs)
                    summary_rows_h.append(row)
                    combined_rows.append(row)

                report = outputs.metrics["test"]
                print(
                    f"[DONE] elapsed={time.time() - t0:.1f}s | "
                    f"any_event_f1={report['any_event']['f1']:.4f} | "
                    f"high_intensity_f1={report['high_intensity']['f1']:.4f} | "
                    f"subtype_macro_f1={report['subtype_cumulative']['subtype_macro_f1']:.4f}"
                )

        cmp_h = pd.DataFrame(summary_rows_h)
        cmp_h.to_csv(horizon_dir / "comparison_metrics_direct_week_ahead.csv", index=False)

        with open(horizon_dir / "all_results_direct_week_ahead.json", "w", encoding="utf-8") as f:
            json.dump(payload_h, f, indent=2)

        payload["horizons"][label] = {
            "results_dir": str(horizon_dir),
            "comparison_metrics_csv": str(horizon_dir / "comparison_metrics_direct_week_ahead.csv"),
            "json_results": str(horizon_dir / "all_results_direct_week_ahead.json"),
        }

    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(outdir / "comparison_metrics_direct_week_ahead_all_horizons.csv", index=False)

    if not combined_df.empty:
        leaderboard = combined_df[combined_df["split"] == "test"].sort_values(
            ["forecast_week_ahead", "any_event_f1", "high_intensity_f1", "subtype_macro_f1"],
            ascending=[True, False, False, False],
        )
        leaderboard.to_csv(outdir / "leaderboard_direct_week_ahead_test.csv", index=False)

    with open(outdir / "all_results_direct_week_ahead_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nSaved:")
    print(f"- {outdir / 'comparison_metrics_direct_week_ahead_all_horizons.csv'}")
    print(f"- {outdir / 'leaderboard_direct_week_ahead_test.csv'}")
    print(f"- {outdir / 'all_results_direct_week_ahead_summary.json'}")


if __name__ == "__main__":
    main()