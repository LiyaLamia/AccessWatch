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
    Dynamically load the base hierarchical forecasting script so this
    direct week-ahead missing-modality runner can reuse its helpers.
    """
    base_path = Path(path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base script not found: {base_path}")

    spec = importlib.util.spec_from_file_location("hier_base_direct_missing", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import base module from {base_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for running direct week-ahead
    missing-modality ablations.
    """
    p = argparse.ArgumentParser(
        description=(
            "Run direct week-ahead missing-modality ablations. "
            "Creates separate targets for week+1, week+2, ... and evaluates full vs dropped-modality feature sets "
            "for all available model families."
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
    Compute regression metrics after cleaning up bad prediction values.
    This helps avoid crashes or nonsense metrics when a model produces
    NaN, inf, or very large outputs.
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
    Each target refers to exactly one future week, not an aggregated window.
    """
    out = base.normalize_ids(master).sort_values(["raion_id", "week_start"]).copy()
    g = out.groupby("raion_id", sort=False)

    # Main hierarchical targets
    for name, cfg in base.MAIN_TARGETS.items():
        src = cfg["source_col"]
        shifted = g[src].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(-week_ahead))

        if cfg["task_type"] == "classification":
            out[f"actual_{name}"] = base.binary_or_nan(shifted)

            # Naive baseline for binary tasks = whether it happened this week
            out[f"naive_source_{name}"] = (pd.to_numeric(out[src], errors="coerce").fillna(0) > 0).astype(float)
        else:
            out[f"actual_{name}"] = shifted

            # Naive baseline for count/regression tasks = carry forward current value
            out[f"naive_source_{name}"] = pd.to_numeric(out[src], errors="coerce").fillna(0)

    # Subtype targets: keep both binary next-week occurrence and shifted counts
    for name, cfg in base.SUBTYPE_TARGETS.items():
        src = cfg["source_col"]
        shifted = g[src].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(-week_ahead))

        out[f"actual_{name}"] = base.binary_or_nan(shifted)
        out[f"naive_source_{name}"] = (pd.to_numeric(out[src], errors="coerce").fillna(0) > 0).astype(float)
        out[f"actual_{name.replace('_any', '_count')}"] = shifted

    # A helper binary target for whether fatalities occur at all
    out["actual_fatalities_any"] = base.binary_or_nan(out["actual_fatalities_sum"])
    out["naive_source_fatalities_any"] = (pd.to_numeric(out["fatalities_sum"], errors="coerce").fillna(0) > 0).astype(float)

    # Direct week-ahead target window covers exactly one week
    out["target_window_start"] = out["week_start"] + pd.Timedelta(days=7 * week_ahead)
    out["target_window_end"] = out["target_window_start"] + pd.Timedelta(days=6)

    # Keep metadata describing this direct horizon
    out["forecast_window_weeks"] = 1
    out["forecast_window_label"] = f"week_plus_{week_ahead}"
    out["forecast_week_ahead"] = week_ahead

    # Drop rows where the future target is unavailable
    return out[out["actual_any_event"].notna()].copy()


def prefixed(cols: Sequence[str], prefixes: Sequence[str]) -> List[str]:
    """
    Keep only columns starting with the given prefixes.
    """
    out = []
    for c in cols:
        lo = c.lower()
        if any(lo.startswith(p) for p in prefixes):
            out.append(c)
    return sorted(set(out))


def without_prefixes(cols: Sequence[str], prefixes: Sequence[str]) -> List[str]:
    """
    Drop columns starting with the given prefixes.
    """
    out = []
    for c in cols:
        lo = c.lower()
        if not any(lo.startswith(p) for p in prefixes):
            out.append(c)
    return sorted(set(out))


def make_missing_feature_groups(model1_features: List[str], model2_features: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    Build the feature subsets used for missing-modality ablation.
    Each config removes one modality (or a pair of modalities)
    from the full feature set.
    """
    firms_prefixes = ("firms_",)
    ntl_prefixes = ("ntl_",)
    unosat_prefixes = ("unosat_",)

    return {
        "model1": {
            "full": sorted(set(model1_features)),
            "drop_ntl": without_prefixes(model1_features, ntl_prefixes),
            "drop_firms": without_prefixes(model1_features, firms_prefixes),
            "drop_ntl_firms": without_prefixes(model1_features, ntl_prefixes + firms_prefixes),
            "drop_unosat": without_prefixes(model1_features, unosat_prefixes),
        },
        "model2": {
            "full": sorted(set(model2_features)),
            "drop_ntl": without_prefixes(model2_features, ntl_prefixes),
            "drop_firms": without_prefixes(model2_features, firms_prefixes),
            "drop_ntl_firms": without_prefixes(model2_features, ntl_prefixes + firms_prefixes),
            "drop_unosat": without_prefixes(model2_features, unosat_prefixes),
        },
    }


def choose_algorithms(base, args) -> List[str]:
    """
    Build the list of model families that are available
    under the current environment and flags.
    """
    algos = ["linear_hierarchical"]

    if base.HAS_LIGHTGBM and not args.skip_lightgbm:
        algos.append("lightgbm_hierarchical")

    if base.HAS_CATBOOST and not args.skip_catboost:
        algos.append("catboost_hierarchical")

    if base.HAS_TORCH and not args.skip_gru:
        algos.append("gru_hierarchical")

    if base.HAS_TORCH and not args.skip_tcn:
        algos.append("tcn_hierarchical")

    return algos


def run_algo(base, algo_name: str, df_model: pd.DataFrame, feat_cols: List[str], dyn: List[str], stat: List[str], args, device):
    """
    Dispatch to the correct training routine for the selected algorithm.
    """
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


def extract_summary_row(week_ahead: int, config_name: str, model_set: str, algorithm: str, split_name: str, outputs) -> Dict[str, Any]:
    """
    Flatten one split's metrics into a compact summary row
    for later comparison across horizons, models, and dropped modalities.
    """
    m = outputs.metrics[split_name]

    row: Dict[str, Any] = {
        "ablation_family": "direct_week_ahead_missing_modality",
        "forecast_week_ahead": week_ahead,
        "forecast_horizon_label": f"week_plus_{week_ahead}",
        "missing_config": config_name,
        "model_set": model_set,
        "algorithm": algorithm,
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


def print_banner(title: str) -> None:
    """
    Print a large separator so long experiment runs are easier to follow.
    """
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92, flush=True)


def persist_tables(outdir: Path, compact_rows: List[Dict[str, Any]], detailed_rows: List[Dict[str, Any]]) -> None:
    """
    Persist the rolling summary tables after progress is made.
    This way partial results are not lost if a long run stops midway.
    """
    compact_df = pd.DataFrame(compact_rows)
    detailed_df = pd.DataFrame(detailed_rows)

    compact_df.to_csv(outdir / "direct_week_ahead_missing_modality_test.csv", index=False)
    detailed_df.to_csv(outdir / "direct_week_ahead_missing_modality_all_splits.csv", index=False)

    if not compact_df.empty:
        compact_df.sort_values(
            ["forecast_week_ahead", "any_event_f1", "high_intensity_f1", "subtype_macro_f1"],
            ascending=[True, False, False, False],
        ).to_csv(outdir / "direct_week_ahead_missing_leaderboard_test.csv", index=False)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading base trainer...")
    base = load_base_module(args.base_script)

    # Patch the base script's regression metrics with the safer version above
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

    missing_groups = make_missing_feature_groups(model1_features, model2_features)
    algorithms = choose_algorithms(base, args)

    manifest: Dict[str, Any] = {
        "summary": {
            "forecast_type": "direct_week_ahead_missing_modality",
            "week_aheads": args.week_aheads,
            "algorithms": algorithms,
            "split_meta": split_meta,
            "feature_counts": {"model1": len(model1_features), "model2": len(model2_features)},
            "missing_configs": list(missing_groups["model1"].keys()),
        },
        "horizons": {},
    }

    compact_rows: List[Dict[str, Any]] = []
    detailed_rows: List[Dict[str, Any]] = []

    print(f"Week-ahead horizons: {args.week_aheads}")
    print(f"Algorithms: {algorithms}")
    print(f"Missing configs: {list(missing_groups['model1'].keys())}")

    for week_ahead in args.week_aheads:
        label = f"week_plus_{week_ahead}"
        print_banner(f"Starting direct week-ahead missing-modality study | {label}")

        horizon_dir = outdir / label
        horizon_dir.mkdir(parents=True, exist_ok=True)

        # Build horizon-specific direct targets
        master_h = make_direct_week_master(base, master, week_ahead)

        # Attach those targets/context to the two feature tables
        model1_h = base.attach_context(master_h, model1)
        model2_h = base.attach_context(master_h, model2)

        payload_h: Dict[str, Any] = {
            "forecast_week_ahead": week_ahead,
            "forecast_horizon_label": label,
            "runs": {}
        }

        for model_set, df_model, feature_group in [
            ("model1", model1_h, missing_groups["model1"]),
            ("model2", model2_h, missing_groups["model2"]),
        ]:
            for missing_config, feat_cols in feature_group.items():
                if not feat_cols:
                    print(f"[SKIP] {label} | {model_set} | {missing_config} has empty feature set", flush=True)
                    continue

                dyn, stat = base.infer_dynamic_static(df_model, feat_cols)

                for algo_name in algorithms:
                    t0 = time.time()
                    print(
                        f"[START] horizon={label} | model_set={model_set} | missing={missing_config} | "
                        f"algo={algo_name} | n_features={len(feat_cols)}",
                        flush=True,
                    )

                    outputs = run_algo(base, algo_name, df_model, feat_cols, dyn, stat, args, device)

                    run_key = f"{model_set}__{missing_config}__{algo_name}"
                    payload_h["runs"][run_key] = {
                        "model_set": model_set,
                        "missing_config": missing_config,
                        "algorithm": algo_name,
                        "n_features": len(feat_cols),
                        "dynamic_cols": dyn,
                        "static_cols": stat,
                        "thresholds": outputs.thresholds,
                        "metrics": outputs.metrics,
                    }

                    # Save one CSV per split for this run
                    for split_name, ex in outputs.exports.items():
                        df_out = ex.copy()
                        df_out["model_set"] = model_set
                        df_out["missing_config"] = missing_config
                        df_out["algorithm"] = algo_name
                        df_out["forecast_type"] = "direct_week_ahead_missing_modality"
                        df_out["forecast_week_ahead"] = week_ahead
                        df_out.to_csv(horizon_dir / f"{run_key}__{split_name}.csv", index=False)

                    # Save both detailed and compact summaries
                    for split_name in ["train", "valid", "test"]:
                        row = extract_summary_row(week_ahead, missing_config, model_set, algo_name, split_name, outputs)
                        detailed_rows.append(row)
                        if split_name == "test":
                            compact_rows.append(row)

                    report = outputs.metrics["test"]
                    print(
                        f"[DONE] elapsed={time.time() - t0:.1f}s | split=test | "
                        f"any_event_f1={report['any_event']['f1']:.4f} | "
                        f"high_intensity_f1={report['high_intensity']['f1']:.4f} | "
                        f"subtype_macro_f1={report['subtype_cumulative']['subtype_macro_f1']:.4f}",
                        flush=True,
                    )

                    # Persist progress after each run
                    persist_tables(outdir, compact_rows, detailed_rows)

        with open(horizon_dir / "all_results_direct_week_ahead_missing_modality.json", "w", encoding="utf-8") as f:
            json.dump(payload_h, f, indent=2)

        manifest["horizons"][label] = {
            "results_dir": str(horizon_dir),
            "json_results": str(horizon_dir / "all_results_direct_week_ahead_missing_modality.json"),
        }

        persist_tables(outdir, compact_rows, detailed_rows)

    with open(outdir / "direct_week_ahead_missing_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nSaved:")
    print(f"- {outdir / 'direct_week_ahead_missing_modality_test.csv'}")
    print(f"- {outdir / 'direct_week_ahead_missing_modality_all_splits.csv'}")
    print(f"- {outdir / 'direct_week_ahead_missing_leaderboard_test.csv'}")
    print(f"- {outdir / 'direct_week_ahead_missing_manifest.json'}")


if __name__ == "__main__":
    main()