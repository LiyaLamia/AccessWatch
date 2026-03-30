#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def load_base_module(path: str):
    """
    Dynamically load the base hierarchical training script so this ablation
    wrapper can reuse its functions instead of duplicating the full pipeline.
    """
    base_path = Path(path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base script not found: {base_path}")

    spec = importlib.util.spec_from_file_location("hier_base", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import base module from {base_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the ablation study runner.
    """
    p = argparse.ArgumentParser(
        description=(
            "Run a P8-focused ablation study using the updated hierarchical multiwindow classifier script. "
            "By default this skips naive reporting and writes summary CSVs for fusion, temporal, architecture, "
            "constraint, and missing-modality ablations."
        )
    )
    p.add_argument("--base_script", default=str(Path(__file__).with_name("19_train_export_hierarchical_multitask_multiwindow_v2.py")))
    p.add_argument("--master_csv", required=True)
    p.add_argument("--model1_csv", required=True)
    p.add_argument("--model2_csv", required=True)
    p.add_argument("--feature_spec_json", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--window_weeks", nargs="+", type=int, default=[1, 2, 4], help="Forecast horizons to evaluate. Default: 1, 2, 4")
    p.add_argument("--report_split", default="test", choices=["train", "valid", "test"], help="Primary split for compact ablation summaries")
    p.add_argument("--valid_weeks", type=int, default=13)
    p.add_argument("--test_weeks", type=int, default=13)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--top_frac", type=float, default=0.2)
    p.add_argument("--seq_lens", nargs="+", type=int, default=[4, 8, 12], help="Sequence lengths for temporal ablation")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--skip_lightgbm", action="store_true")
    p.add_argument("--skip_catboost", action="store_true")
    p.add_argument("--skip_gru", action="store_true")
    p.add_argument("--skip_tcn", action="store_true")
    p.add_argument("--families", nargs="*", default=None, help="Optional subset of ablation families: fusion temporal architecture constraint missing")
    return p.parse_args()


def prefixed(cols: Sequence[str], prefixes: Sequence[str]) -> List[str]:
    """
    Keep only feature columns whose names start with one of the given prefixes.
    """
    out = []
    for c in cols:
        lo = c.lower()
        if any(lo.startswith(p) for p in prefixes):
            out.append(c)
    return sorted(set(out))


def without_prefixes(cols: Sequence[str], prefixes: Sequence[str]) -> List[str]:
    """
    Drop feature columns whose names start with one of the given prefixes.
    """
    out = []
    for c in cols:
        lo = c.lower()
        if not any(lo.startswith(p) for p in prefixes):
            out.append(c)
    return sorted(set(out))


def make_feature_groups(model1_features: List[str], model2_features: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    Organize features into modality-based groups so the different ablation
    families can reuse the same feature selections consistently.
    """
    infra_prefixes = (
        "road_", "rail_", "major_road_", "nearest_border_", "border_crossing", "paved_", "place_", "places_", "pop_"
    )
    firms_prefixes = ("firms_",)
    ntl_prefixes = ("ntl_",)
    unosat_prefixes = ("unosat_",)
    lagged_acled_prefixes = ("acled_", "fatalities_", "air_drone_", "any_event", "high_intensity")

    groups = {
        "model1": {
            "firms_only": prefixed(model1_features, firms_prefixes),
            "ntl_only": prefixed(model1_features, ntl_prefixes),
            "infrastructure_exposure_only": prefixed(model1_features, infra_prefixes),
            "unosat_only": prefixed(model1_features, unosat_prefixes),
            "firms_plus_ntl": sorted(set(prefixed(model1_features, firms_prefixes + ntl_prefixes))),
            "all_non_acled_modalities": sorted(set(model1_features)),
            "all_non_acled_drop_ntl": without_prefixes(model1_features, ntl_prefixes),
            "all_non_acled_drop_firms": without_prefixes(model1_features, firms_prefixes),
            "all_non_acled_drop_ntl_firms": without_prefixes(model1_features, ntl_prefixes + firms_prefixes),
            "all_non_acled_drop_unosat": without_prefixes(model1_features, unosat_prefixes),
        },
        "model2": {
            "all_non_acled_plus_lagged_acled": sorted(set(model2_features)),
            "all_plus_lagged_drop_ntl": without_prefixes(model2_features, ntl_prefixes),
            "all_plus_lagged_drop_firms": without_prefixes(model2_features, firms_prefixes),
            "all_plus_lagged_drop_ntl_firms": without_prefixes(model2_features, ntl_prefixes + firms_prefixes),
            "all_plus_lagged_drop_unosat": without_prefixes(model2_features, unosat_prefixes),
            "lagged_acled_only": prefixed(model2_features, lagged_acled_prefixes),
            "lagged_acled_plus_firms_ntl": sorted(set(prefixed(model2_features, lagged_acled_prefixes + firms_prefixes + ntl_prefixes))),
        },
    }
    return groups


def ensure_nonempty(name: str, cols: List[str]) -> List[str]:
    """
    Fail fast if an ablation config ends up with zero usable features.
    """
    if not cols:
        raise ValueError(f"Feature set is empty for ablation config: {name}")
    return cols


def extract_summary_row(
    ablation_family: str,
    ablation_name: str,
    model_set: str,
    constraint: str,
    algorithm: str,
    horizon: int,
    seq_len: Optional[int],
    split_name: str,
    outputs,
) -> Dict[str, Any]:
    """
    Flatten one experiment/split result into a single CSV-friendly row.
    """
    m = outputs.metrics[split_name]
    row: Dict[str, Any] = {
        "ablation_family": ablation_family,
        "ablation_name": ablation_name,
        "model_set": model_set,
        "constraint": constraint,
        "algorithm": algorithm,
        "forecast_window_weeks": horizon,
        "seq_len": seq_len,
        "split": split_name,
    }

    for target in ["any_event", "high_intensity"]:
        stats = m[target]
        for key in ["accuracy", "precision", "recall", "f1", "avg_precision", "roc_auc", "positive_rate", "predicted_positive_rate"]:
            row[f"{target}_{key}"] = stats.get(key)

    for key in ["subtype_macro_f1", "subtype_weighted_f1", "subtype_micro_f1"]:
        row[key] = m["subtype_cumulative"].get(key)

    for s in ["battle_any", "explosions_remote_any", "violence_against_civilians_any", "air_drone_any", "strategic_developments_any"]:
        row[f"{s}_f1"] = m[s].get("f1")

    diag = m.get("hierarchy_diagnostics", {})
    row["frac_high_gt_event"] = diag.get("frac_high_gt_event")
    row["frac_any_subtype_gt_event"] = diag.get("frac_any_subtype_gt_event")

    for target in ["event_count", "fatalities_sum", "air_drone_strike_count"]:
        stats = m[target]
        for key in ["mae", "rmse", "r2"]:
            row[f"{target}_{key}"] = stats.get(key)

    return row


def attach_context_and_split(base, master_h: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Thin wrapper around the base script's helper for attaching
    horizon-specific targets/context back to a feature table.
    """
    return base.attach_context(master_h, model_df)


def phase_banner(title: str) -> None:
    """
    Print a big separator so long ablation runs are easier to follow in logs.
    """
    print(f"\n{'=' * 88}")
    print(title)
    print(f"{'=' * 88}", flush=True)


def run_status(action: str, ablation_family: str, ablation_name: str, model_set: str, algorithm: str, horizon: int, constraint: str, seq_len: Optional[int] = None) -> None:
    """
    Print a compact one-line status update before each experiment starts.
    """
    extra = f", seq_len={seq_len}" if seq_len is not None else ""
    print(
        f"[{action}] family={ablation_family} | config={ablation_name} | model_set={model_set} | "
        f"algo={algorithm} | constraint={constraint} | horizon={horizon}w{extra}",
        flush=True,
    )


def finish_status(start_time: float, outputs, split_name: str) -> None:
    """
    Print a short completion message with runtime and a few key metrics.
    """
    elapsed = time.perf_counter() - start_time
    try:
        m = outputs.metrics[split_name]
        event_f1 = m["any_event"].get("f1")
        high_f1 = m["high_intensity"].get("f1")
        subtype_macro = m["subtype_cumulative"].get("subtype_macro_f1")
        print(
            f"[DONE] elapsed={elapsed:.1f}s | split={split_name} | "
            f"any_event_f1={event_f1:.4f} | high_intensity_f1={high_f1:.4f} | subtype_macro_f1={subtype_macro:.4f}",
            flush=True,
        )
    except Exception:
        print(f"[DONE] elapsed={elapsed:.1f}s | split={split_name}", flush=True)


# ----------------------------
# Independent-head baseline for constraint ablation
# ----------------------------

def train_independent_tabular(base, df: pd.DataFrame, feature_cols: List[str], family: str, seed: int, top_k: int, top_frac: float):
    """
    Train an independent-head version of the tabular model.
    This is used for the constraint ablation to compare against
    the hierarchy-constrained version from the base script.
    """
    data = df.reset_index(drop=True).copy()
    X = base.to_numeric_frame(data, feature_cols)

    idx_train = data["split"] == "train"
    idx_valid = data["split"] == "valid"
    idx_test = data["split"] == "test"
    X_train = X.loc[idx_train]

    event_model = base.fit_binary_model(X_train, data.loc[idx_train, "actual_any_event"], family, seed)
    high_model = base.fit_binary_model(X_train, data.loc[idx_train, "actual_high_intensity"], family, seed + 1)
    fatal_any_model = base.fit_binary_model(X_train, data.loc[idx_train, "actual_fatalities_any"], family, seed + 2)

    subtype_models = {
        s: base.fit_binary_model(X_train, data.loc[idx_train, f"actual_{s}"], family, seed + 10 + i)
        for i, s in enumerate(base.SUBTYPE_ORDER)
    }

    # Count regressors are trained only on positive rows
    pos_event_count = idx_train & (pd.to_numeric(data["actual_event_count"], errors="coerce").fillna(0) > 0)
    pos_fatal = idx_train & (pd.to_numeric(data["actual_fatalities_sum"], errors="coerce").fillna(0) > 0)
    pos_air = idx_train & (pd.to_numeric(data["actual_air_drone_strike_count"], errors="coerce").fillna(0) > 0)

    event_count_reg = base.fit_positive_regressor(X.loc[pos_event_count], data.loc[pos_event_count, "actual_event_count"], family, seed + 30)
    fatal_reg = base.fit_positive_regressor(X.loc[pos_fatal], data.loc[pos_fatal, "actual_fatalities_sum"], family, seed + 31)
    air_reg = base.fit_positive_regressor(X.loc[pos_air], data.loc[pos_air, "actual_air_drone_strike_count"], family, seed + 32)

    exports: Dict[str, pd.DataFrame] = {}
    for split_name, mask in [("train", idx_train), ("valid", idx_valid), ("test", idx_test)]:
        keep_cols = [
            *base.ID_COLS,
            "split",
            "target_window_start",
            "target_window_end",
            "forecast_window_weeks",
            "forecast_window_label",
            *[c for c in base.RAW_CONTEXT_COLS if c in data.columns],
            *[f"actual_{k}" for k in base.MAIN_TARGETS],
            *[f"actual_{s}" for s in base.SUBTYPE_ORDER],
            *[f"actual_{s.replace('_any', '_count')}" for s in base.SUBTYPE_ORDER],
        ]
        ex = data.loc[mask, keep_cols].copy()
        Xs = X.loc[mask]

        ex["score_any_event"] = base.predict_binary(event_model, Xs)
        ex["score_high_intensity"] = base.predict_binary(high_model, Xs)
        ex["score_fatalities_any"] = base.predict_binary(fatal_any_model, Xs)

        for s, model in subtype_models.items():
            ex[f"score_{s}"] = base.predict_binary(model, Xs)

        # Counts are still gated by their related binary heads
        ex["pred_event_count"] = np.clip(ex["score_any_event"].to_numpy(dtype=float) * base.predict_positive_regressor(event_count_reg, Xs), 0.0, None)
        ex["pred_fatalities_sum"] = np.clip(ex["score_fatalities_any"].to_numpy(dtype=float) * base.predict_positive_regressor(fatal_reg, Xs), 0.0, None)
        ex["pred_air_drone_strike_count"] = np.clip(ex["score_air_drone_any"].to_numpy(dtype=float) * base.predict_positive_regressor(air_reg, Xs), 0.0, None)

        exports[split_name] = ex.reset_index(drop=True)

    thresholds = {
        "any_event": base.sweep_threshold(exports["valid"]["actual_any_event"].to_numpy(dtype=int), exports["valid"]["score_any_event"].to_numpy(dtype=float)),
        "high_intensity": base.sweep_threshold(exports["valid"]["actual_high_intensity"].to_numpy(dtype=int), exports["valid"]["score_high_intensity"].to_numpy(dtype=float)),
    }
    for s in base.SUBTYPE_ORDER:
        thresholds[s] = base.sweep_threshold(
            exports["valid"][f"actual_{s}"].to_numpy(dtype=int),
            exports["valid"][f"score_{s}"].to_numpy(dtype=float),
        )

    metrics: Dict[str, Any] = {}
    for split_name, ex0 in exports.items():
        ex = base.add_common_prediction_fields(ex0, thresholds)
        exports[split_name] = ex

        m = {
            "any_event": base.classification_metrics(ex[["week_start"]], ex["actual_any_event"].to_numpy(dtype=int), ex["score_any_event"].to_numpy(dtype=float), thresholds["any_event"], top_k, top_frac),
            "high_intensity": base.classification_metrics(ex[["week_start"]], ex["actual_high_intensity"].to_numpy(dtype=int), ex["score_high_intensity"].to_numpy(dtype=float), thresholds["high_intensity"], top_k, top_frac),
            "event_count": base.regression_metrics(ex["actual_event_count"].to_numpy(dtype=float), ex["pred_event_count"].to_numpy(dtype=float)),
            "fatalities_sum": base.regression_metrics(ex["actual_fatalities_sum"].to_numpy(dtype=float), ex["pred_fatalities_sum"].to_numpy(dtype=float)),
            "air_drone_strike_count": base.regression_metrics(ex["actual_air_drone_strike_count"].to_numpy(dtype=float), ex["pred_air_drone_strike_count"].to_numpy(dtype=float)),
        }

        for s in base.SUBTYPE_ORDER:
            m[s] = base.classification_metrics(ex[["week_start"]], ex[f"actual_{s}"].to_numpy(dtype=int), ex[f"score_{s}"].to_numpy(dtype=float), thresholds[s], top_k, top_frac)

        m["subtype_cumulative"] = base.multilabel_f1_metrics(ex, thresholds)
        m["hierarchy_diagnostics"] = {
            "frac_high_gt_event": float((ex["score_high_intensity"] > ex["score_any_event"] + 1e-9).mean()),
            "frac_any_subtype_gt_event": float(np.mean([(ex[f"score_{s}"] > ex["score_any_event"] + 1e-9).mean() for s in base.SUBTYPE_ORDER])),
        }
        metrics[split_name] = m

    return base.HierOutputs(thresholds, exports, metrics)


# ----------------------------
# Run blocks
# ----------------------------

def run_single(
    base,
    ablation_family: str,
    ablation_name: str,
    model_set: str,
    df_model: pd.DataFrame,
    feature_cols: List[str],
    algorithm: str,
    constraint: str,
    args,
    device,
):
    """
    Execute one ablation configuration and return its outputs.
    """
    feature_cols = ensure_nonempty(f"{ablation_family}:{ablation_name}:{model_set}", feature_cols)

    if algorithm in {"linear", "lightgbm", "catboost"}:
        if constraint == "hierarchical":
            outputs = base.train_hier_tabular(df_model, feature_cols, algorithm, args.seed, args.top_k, args.top_frac)
        elif constraint == "independent":
            outputs = train_independent_tabular(base, df_model, feature_cols, algorithm, args.seed, args.top_k, args.top_frac)
        else:
            raise ValueError(f"Unknown constraint mode: {constraint}")
        seq_len = None

    elif algorithm in {"gru", "tcn"}:
        if constraint != "hierarchical":
            raise ValueError("Independent constraint ablation is implemented for tabular models only.")
        if not base.HAS_TORCH:
            raise RuntimeError("PyTorch is not available for neural ablations")

        dyn, stat = base.infer_dynamic_static(df_model, feature_cols)
        outputs = base.train_hier_seq(df_model, feature_cols, dyn, stat, algorithm, args, device, args.top_k, args.top_frac)
        seq_len = args.seq_len
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return outputs, seq_len


def candidate_algorithms(base, args) -> List[str]:
    """
    Build the list of algorithms that are available under the current setup.
    """
    algos = ["linear"]
    if base.HAS_LIGHTGBM and not args.skip_lightgbm:
        algos.append("lightgbm")
    if base.HAS_CATBOOST and not args.skip_catboost:
        algos.append("catboost")
    if base.HAS_TORCH and not args.skip_gru:
        algos.append("gru")
    if base.HAS_TORCH and not args.skip_tcn:
        algos.append("tcn")
    return algos


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    phase_banner("Loading base module and input tables")
    base = load_base_module(args.base_script)
    base.set_seed(args.seed)
    device = base.choose_device(args.device)

    master = base.normalize_ids(pd.read_csv(args.master_csv))
    model1 = base.normalize_ids(pd.read_csv(args.model1_csv))
    model2 = base.normalize_ids(pd.read_csv(args.model2_csv))
    feature_spec = base.load_feature_spec(args.feature_spec_json)
    model1_features, model2_features = base.get_feature_sets(feature_spec, model1, model2)
    master, split_meta = base.assign_time_splits(master, args.valid_weeks, args.test_weeks)
    feature_groups = make_feature_groups(model1_features, model2_features)

    families_requested = set(args.families or ["fusion", "temporal", "architecture", "constraint", "missing"])
    all_algos = candidate_algorithms(base, args)
    tree_algos = [a for a in ["catboost", "lightgbm"] if a in all_algos]
    neural_algos = [a for a in ["gru", "tcn"] if a in all_algos]

    print(f"Requested families: {sorted(families_requested)}", flush=True)
    print(f"Forecast horizons: {args.window_weeks}", flush=True)
    print(f"Available algorithms: {all_algos}", flush=True)
    print(f"Report split: {args.report_split}", flush=True)

    detailed_rows: List[Dict[str, Any]] = []
    compact_rows: List[Dict[str, Any]] = []

    manifest: Dict[str, Any] = {
        "base_script": str(args.base_script),
        "report_split": args.report_split,
        "window_weeks": args.window_weeks,
        "seq_lens": args.seq_lens,
        "split_meta": split_meta,
        "available_algorithms": all_algos,
        "notes": {
            "naive_reported": False,
            "constraint_ablation": "independent vs hierarchical only for tabular families",
            "fusion_story": "all_non_acled vs all_non_acled_plus_lagged_acled answers the key added-value question",
        },
        "feature_groups": feature_groups,
    }

    for horizon in args.window_weeks:
        phase_banner(f"Starting horizon {horizon} week(s)")
        master_h = base.make_window_master(master, horizon)
        model1_h = attach_context_and_split(base, master_h, model1)
        model2_h = attach_context_and_split(base, master_h, model2)

        # Fusion ablation: compare different modality combinations
        if "fusion" in families_requested:
            phase_banner(f"Fusion ablation | horizon={horizon}w")
            fusion_plan = [
                ("model1", "firms_only", feature_groups["model1"]["firms_only"]),
                ("model1", "ntl_only", feature_groups["model1"]["ntl_only"]),
                ("model1", "infrastructure_exposure_only", feature_groups["model1"]["infrastructure_exposure_only"]),
                ("model1", "unosat_only", feature_groups["model1"]["unosat_only"]),
                ("model1", "firms_plus_ntl", feature_groups["model1"]["firms_plus_ntl"]),
                ("model1", "all_non_acled_modalities", feature_groups["model1"]["all_non_acled_modalities"]),
                ("model2", "all_non_acled_plus_lagged_acled", feature_groups["model2"]["all_non_acled_plus_lagged_acled"]),
            ]
            fusion_algos = list(all_algos)

            for model_set, ab_name, cols in fusion_plan:
                dfm = model1_h if model_set == "model1" else model2_h
                for algo in fusion_algos:
                    if algo in {"gru", "tcn"}:
                        args.seq_len = 8

                    run_status("START", "fusion", ab_name, model_set, algo, horizon, "hierarchical", seq_len=(args.seq_len if algo in {"gru", "tcn"} else None))
                    _t0 = time.perf_counter()
                    outputs, seq_len = run_single(base, "fusion", ab_name, model_set, dfm, cols, algo, "hierarchical", args, device)
                    finish_status(_t0, outputs, args.report_split)

                    compact_rows.append(extract_summary_row("fusion", ab_name, model_set, "hierarchical", algo, horizon, seq_len, args.report_split, outputs))
                    for split_name in ["train", "valid", "test"]:
                        detailed_rows.append(extract_summary_row("fusion", ab_name, model_set, "hierarchical", algo, horizon, seq_len, split_name, outputs))

        # Temporal ablation: vary sequence length for neural models
        if "temporal" in families_requested and neural_algos:
            phase_banner(f"Temporal ablation | horizon={horizon}w")
            for model_set, cols, dfm in [
                ("model1", feature_groups["model1"]["all_non_acled_modalities"], model1_h),
                ("model2", feature_groups["model2"]["all_non_acled_plus_lagged_acled"], model2_h),
            ]:
                for algo in neural_algos:
                    for seq_len in args.seq_lens:
                        args.seq_len = int(seq_len)

                        run_status("START", "temporal", f"seq{seq_len}", model_set, algo, horizon, "hierarchical", seq_len=seq_len)
                        _t0 = time.perf_counter()
                        outputs, seq_used = run_single(base, "temporal", f"seq{seq_len}", model_set, dfm, cols, algo, "hierarchical", args, device)
                        finish_status(_t0, outputs, args.report_split)

                        compact_rows.append(extract_summary_row("temporal", f"seq{seq_len}", model_set, "hierarchical", algo, horizon, seq_used, args.report_split, outputs))
                        for split_name in ["train", "valid", "test"]:
                            detailed_rows.append(extract_summary_row("temporal", f"seq{seq_len}", model_set, "hierarchical", algo, horizon, seq_used, split_name, outputs))

        # Architecture ablation: compare algorithm families directly
        if "architecture" in families_requested:
            phase_banner(f"Architecture ablation | horizon={horizon}w")
            arch_plan = [
                ("model1", feature_groups["model1"]["all_non_acled_modalities"], model1_h),
                ("model2", feature_groups["model2"]["all_non_acled_plus_lagged_acled"], model2_h),
            ]
            for model_set, cols, dfm in arch_plan:
                for algo in all_algos:
                    if algo in {"gru", "tcn"}:
                        args.seq_len = 8

                    run_status("START", "architecture", f"{algo}", model_set, algo, horizon, "hierarchical", seq_len=(args.seq_len if algo in {"gru", "tcn"} else None))
                    _t0 = time.perf_counter()
                    outputs, seq_used = run_single(base, "architecture", f"{algo}", model_set, dfm, cols, algo, "hierarchical", args, device)
                    finish_status(_t0, outputs, args.report_split)

                    compact_rows.append(extract_summary_row("architecture", f"{algo}", model_set, "hierarchical", algo, horizon, seq_used, args.report_split, outputs))
                    for split_name in ["train", "valid", "test"]:
                        detailed_rows.append(extract_summary_row("architecture", f"{algo}", model_set, "hierarchical", algo, horizon, seq_used, split_name, outputs))

        # Constraint ablation: independent heads vs hierarchy-constrained heads
        if "constraint" in families_requested:
            phase_banner(f"Constraint ablation | horizon={horizon}w")
            constraint_algos = [a for a in ["linear", "catboost", "lightgbm"] if a in all_algos]
            constraint_plan = [
                ("model1", feature_groups["model1"]["all_non_acled_modalities"], model1_h),
                ("model2", feature_groups["model2"]["all_non_acled_plus_lagged_acled"], model2_h),
            ]
            for model_set, cols, dfm in constraint_plan:
                for algo in constraint_algos:
                    for constraint in ["independent", "hierarchical"]:
                        run_status("START", "constraint", constraint, model_set, algo, horizon, constraint)
                        _t0 = time.perf_counter()
                        outputs, seq_used = run_single(base, "constraint", constraint, model_set, dfm, cols, algo, constraint, args, device)
                        finish_status(_t0, outputs, args.report_split)

                        compact_rows.append(extract_summary_row("constraint", constraint, model_set, constraint, algo, horizon, seq_used, args.report_split, outputs))
                        for split_name in ["train", "valid", "test"]:
                            detailed_rows.append(extract_summary_row("constraint", constraint, model_set, constraint, algo, horizon, seq_used, split_name, outputs))

        # Missing-modality ablation: remove one modality at a time
        if "missing" in families_requested:
            phase_banner(f"Missing-modality ablation | horizon={horizon}w")
            miss_plan = [
                ("model1", "drop_ntl", feature_groups["model1"]["all_non_acled_drop_ntl"]),
                ("model1", "drop_firms", feature_groups["model1"]["all_non_acled_drop_firms"]),
                ("model1", "drop_ntl_firms", feature_groups["model1"]["all_non_acled_drop_ntl_firms"]),
                ("model1", "drop_unosat", feature_groups["model1"]["all_non_acled_drop_unosat"]),
                ("model2", "drop_ntl", feature_groups["model2"]["all_plus_lagged_drop_ntl"]),
                ("model2", "drop_firms", feature_groups["model2"]["all_plus_lagged_drop_firms"]),
                ("model2", "drop_ntl_firms", feature_groups["model2"]["all_plus_lagged_drop_ntl_firms"]),
                ("model2", "drop_unosat", feature_groups["model2"]["all_plus_lagged_drop_unosat"]),
            ]
            missing_algos = list(all_algos)

            for model_set, ab_name, cols in miss_plan:
                dfm = model1_h if model_set == "model1" else model2_h
                for algo in missing_algos:
                    if algo in {"gru", "tcn"}:
                        args.seq_len = 8

                    run_status("START", "missing_modality", ab_name, model_set, algo, horizon, "hierarchical", seq_len=(args.seq_len if algo in {"gru", "tcn"} else None))
                    _t0 = time.perf_counter()
                    outputs, seq_used = run_single(base, "missing_modality", ab_name, model_set, dfm, cols, algo, "hierarchical", args, device)
                    finish_status(_t0, outputs, args.report_split)

                    compact_rows.append(extract_summary_row("missing_modality", ab_name, model_set, "hierarchical", algo, horizon, seq_used, args.report_split, outputs))
                    for split_name in ["train", "valid", "test"]:
                        detailed_rows.append(extract_summary_row("missing_modality", ab_name, model_set, "hierarchical", algo, horizon, seq_used, split_name, outputs))

    compact_df = pd.DataFrame(compact_rows)
    detailed_df = pd.DataFrame(detailed_rows)

    phase_banner("Writing ablation summary files")
    compact_path = outdir / f"ablation_summary_{args.report_split}.csv"
    detailed_path = outdir / "ablation_summary_all_splits.csv"
    compact_df.to_csv(compact_path, index=False)
    detailed_df.to_csv(detailed_path, index=False)

    with open(outdir / "ablation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Convenience exports by family plus one overall leaderboard
    if not compact_df.empty:
        leaderboard = compact_df.sort_values(
            ["any_event_f1", "high_intensity_f1", "subtype_macro_f1"],
            ascending=[False, False, False],
        )
        leaderboard.to_csv(outdir / f"ablation_leaderboard_{args.report_split}.csv", index=False)

        fusion_view = compact_df[compact_df["ablation_family"] == "fusion"].copy()
        temporal_view = compact_df[compact_df["ablation_family"] == "temporal"].copy()
        arch_view = compact_df[compact_df["ablation_family"] == "architecture"].copy()
        constraint_view = compact_df[compact_df["ablation_family"] == "constraint"].copy()
        missing_view = compact_df[compact_df["ablation_family"] == "missing_modality"].copy()

        fusion_view.to_csv(outdir / f"fusion_ablation_{args.report_split}.csv", index=False)
        temporal_view.to_csv(outdir / f"temporal_ablation_{args.report_split}.csv", index=False)
        arch_view.to_csv(outdir / f"architecture_ablation_{args.report_split}.csv", index=False)
        constraint_view.to_csv(outdir / f"constraint_ablation_{args.report_split}.csv", index=False)
        missing_view.to_csv(outdir / f"missing_modality_ablation_{args.report_split}.csv", index=False)

    print(f"Saved: {compact_path}")
    print(f"Saved: {detailed_path}")
    print(f"Saved: {outdir / 'ablation_manifest.json'}")
    if compact_df is not None and not compact_df.empty:
        print(f"Saved: {outdir / f'ablation_leaderboard_{args.report_split}.csv'}")


if __name__ == "__main__":
    main()