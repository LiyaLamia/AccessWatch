#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# Core ID columns shared across the modeling tables
ID_COLS = ["raion_id", "raion_name", "oblast_name", "week_start"]

# Current-week context columns that are useful to carry into subtype exports
BASE_CONTEXT_COLS = [
    "high_intensity_week",
    "any_event",
    "acled_event_count",
    "fatalities_sum",
    "air_drone_strike_count",
    "battles_count",
    "explosions_remote_count",
    "violence_against_civilians_count",
    "strategic_developments_count",
    "protests_riots_count",
]

# Mapping from subtype target name to the current-week source count column
# used to build the next-week binary label.
SUBTYPE_TARGET_MAP = OrderedDict(
    [
        ("y_next_battle_any", "battles_count"),
        ("y_next_explosions_remote_any", "explosions_remote_count"),
        ("y_next_violence_against_civilians_any", "violence_against_civilians_count"),
        ("y_next_air_drone_any", "air_drone_strike_count"),
        ("y_next_strategic_developments_any", "strategic_developments_count"),
    ]
)

# Protests/riots are optional because they are often kept separate from the
# more directly violent event types.
PROTESTS_TARGET = ("y_next_protests_riots_any", "protests_riots_count")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training/exporting
    next-week event subtype classifiers.
    """
    p = argparse.ArgumentParser(
        description=(
            "Train/export event subtype classifiers for next-week raion forecasting. "
            "By default excludes protests/riots so battle targets remain purely battle-related."
        )
    )
    p.add_argument("--master_csv", required=True)
    p.add_argument("--model1_csv", required=True)
    p.add_argument("--model2_csv", required=True)
    p.add_argument("--feature_spec_json", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--acled_events_csv", default=None)
    p.add_argument("--valid_weeks", type=int, default=13)
    p.add_argument("--test_weeks", type=int, default=13)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--top_frac", type=float, default=0.20)
    p.add_argument("--skip_gru", action="store_true")
    p.add_argument("--skip_lightgbm", action="store_true")
    p.add_argument("--skip_catboost", action="store_true")
    p.add_argument("--include_protests_riots", action="store_true", help="Also train a subtype target for protests/riots.")
    p.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help=(
            "Optional subset of subtype targets to run. Choices: "
            "y_next_battle_any y_next_explosions_remote_any y_next_violence_against_civilians_any "
            "y_next_air_drone_any y_next_strategic_developments_any y_next_protests_riots_any"
        ),
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--gru_seq_len", type=int, default=8)
    p.add_argument("--gru_epochs", type=int, default=12)
    p.add_argument("--gru_batch_size", type=int, default=128)
    p.add_argument("--gru_hidden_dim", type=int, default=96)
    p.add_argument("--gru_num_layers", type=int, default=2)
    p.add_argument("--gru_dropout", type=float, default=0.2)
    p.add_argument("--gru_lr", type=float, default=1e-3)
    p.add_argument("--gru_weight_decay", type=float, default=1e-4)
    p.add_argument(
        "--drilldown_splits",
        nargs="+",
        default=["test"],
        choices=["train", "valid", "test"],
    )
    return p.parse_args()


def load_mtm_module() -> Any:
    """
    Dynamically load the main multitarget training/export script so this wrapper
    can reuse its helper functions instead of duplicating them.
    """
    here = Path(__file__).resolve().parent
    script_path = here / "12_train_export_multitarget_models.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Required helper script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("mtm", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helper module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_target_map(include_protests: bool) -> OrderedDict[str, str]:
    """
    Build the subtype target mapping, optionally including protests/riots.
    """
    mapping = OrderedDict(SUBTYPE_TARGET_MAP)
    if include_protests:
        mapping[PROTESTS_TARGET[0]] = PROTESTS_TARGET[1]
    return mapping


def add_subtype_targets(df: pd.DataFrame, target_map: Dict[str, str]) -> pd.DataFrame:
    """
    Add current-week binary subtype indicators plus the shifted next-week
    subtype targets.
    """
    out = df.copy()
    out["week_start"] = pd.to_datetime(out["week_start"], errors="coerce")
    out = out.sort_values(["raion_id", "week_start"]).copy()
    g = out.groupby("raion_id", sort=False)

    for target_col, source_count_col in target_map.items():
        current_any_col = target_col.replace("y_next_", "current_")

        # Current-week binary flag: did this subtype happen at all this week?
        out[current_any_col] = (
            pd.to_numeric(out[source_count_col], errors="coerce").fillna(0) > 0
        ).astype(int)

        # Next-week label is just the current-week flag shifted forward by one row
        out[target_col] = g[current_any_col].shift(-1)

    return out


def attach_master_context(master: pd.DataFrame, model_df: pd.DataFrame, target_cols: List[str], extra_context_cols: List[str]) -> pd.DataFrame:
    """
    Merge the master-table split labels, subtype targets, and extra context
    into one of the model-specific feature tables.
    """
    keys = [c for c in ID_COLS if c in master.columns and c in model_df.columns]
    ctx_only = [c for c in (extra_context_cols + target_cols + ["split"]) if c in master.columns]

    out = model_df.copy()

    # Drop any duplicate context columns so the master version is the one we keep
    drop_if_present = [c for c in ctx_only if c in out.columns]
    if drop_if_present:
        out = out.drop(columns=drop_if_present)

    merged = out.merge(master[keys + ctx_only], on=keys, how="left", validate="many_to_one")
    return merged


def build_naive_predictions(df: pd.DataFrame, target_col: str, source_count_col: str, mtm: Any, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build a naive subtype baseline by predicting next week from whether
    the same subtype is already present this week.
    """
    work = df.copy()
    mask = pd.to_numeric(work[target_col], errors="coerce").notna()
    work = work.loc[mask].reset_index(drop=True)

    base = mtm.make_base_prediction_frame(work, target_col)

    # Naive score is binary: 1 if subtype happened this week, else 0
    score = (pd.to_numeric(work[source_count_col], errors="coerce").fillna(0) > 0).astype(float).to_numpy()

    pred_df = base.copy()
    pred_df["prediction_score"] = score
    pred_df["prediction_threshold"] = 0.5
    pred_df["prediction_label"] = (pred_df["prediction_score"] >= 0.5).astype("Int64")
    pred_df["prediction_rank_within_week"] = pred_df.groupby("week_start")["prediction_score"].rank(
        method="first", ascending=False
    ).astype(int)

    metrics = {"source_col": source_count_col}
    for split in ["train", "valid", "test"]:
        sub = pred_df[pred_df["split"] == split]
        y_true = sub["actual_target"].to_numpy(dtype=float)
        y_score = sub["prediction_score"].to_numpy(dtype=float)
        metrics[split] = mtm.classification_metrics(sub, y_true, y_score, 0.5, args.top_k, args.top_frac)

    return pred_df, metrics


def subtype_event_filter(acled_events: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Keep only the ACLED event rows relevant to the requested subtype target.
    This is used for the event-level drilldown export.
    """
    ev = acled_events.copy()
    event_type = ev.get("event_type", pd.Series("", index=ev.index)).astype(str)
    sub_event = ev.get("sub_event_type", pd.Series("", index=ev.index)).astype(str)

    if target_col == "y_next_battle_any":
        return ev[event_type == "Battles"].copy()
    if target_col == "y_next_explosions_remote_any":
        return ev[event_type == "Explosions/Remote violence"].copy()
    if target_col == "y_next_violence_against_civilians_any":
        return ev[event_type == "Violence against civilians"].copy()
    if target_col == "y_next_air_drone_any":
        return ev[sub_event.isin(["Air/drone strike", "Shelling/artillery/missile attack"])].copy()
    if target_col == "y_next_strategic_developments_any":
        return ev[event_type == "Strategic developments"].copy()
    if target_col == "y_next_protests_riots_any":
        return ev[event_type.isin(["Protests", "Riots"])].copy()

    return ev.iloc[0:0].copy()


def save_subtype_event_drilldown(pred_df: pd.DataFrame, acled_events: pd.DataFrame, target_col: str, out_csv: Path, splits: List[str]) -> None:
    """
    Save an event-level drilldown file linking prediction rows
    to actual subtype-specific ACLED events inside the target window.
    """
    sub = pred_df[pred_df["split"].isin(splits)].copy()
    if sub.empty:
        return

    ev = subtype_event_filter(acled_events, target_col)
    if ev.empty:
        return

    ev["event_date"] = pd.to_datetime(ev["event_date"], errors="coerce")

    keep_pred_cols = [
        "raion_id", "raion_name", "oblast_name", "week_start", "target_window_start", "target_window_end", "split",
        "actual_target", "prediction_score", "prediction_label", "prediction_rank_within_week",
    ]
    keep_pred_cols = [c for c in keep_pred_cols if c in sub.columns]

    merged = sub[keep_pred_cols].merge(
        ev,
        on=[c for c in ["raion_id", "raion_name", "oblast_name"] if c in ev.columns and c in sub.columns],
        how="left",
    )

    # Keep only events that actually fall inside the prediction target window
    mask = (
        merged["event_date"].notna()
        & (merged["event_date"] >= pd.to_datetime(merged["target_window_start"]))
        & (merged["event_date"] <= pd.to_datetime(merged["target_window_end"]))
    )
    merged = merged[mask].copy()
    if merged.empty:
        return

    for c in ["week_start", "target_window_start", "target_window_end", "event_date"]:
        merged[c] = pd.to_datetime(merged[c], errors="coerce").dt.strftime("%Y-%m-%d")

    merged.to_csv(out_csv, index=False)


def main() -> None:
    args = parse_args()
    mtm = load_mtm_module()
    mtm.set_seed(args.seed)

    target_map = get_target_map(args.include_protests_riots)

    # If the user specified a subset of subtype targets, filter the mapping here
    if args.targets:
        allowed = set(target_map.keys())
        requested = [t for t in args.targets if t in allowed]
        missing = sorted(set(args.targets) - allowed)
        if missing:
            raise ValueError(f"Unknown subtype targets: {missing}")
        target_map = OrderedDict((t, target_map[t]) for t in requested)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(args.master_csv)
    model1 = pd.read_csv(args.model1_csv)
    model2 = pd.read_csv(args.model2_csv)
    feature_spec = mtm.load_feature_spec(args.feature_spec_json)

    master = mtm.normalize_id_columns(master)
    model1 = mtm.normalize_id_columns(model1)
    model2 = mtm.normalize_id_columns(model2)

    # Add subtype-specific current/next-week labels, then assign train/valid/test splits
    master = add_subtype_targets(master, target_map)
    master, split_meta = mtm.assign_time_splits(master, args.valid_weeks, args.test_weeks)

    current_binary_cols = [t.replace("y_next_", "current_") for t in target_map.keys()]
    extra_context_cols = BASE_CONTEXT_COLS + list(target_map.values()) + current_binary_cols

    # Extend the helper module's context lists so exported CSVs also include subtype fields
    mtm.CURRENT_CONTEXT_COLS = list(
        dict.fromkeys(list(getattr(mtm, "CURRENT_CONTEXT_COLS", [])) + extra_context_cols)
    )
    mtm.ALL_TARGET_COLS = list(
        dict.fromkeys(list(getattr(mtm, "ALL_TARGET_COLS", [])) + list(target_map.keys()))
    )
    mtm.CLASSIFICATION_TARGETS = set(getattr(mtm, "CLASSIFICATION_TARGETS", set())) | set(target_map.keys())

    model1 = attach_master_context(master, model1, list(target_map.keys()), extra_context_cols)
    model2 = attach_master_context(master, model2, list(target_map.keys()), extra_context_cols)

    model1_features, model2_features = mtm.get_feature_sets(feature_spec, model1, model2)

    acled_events = None
    if args.acled_events_csv:
        acled_events = pd.read_csv(args.acled_events_csv)
        if "event_date" in acled_events.columns:
            acled_events["event_date"] = pd.to_datetime(acled_events["event_date"], errors="coerce")

    comparison_rows: List[Dict[str, Any]] = []
    all_results: Dict[str, Any] = {
        "summary": {
            "targets": list(target_map.keys()),
            "split_meta": split_meta,
            "lightgbm_available": bool(getattr(mtm, "HAS_LIGHTGBM", False) and not args.skip_lightgbm),
            "catboost_available": bool(getattr(mtm, "HAS_CATBOOST", False) and not args.skip_catboost),
            "torch_available": bool(getattr(mtm, "HAS_TORCH", False) and not args.skip_gru),
            "feature_counts": {"model1": len(model1_features), "model2": len(model2_features)},
            "include_protests_riots": bool(args.include_protests_riots),
        }
    }

    for target_col, source_count_col in target_map.items():
        target_dir = outdir / target_col
        target_dir.mkdir(parents=True, exist_ok=True)

        target_results: Dict[str, Any] = {
            "task_type": "classification",
            "target_col": target_col,
            "split_meta": split_meta,
            "lightgbm_available": bool(getattr(mtm, "HAS_LIGHTGBM", False) and not args.skip_lightgbm),
            "catboost_available": bool(getattr(mtm, "HAS_CATBOOST", False) and not args.skip_catboost),
            "torch_available": bool(getattr(mtm, "HAS_TORCH", False) and not args.skip_gru),
        }

        # Naive subtype baseline
        naive_df, naive_metrics = build_naive_predictions(master, target_col, source_count_col, mtm, args)
        mtm.save_export_csv(naive_df, target_dir / "naive_baseline.csv")
        if acled_events is not None:
            save_subtype_event_drilldown(
                naive_df,
                acled_events,
                target_col,
                target_dir / "naive_baseline__event_drilldown.csv",
                args.drilldown_splits
            )
        target_results["naive_baseline"] = naive_metrics
        comparison_rows.append({
            "target": target_col,
            "experiment": "naive_baseline",
            "algorithm": "naive",
            "split": "test",
            **naive_metrics["test"]
        })

        experiments = [
            ("model1_non_acled_only", model1, model1_features),
            ("model2_plus_lagged_acled", model2, model2_features),
        ]

        for exp_name, exp_df, feature_cols in experiments:
            exp_results: Dict[str, Any] = {"n_features": len(feature_cols), "feature_cols": feature_cols}

            for algo_name in ["linear", "lightgbm", "catboost"]:
                if algo_name == "lightgbm" and (not getattr(mtm, "HAS_LIGHTGBM", False) or args.skip_lightgbm):
                    continue
                if algo_name == "catboost" and (not getattr(mtm, "HAS_CATBOOST", False) or args.skip_catboost):
                    continue

                pred_df, metrics = mtm.run_standard_experiment(
                    exp_df, feature_cols, target_col, "classification", algo_name, args
                )
                mtm.save_export_csv(pred_df, target_dir / f"{exp_name}__{algo_name}.csv")

                if acled_events is not None:
                    save_subtype_event_drilldown(
                        pred_df,
                        acled_events,
                        target_col,
                        target_dir / f"{exp_name}__{algo_name}__event_drilldown.csv",
                        args.drilldown_splits
                    )

                exp_results[algo_name] = metrics
                comparison_rows.append({
                    "target": target_col,
                    "experiment": exp_name,
                    "algorithm": algo_name,
                    "split": "test",
                    **metrics["test"]
                })

            if getattr(mtm, "HAS_TORCH", False) and not args.skip_gru:
                pred_df, metrics = mtm.run_gru_experiment(
                    exp_df, feature_cols, target_col, "classification", args, args.seed
                )
                mtm.save_export_csv(pred_df, target_dir / f"{exp_name}__gru.csv")

                if acled_events is not None:
                    save_subtype_event_drilldown(
                        pred_df,
                        acled_events,
                        target_col,
                        target_dir / f"{exp_name}__gru__event_drilldown.csv",
                        args.drilldown_splits
                    )

                exp_results["gru"] = metrics
                comparison_rows.append({
                    "target": target_col,
                    "experiment": exp_name,
                    "algorithm": "gru",
                    "split": "test",
                    **metrics["test"]
                })

            target_results[exp_name] = exp_results

        all_results[target_col] = target_results

    comparison_df = pd.DataFrame(comparison_rows)
    if not comparison_df.empty:
        comparison_df.to_csv(outdir / "comparison_metrics_event_subtypes.csv", index=False)

    with open(outdir / "all_results_event_subtypes.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved subtype comparison metrics: {outdir / 'comparison_metrics_event_subtypes.csv'}")
    print(f"Saved subtype results JSON: {outdir / 'all_results_event_subtypes.json'}")
    print(f"Subtype targets: {', '.join(target_map.keys())}")
    if not args.include_protests_riots:
        print("Protests/riots target excluded by default.")


if __name__ == "__main__":
    main()