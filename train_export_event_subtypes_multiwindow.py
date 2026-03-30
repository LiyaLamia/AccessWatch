#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Event subtype targets and how each one should be built from the base master table.
# These are all next-window binary targets, so they use max over the future window.
BASE_TARGETS = {
    "y_next_battle_any": {"source_col": "battles_count", "agg": "max", "task_type": "classification"},
    "y_next_explosions_remote_any": {"source_col": "explosions_remote_count", "agg": "max", "task_type": "classification"},
    "y_next_violence_against_civilians_any": {"source_col": "violence_against_civilians_count", "agg": "max", "task_type": "classification"},
    "y_next_air_drone_any": {"source_col": "air_drone_strike_count", "agg": "max", "task_type": "classification"},
    "y_next_strategic_developments_any": {"source_col": "strategic_developments_count", "agg": "max", "task_type": "classification"},
    "y_next_protests_riots_any": {"source_col": "protests_riots_count", "agg": "max", "task_type": "classification"},
}

# Friendly names used for output folders and metadata
DISPLAY_LABELS = {
    1: "next_week",
    2: "next_2_weeks",
    3: "next_3_weeks",
    4: "next_4_weeks_month",
}


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    """
    Parse wrapper-level arguments and also capture any extra arguments
    that should be passed straight through to the downstream subtype script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create 1/2/3/4-week event subtype forecasting targets and run the existing subtype exporter "
            "for each horizon, saving per-horizon CSV results and metrics."
        )
    )
    parser.add_argument("--master_csv", required=True)
    parser.add_argument("--model1_csv", required=True)
    parser.add_argument("--model2_csv", required=True)
    parser.add_argument("--feature_spec_json", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--acled_events_csv", default=None)
    parser.add_argument(
        "--train_script",
        default=None,
        help="Path to 14_train_export_event_subtypes.py. Defaults to the file next to this wrapper.",
    )
    parser.add_argument(
        "--window_weeks",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help="Forecast windows in weeks. 4 weeks is the month approximation.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=[
            "y_next_battle_any",
            "y_next_explosions_remote_any",
            "y_next_violence_against_civilians_any",
            "y_next_air_drone_any",
            "y_next_strategic_developments_any",
        ],
        choices=list(BASE_TARGETS.keys()),
        help=(
            "Subtype targets to run for every forecast window. Protests/riots is excluded by default, "
            "but can be added explicitly or via --include_protests_riots."
        ),
    )
    parser.add_argument(
        "--include_protests_riots",
        action="store_true",
        help="Also include y_next_protests_riots_any even though it is excluded by default.",
    )
    parser.add_argument(
        "--keep_temps",
        action="store_true",
        help="Keep temporary per-window master CSVs for inspection.",
    )
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def normalize_master(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure the master table has the key columns needed
    and sort it into raion-week order.
    """
    out = df.copy()
    if "week_start" not in out.columns:
        raise ValueError("master_csv must contain 'week_start'")
    if "raion_id" not in out.columns:
        raise ValueError("master_csv must contain 'raion_id'")

    out["week_start"] = pd.to_datetime(out["week_start"], errors="coerce")
    out = out.sort_values(["raion_id", "week_start"]).reset_index(drop=True)
    return out


def build_future_window_target(series: pd.Series, weeks: int, agg: str) -> pd.Series:
    """
    Build a future-looking target over the next N weeks.
    For subtype forecasting this is usually a max, meaning:
    did this subtype occur at least once in the future window?
    """
    future_parts = [series.shift(-i) for i in range(1, weeks + 1)]
    future = pd.concat(future_parts, axis=1)

    # If the full future window is not available, leave the target missing
    incomplete = future.isna().any(axis=1)

    if agg == "max":
        out = future.max(axis=1)
    elif agg == "sum":
        out = future.sum(axis=1)
    else:
        raise ValueError(f"Unsupported future agg: {agg}")

    out[incomplete] = np.nan
    return out


def build_trailing_window_source(series: pd.Series, weeks: int, agg: str) -> pd.Series:
    """
    Build a trailing-window version of the source signal so the naive baseline
    stays aligned with the same forecast window length.
    """
    past_parts = [series.shift(i) for i in range(0, weeks)]
    past = pd.concat(past_parts, axis=1)

    if agg == "max":
        return past.max(axis=1, skipna=True)
    if agg == "sum":
        return past.sum(axis=1, skipna=True)

    raise ValueError(f"Unsupported trailing agg: {agg}")


def make_master_for_window(master: pd.DataFrame, weeks: int, targets: List[str]) -> pd.DataFrame:
    """
    Create one horizon-specific master table where the canonical subtype targets
    are overwritten with next-N-week versions.
    """
    out = master.copy()
    grp = out.groupby("raion_id", sort=False)

    for target_col in targets:
        cfg = BASE_TARGETS[target_col]
        source_col = cfg["source_col"]

        if source_col not in out.columns:
            raise ValueError(f"Required source column missing from master: {source_col}")

        current_any_col = target_col.replace("y_next_", "current_")

        # Current-week subtype presence flag
        out[current_any_col] = (
            pd.to_numeric(out[source_col], errors="coerce").fillna(0) > 0
        ).astype(int)

        # Future target over the requested horizon
        out[target_col] = grp[current_any_col].transform(
            lambda s: build_future_window_target(pd.to_numeric(s, errors="coerce"), weeks, cfg["agg"])
        )

        # Horizon-aligned source column for the downstream naive baseline
        out[source_col] = grp[current_any_col].transform(
            lambda s: build_trailing_window_source(pd.to_numeric(s, errors="coerce"), weeks, cfg["agg"])
        )

    out["forecast_window_weeks"] = weeks
    out["forecast_window_label"] = DISPLAY_LABELS.get(weeks, f"next_{weeks}_weeks")
    return out


def patch_export_csv(path: Path, weeks: int) -> None:
    """
    Patch one exported CSV so the target window dates and horizon metadata
    match the forecast window that produced it.
    """
    df = pd.read_csv(path)

    if "week_start" in df.columns:
        ws = pd.to_datetime(df["week_start"], errors="coerce")
        df["target_window_start"] = (ws + pd.Timedelta(days=7)).dt.strftime("%Y-%m-%d")
        df["target_window_end"] = (ws + pd.Timedelta(days=7 * weeks + 6)).dt.strftime("%Y-%m-%d")

    df["forecast_window_weeks"] = weeks
    df["forecast_window_label"] = DISPLAY_LABELS.get(weeks, f"next_{weeks}_weeks")
    df.to_csv(path, index=False)


def patch_results_dir(horizon_dir: Path, weeks: int) -> None:
    """
    Patch all CSV outputs inside one horizon directory so they carry
    the correct horizon metadata.
    """
    for csv_path in horizon_dir.rglob("*.csv"):
        try:
            if csv_path.name == "comparison_metrics_event_subtypes.csv":
                df = pd.read_csv(csv_path)
                df["forecast_window_weeks"] = weeks
                df["forecast_window_label"] = DISPLAY_LABELS.get(weeks, f"next_{weeks}_weeks")
                df.to_csv(csv_path, index=False)
            else:
                patch_export_csv(csv_path, weeks)
        except Exception:
            # Leave any nonstandard CSVs alone
            continue


def main() -> None:
    args, passthrough = parse_args()
    master = normalize_master(pd.read_csv(args.master_csv))

    targets = list(args.targets)
    if args.include_protests_riots and "y_next_protests_riots_any" not in targets:
        targets.append("y_next_protests_riots_any")

    wrapper_path = Path(__file__).resolve()
    train_script = Path(args.train_script) if args.train_script else wrapper_path.with_name("14_train_export_event_subtypes.py")
    if not train_script.exists():
        raise FileNotFoundError(f"Could not find downstream subtype train/export script: {train_script}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Temporary horizon-specific master files live here unless the user wants to keep them
    temp_root_obj = tempfile.TemporaryDirectory(prefix="subtype_multiwindow_targets_")
    temp_root = Path(temp_root_obj.name)

    if args.keep_temps:
        temp_root = outdir / "_temp_subtype_multiwindow_inputs"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_root_obj.cleanup()
        temp_root_obj = None

    combined_metrics: List[pd.DataFrame] = []
    combined_summary: Dict[str, object] = {
        "summary": {
            "window_weeks": args.window_weeks,
            "window_labels": {str(w): DISPLAY_LABELS.get(w, f"next_{w}_weeks") for w in args.window_weeks},
            "targets": targets,
            "base_train_script": str(train_script),
            "include_protests_riots": bool(args.include_protests_riots),
        },
        "horizons": {},
    }

    for weeks in args.window_weeks:
        horizon_label = DISPLAY_LABELS.get(weeks, f"next_{weeks}_weeks")
        horizon_dir = outdir / horizon_label
        horizon_dir.mkdir(parents=True, exist_ok=True)

        # Build one horizon-specific master input for the subtype exporter
        master_h = make_master_for_window(master, weeks, targets)
        master_h_path = temp_root / f"master_subtypes_{weeks}w.csv"
        master_h.to_csv(master_h_path, index=False)

        cmd = [
            sys.executable,
            str(train_script),
            "--master_csv", str(master_h_path),
            "--model1_csv", args.model1_csv,
            "--model2_csv", args.model2_csv,
            "--feature_spec_json", args.feature_spec_json,
            "--outdir", str(horizon_dir),
            "--targets",
            *targets,
        ]
        if args.acled_events_csv:
            cmd.extend(["--acled_events_csv", args.acled_events_csv])
        if args.include_protests_riots:
            cmd.append("--include_protests_riots")

        # Forward extra flags like GRU/lightgbm settings to the downstream script
        cmd.extend(passthrough)

        print(f"\n=== Running {horizon_label} ({weeks}w) for subtype forecasting ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

        # Patch exports so their target window columns reflect the current horizon
        patch_results_dir(horizon_dir, weeks)

        cmp_path = horizon_dir / "comparison_metrics_event_subtypes.csv"
        if cmp_path.exists():
            cmp_df = pd.read_csv(cmp_path)
            cmp_df["forecast_window_weeks"] = weeks
            cmp_df["forecast_window_label"] = horizon_label
            cmp_df.to_csv(cmp_path, index=False)
            combined_metrics.append(cmp_df)

        combined_summary["horizons"][horizon_label] = {
            "forecast_window_weeks": weeks,
            "forecast_window_label": horizon_label,
            "results_dir": str(horizon_dir),
            "master_input_csv": str(master_h_path),
            "comparison_metrics_csv": str(cmp_path) if cmp_path.exists() else None,
            "json_results": str(horizon_dir / "all_results_event_subtypes.json"),
        }

    # Save one combined comparison table across all subtype forecast windows
    if combined_metrics:
        all_cmp = pd.concat(combined_metrics, ignore_index=True)
        all_cmp.to_csv(outdir / "comparison_metrics_event_subtypes_all_windows.csv", index=False)
        print(f"Saved combined subtype comparison CSV to: {outdir / 'comparison_metrics_event_subtypes_all_windows.csv'}")

    with open(outdir / "all_results_event_subtypes_multiwindow_summary.json", "w", encoding="utf-8") as f:
        json.dump(combined_summary, f, indent=2)
    print(f"Saved subtype multiwindow summary JSON to: {outdir / 'all_results_event_subtypes_multiwindow_summary.json'}")

    # Clean up temporary horizon inputs unless the user asked to keep them
    if temp_root_obj is not None:
        temp_root_obj.cleanup()
    elif not args.keep_temps and temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()