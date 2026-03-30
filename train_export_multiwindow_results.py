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

# Canonical next-week targets from the base master table.
# For multi-week forecasting, these are rebuilt for each requested horizon.
BASE_TARGETS = {
    "y_next_high_intensity": {"source_col": "high_intensity_week", "agg": "max", "task_type": "classification"},
    "y_next_any_event": {"source_col": "any_event", "agg": "max", "task_type": "classification"},
    "y_next_event_count": {"source_col": "acled_event_count", "agg": "sum", "task_type": "regression"},
    "y_next_fatalities_sum": {"source_col": "fatalities_sum", "agg": "sum", "task_type": "regression"},
    "y_next_air_drone_strike_count": {"source_col": "air_drone_strike_count", "agg": "sum", "task_type": "regression"},
}

# Friendly names used for output folders and display metadata
DISPLAY_LABELS = {
    1: "next_week",
    2: "next_2_weeks",
    3: "next_3_weeks",
    4: "next_4_weeks_month",
}


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    """
    Parse wrapper-level arguments and also capture any extra arguments
    that should be passed straight through to the downstream training script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create 1/2/3/4-week forecasting targets and run the existing multitarget exporter "
            "for each horizon, saving per-horizon CSV results."
        )
    )
    parser.add_argument("--master_csv", required=True)
    parser.add_argument("--model1_csv", required=True)
    parser.add_argument("--model2_csv", required=True)
    parser.add_argument("--feature_spec_json", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--train_script",
        default=None,
        help="Path to 12_train_export_multitarget_models.py. Defaults to the file next to this wrapper.",
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
        default=list(BASE_TARGETS.keys()),
        choices=list(BASE_TARGETS.keys()),
        help="Canonical targets to run for every forecast window.",
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
    Make sure the master table has the core columns needed and
    normalize ordering before building forecast windows.
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
    For binary-style targets use max; for count-like targets use sum.
    """
    future_parts = [series.shift(-i) for i in range(1, weeks + 1)]
    future = pd.concat(future_parts, axis=1)

    # If the full future window is not available, mark the target as missing
    incomplete = future.isna().any(axis=1)

    if agg == "max":
        out = future.max(axis=1)
    elif agg == "sum":
        out = future.sum(axis=1)
    else:
        raise ValueError(f"Unsupported future agg: {agg}")

    out[incomplete] = np.nan
    return out


# Used only for the naive baseline source columns inside the downstream script.
def build_trailing_window_source(series: pd.Series, weeks: int, agg: str) -> pd.Series:
    """
    Build a trailing window version of the source column so the naive baseline
    stays aligned with the forecast horizon.
    """
    past_parts = [series.shift(i) for i in range(0, weeks)]
    past = pd.concat(past_parts, axis=1)

    if agg == "max":
        return past.max(axis=1, skipna=True)
    if agg == "sum":
        return past.sum(axis=1, skipna=True)

    raise ValueError(f"Unsupported trailing agg: {agg}")


# The downstream script expects the canonical y_next_* names, so for each horizon we overwrite
# those target columns with the windowed targets for that horizon.
def make_master_for_window(master: pd.DataFrame, weeks: int, targets: List[str]) -> pd.DataFrame:
    """
    Create a horizon-specific master table where the canonical next-week targets
    are replaced with next-N-week versions for the requested window size.
    """
    out = master.copy()
    grp = out.groupby("raion_id", sort=False)

    for target_col in targets:
        cfg = BASE_TARGETS[target_col]
        source_col = cfg["source_col"]

        if source_col not in out.columns:
            raise ValueError(f"Required source column missing from master: {source_col}")

        # Overwrite the canonical target with the future N-week target
        out[target_col] = grp[source_col].transform(
            lambda s: build_future_window_target(pd.to_numeric(s, errors="coerce"), weeks, cfg["agg"])
        )

        # Also overwrite the source column so the downstream naive baseline
        # uses a horizon-matched trailing window instead of only one week
        out[source_col] = grp[source_col].transform(
            lambda s: build_trailing_window_source(pd.to_numeric(s, errors="coerce"), weeks, cfg["agg"])
        )

    out["forecast_window_weeks"] = weeks
    out["forecast_window_label"] = DISPLAY_LABELS.get(weeks, f"next_{weeks}_weeks")
    return out


def patch_export_csv(path: Path, weeks: int) -> None:
    """
    Patch one exported CSV so its target window metadata matches
    the current forecast horizon.
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
    the correct multi-week window metadata.
    """
    # Patch all exported prediction CSVs to show the correct target window.
    for csv_path in horizon_dir.rglob("*.csv"):
        # Also patch the summary CSVs with horizon metadata.
        try:
            if csv_path.name == "comparison_metrics_all_targets.csv":
                df = pd.read_csv(csv_path)
                df["forecast_window_weeks"] = weeks
                df["forecast_window_label"] = DISPLAY_LABELS.get(weeks, f"next_{weeks}_weeks")
                df.to_csv(csv_path, index=False)
            else:
                patch_export_csv(csv_path, weeks)
        except Exception:
            # Leave nonstandard CSVs alone if they do not fit the export shape.
            continue


def main() -> None:
    args, passthrough = parse_args()

    master = normalize_master(pd.read_csv(args.master_csv))

    wrapper_path = Path(__file__).resolve()
    train_script = Path(args.train_script) if args.train_script else wrapper_path.with_name("12_train_export_multitarget_models.py")
    if not train_script.exists():
        raise FileNotFoundError(f"Could not find downstream train/export script: {train_script}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # By default, keep temporary horizon-specific master files in a temp dir,
    # unless the user wants to keep them for inspection.
    temp_root_obj = tempfile.TemporaryDirectory(prefix="multiwindow_targets_")
    temp_root = Path(temp_root_obj.name)

    if args.keep_temps:
        temp_root = outdir / "_temp_multiwindow_inputs"
        temp_root.mkdir(parents=True, exist_ok=True)

        # We created our own persistent temp folder, so the automatic temp object
        # is no longer needed.
        temp_root_obj.cleanup()
        temp_root_obj = None

    combined_metrics: List[pd.DataFrame] = []
    combined_summary: Dict[str, object] = {
        "summary": {
            "window_weeks": args.window_weeks,
            "window_labels": {str(w): DISPLAY_LABELS.get(w, f"next_{w}_weeks") for w in args.window_weeks},
            "targets": args.targets,
            "base_train_script": str(train_script),
        },
        "horizons": {},
    }

    for weeks in args.window_weeks:
        horizon_label = DISPLAY_LABELS.get(weeks, f"next_{weeks}_weeks")
        horizon_dir = outdir / horizon_label
        horizon_dir.mkdir(parents=True, exist_ok=True)

        # Build a horizon-specific master file and save it temporarily
        master_h = make_master_for_window(master, weeks, args.targets)
        master_h_path = temp_root / f"master_{weeks}w.csv"
        master_h.to_csv(master_h_path, index=False)

        # Call the downstream multitarget exporter on this horizon-specific master table
        cmd = [
            sys.executable,
            str(train_script),
            "--master_csv", str(master_h_path),
            "--model1_csv", args.model1_csv,
            "--model2_csv", args.model2_csv,
            "--feature_spec_json", args.feature_spec_json,
            "--outdir", str(horizon_dir),
            "--targets",
            *args.targets,
        ]

        # Forward optional args like --skip_gru, --valid_weeks, etc.
        cmd.extend(passthrough)

        print("\n=== Running", horizon_label, f"({weeks}w) ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

        # Patch the output files so their target windows reflect the horizon correctly
        patch_results_dir(horizon_dir, weeks)

        cmp_path = horizon_dir / "comparison_metrics_all_targets.csv"
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
            "json_results": str(horizon_dir / "all_results_multitarget.json"),
        }

    # Save one combined metrics table across all forecast windows
    if combined_metrics:
        all_cmp = pd.concat(combined_metrics, ignore_index=True)
        all_cmp.to_csv(outdir / "comparison_metrics_all_windows.csv", index=False)
        print(f"Saved combined comparison CSV to: {outdir / 'comparison_metrics_all_windows.csv'}")

    with open(outdir / "all_results_multiwindow_summary.json", "w", encoding="utf-8") as f:
        json.dump(combined_summary, f, indent=2)
    print(f"Saved multiwindow summary JSON to: {outdir / 'all_results_multiwindow_summary.json'}")

    # Clean up temp inputs unless the user asked to keep them
    if temp_root_obj is not None:
        temp_root_obj.cleanup()
    elif not args.keep_temps and temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()