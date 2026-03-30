#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# Core identifier columns expected in the modeling tables
ID_COLS = ["raion_id", "raion_name", "oblast_name", "week_start"]

# Raw ACLED weekly columns that should not be included in Model 1
# because Model 1 is meant to exclude direct ACLED signal.
ACLED_RAW_COLS = [
    "acled_event_count",
    "fatalities_sum",
    "events_with_fatalities",
    "violence_against_civilians_count",
    "explosions_remote_count",
    "battles_count",
    "strategic_developments_count",
    "protests_riots_count",
    "civilian_targeting_count",
    "air_drone_strike_count",
    "precise_geo_event_count",
    "any_event",
    "high_intensity_week",
]

# These FIRMS proximity fields were flagged as potentially unreliable,
# so the script can drop them automatically if they are obviously broken.
BROKEN_FIRMS_PROXIMITY_COLS = [
    "firms_near_roads_count",
    "firms_near_rail_count",
    "firms_near_roads_share",
    "firms_near_rail_share",
]

MODEL1_EXCLUDE = set(ACLED_RAW_COLS) | {"target_week_start"}
MODEL2_EXCLUDE = set(ACLED_RAW_COLS) | {"target_week_start"}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for building the merged master table
    plus the two downstream model tables.
    """
    p = argparse.ArgumentParser(
        description="Build merged raion-week modeling tables for Model 1 (non-ACLED) and Model 2 (+ lagged ACLED history)."
    )
    p.add_argument("--acled_weekly", required=True, help="Path to acled_raion_week.csv")
    p.add_argument("--firms_weekly", required=True, help="Path to firms_weekly_raion.csv")
    p.add_argument("--black_marble_daily_2023", required=True, help="Path to black_marble_daily_raion_2023.csv")
    p.add_argument("--black_marble_weekly_2024", required=True, help="Path to black_marble_weekly_raion_2024.csv")
    p.add_argument("--unosat_static", required=True, help="Path to raion_unosat_features.csv")
    p.add_argument("--exposure_static", required=True, help="Path to raion_exposure_features.csv")
    p.add_argument("--transport_static", required=True, help="Path to raion_transport_features.csv")
    p.add_argument("--out_master", required=True, help="Output CSV for full merged master table")
    p.add_argument("--out_model1", required=True, help="Output CSV for Model 1 table")
    p.add_argument("--out_model2", required=True, help="Output CSV for Model 2 table")
    p.add_argument("--out_feature_json", required=True, help="Output JSON describing feature groups")
    p.add_argument("--start_week", default="2023-01-02", help="Inclusive Monday week_start filter")
    p.add_argument("--end_week", default="2024-12-30", help="Inclusive Monday week_start filter")
    p.add_argument(
        "--drop_broken_firms_proximity",
        action="store_true",
        default=True,
        help="Drop the current near-road / near-rail FIRMS columns if they are all zero or otherwise broken (default: on)",
    )
    p.add_argument(
        "--keep_last_week_without_target",
        action="store_true",
        help="Keep the last available week per raion even though next-week target is missing",
    )
    return p.parse_args()


def ensure_parent(path_str: str) -> None:
    """
    Make sure the parent directory exists before writing an output file.
    """
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def to_monday_week_start(s: pd.Series) -> pd.Series:
    """
    Convert dates to Monday-start week anchors.
    """
    dt = pd.to_datetime(s)
    return (dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.normalize()


def add_group_lags(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    specs: dict[str, tuple[Iterable[int], Iterable[int]]],
) -> pd.DataFrame:
    """
    Add lag and rolling-window features within each group time series.
    """
    df = df.sort_values([group_col, time_col]).copy()
    g = df.groupby(group_col, sort=False)

    for col, (lags, rolls) in specs.items():
        for lag in lags:
            df[f"{col}_lag{lag}"] = g[col].shift(lag)

        for window in rolls:
            shifted = g[col].shift(1)

            # Rolling sums are based only on past weeks, not the current week
            df[f"{col}_roll{window}_sum"] = shifted.groupby(df[group_col]).transform(
                lambda s, w=window: s.rolling(w, min_periods=1).sum()
            )

            # For NTL features, keep rolling means too
            if col.startswith("ntl_"):
                df[f"{col}_roll{window}_mean"] = shifted.groupby(df[group_col]).transform(
                    lambda s, w=window: s.rolling(w, min_periods=1).mean()
                )

    return df


def aggregate_black_marble_2023(daily_path: str) -> pd.DataFrame:
    """
    Aggregate the 2023 daily Black Marble file into weekly raion-level features.
    """
    daily = pd.read_csv(daily_path)

    required = {
        "date", "raion_id", "raion_name", "ntl_mean", "ntl_std_mean",
        "ntl_valid_pixels", "ntl_high_quality_share", "ntl_latest_hq_share"
    }
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"Missing columns in 2023 Black Marble daily file: {sorted(missing)}")

    daily["date"] = pd.to_datetime(daily["date"])
    daily["week_start"] = to_monday_week_start(daily["date"])
    daily = daily.sort_values(["raion_id", "date"])

    # Keep the latest available "latest_hq" value within each raion-week
    latest_idx = daily.groupby(["raion_id", "week_start"], sort=False)["date"].idxmax()
    latest = daily.loc[latest_idx, ["raion_id", "week_start", "ntl_latest_hq_share"]]

    weekly = (
        daily.groupby(["raion_id", "raion_name", "week_start"], as_index=False)
        .agg(
            ntl_week_mean=("ntl_mean", "mean"),
            ntl_week_median=("ntl_mean", "median"),
            ntl_week_std_mean=("ntl_std_mean", "mean"),
            ntl_high_quality_share=("ntl_high_quality_share", "mean"),
            ntl_obs_days=("date", "nunique"),
            ntl_valid_pixels_sum=("ntl_valid_pixels", "sum"),
        )
    )

    weekly = weekly.merge(latest, on=["raion_id", "week_start"], how="left")
    return weekly


def build_black_marble_weekly(daily_2023_path: str, weekly_2024_path: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Combine 2023 weekly features derived from the daily file with the
    already-weekly 2024 file, then add a few temporal NTL features.
    """
    wk23 = aggregate_black_marble_2023(daily_2023_path)

    wk24 = pd.read_csv(weekly_2024_path)
    wk24["week_start"] = pd.to_datetime(wk24["week_start"])

    needed_24 = [
        "raion_id",
        "raion_name",
        "week_start",
        "ntl_week_mean",
        "ntl_week_median",
        "ntl_week_std_mean",
        "ntl_high_quality_share",
        "ntl_latest_hq_share",
        "ntl_obs_days",
        "ntl_valid_pixels_sum",
    ]
    missing = [c for c in needed_24 if c not in wk24.columns]
    if missing:
        raise ValueError(f"Missing columns in 2024 Black Marble weekly file: {missing}")

    wk24 = wk24[needed_24].copy()

    bm = pd.concat([wk23, wk24], ignore_index=True, sort=False)
    bm = bm.sort_values(["raion_id", "week_start"]).drop_duplicates(["raion_id", "week_start"], keep="last")

    g = bm.groupby("raion_id", sort=False)

    # Add a few simple change-based NTL features
    bm["ntl_week_mean_lag1"] = g["ntl_week_mean"].shift(1)
    bm["ntl_week_mean_roll4_mean"] = g["ntl_week_mean"].shift(1).groupby(bm["raion_id"]).transform(
        lambda s: s.rolling(4, min_periods=1).mean()
    )
    bm["ntl_change_vs_prev_week"] = bm["ntl_week_mean"] - bm["ntl_week_mean_lag1"]
    bm["ntl_pct_change_vs_prev_week"] = np.where(
        bm["ntl_week_mean_lag1"].abs() > 1e-12,
        bm["ntl_change_vs_prev_week"] / bm["ntl_week_mean_lag1"],
        np.nan,
    )
    bm["ntl_change_vs_rolling4"] = bm["ntl_week_mean"] - bm["ntl_week_mean_roll4_mean"]
    bm["ntl_change_vs_prev_week_lag1"] = g["ntl_change_vs_prev_week"].shift(1)

    feature_cols = [c for c in bm.columns if c not in ["raion_id", "raion_name", "week_start"]]
    return bm, feature_cols


def build_firms_weekly(firms_path: str, drop_broken_proximity: bool) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Load the weekly FIRMS table and optionally drop proximity columns
    that appear to be completely broken.
    """
    firms = pd.read_csv(firms_path)
    firms["week_start"] = pd.to_datetime(firms["week_start"])

    dropped: list[str] = []
    if drop_broken_proximity:
        for col in BROKEN_FIRMS_PROXIMITY_COLS:
            if col in firms.columns:
                s = pd.to_numeric(firms[col], errors="coerce")
                all_zeroish = s.fillna(0).eq(0).all()
                if all_zeroish:
                    dropped.append(col)

        if dropped:
            firms = firms.drop(columns=dropped)

    keep_id = [c for c in ["raion_id", "raion_name", "oblast_name", "week_start"] if c in firms.columns]
    feature_cols = [c for c in firms.columns if c not in keep_id]
    return firms, feature_cols, dropped


def build_static_table(path: str, drop_cols: Iterable[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Load a static raion-level table and keep only one row per raion.
    """
    df = pd.read_csv(path)
    drop_cols = set(drop_cols or [])

    id_like = [c for c in ["raion_id", "raion_name", "oblast_name"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in id_like and c not in drop_cols]

    keep = ["raion_id"] + feature_cols
    return df[keep].drop_duplicates("raion_id"), feature_cols


def add_targets_and_lagged_acled(acled: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add next-week targets and lagged ACLED history features.
    """
    acled = acled.sort_values(["raion_id", "week_start"]).copy()
    g = acled.groupby("raion_id", sort=False)

    # Next-week targets
    acled["target_week_start"] = g["week_start"].shift(-1)
    acled["y_next_high_intensity"] = g["high_intensity_week"].shift(-1)
    acled["y_next_any_event"] = g["any_event"].shift(-1)
    acled["y_next_event_count"] = g["acled_event_count"].shift(-1)
    acled["y_next_fatalities_sum"] = g["fatalities_sum"].shift(-1)
    acled["y_next_air_drone_strike_count"] = g["air_drone_strike_count"].shift(-1)

    lag_sources = {
        "acled_event_count": ([1], [4]),
        "fatalities_sum": ([1], [4]),
        "air_drone_strike_count": ([1], [4]),
        "any_event": ([1], [4]),
        "high_intensity_week": ([1], [4]),
    }
    acled = add_group_lags(acled, "raion_id", "week_start", lag_sources)

    lagged_acled_cols = [
        c
        for c in acled.columns
        if c.endswith("_lag1") or c.endswith("_roll4_sum")
    ]
    lagged_acled_cols = [
        c for c in lagged_acled_cols
        if c.startswith(("acled_", "fatalities_", "air_drone_", "any_event", "high_intensity_week"))
    ]

    return acled, sorted(lagged_acled_cols)


def choose_name_source(master: pd.DataFrame, preferred: str, fallback: str) -> pd.Series:
    """
    Choose a preferred name column, but fall back to an alternate
    if merges introduced duplicate name fields.
    """
    if preferred in master.columns and fallback in master.columns:
        return master[preferred].combine_first(master[fallback])
    if preferred in master.columns:
        return master[preferred]
    return master[fallback]


def build_master_table(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    """
    Build the full merged raion-week master table from ACLED, FIRMS,
    Black Marble, and several static feature tables.
    """
    acled = pd.read_csv(args.acled_weekly)
    acled["week_start"] = pd.to_datetime(acled["week_start"])

    # Restrict everything to the modeling window
    start_week = pd.Timestamp(args.start_week)
    end_week = pd.Timestamp(args.end_week)
    acled = acled[(acled["week_start"] >= start_week) & (acled["week_start"] <= end_week)].copy()

    acled, lagged_acled_cols = add_targets_and_lagged_acled(acled)

    firms, firms_feature_cols, dropped_firms_cols = build_firms_weekly(
        args.firms_weekly,
        drop_broken_proximity=args.drop_broken_firms_proximity,
    )
    firms = firms[(firms["week_start"] >= start_week) & (firms["week_start"] <= end_week)].copy()

    bm, bm_feature_cols = build_black_marble_weekly(
        args.black_marble_daily_2023,
        args.black_marble_weekly_2024
    )
    bm = bm[(bm["week_start"] >= start_week) & (bm["week_start"] <= end_week)].copy()

    unosat, unosat_feature_cols = build_static_table(args.unosat_static)
    exposure, exposure_feature_cols = build_static_table(args.exposure_static, drop_cols=["area_sqkm"])
    transport, transport_feature_cols = build_static_table(args.transport_static, drop_cols=["area_sqkm"])

    master = acled.copy()

    # Merge all weekly and static feature sources together
    master = master.merge(
        firms.drop(columns=[c for c in ["raion_name", "oblast_name"] if c in firms.columns]),
        on=["raion_id", "week_start"],
        how="left",
        suffixes=("", "_firms"),
    )
    master = master.merge(
        bm.drop(columns=[c for c in ["raion_name"] if c in bm.columns]),
        on=["raion_id", "week_start"],
        how="left",
        suffixes=("", "_bm"),
    )
    master = master.merge(unosat, on="raion_id", how="left", suffixes=("", "_unosat"))
    master = master.merge(exposure, on="raion_id", how="left", suffixes=("", "_exposure"))
    master = master.merge(transport, on="raion_id", how="left", suffixes=("", "_transport"))

    # Restore canonical name columns if any merge created alternate versions
    if "raion_name_firms" in master.columns:
        master["raion_name"] = choose_name_source(master, "raion_name", "raion_name_firms")
        master = master.drop(columns=["raion_name_firms"])

    if "oblast_name_firms" in master.columns:
        master["oblast_name"] = choose_name_source(master, "oblast_name", "oblast_name_firms")
        master = master.drop(columns=["oblast_name_firms"])

    # Fill event-style FIRMS columns with 0 when a week had no matched FIRMS row
    firms_count_like = [
        c for c in firms_feature_cols
        if c.endswith("_count") or c.endswith("_sum") or c.endswith("_share") or c.endswith("_lag1") or c.endswith("_roll4_sum")
    ]
    for col in firms_count_like:
        if col in master.columns:
            master[col] = master[col].fillna(0)

    # Static features should exist for all raions, but fill missing numeric values just in case
    static_feature_cols = unosat_feature_cols + exposure_feature_cols + transport_feature_cols
    for col in static_feature_cols:
        if col in master.columns and pd.api.types.is_numeric_dtype(master[col]):
            master[col] = master[col].fillna(0)

    # By default, drop the last week per raion if it has no next-week target
    if not args.keep_last_week_without_target:
        master = master[master["y_next_high_intensity"].notna()].copy()

    # Format dates consistently for export
    master["week_start"] = pd.to_datetime(master["week_start"]).dt.strftime("%Y-%m-%d")
    if "target_week_start" in master.columns:
        master["target_week_start"] = pd.to_datetime(master["target_week_start"]).dt.strftime("%Y-%m-%d")

    for ycol in [
        "y_next_high_intensity",
        "y_next_any_event",
        "y_next_event_count",
        "y_next_fatalities_sum",
        "y_next_air_drone_strike_count",
    ]:
        if ycol in master.columns:
            # Keep binary targets as nullable integers when possible
            if ycol in {"y_next_high_intensity", "y_next_any_event"}:
                master[ycol] = master[ycol].astype("Int64")

    metadata = {
        "window": {"start_week": args.start_week, "end_week": args.end_week},
        "dropped_firms_cols": dropped_firms_cols,
        "firms_feature_cols": firms_feature_cols,
        "black_marble_feature_cols": bm_feature_cols,
        "unosat_feature_cols": unosat_feature_cols,
        "exposure_feature_cols": exposure_feature_cols,
        "transport_feature_cols": transport_feature_cols,
        "lagged_acled_feature_cols": lagged_acled_cols,
    }
    return master, metadata


def build_model_tables(master: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Split the master table into the two model-specific tables:
    Model 1 without direct ACLED history, and Model 2 with lagged ACLED features added.
    """
    base_cols = [c for c in ID_COLS if c in master.columns]

    target_cols = [
        "target_week_start",
        "y_next_high_intensity",
        "y_next_any_event",
        "y_next_event_count",
        "y_next_fatalities_sum",
        "y_next_air_drone_strike_count",
    ]
    target_cols = [c for c in target_cols if c in master.columns]

    acled_lagged = set(metadata["lagged_acled_feature_cols"])
    firms_features = set(metadata["firms_feature_cols"])
    bm_features = set(metadata["black_marble_feature_cols"])
    unosat_features = set(metadata["unosat_feature_cols"])
    exposure_features = set(metadata["exposure_feature_cols"])
    transport_features = set(metadata["transport_feature_cols"])

    model1_feature_cols = sorted(
        (firms_features | bm_features | unosat_features | exposure_features | transport_features)
        & set(master.columns)
    )
    model2_feature_cols = sorted((set(model1_feature_cols) | acled_lagged) & set(master.columns))

    model1 = master[base_cols + target_cols + model1_feature_cols].copy()
    model2 = master[base_cols + target_cols + model2_feature_cols].copy()

    feature_spec = {
        "id_cols": base_cols,
        "target_cols": target_cols,
        "model1_non_acled_only_features": model1_feature_cols,
        "model2_plus_lagged_acled_features": model2_feature_cols,
        "feature_groups": {
            "firms": sorted(firms_features & set(master.columns)),
            "black_marble": sorted(bm_features & set(master.columns)),
            "unosat": sorted(unosat_features & set(master.columns)),
            "exposure": sorted(exposure_features & set(master.columns)),
            "transport": sorted(transport_features & set(master.columns)),
            "lagged_acled": sorted(acled_lagged & set(master.columns)),
        },
    }

    return model1, model2, feature_spec


def main() -> None:
    args = parse_args()

    for path in [args.out_master, args.out_model1, args.out_model2, args.out_feature_json]:
        ensure_parent(path)

    master, metadata = build_master_table(args)
    model1, model2, feature_spec = build_model_tables(master, metadata)

    master.to_csv(args.out_master, index=False)
    model1.to_csv(args.out_model1, index=False)
    model2.to_csv(args.out_model2, index=False)

    payload = {
        "build_metadata": metadata,
        "feature_spec": feature_spec,
        "row_counts": {
            "master": int(len(master)),
            "model1": int(len(model1)),
            "model2": int(len(model2)),
        },
        "unique_raions": int(master["raion_id"].nunique()),
        "week_start_min": str(master["week_start"].min()) if len(master) else None,
        "week_start_max": str(master["week_start"].max()) if len(master) else None,
    }

    with open(args.out_feature_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved master table: {args.out_master}")
    print(f"Saved Model 1 table: {args.out_model1}")
    print(f"Saved Model 2 table: {args.out_model2}")
    print(f"Saved feature spec: {args.out_feature_json}")
    print(f"Rows in master/model1/model2: {len(master):,} / {len(model1):,} / {len(model2):,}")
    print(f"Unique raions: {master['raion_id'].nunique():,}")

    if metadata["dropped_firms_cols"]:
        print("Dropped broken FIRMS proximity columns:", ", ".join(metadata["dropped_firms_cols"]))
    else:
        print("No FIRMS proximity columns were dropped automatically.")


if __name__ == "__main__":
    main()