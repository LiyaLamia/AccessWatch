#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for merging the different feature tables
    into one final model-ready raion-week dataset.
    """
    p = argparse.ArgumentParser(
        description="Merge ACLED, transport, and Black Marble features into one model-ready raion-week table."
    )
    p.add_argument("--acled_week_csv", required=True)
    p.add_argument("--transport_csv", required=True)
    p.add_argument("--black_marble_week_csv", required=False, default=None)
    p.add_argument("--out_csv", required=True)
    return p.parse_args()


def add_temporal_features(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    """
    For selected columns, add simple lag and rolling-window features
    within each raion's weekly time series.
    """
    df = df.sort_values(["raion_id", "week_start"]).copy()

    for col in base_cols:
        if col not in df.columns:
            continue

        grp = df.groupby("raion_id")[col]

        # Previous one- and two-week values
        df[f"{col}_lag1"] = grp.shift(1)
        df[f"{col}_lag2"] = grp.shift(2)

        # Rolling summaries based only on past weeks, not the current week
        df[f"{col}_roll4_mean"] = (
            grp.shift(1).rolling(4, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df[f"{col}_roll4_sum"] = (
            grp.shift(1).rolling(4, min_periods=1).sum().reset_index(level=0, drop=True)
        )

    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add next-week prediction targets derived from the future values
    of the conflict variables.
    """
    df = df.sort_values(["raion_id", "week_start"]).copy()
    grp = df.groupby("raion_id")

    # Direct next-week regression-style targets for a few key outcomes
    for col in ["acled_event_count", "fatalities_sum", "air_drone_strike_count"]:
        if col in df.columns:
            df[f"target_next_{col}"] = grp[col].shift(-1)

    next_events = grp["acled_event_count"].shift(-1)
    next_fatalities = grp["fatalities_sum"].shift(-1)
    next_vac = grp["violence_against_civilians_count"].shift(-1)
    next_expl = grp["explosions_remote_count"].shift(-1)

    # Binary label for whether next week should be considered high risk
    df["target_next_week_high_risk"] = (
        (next_events >= 5) |
        (next_fatalities >= 1) |
        (next_vac >= 1) |
        (next_expl >= 3)
    ).astype("float")

    # Simpler binary target: did any event occur next week?
    df["target_next_week_any_event"] = (next_events >= 1).astype("float")
    return df


def ensure_required_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    """
    Check that an input table has the columns needed for merging.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def main() -> None:
    args = parse_args()

    # Load the main weekly ACLED table and the static transport table
    acled = pd.read_csv(args.acled_week_csv, parse_dates=["week_start"])
    transport = pd.read_csv(args.transport_csv)

    ensure_required_columns(acled, ["raion_id", "raion_name", "oblast_name", "week_start"], "ACLED weekly CSV")
    ensure_required_columns(transport, ["raion_id", "raion_name", "oblast_name"], "Transport CSV")

    # Transport is static by raion, so merge it onto the weekly ACLED rows
    df = acled.merge(
        transport,
        on=["raion_id", "raion_name", "oblast_name"],
        how="left",
        validate="many_to_one",
    )

    if args.black_marble_week_csv:
        bm = pd.read_csv(args.black_marble_week_csv, parse_dates=["week_start"])

        # Black Marble output does not include oblast_name,
        # so merge only on raion ID/name plus week.
        ensure_required_columns(bm, ["raion_id", "raion_name", "week_start"], "Black Marble weekly CSV")

        dupes = bm.duplicated(subset=["raion_id", "raion_name", "week_start"]).sum()
        if dupes:
            raise ValueError(
                f"Black Marble weekly CSV has {dupes} duplicate raion-week rows. "
                "Expected one row per raion_id, raion_name, week_start."
            )

        df = df.merge(
            bm,
            on=["raion_id", "raion_name", "week_start"],
            how="left",
            validate="one_to_one",
        )

    # These are static transport-style features, so missing values
    # usually just mean there was no matching feature for that raion.
    static_cols = [
        "road_total_length_km", "road_major_length_km", "road_paved_length_km",
        "road_bridge_length_km", "rail_total_length_km", "road_density_km_per_100sqkm",
        "area_sqkm", "major_road_density_km_per_100sqkm", "rail_density_km_per_100sqkm",
        "major_road_share", "paved_road_share", "rail_to_road_ratio",
    ]
    for col in static_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Base columns from which lagged and rolling temporal features are created
    temporal_base_cols = [
        "acled_event_count", "fatalities_sum", "events_with_fatalities",
        "violence_against_civilians_count", "explosions_remote_count", "battles_count",
        "strategic_developments_count", "air_drone_strike_count", "precise_geo_event_count",
        "ntl_week_mean", "ntl_week_std_mean", "ntl_change_vs_prev_week",
        "ntl_change_vs_rolling4", "ntl_high_quality_share",
    ]
    df = add_temporal_features(df, temporal_base_cols)
    df = add_targets(df)

    # A few simple interaction terms combining conflict, infrastructure,
    # and nighttime-light signals
    if {"road_density_km_per_100sqkm", "acled_event_count"}.issubset(df.columns):
        df["events_x_road_density"] = df["acled_event_count"] * df["road_density_km_per_100sqkm"]

    if {"rail_density_km_per_100sqkm", "explosions_remote_count"}.issubset(df.columns):
        df["explosions_x_rail_density"] = df["explosions_remote_count"] * df["rail_density_km_per_100sqkm"]

    if {"ntl_change_vs_prev_week", "air_drone_strike_count"}.issubset(df.columns):
        # Negative NTL change means lights dropped, so multiply by the negative
        # to make larger positive values correspond to stronger drops.
        df["drone_x_ntl_drop"] = df["air_drone_strike_count"] * (-df["ntl_change_vs_prev_week"].fillna(0))

    # Time-based train/validation/test split
    df["split"] = np.where(df["week_start"] < pd.Timestamp("2025-01-01"), "train", "valid")
    df.loc[df["week_start"] >= pd.Timestamp("2026-01-01"), "split"] = "test"

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print(f"Saved model table to {args.out_csv}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print("Split counts:")
    print(df["split"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()