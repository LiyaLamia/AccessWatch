#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for building weekly
    raion-level FIRMS fire features.
    """
    p = argparse.ArgumentParser(
        description="Build weekly raion-level FIRMS VIIRS active fire features for Ukraine."
    )
    p.add_argument("--boundary_zip", required=True, help="Path to ukr_admin_boundaries.geojson.zip or extracted admin2 geojson")
    p.add_argument("--firms_csvs", nargs="+", required=True, help="One or more FIRMS CSV files")
    p.add_argument("--out_csv", required=True, help="Output weekly raion features CSV")
    p.add_argument("--out_events_csv", default=None, help="Optional output CSV for point events with assigned raions")
    p.add_argument("--roads_file", default=None, help="Optional roads GPKG/GeoJSON/SHP or zipped download")
    p.add_argument("--rail_file", default=None, help="Optional rail GPKG/GeoJSON/SHP or zipped download")
    p.add_argument("--admin_id_col", default="adm2_pcode")
    p.add_argument("--admin_name_col", default="adm2_name")
    p.add_argument("--road_buffer_km", type=float, default=2.0)
    p.add_argument("--rail_buffer_km", type=float, default=2.0)
    return p.parse_args()


def resolve_boundary_path(boundary_path: str) -> str:
    """
    Resolve the boundary input into a path geopandas can read.
    If a zip file is provided, point to the admin2 GeoJSON inside it.
    """
    p = Path(boundary_path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as zf:
            # Prefer the expected admin2 file name, but fall back if needed
            candidates = [n for n in zf.namelist() if n.lower().endswith("ukr_admin2.geojson")]
            if not candidates:
                candidates = [n for n in zf.namelist() if n.lower().endswith(".geojson") and "admin2" in n.lower()]
            if not candidates:
                raise FileNotFoundError("Could not find ukr_admin2.geojson inside boundary zip")
            inner = candidates[0]
        return f"zip://{p}!{inner}"
    return boundary_path


def resolve_vector_path(path_str: str | None) -> str | None:
    """
    Resolve an optional network file path. This supports plain vector files
    as well as zipped downloads containing a GPKG, GeoJSON, JSON, or SHP.
    """
    if not path_str:
        return None

    p = Path(path_str)
    if p.suffix.lower() != ".zip":
        return path_str

    with zipfile.ZipFile(p) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]

        # Prefer GPKG if present since it is usually the cleanest option
        gpkg = [n for n in names if n.lower().endswith(".gpkg")]
        if gpkg:
            return f"GPKG:/vsizip/{p}/{gpkg[0]}"

        geojson = [n for n in names if n.lower().endswith((".geojson", ".json"))]
        if geojson:
            return f"zip://{p}!{geojson[0]}"

        shp = [n for n in names if n.lower().endswith(".shp")]
        if shp:
            return f"zip://{p}!{shp[0]}"

    raise FileNotFoundError(f"Could not find a .gpkg, .geojson/.json, or .shp inside {p}")


def read_boundaries(boundary_zip: str, admin_id_col: str, admin_name_col: str) -> gpd.GeoDataFrame:
    """
    Read the raion boundary file and standardize the key columns
    used later in the pipeline.
    """
    path = resolve_boundary_path(boundary_zip)
    gdf = gpd.read_file(path)

    keep_cols = [admin_id_col, admin_name_col, "adm1_name", "geometry"]
    missing = [c for c in [admin_id_col, admin_name_col] if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing boundary columns: {missing}. Available: {list(gdf.columns)}")

    present = [c for c in keep_cols if c in gdf.columns]
    raions = gdf[present].copy().rename(
        columns={
            admin_id_col: "raion_id",
            admin_name_col: "raion_name",
            "adm1_name": "oblast_name"
        }
    )

    # Keep the oblast column even if it is missing in the source
    if "oblast_name" not in raions.columns:
        raions["oblast_name"] = None

    # Use EPSG:4326 before spatial joins with FIRMS points
    if raions.crs is None:
        raions = raions.set_crs(4326)
    else:
        raions = raions.to_crs(4326)

    return raions


def load_firms_csvs(csv_paths: list[str]) -> pd.DataFrame:
    """
    Load one or more FIRMS CSV files, concatenate them,
    and create the derived columns used for weekly aggregation.
    """
    frames = []
    for fp in csv_paths:
        df = pd.read_csv(fp)
        df["source_file"] = Path(fp).name
        frames.append(df)

    if not frames:
        raise ValueError("No FIRMS CSVs loaded.")

    df = pd.concat(frames, ignore_index=True)

    required = ["latitude", "longitude", "acq_date", "frp", "confidence", "daynight", "type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing FIRMS columns: {missing}. Available: {list(df.columns)}")

    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["week_start"] = df["acq_date"] - pd.to_timedelta(df["acq_date"].dt.weekday, unit="D")
    df["confidence"] = df["confidence"].astype(str).str.strip().str.lower()
    df["daynight"] = df["daynight"].astype(str).str.strip().str.upper()

    # These confidence rules are written to work across different FIRMS export styles
    df["is_high_confidence"] = df["confidence"].isin(["h", "high", "9"])
    df["is_nominal_confidence"] = df["confidence"].isin(["n", "nominal", "7", "8"])
    df["is_low_confidence"] = df["confidence"].isin(["l", "low", "0", "1", "2", "3", "4", "5", "6"])

    # FIRMS type 0 is typically vegetation fire, while type 2 is often a static land source
    df["is_type0"] = (pd.to_numeric(df["type"], errors="coerce") == 0)
    df["is_type2"] = (pd.to_numeric(df["type"], errors="coerce") == 2)

    # Convert numeric fields if they exist in the export
    for c in ["frp", "bright_ti4", "bright_ti5", "scan", "track"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def assign_raions(df: pd.DataFrame, raions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert FIRMS rows into point geometry and spatially assign
    each event to a raion polygon.
    """
    points = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        points,
        raions[["raion_id", "raion_name", "oblast_name", "geometry"]],
        how="left",
        predicate="within"
    )
    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])
    return joined


def mark_near_network(points: gpd.GeoDataFrame, network_file: str | None, buffer_km: float, out_col: str) -> gpd.GeoDataFrame:
    """
    Mark each FIRMS point as near or not near a transport network
    using a simple buffer distance.
    """
    points = points.copy()
    points[out_col] = False

    if not network_file:
        return points

    resolved = resolve_vector_path(network_file)
    net = gpd.read_file(resolved)
    if net.empty:
        return points

    if net.crs is None:
        net = net.set_crs(4326)
    else:
        net = net.to_crs(4326)

    # Reproject to meters so the buffer distance is meaningful
    net_m = net.to_crs(3857)
    pts_m = points.to_crs(3857)

    net_buf = net_m.buffer(buffer_km * 1000.0)
    buf_gdf = gpd.GeoDataFrame(geometry=net_buf, crs=3857)

    hits = gpd.sjoin(pts_m[["geometry"]], buf_gdf, how="left", predicate="intersects")
    points[out_col] = hits["index_right"].notna().values

    return points


def aggregate_weekly(points: gpd.GeoDataFrame, raions: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Aggregate FIRMS point events into a full weekly raion panel,
    including lagged and rolling fire-history features.
    """
    valid_weeks = pd.to_datetime(points["week_start"].dropna().unique())
    valid_weeks = sorted(valid_weeks)
    if not valid_weeks:
        raise ValueError("No valid FIRMS weeks found.")

    # Build a complete raion-week panel across the observed time span
    full_index = pd.MultiIndex.from_product(
        [raions["raion_id"], valid_weeks],
        names=["raion_id", "week_start"],
    )

    meta = raions[["raion_id", "raion_name", "oblast_name"]].drop_duplicates()

    # Only aggregate events that were successfully assigned to a raion
    pts = points.dropna(subset=["raion_id"]).copy()

    def pct(series: pd.Series) -> float:
        if len(series) == 0:
            return np.nan
        return float(series.mean())

    weekly = (
        pts.groupby(["raion_id", "week_start"], as_index=False)
        .agg(
            firms_event_count=("geometry", "size"),
            firms_high_conf_count=("is_high_confidence", "sum"),
            firms_nominal_conf_count=("is_nominal_confidence", "sum"),
            firms_low_conf_count=("is_low_confidence", "sum"),
            firms_day_count=("daynight", lambda s: int((s == "D").sum())),
            firms_night_count=("daynight", lambda s: int((s == "N").sum())),
            firms_type0_count=("is_type0", "sum"),
            firms_type2_count=("is_type2", "sum"),
            firms_frp_sum=("frp", "sum"),
            firms_frp_mean=("frp", "mean"),
            firms_frp_max=("frp", "max"),
            firms_bright_ti4_mean=("bright_ti4", "mean"),
            firms_near_roads_count=("near_roads", "sum"),
            firms_near_rail_count=("near_rail", "sum"),
            firms_near_roads_share=("near_roads", pct),
            firms_near_rail_share=("near_rail", pct),
        )
    )

    # Reindex so raion-weeks with no fire events are still present
    weekly = weekly.set_index(["raion_id", "week_start"]).reindex(full_index).reset_index()
    weekly = weekly.merge(meta, on="raion_id", how="left")

    fill_zero_cols = [
        "firms_event_count", "firms_high_conf_count", "firms_nominal_conf_count", "firms_low_conf_count",
        "firms_day_count", "firms_night_count", "firms_type0_count", "firms_type2_count",
        "firms_frp_sum", "firms_near_roads_count", "firms_near_rail_count",
    ]
    for col in fill_zero_cols:
        if col in weekly.columns:
            weekly[col] = weekly[col].fillna(0)

    weekly = weekly.sort_values(["raion_id", "week_start"]).copy()
    grp = weekly.groupby("raion_id")

    # Lagged and rolling summaries help capture recent fire activity history
    weekly["firms_event_count_lag1"] = grp["firms_event_count"].shift(1)
    weekly["firms_event_count_roll4_sum"] = (
        grp["firms_event_count"].shift(1).rolling(4, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    weekly["firms_frp_sum_lag1"] = grp["firms_frp_sum"].shift(1)
    weekly["firms_frp_sum_roll4_sum"] = (
        grp["firms_frp_sum"].shift(1).rolling(4, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    weekly["firms_change_vs_prev_week"] = weekly["firms_event_count"] - weekly["firms_event_count_lag1"]

    return weekly


def main() -> None:
    args = parse_args()

    # Load boundaries and FIRMS events, then spatially assign the events to raions
    raions = read_boundaries(args.boundary_zip, args.admin_id_col, args.admin_name_col)
    firms = load_firms_csvs(args.firms_csvs)
    points = assign_raions(firms, raions)

    # Optionally mark whether each event is close to roads or rail
    points = mark_near_network(points, args.roads_file, args.road_buffer_km, "near_roads")
    points = mark_near_network(points, args.rail_file, args.rail_buffer_km, "near_rail")

    weekly = aggregate_weekly(points, raions)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(args.out_csv, index=False)

    print(f"Saved weekly FIRMS features to: {args.out_csv}")
    print(f"Rows: {len(weekly):,}")
    print(f"Weeks: {weekly['week_start'].nunique():,}")
    print(f"Raions: {weekly['raion_id'].nunique():,}")

    # Optionally save the event-level points with their assigned raions
    if args.out_events_csv:
        out_pts = points.drop(columns="geometry").copy()
        out_pts.to_csv(args.out_events_csv, index=False)
        print(f"Saved FIRMS events with raion assignments to: {args.out_events_csv}")


if __name__ == "__main__":
    main()