#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd


# Europe LAEA works well as a metric CRS for buffering and distance-style work
# in and around Ukraine.
METRIC_CRS = "EPSG:3035"

# Vector formats this loader will try to detect inside extracted archives
VECTOR_SUFFIXES = {".gpkg", ".shp", ".geojson", ".json", ".gdb"}

# Road classes treated as "major" when that option is requested
MAJOR_ROAD_VALUES = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link",
}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for building weekly FIRMS features,
    with optional road/rail proximity based on dissolved buffers.
    """
    p = argparse.ArgumentParser(
        description="Build weekly raion-level FIRMS features with optional fast road/rail proximity via dissolved buffers."
    )
    p.add_argument("--boundary_zip", required=True, help="Path to ukr_admin_boundaries.geojson.zip or extracted admin2 vector.")
    p.add_argument("--firms_csvs", nargs="+", required=True, help="One or more FIRMS CSV files.")
    p.add_argument("--roads_file", default=None, help="Optional HOTOSM roads vector or zip.")
    p.add_argument("--rail_file", default=None, help="Optional HOTOSM rail vector or zip.")
    p.add_argument("--roads_layer", default=None, help="Optional layer name for roads GPKG/GDB.")
    p.add_argument("--rail_layer", default=None, help="Optional layer name for rail GPKG/GDB.")
    p.add_argument("--road_buffer_m", type=float, default=1000.0, help="Road proximity corridor width in meters.")
    p.add_argument("--rail_buffer_m", type=float, default=1000.0, help="Rail proximity corridor width in meters.")
    p.add_argument("--road_simplify_m", type=float, default=50.0, help="Simplify tolerance in meters before dissolving roads.")
    p.add_argument("--rail_simplify_m", type=float, default=50.0, help="Simplify tolerance in meters before dissolving rail.")
    p.add_argument("--major_roads_only", action="store_true", help="If supported by the roads file, keep only major roads.")
    p.add_argument("--skip_transport", action="store_true", help="Skip road/rail proximity entirely.")
    p.add_argument("--bbox_pad_deg", type=float, default=0.25, help="Padding around Ukraine boundary bbox when reading transport data.")
    p.add_argument("--admin_id_col", default="adm2_pcode")
    p.add_argument("--admin_name_col", default="adm2_name")
    p.add_argument("--week_freq", default="W-MON", help="Unused placeholder for compatibility; weeks are Monday starts.")
    p.add_argument("--out_csv", required=True, help="Output weekly raion CSV.")
    p.add_argument("--out_events_csv", default=None, help="Optional output event-level joined CSV.")
    return p.parse_args()


def monday_week_start(series: pd.Series) -> pd.Series:
    """
    Convert dates to Monday-start week anchors.
    """
    dt = pd.to_datetime(series, errors="coerce")
    return dt - pd.to_timedelta(dt.dt.weekday, unit="D")


def normalize_confidence(series: pd.Series) -> pd.Series:
    """
    Normalize different confidence strings into a smaller set of values
    so downstream flags are more consistent across input files.
    """
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "h": "h", "high": "h", "high confidence": "h",
        "n": "n", "nominal": "n", "nominal confidence": "n",
        "l": "l", "low": "l", "low confidence": "l",
    }
    return s.map(mapping).fillna(s)


def extract_zip(path: Path, temp_root: Path) -> Path:
    """
    Extract a zip archive into a temporary folder and return the
    extracted root directory.
    """
    out_dir = temp_root / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(out_dir)
    return out_dir


def find_vector_candidates(root: Path) -> list[Path]:
    """
    Recursively search an extracted directory for vector files
    that geopandas/fiona may be able to read.
    """
    candidates = []
    for p in root.rglob("*"):
        if p.suffix.lower() in VECTOR_SUFFIXES or p.name.lower().endswith(".gdb"):
            # Skip common junk files/folders from Mac zip archives
            if "__macosx" in str(p).lower():
                continue
            if p.name.startswith("._"):
                continue
            candidates.append(p)
    return sorted(set(candidates))


def _read_all_layers(
    path: Path,
    bbox: Optional[tuple[float, float, float, float]],
    requested_layer: Optional[str],
    geometry_only: bool = False,
) -> gpd.GeoDataFrame:
    """
    Read one vector source, including all layers if needed.
    This is mainly for GPKG/GDB inputs that may contain multiple layers.
    """
    suffix = path.suffix.lower()

    if suffix == ".gdb" or path.name.lower().endswith(".gdb"):
        layers = [requested_layer] if requested_layer else list(fiona.listlayers(path))
        frames = []
        for layer in layers:
            if layer is None:
                continue
            g = gpd.read_file(path, layer=layer, bbox=bbox)
            if not g.empty:
                frames.append(g[["geometry"]].copy() if geometry_only else g)
        return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry") if frames else gpd.GeoDataFrame(geometry=[])

    if suffix == ".gpkg":
        if requested_layer:
            g = gpd.read_file(path, layer=requested_layer, bbox=bbox)
            return g[["geometry"]].copy() if geometry_only else g

        try:
            layers = list(fiona.listlayers(path))
        except Exception:
            layers = []

        if not layers:
            g = gpd.read_file(path, bbox=bbox)
            return g[["geometry"]].copy() if geometry_only else g

        frames = []
        for layer in layers:
            g = gpd.read_file(path, layer=layer, bbox=bbox)
            if not g.empty:
                frames.append(g[["geometry"]].copy() if geometry_only else g)
        return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry") if frames else gpd.GeoDataFrame(geometry=[])

    # For simpler formats like shapefile/geojson, just read directly
    g = gpd.read_file(path, bbox=bbox)
    return g[["geometry"]].copy() if geometry_only else g


def load_vector(
    path_str: str,
    bbox: Optional[tuple[float, float, float, float]] = None,
    layer: Optional[str] = None,
    geometry_only: bool = False,
) -> gpd.GeoDataFrame:
    """
    Load a vector dataset from either a plain file path or a zip archive.
    If zipped, try all candidate vector files until one or more read cleanly.
    """
    path = Path(path_str)
    with tempfile.TemporaryDirectory(prefix="vector_read_") as td:
        temp_root = Path(td)

        if path.suffix.lower() == ".zip":
            extracted = extract_zip(path, temp_root)
            candidates = find_vector_candidates(extracted)
            if not candidates:
                raise FileNotFoundError(f"No vector files found inside: {path}")

            frames = []
            for cand in candidates:
                try:
                    g = _read_all_layers(cand, bbox=bbox, requested_layer=layer, geometry_only=geometry_only)
                except Exception:
                    continue
                if not g.empty:
                    frames.append(g)

            if not frames:
                raise FileNotFoundError(f"Could not read any vector data from: {path}")

            gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry")
        else:
            gdf = _read_all_layers(path, bbox=bbox, requested_layer=layer, geometry_only=geometry_only)

    if gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    # Standardize everything to EPSG:4326 first
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf


def read_boundaries(boundary_zip: str, admin_id_col: str, admin_name_col: str) -> gpd.GeoDataFrame:
    """
    Read the Ukraine admin boundaries and standardize the key columns
    used later in the weekly aggregation.
    """
    gdf = load_vector(boundary_zip, geometry_only=False)

    keep_cols = [admin_id_col, admin_name_col, "adm1_name", "geometry"]
    missing = [c for c in [admin_id_col, admin_name_col] if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing boundary columns: {missing}. Available: {list(gdf.columns)}")

    present = [c for c in keep_cols if c in gdf.columns]
    raions = gdf[present].copy().rename(
        columns={
            admin_id_col: "raion_id",
            admin_name_col: "raion_name",
            "adm1_name": "oblast_name",
        }
    )

    if "oblast_name" not in raions.columns:
        raions["oblast_name"] = None

    return raions.to_crs("EPSG:4326")


def compute_bbox_with_pad(gdf: gpd.GeoDataFrame, pad_deg: float) -> tuple[float, float, float, float]:
    """
    Compute a padded bounding box in geographic coordinates.
    This is used to limit transport-data reads to the general study area.
    """
    minx, miny, maxx, maxy = gdf.to_crs("EPSG:4326").total_bounds
    return (minx - pad_deg, miny - pad_deg, maxx + pad_deg, maxy + pad_deg)


def maybe_filter_major_roads(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    If a reasonable road-class field exists, keep only major roads.
    Otherwise just return the original data.
    """
    candidate_cols = ["highway", "fclass", "class", "type", "road_type"]
    found = None
    for c in candidate_cols:
        if c in gdf.columns:
            found = c
            break

    if found is None:
        return gdf

    vals = gdf[found].astype(str).str.strip().str.lower()
    keep = vals.isin(MAJOR_ROAD_VALUES)
    subset = gdf.loc[keep].copy()

    # If the filter removes everything, fall back to the full dataset
    return subset if not subset.empty else gdf


def build_corridor(
    path_str: Optional[str],
    bbox: tuple[float, float, float, float],
    layer: Optional[str],
    buffer_m: float,
    simplify_m: float,
    major_roads_only: bool = False,
) -> Optional[object]:
    """
    Build one dissolved buffered transport corridor geometry.
    This is faster than checking every event against many raw lines.
    """
    if not path_str:
        return None

    raw = load_vector(path_str, bbox=bbox, layer=layer, geometry_only=not major_roads_only)
    if raw.empty:
        return None

    # If major-roads filtering is requested, reload with attributes preserved
    # because the geometry-only read above may have dropped those fields.
    if major_roads_only:
        path = Path(path_str)
        with tempfile.TemporaryDirectory(prefix="vector_attr_read_") as td:
            temp_root = Path(td)
            if path.suffix.lower() == ".zip":
                extracted = extract_zip(path, temp_root)
                candidates = find_vector_candidates(extracted)
                frames = []
                for cand in candidates:
                    try:
                        g = _read_all_layers(cand, bbox=bbox, requested_layer=layer, geometry_only=False)
                    except Exception:
                        continue
                    if not g.empty:
                        frames.append(g)
                if frames:
                    raw = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry")
            else:
                raw = _read_all_layers(path, bbox=bbox, requested_layer=layer, geometry_only=False)

            if raw.crs is None:
                raw = raw.set_crs("EPSG:4326")
            else:
                raw = raw.to_crs("EPSG:4326")

            raw = maybe_filter_major_roads(raw)

    raw = raw[raw.geometry.notna() & ~raw.geometry.is_empty].copy()
    if raw.empty:
        return None

    metric = raw.to_crs(METRIC_CRS)

    # Simplify first to reduce geometry complexity before dissolving/buffering
    if simplify_m and simplify_m > 0:
        metric["geometry"] = metric.geometry.simplify(simplify_m, preserve_topology=True)
        metric = metric[metric.geometry.notna() & ~metric.geometry.is_empty].copy()

    if metric.empty:
        return None

    # Dissolve all linework into one geometry, then buffer once
    try:
        merged = metric.geometry.unary_union
    except Exception:
        merged = metric.dissolve().geometry.iloc[0]

    corridor = gpd.GeoSeries([merged], crs=METRIC_CRS).buffer(buffer_m).iloc[0]
    return corridor


def load_firms_csvs(paths: Iterable[str]) -> pd.DataFrame:
    """
    Load one or more FIRMS CSVs and standardize the main fields
    used later for weekly aggregation.
    """
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = Path(path).name
        frames.append(df)

    if not frames:
        raise ValueError("No FIRMS CSVs were loaded.")

    df = pd.concat(frames, ignore_index=True)

    # Handle minor column-name inconsistencies by matching case-insensitively
    rename_map = {}
    for wanted in ["latitude", "longitude", "acq_date", "confidence", "frp", "daynight", "type", "bright_ti4"]:
        if wanted in df.columns:
            continue
        matches = [c for c in df.columns if c.strip().lower() == wanted]
        if matches:
            rename_map[matches[0]] = wanted
    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["latitude", "longitude", "acq_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"FIRMS CSV missing required columns: {missing}. Available: {list(df.columns)}")

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df[df["acq_date"].notna()].copy()
    df["week_start"] = monday_week_start(df["acq_date"])

    if "confidence" in df.columns:
        df["confidence_norm"] = normalize_confidence(df["confidence"])
    else:
        df["confidence_norm"] = "unknown"

    df["is_high_confidence"] = df["confidence_norm"].eq("h")
    df["is_nominal_confidence"] = df["confidence_norm"].eq("n")
    df["is_low_confidence"] = df["confidence_norm"].eq("l")

    if "type" in df.columns:
        df["type"] = pd.to_numeric(df["type"], errors="coerce").astype("Int64")
    else:
        df["type"] = pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"))

    df["is_type0"] = df["type"].eq(0)
    df["is_type2"] = df["type"].eq(2)
    df["is_type3"] = df["type"].eq(3)

    if "frp" not in df.columns:
        df["frp"] = np.nan
    else:
        df["frp"] = pd.to_numeric(df["frp"], errors="coerce")

    if "bright_ti4" not in df.columns:
        df["bright_ti4"] = np.nan
    else:
        df["bright_ti4"] = pd.to_numeric(df["bright_ti4"], errors="coerce")

    if "daynight" not in df.columns:
        df["daynight"] = ""
    df["daynight"] = df["daynight"].astype(str).str.upper().str.strip()

    return df


def spatially_assign_raions(firms_df: pd.DataFrame, raions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert FIRMS rows into point geometry and assign each event
    to a raion polygon using a spatial join.
    """
    firms_gdf = gpd.GeoDataFrame(
        firms_df.copy(),
        geometry=gpd.points_from_xy(firms_df["longitude"], firms_df["latitude"]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        firms_gdf,
        raions[["raion_id", "raion_name", "oblast_name", "geometry"]],
        how="inner",
        predicate="within",
    ).drop(columns=["index_right"])

    return joined


def add_transport_flags(
    events_gdf: gpd.GeoDataFrame,
    road_corridor,
    rail_corridor,
) -> gpd.GeoDataFrame:
    """
    Mark each FIRMS event as near roads and/or rail by checking
    whether the point intersects the dissolved buffered corridor.
    """
    out = events_gdf.to_crs(METRIC_CRS).copy()

    if road_corridor is None:
        out["near_roads"] = False
    else:
        out["near_roads"] = out.geometry.intersects(road_corridor)

    if rail_corridor is None:
        out["near_rail"] = False
    else:
        out["near_rail"] = out.geometry.intersects(rail_corridor)

    return out.to_crs("EPSG:4326")


def build_week_spine(events: gpd.GeoDataFrame, raions: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Build a complete raion-week scaffold covering the observed date range,
    so weeks with zero events are still present in the output.
    """
    if events.empty:
        raise ValueError("No FIRMS events remained after spatial join to raions.")

    week_min = pd.to_datetime(events["week_start"]).min()
    week_max = pd.to_datetime(events["week_start"]).max()
    weeks = pd.date_range(start=week_min, end=week_max, freq="W-MON")

    meta = raions[["raion_id", "raion_name", "oblast_name"]].drop_duplicates().copy()
    spine = meta.assign(_key=1).merge(
        pd.DataFrame({"week_start": weeks, "_key": 1}),
        on="_key",
        how="outer",
    ).drop(columns="_key")

    return spine


def aggregate_weekly(events: gpd.GeoDataFrame, raions: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Aggregate event-level FIRMS data into weekly raion-level features,
    then add lagged and rolling historical summaries.
    """
    group_cols = ["raion_id", "week_start"]
    weekly = (
        events.groupby(group_cols, dropna=False)
        .agg(
            firms_event_count=("raion_id", "size"),
            firms_high_conf_count=("is_high_confidence", "sum"),
            firms_nominal_conf_count=("is_nominal_confidence", "sum"),
            firms_low_conf_count=("is_low_confidence", "sum"),
            firms_day_count=("daynight", lambda s: (s == "D").sum()),
            firms_night_count=("daynight", lambda s: (s == "N").sum()),
            firms_type0_count=("is_type0", "sum"),
            firms_type2_count=("is_type2", "sum"),
            firms_type3_count=("is_type3", "sum"),
            firms_frp_sum=("frp", "sum"),
            firms_frp_mean=("frp", "mean"),
            firms_frp_max=("frp", "max"),
            firms_near_roads_count=("near_roads", "sum"),
            firms_near_rail_count=("near_rail", "sum"),
            firms_bright_ti4_mean=("bright_ti4", "mean"),
        )
        .reset_index()
    )

    weekly["week_start"] = pd.to_datetime(weekly["week_start"])

    spine = build_week_spine(events, raions)
    out = spine.merge(weekly, on=["raion_id", "week_start"], how="left")

    count_cols = [
        "firms_event_count", "firms_high_conf_count", "firms_nominal_conf_count", "firms_low_conf_count",
        "firms_day_count", "firms_night_count", "firms_type0_count", "firms_type2_count", "firms_type3_count",
        "firms_near_roads_count", "firms_near_rail_count",
    ]
    sum_cols = ["firms_frp_sum"]
    fill_zero_cols = count_cols + sum_cols
    for c in fill_zero_cols:
        out[c] = out[c].fillna(0)

    # Keep shares as NaN for weeks with no events
    denom = out["firms_event_count"].replace(0, np.nan)
    out["firms_near_roads_share"] = out["firms_near_roads_count"] / denom
    out["firms_near_rail_share"] = out["firms_near_rail_count"] / denom

    out = out.sort_values(["raion_id", "week_start"]).reset_index(drop=True)

    by_raion = out.groupby("raion_id", dropna=False)
    out["firms_event_count_lag1"] = by_raion["firms_event_count"].shift(1)
    out["firms_event_count_roll4_sum"] = by_raion["firms_event_count"].shift(1).rolling(4, min_periods=1).sum().reset_index(level=0, drop=True)
    out["firms_frp_sum_lag1"] = by_raion["firms_frp_sum"].shift(1)
    out["firms_frp_sum_roll4_sum"] = by_raion["firms_frp_sum"].shift(1).rolling(4, min_periods=1).sum().reset_index(level=0, drop=True)
    out["firms_change_vs_prev_week"] = out["firms_event_count"] - out["firms_event_count_lag1"]

    return out


def main() -> None:
    args = parse_args()

    # Load boundaries and FIRMS events, then assign each event to a raion
    raions = read_boundaries(args.boundary_zip, args.admin_id_col, args.admin_name_col)
    firms_df = load_firms_csvs(args.firms_csvs)
    events = spatially_assign_raions(firms_df, raions)

    # Limit transport-data reads to a padded Ukraine bounding box
    bbox = compute_bbox_with_pad(raions, args.bbox_pad_deg)

    if args.skip_transport:
        road_corridor = None
        rail_corridor = None
    else:
        road_corridor = build_corridor(
            args.roads_file,
            bbox=bbox,
            layer=args.roads_layer,
            buffer_m=args.road_buffer_m,
            simplify_m=args.road_simplify_m,
            major_roads_only=args.major_roads_only,
        )
        rail_corridor = build_corridor(
            args.rail_file,
            bbox=bbox,
            layer=args.rail_layer,
            buffer_m=args.rail_buffer_m,
            simplify_m=args.rail_simplify_m,
            major_roads_only=False,
        )

    # Add near-road / near-rail flags before weekly aggregation
    events = add_transport_flags(events, road_corridor=road_corridor, rail_corridor=rail_corridor)
    weekly = aggregate_weekly(events, raions)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    weekly = weekly.copy()
    weekly["week_start"] = pd.to_datetime(weekly["week_start"]).dt.strftime("%Y-%m-%d")
    weekly.to_csv(out_path, index=False)

    print(f"Saved weekly FIRMS features to: {out_path}")
    print(f"Weekly rows: {len(weekly):,}")
    print(f"Nonzero near-road events: {int(events['near_roads'].sum()):,}")
    print(f"Nonzero near-rail events: {int(events['near_rail'].sum()):,}")

    if args.out_events_csv:
        events_out = events.drop(columns="geometry").copy()
        events_out["week_start"] = pd.to_datetime(events_out["week_start"]).dt.strftime("%Y-%m-%d")
        events_out["acq_date"] = pd.to_datetime(events_out["acq_date"]).dt.strftime("%Y-%m-%d")
        events_out.to_csv(args.out_events_csv, index=False)
        print(f"Saved event-level FIRMS output to: {args.out_events_csv}")


if __name__ == "__main__":
    main()