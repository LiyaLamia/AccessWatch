#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rasterio.features
from affine import Affine

# Used to extract year + day-of-year from Black Marble filenames
DATE_RE = re.compile(r"A(\d{4})(\d{3})")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for building daily and weekly
    Black Marble features at the raion level.
    """
    p = argparse.ArgumentParser(
        description="Build raion-level daily and weekly Black Marble features from VNP46A2 H5 tiles."
    )
    p.add_argument("--input", required=True, help="Directory containing VNP46A2 *.h5 files (nested folders OK), or a single .h5 file")
    p.add_argument("--boundary_zip", required=True, help="Path to ukr_admin_boundaries.geojson.zip or an extracted admin2 geojson")
    p.add_argument("--out_daily_csv", required=True)
    p.add_argument("--out_weekly_csv", required=True)
    p.add_argument("--admin_id_col", default="adm2_pcode")
    p.add_argument("--admin_name_col", default="adm2_name")
    return p.parse_args()


def resolve_boundary_path(boundary_path: str) -> str:
    """
    Resolve the boundary input into a path geopandas can read.
    If a zip file is given, point to the admin2 GeoJSON inside it.
    """
    p = Path(boundary_path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as zf:
            # Prefer the expected file name, but fall back to any admin2-like GeoJSON
            candidates = [n for n in zf.namelist() if n.lower().endswith("ukr_admin2.geojson")]
            if not candidates:
                candidates = [n for n in zf.namelist() if n.lower().endswith(".geojson") and "admin2" in n.lower()]
            if not candidates:
                raise FileNotFoundError("Could not find ukr_admin2.geojson inside boundary zip")
            inner = candidates[0]
        return f"zip://{p}!{inner}"
    return boundary_path


def read_boundaries(boundary_zip: str, admin_id_col: str, admin_name_col: str) -> gpd.GeoDataFrame:
    """
    Load the admin boundaries, keep the ID/name fields we need,
    and assign an integer zone ID for rasterization.
    """
    path = resolve_boundary_path(boundary_zip)
    gdf = gpd.read_file(path)

    for col in (admin_id_col, admin_name_col):
        if col not in gdf.columns:
            raise ValueError(f"Required column '{col}' not found in boundaries. Available: {list(gdf.columns)}")

    raions = gdf[[admin_id_col, admin_name_col, "geometry"]].copy()
    raions = raions.rename(columns={admin_id_col: "raion_id", admin_name_col: "raion_name"})

    # Black Marble uses geographic coordinates, so keep boundaries in EPSG:4326
    if raions.crs is None:
        raions = raions.set_crs(4326)
    else:
        raions = raions.to_crs(4326)

    # zone_id is used later when rasterizing the polygons into image space
    raions["zone_id"] = np.arange(1, len(raions) + 1, dtype=np.int32)
    return raions


def discover_files(inp: str) -> list[Path]:
    """
    Accept either a single .h5 file or a directory containing
    Black Marble H5 files, including files inside nested folders.
    """
    p = Path(inp)

    if p.is_file():
        if p.suffix.lower() == ".h5":
            return [p]
        raise FileNotFoundError(f"Expected a .h5 file, got: {p.name}")

    if not p.is_dir():
        raise FileNotFoundError(f"Input path not found: {inp}")

    # Search recursively because the input directory may contain subfolders
    files = sorted(p.rglob("VNP46A2*.h5"))
    if files:
        return files

    # Helpful error in case the folder only contains preview images
    jpgs = sorted([*p.rglob("*.jpg"), *p.rglob("*.jpeg"), *p.rglob("*.png")])
    if jpgs:
        example = jpgs[0]
        raise FileNotFoundError(
            "Found preview image files but no VNP46A2 .h5 files. "
            f"Example image: {example}. This script needs the original .h5 Black Marble data files, not JPG previews."
        )

    raise FileNotFoundError(f"No VNP46A2 .h5 files found anywhere under {inp}")


def parse_date_from_filename(name: str) -> pd.Timestamp:
    """
    Parse the acquisition date from the filename using
    the year + day-of-year pattern.
    """
    m = DATE_RE.search(name)
    if not m:
        raise ValueError(f"Could not parse date from {name}")
    year = int(m.group(1))
    doy = int(m.group(2))
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)


def affine_from_latlon(lat: np.ndarray, lon: np.ndarray) -> Affine:
    """
    Build the affine transform for the tile from its 1D lat/lon arrays.
    This is needed to align the raster grid with the boundary polygons.
    """
    dy = float(abs(lat[1] - lat[0]))
    dx = float(abs(lon[1] - lon[0]))
    west = float(lon.min()) - dx / 2.0
    north = float(lat.max()) + dy / 2.0
    return Affine(dx, 0.0, west, 0.0, -dy, north)


def rasterize_zones(raions: gpd.GeoDataFrame, shape: tuple[int, int], transform: Affine) -> np.ndarray:
    """
    Rasterize raion polygons into an integer zone array
    with the same shape as the NTL tile.
    """
    shapes = ((geom, int(zid)) for geom, zid in zip(raions.geometry, raions.zone_id))
    return rasterio.features.rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )


def accumulate_zone_stats(zone_arr: np.ndarray, values: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Accumulate per-zone counts, sums, and squared sums
    so mean and standard deviation can be computed efficiently.
    """
    zones = zone_arr[valid_mask]
    vals = values[valid_mask]
    max_zone = int(zone_arr.max()) if zone_arr.size else 0

    counts = np.bincount(zones, minlength=max_zone + 1)
    sums = np.bincount(zones, weights=vals, minlength=max_zone + 1)
    sums_sq = np.bincount(zones, weights=np.square(vals), minlength=max_zone + 1)

    return counts, sums, sums_sq


def per_file_zone_summary(h5_path: Path, raions: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Build one raion-level summary table from a single Black Marble H5 tile.
    """
    with h5py.File(h5_path, "r") as f:
        root = f["HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"]
        ntl_ds = root["Gap_Filled_DNB_BRDF-Corrected_NTL"]
        ntl = ntl_ds[:].astype(np.float32)
        q = root["Mandatory_Quality_Flag"][:].astype(np.uint8)
        latest = root["Latest_High_Quality_Retrieval"][:].astype(np.uint16)
        lat = root["lat"][:].astype(np.float64)
        lon = root["lon"][:].astype(np.float64)

        # Read fill value and scale factor from the dataset metadata
        fill_val = ntl_ds.attrs.get("_FillValue", -9999)
        if isinstance(fill_val, np.ndarray):
            fill_val = float(fill_val.flat[0])
        else:
            fill_val = float(fill_val)

        scale = ntl_ds.attrs.get("scale_factor", 0.1)
        if isinstance(scale, np.ndarray):
            scale = float(scale.flat[0])
        else:
            scale = float(scale)

    # Convert stored values to scaled NTL values
    ntl = ntl * scale

    transform = affine_from_latlon(lat, lon)
    zone_arr = rasterize_zones(raions, ntl.shape, transform)

    # Valid pixels must belong to a raion and not be fill/missing values
    in_zone = zone_arr > 0
    valid = in_zone & np.isfinite(ntl) & (ntl != fill_val * scale)

    # Two quality-related masks used for extra summary features
    highq = valid & (q == 0)
    latest_hq = valid & (latest > 0)

    counts, sums, sums_sq = accumulate_zone_stats(zone_arr, ntl, valid)
    hq_counts, _, _ = accumulate_zone_stats(zone_arr, ntl, highq)
    latest_counts, _, _ = accumulate_zone_stats(zone_arr, ntl, latest_hq)

    # Count how many raster pixels fall inside each zone, regardless of validity
    pixel_counts = np.bincount(zone_arr[in_zone], minlength=int(zone_arr.max()) + 1)

    rows = []
    file_date = parse_date_from_filename(h5_path.name)

    for _, row in raions[["zone_id", "raion_id", "raion_name"]].iterrows():
        zid = int(row.zone_id)
        c = int(counts[zid]) if zid < len(counts) else 0

        if c == 0:
            mean = np.nan
            std = np.nan
        else:
            mean = float(sums[zid] / c)
            var = max(float(sums_sq[zid] / c) - mean * mean, 0.0)
            std = math.sqrt(var)

        total_pix = int(pixel_counts[zid]) if zid < len(pixel_counts) else 0

        rows.append(
            {
                "date": file_date,
                "raion_id": row.raion_id,
                "raion_name": row.raion_name,
                "ntl_mean": mean,
                "ntl_std": std,
                "ntl_valid_pixels": c,
                "ntl_total_pixels": total_pix,
                "ntl_high_quality_share": (float(hq_counts[zid] / c) if c else np.nan),
                "ntl_latest_hq_share": (float(latest_counts[zid] / c) if c else np.nan),
                "source_file_count": 1,
            }
        )

    return pd.DataFrame(rows)


def combine_same_day_tile_summaries(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine tile-level summaries from the same day into one daily
    raion-level table. This matters when the study area spans multiple tiles.
    """
    daily = pd.concat(frames, ignore_index=True)

    def weighted_mean(g: pd.DataFrame, val_col: str, w_col: str = "ntl_valid_pixels") -> float:
        # Weight by number of valid pixels so tiles with more coverage matter more
        valid = g[[val_col, w_col]].dropna()
        if valid.empty or valid[w_col].sum() == 0:
            return np.nan
        return float(np.average(valid[val_col], weights=valid[w_col]))

    grouped = []
    for (date, rid, rname), g in daily.groupby(["date", "raion_id", "raion_name"], dropna=False):
        total_valid = int(g["ntl_valid_pixels"].sum())
        total_pixels = int(g["ntl_total_pixels"].sum())

        grouped.append(
            {
                "date": date,
                "raion_id": rid,
                "raion_name": rname,
                "ntl_mean": weighted_mean(g, "ntl_mean"),
                "ntl_std_mean": weighted_mean(g, "ntl_std"),
                "ntl_valid_pixels": total_valid,
                "ntl_total_pixels": total_pixels,
                "ntl_high_quality_share": weighted_mean(g, "ntl_high_quality_share"),
                "ntl_latest_hq_share": weighted_mean(g, "ntl_latest_hq_share"),
                "source_tile_count": int((g["ntl_valid_pixels"] > 0).sum()),
            }
        )

    return pd.DataFrame(grouped)


def build_weekly_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily NTL summaries into weekly features and add
    a few simple temporal change features.
    """
    df = daily_df.copy().sort_values(["raion_id", "date"])

    # Use Monday as the week anchor
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")

    weekly = (
        df.groupby(["raion_id", "raion_name", "week_start"], as_index=False)
        .agg(
            ntl_week_mean=("ntl_mean", "mean"),
            ntl_week_median=("ntl_mean", "median"),
            ntl_week_std_mean=("ntl_std_mean", "mean"),
            ntl_high_quality_share=("ntl_high_quality_share", "mean"),
            ntl_latest_hq_share=("ntl_latest_hq_share", "mean"),
            ntl_obs_days=("date", "nunique"),
            ntl_valid_pixels_sum=("ntl_valid_pixels", "sum"),
        )
    )

    weekly = weekly.sort_values(["raion_id", "week_start"])

    # Change compared with the previous observed week
    prev = weekly.groupby("raion_id")["ntl_week_mean"].shift(1)
    weekly["ntl_change_vs_prev_week"] = weekly["ntl_week_mean"] - prev

    # Change compared with the trailing 4-week average, excluding the current week
    rolling4 = weekly.groupby("raion_id")["ntl_week_mean"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    weekly["ntl_change_vs_rolling4"] = weekly["ntl_week_mean"] - rolling4

    # Relative change can be useful, but avoid division by zero
    weekly["ntl_pct_change_vs_prev_week"] = weekly["ntl_change_vs_prev_week"] / prev.replace(0, np.nan)

    return weekly


def main() -> None:
    args = parse_args()

    # Load boundaries and discover all matching Black Marble files
    raions = read_boundaries(args.boundary_zip, args.admin_id_col, args.admin_name_col)
    files = discover_files(args.input)
    print(f"Found {len(files)} VNP46A2 files")

    frames = []
    for i, h5_path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Processing {h5_path.name}")
        frames.append(per_file_zone_summary(h5_path, raions))

    # First build daily features, then aggregate them into weekly features
    daily = combine_same_day_tile_summaries(frames)
    weekly = build_weekly_features(daily)

    Path(args.out_daily_csv).parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.out_daily_csv, index=False)
    weekly.to_csv(args.out_weekly_csv, index=False)

    print(f"Saved daily features to: {args.out_daily_csv}")
    print(f"Saved weekly features to: {args.out_weekly_csv}")


if __name__ == "__main__":
    main()