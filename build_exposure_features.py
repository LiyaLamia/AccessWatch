#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for building static exposure features
    at the raion level.
    """
    p = argparse.ArgumentParser(
        description="Build raion-level exposure features from population, populated places, border crossings, and optional health facilities."
    )
    p.add_argument("--boundary_zip", required=True, help="Path to ukr_admin_boundaries.geojson.zip or extracted ukr_admin2.geojson")
    p.add_argument("--border_crossings", required=True, help="Path to UKR_Border_Crossings.gpkg/geojson/shp")
    p.add_argument("--populated_places_xlsx", required=True, help="Path to ukr-populated-places.xlsx")
    p.add_argument("--population_raster_2022", default=None, help="Optional WorldPop 2022 count GeoTIFF")
    p.add_argument("--population_raster_2023", default=None, help="Optional WorldPop 2023 count GeoTIFF")
    p.add_argument("--health_facilities", default=None, help="Optional health facilities GPKG/GeoJSON/SHP")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--admin_id_col", default="adm2_pcode")
    p.add_argument("--admin_name_col", default="adm2_name")
    p.add_argument("--oblast_name_col", default="adm1_name")
    return p.parse_args()


def resolve_boundary_path(boundary_path: str) -> str:
    """
    Resolve the boundary input into a path geopandas can read.
    If a zip file is given, point to the admin2 GeoJSON inside it.
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


def read_boundaries(boundary_zip: str, admin_id_col: str, admin_name_col: str, oblast_name_col: str) -> gpd.GeoDataFrame:
    """
    Read the raion boundaries and keep the core identifying fields
    plus area and geometry.
    """
    path = resolve_boundary_path(boundary_zip)
    gdf = gpd.read_file(path)

    needed = [admin_id_col, admin_name_col, oblast_name_col, "area_sqkm", "geometry"]
    missing = [c for c in needed if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing boundary columns: {missing}. Available: {gdf.columns.tolist()}")

    raions = gdf[needed].copy().rename(
        columns={
            admin_id_col: "raion_id",
            admin_name_col: "raion_name",
            oblast_name_col: "oblast_name",
        }
    )

    # Keep everything in geographic coordinates for the point-based joins
    if raions.crs is None:
        raions = raions.set_crs(4326)
    else:
        raions = raions.to_crs(4326)

    return raions


def read_border_crossings(path: str) -> gpd.GeoDataFrame:
    """
    Load border crossings and ensure they are in EPSG:4326.
    """
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)
    return gdf


def read_populated_places(path: str) -> pd.DataFrame:
    """
    Read the populated places Excel file and standardize
    the column names used later in the script.
    """
    df = pd.read_excel(path, sheet_name="Populated Places")

    # Rename common source columns to simpler names used downstream
    rename_map = {
        "ADM2_PCODE": "raion_id",
        "ADM2_EN": "raion_name_from_places",
        "ADM1_EN": "oblast_name_from_places",
        "LAT": "lat",
        "LON": "lon",
        "TYPE_EN": "place_type",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    required = ["raion_id", "lat", "lon"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing populated-place columns: {missing}. Available: {df.columns.tolist()}")

    return df


def count_populated_places(places_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count populated places per raion and, if available,
    also break them down by place type.
    """
    out = places_df.groupby("raion_id", as_index=False).size().rename(
        columns={"size": "populated_place_count"}
    )

    if "place_type" in places_df.columns:
        pivot = (
            places_df.assign(place_type=places_df["place_type"].fillna("Unknown"))
            .pivot_table(
                index="raion_id",
                columns="place_type",
                values="lat",
                aggfunc="count",
                fill_value=0
            )
            .reset_index()
        )

        # Make place-type column names consistent and safe for CSV/model use
        new_cols = []
        for c in pivot.columns:
            if c == "raion_id":
                new_cols.append(c)
            else:
                safe = str(c).strip().lower().replace(" ", "_").replace("/", "_")
                safe = "".join(ch for ch in safe if ch.isalnum() or ch == "_")
                new_cols.append(f"place_count_{safe}")
        pivot.columns = new_cols

        out = out.merge(pivot, on="raion_id", how="left")

    return out


def zonal_sum_mean(raster_path: str, raions: gpd.GeoDataFrame, prefix: str) -> pd.DataFrame:
    """
    For each raion polygon, extract raster values and compute
    the total and mean under that polygon.
    """
    rows = []

    with rasterio.open(raster_path) as src:
        if raions.crs != src.crs:
            work = raions.to_crs(src.crs)
        else:
            work = raions

        nodata = src.nodata

        for _, row in work.iterrows():
            try:
                arr, _ = mask(src, [row.geometry], crop=True, filled=False)
            except ValueError:
                # Happens when a polygon does not overlap the raster at all
                data = np.array([], dtype=np.float32)
            else:
                data = arr[0]

                if np.ma.isMaskedArray(data):
                    data = data.compressed()
                else:
                    data = data.ravel()

                if nodata is not None:
                    data = data[data != nodata]

            data = data[np.isfinite(data)] if data.size else data

            total = float(data.sum()) if data.size else 0.0
            mean = float(data.mean()) if data.size else np.nan

            rows.append({
                "raion_id": row.raion_id,
                f"{prefix}_sum": total,
                f"{prefix}_mean": mean,
            })

    return pd.DataFrame(rows)


def count_points_in_raions(points: gpd.GeoDataFrame, raions: gpd.GeoDataFrame, count_col: str) -> pd.DataFrame:
    """
    Spatially join point features to raions and count
    how many fall inside each raion.
    """
    points = points.to_crs(raions.crs)
    joined = gpd.sjoin(points, raions[["raion_id", "geometry"]], how="left", predicate="within")
    out = joined.groupby("raion_id", as_index=False).size().rename(columns={"size": count_col})
    return out


def nearest_crossing_distance_km(crossings: gpd.GeoDataFrame, raions: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compute distance from each raion centroid to the nearest
    border crossing, in kilometers.
    """
    if crossings.empty:
        return pd.DataFrame({
            "raion_id": raions["raion_id"],
            "nearest_border_crossing_km": np.nan
        })

    # Reproject to a metric CRS so distance is measured in meters
    r3857 = raions.to_crs(3857).copy()
    c3857 = crossings.to_crs(3857).copy()

    # union_all is preferred in newer shapely/geopandas versions
    crossing_union = c3857.geometry.union_all() if hasattr(c3857.geometry, 'union_all') else c3857.unary_union

    centroids = r3857.geometry.centroid
    d_km = centroids.distance(crossing_union) / 1000.0

    return pd.DataFrame({
        "raion_id": r3857["raion_id"].values,
        "nearest_border_crossing_km": d_km.values,
    })


def read_health_facilities(path: str) -> gpd.GeoDataFrame:
    """
    Load optional health facility points and ensure
    they are in EPSG:4326.
    """
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)
    return gdf


def main() -> None:
    args = parse_args()

    # Start from the boundary file since it provides the master raion list
    raions = read_boundaries(
        args.boundary_zip,
        args.admin_id_col,
        args.admin_name_col,
        args.oblast_name_col
    )
    out = raions[["raion_id", "raion_name", "oblast_name", "area_sqkm"]].copy()

    # Border crossing features
    crossings = read_border_crossings(args.border_crossings)
    border_counts = count_points_in_raions(crossings, raions, "border_crossing_count")
    border_nearest = nearest_crossing_distance_km(crossings, raions)

    out = out.merge(border_counts, on="raion_id", how="left")
    out = out.merge(border_nearest, on="raion_id", how="left")
    out["border_crossing_count"] = out["border_crossing_count"].fillna(0).astype(int)

    # Populated place features
    places_df = read_populated_places(args.populated_places_xlsx)
    place_counts = count_populated_places(places_df)
    out = out.merge(place_counts, on="raion_id", how="left")

    for c in out.columns:
        if c.startswith("place_count_") or c == "populated_place_count":
            out[c] = out[c].fillna(0).astype(int)

    # Optional population raster summaries
    if args.population_raster_2022:
        pop22 = zonal_sum_mean(args.population_raster_2022, raions, "pop_2022")
        out = out.merge(pop22, on="raion_id", how="left")

    if args.population_raster_2023:
        pop23 = zonal_sum_mean(args.population_raster_2023, raions, "pop_2023")
        out = out.merge(pop23, on="raion_id", how="left")

    # Population change and density-style features
    if {"pop_2022_sum", "pop_2023_sum"}.issubset(out.columns):
        out["pop_change_abs_2022_2023"] = out["pop_2023_sum"] - out["pop_2022_sum"]
        out["pop_change_pct_2022_2023"] = (
            out["pop_change_abs_2022_2023"] / out["pop_2022_sum"].replace(0, np.nan)
        )

    if {"pop_2023_sum", "area_sqkm"}.issubset(out.columns):
        out["pop_2023_per_sqkm"] = out["pop_2023_sum"] / out["area_sqkm"].replace(0, np.nan)

    if {"populated_place_count", "pop_2023_sum"}.issubset(out.columns):
        out["places_per_10k_pop_2023"] = (
            10000 * out["populated_place_count"] / out["pop_2023_sum"].replace(0, np.nan)
        )

    # Optional health-facility features
    if args.health_facilities:
        hf = read_health_facilities(args.health_facilities)
        hf_counts = count_points_in_raions(hf, raions, "health_facility_count")
        out = out.merge(hf_counts, on="raion_id", how="left")
        out["health_facility_count"] = out["health_facility_count"].fillna(0).astype(int)

        if {"health_facility_count", "pop_2023_sum"}.issubset(out.columns):
            out["health_facilities_per_100k_pop_2023"] = (
                100000 * out["health_facility_count"] / out["pop_2023_sum"].replace(0, np.nan)
            )

    # Border-crossing density normalized by area
    out["border_crossings_per_1000sqkm"] = (
        1000 * out["border_crossing_count"] / out["area_sqkm"].replace(0, np.nan)
    )

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"Saved exposure features to {args.out_csv}")
    print(f"Rows: {len(out):,}")
    print(f"Columns: {len(out.columns):,}")
    print("Columns:")
    print("\n".join(out.columns.tolist()))


if __name__ == "__main__":
    main()