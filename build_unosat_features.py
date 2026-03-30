#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd

# Date patterns used when the event date is not stored cleanly in attributes
# and has to be guessed from a layer name or package name.
DATE_PATTERNS = [
    re.compile(r"(20\d{2})(\d{2})(\d{2})"),
    re.compile(r"(20\d{2})[-_](\d{2})[-_](\d{2})"),
]

# Some UNOSAT layers are AOI/extent/helper layers rather than actual features,
# so those should be skipped.
IGNORE_LAYER_PATTERNS = [
    "analysisextent",
    "analysedurbanarea",
    "analyzedurbanarea",
    "analysis_extent",
    "aoi",
]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for building raion-level UNOSAT
    disruption features from zipped shapefile or geodatabase packages.
    """
    p = argparse.ArgumentParser(
        description="Build raion-level UNOSAT disruption features from zipped SHP/GDB packages."
    )
    p.add_argument("--boundary_zip", required=True, help="Path to ukr_admin_boundaries.geojson.zip or extracted admin2 geojson")
    p.add_argument("--unosat_files", nargs="+", required=True, help="One or more UNOSAT .zip packages (.shp.zip or .gdb.zip)")
    p.add_argument("--out_csv", required=True, help="Output raion-level static features CSV")
    p.add_argument("--out_records_csv", default=None, help="Optional output CSV of normalized per-raion records")
    p.add_argument("--admin_id_col", default="adm2_pcode")
    p.add_argument("--admin_name_col", default="adm2_name")
    p.add_argument("--cutoff_date", default="2024-01-01", help="Only include features with event_date < cutoff_date")
    return p.parse_args()


def resolve_boundary_path(boundary_path: str) -> str:
    """
    Resolve the boundary input into a path geopandas can read.
    If the boundary file is zipped, point directly to the admin2 GeoJSON inside it.
    """
    p = Path(boundary_path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as zf:
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
    Read the admin boundary file and standardize the key raion columns
    used later in the aggregation.
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

    if "oblast_name" not in raions.columns:
        raions["oblast_name"] = None

    if raions.crs is None:
        raions = raions.set_crs(4326)
    else:
        raions = raions.to_crs(4326)

    return raions


def should_ignore_layer(name: str) -> bool:
    """
    Decide whether a layer looks like a non-feature helper layer
    such as an AOI or analysis extent.
    """
    s = name.lower().replace(" ", "").replace("-", "").replace("_", "")
    return any(pat.replace("_", "") in s for pat in IGNORE_LAYER_PATTERNS)


def classify_layer(name: str) -> tuple[str, str]:
    """
    Assign a coarse source group and feature kind based on the layer name.
    This is a heuristic, but it helps organize the very mixed UNOSAT inputs.
    """
    s = name.lower()
    source_group = "other"
    feature_kind = "other"

    if any(k in s for k in ["floodextent", "waterextent", "flood", "waterextentoutsideriverbed"]):
        source_group = "flood"
        feature_kind = "flood"
    elif any(k in s for k in ["affectedurbanarea", "affected road", "affectedroad", "damageddam", "damage", "da_", "_da", "cda", "rda"]):
        source_group = "damage"
        if "affectedroad" in s or "road" in s:
            feature_kind = "affected_road"
        elif "affectedurbanarea" in s or "urban" in s:
            feature_kind = "affected_urban"
        elif "point" in s or "damage_point" in s:
            feature_kind = "damage_point"
        else:
            feature_kind = "damage"

    return source_group, feature_kind


def parse_event_date(gdf: gpd.GeoDataFrame, layer_name: str, package_name: str) -> pd.Timestamp | pd.NaT:
    """
    Try to recover an event date either from known date-like attributes
    or, if necessary, from the layer/package name.
    """
    candidates = [
        "SensorDate", "Sensor_Dat", "Date", "ACQ_DATE", "AcqDate", "event_date", "EventDate"
    ]

    for c in candidates:
        if c in gdf.columns:
            vals = pd.to_datetime(gdf[c], errors="coerce")
            if vals.notna().any():
                return vals.dropna().iloc[0]

    search_text = f"{layer_name}_{package_name}"
    for pat in DATE_PATTERNS:
        m = pat.search(search_text)
        if m:
            try:
                return pd.Timestamp(
                    year=int(m.group(1)),
                    month=int(m.group(2)),
                    day=int(m.group(3))
                )
            except Exception:
                pass

    return pd.NaT


def normalize_geometry_type(geom_type: str) -> str:
    """
    Reduce detailed geometry types into a few broad classes
    used later in the aggregation logic.
    """
    if geom_type is None:
        return "unknown"

    g = geom_type.lower()
    if "point" in g:
        return "point"
    if "line" in g:
        return "line"
    if "polygon" in g:
        return "polygon"
    return g


def read_shapefiles_from_zip(zip_path: str) -> list[tuple[gpd.GeoDataFrame, str, str, str]]:
    """
    Read all usable shapefiles inside one zip package.
    Each returned item includes the GeoDataFrame and a bit of source metadata.
    """
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        shps = [
            n for n in zf.namelist()
            if n.lower().endswith(".shp")
            and "__macosx" not in n.lower()
            and not os.path.basename(n).startswith("._")
        ]

    for shp in shps:
        layer_name = Path(shp).stem
        if should_ignore_layer(layer_name):
            continue

        gdf = gpd.read_file(f"zip://{zip_path}!{shp}")
        if gdf.empty:
            continue

        out.append((gdf, layer_name, Path(zip_path).name, zip_path))

    return out


def read_gdb_from_zip(zip_path: str, temp_root: str) -> list[tuple[gpd.GeoDataFrame, str, str, str]]:
    """
    Extract a zipped file geodatabase, read all usable layers,
    and return them with source metadata.
    """
    out = []
    workdir = Path(temp_root) / Path(zip_path).stem

    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(workdir)

    gdbs = list(workdir.rglob("*.gdb"))
    for gdb in gdbs:
        try:
            layers = fiona.listlayers(gdb)
        except Exception:
            continue

        for layer in layers:
            if should_ignore_layer(layer):
                continue
            try:
                gdf = gpd.read_file(gdb, layer=layer)
            except Exception:
                continue
            if gdf.empty:
                continue
            out.append((gdf, layer, Path(zip_path).name, zip_path))

    return out


def load_unosat_sources(paths: list[str]) -> list[tuple[gpd.GeoDataFrame, str, str, str]]:
    """
    Load all supported UNOSAT inputs, whether they are zipped shapefiles,
    zipped geodatabases, or already-extracted vector files.
    """
    all_layers = []
    temp_root = tempfile.mkdtemp(prefix="unosat_")

    try:
        for path in paths:
            p = Path(path)

            if p.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as zf:
                    names = zf.namelist()
                    has_shp = any(
                        n.lower().endswith(".shp")
                        and "__macosx" not in n.lower()
                        and not os.path.basename(n).startswith("._")
                        for n in names
                    )
                    has_gdb = any(".gdb/" in n.lower() or n.lower().endswith('.gdb') for n in names)

                if has_shp:
                    all_layers.extend(read_shapefiles_from_zip(path))
                elif has_gdb:
                    all_layers.extend(read_gdb_from_zip(path, temp_root))

            elif p.suffix.lower() in [".shp", ".gpkg", ".geojson", ".json"]:
                gdf = gpd.read_file(path)
                all_layers.append((gdf, p.stem, p.name, path))

            elif p.suffix.lower() == ".gdb":
                for layer in fiona.listlayers(path):
                    if should_ignore_layer(layer):
                        continue
                    gdf = gpd.read_file(path, layer=layer)
                    all_layers.append((gdf, layer, p.name, path))

        return all_layers

    finally:
        # Intentionally not cleaning temp_root here so extracted geodatabases
        # stay available during the current run.
        pass


def normalize_layers(layer_tuples: list[tuple[gpd.GeoDataFrame, str, str, str]], cutoff_date: pd.Timestamp) -> list[gpd.GeoDataFrame]:
    """
    Standardize all loaded layers into a common schema and attach
    metadata like source group, feature kind, geometry kind, and event date.
    """
    normalized = []

    for gdf, layer_name, package_name, source_path in layer_tuples:
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        else:
            gdf = gdf.to_crs(4326)

        event_date = parse_event_date(gdf, layer_name, package_name)
        if pd.notna(event_date) and event_date >= cutoff_date:
            continue

        source_group, feature_kind = classify_layer(layer_name)
        geom_kind = normalize_geometry_type(gdf.geom_type.iloc[0] if len(gdf) else None)

        use = gdf[["geometry"]].copy()
        use["source_layer"] = layer_name
        use["source_package"] = package_name
        use["source_path"] = source_path
        use["source_group"] = source_group
        use["feature_kind"] = feature_kind
        use["geom_kind"] = geom_kind
        use["event_date"] = event_date

        # Keep a few extra attributes when they exist, since they may be useful later
        for c in ["Main_Damag", "Damage_Sta", "Confidence", "Settlement", "Area_m2", "Area_ha", "EventCode", "Notes"]:
            if c in gdf.columns:
                use[c] = gdf[c]

        normalized.append(use)

    return normalized


def aggregate_to_raions(norm_layers: list[gpd.GeoDataFrame], raions: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign normalized UNOSAT features to raions and aggregate them
    into one static raion-level feature table.
    """
    records = []
    raions_6933 = raions.to_crs(6933)

    for gdf in norm_layers:
        if gdf.empty:
            continue

        geom_kind = gdf["geom_kind"].iloc[0]
        source_group = gdf["source_group"].iloc[0]
        feature_kind = gdf["feature_kind"].iloc[0]
        layer_name = gdf["source_layer"].iloc[0]
        package_name = gdf["source_package"].iloc[0]
        event_date = gdf["event_date"].iloc[0]

        if geom_kind == "point":
            # For points, just count how many fall inside each raion
            joined = gpd.sjoin(
                gdf,
                raions[["raion_id", "raion_name", "oblast_name", "geometry"]],
                how="inner",
                predicate="within"
            )
            if joined.empty:
                continue

            for (rid, rname, oname), grp in joined.groupby(["raion_id", "raion_name", "oblast_name"], dropna=False):
                rec = {
                    "raion_id": rid,
                    "raion_name": rname,
                    "oblast_name": oname,
                    "source_group": source_group,
                    "feature_kind": feature_kind,
                    "geom_kind": geom_kind,
                    "source_layer": layer_name,
                    "source_package": package_name,
                    "event_date": event_date,
                    "feature_count": int(len(grp)),
                    "feature_area_km2": 0.0,
                }
                records.append(rec)

        elif geom_kind in {"polygon", "line"}:
            # For polygons/lines, try intersection overlay so area can be attributed properly
            work = gdf.copy()
            work = work[work.geometry.notna() & ~work.geometry.is_empty].copy()
            if work.empty:
                continue

            work = work.to_crs(6933)

            try:
                inter = gpd.overlay(
                    work,
                    raions_6933[["raion_id", "raion_name", "oblast_name", "geometry"]],
                    how="intersection"
                )
            except Exception:
                # If overlay fails, fall back to a simpler centroid-based join
                cent = work.copy()
                cent["geometry"] = cent.geometry.centroid
                cent = cent.to_crs(4326)

                joined = gpd.sjoin(
                    cent,
                    raions[["raion_id", "raion_name", "oblast_name", "geometry"]],
                    how="inner",
                    predicate="within"
                )
                if joined.empty:
                    continue

                for (rid, rname, oname), grp in joined.groupby(["raion_id", "raion_name", "oblast_name"], dropna=False):
                    records.append({
                        "raion_id": rid,
                        "raion_name": rname,
                        "oblast_name": oname,
                        "source_group": source_group,
                        "feature_kind": feature_kind,
                        "geom_kind": geom_kind,
                        "source_layer": layer_name,
                        "source_package": package_name,
                        "event_date": event_date,
                        "feature_count": int(len(grp)),
                        "feature_area_km2": 0.0,
                    })
                continue

            if inter.empty:
                continue

            inter["part_area_km2"] = inter.geometry.area / 1_000_000.0

            for (rid, rname, oname), grp in inter.groupby(["raion_id", "raion_name", "oblast_name"], dropna=False):
                records.append({
                    "raion_id": rid,
                    "raion_name": rname,
                    "oblast_name": oname,
                    "source_group": source_group,
                    "feature_kind": feature_kind,
                    "geom_kind": geom_kind,
                    "source_layer": layer_name,
                    "source_package": package_name,
                    "event_date": event_date,
                    "feature_count": int(len(grp)),
                    "feature_area_km2": float(grp["part_area_km2"].sum()),
                })

    rec_df = pd.DataFrame(records)

    if rec_df.empty:
        # Return a zero-filled feature table if nothing usable was found
        meta = raions[["raion_id", "raion_name", "oblast_name"]].drop_duplicates().copy()
        empty = meta.copy()
        for c in [
            "unosat_feature_count_pre2024", "unosat_damage_point_count_pre2024", "unosat_damage_polygon_count_pre2024",
            "unosat_damage_area_km2_pre2024", "unosat_flood_feature_count_pre2024", "unosat_flood_area_km2_pre2024",
            "unosat_affected_road_count_pre2024", "unosat_affected_urban_area_km2_pre2024",
            "unosat_feature_count_2022", "unosat_feature_count_2023", "unosat_unique_source_layers"
        ]:
            empty[c] = 0
        return empty, rec_df

    rec_df["event_year"] = pd.to_datetime(rec_df["event_date"], errors="coerce").dt.year

    meta = raions[["raion_id", "raion_name", "oblast_name"]].drop_duplicates().copy()

    def grp_sum(mask, count_col="feature_count"):
        sub = rec_df[mask].groupby("raion_id", as_index=False)[count_col].sum()
        return sub

    def grp_area(mask):
        sub = rec_df[mask].groupby("raion_id", as_index=False)["feature_area_km2"].sum()
        return sub

    out = meta.copy()

    # Each piece below becomes one aggregated raion-level feature
    pieces = [
        (grp_sum(rec_df.index == rec_df.index, "feature_count"), "feature_count", "unosat_feature_count_pre2024"),
        (grp_sum((rec_df["source_group"] == "damage") & (rec_df["geom_kind"] == "point"), "feature_count"), "feature_count", "unosat_damage_point_count_pre2024"),
        (grp_sum((rec_df["source_group"] == "damage") & (rec_df["geom_kind"] == "polygon"), "feature_count"), "feature_count", "unosat_damage_polygon_count_pre2024"),
        (grp_area((rec_df["source_group"] == "damage") & (rec_df["geom_kind"] == "polygon")), "feature_area_km2", "unosat_damage_area_km2_pre2024"),
        (grp_sum(rec_df["source_group"] == "flood", "feature_count"), "feature_count", "unosat_flood_feature_count_pre2024"),
        (grp_area(rec_df["source_group"] == "flood"), "feature_area_km2", "unosat_flood_area_km2_pre2024"),
        (grp_sum(rec_df["feature_kind"] == "affected_road", "feature_count"), "feature_count", "unosat_affected_road_count_pre2024"),
        (grp_area(rec_df["feature_kind"] == "affected_urban"), "feature_area_km2", "unosat_affected_urban_area_km2_pre2024"),
        (grp_sum(rec_df["event_year"] == 2022, "feature_count"), "feature_count", "unosat_feature_count_2022"),
        (grp_sum(rec_df["event_year"] == 2023, "feature_count"), "feature_count", "unosat_feature_count_2023"),
    ]

    for df_piece, old_col, new_col in pieces:
        out = out.merge(df_piece.rename(columns={old_col: new_col}), on="raion_id", how="left")

    unique_layers = (
        rec_df.groupby("raion_id", as_index=False)["source_layer"]
        .nunique()
        .rename(columns={"source_layer": "unosat_unique_source_layers"})
    )
    out = out.merge(unique_layers, on="raion_id", how="left")

    for c in out.columns:
        if c not in ["raion_id", "raion_name", "oblast_name"]:
            out[c] = out[c].fillna(0)

    return out, rec_df


def main() -> None:
    args = parse_args()
    cutoff = pd.Timestamp(args.cutoff_date)

    # Load boundaries, read UNOSAT source packages, normalize them,
    # then aggregate everything to the raion level
    raions = read_boundaries(args.boundary_zip, args.admin_id_col, args.admin_name_col)
    layer_tuples = load_unosat_sources(args.unosat_files)
    normalized = normalize_layers(layer_tuples, cutoff)
    features, records = aggregate_to_raions(normalized, raions)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out_csv, index=False)

    print(f"Saved UNOSAT features to: {args.out_csv}")
    print(f"Rows: {len(features):,}")

    if not records.empty:
        print(f"Normalized records: {len(records):,}")
        print(
            records[['source_package', 'source_layer', 'source_group', 'feature_kind']]
            .drop_duplicates()
            .to_string(index=False)
        )

    if args.out_records_csv:
        records.to_csv(args.out_records_csv, index=False)
        print(f"Saved normalized UNOSAT records to: {args.out_records_csv}")


if __name__ == "__main__":
    main()