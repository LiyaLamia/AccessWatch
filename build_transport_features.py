#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Road classes that we want to treat as the main road network
MAJOR_ROAD_TYPES = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary'}

# Surface types considered paved for a simple paved-road estimate
PAVED_SURFACES = {'asphalt', 'concrete', 'paved', 'paving_stones'}


def extract_member(zip_path: Path, suffix: str) -> Path:
    """
    Find a file inside a zip archive whose name ends with the given suffix,
    extract it, and copy it to a temp path that still exists after return.
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        with zipfile.ZipFile(zip_path) as zf:
            # Ignore MacOS metadata folders and look for the actual target file
            matches = [
                n for n in zf.namelist()
                if n.lower().endswith(suffix.lower()) and '__MACOSX' not in n
            ]
            if not matches:
                raise FileNotFoundError(f'No {suffix} found in {zip_path.name}')

            zf.extract(matches[0], td_path)
            extracted = td_path / matches[0]

            # TemporaryDirectory disappears after this function,
            # so copy the extracted file to a new temp location first.
            final_path = Path(tempfile.mkdtemp()) / Path(matches[0]).name
            final_path.write_bytes(extracted.read_bytes())
            return final_path


def load_admin2(boundary_zip: Path) -> gpd.GeoDataFrame:
    """
    Load the Ukraine admin-2 boundaries and rename the key fields
    into the simpler names used by the rest of the script.
    """
    admin_path = extract_member(boundary_zip, 'ukr_admin2.geojson')
    admin = gpd.read_file(admin_path)

    admin = admin[['adm2_pcode', 'adm2_name', 'adm1_name', 'geometry']].copy()
    admin = admin.rename(columns={
        'adm2_pcode': 'raion_id',
        'adm2_name': 'raion_name',
        'adm1_name': 'oblast_name',
    })

    return admin


def load_gpkg_from_zip(zip_path: Path) -> gpd.GeoDataFrame:
    """
    Extract a GeoPackage from a zip file and load it as a GeoDataFrame.
    """
    gpkg_path = extract_member(zip_path, '.gpkg')
    return gpd.read_file(gpkg_path)


def compute_line_length_features(
    lines: gpd.GeoDataFrame,
    admin: gpd.GeoDataFrame,
    prefix: str
) -> pd.DataFrame:
    """
    Clip line features to raion boundaries and sum total length per raion.
    The prefix controls the output column name.
    """
    # Reproject to a metric CRS so line lengths are measured in meters
    lines = lines.to_crs(3857)
    admin_m = admin.to_crs(3857)

    # Intersect each line with the admin polygons so only the segment
    # inside each raion contributes to that raion's total length
    clipped = gpd.overlay(
        lines[['geometry'] + [c for c in lines.columns if c != 'geometry']],
        admin_m,
        how='intersection'
    )

    clipped[f'{prefix}_length_km'] = clipped.geometry.length / 1000.0

    grouped = (
        clipped.groupby(['raion_id', 'raion_name', 'oblast_name'], dropna=False)
        .agg(**{f'{prefix}_length_km': (f'{prefix}_length_km', 'sum')})
        .reset_index()
    )

    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build static road and rail features by Ukrainian raion.'
    )
    parser.add_argument('--boundary_zip', required=True, type=Path)
    parser.add_argument('--roads_zip', required=True, type=Path)
    parser.add_argument('--rail_zip', required=True, type=Path)
    parser.add_argument('--out_csv', required=True, type=Path)
    args = parser.parse_args()

    # Load raion boundaries plus transport layers
    admin = load_admin2(args.boundary_zip)
    roads = load_gpkg_from_zip(args.roads_zip)
    rail = load_gpkg_from_zip(args.rail_zip)

    # Drop rows with missing geometry before spatial processing
    roads = roads[roads.geometry.notna()].copy()
    rail = rail[rail.geometry.notna()].copy()

    # Keep only the road attributes needed for the feature subsets below
    road_cols = [c for c in ['highway', 'surface', 'bridge'] if c in roads.columns]
    roads_base = roads[road_cols + ['geometry']].copy()

    # Total road length per raion
    all_roads = compute_line_length_features(roads_base, admin, 'road_total')

    # Length of major roads only
    major_roads = compute_line_length_features(
        roads_base[roads_base['highway'].isin(MAJOR_ROAD_TYPES)].copy(),
        admin,
        'road_major'
    )

    # Length of paved roads, if surface information exists
    paved_roads = compute_line_length_features(
        roads_base[roads_base['surface'].fillna('').isin(PAVED_SURFACES)].copy(),
        admin,
        'road_paved'
    ) if 'surface' in roads_base.columns else (
        admin[['raion_id', 'raion_name', 'oblast_name']].copy()
        .assign(road_paved_length_km=0.0)
    )

    # Length of road segments marked as bridges, if that field exists
    bridge_roads = compute_line_length_features(
        roads_base[
            roads_base['bridge'].fillna('').astype(str).str.lower().isin(['yes', 'true', '1'])
        ].copy(),
        admin,
        'road_bridge'
    ) if 'bridge' in roads_base.columns else (
        admin[['raion_id', 'raion_name', 'oblast_name']].copy()
        .assign(road_bridge_length_km=0.0)
    )

    # Total rail length per raion
    rail_base = rail[['railway', 'geometry']].copy() if 'railway' in rail.columns else rail[['geometry']].copy()
    rail_all = compute_line_length_features(rail_base, admin, 'rail_total')

    # Start from the full raion list and merge all feature tables into one output
    out = admin[['raion_id', 'raion_name', 'oblast_name']].drop_duplicates().copy()
    for df in [all_roads, major_roads, paved_roads, bridge_roads, rail_all]:
        out = out.merge(df, on=['raion_id', 'raion_name', 'oblast_name'], how='left')

    # Any missing lengths after the merge mean that raion had no matching features
    length_cols = [c for c in out.columns if c.endswith('_length_km')]
    out[length_cols] = out[length_cols].fillna(0.0)

    # This line is effectively bypassed because of "if False";
    # the real density calculation is done later after area is added.
    out['road_density_km_per_100sqkm'] = (
        out['road_total_length_km'] /
        admin[['raion_id']].merge(
            load_admin2(args.boundary_zip)[['raion_id']].assign(dummy=1),
            on='raion_id',
            how='left'
        )
    ) if False else out['road_total_length_km']

    # Load area values from the original admin file so density features can be computed
    admin_area = load_admin2(args.boundary_zip)
    admin_area_src = extract_member(args.boundary_zip, 'ukr_admin2.geojson')
    admin_full = gpd.read_file(admin_area_src)[['adm2_pcode', 'area_sqkm']].rename(
        columns={'adm2_pcode': 'raion_id'}
    )

    out = out.merge(admin_full, on='raion_id', how='left')

    # Normalize transport lengths by area to make raions more comparable
    out['road_density_km_per_100sqkm'] = out['road_total_length_km'] / (out['area_sqkm'] / 100.0)
    out['major_road_density_km_per_100sqkm'] = out['road_major_length_km'] / (out['area_sqkm'] / 100.0)
    out['rail_density_km_per_100sqkm'] = out['rail_total_length_km'] / (out['area_sqkm'] / 100.0)

    # Ratio-style features for road composition and rail-vs-road balance
    out['major_road_share'] = out['road_major_length_km'] / out['road_total_length_km'].replace(0, pd.NA)
    out['paved_road_share'] = out['road_paved_length_km'] / out['road_total_length_km'].replace(0, pd.NA)
    out['rail_to_road_ratio'] = out['rail_total_length_km'] / out['road_total_length_km'].replace(0, pd.NA)

    # Replace undefined ratios from divide-by-zero cases with 0
    ratio_cols = ['major_road_share', 'paved_road_share', 'rail_to_road_ratio']
    out[ratio_cols] = out[ratio_cols].fillna(0.0)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f'Saved transport features: {args.out_csv}')
    print(f'Rows: {len(out):,}')


if __name__ == '__main__':
    main()