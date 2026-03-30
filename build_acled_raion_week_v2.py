#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd


def find_admin2_geojson(zip_path: Path) -> Path:
    """
    Find the admin-2 GeoJSON inside the boundary zip and copy it
    to a temp location that will still exist after this function returns.
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        with zipfile.ZipFile(zip_path) as zf:
            # Look for the specific Ukraine admin-2 boundary file
            matches = [n for n in zf.namelist() if n.endswith('ukr_admin2.geojson')]
            if not matches:
                raise FileNotFoundError(
                    'Could not find ukr_admin2.geojson inside boundary zip.'
                )

            # Extract just the matching file
            zf.extract(matches[0], td_path)
            extracted = td_path / matches[0]

            # TemporaryDirectory gets deleted as soon as we leave this block,
            # so copy the file into another temp folder we can safely return.
            final_path = Path(tempfile.mkdtemp()) / 'ukr_admin2.geojson'
            final_path.write_bytes(extracted.read_bytes())

            return final_path


def load_admin2(boundary_zip: Path) -> gpd.GeoDataFrame:
    """
    Load Ukraine admin-2 boundaries and keep only the columns
    needed for later spatial joins and aggregation.
    """
    admin2_geojson = find_admin2_geojson(boundary_zip)
    admin = gpd.read_file(admin2_geojson)

    keep = ['adm2_pcode', 'adm2_name', 'adm1_name', 'area_sqkm', 'geometry']
    admin = admin[keep].copy()

    # Rename columns to simpler names used throughout the pipeline
    admin = admin.rename(columns={
        'adm2_pcode': 'raion_id',
        'adm2_name': 'raion_name',
        'adm1_name': 'oblast_name',
    })

    return admin


def load_acled(acled_csv: Path) -> gpd.GeoDataFrame:
    """
    Read the ACLED event file, clean the basic fields we need,
    and convert it into a GeoDataFrame using longitude/latitude.
    """
    usecols = [
        'event_id_cnty', 'event_date', 'year', 'disorder_type', 'event_type',
        'sub_event_type', 'actor1', 'actor2', 'interaction', 'civilian_targeting',
        'admin1', 'admin2', 'admin3', 'location', 'latitude', 'longitude',
        'geo_precision', 'fatalities', 'source_scale', 'notes'
    ]

    df = pd.read_csv(acled_csv, usecols=usecols, low_memory=False)

    # Parse event dates and drop rows that cannot be mapped in space or time
    df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
    df = df.dropna(subset=['event_date', 'latitude', 'longitude']).copy()

    # Turn the tabular ACLED data into point geometry
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
        crs='EPSG:4326',
    )

    return gdf


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add week and month columns so events can be aggregated consistently over time.
    """
    df = df.copy()

    # Use Monday as the start of the week for stable weekly indexing
    df['week_start'] = df['event_date'] - pd.to_timedelta(df['event_date'].dt.weekday, unit='D')
    df['month'] = df['event_date'].dt.to_period('M').astype(str)

    return df


def aggregate_acled(joined: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Aggregate event-level ACLED data into raion-week features.
    """
    df = add_time_columns(joined)

    # Make sure fatalities is numeric before building derived features
    df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0)

    # A few binary indicators that will later be summed per raion-week
    df['has_fatalities'] = (df['fatalities'] > 0).astype(int)
    df['vac_event'] = df['civilian_targeting'].fillna('').astype(str).str.strip().ne('').astype(int)
    df['explosions_remote'] = (df['event_type'] == 'Explosions/Remote violence').astype(int)
    df['battles'] = (df['event_type'] == 'Battles').astype(int)
    df['strategic_dev'] = (df['event_type'] == 'Strategic developments').astype(int)
    df['protests_riots'] = df['event_type'].isin(['Protests', 'Riots']).astype(int)
    df['violence_against_civilians'] = (df['event_type'] == 'Violence against civilians').astype(int)

    # Count both direct air/drone strikes and related shelling/missile attacks
    df['air_drone_strike'] = df['sub_event_type'].fillna('').isin([
        'Air/drone strike',
        'Shelling/artillery/missile attack'
    ]).astype(int)

    # Treat geo_precision 1 or 2 as relatively precise event locations
    df['precise_geo'] = (
        pd.to_numeric(df['geo_precision'], errors='coerce').fillna(99) <= 2
    ).astype(int)

    grp_cols = ['raion_id', 'raion_name', 'oblast_name', 'week_start']

    out = (
        df.groupby(grp_cols, dropna=False)
        .agg(
            acled_event_count=('event_id_cnty', 'count'),
            fatalities_sum=('fatalities', 'sum'),
            events_with_fatalities=('has_fatalities', 'sum'),
            violence_against_civilians_count=('violence_against_civilians', 'sum'),
            explosions_remote_count=('explosions_remote', 'sum'),
            battles_count=('battles', 'sum'),
            strategic_developments_count=('strategic_dev', 'sum'),
            protests_riots_count=('protests_riots', 'sum'),
            civilian_targeting_count=('vac_event', 'sum'),
            air_drone_strike_count=('air_drone_strike', 'sum'),
            precise_geo_event_count=('precise_geo', 'sum'),
        )
        .reset_index()
    )

    out['week_start'] = pd.to_datetime(out['week_start'])
    return out


def build_full_panel(agg: pd.DataFrame, admin: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Build a complete raion-week panel so that even weeks with no events
    are still present in the output.
    """
    min_week = agg['week_start'].min()
    max_week = agg['week_start'].max()
    all_weeks = pd.date_range(min_week, max_week, freq='W-MON')

    # Start with the full list of raions
    base = admin[['raion_id', 'raion_name', 'oblast_name']].drop_duplicates().copy()

    # Cross join raions with all weeks to create the full panel
    panel = (
        base.assign(_key=1)
        .merge(pd.DataFrame({'week_start': all_weeks, '_key': 1}), on='_key')
        .drop(columns='_key')
    )

    # Merge in the aggregated ACLED features
    merged = panel.merge(
        agg,
        on=['raion_id', 'raion_name', 'oblast_name', 'week_start'],
        how='left'
    )

    # Any missing aggregated values mean there were no events that week
    fill_zero_cols = [c for c in merged.columns if c.endswith('_count') or c.endswith('_sum')]
    fill_zero_cols += ['events_with_fatalities']

    for c in fill_zero_cols:
        merged[c] = merged[c].fillna(0)

    # Helpful binary targets/features for downstream modeling
    merged['any_event'] = (merged['acled_event_count'] > 0).astype(int)

    merged['high_intensity_week'] = (
        (merged['fatalities_sum'] >= 5) |
        (merged['acled_event_count'] >= 10) |
        (merged['violence_against_civilians_count'] >= 1)
    ).astype(int)

    merged = merged.sort_values(['raion_id', 'week_start']).reset_index(drop=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build raion-week ACLED panel for Ukraine.'
    )
    parser.add_argument('--acled_csv', required=True, type=Path)
    parser.add_argument('--boundary_zip', required=True, type=Path)
    parser.add_argument('--out_csv', required=True, type=Path)
    parser.add_argument(
        '--out_events_csv',
        type=Path,
        default=None,
        help='Optional event-level joined ACLED CSV with assigned raion.'
    )
    args = parser.parse_args()

    # Load spatial boundaries and ACLED point events
    admin = load_admin2(args.boundary_zip)
    acled = load_acled(args.acled_csv)

    # First try a strict point-in-polygon spatial join
    joined = gpd.sjoin(acled, admin, how='left', predicate='within')

    # Some events may fall exactly on borders or slightly outside polygons.
    # For those, fall back to the nearest admin area.
    missing = joined['raion_id'].isna()
    if missing.any():
        nearest = gpd.sjoin_nearest(
            acled.loc[missing],
            admin,
            how='left',
            distance_col='dist_deg'
        )
        for col in ['raion_id', 'raion_name', 'oblast_name', 'area_sqkm']:
            joined.loc[missing, col] = nearest[col].values

    # Build the weekly aggregated panel
    agg = aggregate_acled(joined)
    panel = build_full_panel(agg, admin)

    # Save the final raion-week table
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out_csv, index=False)

    # Optionally also save the event-level file after spatial assignment
    if args.out_events_csv is not None:
        event_cols = [
            'event_id_cnty', 'event_date', 'year', 'disorder_type', 'event_type',
            'sub_event_type', 'admin1', 'admin2', 'admin3', 'location',
            'latitude', 'longitude', 'geo_precision', 'fatalities',
            'raion_id', 'raion_name', 'oblast_name'
        ]
        joined[event_cols].to_csv(args.out_events_csv, index=False)

    # Simple summary so it is easy to confirm the script ran as expected
    print(f'Saved raion-week panel: {args.out_csv}')
    print(f'Rows: {len(panel):,}')
    print(f'Raions: {panel["raion_id"].nunique():,}')
    print(f'Weeks: {panel["week_start"].nunique():,}')


if __name__ == '__main__':
    main()