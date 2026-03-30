#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="AccessWatch judge dashboard", layout="wide")

# -----------------------------
# Constants
# -----------------------------

# Cleaner display names for event/subevent labels that may appear in the files
PRETTY_EVENTS = {
    "battle": "Battle",
    "battles": "Battle",
    "explosions_remote": "Explosions / Remote violence",
    "explosions remote": "Explosions / Remote violence",
    "explosions / remote violence": "Explosions / Remote violence",
    "violence_against_civilians": "Violence against civilians",
    "violence against civilians": "Violence against civilians",
    "air_drone": "Air / drone strike",
    "air drone": "Air / drone strike",
    "air / drone strike": "Air / drone strike",
    "strategic_developments": "Strategic developments",
    "strategic developments": "Strategic developments",
}

# Subevent score columns that can be shown on the map
SUBEVENT_OPTIONS = [
    ("score_battle_any", "Battle"),
    ("score_explosions_remote_any", "Explosions / Remote violence"),
    ("score_air_drone_any", "Air / drone strike"),
    ("score_violence_against_civilians_any", "Violence against civilians"),
    ("score_strategic_developments_any", "Strategic developments"),
]

# Fixed red color ramp for judge-facing risk maps
RED_SCALE = [
    [0.00, "#fff5f0"],
    [0.10, "#fee0d2"],
    [0.25, "#fcbba1"],
    [0.45, "#fc9272"],
    [0.65, "#fb6a4a"],
    [0.82, "#de2d26"],
    [1.00, "#67000d"],
]

# This dashboard is intentionally fixed to the preferred configuration
TARGET_ALGO_PRIORITY = [
    "gru_hierarchical",
    "gru",
]
TARGET_MODEL_SET = "model2"
TARGET_MISSING = "full"


# -----------------------------
# Helpers for ZIP/folder reading
# -----------------------------
class Source:
    """
    Small helper wrapper so the rest of the code can read
    either from a local folder/path or from an uploaded ZIP in memory.
    """
    def __init__(self, path: Optional[str] = None, uploaded_bytes: Optional[bytes] = None):
        self.path = path
        self.uploaded_bytes = uploaded_bytes

    def exists(self) -> bool:
        return self.uploaded_bytes is not None or (self.path is not None and Path(self.path).exists())

    def is_dir(self) -> bool:
        return self.path is not None and Path(self.path).is_dir()


def _list_names(source: Source) -> List[str]:
    """
    List all file names available inside the current source.
    Works for uploaded ZIPs, local folders, or local ZIP files.
    """
    if source.uploaded_bytes is not None:
        with zipfile.ZipFile(io.BytesIO(source.uploaded_bytes)) as zf:
            return zf.namelist()

    if source.is_dir():
        root = Path(source.path)
        return [str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file()]

    with zipfile.ZipFile(source.path) as zf:
        return zf.namelist()


def _discover_single(names: List[str], suffix: str) -> str:
    """
    Find one file by suffix. If multiple matches exist,
    prefer the shortest path as a simple tie-breaker.
    """
    matches = [n for n in names if n.endswith(suffix)]
    if not matches:
        raise KeyError(f"Could not find file ending with {suffix}")
    return sorted(matches, key=len)[0]


def _read_csv(source: Source, inner_path: str) -> pd.DataFrame:
    """
    Read a CSV from either an uploaded ZIP, a local folder,
    or a local ZIP archive.
    """
    if source.uploaded_bytes is not None:
        with zipfile.ZipFile(io.BytesIO(source.uploaded_bytes)) as zf:
            with zf.open(inner_path) as f:
                return pd.read_csv(f)

    if source.is_dir():
        return pd.read_csv(Path(source.path) / inner_path)

    with zipfile.ZipFile(source.path) as zf:
        with zf.open(inner_path) as f:
            return pd.read_csv(f)


def _read_text(source: Source, inner_path: str) -> str:
    """
    Read a text file from the current source.
    Used mainly for JSON manifests.
    """
    if source.uploaded_bytes is not None:
        with zipfile.ZipFile(io.BytesIO(source.uploaded_bytes)) as zf:
            with zf.open(inner_path) as f:
                return f.read().decode("utf-8")

    if source.is_dir():
        return (Path(source.path) / inner_path).read_text(encoding="utf-8")

    with zipfile.ZipFile(source.path) as zf:
        with zf.open(inner_path) as f:
            return f.read().decode("utf-8")


@st.cache_data(show_spinner=False)
def load_results(path: Optional[str], uploaded_bytes: Optional[bytes]):
    """
    Load the main result bundle:
    - compact summary
    - full summary
    - manifest
    - run index describing per-run CSVs
    """
    source = Source(path=path, uploaded_bytes=uploaded_bytes)
    if not source.exists():
        raise FileNotFoundError("Results source not found")

    names = _list_names(source)

    summary_test_path = _discover_single(names, "direct_week_ahead_missing_modality_test.csv")
    summary_all_path = _discover_single(names, "direct_week_ahead_missing_modality_all_splits.csv")
    manifest_path = _discover_single(names, "direct_week_ahead_missing_manifest.json")

    summary_test = _read_csv(source, summary_test_path)
    summary_all = _read_csv(source, summary_all_path)
    manifest = json.loads(_read_text(source, manifest_path))

    # Build an index of all per-run prediction CSVs so we can load
    # the right file later based on week/model/algorithm/split
    rows = []
    for n in names:
        if not n.endswith(".csv"):
            continue

        parts = Path(n).parts
        week_part = next((p for p in parts if p.startswith("week_plus_")), None)
        if week_part is None:
            continue

        toks = Path(n).stem.split("__")
        if len(toks) != 4:
            continue

        model_set, missing_config, algorithm, split = toks
        try:
            week_num = int(week_part.replace("week_plus_", ""))
        except Exception:
            continue

        rows.append({
            "forecast_week_ahead": week_num,
            "forecast_horizon_label": week_part,
            "model_set": model_set,
            "missing_config": missing_config,
            "algorithm": algorithm,
            "split": split,
            "inner_path": n,
        })

    run_index = pd.DataFrame(rows).sort_values(["forecast_week_ahead", "model_set", "missing_config", "algorithm", "split"])
    if run_index.empty:
        raise ValueError("No per-run prediction CSVs found.")

    return summary_test, summary_all, manifest, run_index


@st.cache_data(show_spinner=False)
def load_prediction_frame(path: Optional[str], uploaded_bytes: Optional[bytes], inner_path: str) -> pd.DataFrame:
    """
    Load one prediction CSV and parse the main date columns.
    """
    source = Source(path=path, uploaded_bytes=uploaded_bytes)
    df = _read_csv(source, inner_path)

    for c in ["week_start", "target_window_start", "target_window_end"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_boundary(boundary_path: Optional[str], boundary_bytes: Optional[bytes]):
    """
    Load the Ukraine admin-2 boundary file from either an uploaded ZIP
    or a local path/ZIP.
    """
    if boundary_bytes is None and not boundary_path:
        raise FileNotFoundError("Boundary not provided")

    if boundary_bytes is not None:
        zf = zipfile.ZipFile(io.BytesIO(boundary_bytes))
        inner = None

        # Prefer the expected admin2 file name
        for n in zf.namelist():
            low = n.lower()
            if low.endswith("ukr_admin2.geojson"):
                inner = n
                break

        # Fall back to any admin2-looking geojson
        if inner is None:
            for n in zf.namelist():
                low = n.lower()
                if low.endswith(".geojson") and "admin2" in low:
                    inner = n
                    break

        if inner is None:
            raise FileNotFoundError("Could not find ukr_admin2.geojson inside boundary zip")

        with zf.open(inner) as f:
            data = json.load(f)

        gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")

    else:
        p = Path(boundary_path)
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as zf:
                inner = None

                for n in zf.namelist():
                    low = n.lower()
                    if low.endswith("ukr_admin2.geojson"):
                        inner = n
                        break

                if inner is None:
                    for n in zf.namelist():
                        low = n.lower()
                        if low.endswith(".geojson") and "admin2" in low:
                            inner = n
                            break

                if inner is None:
                    raise FileNotFoundError("Could not find ukr_admin2.geojson inside boundary zip")

            gdf = gpd.read_file(f"zip://{p}!{inner}")
        else:
            gdf = gpd.read_file(p)

    # Standardize the key identifiers used later in merges
    rename = {}
    if "adm2_pcode" in gdf.columns:
        rename["adm2_pcode"] = "raion_id"
    if "adm2_name" in gdf.columns:
        rename["adm2_name"] = "raion_name"
    if "adm1_name" in gdf.columns:
        rename["adm1_name"] = "oblast_name"

    gdf = gdf.rename(columns=rename)

    keep = [c for c in ["raion_id", "raion_name", "oblast_name", "geometry"] if c in gdf.columns]
    gdf = gdf[keep].copy()

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)

    return gdf


# -----------------------------
# Display helpers
# -----------------------------
def clean_event_list(val: object) -> str:
    """
    Clean a stored event list into a nicer judge-facing string.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"

    s = str(val).strip()
    if not s:
        return "—"

    parts = [p.strip() for p in re.split(r"[;|,]", s) if p.strip()]
    if not parts:
        return "—"

    return "; ".join(PRETTY_EVENTS.get(p.lower().strip(), p.replace("_", " ").title()) for p in parts)


def prep_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cleaned display columns for predicted and actual events.
    """
    out = df.copy()

    if "predicted_subtype_list" in out.columns:
        out["Predicted events"] = out["predicted_subtype_list"].apply(clean_event_list)
    else:
        out["Predicted events"] = "—"

    if "actual_subtype_list" in out.columns:
        out["Actual events"] = out["actual_subtype_list"].apply(clean_event_list)
    elif "actual_event_name" in out.columns:
        out["Actual events"] = out["actual_event_name"].apply(clean_event_list)
    else:
        out["Actual events"] = "—"

    return out


def prettify_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cleaner display labels to the summary table.
    """
    out = df.copy()

    out["Forecast week ahead"] = "Week +" + out["forecast_week_ahead"].astype(str)
    out["Model family"] = out["algorithm"].map({
        "linear_hierarchical": "Linear",
        "lightgbm_hierarchical": "LightGBM",
        "catboost_hierarchical": "CatBoost",
        "gru_hierarchical": "GRU",
        "tcn_hierarchical": "TCN",
        "linear": "Linear",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "gru": "GRU",
        "tcn": "TCN",
    }).fillna(out["algorithm"])

    out["Missing setting"] = out["missing_config"].map({
        "full": "Full model",
        "drop_ntl": "Drop night lights",
        "drop_firms": "Drop FIRMS",
        "drop_ntl_firms": "Drop night lights + FIRMS",
        "drop_unosat": "Drop UNOSAT",
    }).fillna(out["missing_config"])

    return out


def choose_fixed_config(summary_test: pd.DataFrame) -> pd.DataFrame:
    """
    Pick one fixed configuration per forecast horizon.
    By design this prefers Model 2 + GRU + full model.
    """
    sub = summary_test[
        (summary_test["model_set"] == TARGET_MODEL_SET)
        & (summary_test["missing_config"] == TARGET_MISSING)
    ].copy()

    picks = []
    for wk, grp in sub.groupby("forecast_week_ahead"):
        chosen = None

        for algo in TARGET_ALGO_PRIORITY:
            hit = grp[grp["algorithm"] == algo]
            if not hit.empty:
                chosen = hit.iloc[0]
                break

        # If GRU is unavailable for that horizon, fall back to the strongest row
        if chosen is None:
            chosen = grp.sort_values(
                ["high_intensity_f1", "subtype_macro_f1", "any_event_f1"],
                ascending=False
            ).iloc[0]

        picks.append(chosen)

    return pd.DataFrame(picks).sort_values("forecast_week_ahead")


def week_range(df: pd.DataFrame) -> str:
    """
    Return the date span covered by the loaded prediction rows.
    """
    vals = df["week_start"].dropna().dt.normalize().unique().tolist()
    vals = sorted(vals)
    if not vals:
        return "No weeks"
    return f"{pd.Timestamp(vals[0]).strftime('%Y-%m-%d')} → {pd.Timestamp(vals[-1]).strftime('%Y-%m-%d')}"


def boundary_view_params(gdf: gpd.GeoDataFrame) -> Tuple[Dict[str, float], float]:
    """
    Choose a reasonable center and zoom level based on the boundary extent.
    """
    minx, miny, maxx, maxy = gdf.total_bounds
    center = {"lon": float((minx + maxx) / 2), "lat": float((miny + maxy) / 2)}
    span = max(float(maxx - minx), float(maxy - miny))

    if span >= 20:
        zoom = 4.15
    elif span >= 14:
        zoom = 4.45
    elif span >= 10:
        zoom = 4.8
    else:
        zoom = 5.2

    return center, zoom


def prep_geojson_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert datetime columns to strings before passing the frame to Plotly.
    """
    out = gdf.copy()
    for col in out.columns:
        if col == "geometry":
            continue
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
            out[col] = out[col].where(out[col].notna(), None)
    return out


def make_map(gdf: gpd.GeoDataFrame, color_col: str, title: str, hover_cols: List[str], zmin: float, zmax: float):
    """
    Build the choropleth map used in the dashboard.
    """
    gplot = prep_geojson_gdf(gdf)
    center, zoom = boundary_view_params(gplot)

    fig = px.choropleth_mapbox(
        gplot,
        geojson=json.loads(gplot.to_json()),
        locations="raion_id",
        featureidkey="properties.raion_id",
        color=color_col,
        color_continuous_scale=RED_SCALE,
        range_color=(zmin, zmax),
        mapbox_style="white-bg",
        center=center,
        zoom=zoom,
        opacity=0.85,
        hover_name="raion_name",
        hover_data={c: True for c in hover_cols if c in gplot.columns},
        title=title,
        height=820,
    )

    fig.update_traces(marker_line_color="#2a2a2a", marker_line_width=0.9)
    fig.update_layout(margin={"l": 0, "r": 0, "t": 55, "b": 0}, coloraxis_colorbar_title="Risk")
    return fig


def metric_trend_chart(df: pd.DataFrame):
    """
    Build the single F1 trend chart shown at the top of the dashboard.
    """
    plot_df = df[["forecast_week_ahead", "any_event_f1", "high_intensity_f1", "subtype_macro_f1"]].copy()

    long_df = plot_df.melt(
        id_vars=["forecast_week_ahead"],
        value_vars=["any_event_f1", "high_intensity_f1", "subtype_macro_f1"],
        var_name="metric",
        value_name="f1_score",
    )

    long_df["metric"] = long_df["metric"].map({
        "any_event_f1": "Any-event F1-score",
        "high_intensity_f1": "High-risk F1-score",
        "subtype_macro_f1": "Subevent macro F1-score",
    })

    fig = px.line(
        long_df,
        x="forecast_week_ahead",
        y="f1_score",
        color="metric",
        markers=True,
        title="F1-score vs forecast week ahead (fixed Model 2 + GRU, full model)",
        labels={"forecast_week_ahead": "Forecast week ahead", "f1_score": "F1-score", "metric": ""},
    )

    fig.update_xaxes(dtick=1)

    ymin = max(0.0, float(long_df["f1_score"].min()) - 0.03)
    ymax = min(1.0, float(long_df["f1_score"].max()) + 0.03)
    fig.update_yaxes(range=[ymin, ymax])
    fig.update_layout(margin={"l": 0, "r": 0, "t": 55, "b": 0}, legend_title_text="")

    return fig


def subtype_explainer(name: str) -> str:
    """
    Short subtype descriptions shown in the sidebar/control area.
    """
    return {
        "Battle": "Direct armed clashes or weapon engagement between organized forces.",
        "Explosions / Remote violence": "Shelling, missiles, artillery, or other remote explosive violence.",
        "Air / drone strike": "Air-delivered or drone-delivered strike activity.",
        "Violence against civilians": "Civilian-targeted violence or attacks against civilians.",
        "Strategic developments": "Non-battle strategic actions such as movement, control change, or military posture shifts.",
    }.get(name, "")


# -----------------------------
# UI
# -----------------------------
st.title("AccessWatch: judge-facing forecast prototype")
st.caption("Fixed to Model 2 + GRU, full model. One clean trend, one hierarchy-correct map, one ranked table.")

with st.sidebar:
    st.header("Inputs")
    uploaded_results = st.file_uploader("Upload result ZIP", type=["zip"], key="results_zip")
    results_path = st.text_input("Or local result ZIP/folder", value="")
    uploaded_boundary = st.file_uploader("Upload Ukraine boundary ZIP", type=["zip"], key="boundary_zip")
    boundary_path = st.text_input("Or local boundary ZIP", value="")

results_bytes = uploaded_results.getvalue() if uploaded_results is not None else None
boundary_bytes = uploaded_boundary.getvalue() if uploaded_boundary is not None else None

if not results_bytes and not results_path:
    st.info("Upload the result ZIP or provide a local path.")
    st.stop()

try:
    summary_test_raw, summary_all_raw, manifest, run_index = load_results(results_path or None, results_bytes)
except Exception as e:
    st.exception(e)
    st.stop()

summary_test = prettify_summary(summary_test_raw)
fixed_cfg = choose_fixed_config(summary_test_raw)

# The trend chart uses the fixed config chosen above
trend_fig = metric_trend_chart(fixed_cfg)
st.plotly_chart(trend_fig, use_container_width=True)

metric_cols = st.columns(3)
metric_cols[0].metric("Fixed model", "Model 2 + GRU")
metric_cols[1].metric("Covered horizons", ", ".join([f"Week +{int(x)}" for x in fixed_cfg["forecast_week_ahead"].tolist()]))
metric_cols[2].metric("Setup", "Full model only")

# Main dashboard controls
left, right = st.columns([1.2, 2.0])

with left:
    forecast_week = st.selectbox(
        "Forecast week ahead",
        fixed_cfg["forecast_week_ahead"].tolist(),
        format_func=lambda x: f"Week +{x}",
    )

    cfg_row = fixed_cfg[fixed_cfg["forecast_week_ahead"] == forecast_week].iloc[0]

    # Load valid + test rows for the selected fixed configuration
    pred_match = run_index[
        (run_index["forecast_week_ahead"] == int(forecast_week))
        & (run_index["model_set"] == cfg_row["model_set"])
        & (run_index["missing_config"] == cfg_row["missing_config"])
        & (run_index["algorithm"] == cfg_row["algorithm"])
        & (run_index["split"].isin(["valid", "test"]))
    ].copy()

    pred_frames = []
    for _, r in pred_match.iterrows():
        part = load_prediction_frame(results_path or None, results_bytes, r["inner_path"])
        part["split"] = r["split"]
        part = prep_predictions(part)
        pred_frames.append(part)

    pred_df = pd.concat(pred_frames, ignore_index=True)
    pred_df["week_start"] = pd.to_datetime(pred_df["week_start"], errors="coerce")

    if "target_window_start" in pred_df.columns:
        pred_df["target_window_start"] = pd.to_datetime(pred_df["target_window_start"], errors="coerce")
    if "target_window_end" in pred_df.columns:
        pred_df["target_window_end"] = pd.to_datetime(pred_df["target_window_end"], errors="coerce")

    available_weeks = sorted(pred_df["week_start"].dropna().dt.normalize().unique().tolist())
    anchor_week = st.selectbox(
        "Anchor week",
        available_weeks,
        format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d"),
    )

    main_view = st.radio(
        "Map view",
        ["High-risk severity", "Subevent"],
        index=0,
    )

    # Hierarchy gate: only show sub-risk when any-event probability is large enough
    any_event_gate = st.slider(
        "Any-event gate",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.05,
        help="Only display high-risk or subevent values where any-event probability is at least this threshold.",
    )

    subevent_choice = None
    if main_view == "Subevent":
        present = [(c, lab) for c, lab in SUBEVENT_OPTIONS if c in pred_df.columns]
        subevent_choice = st.selectbox("Subevent", present, format_func=lambda x: x[1]) if present else None

    st.markdown("**What high-risk severity means**")
    st.markdown(
        "High-risk severity indicates how likely a raion is to face a serious conflict week in the selected future week. "
        "A week is treated as high risk when **fatalities are at least 5, total event count is at least 10, or violence against civilians occurs**."
    )

    if main_view == "Subevent" and subevent_choice is not None:
        st.markdown(f"**{subevent_choice[1]}**")
        st.caption(subtype_explainer(subevent_choice[1]))

with right:
    st.markdown(
        f"**Active configuration:** {PRETTY_EVENTS.get('air_drone','Air / drone strike')} info hidden • "
        f"Model 2 • GRU • Full model • valid+test dates: {week_range(pred_df)}"
    )

# Filter to the selected anchor week
sel = pred_df[pred_df["week_start"].dt.normalize() == pd.Timestamp(anchor_week).normalize()].copy()
if sel.empty:
    st.warning("No rows found for the selected anchor week.")
    st.stop()

# Apply hierarchy gating based on any-event risk
if "score_any_event" in sel.columns:
    gate_mask = pd.to_numeric(sel["score_any_event"], errors="coerce") >= float(any_event_gate)
else:
    gate_mask = pd.Series([True] * len(sel), index=sel.index)

if main_view == "High-risk severity":
    map_col = "score_high_intensity"
    legend_title = "High-risk severity"
    if map_col in sel.columns:
        sel[map_col] = pd.to_numeric(sel[map_col], errors="coerce").where(gate_mask)
else:
    map_col = subevent_choice[0]
    legend_title = subevent_choice[1]
    sel[map_col] = pd.to_numeric(sel[map_col], errors="coerce").where(gate_mask)

# Load the boundary and render the map
try:
    boundary = load_boundary(boundary_path or None, boundary_bytes)
    join_keys = [c for c in ["raion_id", "raion_name", "oblast_name"] if c in sel.columns and c in boundary.columns]
    merged = boundary.merge(sel, on=join_keys, how="left")

    visible = pd.to_numeric(merged[map_col], errors="coerce")
    if visible.notna().any():
        zmin = max(0.0, float(visible.min()))
        zmax = min(1.0, float(visible.max()))
        if math.isclose(zmin, zmax):
            zmin, zmax = 0.0, 1.0
    else:
        zmin, zmax = 0.0, 1.0

    hover_cols = [
        "oblast_name",
        "split",
        "score_any_event",
        "score_high_intensity",
        "Predicted events",
        "Actual events",
        "pred_event_count",
        "actual_event_count",
        "pred_fatalities_sum",
        "actual_fatalities_sum",
    ]

    title = f"{legend_title} • Week +{forecast_week} • {pd.Timestamp(anchor_week).strftime('%Y-%m-%d')}"
    fig = make_map(merged, map_col, title, hover_cols, zmin, zmax)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Displayed color range: {zmin:.3f} to {zmax:.3f}. Areas below the any-event gate are hidden.")

except Exception as e:
    st.warning(f"Boundary/map could not be loaded: {e}")

# Ranked table
st.subheader("Ranked forecast table")

view = sel.copy()
view["Any-event probability"] = pd.to_numeric(view.get("score_any_event"), errors="coerce")
view["High-risk probability"] = pd.to_numeric(view.get("score_high_intensity"), errors="coerce")
view["Predicted event count"] = pd.to_numeric(view.get("pred_event_count"), errors="coerce")
view["Actual event count"] = pd.to_numeric(view.get("actual_event_count"), errors="coerce")

keep = [
    "raion_name",
    "oblast_name",
    "split",
    "week_start",
    "target_window_start",
    "target_window_end",
    "Any-event probability",
    "High-risk probability",
    "Predicted events",
    "Actual events",
    "Predicted event count",
    "Actual event count",
]
keep = [c for c in keep if c in view.columns]

view = view[keep].copy().rename(columns={
    "raion_name": "Raion",
    "oblast_name": "Oblast",
    "split": "Split",
    "week_start": "Anchor week",
    "target_window_start": "Target week start",
    "target_window_end": "Target week end",
})

for c in ["Anchor week", "Target week start", "Target week end"]:
    if c in view.columns:
        view[c] = pd.to_datetime(view[c], errors="coerce").dt.strftime("%Y-%m-%d")

for c in ["Any-event probability", "High-risk probability", "Predicted event count", "Actual event count"]:
    if c in view.columns:
        view[c] = pd.to_numeric(view[c], errors="coerce").round(3)

view = view.sort_values(["High-risk probability", "Any-event probability"], ascending=False)
st.dataframe(view, use_container_width=True, hide_index=True)