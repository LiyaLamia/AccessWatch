#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Ukraine forecast map (non-ACLED preferred)", layout="wide")

# Friendly labels for the forecast horizons used in the app
WINDOW_LABELS = {
    "next_week": "Next week",
    "next_2_weeks": "Next 2 weeks",
    "next_3_weeks": "Next 3 weeks",
    "next_4_weeks_month": "Next 4 weeks",
}
WINDOW_ORDER = list(WINDOW_LABELS.keys())

# Preferred model/algo order for auto-selection.
# The app will use the first one it finds for the chosen horizon.
ALGO_PREFERENCE = [
    ("model1_non_acled_only", "gru_hierarchical"),
    ("model1_non_acled_only", "gru"),
    ("model1_non_acled_only", "catboost_hierarchical"),
    ("model1_non_acled_only", "catboost"),
    ("model1_non_acled_only", "lightgbm_hierarchical"),
    ("model1_non_acled_only", "lightgbm"),
    ("model1_non_acled_only", "linear_hierarchical"),
    ("model1_non_acled_only", "linear"),
]

# Pretty display names for event/subtype strings that may show up in outputs
PRETTY_EVENT = {
    "battle": "Battle",
    "battles": "Battle",
    "explosions_remote": "Explosions / Remote violence",
    "explosions remote": "Explosions / Remote violence",
    "explosions / remote violence": "Explosions / Remote violence",
    "violence_against_civilians": "Violence against civilians",
    "violence against civilians": "Violence against civilians",
    "air_drone": "Air / drone strike",
    "air drone": "Air / drone strike",
    "air/drone": "Air / drone strike",
    "air / drone strike": "Air / drone strike",
    "strategic_developments": "Strategic developments",
    "strategic developments": "Strategic developments",
    "protests_riots": "Protests / Riots",
    "protests riots": "Protests / Riots",
}

# Prettier column labels for the priority table
PRETTY_COLUMNS = {
    "raion_name": "Raion",
    "oblast_name": "Oblast",
    "week_start": "Anchor week",
    "target_window_label": "Target window",
    "score_operational_risk": "High-risk severity score",
    "score_any_event": "Any-event risk",
    "predicted_subtype_list": "Predicted events",
    "actual_subtype_list": "Actual events",
    "pred_any_event": "Predicted any event",
    "actual_any_event": "Observed any event",
    "pred_high_intensity": "Predicted high risk",
    "actual_high_intensity": "Observed high risk",
    "pred_event_count": "Predicted event count",
    "actual_event_count": "Observed event count",
    "pred_fatalities_sum": "Predicted fatalities",
    "actual_fatalities_sum": "Observed fatalities",
    "pred_air_drone_strike_count": "Predicted air/drone count",
    "actual_air_drone_strike_count": "Observed air/drone count",
}


@st.cache_data(show_spinner=False)
def load_boundary(boundary_path: str) -> gpd.GeoDataFrame:
    """
    Load the Ukraine admin-2 boundary file from either a direct vector file
    or a zip archive containing the GeoJSON.
    """
    p = Path(boundary_path)
    if not p.exists():
        raise FileNotFoundError(f"Boundary path not found: {boundary_path}")

    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as zf:
            # Prefer the expected admin2 file name, but fall back if needed
            candidates = [n for n in zf.namelist() if n.lower().endswith("ukr_admin2.geojson")]
            if not candidates:
                candidates = [n for n in zf.namelist() if n.lower().endswith(".geojson") and "admin2" in n.lower()]
            if not candidates:
                raise FileNotFoundError("Could not find ukr_admin2.geojson inside boundary zip")
            inner = candidates[0]
        gdf = gpd.read_file(f"zip://{p}!{inner}")
    else:
        gdf = gpd.read_file(p)

    keep = [c for c in ["adm2_pcode", "adm2_name", "adm1_name", "geometry"] if c in gdf.columns]
    gdf = gdf[keep].copy().rename(
        columns={
            "adm2_pcode": "raion_id",
            "adm2_name": "raion_name",
            "adm1_name": "oblast_name",
        }
    )

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)

    return gdf


def _read_csv_any(source: Path, inner_path: str) -> pd.DataFrame:
    """
    Read a CSV either from a directory tree or from inside a zip archive.
    """
    if source.is_dir():
        return pd.read_csv(source / inner_path)

    with zipfile.ZipFile(source) as zf:
        with zf.open(inner_path) as f:
            return pd.read_csv(f)


def _list_csvs(source: Path) -> List[str]:
    """
    List all CSV paths available under a directory or inside a zip archive.
    """
    if source.is_dir():
        return [str(p.relative_to(source)).replace("\\", "/") for p in source.rglob("*.csv")]

    with zipfile.ZipFile(source) as zf:
        return [n for n in zf.namelist() if n.lower().endswith(".csv")]


def _detect_prefix(names: List[str]) -> str:
    """
    Try to detect the common prefix before the horizon folders.
    This makes the loader work whether the files are stored directly under
    next_week/... or inside an extra parent folder.
    """
    for pref in ["hierarchical_multiwindow_exports/", "", "data/outputs/hierarchical_multiwindow_exports/"]:
        if pref == "":
            if any(
                re.match(
                    r"(next_week|next_2_weeks|next_3_weeks|next_4_weeks_month)/model.+__.+__(train|valid|test)\.csv$",
                    n,
                )
                for n in names
            ):
                return ""
        elif any(n.startswith(pref) for n in names):
            return pref

    # Fallback: infer the prefix from the first matching path
    for n in names:
        m = re.match(
            r"(.*?)(next_week|next_2_weeks|next_3_weeks|next_4_weeks_month)/model.+__.+__(train|valid|test)\.csv$",
            n,
        )
        if m:
            return m.group(1)

    return ""


@st.cache_data(show_spinner=False)
def discover_runs(exports_path: str) -> pd.DataFrame:
    """
    Scan the forecast output folder/zip and build an index
    of available horizon/model/algo/split CSVs.
    """
    source = Path(exports_path)
    names = _list_csvs(source)
    prefix = _detect_prefix(names)

    pat = re.compile(
        rf"^{re.escape(prefix)}(?P<window>next_week|next_2_weeks|next_3_weeks|next_4_weeks_month)/(?P<model>model[12]_[^/]+?)__(?P<algo>[^/]+?)__(?P<split>train|valid|test)\.csv$"
    )

    rows = []
    for n in names:
        m = pat.match(n)
        if m:
            rows.append(
                {
                    "window": m.group("window"),
                    "model_key": m.group("model"),
                    "algo_key": m.group("algo"),
                    "split": m.group("split"),
                    "inner_path": (
                        f"{prefix}{m.group('window')}/{m.group('model')}__{m.group('algo')}__{m.group('split')}.csv"
                        if prefix
                        else f"{m.group('window')}/{m.group('model')}__{m.group('algo')}__{m.group('split')}.csv"
                    ),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No forecast CSVs found in the provided folder or zip.")

    return out.sort_values(["window", "model_key", "algo_key", "split"]).reset_index(drop=True)


def pick_best_model(run_index: pd.DataFrame, horizon: str) -> Optional[tuple[str, str]]:
    """
    Pick the preferred model/algo pair for a given horizon.
    """
    sub = run_index[run_index["window"] == horizon]
    if sub.empty:
        return None

    available = {(r.model_key, r.algo_key) for r in sub.itertuples()}
    for choice in ALGO_PREFERENCE:
        if choice in available:
            return choice

    # Fallback to the first available combination
    first = sub.iloc[0]
    return (first["model_key"], first["algo_key"])


@st.cache_data(show_spinner=False)
def load_frame_for_model(exports_path: str, run_index_json: str, horizon: str, model_key: str, algo_key: str) -> pd.DataFrame:
    """
    Load all split CSVs for one horizon/model/algo combination
    and concatenate them into one dataframe.
    """
    source = Path(exports_path)
    run_index = pd.read_json(run_index_json)

    sub = run_index[
        (run_index["window"] == horizon)
        & (run_index["model_key"] == model_key)
        & (run_index["algo_key"] == algo_key)
    ].copy()

    frames = []
    for row in sub.itertuples():
        df = _read_csv_any(source, row.inner_path)
        df["split"] = row.split
        frames.append(df)

    if not frames:
        raise ValueError("No prediction files found for the selected horizon/model.")

    df = pd.concat(frames, ignore_index=True)
    for col in ["week_start", "target_window_start", "target_window_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def _pretty_event_piece(piece: str) -> str:
    """
    Clean and prettify one event/subtype name.
    """
    s = piece.strip().lower().replace(";", "")
    s = s.replace("__", "_")
    return PRETTY_EVENT.get(s, piece.replace("_", " ").strip().title())


def clean_event_string(value: object) -> str:
    """
    Turn a semicolon/comma/pipe-separated event list into a cleaner display string.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"

    s = str(value).strip()
    if not s:
        return "—"

    parts = [p.strip() for p in re.split(r"[;|,]", s) if p.strip()]
    if not parts:
        return "—"

    return "; ".join(_pretty_event_piece(p) for p in parts)


def build_event_list_from_scores(row: pd.Series) -> str:
    """
    If the exported file does not already contain a predicted subtype list,
    build one from the subtype score columns.
    """
    mapping = [
        ("score_battle_any", "Battle"),
        ("score_explosions_remote_any", "Explosions / Remote violence"),
        ("score_violence_against_civilians_any", "Violence against civilians"),
        ("score_air_drone_any", "Air / drone strike"),
        ("score_strategic_developments_any", "Strategic developments"),
    ]

    vals = []
    for col, label in mapping:
        if col in row.index:
            v = pd.to_numeric(row[col], errors="coerce")
            if pd.notna(v) and v >= 0.25:
                vals.append((float(v), label))

    vals.sort(reverse=True)

    # Show up to 3 reasonably strong subtype signals
    if not vals:
        best = None
        for col, label in mapping:
            if col in row.index:
                v = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(v):
                    best = max(best, (float(v), label)) if best else (float(v), label)
        return best[1] if best else "—"

    return "; ".join([label for _, label in vals[:3]])


def risk_band(score: float) -> str:
    """
    Convert a numeric risk score into a simple label.
    """
    if pd.isna(score):
        return "Unknown"
    if score >= 0.75:
        return "Very high"
    if score >= 0.50:
        return "High"
    if score >= 0.25:
        return "Moderate"
    return "Low"


def prepare_frame(df: pd.DataFrame, window_label: str) -> pd.DataFrame:
    """
    Normalize the loaded prediction frame so the UI can work
    even when some optional columns are missing.
    """
    out = df.copy()

    for c in out.columns:
        if c.startswith(("score_", "actual_", "pred_", "threshold_")):
            out[c] = pd.to_numeric(out[c], errors="ignore")

    # Use the high-intensity score when available, otherwise fall back to any-event risk
    if "score_high_intensity" in out.columns:
        out["score_operational_risk"] = pd.to_numeric(out["score_high_intensity"], errors="coerce")
    elif "score_any_event" in out.columns:
        out["score_operational_risk"] = pd.to_numeric(out["score_any_event"], errors="coerce")
    else:
        out["score_operational_risk"] = np.nan

    out["high_risk_band"] = out["score_operational_risk"].apply(risk_band)

    if "predicted_subtype_list" in out.columns:
        out["predicted_subtype_list"] = out["predicted_subtype_list"].apply(clean_event_string)
    else:
        out["predicted_subtype_list"] = out.apply(build_event_list_from_scores, axis=1)

    if "actual_subtype_list" in out.columns:
        out["actual_subtype_list"] = out["actual_subtype_list"].apply(clean_event_string)
    elif "actual_event_name" in out.columns:
        out["actual_subtype_list"] = out["actual_event_name"].apply(clean_event_string)
    else:
        out["actual_subtype_list"] = "—"

    if "target_window_start" in out.columns and "target_window_end" in out.columns:
        out["target_window_label"] = (
            pd.to_datetime(out["target_window_start"], errors="coerce").dt.strftime("%Y-%m-%d")
            + " → "
            + pd.to_datetime(out["target_window_end"], errors="coerce").dt.strftime("%Y-%m-%d")
        )
    else:
        out["target_window_label"] = WINDOW_LABELS.get(window_label, window_label)

    out["predicted_subtype_list"] = out["predicted_subtype_list"].fillna("—")
    out["actual_subtype_list"] = out["actual_subtype_list"].fillna("—")
    return out


def prepare_geojson_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert datetime-like columns to strings before pushing the dataframe
    into Plotly/GeoJSON.
    """
    out = gdf.copy()
    for col in out.columns:
        if col == "geometry":
            continue
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
            out[col] = out[col].where(out[col].notna(), None)
        elif out[col].dtype == "object":
            out[col] = out[col].apply(lambda x: x.strftime("%Y-%m-%d") if isinstance(x, pd.Timestamp) else x)
    return out


def make_numeric_map(gdf: gpd.GeoDataFrame, color_col: str, title: str, hover_cols: List[str]):
    """
    Build the Plotly choropleth map.
    """
    gplot = prepare_geojson_gdf(gdf)

    fig = px.choropleth_mapbox(
        gplot,
        geojson=json.loads(gplot.to_json()),
        locations="raion_id",
        featureidkey="properties.raion_id",
        color=color_col,
        color_continuous_scale="YlOrRd",
        mapbox_style="carto-positron",
        zoom=4.5,
        center={"lat": 48.8, "lon": 31.3},
        opacity=0.76,
        hover_name="raion_name",
        hover_data={c: True for c in hover_cols if c in gplot.columns},
        title=title,
        height=820,
    )

    fig.update_layout(margin={"r": 0, "t": 55, "l": 0, "b": 0})
    return fig


def format_priority_table(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Build the top-N priority table shown below the map.
    """
    work = df.sort_values(["score_any_event", "score_operational_risk"], ascending=False).head(top_n).copy()

    cols = [
        "raion_name",
        "oblast_name",
        "week_start",
        "target_window_label",
        "score_any_event",
        "score_operational_risk",
        "predicted_subtype_list",
        "actual_subtype_list",
        "pred_any_event",
        "actual_any_event",
        "pred_high_intensity",
        "actual_high_intensity",
        "pred_event_count",
        "actual_event_count",
        "pred_fatalities_sum",
        "actual_fatalities_sum",
        "pred_air_drone_strike_count",
        "actual_air_drone_strike_count",
        "split",
    ]
    keep = [c for c in cols if c in work.columns]

    view = work[keep].copy().rename(columns=PRETTY_COLUMNS)

    if "Anchor week" in view.columns:
        view["Anchor week"] = pd.to_datetime(view["Anchor week"], errors="coerce").dt.strftime("%Y-%m-%d")

    for c in [
        "Any-event risk", "High-risk severity score", "Predicted event count", "Observed event count",
        "Predicted fatalities", "Observed fatalities", "Predicted air/drone count", "Observed air/drone count",
    ]:
        if c in view.columns:
            view[c] = pd.to_numeric(view[c], errors="coerce").round(3)

    return view


st.markdown(
    """
    <style>
      div[data-testid="stHorizontalBlock"] > div:nth-child(2) p,
      div[data-testid="stHorizontalBlock"] > div:nth-child(2) li {
          font-size: 0.88rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Ukraine forecast map")
st.caption("Choose a horizon and anchor week. The app auto-selects one preferred model and shows predicted events and actual events side by side.")

with st.sidebar:
    st.header("Inputs")
    boundary_path = st.text_input("Boundary zip path", value=st.session_state.get("boundary_path", ""), key="boundary_path")
    exports_path = st.text_input("Main multiwindow folder or zip", value=st.session_state.get("exports_path", ""), key="exports_path")

if not boundary_path or not exports_path:
    st.info("Enter your Ukraine boundary zip and your main forecast folder or zip.")
    st.stop()

if not Path(boundary_path).exists() or not Path(exports_path).exists():
    st.error("Please provide valid local paths for both inputs.")
    st.stop()

try:
    boundary = load_boundary(boundary_path)
    run_index = discover_runs(exports_path)
except Exception as e:
    st.exception(e)
    st.stop()

left_top, right_top = st.columns([1.2, 1.1])
with left_top:
    horizon = st.selectbox(
        "Prediction horizon",
        [w for w in WINDOW_ORDER if w in run_index["window"].unique().tolist()],
        format_func=lambda x: WINDOW_LABELS[x],
    )
with right_top:
    map_mode = st.selectbox("Map layer", ["Any-event risk", "High-risk severity score", "Selected subtype risk"])

chosen = pick_best_model(run_index, horizon)
if chosen is None:
    st.error("No available model was found for this horizon.")
    st.stop()

model_key, algo_key = chosen

run_index_json = run_index.to_json(orient="records")
try:
    frame = prepare_frame(load_frame_for_model(exports_path, run_index_json, horizon, model_key, algo_key), horizon)
except Exception as e:
    st.exception(e)
    st.stop()

if "week_start" not in frame.columns or frame["week_start"].isna().all():
    st.error("This forecast file does not contain usable anchor weeks.")
    st.stop()

available_weeks = sorted(frame["week_start"].dropna().dt.normalize().unique().tolist())
selected_week_str = st.selectbox(
    "Anchor week",
    [pd.Timestamp(w).strftime("%Y-%m-%d") for w in available_weeks],
    index=len(available_weeks) - 1,
)
selected_week = pd.Timestamp(selected_week_str)
filtered = frame[frame["week_start"].dt.normalize() == selected_week.normalize()].copy()

if filtered.empty:
    st.warning("No rows match the selected horizon and anchor week.")
    st.stop()

control_a, control_b = st.columns([1.2, 1.0])
with control_a:
    top_n = st.slider("Top rows", min_value=10, max_value=100, value=25, step=5)
with control_b:
    subtype_options = [
        ("score_battle_any", "Battle risk"),
        ("score_explosions_remote_any", "Explosions / Remote violence risk"),
        ("score_violence_against_civilians_any", "Violence against civilians risk"),
        ("score_air_drone_any", "Air / drone strike risk"),
        ("score_strategic_developments_any", "Strategic developments risk"),
    ]
    present = [(c, lab) for c, lab in subtype_options if c in filtered.columns]

    if map_mode == "Selected subtype risk" and not present:
        st.warning("No subtype score columns were found in this file. Switching to Any-event risk.")
        map_mode = "Any-event risk"

    subtype_choice = (
        st.selectbox("Subtype risk layer", [c for c, _ in present], format_func=lambda x: dict(present)[x])
        if (map_mode == "Selected subtype risk" and present)
        else None
    )

if map_mode == "High-risk severity score":
    map_col = "score_operational_risk"
    map_title = "High-risk severity score"
elif map_mode == "Any-event risk":
    map_col = "score_any_event"
    map_title = "Any-event risk"
else:
    map_col = subtype_choice
    map_title = dict(present)[subtype_choice]

join_keys = [c for c in ["raion_id", "raion_name", "oblast_name"] if c in filtered.columns]
merged = boundary.merge(filtered, on=join_keys, how="left")

hover_cols = [
    "oblast_name", "target_window_label", "score_any_event", "score_operational_risk",
    "predicted_subtype_list", "actual_subtype_list", "pred_event_count", "actual_event_count",
    "pred_fatalities_sum", "actual_fatalities_sum", "pred_air_drone_strike_count", "actual_air_drone_strike_count",
]

col_map, col_notes = st.columns([4.0, 1.0])
with col_map:
    title_suffix = f"{WINDOW_LABELS[horizon]} • {selected_week_str} • {model_key.replace('_', ' ')} / {algo_key.replace('_', ' ')}"
    fig = make_numeric_map(merged, map_col, f"Ukraine forecast map — {map_title} — {title_suffix}", hover_cols)
    st.plotly_chart(fig, use_container_width=True)

with col_notes:
    st.subheader("How to read this")
    st.markdown(
        "- **Any-event risk** is the broad activity probability.\n"
        "- **High-risk severity score** indicates how likely a raion is to face a serious conflict week in the selected future window, where a week is treated as high risk if fatalities are at least 5, total event count is at least 10, or violence against civilians occurs.\n"
        "- **Predicted events** shows the predicted event-type list.\n"
        "- **Actual events** shows what ACLED recorded in the target window."
    )

    st.subheader("Event names")
    st.markdown(
        "- **Explosions / Remote violence**: shelling, missile, remote explosive-type violence.\n"
        "- **Air / drone strike**: air or drone strike-related activity.\n"
        "- **Battle**: direct battlefield clashes.\n"
        "- **Violence against civilians**: civilian-targeted violence.\n"
        "- **Strategic developments**: control changes, military movements, non-battle strategic actions."
    )

st.subheader("Priority view")
st.dataframe(format_priority_table(filtered, top_n), use_container_width=True, hide_index=True)