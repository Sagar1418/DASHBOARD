#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HT Cable-Health Dashboard (Interactive + Edge-Click Inspector)

Updates:
- Default 11 kV Scope is now "All rows".
- Default 11 kV Feeder is now the FIRST item in the list.
- Minimalist Uploader Styling.
- Fixed Timezone/PeriodArray UserWarnings.
"""
import csv, io, json, tempfile
from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import networkx as nx
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from pyvis.network import Network

# Try interactive component
try:
    from streamlit_vis_network import streamlit_vis_network as vis_network
    _USE_VIS = True
except Exception:
    vis_network = None
    _USE_VIS = False

# ──────────────────────────────────────────────────────────────────────────────
# Theme / CSS
# ──────────────────────────────────────────────────────────────────────────────
THEMES = {
    "Dark": {"bg":"#0b1220","bg2":"#0f172a","card":"#111827","text":"#e5e7eb","muted":"#a1a1aa","primary":"#10b981",
             "node":"#60a5fa","good":"#22c55e","mod":"#f59e0b","poor":"#ef4444","matrix_text":"#000"},
    "Light":{"bg":"#f5f7fb","bg2":"#ffffff","card":"#ffffff","text":"#0f172a","muted":"#475569","primary":"#059669",
             "node":"#2563eb","good":"#16a34a","mod":"#d97706","poor":"#b91c1c","matrix_text":"#000"},
}
BAND_COLORS: Dict[str, str] = {}

def inject_theme_css(t: Dict[str, str]) -> None:
    st.markdown(f"""
        <style>
            /* --- MAIN APP & LAYOUT --- */
            .stApp {{ background: linear-gradient(180deg, {t['bg']} 0%, {t['bg2']} 100%); }}
            .block-container {{ padding: 1rem 2rem 2rem 2rem; }}

            /* --- TABS --- */
            .stTabs [data-baseweb="tab"][aria-selected="true"] {{ border-bottom-color: {t['primary']}; }}
            .stTabs div[role="tabpanel"] {{
                background-color: {t['card']}; border-radius: 0 8px 8px 8px;
                padding: 1.2rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            }}

            /* --- METRICS & TEXT --- */
            div[data-testid="stMetricValue"] {{ color: {t['primary']}; }}
            .stDataFrame thead tr th {{ background-color: {t['bg2']} !important; color: {t['text']} !important; }}

            /* --- CHIPS (TAGS) --- */
            .chip {{
                display:inline-block; padding:3px 10px; border-radius:12px; margin-right:6px;
                font-size:11px; font-weight:600; background:{t['bg2']}; border: 1px solid {t['muted']}33;
                color: {t['text']}; letter-spacing: 0.5px;
            }}

            /* ─────────────────────────────────────────────────────────────
               FIXED MINIMALIST UPLOADER STYLE
            ───────────────────────────────────────────────────────────── */
            
            /* 1. Reset the Dropzone Box */
            [data-testid='stFileUploader'] section {{
                background-color: transparent;
                border: 1px dashed {t['muted']}66;
                border-radius: 8px;
                padding: 0; 
                transition: all 0.2s;
            }}
            
            [data-testid='stFileUploader'] section:hover {{
                border-color: {t['primary']};
                background-color: {t['bg2']}44;
            }}

            /* 2. THE CRITICAL FIX: Force Vertical Stacking & Centering */
            [data-testid='stFileUploader'] section > div {{
                display: flex !important;
                flex-direction: column !important; /* Stack top-to-bottom */
                align-items: center !important;    /* Center horizontally */
                justify-content: center !important; /* Center vertically */
                gap: 8px !important;
                padding: 1.5rem 1rem !important;
            }}

            /* 3. Hide the Default Streamlit Icon (The one causing the double icon issue) */
            [data-testid='stFileUploader'] section svg {{ 
                display: none !important; 
            }}
            
            /* 4. Add Custom Cloud Icon */
            [data-testid='stFileUploader'] section > div::before {{
               
                display: block; 
                font-size: 36px; 
                opacity: 0.6; 
                filter: grayscale(100%);
                line-height: 1;
                margin-bottom: 5px;
            }}

            /* 5. Style the Button */
            [data-testid='stFileUploader'] section button {{
                border: 1px solid {t['primary']};
                background-color: transparent; 
                color: {t['primary']};
                border-radius: 20px; 
                padding: 0.3rem 1.2rem;
                font-size: 13px; 
                font-weight: 500;
                width: auto !important; /* Stop it from stretching */
                margin: 0 auto !important; /* Ensure centering */
            }}
            
            [data-testid='stFileUploader'] section button:hover {{
                background-color: {t['primary']}; 
                color: {t['bg2']};
            }}

            /* ─────────────────────────────────────────────────────────────
               CUSTOM TEXT REPLACEMENT
            ───────────────────────────────────────────────────────────── */

            /* Hide Default Large Text */
            [data-testid='stFileUploader'] section > div > div > span {{
                display: none;
            }}

            /* Hide Default Small Text */
            [data-testid='stFileUploader'] section > div > div > small {{
                font-size: 0px !important;
                color: transparent !important;
            }}

            /* Inject New Custom Text */
            [data-testid='stFileUploader'] section > div > div > small::after {{
                font-size: 11px;
                color: {t['muted']};
                display: block;
                text-align: center;
                margin-top: -2px;
                content: "Drop files here"; /* Default fallback */
            }}

            /* Specific Text: Column 1 (11 kV) */
            div[data-testid="column"]:nth-of-type(1) [data-testid='stFileUploader'] small::after {{
                content: "Req: SWITCH_ID, health_score, health_band";
            }}

            /* Specific Text: Column 2 (22/33 kV) */
            div[data-testid="column"]:nth-of-type(2) [data-testid='stFileUploader'] small::after {{
                content: "Req: SWITCH_ID, health_score, health_band";
            }}

            /* Specific Text: Column 3 (Faults) */
            div[data-testid="column"]:nth-of-type(3) [data-testid='stFileUploader'] small::after {{
                content: "Req: SWITCH_ID, TIME_OUTAGE";
            }}

            /* Clean up uploaded file list */
            [data-testid='stFileUploader'] ul > li {{
                background: {t['bg2']} !important;
                border-left: 3px solid {t['primary']} !important;
            }}
        </style>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────
def sniff_delimiter(buf: bytes) -> str:
    sample = buf[:10_000].decode("utf-8", errors="ignore")
    try: return csv.Sniffer().sniff(sample).delimiter
    except csv.Error: return ","

def normalize_column(series: pd.Series) -> pd.Series:
    s = (series.astype(str).str.upper().str.strip()
         .str.replace(r"^(SWNO_|SWNO|SW|S)\s*", "", regex=True)
         .str.replace(r"\D+", "", regex=True)
         .replace("", np.nan))
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def find_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    return next((c for c in candidates if c in df.columns), None)

@st.cache_data(show_spinner="Processing scored data...")
def load_and_process_file(upload) -> pd.DataFrame:
    raw = upload.read(); upload.seek(0)
    df = pd.read_csv(io.BytesIO(raw), sep=sniff_delimiter(raw), low_memory=False)

    # normalize key columns
    rename_map = {c: "SOURCE_SS" for c in df.columns if str(c).startswith("SOURCE_SS")}
    rename_map.update({c: "DESTINATION_SS" for c in df.columns if str(c).startswith("DESTINATION_SS")})
    df = df.rename(columns=rename_map).loc[:, ~df.columns.duplicated()]

    need = {"health_score", "health_score_10", "health_band"}
    if not need.issubset(df.columns):
        miss = ", ".join(sorted(need - set(df.columns)))
        raise ValueError(f"CSV missing required columns: {miss}")

    if "SWITCH_ID" not in df.columns:
        cand = find_first_column(df, ["TO_SWITCH", "EDGE_KEY", "FROM_SWITCH"])
        df["SWITCH_ID"] = normalize_column(df[cand]) if cand else pd.Series(index=df.index, dtype="Int64")

    if "SWNO" in df.columns:
        df["SWNO"] = normalize_column(df["SWNO"])

    df["health_score"] = pd.to_numeric(df["health_score"], errors="coerce").clip(0, 100)
    df["health_score_10"] = pd.to_numeric(df["health_score_10"], errors="coerce")

    for col in ["INSTALL_DATE","INSTALLATION_DATE","COMMISSIONEDDATE","COMMISSION_DATE","DATE_INSTALLED"]:
        if col in df.columns: df[col] = pd.to_datetime(df[col], errors="coerce")
    for c in ["LENGTH_M","CABLE_LENGTH_M","MEASUREDLENGTH","LENGTH"]:
        if c in df.columns:
            df["__LENGTH_M__"] = pd.to_numeric(df[c], errors="coerce"); break
    return df

def sort_by_num(df: pd.DataFrame, col: str, ascending=True) -> pd.DataFrame:
    k = f"__sort_{col}__"
    out = df.assign(**{k: pd.to_numeric(df[col], errors="coerce")})
    return out.sort_values(k, ascending=ascending, na_position="last").drop(columns=k)

def enrich_faults(df_ft: pd.DataFrame) -> pd.DataFrame:
    df = df_ft.copy()
    dt = pd.to_datetime(df.get("TIME_OUTAGE", pd.NaT), errors="coerce", utc=True)
    df["TIME_OUTAGE"] = dt
    df = df.dropna(subset=["TIME_OUTAGE"]).copy()

    if "VOLTAGE" in df.columns:
        vnum = pd.to_numeric(df["VOLTAGE"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
        df["VOLTAGE_BUCKET"] = np.select(
            [abs(vnum-11)<=1.1, abs(vnum-22)<=2.2, abs(vnum-33)<=3.3],
            ["11 kV","22 kV","33 kV"], "Other")
    else:
        df["VOLTAGE_BUCKET"] = "Unknown"

    df["MONTH"] = df["TIME_OUTAGE"].dt.tz_localize(None).dt.to_period("M").dt.start_time
    df["WEEKDAY"] = df["TIME_OUTAGE"].dt.day_name()
    df["HOUR"] = df["TIME_OUTAGE"].dt.hour

    swc = find_first_column(df, ["TO_SWITCH","SWITCH_ID"])
    if swc: df["SWITCH_ID"] = normalize_column(df[swc])

    if "TIME_DIFFERENCE_HOURS" in df.columns:
        df["DURATION_H"] = pd.to_numeric(df["TIME_DIFFERENCE_HOURS"], errors="coerce")
    else:
        df["DURATION_H"] = np.nan
    return df

def compute_actual_failures(faults_enriched: pd.DataFrame, year: int) -> pd.Series:
    if "SWITCH_ID" not in faults_enriched.columns or "TIME_OUTAGE" not in faults_enriched.columns:
        return pd.Series([], dtype="Int64")
    tmp = faults_enriched.copy()
    tmp["TIME_OUTAGE"] = pd.to_datetime(tmp["TIME_OUTAGE"], errors="coerce", utc=True)
    return (tmp.loc[tmp["TIME_OUTAGE"].dt.year.eq(year), "SWITCH_ID"]
              .dropna().astype("Int64").unique())

def vis_with_height(nodes, edges, options, h, key=None):
    kwargs = {"nodes": nodes, "edges": edges, "options": options, "width": "100%"}
    if key: kwargs["key"] = key
    try: return vis_network(height=h, **kwargs)
    except TypeError: pass
    try: return vis_network(height=f"{h}px", **kwargs)
    except TypeError: pass
    try: return vis_network(height_px=h, **kwargs)
    except TypeError: pass
    opts = dict(options or {})
    opts["height"] = f"{h}px"
    kwargs["options"] = opts
    return vis_network(**kwargs)

# ──────────────────────────────────────────────────────────────────────────────
# Metrics & plots
# ──────────────────────────────────────────────────────────────────────────────
def calculate_metrics(df_eval: pd.DataFrame) -> Optional[dict]:
    if df_eval is None or df_eval.empty or "ACTUAL_FAIL_YEAR" not in df_eval.columns:
        return None
    y_true = df_eval["ACTUAL_FAIL_YEAR"].astype(int).values
    y_pred = df_eval["health_band"].map({"Poor":1, "Moderate":1, "Good":0}).fillna(0).astype(int).values
    risk = 1 - (pd.to_numeric(df_eval["health_score"], errors="coerce").fillna(50).values/100)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    return {
        "cm": cm,
        "Accuracy": (cm[0,0]+cm[1,1])/cm.sum() if cm.sum() else 0,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "AUROC": roc_auc_score(y_true, risk) if len(np.unique(y_true))>1 else np.nan
    }

def render_confusion_matrix(metrics: dict, title: str, theme: Dict[str, str]):
    st.subheader(title)
    if not metrics:
        st.warning("Not enough data to calculate metrics.")
        return
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown(f"""
| Metric      | Value     |
|-------------|-----------|
| Accuracy    | `{metrics["Accuracy"]:.2%}` |
| Precision   | `{metrics["Precision"]:.2%}` |
| Recall      | `{metrics["Recall"]:.2%}`   |
| F1-Score    | `{metrics["F1-Score"]:.2%}`  |
| **AUROC** | `{metrics["AUROC"]:.3f}`    |
""")
    with c2:
        fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=150, facecolor=theme["card"])
        sns.heatmap(metrics["cm"], annot=True, fmt="d", cbar=False,
                    xticklabels=["Predicted OK","Predicted Fail"],
                    yticklabels=["Actual OK","Actual Fail"], ax=ax)
        ax.set_facecolor(theme["card"])
        ax.set_xlabel("Prediction", color=theme["muted"])
        ax.set_ylabel("Reality", color=theme["muted"])
        ax.tick_params(colors=theme["text"])
        st.pyplot(fig)

def plot_health_distribution(scores: pd.Series, theme: Dict[str, str]):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120, facecolor=theme["card"])
    ax.set_facecolor(theme["card"])
    x = pd.to_numeric(scores, errors="coerce").dropna().clip(0, 100)
    if x.empty:
        ax.text(0.5, 0.5, "No data in view", ha='center', color=theme["muted"])
        return fig
    bins = np.arange(0, 101, 5)
    counts, edges = np.histogram(x, bins=bins)
    colors = [BAND_COLORS.get("Poor") if e < 40 else BAND_COLORS.get("Moderate") if e < 60 else BAND_COLORS.get("Good") for e in edges[:-1]]
    ax.bar(edges[:-1], counts, width=5, align="edge", color=colors, edgecolor=theme["bg2"], linewidth=0.5)
    ax.set(xlim=(0, 100), xlabel="Health Score", ylabel="Cable Count", title="Health Score Distribution (Current View)")
    ax.title.set_color(theme["text"])
    ax.xaxis.label.set_color(theme["muted"]); ax.yaxis.label.set_color(theme["muted"])
    ax.tick_params(colors=theme["text"]); ax.yaxis.grid(True, linestyle=":", alpha=0.3)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(theme["muted"])
    fig.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# Edge → Inspector helpers
# ──────────────────────────────────────────────────────────────────────────────
INSPECT_CORE = [
    "EDGE_KEY","FROM_SWITCH","TO_SWITCH",
    "SOURCE_SS","DESTINATION_SS","FEEDERID_FULL",
    "CABLETYPE","STD_CABLE_SIZE",
    "AGG_MEASUREDLENGTH","NO_OF_SEGMENT",
    "COMMISSIONEDDATE_DT_OLDEST","CLUSTER_TYPE","REMARKS",
    "health_score","health_band","primary_health_driver","top3_health_drivers"
]


def edge_properties_frame(row: pd.Series) -> pd.DataFrame:
    keep = []
    for c in INSPECT_CORE:
        if c in row.index and pd.notna(row[c]): 
            # FIX: Convert the value to a string explicitly to avoid mixed-type serialization error
            display_value = str(row[c]) 
            keep.append((c, display_value))
            
    return pd.DataFrame(keep, columns=["Property","Value"])

def mini_series_from_months(row: pd.Series, prefix: str) -> Optional[pd.DataFrame]:
    cols = [c for c in row.index if c.startswith(f"{prefix}_Month_")]
    if not cols: return None
    cols = sorted(cols, key=lambda x: int(x.split("_")[-1]))
    vals = [pd.to_numeric(row[c], errors="coerce") for c in cols]
    idx = [c.split("_")[-1] for c in cols]
    return pd.DataFrame({"month": idx, prefix: vals})

def render_edge_inspector(row: pd.Series, faults_enriched: Optional[pd.DataFrame], theme: Dict[str, str]) -> None:
    st.markdown("### 🔎 Edge Inspector")
    sid = row.get("FEEDER_ID_x", "—")
    hs  = row.get("health_score", np.nan)
    band = str(row.get("health_band", "—"))
    pri  = row.get("primary_health_driver", "—")
    band_color = theme["good"]
    if isinstance(band, str) and band.lower().startswith("poor"): band_color = theme["poor"]
    elif isinstance(band, str) and band.lower().startswith("mod"): band_color = theme["mod"]

    chips = (
        f"<span class='chip'>FEEDER_ID {sid}</span>"
        f"<span class='chip'>HS {'' if pd.isna(hs) else int(hs)}</span>"
        f"<span class='chip' style='background:{band_color}26;border:1px solid {band_color}55;'>Band {band}</span>"
        f"<span class='chip'>Primary concern: {pri}</span>"
    )
    st.markdown(chips, unsafe_allow_html=True)

    pcols = st.columns([1.6, 1.4])
    with pcols[0]:
        props = edge_properties_frame(row)
        if props.empty: st.info("No properties available for this edge.")
        else: st.dataframe(props, hide_index=True, use_container_width=True, height=360)

    with pcols[1]:
        var_df = mini_series_from_months(row, "VAR")
        cyc_df = mini_series_from_months(row, "CYCLE")
        if var_df is not None:
            fig_v = px.bar(var_df, x="month", y="VAR", title="Variation by Month")
            fig_v.update_layout(height=200, margin=dict(l=10,r=10,t=35,b=10),
                                paper_bgcolor=theme["card"], plot_bgcolor=theme["card"], font_color=theme["text"])
            st.plotly_chart(fig_v, use_container_width=True)
        if cyc_df is not None:
            fig_cy = px.line(cyc_df, x="month", y="CYCLE", markers=True, title="Cycle Count by Month")
            fig_cy.update_layout(height=200, margin=dict(l=10,r=10,t=35,b=10),
                                 paper_bgcolor=theme["card"], plot_bgcolor=theme["card"], font_color=theme["text"])
            st.plotly_chart(fig_cy, use_container_width=True)

        sid_sw = row.get("SWITCH_ID")
        if pd.notna(sid_sw) and faults_enriched is not None and "SWITCH_ID" in faults_enriched.columns:
            df_f = faults_enriched.loc[faults_enriched["SWITCH_ID"] == sid_sw].copy()
            if not df_f.empty:
                if "MONTH" not in df_f.columns:
                    df_f["MONTH"] = pd.to_datetime(df_f["TIME_OUTAGE"], errors="coerce", utc=True).dt.tz_localize(None).dt.to_period("M").dt.start_time
                spark = df_f.groupby("MONTH").size().rename("faults").reset_index()
                fig_s = px.line(spark, x="MONTH", y="faults", markers=True, title="Fault History (per month)")
                fig_s.update_layout(height=200, margin=dict(l=10,r=10,t=35,b=0),
                                    paper_bgcolor=theme["card"], plot_bgcolor=theme["card"], font_color=theme["text"])
                st.plotly_chart(fig_s, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit Page setup
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="HT Cable-Health Dashboard", layout="wide")

if "data" not in st.session_state:
    st.session_state.data = {"11 kV": None, "22/33 kV": None}
if "faults_enriched" not in st.session_state:
    st.session_state.faults_enriched = None
if "selected_graph_idx" not in st.session_state:
    st.session_state.selected_graph_idx = None

with st.sidebar:
    st.markdown("### Appearance")
    theme_choice = st.selectbox("Theme", list(THEMES.keys()), index=0)
theme = THEMES[theme_choice]
inject_theme_css(theme)
BAND_COLORS.update({"Poor": theme["poor"], "Moderate": theme["mod"], "Good": theme["good"]})

st.title("HT Cable-Health Comparison Dashboard")

# Uploads
st.header("1. Upload Data Files")
c1, c2, c3 = st.columns(3)
file_11kv = c1.file_uploader("11 kV Scored Data", type=["csv","tsv"])
file_22_33kv = c2.file_uploader("22/33 kV Scored Data", type=["csv","tsv"])
file_faults = c3.file_uploader("Faults Data (Common)", type=["csv","tsv"])

if file_11kv:    st.session_state.data["11 kV"] = load_and_process_file(file_11kv)
if file_22_33kv: st.session_state.data["22/33 kV"] = load_and_process_file(file_22_33kv)

selected_year = None
if file_faults:
    raw = file_faults.read(); file_faults.seek(0)
    df_ft = pd.read_csv(io.BytesIO(raw), sep=sniff_delimiter(raw), low_memory=False)
    st.session_state.faults_enriched = enrich_faults(df_ft)
    years = sorted(st.session_state.faults_enriched["TIME_OUTAGE"].dt.year.dropna().unique().astype(int))
    if years:
        with st.sidebar:
            st.subheader(" Fault Year for Performance")
            default_idx = years.index(2024) if 2024 in years else len(years)-1
            selected_year = st.selectbox("Select Year", years, index=default_idx)
        failed_switches = compute_actual_failures(st.session_state.faults_enriched, selected_year)
        for df in st.session_state.data.values():
            if df is not None:
                df["ACTUAL_FAIL_YEAR"] = df["SWITCH_ID"].isin(failed_switches).astype(int)

available = [k for k,v in st.session_state.data.items() if v is not None]
if not available:
    st.info("Please upload at least one scored data file to begin analysis.")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header(" Dashboard Controls")
    selected_dataset_key = st.radio("Dataset:", available, horizontal=True)
    df_display = st.session_state.data[selected_dataset_key]
    is_11kv = (selected_dataset_key == "11 kV")

    # Detect feeder column for 11 kV filtering
    FEEDER_COL_CANDIDATES = ["FEEDER_ID_x","FEEDER_ID","SWNO","FEEDERID_FULL","FEEDERID","FEEDER_NO"]
    feeder_col = find_first_column(df_display, FEEDER_COL_CANDIDATES)

    with st.expander("Scope & Filters", expanded=True):
        if is_11kv:
            # CHANGED: Index=1 sets default to "All rows"
            scope_mode = st.radio("Scope", ["Top-N worst","All rows"], index=1)
            top_n = st.number_input("N (Top-N worst)", min_value=100, max_value=len(df_display), value=min(500, len(df_display)), step=100)

            # NEW: Feeder ID filter (MULTI-SELECT)
            feeder_selected = []
            expand_component = False
            if feeder_col:
                feeder_values = (df_display[feeder_col]
                                 .dropna()
                                 .astype(str)
                                 .drop_duplicates()
                                 .sort_values()
                                 .tolist())
                # CHANGED: Select First Feeder by default
                default_feeder = [feeder_values[0]] if feeder_values else []
                feeder_selected = st.multiselect(f"Feeder IDs (from {feeder_col})", feeder_values, default=default_feeder)
                expand_component = st.checkbox("Expand to connected network (all edges touching its stations)", True)

        decile_range = st.slider("Health Decile (1=worst)", 1, 10, (1, 10))
        band_selected = st.multiselect("Health Band",
                                       sorted(df_display["health_band"].dropna().unique().tolist()),
                                       default=sorted(df_display["health_band"].dropna().unique().tolist()))
        keyword = st.text_input("Keyword Search (station/switch)", "").upper()
        if "SWNO" in df_display.columns:
            swno_selected = st.multiselect("Feeder IDs (SWNO)", sorted(df_display["SWNO"].dropna().unique().tolist()))
        else:
            swno_selected = []

    with st.expander(" Graph Customization", expanded=True):
        graph_mode = st.radio("Graph Renderer", ["Interactive (vis-network)", "PyVis (fallback)", "Both"])
        hard_cap = int(min(15000, len(df_display)))
        edge_cap = st.slider("Max Edges", 50, hard_cap, min(1000, hard_cap), step=50)
        canvas_height = st.slider("Canvas Height (px)", 100, 2600, 1200, step=50)
        label_color = st.color_picker("Node label font color", value=theme["text"])
        label_size = st.slider("Node label font size", 8, 28, 16)
        pyvis_bg = st.color_picker("PyVis background color", value=theme["bg2"])

    with st.expander(" Alert Configuration", expanded=False):
        alert_threshold = st.slider("Alert on Health Score below:", 0, 100, 20)

    with st.expander("🔬 Fault Analysis Controls", expanded=False):
        fault_controls = {
            'time_agg': st.selectbox("Time Aggregation", ["Monthly", "Weekly", "Daily"]),
            'rolling_window': st.slider("Rolling Avg Window", 1, 12, 1),
            'breakdown_dim': st.selectbox("Breakdown Dimension", ["STATION_NAME", "CABLE_TYPE", "STD_CABLE_SIZE"]),
            'top_k': st.slider("Top-K Categories", 3, 20, 10)
        }

# Build filtered view
scope_note = "All rows"
if is_11kv and 'scope_mode' in locals() and scope_mode == "Top-N worst":
    scope_df = sort_by_num(df_display, "health_score", ascending=True).head(int(top_n)).copy()
    scope_note = f"Top-{int(top_n)} worst"
else:
    scope_df = df_display.copy()

# Base mask
m = pd.Series(True, index=scope_df.index)
m &= pd.to_numeric(scope_df["health_score_10"], errors="coerce").between(*decile_range)
m &= scope_df["health_band"].isin(band_selected)

# Keyword match
if keyword:
    cols = [c for c in ["SOURCE_SS","DESTINATION_SS","SWITCH_ID"] if c in scope_df.columns]
    if cols:
        m &= scope_df[cols].astype(str).apply(lambda s: s.str.upper().str.contains(keyword, na=False)).any(axis=1)

# Additional SWNO multiselect (if present)
if 'SWNO' in scope_df.columns and swno_selected:
    m &= scope_df["SWNO"].isin(swno_selected)

# NEW: Feeder selection logic (now multiselect)
if is_11kv and feeder_col and 'feeder_selected' in locals() and feeder_selected:
    # 1) Find all edges that belong to ANY of the selected feeders
    feeder_edges = df_display[df_display[feeder_col].astype(str).isin(feeder_selected)].copy()
    # 2) Collect all stations that appear in those edges
    feeder_nodes = set(feeder_edges["SOURCE_SS"].astype(str).dropna()) | set(feeder_edges["DESTINATION_SS"].astype(str).dropna())
    # 3) If expand_component is True, include *all* edges that touch any of those stations
    if expand_component:
        in_component = (scope_df["SOURCE_SS"].astype(str).isin(feeder_nodes) |
                        scope_df["DESTINATION_SS"].astype(str).isin(feeder_nodes))
        m &= in_component
        scope_note = f"Connected network for {len(feeder_selected)} feeder(s)"
    else:
        m &= (scope_df[feeder_col].astype(str).isin(feeder_selected))
        display_feeders = ', '.join(feeder_selected[:3])
        if len(feeder_selected) > 3: display_feeders += f" (+{len(feeder_selected)-3} more)"
        scope_note = f"Feeders: {display_feeders}"

df_view = scope_df[m].copy()

if df_view.empty:
    st.warning("No data matches the current settings.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
st.header(f"2. Analysis for: **{selected_dataset_key}**")
tabs = st.tabs(["🕸️ Network Graph", " Summary", " Alerts", " Fault Analysis", " Performance", " Data"])

# ──────────────────────────────────────────────────────────────────────────────
# NETWORK
# ──────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.caption(
        f"Scope: **{scope_note}**. "
        f"Filtered: **{len(df_view):,}** cables. "
        f"Rendering **{min(edge_cap, len(df_view))}** edges."
    )
    df_graph = sort_by_num(df_view, "health_score", ascending=True).head(int(min(edge_cap, len(df_view)))).copy()

    nodes_dict: Dict[str, dict] = {}
    vis_edges: List[dict] = []
    G = nx.MultiDiGraph()
    pair_to_edge_ids: Dict[Tuple[str, str], list] = {}
    pos_to_row: Dict[str, int] = {}
    selection_options = []

    for idx, r in df_graph.iterrows():
        src = str(r.get("SOURCE_SS", r.get("FROM_SWITCH", "SRC")))
        dst = str(r.get("DESTINATION_SS", r.get("TO_SWITCH", "DST")))
        hs  = pd.to_numeric(r.get("health_score"), errors="coerce")
        dec = pd.to_numeric(r.get("health_score_10"), errors="coerce")
        band = str(r.get("health_band", ""))
        color = {"Poor": BAND_COLORS["Poor"], "Moderate": BAND_COLORS["Moderate"], "Good": BAND_COLORS["Good"]}.get(band, "#808080")
        sid_txt = r.get("SWITCH_ID", "-")
        hs_txt = "-" if pd.isna(hs) else f"{int(hs)}"
        selection_options.append((idx, f"{src} → {dst} | SWITCH {sid_txt} | HS {hs_txt}"))

        inst_date = pd.to_datetime(r.get("COMMISSIONEDDATE", r.get("DATE_INSTALLED", "")), errors="coerce")
        inst_str = inst_date.strftime("%Y-%m-%d") if not pd.isna(inst_date) else "N/A"
        tooltip_lines = list(filter(None, [
            f"From_switch : {r.get('FROM_SWITCH','-')}",
            f"To_switch : {r.get('TO_SWITCH','-')}",
            f"Band : {r.get('health_band','-')}",
            f"Health-score : {hs if not pd.isna(hs) else '-'}",
            f"Decile : {dec if not pd.isna(dec) else '-'}",
            f"Primary driver : {r.get('primary_health_driver','-')}",
            f"Top3 drivers : {r.get('top3_health_drivers','-')}",
            f"SWITCH_ID : {r.get('SWITCH_ID','-')}",
            (f"SWNO : {r.get('SWNO','-')}" if 'SWNO' in r else None),
            f"Length (m) : {r.get('LENGTH_M','-')}",
            f"Installed : {inst_str}",
        ]))
        title = "\n".join(tooltip_lines)
        width = float(2 + max(0, 11 - (0 if pd.isna(dec) else int(dec))) * 1.5)
        edge_id = str(idx)
        pos_to_row[edge_id] = idx
        pair_to_edge_ids.setdefault((src, dst), []).append(edge_id)

        if src not in nodes_dict: nodes_dict[src] = {"id": src, "label": src}
        if dst not in nodes_dict: nodes_dict[dst] = {"id": dst, "label": dst}
        vis_edges.append({"id": edge_id, "from": src, "to": dst,
                          "color": {"color": color}, "arrows": "to",
                          "title": title, "width": width})
        if src not in G: G.add_node(src, title=f"Station: {src}")
        if dst not in G: G.add_node(dst, title=f"Station: {dst}")
        G.add_edge(src, dst, title=title, label="", color=color, width=width)

    st.markdown("#### Network")

    def render_interactive():
        if not (_USE_VIS and callable(vis_network)):
            st.warning("Interactive graph component not available in this environment.")
            return
        options = {
            "nodes": {"shape": "dot", "size": 30, "font": {"color": label_color, "size": label_size}},
            "edges": {"color": {"inherit": False}},
            "physics": {
                "stabilization": {"enabled": True, "iterations": 150},
                "forceAtlas2Based": {"gravitationalConstant": -100, "centralGravity": 0.005,
                                     "springLength": 250, "springConstant": 0.09, "avoidOverlap": 0.5},
                "solver": "forceAtlas2Based", "timestep": 0.7
            },
            "interaction": {"hover": True}
        }
        graph_key = f"network_{len(vis_edges)}_{selected_dataset_key}"
        selection = vis_with_height(
            nodes=list(nodes_dict.values()),
            edges=vis_edges,
            options=options,
            h=canvas_height,
            key=graph_key
        )
        new_selection_idx = None
        def as_list(x):
            if x is None: return []
            if isinstance(x, (list, tuple, set)): return list(x)
            return [x]
        try:
            if isinstance(selection, dict):
                raw = selection.get("edges") or selection.get("selectedEdges") or selection.get("selected")
                if raw:
                    ids = [str(v) for v in as_list(raw) if str(v) in pos_to_row]
                    if ids: new_selection_idx = pos_to_row[ids[-1]]
            elif isinstance(selection, (list, tuple)):
                if len(selection) >= 2 and isinstance(selection[1], (list, tuple)):
                    candidates = []
                    for item in selection[1]:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            a, b = str(item[0]), str(item[1])
                            for key in ((a,b),(b,a)):
                                for eid in pair_to_edge_ids.get(key, []):
                                    ridx = pos_to_row[eid]
                                    hs_val = pd.to_numeric(df_graph.loc[ridx].get("health_score"), errors="coerce")
                                    hs_val = hs_val if not pd.isna(hs_val) else 999.0
                                    candidates.append((eid, ridx, hs_val))
                    if candidates:
                        candidates.sort(key=lambda x: x[2])
                        new_selection_idx = candidates[0][1]
                if new_selection_idx is None:
                    ids = [str(v) for v in selection if str(v) in pos_to_row]
                    if ids: new_selection_idx = pos_to_row[ids[-1]]
            elif isinstance(selection, (int, np.integer, str)):
                sid = str(selection)
                if sid in pos_to_row: new_selection_idx = pos_to_row[sid]
        except Exception as e:
            st.warning(f"Could not parse selection: {e}")
        if new_selection_idx is not None:
             st.session_state.selected_graph_idx = new_selection_idx

    def render_pyvis():
        net = Network(height=f"{canvas_height}px", width="100%", bgcolor=pyvis_bg, directed=True)
        net.from_nx(G)
        net.set_options(json.dumps({
            "nodes":{"size":35,"font":{"color": label_color, "size": label_size, "strokeWidth":1}},
            "edges":{"arrows":{"to":{"enabled":True,"scaleFactor":0.7}},"smooth":{"type":"dynamic"}},
            "physics":{"forceAtlas2Based":{"gravitationalConstant":-100,"centralGravity":0.005,
                                           "springLength":250,"springConstant":0.09,"avoidOverlap":0.5},
                       "solver":"forceAtlas2Based","timestep":0.7,
                       "stabilization":{"enabled":True,"iterations":150}}
        }))
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            net.save_graph(tmp.name)
            html = open(tmp.name, encoding="utf-8").read() + \
                   "<script>setTimeout(()=>{if(window.network)network.setOptions({physics:false});},25000);</script>"
            st.components.v1.html(html, height=canvas_height + 20, scrolling=True)

    if graph_mode == "Interactive (vis-network)":
        render_interactive()
    elif graph_mode == "PyVis (fallback)":
        render_pyvis()
    else:
        st.markdown("**Interactive (clickable)**")
        render_interactive()
        st.markdown("---")
        st.markdown("**PyVis (static fallback)**")
        render_pyvis()

    st.markdown("---")
    effective_index = st.session_state.selected_graph_idx
    if effective_index not in df_graph.index:
        effective_index = df_graph.index[0] if not df_graph.empty else None
        st.session_state.selected_graph_idx = effective_index

    current_option = next((opt for opt in selection_options if opt[0] == effective_index), selection_options[0] if selection_options else None)
    def on_manual_change():
        sel_str = st.session_state["manual_selector"]
        idx_val = int(sel_str.split(":")[0])
        st.session_state.selected_graph_idx = idx_val

    if current_option:
        st.selectbox(
            "Inspect specific edge:",
            options=[f"{i}: {lbl}" for i, lbl in selection_options],
            index=selection_options.index(current_option),
            key="manual_selector",
            on_change=on_manual_change
        )
    if effective_index is not None:
        row = df_graph.loc[effective_index]
        render_edge_inspector(row, st.session_state.get("faults_enriched"), THEMES[theme_choice])

# SUMMARY
with tabs[1]:
    c1, c2 = st.columns([1.5, 2])
    with c1:
        st.metric("Cables in View", f"{len(df_view):,}")
        st.metric("Mean Health Score", f"{pd.to_numeric(df_view['health_score'], errors='coerce').mean():.1f}")
        st.subheader("Top 10 Worst Cables")
        cols = [c for c in ["FROM_SWITCH","TO_SWITCH","SWNO","SOURCE_SS","DESTINATION_SS","health_score","health_band"] if c in df_view.columns]
        st.dataframe(sort_by_num(df_view, "health_score", ascending=True).head(10)[cols],
                     use_container_width=True, height=385)
    with c2:
        st.pyplot(plot_health_distribution(df_view["health_score"], THEMES[theme_choice]))

# ALERTS
with tabs[2]:
    st.subheader(" Live Attention Needed")
    df_alerts = sort_by_num(df_display[df_display["health_score"] < alert_threshold].copy(),
                            "health_score", ascending=True)
    st.metric("Cables Requiring Immediate Attention", len(df_alerts))
    if df_alerts.empty:
        st.success("No cables are currently below the alert threshold.")
    else:
        st.caption(f"Cables from **{selected_dataset_key}** with health score below **{alert_threshold}**.")
        for _, r in df_alerts.iterrows():
            hs, sw_id = r.get("health_score", np.nan), r.get("SWITCH_ID", "N/A")
            title = f"**SWITCH ID:** {sw_id} | **HS:** {hs:.0f}" if pd.notna(hs) else f"**SWITCH ID:** {sw_id}"
            with st.expander(title):
                essentials = {k: v for k, v in r.to_dict().items()
                              if k in INSPECT_CORE and pd.notna(v)}
                st.json(essentials)

# FAULT ANALYSIS
with tabs[3]:
    if st.session_state.faults_enriched is None:
        st.info("Upload a faults data file to see historical trend analysis.")
    else:
        sub_tabs = st.tabs(["Common (All)", "11 kV", "22 kV", "33 kV"])
        def generate_fault_analysis_layout(df_faults, theme, controls, tab_key):
            if df_faults is None or df_faults.empty:
                st.info("No fault data available for this selection."); return
            min_date = df_faults["TIME_OUTAGE"].min().date()
            max_date = df_faults["TIME_OUTAGE"].max().date()
            date_range = st.slider("Filter by Date Range", min_date, max_date, (min_date, max_date),
                                   format="MMM YYYY", key=f"slider_{tab_key}")
            filtered = df_faults[(df_faults["TIME_OUTAGE"].dt.date >= date_range[0]) &
                                 (df_faults["TIME_OUTAGE"].dt.date <= date_range[1])].copy()
            if filtered.empty: st.warning("No faults in selected date range."); return
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Faults", len(filtered))
            if "CABLE_TYPE" in filtered.columns and not filtered["CABLE_TYPE"].mode().empty:
                c2.metric("Top Cable Type", filtered["CABLE_TYPE"].mode()[0])
            c3.metric("Avg Duration (h)", f"{filtered['DURATION_H'].mean():.2f}" if "DURATION_H" in filtered.columns and filtered["DURATION_H"].notna().any() else "N/A")
            freq_map = {"Daily":"D","Weekly":"W","Monthly":"M"}
            freq = freq_map.get(controls.get("time_agg","Monthly"), "M")
            filtered["__bucket__"] = filtered["TIME_OUTAGE"].dt.to_period(freq).dt.start_time
            ts = filtered.groupby("__bucket__").size().rename("Faults")
            suffix = ""
            if controls.get("rolling_window", 1) > 1:
                ts = ts.rolling(controls["rolling_window"], min_periods=1).mean()
                suffix = f" ({controls['rolling_window']}-period rolling avg)"
            fig = px.line(ts.reset_index(), x="__bucket__", y="Faults", markers=True,
                          title=f"{controls.get('time_agg','Monthly')} Faults Trend{suffix}",
                          labels={"__bucket__":"Date","Faults":"Number of Faults"})
            fig.update_layout(paper_bgcolor=theme["card"], plot_bgcolor=theme["card"],
                              font_color=theme["text"], margin=dict(t=40,l=0,r=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
            dim = controls.get("breakdown_dim", "CABLE_TYPE")
            top_k = int(controls.get("top_k", 10))
            c1, c2 = st.columns(2)
            if dim in filtered.columns and filtered[dim].notna().any():
                vc = filtered[dim].value_counts().nlargest(top_k)
                with c1:
                    st.subheader(f"Top {top_k} by {dim}")
                    fig_bar = px.bar(vc.sort_values(), x=vc.sort_values().values, y=vc.sort_values().index,
                                     orientation='h', labels={"x":"Number of Faults","y":""})
                    fig_bar.update_layout(paper_bgcolor=theme["card"], plot_bgcolor=theme["card"],
                                          font_color=theme["text"], margin=dict(t=20, l=0, r=0, b=0))
                    st.plotly_chart(fig_bar, use_container_width=True)
                with c2:
                    st.subheader(f"Fault Share by {dim}")
                    fig_pie = px.pie(vc, values=vc.values, names=vc.index, hole=0.4)
                    fig_pie.update_layout(paper_bgcolor=theme["card"], font_color=theme["text"],
                                          margin=dict(t=20, l=0, r=0, b=0), legend_title_text=dim)
                    st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info(f"No data available for dimension '{dim}'.")
        with sub_tabs[0]:
            generate_fault_analysis_layout(st.session_state.faults_enriched, THEMES[theme_choice], fault_controls, "common")
        with sub_tabs[1]:
            generate_fault_analysis_layout(st.session_state.faults_enriched.query('VOLTAGE_BUCKET == "11 kV"'), THEMES[theme_choice], fault_controls, "v11")
        with sub_tabs[2]:
            generate_fault_analysis_layout(st.session_state.faults_enriched.query('VOLTAGE_BUCKET == "22 kV"'), THEMES[theme_choice], fault_controls, "v22")
        with sub_tabs[3]:
            generate_fault_analysis_layout(st.session_state.faults_enriched.query('VOLTAGE_BUCKET == "33 kV"'), THEMES[theme_choice], fault_controls, "v33")

# PERFORMANCE
with tabs[4]:
    if selected_year is None:
        st.info("Upload faults data and select a year to see performance comparisons.")
    else:
        st.info(f"Comparing predictions vs actual faults for **{selected_year}**.")
        m11 = calculate_metrics(st.session_state.data.get("11 kV"))
        m23 = calculate_metrics(st.session_state.data.get("22/33 kV"))
        c1, c2 = st.columns(2)
        with c1:
            if m11: render_confusion_matrix(m11, "11 kV Performance", THEMES[theme_choice])
            else: st.warning("No 11 kV data / faults.")
        with c2:
            if m23: render_confusion_matrix(m23, "22/33 kV Performance", THEMES[theme_choice])
            else: st.warning("No 22/33 kV data / faults.")

# DATA
with tabs[5]:
    st.subheader(f"Full Filtered Data — {selected_dataset_key}")
    cols = [c for c in ["SWITCH_ID","SWNO","SOURCE_SS","DESTINATION_SS",
                        "health_score_10","health_band","health_score"] if c in df_view.columns]
    st.dataframe(sort_by_num(df_view, "health_score", ascending=True)[cols], use_container_width=True)