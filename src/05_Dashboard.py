"""
GridLock Patrol Intelligence — Streamlit Dashboard
Reads the pre-computed CSVs produced by notebooks 01-04.

UI/UX redesigned for a command-center intelligence aesthetic.
All widgets, functions and data logic are preserved unchanged.
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from config import (
    ANOMALY_ANALYSIS,
    BEST_PATROL_WINDOWS,
    BEST_PATROL_WINDOWS_DETAILED,
    HOTSPOT_CLUSTERED,
    HOTSPOT_CLUSTER_STATS,
    PATROL_ACTION_PLAN,
    RAW_DATA,
    TOP_ENFORCEMENT_ZONES,
    TRAJECTORY_ANALYSIS,
)
from temporal_utils import parse_local_naive_series

st.set_page_config(
    page_title="Gridlock Patrol Intelligence",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
RISK_ORDER = ["Low", "Medium", "High", "Critical"]
RISK_COLORS = {
    "Low":      "#22C55E",
    "Medium":   "#F2C14E",
    "High":     "#F78154",
    "Critical": "#EF4444",
}
TRAJECTORY_COLORS = {
    "Escalating":           "#EF4444",
    "Stable":               "#3B82F6",
    "Declining":            "#94A3B8",
    "Insufficient History": "#64748B",
}
DEPLOYMENT_COLORS = {
    "Routine Monitoring":   "#22C55E",
    "Targeted Patrol":      "#F2C14E",
    "Immediate Enforcement":"#EF4444",
}

PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, system-ui, sans-serif", color="#cbd5e1", size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.55)",
    xaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.15)"),
    yaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.15)"),
    legend=dict(bgcolor="rgba(15,23,42,0.4)", bordercolor="rgba(148,163,184,0.15)",
                borderwidth=1, font=dict(size=11)),
    title=dict(font=dict(size=13, color="#e2e8f0"), x=0.01, xanchor="left"),
)

def style_fig(fig, *, transparent_plot=False):
    layout = {k: v for k, v in PLOTLY_LAYOUT.items()}
    if transparent_plot:
        layout["plot_bgcolor"] = "rgba(0,0,0,0)"
    fig.update_layout(**layout)
    return fig


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

.stApp {
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(59,130,246,.08), transparent 60%),
        radial-gradient(900px 500px at 110% 10%, rgba(239,68,68,.06), transparent 60%),
        #070b16;
    color: #e2e8f0;
}
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px; }

.main-header {
    position: relative;
    background: linear-gradient(135deg, rgba(15,23,42,.95) 0%, rgba(30,58,95,.85) 55%, rgba(26,47,78,.95) 100%);
    padding: 1.6rem 1.9rem; border-radius: 14px; margin-bottom: 1.1rem;
    border: 1px solid rgba(59,130,246,0.28);
    box-shadow: 0 10px 40px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,.04);
    overflow: hidden;
}
.main-header::before {
    content:""; position:absolute; inset:0;
    background: repeating-linear-gradient(90deg, rgba(59,130,246,.04) 0 2px, transparent 2px 60px);
    pointer-events:none;
}
.main-header h1 { color:#f0f9ff; font-size:1.7rem; font-weight:800;
    margin:0 0 .3rem 0; letter-spacing:-0.02em; display:flex; align-items:center; gap:.6rem; }
.main-header p { color:#94a3b8; font-size:.88rem; margin:.25rem 0 0 0; }
.header-meta { margin-top:.9rem; display:flex; flex-wrap:wrap; gap:.45rem; align-items:center;
    font-family:'JetBrains Mono', monospace; font-size:.72rem; color:#94a3b8; }
.header-meta .dot { width:8px; height:8px; border-radius:50%; background:#22c55e;
    box-shadow:0 0 0 4px rgba(34,197,94,.18); animation: pulse 1.8s infinite; }
@keyframes pulse {
    0%,100% { box-shadow:0 0 0 4px rgba(34,197,94,.18); }
    50% { box-shadow:0 0 0 7px rgba(34,197,94,.04); }
}
.header-meta .sep { opacity:.4; }

.badge { display:inline-block; padding:3px 11px; border-radius:999px;
    font-size:.68rem; font-weight:700; letter-spacing:.08em; margin-right:6px;
    text-transform:uppercase; font-family:'JetBrains Mono', monospace; }
.badge-live   { background:rgba(34,197,94,.15); color:#86efac; border:1px solid rgba(34,197,94,.3); }
.badge-dbscan { background:rgba(59,130,246,.15); color:#bfdbfe; border:1px solid rgba(59,130,246,.3); }

[data-testid="metric-container"] {
    background: linear-gradient(180deg, rgba(30,41,59,.92), rgba(15,23,42,.92));
    border-radius:12px; padding:1rem 1.1rem;
    border:1px solid rgba(148,163,184,.14);
    box-shadow:0 4px 14px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.03);
    transition: transform .15s ease, border-color .15s ease;
}
[data-testid="metric-container"]:hover { transform:translateY(-1px); border-color:rgba(59,130,246,.35); }
[data-testid="metric-container"] label { color:#94a3b8 !important; font-size:.7rem !important;
    font-weight:600 !important; letter-spacing:.08em; text-transform:uppercase; }
[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-size:1.7rem !important; font-weight:700; color:#f1f5f9 !important;
    font-family:'JetBrains Mono', monospace; letter-spacing:-.01em;
}

.section-header { color:#f1f5f9; font-size:1.05rem; font-weight:700;
    padding:.35rem 0 .55rem 0; margin:0 0 1rem 0;
    border-bottom:1px solid rgba(148,163,184,.14);
    display:flex; align-items:center; gap:.55rem; letter-spacing:-.01em; }
.section-header::before { content:""; width:4px; height:18px; border-radius:2px;
    background: linear-gradient(180deg,#3b82f6,#1d4ed8);
    box-shadow:0 0 12px rgba(59,130,246,.5); }

.info-box { background: linear-gradient(90deg, rgba(30,64,175,.18), rgba(30,64,175,.05));
    border-left:3px solid #3b82f6; border-radius:8px;
    padding:.75rem 1rem; color:#cfe1ff; font-size:.82rem;
    margin-bottom:.9rem; line-height:1.55; }
.warn-box { background: linear-gradient(90deg, rgba(180,83,9,.2), rgba(180,83,9,.05));
    border-left:3px solid #f59e0b; border-radius:8px;
    padding:.75rem 1rem; color:#fde68a; font-size:.82rem;
    margin-bottom:.9rem; line-height:1.55; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1020 0%, #0d1426 100%);
    border-right: 1px solid rgba(148,163,184,.1);
}
[data-testid="stSidebar"] .block-container { padding-top:1.2rem; }
.sidebar-title { font-family:'JetBrains Mono', monospace; font-size:.7rem; font-weight:700;
    color:#60a5fa; letter-spacing:.16em; margin-bottom:.4rem; text-transform:uppercase; }
.sidebar-sub { color:#cbd5e1; font-size:.95rem; font-weight:600;
    margin-bottom:1rem; display:flex; align-items:center; gap:.4rem; }

.cluster-card { background: linear-gradient(180deg, rgba(30,41,59,.95), rgba(15,23,42,.95));
    border-radius:12px; padding:1rem 1.05rem;
    border:1px solid rgba(148,163,184,.16);
    box-shadow:0 4px 14px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.03);
    margin-top:.4rem; }
.cluster-card .card-head { display:flex; justify-content:space-between; align-items:center;
    font-family:'JetBrains Mono', monospace; font-size:.7rem; color:#60a5fa;
    letter-spacing:.12em; margin-bottom:.75rem;
    padding-bottom:.55rem; border-bottom:1px dashed rgba(148,163,184,.18); }
.cluster-card .card-head .live { width:7px; height:7px; border-radius:50%; background:#22c55e;
    box-shadow:0 0 0 3px rgba(34,197,94,.2); }
.cluster-card .row { display:flex; justify-content:space-between; align-items:center;
    padding:.32rem 0; border-bottom:1px dotted rgba(148,163,184,.08); }
.cluster-card .row:last-child { border-bottom:none; }
.cluster-card .label { color:#94a3b8; font-size:.76rem; letter-spacing:.02em; }
.cluster-card .val   { font-weight:600; font-size:.82rem; color:#f1f5f9;
    font-family:'JetBrains Mono', monospace; }

[data-baseweb="tab-list"] { gap:4px; background:rgba(15,23,42,.6);
    padding:5px; border-radius:12px; border:1px solid rgba(148,163,184,.12);
    margin-bottom:1rem; }
[data-baseweb="tab"] { border-radius:8px !important; padding:.55rem 1.05rem !important;
    font-size:.82rem !important; font-weight:500 !important;
    color:#94a3b8 !important; background:transparent !important;
    transition: all .15s ease; }
[data-baseweb="tab"]:hover { color:#e2e8f0 !important; background:rgba(59,130,246,.08) !important; }
[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(180deg, rgba(59,130,246,.22), rgba(59,130,246,.12)) !important;
    color:#f0f9ff !important; font-weight:600 !important;
    box-shadow: inset 0 -2px 0 #3b82f6;
}
[data-baseweb="tab-highlight"] { display:none !important; }

[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div {
    background-color: rgba(15,23,42,.85) !important;
    border-color: rgba(148,163,184,.2) !important;
    border-radius: 8px !important;
}

[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden;
    border:1px solid rgba(148,163,184,.12); }

#MainMenu, footer, header[data-testid="stHeader"] { visibility:hidden; height:0; }
</style>
""", unsafe_allow_html=True)

# ── Guard: check all required CSVs ────────────────────────────────────────────
REQUIRED = [
    HOTSPOT_CLUSTERED,
    HOTSPOT_CLUSTER_STATS,
    TRAJECTORY_ANALYSIS,
    ANOMALY_ANALYSIS,
    PATROL_ACTION_PLAN,
    BEST_PATROL_WINDOWS_DETAILED,
]
missing = [path for path in REQUIRED if not path.exists()]
if missing:
    st.error(
        f"**Missing data files:** `{'`, `'.join(path.name for path in missing)}`\n\n"
        f"Run the pipeline scripts to generate all required CSVs from `{RAW_DATA.as_posix()}` into `outputs/`."
    )
    st.stop()


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading datasets…")
def load_data():
    hotspots = pd.read_csv(HOTSPOT_CLUSTERED, low_memory=False)
    hotspots = hotspots[hotspots["cluster_id"] >= 0].copy()
    hotspots["created_datetime"] = parse_local_naive_series(
        hotspots["created_datetime"],
        "created_datetime",
    )
    hotspots["year_month"] = hotspots["created_datetime"].dt.to_period("M").astype(str)

    cluster_stats = pd.read_csv(HOTSPOT_CLUSTER_STATS)
    cluster_stats["cluster_id"] = cluster_stats["cluster_id"].astype(float)

    trajectory = pd.read_csv(TRAJECTORY_ANALYSIS)
    trajectory["cluster_id"] = trajectory["cluster_id"].astype(float)

    anomalies = pd.read_csv(ANOMALY_ANALYSIS)
    anomalies["cluster_id"] = anomalies["cluster_id"].astype(float)

    patrol_plan = pd.read_csv(PATROL_ACTION_PLAN)
    patrol_plan["cluster_id"] = patrol_plan["cluster_id"].astype(float)

    top_windows = pd.read_csv(BEST_PATROL_WINDOWS_DETAILED)
    top_windows["cluster_id"] = top_windows["cluster_id"].astype(float)

    windows_summary = pd.read_csv(BEST_PATROL_WINDOWS)
    windows_summary["cluster_id"] = windows_summary["cluster_id"].astype(float)

    top_zones = pd.read_csv(TOP_ENFORCEMENT_ZONES)
    top_zones["cluster_id"] = top_zones["cluster_id"].astype(float)

    patrol_cols = ["cluster_id", "patrol_priority", "recommended_windows", "deployment_level"]
    available_patrol = [c for c in patrol_cols if c in patrol_plan.columns]

    cluster_master = (
        cluster_stats
        .merge(trajectory, on="cluster_id", how="left")
        .merge(anomalies,  on="cluster_id", how="left")
        .merge(patrol_plan[available_patrol], on="cluster_id", how="left")
    )

    monthly_trends = (
        hotspots.groupby(["cluster_id", "year_month"]).size().reset_index(name="violations")
    )

    risk_counts = (
        cluster_master["risk_level"].value_counts()
        .reindex(RISK_ORDER, fill_value=0)
        .rename_axis("risk_level").reset_index(name="hotspots")
    )

    watchlist = (
        cluster_master[cluster_master["anomaly_status"] == "Abnormal Surge"]
        .sort_values(["z_score", "patrol_priority"], ascending=[False, False]).copy()
    )

    hour_dist  = hotspots.groupby("hour")["cluster_id"].count().reset_index(name="violations")
    dow_dist   = hotspots.groupby("day_of_week")["cluster_id"].count().reset_index(name="violations")
    month_dist = hotspots.groupby("year_month")["cluster_id"].count().reset_index(name="violations")

    return (
        hotspots, cluster_master, monthly_trends, risk_counts,
        watchlist, patrol_plan, top_windows, windows_summary, top_zones,
        hour_dist, dow_dist, month_dist,
    )


(
    hotspots, cluster_master, monthly_trends, risk_counts,
    watchlist, patrol_plan, top_windows, windows_summary, top_zones,
    hour_dist, dow_dist, month_dist,
) = load_data()


# ── Helpers ───────────────────────────────────────────────────────────────────
def kpi(label, value, delta=None, help_txt=None):
    st.metric(label, value, delta=delta, help=help_txt)


# Pre-computed normalization anchors (fixed to original data, not re-computed on simulation)
_LOG_MIN = None
_LOG_MAX = None
_QCUT_BINS = None   # [q50, q80, q95] from original priority_score


def _build_sim_anchors(df: pd.DataFrame):
    """Call once after load_data to fix normalization references."""
    global _LOG_MIN, _LOG_MAX, _QCUT_BINS
    _LOG_MIN = float(np.log1p(df["violations"]).min())
    _LOG_MAX = float(np.log1p(df["violations"]).max())
    # Match the exact qcut bins used in notebook 02
    _QCUT_BINS = df["priority_score"].quantile([0.50, 0.80, 0.95]).tolist()


def priority_components_after_reduction(df: pd.DataFrame, pct: float) -> pd.DataFrame:
    """
    Simulate what happens when enforcement reduces violations by `pct`%
    across ALL clusters (not just Critical), using FIXED normalization bounds
    from the original data so the effect is not diluted by re-scaling.
    """
    sim = df.copy()
    factor = 1 - pct / 100.0

    # Apply reduction to every cluster
    sim["sim_violations"] = (sim["violations"] * factor).clip(lower=0)
    sim["sim_log_v"] = np.log1p(sim["sim_violations"])

    # Use FIXED min/max from original data — prevents renormalization dilution
    lo = _LOG_MIN if _LOG_MIN is not None else float(sim["sim_log_v"].min())
    hi = _LOG_MAX if _LOG_MAX is not None else float(sim["sim_log_v"].max())
    sim["sim_log_v_norm"] = ((sim["sim_log_v"] - lo) / (hi - lo)).clip(0, 1) if hi > lo else 0.0

    sim["sim_priority_score"] = (
        0.45 * sim["sim_log_v_norm"]
        + 0.20 * sim["avg_severity_norm"]
        + 0.10 * sim["avg_vehicle_weight_norm"]
        + 0.10 * sim["avg_junction_weight_norm"]
        + 0.10 * sim["active_months_norm"]
        + 0.05 * sim["avg_num_violations_norm"]
    )

    # Use the SAME qcut thresholds as notebook 02 (q50 / q80 / q95)
    if _QCUT_BINS is not None:
        q50, q80, q95 = _QCUT_BINS
    else:
        q50, q80, q95 = df["priority_score"].quantile([0.50, 0.80, 0.95]).tolist()

    sim["sim_risk_level"] = pd.cut(
        sim["sim_priority_score"],
        bins=[-np.inf, q50, q80, q95, np.inf],
        labels=["Low", "Medium", "High", "Critical"],
        include_lowest=True,
    )
    sim["score_delta"] = sim["sim_priority_score"] - sim["priority_score"]
    return sim


# Build fixed normalization anchors for the Impact Simulator
_build_sim_anchors(cluster_master)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">◢ OPERATIONS CONTROL</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">🔍 Intelligence Filters</div>', unsafe_allow_html=True)

    stations = ["All"] + sorted(cluster_master["top_station"].dropna().unique().tolist())
    sel_station = st.selectbox("Police Station", stations)

    sel_risks = st.multiselect("Risk Levels", RISK_ORDER, default=RISK_ORDER)

    sel_traj = st.multiselect(
        "Trajectory",
        ["Escalating", "Stable", "Declining", "Insufficient History"],
        default=["Escalating", "Stable", "Declining", "Insufficient History"],
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-title">◢ FOCUS TARGET</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">🎯 Focus Cluster</div>', unsafe_allow_html=True)

    fc = cluster_master.copy()
    if sel_station != "All":
        fc = fc[fc["top_station"] == sel_station]
    fc = fc[fc["risk_level"].isin(sel_risks)]
    if sel_traj:
        fc = fc[fc["trajectory"].isin(sel_traj)]
    filtered_clusters = fc

    choices = (
        fc.sort_values("patrol_priority", ascending=False, na_position="last")["cluster_id"]
        .astype(int).tolist()
    )
    if not choices:
        choices = cluster_master["cluster_id"].astype(int).tolist()

    sel_cluster = st.selectbox("Cluster ID", choices)
    focus_row = cluster_master[
        cluster_master["cluster_id"].astype(int) == int(sel_cluster)
    ].iloc[0]

    rc = RISK_COLORS.get(str(focus_row.get("risk_level", "")), "#7FB069")
    tc = TRAJECTORY_COLORS.get(str(focus_row.get("trajectory", "")), "#9CA3AF")
    pp = focus_row.get("patrol_priority", float("nan"))
    pp_str = f"{pp:.3f}" if pd.notna(pp) else "–"

    st.markdown(f"""
    <div class="cluster-card">
      <div class="card-head">
        <span>CLUSTER · {int(sel_cluster):04d}</span>
        <span class="live"></span>
      </div>
      <div class="row"><span class="label">Risk Level</span>
        <span class="val" style="color:{rc};">● {focus_row.get("risk_level","–")}</span></div>
      <div class="row"><span class="label">Trajectory</span>
        <span class="val" style="color:{tc};">▲ {focus_row.get("trajectory","–")}</span></div>
      <div class="row"><span class="label">Anomaly</span>
        <span class="val">{focus_row.get("anomaly_status","–")}</span></div>
      <div class="row"><span class="label">Violations</span>
        <span class="val">{int(focus_row.get("violations",0)):,}</span></div>
      <div class="row"><span class="label">Patrol Priority</span>
        <span class="val">{pp_str}</span></div>
      <div class="row"><span class="label">Station</span>
        <span class="val" style="font-size:.74rem;">{focus_row.get("top_station","–")}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        f"**{len(filtered_clusters):,}** clusters shown  ·  "
        f"**{len(cluster_master):,}** total"
    )
    st.caption(f"🛰️ Analytics ready · {datetime.now().strftime('%H:%M:%S')} local")


# ── Header banner ─────────────────────────────────────────────────────────────
n_clusters = int(cluster_master["cluster_id"].max()) + 1
st.markdown(f"""
<div class="main-header">
  <h1>🚔 Gridlock Patrol Intelligence</h1>
  <p>Data-driven parking enforcement intelligence— spatial hotspots, trajectory analysis,
     anomaly detection, and patrol deployment.</p>
  <div class="header-meta">
    <span class="dot"></span> SYSTEM ONLINE
    <span class="sep">│</span> <span class="badge badge-live">PIPELINE OUTPUT</span>
    <span class="badge badge-dbscan">DBSCAN · {n_clusters} HOTSPOTS</span>
    <span class="sep">│</span> SECTOR: BANGALORE
    <span class="sep">│</span> {datetime.now().strftime('%a %d %b %Y · %H:%M')}
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Operational Snapshot</div>', unsafe_allow_html=True)
col_kpi = st.columns(5)
with col_kpi[0]: kpi("Active Hotspots", f"{len(filtered_clusters):,}")
with col_kpi[1]: kpi("Critical Hotspots", f"{(filtered_clusters['risk_level']=='Critical').sum():,}")
with col_kpi[2]: kpi("Escalating", f"{(filtered_clusters['trajectory']=='Escalating').sum():,}")
with col_kpi[3]: kpi("Abnormal Surges", f"{(filtered_clusters['anomaly_status']=='Abnormal Surge').sum():,}")
with col_kpi[4]:
    imm = filtered_clusters["deployment_level"].eq("Immediate Enforcement").sum()
    kpi("Immediate Enforcement", f"{imm:,}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
(
    tab_map, tab_risk, tab_trend, tab_temporal,
    tab_watch, tab_patrol, tab_impact, tab_about,
) = st.tabs([
    "🗺️ Hotspot Map", "⚠️ Risk Zones", "📈 Trajectory",
    "⏰ Temporal Patterns", "🔴 Watchlist", "🚓 Patrol Plan",
    "🎯 Impact Simulator", "ℹ️ About",
])


# ── TAB 1: HOTSPOT MAP ────────────────────────────────────────────────────────
with tab_map:
    st.markdown('<div class="section-header">DBSCAN Hotspot Map</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">506 hotspots identified with DBSCAN '
        '(eps ≈ 61 m, min_samples = 20, haversine + ball_tree). '
        'Bubble size = violation count. Colour = risk level.</div>',
        unsafe_allow_html=True,
    )

    map_df = filtered_clusters.dropna(subset=["centroid_lat", "centroid_lon"]).copy()
    map_df["cluster_label"] = map_df["cluster_id"].astype(int).astype(str)

    c_map, c_ctrl = st.columns([4, 1])
    with c_ctrl:
        st.markdown("**🛰️ Map Console**")
        map_style = st.selectbox(
            "Map style",
            ["carto-darkmatter", "open-street-map", "carto-positron"],
            index=0,
        )
        show_tbl = st.checkbox("Show table below map", value=True)
        st.markdown("---")
        st.caption("**Legend**")
        for lvl in RISK_ORDER:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:.5rem;font-size:.8rem;color:#cbd5e1;margin:.15rem 0;'>"
                f"<span style='width:10px;height:10px;border-radius:50%;background:{RISK_COLORS[lvl]};"
                f"box-shadow:0 0 8px {RISK_COLORS[lvl]}80;'></span>{lvl}</div>",
                unsafe_allow_html=True,
            )

    with c_map:
        fig_map = px.scatter_mapbox(
            map_df,
            lat="centroid_lat", lon="centroid_lon",
            color="risk_level", size="violations",
            hover_name="cluster_label",
            hover_data={
                "top_station": True, "trajectory": True, "anomaly_status": True,
                "patrol_priority": ":.3f", "violations": ":,",
                "priority_score": ":.3f",
                "centroid_lat": False, "centroid_lon": False,
            },
            color_discrete_map=RISK_COLORS,
            category_orders={"risk_level": RISK_ORDER},
            size_max=30, zoom=10, height=620,
        )
        fig_map.update_layout(
            mapbox_style=map_style, margin=dict(l=0, r=0, t=0, b=0),
            legend_title_text="Risk Zone",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#cbd5e1"),
        )
        st.plotly_chart(fig_map, width='stretch')

    if show_tbl:
        with st.expander("📋 Filtered hotspot table", expanded=True):
            show_cols = [c for c in [
                "cluster_id","top_station","risk_level","trajectory",
                "anomaly_status","priority_score","patrol_priority","deployment_level",
            ] if c in map_df.columns]
            st.dataframe(
                map_df[show_cols]
                .sort_values("patrol_priority", ascending=False, na_position="last")
                .reset_index(drop=True),
                width='stretch', hide_index=True,
            )


# ── TAB 2: RISK ZONES ─────────────────────────────────────────────────────────
with tab_risk:
    st.markdown('<div class="section-header">Risk Zone Distribution</div>', unsafe_allow_html=True)

    left, right = st.columns([1.3, 1])

    with left:
        risk_fig = px.bar(
            risk_counts, x="risk_level", y="hotspots",
            color="risk_level", color_discrete_map=RISK_COLORS,
            category_orders={"risk_level": RISK_ORDER},
            text="hotspots", height=360, template="plotly_dark",
            title="Hotspot Count by Risk Level",
        )
        risk_fig.update_traces(textposition="outside", textfont=dict(color="#e2e8f0"))
        risk_fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Hotspot Count",
                               margin=dict(t=50, b=0, l=0, r=0))
        st.plotly_chart(style_fig(risk_fig), width='stretch')

        traj_risk = (
            filtered_clusters.groupby(["risk_level", "trajectory"]).size()
            .reset_index(name="count")
        )
        traj_fig = px.bar(
            traj_risk, x="risk_level", y="count", color="trajectory",
            color_discrete_map=TRAJECTORY_COLORS,
            category_orders={"risk_level": RISK_ORDER},
            barmode="stack", height=300, template="plotly_dark",
            title="Trajectory Mix per Risk Level",
        )
        traj_fig.update_layout(margin=dict(t=50, b=0, l=0, r=0))
        st.plotly_chart(style_fig(traj_fig), width='stretch')

    with right:
        st.markdown("**🏆 Top 15 Hotspots by Priority**")
        show_cols = [c for c in [
            "cluster_id","top_station","violations","priority_score","risk_level","trajectory",
        ] if c in filtered_clusters.columns]
        st.dataframe(
            filtered_clusters[show_cols]
            .sort_values("priority_score", ascending=False)
            .head(15).reset_index(drop=True),
            width='stretch', hide_index=True,
        )

        dep_counts = filtered_clusters["deployment_level"].value_counts().reset_index()
        dep_counts.columns = ["deployment_level", "count"]
        if len(dep_counts):
            dep_fig = px.pie(
                dep_counts, names="deployment_level", values="count",
                color="deployment_level", color_discrete_map=DEPLOYMENT_COLORS,
                height=300, template="plotly_dark",
                title="Deployment Level Breakdown", hole=0.55,
            )
            dep_fig.update_traces(textfont=dict(color="#f1f5f9", size=12))
            dep_fig.update_layout(margin=dict(t=50, b=0, l=0, r=0))
            st.plotly_chart(style_fig(dep_fig, transparent_plot=True), width='stretch')


# ── TAB 3: TRAJECTORY ─────────────────────────────────────────────────────────
with tab_trend:
    st.markdown('<div class="section-header">Hotspot Trajectory Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Trajectory determined by fitting LinearRegression on monthly '
        'violation counts per cluster. '
        'Escalating = slope ≥ 75th pct · Declining = slope ≤ 25th pct · '
        'R² reflects how well the linear trend fits. No XGBoost — no data leakage.</div>',
        unsafe_allow_html=True,
    )

    focus = cluster_master[
        cluster_master["cluster_id"].astype(int) == int(sel_cluster)
    ].iloc[0]

    trend_df = (
        monthly_trends[monthly_trends["cluster_id"].astype(int) == int(sel_cluster)]
        .sort_values("year_month").copy()
    )

    kpi_cols = st.columns(5)
    with kpi_cols[0]: kpi("Trajectory", str(focus.get("trajectory","–")))
    with kpi_cols[1]:
        slope = focus.get("slope", float("nan"))
        kpi("Slope", f"{slope:.2f}" if pd.notna(slope) else "N/A")
    with kpi_cols[2]:
        r2 = focus.get("r2", float("nan"))
        kpi("R² Confidence", f"{r2:.2f}" if pd.notna(r2) else "N/A")
    with kpi_cols[3]: kpi("Anomaly", str(focus.get("anomaly_status","–")))
    with kpi_cols[4]:
        cm = focus.get("current_month", float("nan"))
        kpi("Current Month Violations", f"{int(cm):,}" if pd.notna(cm) else "N/A")

    st.markdown(f"##### Trajectory Profile · Cluster {int(sel_cluster)}")
    tc = TRAJECTORY_COLORS.get(str(focus.get("trajectory","")), "#3B82F6")
    trend_fig = go.Figure()

    if len(trend_df) > 1 and pd.notna(slope):
        xi = list(range(len(trend_df)))
        intercept = float(trend_df["violations"].iloc[0]) - slope * xi[0]
        trend_fig.add_trace(go.Scatter(
            x=trend_df["year_month"],
            y=[slope * x + intercept for x in xi],
            mode="lines", line=dict(color=tc, width=1.5, dash="dot"),
            name="Linear trend", opacity=0.6,
        ))

    trend_fig.add_trace(go.Scatter(
        x=trend_df["year_month"], y=trend_df["violations"],
        mode="lines+markers",
        line=dict(color=tc, width=3),
        marker=dict(size=10, color=tc, line=dict(width=2, color="#0f172a")),
        name="Monthly Violations",
        fill="tozeroy",
        fillcolor=f"rgba({int(tc[1:3],16)},{int(tc[3:5],16)},{int(tc[5:7],16)},0.10)",
    ))
    trend_fig.update_layout(
        height=380, template="plotly_dark",
        xaxis_title="Month", yaxis_title="Violations",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(style_fig(trend_fig), width='stretch')

    st.markdown("**📊 Slope Distribution Across All 506 Hotspots**")
    slope_fig = px.histogram(
        cluster_master.dropna(subset=["slope"]),
        x="slope", color="trajectory", nbins=60,
        color_discrete_map=TRAJECTORY_COLORS,
        height=280, template="plotly_dark",
        labels={"slope": "Linear Slope (violations/month)"},
    )
    slope_fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(style_fig(slope_fig, transparent_plot=True), width='stretch')

    with st.expander("📋 Monthly trend data — current cluster"):
        st.dataframe(trend_df.rename(columns={"year_month":"month"}),
                     width='stretch', hide_index=True)


# ── TAB 4: TEMPORAL PATTERNS ──────────────────────────────────────────────────
with tab_temporal:
    st.markdown('<div class="section-header">Temporal Enforcement Patterns</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="warn-box">⚠️ <b>Detection Hour Reframe:</b> Timestamps reflect when '
        'violations were recorded, not necessarily when illegal parking began. Recorded '
        'parking-violation detections are concentrated during the morning, particularly '
        'around 8 AM–12 PM, with the highest counts around 10–11 AM. These patterns '
        'describe observed detection activity and should not be interpreted as the exact '
        'time when parking violations began. Dataset covers Nov 2023 – Apr 2024 '
        '(winter–spring window only).</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        h_fig = px.bar(
            hour_dist, x="hour", y="violations",
            color="violations", color_continuous_scale="Blues",
            height=340, template="plotly_dark",
            title="Detections by Hour of Day",
            labels={"hour": "Hour (0–23)", "violations": "Count"},
        )
        h_fig.update_layout(coloraxis_showscale=False, margin=dict(t=50,b=0,l=0,r=0))
        st.plotly_chart(style_fig(h_fig, transparent_plot=True), width='stretch')

    with c2:
        DOW_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_sorted = dow_dist.copy()
        dow_sorted["day_of_week"] = pd.Categorical(
            dow_sorted["day_of_week"], categories=DOW_ORDER, ordered=True
        )
        dow_sorted = dow_sorted.sort_values("day_of_week")
        d_fig = px.bar(
            dow_sorted, x="day_of_week", y="violations",
            color="violations", color_continuous_scale="Blues",
            height=340, template="plotly_dark",
            title="Detections by Day of Week",
            labels={"day_of_week":"", "violations":"Count"},
        )
        d_fig.update_layout(coloraxis_showscale=False, margin=dict(t=50,b=0,l=0,r=0))
        st.plotly_chart(style_fig(d_fig, transparent_plot=True), width='stretch')

    m_fig = px.area(
        month_dist.sort_values("year_month"),
        x="year_month", y="violations",
        height=280, template="plotly_dark",
        title="Monthly Detection Volume",
        labels={"year_month":"Month","violations":"Count"},
        color_discrete_sequence=["#3b82f6"],
    )
    m_fig.update_traces(line_color="#3b82f6", line_width=2.5,
                        fillcolor="rgba(59,130,246,0.12)")
    m_fig.update_layout(margin=dict(t=50,b=0,l=0,r=0))
    st.plotly_chart(style_fig(m_fig), width='stretch')

    st.markdown("**🌡️ Enforcement Heatmap — Hour × Day**")
    hm = (
        hotspots.groupby(["day_of_week","hour"])["cluster_id"]
        .count().reset_index(name="count")
    )
    hm["day_of_week"] = pd.Categorical(hm["day_of_week"], categories=DOW_ORDER, ordered=True)
    hm_piv = hm.pivot(index="day_of_week", columns="hour", values="count").fillna(0)
    hm_fig = go.Figure(go.Heatmap(
        z=hm_piv.values,
        x=[str(h) for h in hm_piv.columns],
        y=[str(d) for d in hm_piv.index],
        colorscale="Blues", hoverongaps=False,
        colorbar=dict(thickness=10, tickfont=dict(color="#cbd5e1", size=10)),
    ))
    hm_fig.update_layout(
        height=320, template="plotly_dark",
        xaxis_title="Hour of Day", yaxis_title="",
        margin=dict(t=10,b=0,l=0,r=0),
    )
    st.plotly_chart(style_fig(hm_fig, transparent_plot=True), width='stretch')


# ── TAB 5: WATCHLIST ──────────────────────────────────────────────────────────
with tab_watch:
    st.markdown('<div class="section-header">Watchlist: Abnormal Surges</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Clusters where the latest month materially exceeded '
        'historical baseline. Z-score threshold = 1.5 σ. '
        'Historical mean and std computed from all prior months per cluster.</div>',
        unsafe_allow_html=True,
    )

    w_cols = st.columns(4)
    with w_cols[0]: kpi("Watchlist Size", f"{len(watchlist):,}")
    with w_cols[1]:
        mz = watchlist["z_score"].max() if len(watchlist) else 0
        kpi("Max Z-Score", f"{mz:.2f}")
    with w_cols[2]:
        kpi("Critical In Watchlist", f"{(watchlist['risk_level']=='Critical').sum():,}")
    with w_cols[3]:
        cm_avg = watchlist["current_month"].mean() if len(watchlist) else 0
        kpi("Avg Current Month Violations", f"{cm_avg:.0f}")

    if len(watchlist) == 0:
        st.info("No abnormal surges detected for the current filter selection.")
    else:
        w_scatter = px.scatter(
            watchlist.reset_index(drop=True),
            x="z_score", y="patrol_priority",
            color="risk_level", color_discrete_map=RISK_COLORS,
            size="violations",
            hover_name=watchlist["cluster_id"].astype(int).astype(str).values,
            hover_data={
                "top_station": True, "trajectory": True,
                "z_score": ":.2f", "current_month": True,
            },
            height=340, template="plotly_dark",
            title="Surge Intensity vs Patrol Priority",
            labels={"z_score": "Z-Score", "patrol_priority": "Patrol Priority"},
        )
        w_scatter.update_layout(margin=dict(t=50,b=0,l=0,r=0))
        st.plotly_chart(style_fig(w_scatter), width='stretch')

        with st.expander("📋 Full watchlist roster", expanded=True):
            show_w = [c for c in [
                "cluster_id","top_station","risk_level","trajectory",
                "z_score","mean","current_month","patrol_priority","deployment_level",
            ] if c in watchlist.columns]
            st.dataframe(watchlist[show_w].reset_index(drop=True),
                         width='stretch', hide_index=True)


# ── TAB 6: PATROL PLAN ────────────────────────────────────────────────────────
with tab_patrol:
    st.markdown('<div class="section-header">Patrol Recommendations</div>', unsafe_allow_html=True)

    plan_cols = [c for c in [
        "cluster_id","top_station","risk_level","trajectory","anomaly_status",
        "patrol_priority","deployment_level","recommended_windows",
    ] if c in patrol_plan.columns]
    top20_plan = patrol_plan[plan_cols].head(20)

    pl, pr = st.columns([1.2, 1])
    with pl:
        st.markdown("**🚓 Top 20 Enforcement Zones**")
        st.dataframe(top20_plan, width='stretch', hide_index=True)

        dep_data = top20_plan["deployment_level"].value_counts().reset_index()
        dep_data.columns = ["level","count"]
        dep_bar = px.bar(
            dep_data, x="level", y="count",
            color="level", color_discrete_map=DEPLOYMENT_COLORS,
            height=250, template="plotly_dark",
            title="Deployment Level — Top 20",
            text="count",
        )
        dep_bar.update_traces(textposition="outside", textfont=dict(color="#e2e8f0"))
        dep_bar.update_layout(showlegend=False, margin=dict(t=50,b=0,l=0,r=0))
        st.plotly_chart(style_fig(dep_bar, transparent_plot=True), width='stretch')

    with pr:
        st.markdown(f"**⏱️ Patrol Windows — Cluster {int(sel_cluster)}**")

        focus_win = top_windows[
            top_windows["cluster_id"].astype(int) == int(sel_cluster)
        ].copy()

        if len(focus_win) == 0:
            st.info(
                "This cluster is not in the top-20 patrol list. "
                "Select a higher-priority cluster to see its patrol windows."
            )
        else:
            win_cols = [c for c in [
                "day_of_week","hour","patrol_window","window_violations",
                "hour_percentage","window_score","window_rank",
            ] if c in focus_win.columns]
            focus_win["hour_share_pct"] = (
                100 * focus_win["hour_percentage"]
            ).round(1).astype(str) + "%"
            st.dataframe(
                focus_win.sort_values("window_score", ascending=False)[win_cols].head(10),
                width='stretch', hide_index=True,
            )

            if "patrol_window" in focus_win.columns:
                wf = px.bar(
                    focus_win.sort_values("window_score", ascending=False).head(10),
                    x="patrol_window", y="window_score",
                    color="window_score", color_continuous_scale="Blues",
                    height=320, template="plotly_dark",
                    title="Window Score Ranking",
                )
                wf.update_layout(
                    coloraxis_showscale=False, xaxis_tickangle=-30,
                    margin=dict(t=50,b=80,l=0,r=0),
                )
                st.plotly_chart(style_fig(wf, transparent_plot=True), width='stretch')

    with st.expander("📅 All Top-20 Cluster Windows", expanded=False):
        sum_cols = [c for c in [
            "cluster_id","top_station","risk_level","patrol_priority","recommended_windows",
        ] if c in windows_summary.columns]
        st.dataframe(windows_summary[sum_cols], width='stretch', hide_index=True)


# ── TAB 7: IMPACT SIMULATOR ───────────────────────────────────────────────────
with tab_impact:
    st.markdown('<div class="section-header">Impact Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box"><b>No separate ML model needed.</b> The simulator uses the '
        'same priority score formula from Notebook 02 (Hotspot Engineering): '
        '<code>0.45×log_violations + 0.20×severity + 0.10×vehicle + 0.10×junction + '
        '0.10×persistence + 0.05×multi_violation</code>. '
        'Reducing patrol violations shifts each cluster\'s score, potentially moving it to '
        'a lower risk tier. Fixed normalization bounds prevent dilution.</div>',
        unsafe_allow_html=True,
    )

    # Controls row
    ctrl_l, ctrl_r = st.columns([2, 1])
    with ctrl_l:
        reduction_pct = st.slider(
            "Simulated violation reduction across ALL hotspots (%)",
            0, 60, 20, step=5,
            help="Represents the expected reduction in violations if patrol enforcement "
                 "is applied proportionally across all hotspots.",
        )
    with ctrl_r:
        show_all_tiers = st.checkbox("Show all tier migrations", value=True)

    required_sim_cols = [
        "risk_level", "violations", "avg_severity_norm", "avg_vehicle_weight_norm",
        "avg_junction_weight_norm", "active_months_norm", "avg_num_violations_norm",
        "priority_score",
    ]
    has_sim_cols = all(c in cluster_master.columns for c in required_sim_cols)

    if not has_sim_cols:
        missing_cols = [c for c in required_sim_cols if c not in cluster_master.columns]
        st.warning(
            f"Missing columns from {HOTSPOT_CLUSTER_STATS.name}: {missing_cols}"
        )
    else:
        sim = priority_components_after_reduction(cluster_master, reduction_pct)

        # ── KPI row ────────────────────────────────────────────────────────────
        # Count tier changes for each original level
        tier_changes = {}
        for tier in RISK_ORDER:
            orig_in_tier = sim[sim["risk_level"] == tier]
            moved = (orig_in_tier["sim_risk_level"] != tier).sum()
            tier_changes[tier] = {"total": len(orig_in_tier), "moved": int(moved)}

        total_changed = sum(v["moved"] for v in tier_changes.values())
        crit_moved = tier_changes["Critical"]["moved"]
        crit_total = tier_changes["Critical"]["total"]
        avg_delta = sim["score_delta"].mean()

        ic = st.columns(5)
        with ic[0]: kpi("Clusters Simulated", f"{len(sim):,}")
        with ic[1]: kpi("Tier Downgrades",    f"{total_changed:,}")
        with ic[2]: kpi("Critical → Lower",   f"{crit_moved:,} / {crit_total}")
        with ic[3]:
            pct_r = 0 if crit_total == 0 else 100 * crit_moved / crit_total
            kpi("Critical Reduction %", f"{pct_r:.0f}%")
        with ic[4]: kpi("Avg Score Delta",    f"{avg_delta:+.4f}")

        st.markdown("<br>", unsafe_allow_html=True)
        il, ir = st.columns(2)

        # ── Left: Before vs After bar chart ────────────────────────────────────
        with il:
            before = (
                sim["risk_level"].value_counts()
                .reindex(RISK_ORDER, fill_value=0).reset_index()
            )
            before.columns = ["risk_level", "count"]
            before["scenario"] = "Before enforcement"
            after = (
                sim["sim_risk_level"].value_counts()
                .reindex(RISK_ORDER, fill_value=0).reset_index()
            )
            after.columns = ["risk_level", "count"]
            after["scenario"] = f"After {reduction_pct}% reduction"
            cmp = pd.concat([before, after], ignore_index=True)
            cmp_fig = px.bar(
                cmp, x="risk_level", y="count", color="scenario",
                barmode="group",
                category_orders={"risk_level": RISK_ORDER},
                height=380, template="plotly_dark",
                title="Risk Tier Distribution — Before vs After",
                color_discrete_sequence=["#475569", "#3b82f6"],
                text="count",
            )
            cmp_fig.update_traces(textposition="outside", textfont=dict(color="#e2e8f0"))
            cmp_fig.update_layout(
                legend_title_text="",
                margin=dict(t=50, b=0, l=0, r=0),
            )
            st.plotly_chart(style_fig(cmp_fig, transparent_plot=True), width='stretch')

        # ── Right: Score delta scatter (all clusters) ──────────────────────────
        with ir:
            delta_fig = px.scatter(
                sim.dropna(subset=["score_delta"]),
                x="priority_score", y="score_delta",
                color="risk_level", color_discrete_map=RISK_COLORS,
                symbol="sim_risk_level",
                size="violations",
                hover_name=sim["cluster_id"].astype(int).astype(str).values,
                hover_data={"top_station": True, "priority_score": ":.4f",
                            "sim_priority_score": ":.4f", "score_delta": ":.5f"},
                height=380, template="plotly_dark",
                title="Score Delta per Cluster (original score → how much it dropped)",
                labels={"priority_score": "Original Priority Score",
                        "score_delta": "Score Change (sim − original)"},
            )
            delta_fig.add_hline(y=0, line_dash="dash",
                                line_color="rgba(148,163,184,0.4)", line_width=1)
            delta_fig.update_layout(
                margin=dict(t=50, b=0, l=0, r=0),
                legend_title_text="Original Tier",
            )
            st.plotly_chart(style_fig(delta_fig), width='stretch')

        # ── Tier migration table ────────────────────────────────────────────────
        st.markdown("**Tier Migration Breakdown**")
        migration_rows = []
        tiers_to_show = RISK_ORDER if show_all_tiers else ["Critical", "High"]
        for tier in tiers_to_show:
            grp = sim[sim["risk_level"] == tier]
            for dest_tier in RISK_ORDER:
                cnt = (grp["sim_risk_level"] == dest_tier).sum()
                if cnt > 0:
                    arrow = "→" if tier == dest_tier else "↘"
                    migration_rows.append({
                        "From": tier,
                        "To": dest_tier,
                        "Clusters": int(cnt),
                        "Status": "Unchanged" if tier == dest_tier else "Downgraded",
                    })
        if migration_rows:
            mig_df = pd.DataFrame(migration_rows)
            st.dataframe(mig_df, width='stretch', hide_index=True)

        # ── Detail table for clusters that actually changed ─────────────────────
        changed = sim[sim["risk_level"] != sim["sim_risk_level"]].copy()
        if len(changed) > 0:
            with st.expander(
                f"📋 {len(changed)} clusters that changed tier — detail view",
                expanded=True,
            ):
                detail_cols = [c for c in [
                    "cluster_id", "top_station", "violations", "sim_violations",
                    "priority_score", "sim_priority_score", "score_delta",
                    "risk_level", "sim_risk_level",
                ] if c in changed.columns]
                st.dataframe(
                    changed[detail_cols].sort_values("score_delta"),
                    width='stretch', hide_index=True,
                )
        else:
            st.info(
                f"At {reduction_pct}% violation reduction, no clusters change tier. "
                "The priority score formula weights log(violations) at 45%, so violations "
                "must drop significantly to push a cluster below its tier boundary. "
                "Try increasing the slider to 40–60%."
            )


# ── TAB 8: ABOUT ──────────────────────────────────────────────────────────────
with tab_about:
    st.markdown('<div class="section-header">About this System</div>', unsafe_allow_html=True)

    n_crit    = (cluster_master["risk_level"] == "Critical").sum()
    n_esc     = (cluster_master["trajectory"] == "Escalating").sum()
    n_surges  = (cluster_master["anomaly_status"] == "Abnormal Surge").sum()
    n_imm     = cluster_master["deployment_level"].eq("Immediate Enforcement").sum()

    st.markdown(f"""
**GridLock Patrol Intelligence** is a fully data-driven AI parking enforcement platform
built on Bangalore's parking violation dataset.

#### Pipeline — Module Summary

| Module | Method | Output |
|---|---|---|
| **0 – Data Prep** | Filter approved + NaN; parse violation lists; engineer severity / vehicle / junction weights | 240,654 cleaned records |
| **1 – Spatial Hotspot Detection** | DBSCAN (eps ≈ 61 m, haversine, ball_tree, min_samples = 20) | **{n_clusters} hotspot clusters**, 6.2% noise |
| **2 – Temporal Intelligence** | Hour / day / month distributions, framed as *detection* hours | Temporal profiles per cluster |
| **3 – Risk Engine** | `0.45 × log_violations + 0.20 × severity + 0.10 × vehicle + 0.10 × junction + 0.10 × persistence + 0.05 × multi_violation` | priority_score → Low / Medium / High / Critical |
| **4 – Trajectory Analysis** | LinearRegression on monthly counts; slope percentile thresholds (p25 / p75) | Escalating / Stable / Declining |
| **5 – Anomaly Detection** | Z-score vs per-cluster historical mean/std; threshold = 1.5 σ | {n_surges} Abnormal Surges |
| **6 – Patrol Allocation** | `0.50 × priority_norm + 0.30 × trajectory_weight + 0.20 × anomaly_weight` | Top-20 patrol zones + deployment level |
| **7 – Dashboard** | This app | — |

#### Key Design Decisions
- **Haversine DBSCAN** — lat/lon degrees are not isotropic; Euclidean was wrong.
- **LinearRegression for trajectory** — replaced XGBoost to avoid data-leakage (priority_score would be both feature and label).
- **Graded junction weight** — normalised violation count per junction, not binary 0/1.
- **Vehicle type keys corrected** — "MOTOR CYCLE" (with space), "PASSENGER AUTO" — fixes a silent bug affecting ~26% of records.
- **Impact Simulator scoped to risk score reduction** — no traffic-flow claims.

#### Congestion Disclaimer
> Direct traffic-flow measurements are unavailable in the dataset.
> Congestion impact is estimated using a surrogate model combining violation density,
> violation severity, vehicle size, hotspot persistence, and junction importance.
""")

    a, b = st.columns(2)
    with a:
        st.markdown(f"""
**Dataset Stats**
| Metric | Value |
|---|---|
| Raw records | 298,450 |
| After filter (approved + NaN) | 240,654 |
| Hotspot clusters | **{n_clusters}** |
| Critical | **{n_crit}** |
| Escalating | **{n_esc}** |
| Abnormal Surges | **{n_surges}** |
| Immediate Enforcement zones | **{n_imm}** |
""")
    with b:
        st.markdown("""
**Tech Stack**
| Component | Tool |
|---|---|
| Language | Python 3.12 |
| Data | Pandas · NumPy |
| Clustering | scikit-learn DBSCAN |
| Regression | scikit-learn LinearRegression |
| Visualisation | Plotly Express / Graph Objects |
| App framework | Streamlit |
| Knowledge graph | graphify |
""")

