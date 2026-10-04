from IPython.display import display

"""
# Notebook 4 - Patrol Optimizer

This notebook converts hotspot analytics into an enforcement action plan.

It combines:
- `priority_score`
- `risk_level`
- `trajectory`
- `anomaly_status`

into a single `patrol_priority` score, then generates:
- Top 20 Enforcement Zones
- Best Patrol Windows

This is the operational answer to the hackathon prompt: how AI-driven parking intelligence can enable targeted enforcement.
"""

# In[]:
import pandas as pd
import numpy as np
from config import (
    ANOMALY_ANALYSIS,
    BEST_PATROL_WINDOWS,
    BEST_PATROL_WINDOWS_DETAILED,
    HOTSPOT_CLUSTERED,
    HOTSPOT_CLUSTER_STATS,
    OUTPUT_DIR,
    PATROL_ACTION_PLAN,
    TOP_ENFORCEMENT_ZONES,
    TRAJECTORY_ANALYSIS,
)

from sklearn.preprocessing import MinMaxScaler

"""
## Load Inputs
"""

# In[]:
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

hotspot_events = pd.read_csv(HOTSPOT_CLUSTERED)
cluster_stats = pd.read_csv(HOTSPOT_CLUSTER_STATS)
trajectory_df = pd.read_csv(TRAJECTORY_ANALYSIS)
anomaly_df = pd.read_csv(ANOMALY_ANALYSIS)

print(hotspot_events.shape)
print(cluster_stats.shape)
print(trajectory_df.shape)
print(anomaly_df.shape)

"""
## Build Patrol Priority
"""

# In[]:
# patrol_priority is a project-defined heuristic score combining existing
# hotspot priority, trajectory behaviour, and anomaly status to support
# relative enforcement prioritization. It is not a probability, causal
# estimate, or learned prediction.
# In[]:
patrol_df = (
    cluster_stats
    .merge(
        trajectory_df,
        on="cluster_id",
        how="left"
    )
    .merge(
        anomaly_df,
        on="cluster_id",
        how="left"
    )
)

trajectory_base_map = {
    "Declining": 0.25,
    "Stable": 0.50,
    "Escalating": 1.00,
    "Insufficient History": 0.10,
}

anomaly_base_map = {
    "Normal": 0.40,
    "Abnormal Surge": 1.00,
    "Abnormal Drop": 0.20,
}

patrol_df["trajectory_base_score"] = (
    patrol_df["trajectory"]
    .map(trajectory_base_map)
    .fillna(0.25)
)

patrol_df["priority_norm"] = MinMaxScaler().fit_transform(
    patrol_df[["priority_score"]]
)

patrol_df["r2_filled"] = patrol_df["r2"].fillna(0)

# The trajectory contribution is weighted by R² so that trends with stronger
# linear fit contribute more strongly to patrol prioritization. Missing R²
# values are filled with zero, resulting in a 0.5 multiplier.
patrol_df["trajectory_weight"] = (
    patrol_df["trajectory_base_score"]
    * (0.50 + 0.50 * patrol_df["r2_filled"])
)

patrol_df["anomaly_base_score"] = (
    patrol_df["anomaly_status"]
    .map(anomaly_base_map)
    .fillna(0.40)
)

patrol_df["surge_intensity"] = patrol_df["z_score"].clip(lower=0).fillna(0)
patrol_df["surge_intensity_norm"] = MinMaxScaler().fit_transform(
    patrol_df[["surge_intensity"]]
)

patrol_df["anomaly_weight"] = (
    0.60 * patrol_df["anomaly_base_score"]
    + 0.40 * patrol_df["surge_intensity_norm"]
)

patrol_df["patrol_priority"] = (
    0.50 * patrol_df["priority_norm"]
    + 0.30 * patrol_df["trajectory_weight"]
    + 0.20 * patrol_df["anomaly_weight"]
)

patrol_df[[
    "cluster_id",
    "priority_score",
    "priority_norm",
    "risk_level",
    "trajectory",
    "anomaly_status",
    "patrol_priority"
]].head()

"""
## Top 20 Enforcement Zones
"""

# In[]:
top_enforcement_zones = (
    patrol_df[
        [
            "cluster_id",
            "top_station",
            "centroid_lat",
            "centroid_lon",
            "violations",
            "avg_severity",
            "active_months",
            "priority_score",
            "risk_level",
            "trajectory",
            "r2",
            "current_month",
            "z_score",
            "anomaly_status",
            "patrol_priority"
        ]
    ]
    .sort_values(
        "patrol_priority",
        ascending=False
    )
    .head(20)
    .reset_index(drop=True)
)

top_enforcement_zones

"""
## Best Patrol Windows
"""

"""
## Dashboard Centerpiece
"""

# In[]:
top_enforcement_zones[
    [
        "cluster_id",
        "top_station",
        "risk_level",
        "trajectory",
        "anomaly_status",
        "patrol_priority"
    ]
].head(20)

# In[]:
day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

top_cluster_ids = top_enforcement_zones["cluster_id"]

window_df = hotspot_events[
    hotspot_events["cluster_id"].isin(top_cluster_ids)
].copy()

window_df["day_of_week"] = pd.Categorical(
    window_df["day_of_week"],
    categories=day_order,
    ordered=True
)

window_df = window_df.dropna(subset=["hour", "day_of_week"]).copy()
window_df["hour"] = window_df["hour"].astype(int)

cluster_totals = (
    window_df
    .groupby("cluster_id")
    .agg(cluster_total_violations=("id", "count"))
    .reset_index()
)

window_profiles = (
    window_df
    .groupby(
        ["cluster_id", "day_of_week", "hour"],
        observed=True
    )
    .agg(
        window_violations=("id", "count"),
        avg_window_severity=("max_severity", "mean"),
        weekend_share=("is_weekend", "mean")
    )
    .reset_index()
    .merge(
        cluster_totals,
        on="cluster_id",
        how="left"
    )
)

window_profiles["hour_percentage"] = (
    window_profiles["window_violations"]
    / window_profiles["cluster_total_violations"]
)

window_profiles["window_violations_norm"] = MinMaxScaler().fit_transform(
    window_profiles[["window_violations"]]
)
window_profiles["avg_window_severity_norm"] = MinMaxScaler().fit_transform(
    window_profiles[["avg_window_severity"]]
)

window_profiles["window_score"] = (
    0.75 * window_profiles["window_violations_norm"]
    + 0.25 * window_profiles["avg_window_severity_norm"]
)

window_profiles["window_rank"] = (
    window_profiles
    .groupby("cluster_id")["window_score"]
    .rank(method="first", ascending=False)
)

window_profiles["patrol_window"] = (
    window_profiles["day_of_week"].astype(str)
    + " "
    + window_profiles["hour"].map(lambda x: f"{x:02d}:00-{(x + 1):02d}:00")
)

best_windows_detailed = (
    window_profiles
    .sort_values(
        ["cluster_id", "window_score"],
        ascending=[True, False]
    )
    .query("window_rank <= 3")
    .copy()
)
best_windows_detailed["hour_percentage_label"] = (
    (100 * best_windows_detailed["hour_percentage"])
    .round(1)
    .astype(str) +
    "% of hotspot activity with valid time data"
)

best_windows_detailed.head(20)

# In[]:
best_windows_summary = (
    best_windows_detailed
    .assign(
        patrol_window_summary=(
            best_windows_detailed["patrol_window"]
            + " ("
            + best_windows_detailed["hour_percentage_label"]
            + ")"
        )
    )
    .groupby("cluster_id")
    .agg(
        recommended_windows=(
            "patrol_window_summary",
            lambda x: " | ".join(x)
        )
    )
    .reset_index()
)

best_patrol_windows = (
    top_enforcement_zones
    .merge(
        best_windows_summary,
        on="cluster_id",
        how="left"
    )
    [[
        "cluster_id",
        "top_station",
        "risk_level",
        "trajectory",
        "anomaly_status",
        "patrol_priority",
        "recommended_windows"
    ]]
)

best_patrol_windows

"""
## Final Enforcement Action Plan
"""

# In[]:
patrol_action_plan = (
    top_enforcement_zones
    .merge(
        best_windows_summary,
        on="cluster_id",
        how="left"
    )
)

patrol_action_plan["deployment_level"] = pd.cut(
    patrol_action_plan["patrol_priority"],
    bins=[0, 0.4, 0.7, 1.0],
    labels=[
        "Routine Monitoring",
        "Targeted Patrol",
        "Immediate Enforcement"
    ],
    include_lowest=True
)

patrol_action_plan[[
    "cluster_id",
    "risk_level",
    "trajectory",
    "anomaly_status",
    "patrol_priority",
    "deployment_level",
    "recommended_windows"
]]

"""
## Save Outputs
"""

# In[]:
top_enforcement_zones.to_csv(
    TOP_ENFORCEMENT_ZONES,
    index=False
)

best_windows_detailed.to_csv(
    BEST_PATROL_WINDOWS_DETAILED,
    index=False
)

best_patrol_windows.to_csv(
    BEST_PATROL_WINDOWS,
    index=False
)

patrol_action_plan.to_csv(
    PATROL_ACTION_PLAN,
    index=False
)

print("Saved")

# In[]:
patrol_action_plan["deployment_level"].value_counts()

# In[]:
patrol_action_plan[
[
    "cluster_id",
    "risk_level",
    "trajectory",
    "anomaly_status",
    "patrol_priority",
    "deployment_level",
    "recommended_windows"
]
].head(20)
