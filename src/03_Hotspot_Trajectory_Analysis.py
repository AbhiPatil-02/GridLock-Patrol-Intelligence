from IPython.display import display

"""
Cell 1 — Imports
"""

# In[]:
import pandas as pd
import numpy as np
from config import (
    ANOMALY_ANALYSIS,
    HOTSPOT_CLUSTERED,
    OUTPUT_DIR,
    TRAJECTORY_ANALYSIS,
)
from temporal_utils import parse_local_naive_series

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns

"""
Cell 2 — Load Data
"""

# In[]:
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(HOTSPOT_CLUSTERED)

print(df.shape)
df.head()

"""
Cell 3 — Datetime
"""

# In[]:
# `created_datetime` was converted from UTC to Asia/Kolkata in the
# preprocessing stage before being saved to outputs/hotspot_clustered.csv.
df["created_datetime"] = parse_local_naive_series(
    df["created_datetime"],
    "created_datetime",
)

df["year_month"] = (
    df["created_datetime"]
    .dt.to_period("M")
)

monthly_counts_global = (
    df.groupby("year_month")
    .size()
    .sort_index()
)

print(monthly_counts_global)

"""
Cell 4 — Monthly Violation Counts Per Cluster
"""

# In[]:
monthly_counts = (
    df[df["cluster_id"] != -1]
    .groupby(
        ["cluster_id", "year_month"]
    )
    .size()
    .reset_index(name="violations")
)

monthly_counts.head()

"""
Cell 5 — Create Pivot Table  Rows: cluster, Columns: month, Values: violation
"""

# In[]:
monthly_pivot = (
    monthly_counts
    .pivot(
        index="cluster_id",
        columns="year_month",
        values="violations"
    )
    .fillna(0)
)

monthly_pivot.head()

"""
Cell 6 — Verify Shape
"""

# In[]:
print(monthly_pivot.shape)

"""
Cell 7 — Trajectory Slopes
"""

# In[]:
# The trajectory regression uses data up to and including the latest complete month.
# April 2024 is excluded as it is an incomplete observation period.
trajectory_pivot = monthly_pivot.loc[:, monthly_pivot.columns <= pd.Period("2024-03", freq="M")]

slopes = []

for cluster_id, row in trajectory_pivot.iterrows():

    y = row.values
    active_months = np.count_nonzero(y)

    if active_months < 2:
        slopes.append(
            [cluster_id, np.nan, np.nan]
        )
        continue

    X = np.arange(len(y)).reshape(-1,1)

    model = LinearRegression()
    model.fit(X, y)

    # R² measures how well the linear trend explains the observed monthly variation in recorded violation counts.
    # It does not establish causality or prove actual changes in illegal parking.
    r2 = r2_score(y, model.predict(X))

    slopes.append(
        [cluster_id, model.coef_[0], r2]
    )

trajectory_df = pd.DataFrame(
    slopes,
    columns=["cluster_id", "slope", "r2"]
)

# The trajectory represents the change in recorded violation counts over the observed months.
# It does not claim to establish causality for actual illegal parking increases or decreases.

trajectory_df.head()

"""
Cell 8 — Visualize Slope Distribution
"""

# In[]:
plt.figure(figsize=(10,5))

sns.histplot(
    trajectory_df["slope"],
    bins=50
)

plt.title("Trajectory Slope Distribution")
plt.show()

"""
Cell 9 — Data Driven Thresholds
"""

# In[]:
valid_slopes = trajectory_df["slope"].dropna()

declining_threshold = (
    valid_slopes
    .quantile(0.25)
)

escalating_threshold = (
    valid_slopes
    .quantile(0.75)
)

print("Declining:", declining_threshold)
print("Escalating:", escalating_threshold)

"""
Cell 10 — Assign Labels
"""

# In[]:
def classify_slope(x):

    if pd.isna(x):
        return "Insufficient History"

    if x >= escalating_threshold:
        return "Escalating"

    elif x <= declining_threshold:
        return "Declining"

    else:
        return "Stable"

trajectory_df["trajectory"] = (
    trajectory_df["slope"]
    .apply(classify_slope)
)

trajectory_df.head()

"""
Cell 11 — Check Distribution
"""

# In[]:
trajectory_df["trajectory"].value_counts()

"""
Cell 12 — Top Escalating Hotspots
"""

# In[]:
trajectory_df.sort_values(
    "slope",
    ascending=False
)[
    [
        "cluster_id",
        "slope",
        "r2",
        "trajectory"
    ]
].head(20)

"""
Cell 13 — Anomaly Detection
"""

# In[]:
# Calculate historical mean and standard deviation for anomaly detection.
# March 2024 (latest_month) is excluded from the historical baseline
# to avoid including the current month in its own evaluation.

latest_month = pd.Period("2024-03", freq="M")

historical_months = monthly_pivot.columns[
    monthly_pivot.columns < latest_month
]

latest_month = pd.Period("2024-03", freq="M")

cluster_month_stats = (
    monthly_pivot[historical_months]
    .agg(
        ["mean", "std"],
        axis=1
    )
)

cluster_month_stats.head()

"""
Cell 14 — Current Month
"""

# In[]:
# March 2024 is used as the current analysis month because it is the latest complete month in the available dataset.
# April is excluded from current-month anomaly analysis because its observation period is incomplete.

print(monthly_counts_global)

latest_month = pd.Period("2024-03", freq="M")

print(latest_month)

"""
Cell 15 — Current Month Count
"""

# In[]:
cluster_month_stats["current_month"] = (
    monthly_pivot[latest_month]
)

"""
Cell 16 — Z-score
"""

# In[]:
def compute_z(row):

    if pd.isna(row["std"]):
        return np.nan

    if row["std"] == 0:
        return np.nan

    if row["mean"] == 0:
        return np.nan

    return (
        row["current_month"]
        - row["mean"]
    ) / row["std"]

cluster_month_stats["z_score"] = (
    cluster_month_stats
    .apply(compute_z, axis=1)
)

"""
Cell 17 — Anomaly Labels
"""

# In[]:
# A Z-score above 1.5 is used as a project-defined operational threshold
# for flagging an unusually high current-month violation count relative
# to the hotspot's historical baseline.
def anomaly_label(z):

    if pd.isna(z):
        return "Insufficient History"

    elif z > 1.5:
        return "Abnormal Surge"

    else:
        return "Normal"

cluster_month_stats["anomaly_status"] = (
    cluster_month_stats["z_score"]
    .apply(anomaly_label)
)

"""
Cell 18 — Watchlist
"""

# In[]:
watchlist = (
    cluster_month_stats[
        cluster_month_stats[
            "anomaly_status"
        ] == "Abnormal Surge"
    ]
    .sort_values(
        "z_score",
        ascending=False
    )
)

watchlist.head(20)

"""
Cell 19 — Save Outputs
"""

# In[]:
trajectory_df.to_csv(
    TRAJECTORY_ANALYSIS,
    index=False
)

cluster_month_stats.to_csv(
    ANOMALY_ANALYSIS
)

print("Saved")

# In[]:
monthly_counts_global

# In[]:
