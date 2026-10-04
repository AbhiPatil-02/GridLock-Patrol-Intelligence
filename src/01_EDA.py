from IPython.display import display

"""
Section 1 — Load Dataset
"""

# In[]:
import pandas as pd
import numpy as np
from config import RAW_DATA
from temporal_utils import (
    PRIMARY_EVENT_TIMESTAMP,
    add_temporal_features,
    convert_utc_series_to_local_naive,
)

df = pd.read_csv(RAW_DATA)

print(df.shape)
df.head()

"""
Section 2 — Dataset Overview
"""

# In[]:
print(df.info())

display(df.sample(5))

"""
Section 3 — Missing Value Analysis
"""

# In[]:
nulls = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": round(df.isna().mean()*100,2)
})

nulls.sort_values("missing_percent", ascending=False)

# In[]:
important_cols = [
    "action_taken_timestamp",
    "closed_datetime",
    "junction_name",
    "vehicle_type",
    "violation_type",
    "validation_status"
]

df[important_cols].isna().mean()*100

# In[]:
print("Duplicate rows:", df.duplicated().sum())

# In[]:
df["validation_status"].value_counts(dropna=False)

"""
Section 4 — GPS Validation
"""

# In[]:
df["latitude"].describe()

# In[]:
df["longitude"].describe()

# In[]:
print(df["latitude"].min(), df["latitude"].max())
print(df["longitude"].min(), df["longitude"].max())

"""
Section 5 — Timestamp Conversion
"""

# In[]:
date_cols = [
    "created_datetime",
    "closed_datetime",
    "modified_datetime",
    "action_taken_timestamp",
    "validation_timestamp"
]

for col in date_cols:
    if col not in df.columns:
        continue
    df[col], _ = convert_utc_series_to_local_naive(df[col], col)

"""
Section 6 — Date Coverage
"""

# In[]:
print(df["created_datetime"].min())
print(df["created_datetime"].max())

"""
Section 7 — Feature Engineering
"""

# In[]:
# `created_datetime` is the violation creation timestamp, so it is the
# correct event clock for hotspot activity by local hour/day/week.
add_temporal_features(
    df,
    PRIMARY_EVENT_TIMESTAMP,
    include_week=True,
    already_localized=True,
)

"""
Section 8 — Violation Types
"""

# In[]:
df["violation_type"].value_counts().head(20)

# In[]:
df["violation_type"].sample(20)

"""
Section 9 — Vehicle Types
"""

# In[]:
df["vehicle_type"].value_counts(dropna=False)

"""
Section 10 — Police Stations
"""

# In[]:
df["police_station"].value_counts().head(20)

"""
Section 11 — Junction Analysis
"""

# In[]:
df["junction_name"].value_counts().head(20)

"""
Section 12 — Hourly Violation Pattern
"""

# In[]:
hourly = df.groupby("hour").size()
hourly.sort_values(ascending=False).head(10)

"""
Section 13 — Day-of-Week Pattern
"""

# In[]:
df.groupby("day_of_week").size().sort_values(ascending=False)

"""
Section 14 — Monthly Trend
"""

# In[]:
df.groupby("month").size()

"""
Section 15 — Response Time Feasibility
"""

# In[]:
action_timestamp_null_pct = round(
    df["action_taken_timestamp"].isna().mean()*100,
    2
)
closed_datetime_null_pct = round(
    df["closed_datetime"].isna().mean()*100,
    2
)

print("Action Timestamp Null %")
print(action_timestamp_null_pct)

print("Closed Datetime Null %")
print(closed_datetime_null_pct)

if action_timestamp_null_pct == 100:
    print(
        "Skipping response_time_minutes because action_taken_timestamp "
        "contains 100% missing values."
    )
else:
    df["response_time_minutes"] = (
        df["action_taken_timestamp"]
        - df["created_datetime"]
    ).dt.total_seconds()/60
    df["response_time_minutes"].describe()

"""
Section 18 — Closure Time Feasibility
"""

# In[]:
if closed_datetime_null_pct == 100:
    print(
        "Skipping closure_time_hours because closed_datetime contains "
        "100% missing values."
    )
else:
    df["closure_time_hours"] = (
        df["closed_datetime"]
        - df["created_datetime"]
    ).dt.total_seconds()/3600
    df["closure_time_hours"].describe()

"""
Section 19 — Spatial Density Readiness
"""

# In[]:
print(
    df[["latitude","longitude"]]
    .dropna()
    .shape
)

"""
Section 20 — Hotspot Candidate Count
"""

# In[]:
df["lat_grid"] = df["latitude"].round(3)
df["lon_grid"] = df["longitude"].round(3)

hotspots = (
    df.groupby(["lat_grid","lon_grid"])
    .size()
    .sort_values(ascending=False)
)

hotspots.head(20)

"""
A. Validation Status Breakdown by Month
"""

# In[]:
pd.crosstab(
    df["month"],
    df["validation_status"],
    margins=True
)

"""
B. Unique Violation Types
"""

# In[]:
import ast

violations = set()

for row in df["violation_type"]:
    try:
        vals = ast.literal_eval(row)
        for v in vals:
            violations.add(v)
    except:
        pass

sorted(violations)

# In[]:
df.groupby("hour").size().sort_index()

"""
Step 1: Validate Parking-Violation Coverage
"""

# In[]:
parking_df = df[
    df["violation_type"].str.contains(
        "PARKING",
        na=False
    )
].copy()

print("Original:", len(df))
print("Parking:", len(parking_df))
print("Percentage:", round(len(parking_df)/len(df)*100,2))

"""
Step 2: Parking Filter Redundancy Check
"""

# In[]:
if len(parking_df) == len(df):
    print(
        "All records already contain at least one parking-related offence. "
        "Repeating hourly, day-of-week, and spatial analysis on parking_df "
        "would be redundant."
    )
else:
    parking_df["violation_type"].value_counts()

"""
Step 7: Persistence Analysis
"""

# In[]:
df["year_month"] = (
    df["created_datetime"]
    .dt.to_period("M")
)

# In[]:
station_persistence = (
    df
    .groupby("police_station")["year_month"]
    .nunique()
    .sort_values(ascending=False)
)

station_persistence.head(20)

"""
Step 8: Validation Status Investigation
"""

# In[]:
df["validation_status"].value_counts(dropna=False)

# In[]:
pd.crosstab(
    df["month"],
    df["validation_status"],
    margins=True
)

"""
Step 9: Multi-Violation Analysis
"""

# In[]:
import ast

df["num_violations"] = (
    df["violation_type"]
    .apply(lambda x: len(ast.literal_eval(x)))
)

df["num_violations"].value_counts()

"""
Step 10: Determine DBSCAN Parameters
"""

# In[]:
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

coords = df[
    ["latitude","longitude"]
].dropna()

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(coords)

distances, indices = neighbors_fit.kneighbors(coords)

distances = np.sort(distances[:,4])

plt.figure(figsize=(10,5))
plt.plot(distances)
plt.title("K-distance plot")
plt.show()

# In[]:


# In[]:

