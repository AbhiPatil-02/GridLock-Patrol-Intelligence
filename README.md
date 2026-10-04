# GridLock Patrol Intelligence

Data-driven parking violation hotspot analysis and patrol optimization for traffic enforcement teams.

## Repository Structure

```text
parking-violation-analysis/
|
+-- data/
|   +-- raw/
|       +-- dataset.csv
|
+-- outputs/
|   +-- engineered_dataset.csv
|   +-- hotspot_clustered.csv
|   +-- hotspot_cluster_stats.csv
|   +-- trajectory_analysis.csv
|   +-- anomaly_analysis.csv
|   +-- top_enforcement_zones.csv
|   +-- best_patrol_windows.csv
|   +-- best_patrol_windows_detailed.csv
|   +-- patrol_action_plan.csv
|
+-- notebooks/
|   +-- 01_EDA.ipynb
|   +-- 02_Hotspot_Engineering.ipynb
|   +-- 03_Hotspot_Trajectory_Analysis.ipynb
|   +-- 04_Patrol_Optimizer.ipynb
|
+-- src/
|   +-- config.py
|   +-- 01_EDA.py
|   +-- 02_Hotspot_Engineering.py
|   +-- 03_Hotspot_Trajectory_Analysis.py
|   +-- 04_Patrol_Optimizer.py
|   +-- 05_Dashboard.py
|
+-- Docs/
+-- README.md
+-- DOCUMENTATION.md
+-- requirements.txt
```

## Data Layout

- Raw input data lives at `data/raw/dataset.csv`.
- All generated CSV artifacts are written to `outputs/`.
- Shared path constants are defined in `src/config.py`.
- Raw timestamp fields are stored in UTC and are converted to `Asia/Kolkata`
  before any temporal features are created.
- `created_datetime` is the primary event timestamp used for `hour`,
  `day_of_week`, `month`, `week`, and `is_weekend`.
- Timestamp columns with 100% null values are profiled and intentionally
  skipped instead of producing meaningless derived features.

## Timestamp Handling

- The raw feed contains UTC offsets such as `+00:00`.
- The preprocessing pipeline converts usable timestamps from UTC to
  `Asia/Kolkata`, then drops timezone metadata only to preserve the existing
  CSV schema.
- Temporal features therefore represent Bengaluru local event time, not UTC.
- Before each conversion the pipeline prints total rows, valid timestamps, and
  null percentage. Columns with less than 10% usable timestamps raise a
  warning, and columns with 100% nulls are skipped.

## Key EDA Findings

- Violations are concentrated during the morning period, especially from
  `08:00` to `12:00`, with the highest counts around `10:00` and `11:00`.
- Sunday records the highest number of violations, followed by Saturday,
  while Monday records the fewest.
- All `298,450` records already contain at least one parking-related offence,
  so filtering to a separate parking-only dataset does not change the sample.
- Monthly comparisons should be treated cautiously because the dataset covers
  only about six months and November/April appear to be partial months.
- `created_datetime` must be converted from UTC to `Asia/Kolkata` before
  extracting temporal features; otherwise hourly and weekday patterns shift by
  `5.5` hours.
- `action_taken_timestamp` and `closed_datetime` are `100%` missing in the raw
  data, so response-time and closure-time features are intentionally excluded.

## Pipeline

```text
data/raw/dataset.csv
        |
        v
src/01_EDA.py
        |
        v
src/02_Hotspot_Engineering.py
        +--> outputs/engineered_dataset.csv
        +--> outputs/hotspot_clustered.csv
        +--> outputs/hotspot_cluster_stats.csv
        |
        v
src/03_Hotspot_Trajectory_Analysis.py
        +--> outputs/trajectory_analysis.csv
        +--> outputs/anomaly_analysis.csv
        |
        v
src/04_Patrol_Optimizer.py
        +--> outputs/top_enforcement_zones.csv
        +--> outputs/best_patrol_windows.csv
        +--> outputs/best_patrol_windows_detailed.csv
        +--> outputs/patrol_action_plan.csv
        |
        v
src/05_Dashboard.py
```

### Hotspot Clustering

- Spatial hotspots are identified using DBSCAN.
- `eps = 0.00000962` radians (approximately 61 meters)
- `min_samples = 20`
- Distance metric: Haversine
- Algorithm: BallTree
- Final result: 506 hotspots with approximately 6.17% noise points.

## Setup

```bash
git clone https://github.com/Anurag1734/parking-violation-analysis.git
cd parking-violation-analysis
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Run The Pipeline

Run each stage from the repository root:

```bash
python src/01_EDA.py
python src/02_Hotspot_Engineering.py
python src/03_Hotspot_Trajectory_Analysis.py
python src/04_Patrol_Optimizer.py
```

The notebooks in `notebooks/` are updated to use the same centralized path configuration and can still be run for exploratory work.

## Run The Dashboard

```bash
streamlit run src/05_Dashboard.py
```

The dashboard reads all downstream artifacts from `outputs/`.

## Generated Outputs

After the pipeline completes, these files will be available in `outputs/`:

- `outputs/engineered_dataset.csv`
- `outputs/hotspot_clustered.csv`
- `outputs/hotspot_cluster_stats.csv`
- `outputs/trajectory_analysis.csv`
- `outputs/anomaly_analysis.csv`
- `outputs/top_enforcement_zones.csv`
- `outputs/best_patrol_windows.csv`
- `outputs/best_patrol_windows_detailed.csv`
- `outputs/patrol_action_plan.csv`

## Verification

PowerShell:

```powershell
Test-Path outputs\engineered_dataset.csv
Test-Path outputs\hotspot_clustered.csv
Test-Path outputs\hotspot_cluster_stats.csv
Test-Path outputs\trajectory_analysis.csv
Test-Path outputs\anomaly_analysis.csv
Test-Path outputs\top_enforcement_zones.csv
Test-Path outputs\best_patrol_windows.csv
Test-Path outputs\best_patrol_windows_detailed.csv
Test-Path outputs\patrol_action_plan.csv
```

Example row-count check:

```bash
python -c "import pandas as pd; print(len(pd.read_csv('outputs/patrol_action_plan.csv')))"
```
