from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = ROOT / "outputs"

RAW_DATA = RAW_DATA_DIR / "dataset.csv"

ENGINEERED_DATASET = OUTPUT_DIR / "engineered_dataset.csv"
HOTSPOT_CLUSTERED = OUTPUT_DIR / "hotspot_clustered.csv"
HOTSPOT_CLUSTER_STATS = OUTPUT_DIR / "hotspot_cluster_stats.csv"
TRAJECTORY_ANALYSIS = OUTPUT_DIR / "trajectory_analysis.csv"
ANOMALY_ANALYSIS = OUTPUT_DIR / "anomaly_analysis.csv"
TOP_ENFORCEMENT_ZONES = OUTPUT_DIR / "top_enforcement_zones.csv"
BEST_PATROL_WINDOWS = OUTPUT_DIR / "best_patrol_windows.csv"
BEST_PATROL_WINDOWS_DETAILED = OUTPUT_DIR / "best_patrol_windows_detailed.csv"
PATROL_ACTION_PLAN = OUTPUT_DIR / "patrol_action_plan.csv"
