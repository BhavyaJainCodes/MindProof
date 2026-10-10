import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "research" / "metrics.json"

def load_metrics():
    if not METRICS.exists():
        return {}
    return json.loads(METRICS.read_text(encoding="utf-8"))

def topic_dataframe():
    p = ROOT / "research" / "topic_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def error_dataframe():
    p = ROOT / "research" / "error_analysis.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def ablation_dataframe():
    p = ROOT / "research" / "ablation_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()
