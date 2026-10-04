"""Synthetic demo data and CSV validation."""
import numpy as np
import pandas as pd

def generate_demo_data(days=14, seed=42, interval_minutes=15):
    if days < 2 or interval_minutes < 1:
        raise ValueError("days must be >= 2 and interval_minutes must be >= 1")
    rng = np.random.default_rng(seed)
    n = int(days * 24 * 60 / interval_minutes)
    ts = pd.date_range("2025-01-01", periods=n, freq=f"{interval_minutes}min", tz="UTC")
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60
    baseline = .18 + .8*np.exp(-((hour-7.3)/1.5)**2) + 1.1*np.exp(-((hour-19.2)/2)**2)
    flow = np.maximum(.02, baseline + rng.normal(0, .12, n))
    pressure = 320 + rng.normal(0, 5, n)
    labels = np.zeros(n, dtype=int)
    for a,b,extra in [(int(n*.28),int(n*.34),.75),(int(n*.61),int(n*.66),1.25),(int(n*.82),int(n*.88),.45)]:
        flow[a:b] += extra + rng.normal(0,.05,b-a)
        pressure[a:b] -= 10 + 3*extra
        labels[a:b] = 1
    return pd.DataFrame({"timestamp":ts,"flow_lpm":flow.round(4),"pressure_kpa":pressure.round(2),"leak_label":labels,"sensor_id":"demo-meter-01","data_origin":"SIMULATED"})

def validate_readings(frame):
    missing = {"timestamp","flow_lpm"} - set(frame.columns)
    if missing: raise ValueError("Missing required column(s): " + ", ".join(sorted(missing)))
    f = frame.copy()
    f["timestamp"] = pd.to_datetime(f.timestamp, errors="coerce", utc=True)
    f["flow_lpm"] = pd.to_numeric(f.flow_lpm, errors="coerce")
    f = f.dropna(subset=["timestamp","flow_lpm"]).sort_values("timestamp").reset_index(drop=True)
    if f.empty: raise ValueError("No valid readings found")
    if (f.flow_lpm < 0).any(): raise ValueError("flow_lpm cannot be negative")
    if "pressure_kpa" not in f: f["pressure_kpa"] = np.nan
    else: f["pressure_kpa"] = pd.to_numeric(f.pressure_kpa, errors="coerce")
    if "leak_label" in f: f["leak_label"] = pd.to_numeric(f.leak_label, errors="coerce")
    return f
