"""Isolation Forest anomaly detection and alert summaries."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from .data import validate_readings

def _features(f):
    x = pd.DataFrame(index=f.index)
    x["flow"] = f.flow_lpm
    x["delta"] = f.flow_lpm.diff().fillna(0)
    x["rolling_mean"] = f.flow_lpm.rolling(4,min_periods=1).mean()
    x["rolling_std"] = f.flow_lpm.rolling(4,min_periods=1).std().fillna(0)
    x["hour"] = f.timestamp.dt.hour + f.timestamp.dt.minute/60
    x["overnight"] = ((x.hour >= 0) & (x.hour < 5)).astype(int)
    pressure = f.pressure_kpa
    x["pressure"] = pressure.fillna(pressure.median() if pressure.notna().any() else 0)
    return x

def analyze_readings(readings, contamination=.025):
    data = validate_readings(readings)
    x = _features(data)
    normal = data.leak_label.fillna(0).to_numpy() == 0 if "leak_label" in data and data.leak_label.notna().any() else np.ones(len(data), dtype=bool)
    if normal.sum() < 20: normal = np.ones(len(data), dtype=bool)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    model.fit(x.loc[normal])
    scores = -model.score_samples(x)
    cutoff = float(np.quantile(scores[normal], 1-contamination))
    flagged = scores > cutoff
    persistent = pd.Series(flagged).rolling(2,min_periods=1).sum().to_numpy() >= 2
    data["anomaly_score"] = scores.round(5)
    data["alert"] = persistent
    data["severity"] = np.where(persistent & (scores > cutoff*1.5),"high",np.where(persistent,"medium","normal"))
    data["reason"] = np.where(persistent,"Reading pattern differs from this meter's learned baseline","No sustained anomaly detected")
    data.attrs["threshold"] = cutoff
    return data

def summarize(results):
    alerts = results[results.alert]
    return {"readings":int(len(results)),"alerts":int(len(alerts)),"high_severity":int((alerts.severity=="high").sum()),"estimated_extra_liters":round(float(np.maximum(0,alerts.flow_lpm-results.flow_lpm.median()).sum()*.25),2)}
