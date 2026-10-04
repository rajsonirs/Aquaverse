from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import streamlit as st
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from aquaguard.data import generate_demo_data, validate_readings
from aquaguard.detection import analyze_readings, summarize
from aquaguard.metrics import evaluate
st.set_page_config(page_title="AquaGuard",page_icon="💧",layout="wide")
st.title("💧 AquaGuard — Water Leak Detection")
st.caption("AI-assisted anomaly detection for water meter readings")
st.info("Decision-support prototype. Built-in readings are simulated, not field measurements; alerts are not proof of a leak.")
upload=st.file_uploader("Upload CSV: timestamp, flow_lpm (optional: pressure_kpa, leak_label)",type="csv")
try:
    if upload: readings=validate_readings(pd.read_csv(upload)); origin="Uploaded data"
    else: readings=generate_demo_data(); origin="SIMULATED demonstration data"
    results=analyze_readings(readings); summary=summarize(results)
except Exception as e:
    st.error(str(e)); st.stop()
st.caption(f"Source: **{origin}** · {len(readings):,} readings")
c=st.columns(4)
c[0].metric("Readings",f"{summary['readings']:,}"); c[1].metric("Sustained alerts",f"{summary['alerts']:,}")
c[2].metric("High severity",f"{summary['high_severity']:,}"); c[3].metric("Estimated excess liters*",f"{summary['estimated_extra_liters']:,.1f}")
st.caption("*Rough estimate versus median flow; validate against a calibrated meter before operational use.")
fig=px.line(results,x="timestamp",y="flow_lpm",title="Water flow over time",labels={"flow_lpm":"Flow (L/min)"})
a=results[results.alert]
if not a.empty: fig.add_scatter(x=a.timestamp,y=a.flow_lpm,mode="markers",name="Alert",marker={"color":"red","size":8})
st.plotly_chart(fig,use_container_width=True)
metrics=evaluate(results)
if metrics:
    st.subheader("Metrics against labels (diagnostic only)")
    st.caption("For an honest score, hold out complete dates, meters, or experiments before tuning.")
    cols=st.columns(4)
    for col,key in zip(cols,["precision","recall","f1","false_alarms"]): col.metric(key.replace("_"," ").title(),metrics[key])
with st.expander("Recent alerts"):
    st.dataframe(results.loc[results.alert,["timestamp","flow_lpm","pressure_kpa","anomaly_score","severity","reason"]].tail(100),use_container_width=True)
st.download_button("Download analyzed CSV",results.to_csv(index=False),"aquaguard_analysis.csv","text/csv")
with st.expander("How it works and limitations"):
    st.write("Isolation Forest learns typical patterns and scores unusual behavior. Consecutive suspicious readings become alerts. Synthetic demo performance does not establish real-world accuracy; validate with representative labeled data first.")
