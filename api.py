"""FastAPI service for scoring sensor readings."""
from io import BytesIO
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from .data import generate_demo_data
from .detection import analyze_readings, summarize
app=FastAPI(title="AquaGuard API",version="1.0.0",description="Prototype water-flow anomaly scoring API")
class Reading(BaseModel):
    timestamp:str
    flow_lpm:float
    pressure_kpa:float|None=None
    leak_label:int|None=None
@app.get("/health")
def health(): return {"status":"ok"}
@app.get("/demo")
def demo():
    r=analyze_readings(generate_demo_data())
    return {"notice":"SIMULATED demonstration data","summary":summarize(r),"results":r.astype(object).where(pd.notna(r),None).to_dict(orient="records")}
@app.post("/analyze")
def analyze(readings:list[Reading]):
    if len(readings)<20: raise HTTPException(400,"Provide at least 20 readings")
    try:
        r=analyze_readings(pd.DataFrame([x.model_dump() for x in readings]))
        return {"summary":summarize(r),"results":r.astype(object).where(pd.notna(r),None).to_dict(orient="records")}
    except ValueError as e: raise HTTPException(422,str(e)) from e
@app.post("/analyze-csv")
async def analyze_csv(file:UploadFile=File(...)):
    if not (file.filename or "").lower().endswith(".csv"): raise HTTPException(400,"Upload a .csv file")
    try:
        r=analyze_readings(pd.read_csv(BytesIO(await file.read())))
        return {"summary":summarize(r),"results":r.astype(object).where(pd.notna(r),None).to_dict(orient="records")}
    except Exception as e: raise HTTPException(422,str(e)) from e
