from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from aquaguard.data import generate_demo_data
out=ROOT/"data"/"generated"/"demo_readings.csv"
out.parent.mkdir(parents=True,exist_ok=True)
generate_demo_data().to_csv(out,index=False)
print(f"Wrote simulated sample data: {out}")
