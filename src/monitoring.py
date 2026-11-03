import argparse,json
from pathlib import Path
import pandas as pd

def data_quality_summary(path):
    df=pd.read_csv(path,dtype={"TotalCharges":"string"})
    blanks={c:int(df[c].astype("string").str.strip().eq("").fillna(False).sum()) for c in df.select_dtypes(include=["object","string"]).columns}
    return {"rows":len(df),"columns":len(df.columns),"duplicate_rows":int(df.duplicated().sum()),"missing_values":{k:int(v) for k,v in df.isna().sum().items() if v},"blank_values":{k:v for k,v in blanks.items() if v},"churn_rate":float(df.Churn.eq("Yes").mean())}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--output",default="artifacts/monitoring.json"); a=p.parse_args()
    result=data_quality_summary(a.data); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
