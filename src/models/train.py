import argparse, json
from pathlib import Path
import joblib, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.data.validation import load_dataset, validate_dataframe, TARGET, ID_COLUMN
from src.features.preprocessing import build_preprocessor

def train(data_path, artifact_dir="artifacts", test_size=.2, random_state=42):
    df=load_dataset(data_path)
    report=validate_dataframe(df,data_path)
    if not report.passed: raise ValueError("Dataset validation failed: "+"; ".join(report.errors))
    df["TotalCharges"]=pd.to_numeric(df["TotalCharges"].astype("string").str.strip().replace("",pd.NA),errors="coerce")
    y=df[TARGET].map({"No":0,"Yes":1})
    X=df.drop(columns=[TARGET,ID_COLUMN])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y)
    model=Pipeline([("preprocessor",build_preprocessor(Xtr)),("classifier",LogisticRegression(max_iter=2000,class_weight="balanced",random_state=random_state))])
    model.fit(Xtr,ytr)
    out=Path(artifact_dir); out.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,out/"churn_pipeline.joblib")
    metadata={"model":"LogisticRegression","target":TARGET,"positive_class":"Yes","features":X.columns.tolist(),"dropped_columns":[ID_COLUMN,TARGET],"test_size":test_size,"random_state":random_state,"train_rows":len(Xtr),"test_rows":len(Xte),"threshold":.5}
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return model,Xte,yte,metadata

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"); p.add_argument("--artifacts",default="artifacts"); a=p.parse_args()
    _,_,y,m=train(a.data,a.artifacts); print(json.dumps({"status":"trained",**m},indent=2)); print("Holdout rows:",len(y))
if __name__=="__main__": main()
