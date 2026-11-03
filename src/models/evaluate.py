import argparse,json
from pathlib import Path
import joblib,pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from src.data.validation import load_dataset,TARGET,ID_COLUMN

def evaluate(data_path,artifact_dir="artifacts",test_size=.2,random_state=42):
    df=load_dataset(data_path)
    df["TotalCharges"]=pd.to_numeric(df["TotalCharges"].astype("string").str.strip().replace("",pd.NA),errors="coerce")
    y=df[TARGET].map({"No":0,"Yes":1}); X=df.drop(columns=[TARGET,ID_COLUMN])
    _,Xte,_,yte=train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y)
    model=joblib.load(Path(artifact_dir)/"churn_pipeline.joblib")
    prob=model.predict_proba(Xte)[:,1]; pred=(prob>=.5).astype(int)
    result={"accuracy":float(accuracy_score(yte,pred)),"precision":float(precision_score(yte,pred,zero_division=0)),"recall":float(recall_score(yte,pred,zero_division=0)),"f1":float(f1_score(yte,pred,zero_division=0)),"roc_auc":float(roc_auc_score(yte,prob)),"confusion_matrix":confusion_matrix(yte,pred).tolist(),"classification_report":classification_report(yte,pred,target_names=["No churn","Churn"],output_dict=True,zero_division=0),"threshold":.5,"test_rows":len(yte)}
    out=Path(artifact_dir); (out/"evaluation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"); p.add_argument("--artifacts",default="artifacts"); a=p.parse_args(); print(json.dumps(evaluate(a.data,a.artifacts),indent=2))
if __name__=="__main__": main()
