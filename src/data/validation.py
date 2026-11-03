from dataclasses import asdict, dataclass
from pathlib import Path
import json
import pandas as pd

TARGET = "Churn"
ID_COLUMN = "customerID"
EXPECTED_COLUMNS = {"customerID","gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges","Churn"}
NUMERIC_COLUMNS = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]
TARGET_LABELS = {"No","Yes"}

@dataclass
class ValidationReport:
    file: str
    rows: int
    columns: int
    missing_values: dict
    duplicate_rows: int
    duplicate_customer_ids: int
    target_distribution: dict
    unexpected_columns: list
    missing_columns: list
    invalid_target_values: list
    blank_values: dict
    passed: bool
    errors: list
    def to_dict(self): return asdict(self)

def load_dataset(path):
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path,dtype={"TotalCharges":"string"})

def validate_dataframe(df, source="<dataframe>"):
    errors=[]
    missing=sorted(EXPECTED_COLUMNS-set(df.columns))
    unexpected=sorted(set(df.columns)-EXPECTED_COLUMNS)
    if missing: errors.append(f"Missing required columns: {missing}")
    if unexpected: errors.append(f"Unexpected columns: {unexpected}")
    nulls={str(k):int(v) for k,v in df.isna().sum().items() if int(v)}
    blanks={}
    for c in df.select_dtypes(include=["object","string"]).columns:
        n=int(df[c].astype("string").str.strip().eq("").fillna(False).sum())
        if n: blanks[c]=n
    if nulls: errors.append(f"Null values found: {nulls}")
    unsupported={k:v for k,v in blanks.items() if k!="TotalCharges"}
    if unsupported: errors.append(f"Blank strings in unsupported columns: {unsupported}")
    duplicates=int(df.duplicated().sum())
    duplicate_ids=int(df[ID_COLUMN].duplicated().sum()) if ID_COLUMN in df else 0
    if duplicates: errors.append(f"Duplicate rows found: {duplicates}")
    if duplicate_ids: errors.append(f"Duplicate customer IDs found: {duplicate_ids}")
    labels=set(df[TARGET].dropna().astype(str).unique()) if TARGET in df else set()
    invalid=sorted(labels-TARGET_LABELS)
    if TARGET not in df: errors.append("Target column Churn is missing")
    elif invalid: errors.append(f"Unexpected target labels: {invalid}")
    for c in NUMERIC_COLUMNS:
        if c in df:
            raw=df[c].astype("string").str.strip()
            parsed=pd.to_numeric(raw.replace("",pd.NA),errors="coerce")
            bad=int((parsed.isna() & raw.notna() & raw.ne("")).sum())
            if bad: errors.append(f"Column {c!r} has {bad} non-numeric values")
    if len(df)==0: errors.append("Dataset is empty")
    distribution={str(k):int(v) for k,v in df[TARGET].value_counts(dropna=False).items()} if TARGET in df else {}
    return ValidationReport(source,len(df),len(df.columns),nulls,duplicates,duplicate_ids,distribution,unexpected,missing,invalid,blanks,not errors,errors)

def validate_file(path, report_path=None):
    report=validate_dataframe(load_dataset(path),str(path))
    if report_path:
        out=Path(report_path); out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report.to_dict(),indent=2),encoding="utf-8")
    return report
