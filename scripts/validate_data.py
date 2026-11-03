import argparse,json
from src.data.validation import validate_file
def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"); p.add_argument("--report",default="artifacts/data_validation.json"); a=p.parse_args()
    report=validate_file(a.data,a.report); print(json.dumps(report.to_dict(),indent=2))
    if not report.passed: raise SystemExit(1)
if __name__=="__main__": main()
