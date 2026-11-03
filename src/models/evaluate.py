import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data.validation import ID_COLUMN, TARGET, load_dataset
from src.mlops.tracking import configure_tracking


def evaluate(data_path, artifact_dir="artifacts", test_size=0.2, random_state=42):
    df = load_dataset(data_path)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype("string").str.strip().replace("", pd.NA),
        errors="coerce",
    )
    y = df[TARGET].map({"No": 0, "Yes": 1})
    X = df.drop(columns=[TARGET, ID_COLUMN])
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model_path = Path(artifact_dir) / "churn_pipeline.joblib"
    model = joblib.load(model_path)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    result = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["No churn", "Churn"],
            output_dict=True,
            zero_division=0,
        ),
        "threshold": 0.5,
        "test_rows": len(y_test),
    }

    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "evaluation.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    mlflow = configure_tracking()
    with mlflow.start_run(run_name="holdout-evaluation"):
        mlflow.log_metrics({
            key: result[key]
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
        })
        mlflow.log_params({
            "threshold": 0.5,
            "test_size": test_size,
            "random_state": random_state,
            "test_rows": len(y_test),
        })
        mlflow.set_tags({
            "dataset": Path(data_path).name,
            "stage": "evaluation",
        })
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    result = evaluate(args.data, args.artifacts, args.test_size, args.random_state)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
