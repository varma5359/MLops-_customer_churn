import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.validation import ID_COLUMN, TARGET, load_dataset, validate_dataframe
from src.features.preprocessing import build_preprocessor
from src.mlops.tracking import configure_tracking


def train(data_path, artifact_dir="artifacts", test_size=0.2, random_state=42):
    df = load_dataset(data_path)
    report = validate_dataframe(df, data_path)
    if not report.passed:
        raise ValueError("Dataset validation failed: " + "; ".join(report.errors))

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype("string").str.strip().replace("", pd.NA),
        errors="coerce",
    )
    y = df[TARGET].map({"No": 0, "Yes": 1})
    X = df.drop(columns=[TARGET, ID_COLUMN])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )
    model = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("classifier", classifier),
    ])
    model.fit(X_train, y_train)

    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "churn_pipeline.joblib"
    metadata_path = out / "metadata.json"
    joblib.dump(model, model_path)

    metadata = {
        "model": "LogisticRegression",
        "target": TARGET,
        "positive_class": "Yes",
        "features": X.columns.tolist(),
        "dropped_columns": [ID_COLUMN, TARGET],
        "test_size": test_size,
        "random_state": random_state,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "threshold": 0.5,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # The MLflow run records the model together with the exact training settings.
    mlflow = configure_tracking()
    with mlflow.start_run(run_name="logistic-regression-training"):
        mlflow.log_params({
            "model_type": "LogisticRegression",
            "test_size": test_size,
            "random_state": random_state,
            "class_weight": "balanced",
            "max_iter": 2000,
            "feature_count": X.shape[1],
        })
        mlflow.log_metrics({
            "train_rows": float(len(X_train)),
            "holdout_rows": float(len(X_test)),
        })
        mlflow.set_tags({
            "dataset": Path(data_path).name,
            "target": TARGET,
            "stage": "training",
        })
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(metadata_path), artifact_path="metadata")

    return model, X_test, y_test, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    _, _, y_test, metadata = train(
        args.data, args.artifacts, args.test_size, args.random_state
    )
    print(json.dumps({"status": "trained", **metadata}, indent=2))
    print("Holdout rows:", len(y_test))


if __name__ == "__main__":
    main()
