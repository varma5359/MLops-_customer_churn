"""Configure MLflow tracking locally or on the DagsHub-hosted tracking server."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def configure_tracking():
    """Return the configured MLflow module.

    DagsHub is the default remote. Set MLFLOW_TRACKING_URI to override it.
    Authentication is read from environment variables or the local DagsHub login.
    """
    import mlflow

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")
        if username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
        if password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    elif os.getenv("DAGSHUB_MLFLOW", "true").lower() in {"1", "true", "yes"}:
        import dagshub

        dagshub.init(
            repo_owner=os.getenv("DAGSHUB_REPO_OWNER", "varma5359"),
            repo_name=os.getenv("DAGSHUB_REPO_NAME", "mlops-customer-churn"),
            mlflow=True,
            dvc=False,
        )

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "telco-customer-churn"))
    return mlflow
