# Telco Customer Churn — End-to-End MLOps Project

An end-to-end binary classification project that predicts whether a telecom customer is likely to churn. The project combines reproducible ML development with **Git, DVC, DagsHub, MLflow, FastAPI, Docker, and GitHub Actions**.

## MLOps architecture

| Tool | Purpose | Evidence in this repository |
|---|---|---|
| Git / GitHub | Source-code version control and CI | Code, tests, pull requests, GitHub Actions |
| DVC | Dataset, pipeline and model-output versioning | .dvc files, dvc.yaml, params.yaml, dvc.lock |
| DagsHub | Shared DVC remote and hosted MLflow tracking | DVC storage remote and experiment UI |
| MLflow | Experiment tracking and model/artifact logging | Training and evaluation runs, params, metrics, tags and model |
| FastAPI | Model inference service | /health and /predict endpoints |
| Docker | Repeatable API runtime | Dockerfile |
| pytest | Automated tests | tests/ |
| GitHub Actions | Automated test execution | .github/workflows/ci.yml |

### End-to-end flow

```text
Telco CSV (local)
      |
      v
DVC add / DagsHub DVC storage
      |
      v
DVC pipeline (dvc repro)
      |
      +--> validate data
      |
      +--> train preprocessing + Logistic Regression
      |       |
      |       +--> model + metadata tracked by DVC
      |       +--> training run logged to MLflow
      |
      +--> evaluate deterministic holdout
              |
              +--> evaluation metrics tracked by DVC
              +--> evaluation run logged to MLflow

Git source + DVC metadata --> GitHub and DagsHub
MLflow runs/artifacts ------> DagsHub hosted MLflow UI
Model artifact -------------> FastAPI / Docker inference
```

## Dataset contract

- Dataset: IBM/Kaggle Telco Customer Churn CSV.
- Local path: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
- Shape: 7,043 rows × 21 columns.
- Target: Churn (Yes = 1,869; No = 5,174).
- customerID is an identifier and is excluded from model features.
- TotalCharges contains 11 blank strings. They are converted to missing numeric values and imputed inside the training pipeline.
- No exact duplicate rows or duplicate customer IDs were observed in the supplied CSV.
- The dataset is not committed to Git. DVC tracks it and sends its content to the DagsHub DVC remote.

## Repository structure

```text
.
├── .dvc/config                 # DagsHub DVC remote (no credentials)
├── .dvcignore
├── .env.example                # Safe environment-variable template
├── .github/workflows/ci.yml
├── data/raw/                   # Local CSV; DVC pointer is committed
├── docs/MLops-workflow.md      # Full setup, execution and verification guide
├── dvc.yaml                    # validate -> train -> evaluate
├── params.yaml                 # Data path and training parameters
├── scripts/validate_data.py
├── src/
│   ├── api/                    # FastAPI inference service
│   ├── data/                   # Dataset schema and validation
│   ├── features/               # Train-fitted preprocessing
│   ├── mlops/tracking.py       # DagsHub/MLflow configuration
│   └── models/                 # Training and evaluation with MLflow logging
├── tests/
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Quick start (Windows PowerShell)

Use Python 3.11 for this project.

```powershell
git clone https://github.com/varma5359/MLops-_customer_churn.git
cd MLops-_customer_churn

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy the supplied CSV into data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv.

## Configure DagsHub remotes and credentials

The configured DagsHub repository is:

- Repository: https://dagshub.com/varma5359/mlops-customer-churn
- MLflow tracking URI: https://dagshub.com/varma5359/mlops-customer-churn.mlflow
- DVC endpoint: https://dagshub.com/varma5359/mlops-customer-churn.s3

Add DagsHub as a Git remote if this local project is not already connected:

```powershell
git remote add dagshub https://dagshub.com/varma5359/mlops-customer-churn.git
git remote -v
```

If the DagsHub repository was initialized separately with its own first commit, fetch and merge that history once before pushing. Follow the exact merge instructions in docs/MLops-workflow.md. Do not merge unrelated histories if DagsHub is already mirroring this GitHub repository.

DVC's non-secret remote configuration is committed in .dvc/config. Configure the access token locally (never commit it):

```powershell
dvc remote modify origin --local access_key_id YOUR_DAGSHUB_TOKEN
dvc remote modify origin --local secret_access_key YOUR_DAGSHUB_TOKEN
git check-ignore .dvc/config.local
```

For MLflow, the training code uses dagshub.init() to configure the hosted tracking URI. Authenticate using the DagsHub client when prompted. You can also configure manual MLflow credentials in a local .env file copied from .env.example. Never commit .env or a token.

## Track the dataset with DVC

```powershell
dvc add data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
git add data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv.dvc data/raw/.gitignore
git commit -m "Track Telco churn dataset with DVC"
dvc push -r origin
git push origin main
git push dagshub main
```

Git stores the small DVC pointer file. DagsHub DVC storage stores the actual CSV content. The raw CSV remains excluded from Git.

## Run the complete DVC pipeline

```powershell
dvc repro
```

Pipeline stages:

1. **validate** — validates expected columns, target labels, duplicates, nulls and blank values.
2. **train** — creates a stratified train/holdout split, fits preprocessing only on training data, trains class-weighted Logistic Regression, saves the model and logs a training run to MLflow.
3. **evaluate** — reloads the saved model, evaluates the same deterministic holdout split, writes evaluation metrics and logs a separate MLflow evaluation run.

Inspect the pipeline:

```powershell
dvc dag
dvc status
dvc metrics show
dvc params diff
```

Commit DVC's pipeline lockfile and metadata, then push Git and DVC data:

```powershell
git add dvc.yaml dvc.lock params.yaml
git commit -m "Add reproducible DVC churn pipeline"
git push origin main
git push dagshub main
dvc push -r origin
```

## MLflow experiment tracking

Both the training and evaluation scripts log to the DagsHub-hosted MLflow tracking server.

Training run:
- Model type and hyperparameters
- Train/holdout row counts
- Dataset and stage tags
- Serialized sklearn pipeline model
- Model metadata

Evaluation run:
- Accuracy, precision, recall, F1 and ROC-AUC
- Decision threshold, split parameters and test-row count
- Dataset and stage tags
- evaluation.json artifact

Open the DagsHub repository's Experiments/MLflow view or visit the tracking URI above. Your DagsHub account must have contributor/write access to log runs.

## Prove that the pipeline is reproducible

Run the pipeline a second time without changing source, data or parameters:

```powershell
dvc repro
dvc status
```

DVC should report that stages are up to date and skip unchanged stages.

Now change train.random_state in params.yaml and rerun:

```powershell
dvc repro
dvc metrics show
```

The training and evaluation stages should rerun. The new evaluation should be visible as another MLflow run. This demonstrates parameter-based invalidation and experiment comparison.

To reproduce a committed version on another machine:

```powershell
git pull
dvc pull -r origin
dvc repro
```

For the detailed explanation, setup commands, DagsHub Git synchronization, credentials, and troubleshooting, read [docs/MLops-workflow.md](docs/MLops-workflow.md).

## Run tests

```powershell
pytest
```

GitHub Actions runs the test suite on pushes and pull requests to main. CI does not download the private/local dataset or require DagsHub credentials; unit tests use small synthetic examples.

## Run the API

After producing the model artifact with the pipeline:

```powershell
uvicorn src.api.main:app --reload
```

- Swagger UI: http://127.0.0.1:8000/docs
- Health: GET http://127.0.0.1:8000/health
- Prediction: POST http://127.0.0.1:8000/predict

## Docker

After model artifacts are available locally:

```powershell
docker build -t telco-churn-api .
docker run --rm -p 8000:8000 telco-churn-api
```

## Baseline model

The project uses an 80/20 stratified split (seed 42), median imputation and scaling for numeric columns, most-frequent imputation and one-hot encoding for categorical columns, and class-weighted Logistic Regression.

Previously recorded local baseline metrics on the supplied dataset:

| Metric | Value |
|---|---:|
| Accuracy | 0.7381 |
| Churn precision | 0.5043 |
| Churn recall | 0.7834 |
| Churn F1 | 0.6136 |
| ROC-AUC | 0.8413 |

These are baseline reference values, not a production guarantee. Re-run the DVC pipeline and inspect the actual MLflow run before using them in a presentation. The decision threshold should be chosen using retention costs and intervention capacity.

## Production limitations

This repository demonstrates an MLOps workflow; it is not yet a production deployment. Before real customer use, add authentication and rate limiting, model calibration and threshold governance, drift detection/alerts, fairness review, registry promotion/rollback policy, secure infrastructure, and privacy/legal approval.
