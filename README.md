# Telco Customer Churn Prediction — Machine Learning & MLOps

End-to-end binary churn classification using the uploaded Telco Customer Churn dataset.

## Dataset contract
- File (local only): `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- 7,043 rows, 21 columns; target: `Churn` (`Yes` = churn, `No` = retained).
- Distribution: 1,869 Yes and 5,174 No.
- `customerID` is an identifier and is excluded from features.
- `TotalCharges` has 11 blank strings. Validation reports them; numeric coercion turns them into missing values and the training pipeline imputes them.
- No duplicate rows or duplicate customer IDs were observed in the supplied file.
- The dataset is not committed. The existing .gitignore excludes raw data and model artifacts.

## Architecture
```
src/data/validation.py       schema and data-quality validation
src/features/preprocessing.py train-fitted numeric/categorical transforms
src/models/train.py          stratified training and model persistence
src/models/evaluate.py       holdout classification metrics
src/api/                     FastAPI prediction service
src/monitoring.py             lightweight data-quality summary
scripts/validate_data.py      validation CLI
tests/                       unit/API tests
.github/workflows/ci.yml      GitHub Actions test workflow
Dockerfile                    container runtime
```

## Setup (Python 3.11 recommended)
Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
macOS/Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the CSV into `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`. Never commit the customer dataset.

## Run in order
```bash
python -m scripts.validate_data
python -m src.models.train
python -m src.models.evaluate
pytest
uvicorn src.api.main:app --reload
```

Validation writes `artifacts/data_validation.json`. Training writes `artifacts/churn_pipeline.joblib` and `artifacts/metadata.json`. Evaluation writes `artifacts/evaluation.json`.

Training uses an 80/20 stratified split (seed 42), class-weighted Logistic Regression, median imputation and scaling for numeric fields, and most-frequent imputation plus one-hot encoding for categorical fields. Preprocessing is fitted inside the training pipeline to avoid train/test leakage.

Metrics include accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and class-level report. The recorded local holdout run produced:
- Accuracy: 0.7381
- Churn precision: 0.5043
- Churn recall: 0.7834
- Churn F1: 0.6136
- ROC-AUC: 0.8413
- Confusion matrix (actual rows No/Yes, predicted columns No/Yes): [[747, 288], [81, 293]]

These are a baseline run on the supplied CSV and fixed split, not a production guarantee. Threshold selection must be aligned with retention costs and intervention capacity.

## API
Train first, then launch the server. Open `http://127.0.0.1:8000/docs`; health check is `GET /health`, prediction is `POST /predict`. The request accepts a `customers` array (maximum 500), each with the 19 non-ID, non-target feature columns. Response contains `churn_prediction`, `churn_probability`, and threshold. Prediction events are logged without direct customer identifiers to `logs/predictions.jsonl`.

## Docker
After local training:
```bash
docker build -t telco-churn-api .
docker run --rm -p 8000:8000 telco-churn-api
```
Check `http://localhost:8000/health`.

## Monitoring
```bash
python -m src.monitoring --data data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```
This reports row count, duplicate rows, nulls/blanks and churn rate. It is a basic data-quality monitor, not a statistical drift detector.

## Tests and CI
Run `pytest`. GitHub Actions runs the test suite for pushes and pull requests to main.

## Production hardening still required
This is a runnable end-to-end engineering project, not a deployed production service. Before real customer use, add authentication, rate limiting, threshold/cost validation, calibration and fairness review, data/model versioning, MLflow or equivalent experiment tracking, statistical drift alerts, model registry/rollback, secure infrastructure, and privacy/legal approval.
