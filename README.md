# Customer Churn Prediction — Machine Learning & MLOps

An end-to-end customer churn prediction project developed with production-oriented Machine Learning Engineering and MLOps practices.

The goal is not only to train a classifier. The project will establish a reproducible workflow for validating data, building preprocessing and training pipelines, evaluating models, packaging inference, serving predictions, testing, and automating quality checks.

> **Project status:** Initial repository setup. Implementation will be delivered incrementally, one module at a time, with a separate Git commit for each completed module.

## 1. Problem statement

Customer churn occurs when an existing customer stops using a company's products or services. A churn prediction system estimates which customers are likely to leave so that business teams can prioritize retention actions.

This is a supervised binary-classification problem. The target in the planned BankChurners dataset is `Attrition_Flag`. The positive class will be defined explicitly during data preparation and documented consistently throughout training and inference.

## 2. Engineering objectives

- Validate raw data before training.
- Prevent target leakage and exclude identifier-only fields from model features.
- Use deterministic, reusable preprocessing and training code.
- Split data before fitting preprocessing steps to avoid leakage.
- Evaluate beyond accuracy, with emphasis on churn-class precision, recall, F1-score, confusion matrix, and ROC-AUC where appropriate.
- Persist the complete inference pipeline and its metadata.
- Keep training and prediction code modular and testable.
- Expose predictions through a documented API.
- Containerize the service after local functionality is verified.
- Add automated tests and CI checks before deployment.
- Add monitoring and data/model drift foundations as later milestones.

## 3. Planned architecture

```text
customer-churn-mlops/
├── data/
│   ├── raw/                  # Source dataset (not committed if large/sensitive)
│   └── processed/            # Generated data (not committed)
├── notebooks/                # Exploration and experiments
├── src/
│   ├── data/                  # Loading and data validation
│   ├── features/              # Reusable preprocessing
│   ├── models/                # Training and evaluation
│   ├── pipelines/             # End-to-end pipeline orchestration
│   └── api/                   # Prediction service
├── tests/                     # Unit and integration tests
├── configs/                   # Model and pipeline configuration
├── artifacts/                 # Locally generated model artifacts
├── scripts/                   # Repeatable command-line entry points
├── .github/workflows/         # CI automation
├── requirements.txt
├── .gitignore
└── README.md
```

The exact structure may be refined after the source notebook and dataset schema are reviewed. Existing user data and working decisions will not be overwritten.

## 4. Dataset contract

The initial implementation is planned around the BankChurners customer churn dataset:

- Expected target: `Attrition_Flag`
- Expected task: binary classification
- `CLIENTNUM` is an identifier, not a predictive feature.
- The two `Naive_Bayes_Classifier_Attrition_Flag_*` columns are excluded because they encode target-derived information and would leak the answer.
- The actual column names, target labels, row count, missing values, duplicates, and data types must be verified from the supplied CSV before preprocessing or training is implemented.

Place the source file at:

```text
data/raw/BankChurners.csv
```

Do not commit private customer data or large generated artifacts. Dataset handling and reproducibility instructions will be documented in the data module.

## 5. Development roadmap

| Stage | Module | Deliverable | Verification |
|---|---|---|---|
| 0 | Repository foundation | Project README, structure, ignore rules, dependency strategy | Repository and setup review |
| 1 | Data validation | Schema, target, missing-value, duplicate, and leakage checks | Valid and invalid dataset tests |
| 2 | Preprocessing | Reusable train/inference transformations | Fit/transform and leakage tests |
| 3 | Model training | Reproducible training pipeline and saved artifact | Training run and artifact checks |
| 4 | Evaluation | Classification metrics, confusion matrix, ROC-AUC, threshold analysis | Evaluation report |
| 5 | Experiment tracking | Parameters, metrics, and artifact lineage | Reproducible experiment record |
| 6 | Prediction interface | Validated prediction service and request/response schemas | API unit and integration tests |
| 7 | Packaging | Docker image and runtime configuration | Container build and health check |
| 8 | CI/CD foundations | Automated linting and tests in GitHub Actions | Successful workflow run |
| 9 | Monitoring | Prediction/data quality and drift foundations | Monitoring checks and report |
| 10 | Documentation | End-to-end execution and troubleshooting guide | Clean-environment walkthrough |

Each stage will be reviewed and committed independently. Tools such as MLflow, FastAPI, Docker, and GitHub Actions will be introduced when their corresponding module is reached rather than added without a working use case.

## 6. Development principles

1. **One module at a time:** each module has a clear purpose, implementation, and acceptance criteria.
2. **Small commits:** each completed file or cohesive module receives a descriptive commit.
3. **No hidden assumptions:** dataset columns, labels, metrics, and runtime settings are verified.
4. **Reproducibility:** random seeds and configuration are explicit.
5. **Leakage prevention:** preprocessing is fitted only on training data; identifiers and target-derived fields are excluded.
6. **Testability:** core logic is separated from command-line and API layers.
7. **Explainability:** each module documents what it does, why it exists, how to run it, and how to verify it.

## 7. Current setup status

The GitHub repository is newly initialized. No source code, dataset, dependency file, or existing README was present during the initial inspection. This README establishes the project contract; implementation begins with the data-validation module after the dataset schema is confirmed.

## 8. Next step

Provide the BankChurners CSV (or confirm the exact dataset file and column schema to use). The first implementation module will validate the raw dataset and produce a clear pass/fail report without modifying the source data.
