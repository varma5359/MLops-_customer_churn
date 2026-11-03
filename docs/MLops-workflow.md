# MLOps workflow: Git + DVC + DagsHub + MLflow

## Responsibilities

| Tool | Responsibility | What is stored/versioned |
|---|---|---|
| Git / GitHub | Source-code history and review | Python source, tests, configuration, DVC pointers and pipeline definition |
| DVC | Dataset and pipeline versioning | Dataset pointer files, pipeline stages, lockfile, model outputs |
| DagsHub | Shared remote for this project | Git repository view, DVC remote storage, hosted MLflow tracking UI |
| MLflow | Experiment tracking | Hyperparameters, run tags, evaluation metrics, model and evaluation artifacts |

The CSV itself is not committed to Git. DVC stores its content in the DagsHub DVC remote; Git stores the small .dvc pointer file. Never put a DagsHub token in a tracked file.

## 1. Clone the code repository

This project currently has a GitHub source repository and a DagsHub destination repository:

- GitHub: https://github.com/varma5359/MLops-_customer_churn
- DagsHub: https://dagshub.com/varma5359/mlops-customer-churn

Clone the GitHub source repository if you are starting from a clean machine:

```powershell
git clone https://github.com/varma5359/MLops-_customer_churn.git
cd MLops-_customer_churn
```

To publish the same Git history to DagsHub, configure it as an additional Git remote:

```powershell
git remote add dagshub https://dagshub.com/varma5359/mlops-customer-churn.git
git remote -v
```

If the DagsHub repository already contains a separate initial commit, fetch and merge that history once before pushing. Because both repositories may contain a README, resolve any README conflict by keeping the complete project README, then commit the merge:

```powershell
git fetch dagshub
git merge dagshub/main --allow-unrelated-histories --no-commit
# Keep the complete project README if the initial README conflicts:
git checkout --ours README.md
git add README.md
git commit -m "Merge initial DagsHub repository history"
git push dagshub main
```

If DagsHub is already connected to this GitHub repository and mirrors its history, do not perform the unrelated-history merge; simply push the current branch to the configured DagsHub remote.

## 2. Create and activate the environment

Python 3.11 is the project target.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify the toolchain:

```powershell
python --version
git --version
dvc version
python -c "import mlflow, dagshub; print('MLflow:', mlflow.__version__); print('DagsHub client:', dagshub.__version__)"
```

## 3. Authenticate to DagsHub

Create a DagsHub access token in your account settings. Do not paste it into chat, source code, .env.example, or a Git commit.

For Git pushes, authenticate using your normal Git credential manager or the DagsHub username/token prompt.

For DVC storage, configure the credentials locally. These commands write secrets to .dvc/config.local, which is ignored by Git:

```powershell
dvc remote modify origin --local access_key_id YOUR_DAGSHUB_TOKEN
dvc remote modify origin --local secret_access_key YOUR_DAGSHUB_TOKEN
```

The committed .dvc/config contains only the remote name, endpoint and non-secret URL. Confirm that local secrets are not staged:

```powershell
git status --short
git check-ignore .dvc/config.local
```

## 4. Add the local Telco dataset to DVC

Place the supplied dataset at exactly:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Then track it with DVC:

```powershell
dvc add data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
git add data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv.dvc data/raw/.gitignore
git commit -m "Track Telco churn dataset with DVC"
dvc push -r origin
git push origin main
git push dagshub main
```

The .csv.dvc file is a small pointer. The actual CSV content is uploaded to DagsHub DVC storage. Do not use git add -f on the CSV.

## 5. Run the reproducible pipeline

The pipeline is declared in dvc.yaml and its tunable values are in params.yaml.

```powershell
dvc repro
```

Stages run in dependency order:

1. validate: checks the input schema and data quality.
2. train: fits preprocessing and Logistic Regression on the training split, saves the model and metadata, and logs a training run to MLflow.
3. evaluate: evaluates the same deterministic stratified holdout split and logs metrics/artifacts to MLflow.

Inspect the pipeline and results:

```powershell
dvc dag
dvc status
dvc metrics show
dvc params diff
```

DVC creates/updates dvc.lock. Commit it with the pipeline and parameter changes:

```powershell
git add dvc.yaml dvc.lock params.yaml
git commit -m "Add reproducible DVC churn training pipeline"
git push origin main
git push dagshub main
dvc push -r origin
```

## 6. View MLflow experiments

The training and evaluation scripts call configure_tracking(). By default it initializes the DagsHub-hosted MLflow server for this repository:

https://dagshub.com/varma5359/mlops-customer-churn.mlflow

Authenticate once with the DagsHub client if prompted. Alternatively, create a local .env from .env.example and set the manual MLflow tracking variables there. The token must remain local and must never be committed.

Open the DagsHub repository and select its Experiments/MLflow view. Each run should show parameters, metrics, tags, and model/evaluation artifacts. Only repository contributors with write access can log experiments.

## 7. Prove reproducibility

Run the same pipeline a second time without changing code, data, or parameters:

```powershell
dvc repro
dvc status
```

DVC should report that stages are up to date and avoid rerunning unchanged work. Change train.random_state or train.test_size in params.yaml, then run dvc repro again. The train and evaluate stages should rerun, and the new metrics should appear as a separate MLflow run.

To reproduce a committed version on another machine:

```powershell
git pull
dvc pull -r origin
dvc repro
```

## Troubleshooting

- DVC 401/403: confirm that the token is valid and that you have write access to the DagsHub repository. Re-run the two dvc remote modify ... --local commands.
- MLflow authentication error: sign in to DagsHub and verify that your account can push to the repository. Check that MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD are not stale.
- Dataset not found: check the exact CSV filename and folder path in params.yaml.
- DVC says no remote: inspect dvc remote list and .dvc/config; the configured remote is named origin.
- DagsHub Git push rejected: check git remote -v, fetch the DagsHub branch, and resolve any history divergence before pushing.
