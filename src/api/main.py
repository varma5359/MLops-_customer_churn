import json,os
from contextlib import asynccontextmanager
from datetime import datetime,timezone
from pathlib import Path
import joblib,pandas as pd
from fastapi import FastAPI,HTTPException
from src.api.schemas import PredictionRequest,PredictionResponse,PredictionItem
ARTIFACT_DIR=Path(os.getenv("ARTIFACT_DIR","artifacts"))
MODEL_PATH=ARTIFACT_DIR/"churn_pipeline.joblib"
META_PATH=ARTIFACT_DIR/"metadata.json"
LOG_PATH=Path(os.getenv("PREDICTION_LOG","logs/predictions.jsonl"))
model=None; threshold=.5
@asynccontextmanager
async def lifespan(app):
    global model,threshold
    if MODEL_PATH.is_file():
        model=joblib.load(MODEL_PATH)
        if META_PATH.is_file(): threshold=float(json.loads(META_PATH.read_text()).get("threshold",.5))
    yield
app=FastAPI(title="Telco Customer Churn API",version="1.0.0",lifespan=lifespan)
@app.get("/health")
def health(): return {"status":"ok","model_loaded":model is not None}
@app.post("/predict",response_model=PredictionResponse)
def predict(request:PredictionRequest):
    if model is None: raise HTTPException(503,"Model artifact is not loaded. Train the model first.")
    rows=[x.model_dump() for x in request.customers]
    try: probabilities=model.predict_proba(pd.DataFrame(rows))[:,1]
    except Exception as exc: raise HTTPException(422,f"Could not process features: {exc}") from exc
    LOG_PATH.parent.mkdir(parents=True,exist_ok=True); results=[]
    with LOG_PATH.open("a",encoding="utf-8") as log:
        for probability in probabilities:
            probability=float(probability); label="Yes" if probability>=threshold else "No"
            results.append(PredictionItem(churn_prediction=label,churn_probability=round(probability,6),threshold=threshold))
            log.write(json.dumps({"timestamp":datetime.now(timezone.utc).isoformat(),"prediction":label,"probability":probability})+"\n")
    return PredictionResponse(predictions=results)
