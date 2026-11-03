from fastapi.testclient import TestClient
from src.api.main import app
def test_health_endpoint():
    c=TestClient(app); r=c.get("/health"); assert r.status_code==200 and r.json()["status"]=="ok"
def test_predict_requires_model(monkeypatch):
    import src.api.main as api
    monkeypatch.setattr(api,"model",None)
    c=TestClient(app)
    body={"customers":[{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":1,"PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":29.85}]}
    assert c.post("/predict",json=body).status_code==503
