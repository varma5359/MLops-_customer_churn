from pydantic import BaseModel,Field,ConfigDict
class CustomerRecord(BaseModel):
    model_config=ConfigDict(extra="forbid")
    gender:str
    SeniorCitizen:int=Field(ge=0,le=1)
    Partner:str
    Dependents:str
    tenure:int=Field(ge=0)
    PhoneService:str
    MultipleLines:str
    InternetService:str
    OnlineSecurity:str
    OnlineBackup:str
    DeviceProtection:str
    TechSupport:str
    StreamingTV:str
    StreamingMovies:str
    Contract:str
    PaperlessBilling:str
    PaymentMethod:str
    MonthlyCharges:float=Field(ge=0)
    TotalCharges:float=Field(ge=0)
class PredictionRequest(BaseModel):
    customers:list[CustomerRecord]=Field(min_length=1,max_length=500)
class PredictionItem(BaseModel):
    churn_prediction:str
    churn_probability:float
    threshold:float
class PredictionResponse(BaseModel):
    predictions:list[PredictionItem]
