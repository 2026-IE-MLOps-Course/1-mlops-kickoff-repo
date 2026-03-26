from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib

from src.clean_data import clean_data
from src.validate import validate_dataframe
from src.features import engineer_features
from src.infer import run_inference

app = FastAPI(title="Churn API")

MODEL_PATH = "models/model.joblib"
model = joblib.load(MODEL_PATH)


class Customer(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str


class RequestData(BaseModel):
    customers: List[Customer]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: RequestData):
    try:
        df = pd.DataFrame([c.model_dump() for c in data.customers])

        df_clean = clean_data(df, drop_na=False)

        # Add dummy target BEFORE validation
        df_clean["Churn"] = "No"

        validate_dataframe(df_clean, df_clean.columns.tolist())

        ids = df_clean["customerID"].copy()

        df_features = engineer_features(df_clean)

        if "Churn" in df_features.columns:
            df_features = df_features.drop(columns=["Churn"])

        preds = run_inference(model, df_features)
        preds["customerID"] = ids.values

        return {"predictions": preds.to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
