from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np
from src.model import train_and_save
from contextlib import asynccontextmanager

MODEL_PATH = "phishing_model.joblib"
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        train_and_save(MODEL_PATH)
    ml_models["model"] = joblib.load(MODEL_PATH)
    yield
    ml_models.clear()

app = FastAPI(title="Phishing Detection API", lifespan=lifespan)


class PredictionRequest(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float

class PredictionResponse(BaseModel):
    is_phishing: bool
    probability: float


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = pd.DataFrame([[
            request.feature_1,
            request.feature_2,
            request.feature_3,
            request.feature_4,
            request.feature_5
        ]], columns=["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"])
        
        prediction = ml_models["model"].predict(features)[0]
        prob = ml_models["model"].predict_proba(features)[0][1]
        
        return PredictionResponse(is_phishing=bool(prediction), probability=float(prob))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
