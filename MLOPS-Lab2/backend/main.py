from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_TITLE = "Wine Class Prediction API"
MODEL_PATH = Path(__file__).resolve().parent / "wine_model.pkl"

# Feature order must match training data columns.
FEATURE_NAMES: List[str] = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
]

CLASS_NAMES = {
    0: "class_0 (Cultivar A)",
    1: "class_1 (Cultivar B)",
    2: "class_2 (Cultivar C)",
}

class PredictRequest(BaseModel):
    alcohol: float = Field(..., ge=0)
    malic_acid: float = Field(..., ge=0)
    ash: float = Field(..., ge=0)
    alcalinity_of_ash: float = Field(..., ge=0)
    magnesium: float = Field(..., ge=0)
    total_phenols: float = Field(..., ge=0)
    flavanoids: float = Field(..., ge=0)
    nonflavanoid_phenols: float = Field(..., ge=0)
    proanthocyanins: float = Field(..., ge=0)
    color_intensity: float = Field(..., ge=0)
    hue: float = Field(..., ge=0)
    od280_315_of_diluted_wines: float = Field(..., ge=0, alias="od280/od315_of_diluted_wines")
    proline: float = Field(..., ge=0)

    class Config:
        populate_by_name = True

class PredictResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probabilities: List[float]

app = FastAPI(title=APP_TITLE)

def _load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

MODEL = _load_model()

@app.get("/")
def root():
    return {"status": "ok", "message": "Wine prediction backend is running."}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        row = np.array([[
            payload.alcohol,
            payload.malic_acid,
            payload.ash,
            payload.alcalinity_of_ash,
            payload.magnesium,
            payload.total_phenols,
            payload.flavanoids,
            payload.nonflavanoid_phenols,
            payload.proanthocyanins,
            payload.color_intensity,
            payload.hue,
            payload.od280_315_of_diluted_wines,
            payload.proline,
        ]], dtype=float)

        probs = MODEL.predict_proba(row)[0].tolist()
        pred = int(np.argmax(probs))
        return PredictResponse(
            predicted_class=pred,
            predicted_label=CLASS_NAMES.get(pred, f"class_{pred}"),
            probabilities=probs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
