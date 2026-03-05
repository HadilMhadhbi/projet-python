"""
FastAPI Application
Drug Consumption Prediction API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Drug Consumption Prediction API")

# Load model
model = None
if os.path.exists('data/best_model.pkl'):
    model = joblib.load('data/best_model.pkl')

class PredictionInput(BaseModel):
    Age: float
    Gender: int
    Education: int
    Country: int
    Ethnicity: int
    Nscore: float
    Escore: float
    Oscore: float
    Ascore: float
    Cscore: float
    Impulsive: float
    SS: float

@app.get("/")
def root():
    return {"message": "Drug Consumption Prediction API", "status": "active"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    features = np.array([[
        input_data.Age, input_data.Gender, input_data.Education,
        input_data.Country, input_data.Ethnicity, input_data.Nscore,
        input_data.Escore, input_data.Oscore, input_data.Ascore,
        input_data.Cscore, input_data.Impulsive, input_data.SS
    ]])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "risk_level": "Low" if probability < 0.3 else "Medium" if probability < 0.7 else "High"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)