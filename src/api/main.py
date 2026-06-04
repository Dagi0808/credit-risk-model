from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import pandas as pd
from src.api.pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(title="Credit Risk Prediction API")

model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = mlflow.sklearn.load_model("models:/Credit_Risk_Best_Model/latest")
        print("✅ Model loaded successfully from MLflow")
    except Exception as e:
        print(f"⚠️ Warning: Could not load model: {e}")


@app.get("/")
async def root():
    return {"message": "Credit Risk Prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please try again later.")

    try:
        input_df = pd.DataFrame([request.dict()])
        probability = model.predict_proba(input_df)[0][1]
        prediction = 1 if probability >= 0.5 else 0

        return PredictionResponse(
            risk_probability=round(float(probability), 4),
            is_high_risk=bool(prediction),
            message="High Risk" if prediction == 1 else "Low Risk"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))