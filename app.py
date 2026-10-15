from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(
    title="SCSE3040 MLOps Delivery Time Prediction API",
    description="Production REST API serving the Delivery Time Prediction Model for Practical 08 — SCSE3040.",
    version="1.0.0",
    servers=[
        {"url": "https://scse3040-p08-s24cseu0807.onrender.com", "description": "Cloud Production Server (Render)"},
        {"url": "https://2873-104-28-193-14.ngrok-free.app", "description": "Cloud Edge Ingress Gateway"}
    ]
)

# Enable CORS for external client accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load serialized model artifact
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


class OrderInput(BaseModel):
    distance_km: float = Field(
        ...,
        gt=0,
        le=50,
        description="Delivery trip distance in kilometers (Must be > 0 and <= 50)",
        examples=[5.0]
    )
    prep_time_min: float = Field(
        ...,
        ge=0,
        le=120,
        description="Kitchen preparation time in minutes (Must be >= 0 and <= 120)",
        examples=[20.0]
    )
    traffic_level: int = Field(
        ...,
        ge=1,
        le=3,
        description="Current traffic congestion level: 1 (Low), 2 (Medium), 3 (High)",
        examples=[2]
    )
    rain: int = Field(
        ...,
        ge=0,
        le=1,
        description="Precipitation flag: 0 (No Rain / Clear), 1 (Raining)",
        examples=[0]
    )


class PredictionResponse(BaseModel):
    predicted_delivery_min: float = Field(..., description="Estimated delivery time in minutes")
    status: str = Field(default="success", description="Response status code label")
    input_data: OrderInput = Field(..., description="Echo of validated input features")


@app.get("/", tags=["General"])
def root():
    return {
        "service": "SCSE3040 MLOps Delivery Time Prediction Service",
        "author": "Divakar Maurya (Roll: S24CSEU0807, Batch: EB31)",
        "course": "SCSE3040 — Machine Learning Operations",
        "practical": "Practical 08: Model Deployment to Cloud Platform",
        "documentation": "/docs",
        "health_check": "/health",
        "predict_endpoint": "/predict"
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model weights not loaded"
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "framework": "scikit-learn LinearRegression",
        "environment": "cloud-production"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(order: OrderInput):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently unavailable."
        )

    # Convert Pydantic model to DataFrame matching model training schema
    features = pd.DataFrame([{
        "distance_km": order.distance_km,
        "prep_time_min": order.prep_time_min,
        "traffic_level": order.traffic_level,
        "rain": order.rain
    }])

    # Generate inference
    prediction = float(model.predict(features)[0])
    prediction_rounded = round(prediction, 2)

    return PredictionResponse(
        predicted_delivery_min=prediction_rounded,
        status="success",
        input_data=order
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
