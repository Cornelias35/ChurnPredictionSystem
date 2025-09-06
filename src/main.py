import os
from dotenv import load_dotenv
import logging
from fastapi import FastAPI
from .serving import ModelResponse, PredictionResponse, PredictionRequest, TrainingRequest, AVAILABLE_MODELS, AVAILABLE_METRICS
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_dotenv(dotenv_path="../.env")
        os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
    except FileNotFoundError:
        logging.error('Could not find WANDB_API_KEY')
        raise FileNotFoundError
    yield

app = FastAPI(title="Customer Churn Prediction", version="1.0", lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "ML Model Training and Prediction API"}

@app.get("/models")
async def get_available_models():
    """Get list of available models"""
    return {"available_models": list(AVAILABLE_MODELS.keys())}

@app.get("/metrics")
async def get_available_metrics():
    """Get list of available metrics for optimization"""
    return {"available_metrics": AVAILABLE_METRICS}

@app.post("/train_model")
def train_model_endpoint(training_request: TrainingRequest) -> ModelResponse:
    """Train a model"""
    # doing something
@app.post("/prediction")
def prediction_endpoint(prediction_request: PredictionRequest) -> PredictionResponse:
    """Prediction"""
    #doing something