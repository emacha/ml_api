from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict
import training
from pathlib import Path


class PredictRequest(BaseModel):
    features: Dict[str, float]


class ModelResponse(BaseModel):
    predictions: Optional[List[Dict[str, float]]]
    error: Optional[str]


app = FastAPI()

@app.get("/", response_model=ModelResponse)
async def root() -> ModelResponse:
    return ModelResponse(error="This is a test endpoint.")


@app.get("/predict", response_model=ModelResponse)
async def explain_api() -> ModelResponse:
    return ModelResponse(
        error="Send a POST request to this endpoint with 'features' data."
    )


@app.post("/predict")
async def get_model_predictions(request: PredictRequest) -> ModelResponse:
    model = training.EnsembleModel.load(Path("ensemble_model"))

    try:
        prediction = model.predict(request)
    except training.IncorrectColumnsError:
        return ModelResponse(error="Incorrect columns provided!")    

    return ModelResponse(predictions=[{"probability": prediction}])
