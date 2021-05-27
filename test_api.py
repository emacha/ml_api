from fastapi.testclient import TestClient

from api import app, PredictRequest
import training
import pytest
import json
from pathlib import Path


client = TestClient(app)


@pytest.fixture(scope="function")
def sample_data():
    return json.loads(Path("sample_data.json").read_text())


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"predictions": None, "error": "This is a test endpoint."}


def test_incorrect_url():
    response = client.get("/no_such")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_wrong_method():
    response = client.get("/predict")
    assert response.status_code == 200
    assert response.json() == {"error": "Send a POST request to this endpoint with 'features' data.", "predictions": None}


def test_correct_prediction(sample_data):
    response = client.post("/predict", json={"features": sample_data})

    assert response.status_code == 200
    assert response.json() == {"predictions": [{"probability": 0.40215089157255884}], "error": None}


def test_missing_column(sample_data):
    sample_data.pop("age")
    response = client.post("/predict", json={"features": sample_data})

    assert response.status_code == 200
    assert response.json() == {"predictions": None, "error": "Incorrect columns provided!"}


def test_bad_data_type(sample_data):
    sample_data["age"] = "not a number"
    response = client.post("/predict", json={"features": sample_data})

    # FastAPI does the validation
    assert response.status_code == 422


def test_model_saving_loading(tmp_path, sample_data):
    model = training.train_model()
    path = tmp_path / "ensemble_model"
    model.save(path)
    loaded_model = training.EnsembleModel.load(path)

    row = PredictRequest(features=sample_data)

    assert model.predict(row) == loaded_model.predict(row)
