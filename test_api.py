from fastapi.testclient import TestClient

from api import app, PredictRequest

client = TestClient(app)

# The interfaces should contain basic tests for incorrect URLs, data, etc.
# Your APIs should perform basic tests for incorrect URLs parameters, headers, data,

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


def test_correct_prediction():
    features_ = {'age': 65.0, 'anaemia': 1.0, 'creatinine_phosphokinase': 52.0, 'diabetes': 0.0, 'ejection_fraction': 25.0, 'high_blood_pressure': 1.0, 'platelets': 276000.0, 'serum_creatinine': 1.3, 'serum_sodium': 137.0, 'sex': 0.0, 'smoking': 0.0, 'time': 16.0, 'DEATH_EVENT': 0.0}
    response = client.post("/predict", json={"features": features_})

    assert response.status_code == 200
    assert response.json() == {"predictions": [{"probability": 0.40215089157255884}], "error": None}


def test_missing_column():
    features_ = {'age': 65.0, 'anaemia': 1.0, 'creatinine_phosphokinase': 52.0, 'diabetes': 0.0, 'ejection_fraction': 25.0, 'high_blood_pressure': 1.0, 'platelets': 276000.0, 'serum_creatinine': 1.3, 'serum_sodium': 137.0, 'sex': 0.0, 'smoking': 0.0, 'time': 16.0, 'DEATH_EVENT': 0.0}
    features_.pop("age")
    response = client.post("/predict", json={"features": features_})

    assert response.status_code == 200
    assert response.json() == {"predictions": None, "error": "Incorrect columns provided!"}
