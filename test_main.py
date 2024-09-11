import pytest
from fastapi.testclient import TestClient
from main import app, ml_models  # Adjust the import based on your app's file name and structure
from unittest.mock import patch, MagicMock

# Initialize the TestClient
client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to my APP IRIS prediction!"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "up and running"}

def test_list_models():
    response = client.get("/models")
    assert response.status_code == 200
    assert "models" in response.json()

@pytest.fixture
def mock_model():

    model = MagicMock()
    model.predict.return_value = [1]  
    return model

def test_predict_valid_model(mock_model):
  
    ml_models["mock_model"] = mock_model


    valid_input = {
        "length_sepals": 5.0,
        "width_sepals": 3.5,
        "length_petals": 1.4,
        "width_petals": 0.2
    }

    response = client.post("/predict/mock_model", json=valid_input)
    assert response.status_code == 200
    assert response.json() == {"prediction": "Iris-versicolor"}

    # Clean up the mock model from ml_models
    del ml_models["mock_model"]

def test_predict_invalid_model():
   
    valid_input = {
        "length_sepals": 5.0,
        "width_sepals": 3.5,
        "length_petals": 1.4,
        "width_petals": 0.2
    }

    response = client.post("/predict/invalid_model", json=valid_input)
    assert response.status_code == 404
    assert response.json() == {"detail": "Model not found"}

def test_predict_with_mocked_model(mocker):
  
    mock_model = mocker.patch.dict(ml_models, {"mock_model": MagicMock()})
    mock_model["mock_model"].predict.return_value = [0]


    valid_input = {
        "length_sepals": 5.0,
        "width_sepals": 3.5,
        "length_petals": 1.4,
        "width_petals": 0.2
    }

    response = client.post("/predict/mock_model", json=valid_input)
    assert response.status_code == 200
    assert response.json() == {"prediction": "Iris-setosa"}

   
    del ml_models["mock_model"]
