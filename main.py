from fastapi import FastAPI, HTTPException
import os
import joblib
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Dict
import asyncio
from dotenv import load_dotenv


from dotenv import load_dotenv
load_dotenv()


class PredictionInput(BaseModel):
    length_sepals: float = Field(..., description="length sepals", ge=4.3, le=7.9,  example=4.5)
    width_sepals: float = Field(..., description="width sepals", ge=2.0, le=4.4, example=3.0)
    length_petals: float = Field(..., description="length petals", ge=1.0, le=6.9,  example=3.0)
    width_petals: float = Field(..., description="width petals", ge=0.1, le=2.5, example=2.0)



class_names = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}

logreg_model_path = os.getenv("LOGREG_MODELP")
rf_model_path = os.getenv("RF_MODELP")


ml_models = {}

def load_models():
  
    print(f"logreg_model_path: {logreg_model_path}")
    print(f"rf_model_path: {rf_model_path}")


    logistic_model = joblib.load(logreg_model_path)
    random_forest_model = joblib.load(rf_model_path)

    return logistic_model, random_forest_model

asynccontextmanager
async def lifespan(app: FastAPI):
  
    logistic_model, random_forest_model = load_models()
    ml_models["logistic regression"] = logistic_model
    ml_models["random forest"] = random_forest_model
    try:
        yield
    finally:
      
        ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Welcome to my APP IRIS prediction!"}

@app.get("/health")
def health_check():
    return {"status": "up and running"}

@app.get("/models")
def list_models():
    return {"models": list(ml_models.keys())}


@app.post("/predict/{model_name}")
async def predict(model_name: str, input_data: PredictionInput):
    model = ml_models.get(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    await asyncio.sleep(5)
    features = [[input_data.length_sepals, input_data.width_sepals, input_data.length_petals, input_data.width_petals]]
    prediction = model.predict(features)
    predicted_class = class_names.get(prediction[0], "Unknown")
    return {"prediction": predicted_class}