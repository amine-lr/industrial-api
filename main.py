from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.joblib")

class Sensors(BaseModel):
    Type: int
    air_temp: float
    process_temp: float
    speed: float
    torque: float
    tool_wear: float

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(s: Sensors):
    X = [[s.Type, s.air_temp, s.process_temp, s.speed, s.torque, s.tool_wear]]
    risk = float(model.predict_proba(X)[0][1])
    return {"risk": risk, "status": "model"}