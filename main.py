from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Sensors(BaseModel):
    temperature: float
    vibration:float

@app.get("/health")
def health():
    return{"ok": True}

@app.post("/predict")
def predict(data: Sensors):
    return{"risk":0.5, "status":"fake"}