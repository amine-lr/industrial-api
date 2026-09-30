from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import sqlite3, time
import json as js

app = FastAPI()
con = sqlite3.connect("predictions.db", check_same_thread=False)
con.execute("""CREATE TABLE IF NOT EXISTS predictions
(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, inputs TEXT, risk REAL)""")
con.commit()
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
    con.execute("INSERT INTO predictions (ts, inputs, risk) VALUES (?,?,?)",
            (time.time(), js.dumps(X[0]), risk))
    con.commit()
    return {"risk": risk, "status": "model"}
@app.get("/history")
def history(limit: int = 10):
    cur = con.execute("SELECT ts, inputs, risk FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
    return [{"ts": r[0], "inputs": r[1], "risk": r[2]} for r in cur.fetchall()]