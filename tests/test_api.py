from fastapi.testclient import TestClient
from main import app
c = TestClient(app)
def test_health():
    assert c.get("/health").json() == {"ok": True}
def test_healthy_low():
    r = c.post("/predict", json={"Type":1,"air_temp":298.1,"process_temp":308.6,"speed":1551,"torque":42.8,"tool_wear":0}).json()
    assert r["risk"] < 0.3
def test_worn_high():
    r = c.post("/predict", json={"Type":1,"air_temp":300.0,"process_temp":310.0,"speed":1400,"torque":70.0,"tool_wear":250}).json()
    assert r["risk"] > 0.5