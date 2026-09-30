# 📖 Full Guide — every concept in industrial-api, with examples

> For 42 students and recruiters. Read top to bottom, run each block yourself.
> Project: `~/personal_projects/industrual-api` · Dataset: `data/raw.csv` (AI4I 10k)

---

## 0. Mental map (C → Python)

| 42 C world | This project | Why |
|---|---|---|
| `a.out` + terminal | `uvicorn` + HTTP + JSON | programs talk over internet, not just terminal |
| `struct s_data {float t;}` | `{"air_temp": 298.1}` JSON + Pydantic `Sensors` | text format, every language reads it |
| `prog1 \| prog2` (PIPEX) | `sensor → POST /predict → risk` | same pipe idea, bigger pipe (HTTP) |
| `Makefile` deps | `requirements.txt` + `venv/` | isolated lib box per project |
| `if (!is_number) return` | FastAPI auto-validation (422) | free input check before your code runs |
| VM in born2beroot | `Dockerfile` | light VM: code + libs packed as image |

---

## 1. M0 — Setup: venv, requirements, git

### 1.1 `venv/` — isolated Python box
```bash
python3 -m venv venv
source venv/bin/activate
# prompt becomes (venv) ... — proof you're inside the box
which python   # → .../industrual-api/venv/bin/python (not /usr/bin)
```
Why: each project has its own libs. In C you have 1 global `gcc`. In Python `fastapi==x` in one project must not break another.

Snapshot you saw:
```
(venv) void ~/personal_projects/industrual-api $
```

### 1.2 `requirements.txt` — deps list
```
fastapi
uvicorn
pandas
matplotlib
jupyter
scikit-learn
```
Install: `pip install -r requirements.txt` · Check: `pip list | grep pandas` → `pandas 3.0.6`
Like `Makefile` `LIBS`, but for Python.

### 1.3 `.gitignore` — what git ignores
```
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
predictions.db
```
Why: never commit venv (hundreds of MB), cache, local DB. Same as ignoring `a.out` in C.

---

## 2. M1 — Hello API: FastAPI + uvicorn

### 2.1 The 3 actors
```
Client (curl/browser/factory) ──HTTP──▶ uvicorn (waiter, infinite loop on :8000)
                                         ──▶ FastAPI (chef, picks function by URL)
                                                ──▶ your function → return dict → JSON
```
- **uvicorn**: `while(1) accept()` you don't write. `uvicorn main:app --reload` means file `main.py`, var `app`, restart on save (dev only).
- **FastAPI**: router + validator + doc generator.
- **Client**: anything sending text.

### 2.2 Minimal `main.py` (M1 fake)
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Sensors(BaseModel):
    temperature: float
    vibration: float

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(data: Sensors):
    return {"risk": 0.5, "status": "fake"}
```
- `@app.get("/health")` = register function for that URL (function-pointer table).
- `data: Sensors` = auto-check types. Send `"temperature": "hot"` → FastAPI returns `422` before your code. In C you'd `if` manually.
- `return dict` → FastAPI converts to JSON + status 200.
- Free test UI at `/docs`.

### 2.3 Run + prove (snapshots from your terminal)
Terminal 1:
```bash
uvicorn main:app --reload
# INFO: Uvicorn running on http://127.0.0.1:8000
```
Terminal 2:
```bash
curl http://127.0.0.1:8000/health
# {"ok":true}

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature":90,"vibration":0.6}'
# {"risk":0.5,"status":"fake"}
```
Error you hit and fixed:
```
ERROR: Attribute "app" not found in module "main"
→ causes: wrong folder, file not saved, `App` vs `app`. Fix: `pwd; ls; cat main.py`
```

---

## 3. M2 — Data: pandas, EDA, imbalance

### 3.1 Get data
```bash
mkdir -p data notebooks scripts
# download AI4I 2020 predictive maintenance zip from UCI/Kaggle
python3 -c "import zipfile; zipfile.ZipFile('~/Downloads/ai4i+2020+predictive+maintenance+dataset.zip').extract('ai4i2020.csv','/tmp')"
cp /tmp/ai4i2020.csv data/raw.csv
ls -lh data/  # → raw.csv 510K
head -2 data/raw.csv
# UDI,Product ID,Type,Air temperature [K],...,Torque [Nm],Tool wear [min],Machine failure,...
```

### 3.2 `df = pd.read_csv()` — Excel in 1 line
In `notebooks/01_eda.ipynb`:
```python
import pandas as pd
df = pd.read_csv("../data/raw.csv")
df.head()   # first 5 machines
```
`../` = from `notebooks/` go up, into `data/`.

### 3.3 `df.shape` — size
```python
df.shape
# (10000, 14)  → 10k rows, 14 cols. Confirms right file.
```

### 3.4 `value_counts()` — imbalance (key interview word)
```python
df["Machine failure"].value_counts()
# 0    9661
# 1     339
```
→ `339/10000 = 3.39%` fail. **Imbalanced, normal.** Factories rarely break. Never train blind without saying this.

### 3.5 Graphs — see separation
```python
import matplotlib.pyplot as plt
df["Machine failure"].value_counts().plot(kind="bar"); plt.show()

df[df["Machine failure"]==0]["Torque [Nm]"].hist(alpha=0.5, label="ok")
df[df["Machine failure"]==1]["Torque [Nm]"].hist(alpha=0.5, label="fail")
plt.legend(); plt.title("Torque ok vs fail"); plt.show()
# repeat for "Tool wear [min]" (best signal) and "Air temperature [K]" (weak)
```
Lesson: **Torque + Tool wear separate best. Temperature overlaps → weak alone.**

### 3.6 `isnull().sum()` — NULL check
```python
df.isnull().sum()
# every col → 0. No missing → "no imputation needed" (recruiter sentence).
```

### 3.7 Jupyter fix you lived (worth remembering)
```
jupyter notebook → no output, exit instantly
→ cause: venv metadata corrupted, 70 pkgs Version: None
   ImportError: Instance / ensure_async / TemplateNotFound in chain
→ fix: rm -rf venv; python3 -m venv venv; pip install fastapi uvicorn pandas matplotlib scikit-learn jupyter
→ verify: jupyter notebook --ip=127.0.0.1 --no-browser --port=8889 → token URL prints
```
Also: suspended `uvicorn` holds :8000 → `jobs; kill %1; jobs` (empty = free).

---

## 4. M3 — Model: train/test, RandomForest, metrics

### 4.1 Block 1 — load + stratify split (`scripts/train.py`)
```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw.csv")
df["Type"] = df["Type"].map({"L":0, "M":1, "H":2})  # letters → numbers, model needs numbers

FEATURES = ["Type","Air temperature [K]","Process temperature [K]",
            "Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]
X = df[FEATURES]; y = df["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```
Snapshot:
```
train (8000, 6) test (2000, 6)
0    0.966
1    0.034
```
- `test_size=0.2` = hide 20% for exam (2000 rows). Like piscine vs exam.
- `stratify=y` = keep 3.4% fail in both sets. Without it test could have 0 fails → fake 100%.
- Typo you fixed: `df =["Type"] = ...` → `SyntaxError`. Right: `df["Type"] = ...`.

### 4.2 Block 2 — RandomForest + report + save
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib, json

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(classification_report(y_test, pred))
joblib.dump(clf, "model.joblib")
```
- **RandomForest** = 100 decision trees vote, majority wins. Like 100 peers reviewing. No GPU, robust for factories.
- **`model.joblib`** (2.6M) = saved brain. Train once, API loads each start. No retrain per request.
- **`metrics.json`** = report card for README/CV.

Your real snapshot:
```
              precision    recall  f1-score   support
           0       0.99      1.00      0.99      1932
           1       0.88      0.62      0.72        68
    accuracy                           0.98      2000
```
Read it: precision 0.88 = alarms 88% true. Recall 0.62 = catch 42/68, miss 26. Accuracy 0.98 lies (always say "ok" → 96%). Quote precision/recall in interviews.

### 4.3 Wire to `main.py` (kill fake 0.5)
```python
import joblib
model = joblib.load("model.joblib")

class Sensors(BaseModel):
    Type: int; air_temp: float; process_temp: float
    speed: float; torque: float; tool_wear: float

@app.post("/predict")
def predict(s: Sensors):
    X = [[s.Type, s.air_temp, s.process_temp, s.speed, s.torque, s.tool_wear]]
    risk = float(model.predict_proba(X)[0][1])  # proba of class 1
    return {"risk": risk, "status": "model"}
```
Proof snapshots:
```bash
# healthy
curl -X POST ... -d '{"Type":1,"air_temp":298.1,...,"tool_wear":0}'
# {"risk":0.0,"status":"model"}
# worn
curl -X POST ... -d '{"Type":1,...,"torque":70.0,"tool_wear":250}'
# {"risk":0.88,"status":"model"}
```
Low < high → fake killed.

---

## 5. M4 — Pro: SQLite, pytest, Docker

### 5.1 SQLite log (`predictions.db`)
```python
import sqlite3, time, json as js
con = sqlite3.connect("predictions.db", check_same_thread=False)
con.execute("""CREATE TABLE IF NOT EXISTS predictions
(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, inputs TEXT, risk REAL)""")
con.commit()
# in predict(), before return:
con.execute("INSERT INTO predictions (ts,inputs,risk) VALUES (?,?,?)",
            (time.time(), js.dumps(X[0]), risk)); con.commit()

@app.get("/history")
def history(limit: int = 10):
    cur = con.execute("SELECT ts,inputs,risk FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
    return [{"ts":r[0],"inputs":r[1],"risk":r[2]} for r in cur.fetchall()]
```
- SQLite = file DB, no server. `check_same_thread=False` because FastAPI uses threads.
- Prove: `curl "http://127.0.0.1:8000/history?limit=2"` (quotes needed in zsh for `?`) → 2 rows.
- Your snapshot: `[{"risk":0.88,...},{"risk":0.0,...}]` ✅

### 5.2 pytest (3 tests)
`tests/test_api.py`:
```python
from fastapi.testclient import TestClient
from main import app
c = TestClient(app)
def test_health(): assert c.get("/health").json() == {"ok": True}
def test_healthy_low(): assert c.post("/predict", json={...wear:0}).json()["risk"] < 0.3
def test_worn_high(): assert c.post("/predict", json={...wear:250}).json()["risk"] > 0.5
```
Run: `PYTHONPATH=. pytest -v` → `3 passed`. Why `PYTHONPATH=.`? pytest from `tests/` can't see root `main.py` without it. Error you fixed: `ModuleNotFoundError: No module named 'main'`.

### 5.3 Dockerfile (prod image)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py model.joblib ./
EXPOSE 8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
```
- Like born2beroot VM but light. `--host 0.0.0.0` = listen on all interfaces (Docker needs it, localhost fails inside container).
- Your env has no `docker` daemon → validated by `uvicorn --host 0.0.0.0` + curl + file review instead. Accepted for internship.
- `.dockerignore`: `venv/ __pycache__/ notebooks/ data/ .git/ predictions.db` → small image.

---

## 6. Git hygiene you applied

```bash
git add scripts/train.py main.py requirements.txt metrics.json .gitignore
git status   # caught junk: .ipynb_checkpoints, __pycache__
git restore --staged notebooks/.ipynb_checkpoints/...
echo ".ipynb_checkpoints/" >> .gitignore
git commit -m "M3: real RandomForest precision 0.88 recall 0.62"
git push  # → aee820d, 661KB. model.joblib 2.6M ok (<25MB GitHub limit)
```

---

## 7. 30-second interview script (memorize)

> "I built a predictive-maintenance API. FastAPI exposes health, predict, history. RandomForest trained 80/20 stratify on 10k machines, 3.4% fails. Precision 88, recall 62. Every prediction logged to SQLite. Tested with pytest, Dockerized. Best signals tool wear and torque from EDA."

If you can say that + draw sensor→API→model→DB on paper, you own this project.
