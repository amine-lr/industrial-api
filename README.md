# industrial-api — Machine Failure Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white" alt="SQLite" />
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white" alt="pytest" />
</p>

<p align="center"><b>Sensors in → risk out.</b> FastAPI + scikit-learn API for predictive maintenance.<br/>Built at 42 Urduliz (Bilbao) — looking for internships in Bizkaia / Navarra.</p>

## What it does

Factories lose money when machines break without warning. This API warns **before** it breaks.

```text
Factory sensors --JSON--> POST /predict --> {"risk": 0.88, "status": "model"}
                                              ^
                                     RandomForest (100 trees)
                                     trained on 10,000 machines
```

|  | Healthy machine | Worn machine |
| --- | --- | --- |
| Input | `tool_wear: 0, torque: 42.8` | `tool_wear: 250, torque: 70` |
| Output | `{"risk": 0.0}` keep running | `{"risk": 0.88}` maintenance needed |

## Results (hidden 20% test set)

| Metric (failures = 1) | Score | Meaning |
| --- | --- | --- |
| Precision | **0.875** | When it says "will break", it is right 88% of the time |
| Recall | **0.618** | Of 68 real failures, it catches 42 and misses 26 |
| F1 | **0.724** | Balance of both |
| Accuracy | 0.98 | Misleading here — never quote it alone (only 3.39% fail) |

```text
              precision    recall  f1-score   support
           0       0.99      1.00      0.99      1932
           1       0.88      0.62      0.72        68
```

Dataset: AI4I 2020 Predictive Maintenance — 10,000 rows x 14 cols, 339 fails (3.39%), 0 missing values. Best signals from EDA: **Tool wear + Torque**. Temperature alone does not separate well.

> Full learning story with every concept explained: [docs/GUIDE.md](docs/GUIDE.md)

## Architecture

```text
sensor JSON --> FastAPI (main.py)
                |-- GET  /health   -> {"ok": true}
                |-- POST /predict  -> {"risk": ...}
                |-- GET  /history  -> last N logs
                        |                |
                        v                v
                 model.joblib      predictions.db
                 RandomForest      SQLite logs
                 100 trees         (ts, inputs, risk)
                        ^
                        |
                 scripts/train.py
                 train 80% / test 20% (stratified)
```

## Quickstart

Local (venv):

```bash
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs

Docker (prod-like):

```bash
docker build -t industrial-api .
docker run -p 8000:8000 industrial-api
```

Tests:

```bash
PYTHONPATH=. pytest -v
# 3 passed (health, healthy-low, worn-high)
```

## API reference

### GET /health

```bash
curl http://127.0.0.1:8000/health
# {"ok":true}
```

### POST /predict

```bash
# healthy — expect risk ~0.0
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Type":1,"air_temp":298.1,"process_temp":308.6,"speed":1551,"torque":42.8,"tool_wear":0}'
# {"risk":0.0,"status":"model"}

# worn — expect risk ~0.88
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Type":1,"air_temp":300.0,"process_temp":310.0,"speed":1400,"torque":70.0,"tool_wear":250}'
# {"risk":0.88,"status":"model"}
```

### GET /history

```bash
curl "http://127.0.0.1:8000/history?limit=2"
# [{"ts":...,"inputs":"[1, 300.0, ...]","risk":0.88},
#  {"ts":...,"inputs":"[1, 298.1, ...]","risk":0.0}]
```

Note for zsh users: quote the URL because of `?`.

## Project layout

```text
main.py                    FastAPI app: 3 routes + SQLite log + model load
scripts/train.py           train 80/20 stratified -> model.joblib + metrics.json
tests/test_api.py          3 tests: health, low-risk, high-risk
notebooks/01_eda.ipynb     EDA: shape, imbalance, graphs, missing check
data/raw.csv               AI4I 10k machines
model.joblib               trained RandomForest (~2.6 MB)
metrics.json               precision / recall / F1 for CV
Dockerfile                 python:3.11-slim prod image
docs/GUIDE.md              full concept guide with examples
```

## Roadmap

- [x] M0 setup: venv, requirements, git
- [x] M1 hello API: `/health` + fake `/predict`
- [x] M2 EDA: 10k rows, 3.39% fail, torque / tool-wear signals
- [x] M3 real model: RandomForest precision 0.88 / recall 0.62
- [x] M4 pro: SQLite log + pytest 3/3 + Dockerfile + 0.0.0.0 proof
- [ ] Next: `class_weight=balanced` + threshold tuning (recall -> 0.75?)
- [ ] Next: Postgres via docker-compose + Alembic migration
- [ ] Next: GitHub Actions CI + Prometheus `/metrics`

---

**Mohamed Amine Larioui** — 42 Urduliz Bilbao · [github.com/amine-lr](https://github.com/amine-lr)

Looking for: **practicas Bizkaia / Navarra** — Backend Python, Data, IA aplicada.
