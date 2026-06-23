# CLV XGBoost LightGBM Platform

Production-ready Customer Lifetime Value prediction platform with dual-model inference.

## Features

- XGBoost and LightGBM CLV regression models
- Ensemble, XGBoost-only, or LightGBM-only prediction modes
- FastAPI REST service with SQLite/PostgreSQL persistence
- Reproducible training pipeline with synthetic data fallback

## Quick Start

```bash
pip install -r requirements.txt
python ml/training/train_models.py
uvicorn app.main:app --reload --port 8001
```

API docs: http://127.0.0.1:8001/docs

See `usermanual.md` for full end-to-end operations guide.
