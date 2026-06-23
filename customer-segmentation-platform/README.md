# Customer Segmentation Platform

K-Means customer segmentation with FastAPI inference and persistence.

## Quick Start

```bash
pip install -r requirements.txt
python ml-platform/training/train_kmeans.py
uvicorn app.main:app --reload --port 8002
```

See `usermanual.md` for full guide.
