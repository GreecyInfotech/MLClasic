# Container Intelligence Platform

Smart port container tracking with dwell-time prediction and risk scoring.

## Quick Start

```bash
pip install -r requirements.txt
python ml/training/train_dwell_time.py
uvicorn app.main:app --reload --port 8005
```

See `usermanual.md` for full guide.
