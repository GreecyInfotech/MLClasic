# ML Domain Models Bundle

Unified production API for six domain ML models.

## Models

- `berth-occupancy-lightgbm` - port berth occupancy prediction
- `claim-settlement-random-forest` - insurance claim approval
- `customer-risk-scoring-gradient-boosting` - customer risk score
- `fraud-risk-score-gradient-boosting` - fraud detection
- `suspicious-transactions-autoencoder` - transaction anomaly detection
- `vessel-delay-prediction-xgboost` - vessel delay hours

## Quick Start

```bash
pip install -r requirements.txt
python ml/training/train_all.py
uvicorn app.main:app --reload --port 8006
```

See `usermanual.md` for full guide.
