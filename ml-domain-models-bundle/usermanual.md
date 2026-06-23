# User Manual - ML Domain Models Bundle

## Overview

Single FastAPI service exposing six domain-specific ML models with audit logging.

## End-to-End Flow

1. Train all models: `python ml/training/train_all.py`
2. Start API: `uvicorn app.main:app --port 8006`
3. Call model endpoint
4. Retrieve audit record: `GET /v1/predictions/{prediction_id}`

## Endpoints

| Model | Endpoint |
|-------|----------|
| Berth Occupancy | `POST /v1/models/berth-occupancy/predict` |
| Claim Settlement | `POST /v1/models/claim-settlement/predict` |
| Customer Risk | `POST /v1/models/customer-risk/predict` |
| Fraud Risk | `POST /v1/models/fraud-risk/predict` |
| Suspicious Transactions | `POST /v1/models/suspicious-transactions/predict` |
| Vessel Delay | `POST /v1/models/vessel-delay/predict` |

## Sample Requests

### Berth Occupancy
```bash
curl -X POST "http://127.0.0.1:8006/v1/models/berth-occupancy/predict" \
  -H "Content-Type: application/json" \
  -d "{\"vessel_count\":12,\"berth_capacity\":16,\"hour\":14,\"day_of_week\":2}"
```

### Fraud Risk
```bash
curl -X POST "http://127.0.0.1:8006/v1/models/fraud-risk/predict" \
  -H "Content-Type: application/json" \
  -d "{\"transaction_amount\":4500,\"velocity_24h\":18,\"geo_mismatch\":0.8,\"merchant_risk\":0.6}"
```

### Vessel Delay
```bash
curl -X POST "http://127.0.0.1:8006/v1/models/vessel-delay/predict" \
  -H "Content-Type: application/json" \
  -d "{\"weather_score\":0.7,\"port_congestion\":0.55,\"cargo_volume_teu\":2200}"
```

## Health Check

`GET /health` returns loaded model list. Status is `ok` when all 6 models are loaded.

## Artifacts

Models are stored under `ml/artifacts/{model-name}/model.pkl`.

## Production Checklist

- Use PostgreSQL via `DATABASE_URL`
- Add API authentication per tenant/model
- Monitor per-model latency and error rates
- Schedule `train_all.py` retraining pipeline
