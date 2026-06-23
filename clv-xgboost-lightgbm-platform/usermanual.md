# User Manual - CLV XGBoost LightGBM Platform

## Overview

Predicts Customer Lifetime Value (CLV) using XGBoost and LightGBM regressors with optional ensemble averaging.

## End-to-End Flow

1. Train models: `python ml/training/train_models.py`
2. Start API: `uvicorn app.main:app --host 0.0.0.0 --port 8001`
3. Submit prediction: `POST /v1/clv/predict`
4. Retrieve audit record: `GET /v1/clv/predictions/{prediction_id}`

## Configuration

Copy `.env.example` to `.env`:

- `DATABASE_URL` - SQLAlchemy connection string
- `XGBOOST_MODEL_PATH` - XGBoost artifact path
- `LIGHTGBM_MODEL_PATH` - LightGBM artifact path
- `DEFAULT_MODEL` - `ensemble`, `xgboost`, or `lightgbm`

## Sample Request

```bash
curl -X POST "http://127.0.0.1:8001/v1/clv/predict" \
  -H "Content-Type: application/json" \
  -d "{\"customer_id\":\"C1001\",\"purchase_count\":42,\"avg_order_value\":85.5,\"tenure_months\":24,\"recency_days\":12,\"frequency_per_month\":1.75,\"product_categories\":6}"
```

## Model Selection

Set `"model": "xgboost"` or `"model": "lightgbm"` in request body. Omit for default ensemble.

## Production Checklist

- Use PostgreSQL for `DATABASE_URL`
- Schedule periodic retraining with fresh customer features
- Monitor MAE/R2 from `ml/artifacts/training_metrics.json`
- Add API authentication and rate limiting
