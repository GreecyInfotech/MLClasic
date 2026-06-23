# User Manual - Inventory Forecast XGBoost

## Overview

Forecasts product demand using XGBoost and computes inventory planning metrics.

## End-to-End Flow

1. Train: `python ml/training/train_inventory_forecast.py`
2. Run API: `uvicorn app.main:app --port 8003`
3. Forecast: `POST /v1/forecast`
4. Retrieve: `GET /v1/forecast/{forecast_id}`

## Sample Request

```bash
curl -X POST "http://127.0.0.1:8003/v1/forecast" \
  -H "Content-Type: application/json" \
  -d "{\"product_id\":\"P100\",\"forecast_date\":\"2026-06-23\",\"lag_1\":120,\"lag_7\":110,\"lag_30\":95,\"day_of_week\":2,\"month\":6,\"promo_flag\":1,\"stock_level\":400,\"demand_std\":8}"
```

## Outputs

- `predicted_demand` - model forecast for the period
- `safety_stock` - buffer based on service level and lead time
- `reorder_point` - when to trigger replenishment

## Configuration

- `SERVICE_LEVEL_Z` - z-score for safety stock (default 1.65 = ~95%)
- `LEAD_TIME_DAYS` - supplier lead time in days
