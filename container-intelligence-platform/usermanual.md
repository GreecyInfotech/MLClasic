# User Manual - Container Intelligence Platform

## Overview

End-to-end container intelligence workflow:
1. Register/analyze container
2. Predict yard dwell time
3. Compute risk score and flags
4. Assign operational status

## End-to-End Flow

1. Train dwell model: `python ml/training/train_dwell_time.py`
2. Run API: `uvicorn app.main:app --port 8005`
3. Analyze container: `POST /v1/containers/analyze`
4. Fetch record: `GET /v1/containers/{container_number}`

## Sample Request

```bash
curl -X POST "http://127.0.0.1:8005/v1/containers/analyze" \
  -H "Content-Type: application/json" \
  -d "{\"container_number\":\"MSCU1234567\",\"container_type\":\"reefer\",\"weight_tons\":22.5,\"origin_port\":\"SGSIN\",\"destination_port\":\"USLAX\",\"customs_hold\":false,\"yard_congestion\":0.62,\"vessel_delay_hours\":8}"
```

## Status Outcomes

- `at_yard` - normal processing
- `customs_review` - customs hold active
- `high_risk_hold` - risk score above threshold
- `in_transit`, `released` - lifecycle states

## Configuration

- `DWELL_MODEL_PATH` - dwell time model artifact
- `RISK_THRESHOLD` - threshold for high-risk hold (default 0.7)
