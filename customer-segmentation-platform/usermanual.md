# User Manual - Customer Segmentation Platform

## Overview

Segments customers using K-Means on RFM + demographic features.

## End-to-End Flow

1. Train: `python ml-platform/training/train_kmeans.py`
2. Run API: `uvicorn app.main:app --port 8002`
3. Segment customer: `POST /v1/segmentation/predict`
4. Fetch result: `GET /v1/segmentation/results/{result_id}`

## Sample Request

```bash
curl -X POST "http://127.0.0.1:8002/v1/segmentation/predict" \
  -H "Content-Type: application/json" \
  -d "{\"customer_id\":\"C2001\",\"recency\":15,\"frequency\":12,\"monetary\":2400,\"age\":34,\"income_band\":3}"
```

## Segment Labels

- `champions`, `loyal`, `at_risk`, `hibernating` (default 4 clusters)

## Configuration

- `N_CLUSTERS` - number of K-Means clusters
- `MODEL_PATH` - trained artifact path
- `DATABASE_URL` - persistence backend
