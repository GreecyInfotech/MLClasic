# User Manual - Next Best Offer Gradient Boosting

## Overview

Recommends the best offer from a candidate list using a Gradient Boosting classifier.

## End-to-End Flow

1. Train: `python ml/training/train_gradient_boosting.py`
2. Run API: `uvicorn app.main:app --port 8004`
3. Recommend: `POST /v1/offers/recommend`
4. Retrieve: `GET /v1/offers/recommendations/{recommendation_id}`

## Sample Request

```bash
curl -X POST "http://127.0.0.1:8004/v1/offers/recommend" \
  -H "Content-Type: application/json" \
  -d "{\"customer_id\":\"C3001\",\"customer_age\":35,\"income\":72000,\"past_purchases\":18,\"channel\":\"email\",\"offers\":[{\"offer_id\":\"O1\",\"offer_type\":\"discount\",\"discount_pct\":15},{\"offer_id\":\"O2\",\"offer_type\":\"loyalty\",\"discount_pct\":5}]}"
```

## Offer Types

Supported encodings: `discount`, `bundle`, `loyalty`, `upgrade`, `cashback`

## Channels

`email`, `sms`, `app`, `web`
