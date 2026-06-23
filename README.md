# MLClasic

Production-ready machine learning platforms monorepo.

## Projects

| Project | Description | Port |
|---------|-------------|------|
| [ai-loan-approval-system](./ai-loan-approval-system) | Loan approval with XGBoost + fraud/risk scoring | 8000 |
| [clv-xgboost-lightgbm-platform](./clv-xgboost-lightgbm-platform) | Customer Lifetime Value (XGBoost + LightGBM) | 8001 |
| [customer-segmentation-platform](./customer-segmentation-platform) | K-Means customer segmentation | 8002 |
| [inventory-forecast-xgboost](./inventory-forecast-xgboost) | Demand forecasting + safety stock | 8003 |
| [next-best-offer-gradient-boosting](./next-best-offer-gradient-boosting) | Offer recommendation ranking | 8004 |
| [container-intelligence-platform](./container-intelligence-platform) | Container dwell time + risk scoring | 8005 |
| [ml-domain-models-bundle](./ml-domain-models-bundle) | Unified API for 6 domain ML models | 8006 |

## Quick Start (any project)

```bash
cd <project-folder>
pip install -r requirements.txt
python <train-script>   # see project usermanual.md
uvicorn app.main:app --reload --port <port>
```

## Production

See [PRODUCTION.md](./PRODUCTION.md) for Docker, health probes, PostgreSQL, and deployment across all platforms.

## Repository

https://github.com/GreecyInfotech/MLClasic
