# Inventory Forecast XGBoost

Demand forecasting with safety stock and reorder point calculation.

## Quick Start

```bash
pip install -r requirements.txt
python ml/training/train_inventory_forecast.py
uvicorn app.main:app --reload --port 8003
```

See `usermanual.md` for full guide.
