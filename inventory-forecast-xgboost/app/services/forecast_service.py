import math
from pathlib import Path

import joblib
import numpy as np

from app.config import get_settings
from app.schemas import ForecastRequest

FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_30",
    "day_of_week",
    "month",
    "promo_flag",
    "stock_level",
]


class ForecastService:
    def __init__(self) -> None:
        settings = get_settings()
        path = Path(settings.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Forecast model not found: {path}")
        self.model = joblib.load(path)
        self.settings = settings

    def predict(self, payload: ForecastRequest) -> dict:
        features = np.array([[getattr(payload, col) for col in FEATURE_COLUMNS]])
        demand = float(self.model.predict(features)[0])
        demand = max(demand, 0.0)

        lead_time = self.settings.lead_time_days
        z = self.settings.service_level_z
        safety_stock = z * payload.demand_std * math.sqrt(lead_time)
        reorder_point = demand * lead_time + safety_stock

        return {
            "predicted_demand": round(demand, 2),
            "safety_stock": round(safety_stock, 2),
            "reorder_point": round(reorder_point, 2),
        }
