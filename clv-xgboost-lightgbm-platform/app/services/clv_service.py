from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from app.config import get_settings
from app.schemas import CLVPredictionRequest, ModelChoice

FEATURE_COLUMNS = [
    "purchase_count",
    "avg_order_value",
    "tenure_months",
    "recency_days",
    "frequency_per_month",
    "product_categories",
]


class CLVModelService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.xgb_model = self._load_xgboost(self.settings.xgboost_model_path)
        self.lgb_model = self._load_lightgbm(self.settings.lightgbm_model_path)

    @staticmethod
    def _load_xgboost(path: str):
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost model not found: {path}")
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        return model

    @staticmethod
    def _load_lightgbm(path: str):
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"LightGBM model not found: {path}")
        return lgb.Booster(model_file=str(model_path))

    def _to_features(self, payload: CLVPredictionRequest) -> np.ndarray:
        return np.array(
            [[getattr(payload, col) for col in FEATURE_COLUMNS]], dtype=float
        )

    def predict(self, payload: CLVPredictionRequest, model: ModelChoice) -> dict:
        features = self._to_features(payload)
        xgb_pred = float(self.xgb_model.predict(features)[0])
        lgb_pred = float(self.lgb_model.predict(features)[0])

        if model == ModelChoice.xgboost:
            final = xgb_pred
        elif model == ModelChoice.lightgbm:
            final = lgb_pred
        else:
            final = (xgb_pred + lgb_pred) / 2.0

        return {
            "predicted_clv": round(final, 2),
            "xgboost_clv": round(xgb_pred, 2),
            "lightgbm_clv": round(lgb_pred, 2),
            "model_name": model.value,
        }
