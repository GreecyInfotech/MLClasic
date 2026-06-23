from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelChoice(str, Enum):
    xgboost = "xgboost"
    lightgbm = "lightgbm"
    ensemble = "ensemble"


class CLVPredictionRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=64)
    purchase_count: float = Field(ge=0)
    avg_order_value: float = Field(gt=0)
    tenure_months: float = Field(ge=0, le=600)
    recency_days: float = Field(ge=0)
    frequency_per_month: float = Field(ge=0)
    product_categories: float = Field(ge=1, le=50)
    model: Optional[ModelChoice] = None


class CLVPredictionResponse(BaseModel):
    prediction_id: str
    customer_id: str
    predicted_clv: float
    xgboost_clv: float
    lightgbm_clv: float
    model_name: str
    timestamp: datetime


class CLVRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    customer_id: str
    purchase_count: float
    avg_order_value: float
    tenure_months: float
    recency_days: float
    frequency_per_month: float
    product_categories: float
    predicted_clv: float
    xgboost_clv: float
    lightgbm_clv: float
    model_name: str
    created_at: datetime
