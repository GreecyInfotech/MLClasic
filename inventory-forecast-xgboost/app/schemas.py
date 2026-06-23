from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class ForecastRequest(BaseModel):
    product_id: str = Field(min_length=1, max_length=64)
    forecast_date: date
    lag_1: float = Field(ge=0)
    lag_7: float = Field(ge=0)
    lag_30: float = Field(ge=0)
    day_of_week: float = Field(ge=0, le=6)
    month: float = Field(ge=1, le=12)
    promo_flag: float = Field(ge=0, le=1)
    stock_level: float = Field(ge=0)
    demand_std: float = Field(default=5.0, ge=0)


class ForecastResponse(BaseModel):
    forecast_id: str
    product_id: str
    forecast_date: date
    predicted_demand: float
    safety_stock: float
    reorder_point: float
    timestamp: datetime


class ForecastRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    product_id: str
    forecast_date: date
    predicted_demand: float
    safety_stock: float
    reorder_point: float
    created_at: datetime
