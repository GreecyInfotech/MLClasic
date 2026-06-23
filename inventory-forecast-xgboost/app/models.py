import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class InventoryForecast(Base):
    __tablename__ = "inventory_forecast"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    product_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    forecast_date: Mapped[date] = mapped_column(Date, nullable=False)
    lag_1: Mapped[float] = mapped_column(Float, nullable=False)
    lag_7: Mapped[float] = mapped_column(Float, nullable=False)
    lag_30: Mapped[float] = mapped_column(Float, nullable=False)
    day_of_week: Mapped[float] = mapped_column(Float, nullable=False)
    month: Mapped[float] = mapped_column(Float, nullable=False)
    promo_flag: Mapped[float] = mapped_column(Float, nullable=False)
    stock_level: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_demand: Mapped[float] = mapped_column(Float, nullable=False)
    safety_stock: Mapped[float] = mapped_column(Float, nullable=False)
    reorder_point: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
