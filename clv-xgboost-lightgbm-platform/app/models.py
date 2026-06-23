import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class CLVPrediction(Base):
    __tablename__ = "customer_clv"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    customer_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    purchase_count: Mapped[float] = mapped_column(Float, nullable=False)
    avg_order_value: Mapped[float] = mapped_column(Float, nullable=False)
    tenure_months: Mapped[float] = mapped_column(Float, nullable=False)
    recency_days: Mapped[float] = mapped_column(Float, nullable=False)
    frequency_per_month: Mapped[float] = mapped_column(Float, nullable=False)
    product_categories: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_clv: Mapped[float] = mapped_column(Float, nullable=False)
    xgboost_clv: Mapped[float] = mapped_column(Float, nullable=False)
    lightgbm_clv: Mapped[float] = mapped_column(Float, nullable=False)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
