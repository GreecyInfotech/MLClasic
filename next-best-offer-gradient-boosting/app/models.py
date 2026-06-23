import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class OfferRecommendation(Base):
    __tablename__ = "offer_recommendations"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    customer_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    customer_age: Mapped[float] = mapped_column(Float, nullable=False)
    income: Mapped[float] = mapped_column(Float, nullable=False)
    past_purchases: Mapped[float] = mapped_column(Float, nullable=False)
    channel: Mapped[str] = mapped_column(String(30), nullable=False)
    recommended_offer: Mapped[str] = mapped_column(String(100), nullable=False)
    acceptance_probability: Mapped[float] = mapped_column(Float, nullable=False)
    ranked_offers: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
