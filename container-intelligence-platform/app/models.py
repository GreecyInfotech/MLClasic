import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Container(Base):
    __tablename__ = "containers"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    container_number: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    container_type: Mapped[str] = mapped_column(String(20), nullable=False)
    weight_tons: Mapped[float] = mapped_column(Float, nullable=False)
    origin_port: Mapped[str] = mapped_column(String(50), nullable=False)
    destination_port: Mapped[str] = mapped_column(String(50), nullable=False)
    customs_hold: Mapped[int] = mapped_column(nullable=False, default=0)
    yard_congestion: Mapped[float] = mapped_column(Float, nullable=False)
    vessel_delay_hours: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="in_transit")
    predicted_dwell_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
