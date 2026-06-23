import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class SegmentationResult(Base):
    __tablename__ = "segmentation_results"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    customer_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    recency: Mapped[float] = mapped_column(Float, nullable=False)
    frequency: Mapped[float] = mapped_column(Float, nullable=False)
    monetary: Mapped[float] = mapped_column(Float, nullable=False)
    age: Mapped[float] = mapped_column(Float, nullable=False)
    income_band: Mapped[float] = mapped_column(Float, nullable=False)
    segment_id: Mapped[int] = mapped_column(Integer, nullable=False)
    segment_label: Mapped[str] = mapped_column(String(50), nullable=False)
    distance_to_centroid: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
