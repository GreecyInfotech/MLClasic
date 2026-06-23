import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class LoanApplication(Base):
    __tablename__ = "loan_applications"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    annual_income: Mapped[float] = mapped_column(Float, nullable=False)
    requested_amount: Mapped[float] = mapped_column(Float, nullable=False)
    employment_years: Mapped[float] = mapped_column(Float, nullable=False)
    credit_score: Mapped[int] = mapped_column(nullable=False)
    existing_debt: Mapped[float] = mapped_column(Float, nullable=False)
    delinquencies: Mapped[int] = mapped_column(nullable=False)
    purpose: Mapped[str] = mapped_column(String(100), nullable=False)

    model_probability: Mapped[float] = mapped_column(Float, nullable=False)
    fraud_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    reasons: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
