from datetime import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class LoanPurpose(str, Enum):
    personal = "personal"
    education = "education"
    business = "business"
    home_improvement = "home_improvement"
    medical = "medical"


class LoanApplicationRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    annual_income: float = Field(gt=0)
    requested_amount: float = Field(gt=0)
    employment_years: float = Field(ge=0, le=60)
    credit_score: int = Field(ge=300, le=850)
    existing_debt: float = Field(ge=0)
    delinquencies: int = Field(ge=0, le=25)
    purpose: LoanPurpose


class DecisionType(str, Enum):
    approved = "approved"
    manual_review = "manual_review"
    rejected = "rejected"


class LoanDecisionResponse(BaseModel):
    application_id: str
    decision: DecisionType
    model_probability: float
    fraud_score: float
    risk_score: float
    reasons: List[str]
    timestamp: datetime


class LoanApplicationRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    full_name: str
    email: EmailStr
    annual_income: float
    requested_amount: float
    employment_years: float
    credit_score: int
    existing_debt: float
    delinquencies: int
    purpose: str
    decision: DecisionType
    model_probability: float
    fraud_score: float
    risk_score: float
    reasons: str
    created_at: datetime
