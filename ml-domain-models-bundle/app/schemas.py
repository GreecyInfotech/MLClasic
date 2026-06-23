from typing import Any

from pydantic import BaseModel, Field


class BerthOccupancyRequest(BaseModel):
    vessel_count: float = Field(ge=0)
    berth_capacity: float = Field(gt=0)
    hour: float = Field(ge=0, le=23)
    day_of_week: float = Field(ge=0, le=6)


class ClaimSettlementRequest(BaseModel):
    claim_amount: float = Field(gt=0)
    policy_age_years: float = Field(ge=0)
    prior_claims: float = Field(ge=0)
    incident_severity: float = Field(ge=1, le=5)


class CustomerRiskRequest(BaseModel):
    credit_score: float = Field(ge=300, le=850)
    debt_ratio: float = Field(ge=0, le=2)
    delinquencies: float = Field(ge=0)
    annual_income: float = Field(gt=0)


class FraudRiskRequest(BaseModel):
    transaction_amount: float = Field(gt=0)
    velocity_24h: float = Field(ge=0)
    geo_mismatch: float = Field(ge=0, le=1)
    merchant_risk: float = Field(ge=0, le=1)


class SuspiciousTransactionRequest(BaseModel):
    amount: float = Field(gt=0)
    hour: float = Field(ge=0, le=23)
    merchant_category: float = Field(ge=0, le=20)
    distance_from_home_km: float = Field(ge=0)


class VesselDelayRequest(BaseModel):
    weather_score: float = Field(ge=0, le=1)
    port_congestion: float = Field(ge=0, le=1)
    cargo_volume_teu: float = Field(gt=0)


class PredictionResponse(BaseModel):
    prediction_id: str
    model_name: str
    result: dict[str, Any]
