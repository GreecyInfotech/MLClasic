from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ContainerType(str, Enum):
    dry = "dry"
    reefer = "reefer"
    tank = "tank"
    flat_rack = "flat_rack"


class ContainerStatus(str, Enum):
    in_transit = "in_transit"
    at_yard = "at_yard"
    customs_review = "customs_review"
    released = "released"
    high_risk_hold = "high_risk_hold"


class ContainerCreate(BaseModel):
    container_number: str = Field(min_length=5, max_length=20)
    container_type: ContainerType
    weight_tons: float = Field(gt=0, le=40)
    origin_port: str
    destination_port: str
    customs_hold: bool = False
    yard_congestion: float = Field(ge=0, le=1)
    vessel_delay_hours: float = Field(ge=0)


class ContainerIntelligenceResponse(BaseModel):
    container_id: str
    container_number: str
    predicted_dwell_hours: float
    risk_score: float
    status: ContainerStatus
    risk_flags: list[str]
    timestamp: datetime


class ContainerRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    container_number: str
    container_type: str
    weight_tons: float
    origin_port: str
    destination_port: str
    customs_hold: int
    yard_congestion: float
    vessel_delay_hours: float
    status: str
    predicted_dwell_hours: Optional[float]
    risk_score: Optional[float]
    created_at: datetime
    updated_at: datetime
