from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


SEGMENT_LABELS = {
    0: "champions",
    1: "loyal",
    2: "at_risk",
    3: "hibernating",
}


class CustomerFeatures(BaseModel):
    customer_id: str = Field(min_length=1, max_length=64)
    recency: float = Field(ge=0)
    frequency: float = Field(ge=0)
    monetary: float = Field(ge=0)
    age: float = Field(ge=18, le=100)
    income_band: float = Field(ge=1, le=5)


class SegmentationResponse(BaseModel):
    result_id: str
    customer_id: str
    segment_id: int
    segment_label: str
    distance_to_centroid: float
    timestamp: datetime


class SegmentationRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    customer_id: str
    recency: float
    frequency: float
    monetary: float
    age: float
    income_band: float
    segment_id: int
    segment_label: str
    distance_to_centroid: float
    created_at: datetime
