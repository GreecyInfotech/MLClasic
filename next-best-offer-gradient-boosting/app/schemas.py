from datetime import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class Channel(str, Enum):
    email = "email"
    sms = "sms"
    app = "app"
    web = "web"


class OfferCandidate(BaseModel):
    offer_id: str
    offer_type: str = Field(min_length=1, max_length=50)
    discount_pct: float = Field(ge=0, le=100)


class OfferRecommendationRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=64)
    customer_age: float = Field(ge=18, le=100)
    income: float = Field(gt=0)
    past_purchases: float = Field(ge=0)
    channel: Channel
    offers: List[OfferCandidate] = Field(min_length=1, max_length=20)


class RankedOffer(BaseModel):
    offer_id: str
    offer_type: str
    discount_pct: float
    acceptance_probability: float


class OfferRecommendationResponse(BaseModel):
    recommendation_id: str
    customer_id: str
    recommended_offer: str
    acceptance_probability: float
    ranked_offers: List[RankedOffer]
    timestamp: datetime


class OfferRecommendationRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    customer_id: str
    recommended_offer: str
    acceptance_probability: float
    ranked_offers: str
    created_at: datetime
