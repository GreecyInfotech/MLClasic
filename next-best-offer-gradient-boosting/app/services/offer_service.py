import json
from pathlib import Path

import joblib
import numpy as np

from app.config import get_settings
from app.schemas import OfferCandidate, OfferRecommendationRequest

CHANNEL_MAP = {"email": 0, "sms": 1, "app": 2, "web": 3}
OFFER_TYPE_MAP = {
    "discount": 0,
    "bundle": 1,
    "loyalty": 2,
    "upgrade": 3,
    "cashback": 4,
}


class OfferService:
    def __init__(self) -> None:
        settings = get_settings()
        path = Path(settings.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Offer model not found: {path}")
        self.model = joblib.load(path)

    def _features(
        self, payload: OfferRecommendationRequest, offer: OfferCandidate
    ) -> np.ndarray:
        offer_type = OFFER_TYPE_MAP.get(offer.offer_type.lower(), 0)
        channel = CHANNEL_MAP[payload.channel.value]
        return np.array(
            [
                [
                    payload.customer_age,
                    payload.income,
                    payload.past_purchases,
                    channel,
                    offer.discount_pct,
                    offer_type,
                ]
            ]
        )

    def recommend(self, payload: OfferRecommendationRequest) -> dict:
        ranked = []
        for offer in payload.offers:
            prob = float(self.model.predict_proba(self._features(payload, offer))[0][1])
            ranked.append(
                {
                    "offer_id": offer.offer_id,
                    "offer_type": offer.offer_type,
                    "discount_pct": offer.discount_pct,
                    "acceptance_probability": round(prob, 4),
                }
            )
        ranked.sort(key=lambda x: x["acceptance_probability"], reverse=True)
        best = ranked[0]
        return {
            "recommended_offer": best["offer_id"],
            "acceptance_probability": best["acceptance_probability"],
            "ranked_offers": ranked,
        }
