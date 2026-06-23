import json, logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging
from app.core.middleware import RequestLoggingMiddleware
from app.core.security import verify_api_key
from app.db import Base, engine, get_db
from app.models import OfferRecommendation
from app.schemas import OfferRecommendationRecord, OfferRecommendationRequest, OfferRecommendationResponse, RankedOffer
from app.services.offer_service import OfferService
settings = get_settings(); logger = logging.getLogger(__name__)
offer_service: OfferService | None = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global offer_service
    setup_logging(settings.log_level); Base.metadata.create_all(bind=engine)
    try: offer_service = OfferService(); logger.info("Offer model loaded")
    except FileNotFoundError: offer_service = None
    yield
app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RequestLoggingMiddleware); register_exception_handlers(app)
@app.get("/health")
def health(): return {"status": "ok", "environment": settings.environment}
@app.get("/ready")
def ready():
    if offer_service is None: raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded.")
    return {"status": "ready"}
@app.post("/v1/offers/recommend", response_model=OfferRecommendationResponse, dependencies=[Depends(verify_api_key)])
def recommend_offer(payload: OfferRecommendationRequest, db: Session = Depends(get_db)):
    if offer_service is None: raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded.")
    result = offer_service.recommend(payload)
    record = OfferRecommendation(customer_id=payload.customer_id, customer_age=payload.customer_age, income=payload.income, past_purchases=payload.past_purchases, channel=payload.channel.value, recommended_offer=result["recommended_offer"], acceptance_probability=result["acceptance_probability"], ranked_offers=json.dumps(result["ranked_offers"]))
    db.add(record); db.commit(); db.refresh(record)
    return OfferRecommendationResponse(recommendation_id=record.id, customer_id=record.customer_id, recommended_offer=record.recommended_offer, acceptance_probability=record.acceptance_probability, ranked_offers=[RankedOffer(**i) for i in result["ranked_offers"]], timestamp=datetime.utcnow())
@app.get("/v1/offers/recommendations/{recommendation_id}", response_model=OfferRecommendationRecord, dependencies=[Depends(verify_api_key)])
def get_recommendation(recommendation_id: str, db: Session = Depends(get_db)):
    record = db.query(OfferRecommendation).filter(OfferRecommendation.id == recommendation_id).one_or_none()
    if record is None: raise HTTPException(status.HTTP_404_NOT_FOUND, "Recommendation not found.")
    return record
