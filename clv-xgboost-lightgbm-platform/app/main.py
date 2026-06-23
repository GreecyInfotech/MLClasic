import logging
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
from app.models import CLVPrediction
from app.schemas import CLVPredictionRequest, CLVPredictionResponse, CLVRecord, ModelChoice
from app.services.clv_service import CLVModelService

settings = get_settings()
logger = logging.getLogger(__name__)
model_service: CLVModelService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_service
    setup_logging(settings.log_level)
    Base.metadata.create_all(bind=engine)
    try:
        model_service = CLVModelService()
        logger.info("CLV models loaded")
    except FileNotFoundError:
        model_service = None
        logger.warning("CLV models not found")
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)
register_exception_handlers(app)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "environment": settings.environment}


@app.get("/ready")
def ready() -> dict:
    if model_service is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Models not loaded.")
    return {"status": "ready", "default_model": settings.default_model}


@app.post("/v1/clv/predict", response_model=CLVPredictionResponse, dependencies=[Depends(verify_api_key)])
def predict_clv(payload: CLVPredictionRequest, db: Session = Depends(get_db)):
    if model_service is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Models not loaded.")

    model = payload.model or ModelChoice(settings.default_model)
    result = model_service.predict(payload, model)

    record = CLVPrediction(
        customer_id=payload.customer_id,
        purchase_count=payload.purchase_count,
        avg_order_value=payload.avg_order_value,
        tenure_months=payload.tenure_months,
        recency_days=payload.recency_days,
        frequency_per_month=payload.frequency_per_month,
        product_categories=payload.product_categories,
        predicted_clv=result["predicted_clv"],
        xgboost_clv=result["xgboost_clv"],
        lightgbm_clv=result["lightgbm_clv"],
        model_name=result["model_name"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return CLVPredictionResponse(
        prediction_id=record.id,
        customer_id=record.customer_id,
        predicted_clv=record.predicted_clv,
        xgboost_clv=record.xgboost_clv,
        lightgbm_clv=record.lightgbm_clv,
        model_name=record.model_name,
        timestamp=datetime.utcnow(),
    )


@app.get(
    "/v1/clv/predictions/{prediction_id}",
    response_model=CLVRecord,
    dependencies=[Depends(verify_api_key)],
)
def get_prediction(prediction_id: str, db: Session = Depends(get_db)):
    record = db.query(CLVPrediction).filter(CLVPrediction.id == prediction_id).one_or_none()
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Prediction not found.")
    return record
