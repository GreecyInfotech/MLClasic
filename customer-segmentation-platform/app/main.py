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
from app.models import SegmentationResult
from app.schemas import CustomerFeatures, SegmentationRecord, SegmentationResponse
from app.services.segmentation_service import SegmentationService

settings = get_settings()
logger = logging.getLogger(__name__)
segmentation_service: SegmentationService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global segmentation_service
    setup_logging(settings.log_level)
    Base.metadata.create_all(bind=engine)
    try:
        segmentation_service = SegmentationService()
        logger.info("Segmentation model loaded")
    except FileNotFoundError:
        segmentation_service = None
        logger.warning("Segmentation model not found")
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RequestLoggingMiddleware)
register_exception_handlers(app)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "environment": settings.environment}


@app.get("/ready")
def ready() -> dict:
    if segmentation_service is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded.")
    return {"status": "ready"}


@app.post("/v1/segmentation/predict", response_model=SegmentationResponse, dependencies=[Depends(verify_api_key)])
def predict_segment(payload: CustomerFeatures, db: Session = Depends(get_db)):
    if segmentation_service is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded.")
    result = segmentation_service.segment(payload)
    record = SegmentationResult(
        customer_id=payload.customer_id, recency=payload.recency, frequency=payload.frequency,
        monetary=payload.monetary, age=payload.age, income_band=payload.income_band,
        segment_id=result["segment_id"], segment_label=result["segment_label"],
        distance_to_centroid=result["distance_to_centroid"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return SegmentationResponse(
        result_id=record.id, customer_id=record.customer_id, segment_id=record.segment_id,
        segment_label=record.segment_label, distance_to_centroid=record.distance_to_centroid,
        timestamp=datetime.utcnow(),
    )


@app.get("/v1/segmentation/results/{result_id}", response_model=SegmentationRecord, dependencies=[Depends(verify_api_key)])
def get_result(result_id: str, db: Session = Depends(get_db)):
    record = db.query(SegmentationResult).filter(SegmentationResult.id == result_id).one_or_none()
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Result not found.")
    return record
