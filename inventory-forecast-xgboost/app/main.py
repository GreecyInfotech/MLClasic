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
from app.models import InventoryForecast
from app.schemas import ForecastRecord, ForecastRequest, ForecastResponse
from app.services.forecast_service import ForecastService
settings = get_settings(); logger = logging.getLogger(__name__)
forecast_service: ForecastService | None = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global forecast_service
    setup_logging(settings.log_level); Base.metadata.create_all(bind=engine)
    try: forecast_service = ForecastService(); logger.info("Forecast model loaded")
    except FileNotFoundError: forecast_service = None; logger.warning("Forecast model not found")
    yield
app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RequestLoggingMiddleware); register_exception_handlers(app)
@app.get("/health")
def health(): return {"status": "ok", "environment": settings.environment}
@app.get("/ready")
def ready():
    if forecast_service is None: raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded.")
    return {"status": "ready"}
@app.post("/v1/forecast", response_model=ForecastResponse, dependencies=[Depends(verify_api_key)])
def create_forecast(payload: ForecastRequest, db: Session = Depends(get_db)):
    if forecast_service is None: raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded.")
    result = forecast_service.predict(payload)
    record = InventoryForecast(product_id=payload.product_id, forecast_date=payload.forecast_date, lag_1=payload.lag_1, lag_7=payload.lag_7, lag_30=payload.lag_30, day_of_week=payload.day_of_week, month=payload.month, promo_flag=payload.promo_flag, stock_level=payload.stock_level, predicted_demand=result["predicted_demand"], safety_stock=result["safety_stock"], reorder_point=result["reorder_point"])
    db.add(record); db.commit(); db.refresh(record)
    return ForecastResponse(forecast_id=record.id, product_id=record.product_id, forecast_date=record.forecast_date, predicted_demand=record.predicted_demand, safety_stock=record.safety_stock, reorder_point=record.reorder_point, timestamp=datetime.utcnow())
@app.get("/v1/forecast/{forecast_id}", response_model=ForecastRecord, dependencies=[Depends(verify_api_key)])
def get_forecast(forecast_id: str, db: Session = Depends(get_db)):
    record = db.query(InventoryForecast).filter(InventoryForecast.id == forecast_id).one_or_none()
    if record is None: raise HTTPException(status.HTTP_404_NOT_FOUND, "Forecast not found.")
    return record
