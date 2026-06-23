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
from app.models import Container
from app.schemas import ContainerCreate, ContainerIntelligenceResponse, ContainerRecord, ContainerStatus
from app.services.intelligence_service import DwellTimeService, RiskService
settings = get_settings(); logger = logging.getLogger(__name__)
dwell_service: DwellTimeService | None = None; risk_service = RiskService()
@asynccontextmanager
async def lifespan(app: FastAPI):
    global dwell_service
    setup_logging(settings.log_level); Base.metadata.create_all(bind=engine)
    try: dwell_service = DwellTimeService(); logger.info("Dwell model loaded")
    except FileNotFoundError: dwell_service = None
    yield
app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RequestLoggingMiddleware); register_exception_handlers(app)
@app.get("/health")
def health(): return {"status": "ok", "environment": settings.environment}
@app.get("/ready")
def ready():
    if dwell_service is None: raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded.")
    return {"status": "ready"}
@app.post("/v1/containers/analyze", response_model=ContainerIntelligenceResponse, dependencies=[Depends(verify_api_key)])
def analyze_container(payload: ContainerCreate, db: Session = Depends(get_db)):
    if dwell_service is None: raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Dwell model not loaded.")
    dwell = dwell_service.predict_dwell(payload); risk_score, flags = risk_service.score(payload, dwell)
    container_status = risk_service.status_from_risk(risk_score, payload.customs_hold)
    existing = db.query(Container).filter(Container.container_number == payload.container_number).one_or_none()
    if existing:
        record = existing; record.container_type = payload.container_type.value; record.weight_tons = payload.weight_tons
        record.origin_port = payload.origin_port; record.destination_port = payload.destination_port
        record.customs_hold = int(payload.customs_hold); record.yard_congestion = payload.yard_congestion
        record.vessel_delay_hours = payload.vessel_delay_hours; record.predicted_dwell_hours = dwell
        record.risk_score = risk_score; record.status = container_status
    else:
        record = Container(container_number=payload.container_number, container_type=payload.container_type.value, weight_tons=payload.weight_tons, origin_port=payload.origin_port, destination_port=payload.destination_port, customs_hold=int(payload.customs_hold), yard_congestion=payload.yard_congestion, vessel_delay_hours=payload.vessel_delay_hours, predicted_dwell_hours=dwell, risk_score=risk_score, status=container_status)
        db.add(record)
    db.commit(); db.refresh(record)
    return ContainerIntelligenceResponse(container_id=record.id, container_number=record.container_number, predicted_dwell_hours=round(dwell, 2), risk_score=round(risk_score, 4), status=ContainerStatus(container_status), risk_flags=flags, timestamp=datetime.utcnow())
@app.get("/v1/containers/{container_number}", response_model=ContainerRecord, dependencies=[Depends(verify_api_key)])
def get_container(container_number: str, db: Session = Depends(get_db)):
    record = db.query(Container).filter(Container.container_number == container_number).one_or_none()
    if record is None: raise HTTPException(status.HTTP_404_NOT_FOUND, "Container not found.")
    return record
