from pathlib import Path

import joblib
import numpy as np

from app.config import get_settings
from app.schemas import ContainerCreate, ContainerType

TYPE_MAP = {
    ContainerType.dry: 0,
    ContainerType.reefer: 1,
    ContainerType.tank: 2,
    ContainerType.flat_rack: 3,
}

FEATURE_COLUMNS = [
    "container_type_code",
    "weight_tons",
    "customs_hold",
    "yard_congestion",
    "vessel_delay_hours",
]


class DwellTimeService:
    def __init__(self) -> None:
        settings = get_settings()
        path = Path(settings.dwell_model_path)
        if not path.exists():
            raise FileNotFoundError(f"Dwell model not found: {path}")
        self.model = joblib.load(path)

    def predict_dwell(self, payload: ContainerCreate) -> float:
        features = np.array(
            [
                [
                    TYPE_MAP[payload.container_type],
                    payload.weight_tons,
                    int(payload.customs_hold),
                    payload.yard_congestion,
                    payload.vessel_delay_hours,
                ]
            ]
        )
        return float(max(self.model.predict(features)[0], 0))


class RiskService:
    def __init__(self) -> None:
        self.threshold = get_settings().risk_threshold

    def score(self, payload: ContainerCreate, dwell_hours: float) -> tuple[float, list[str]]:
        flags: list[str] = []
        score = 0.05

        if payload.customs_hold:
            score += 0.35
            flags.append("customs_hold_active")
        if payload.yard_congestion > 0.75:
            score += 0.2
            flags.append("high_yard_congestion")
        if payload.vessel_delay_hours > 24:
            score += 0.15
            flags.append("vessel_delay_exceeded")
        if dwell_hours > 72:
            score += 0.2
            flags.append("extended_dwell_predicted")
        if payload.weight_tons > 30:
            score += 0.1
            flags.append("heavy_container")

        return min(score, 1.0), flags

    def status_from_risk(self, risk_score: float, customs_hold: bool) -> str:
        if risk_score >= self.threshold:
            return "high_risk_hold"
        if customs_hold:
            return "customs_review"
        return "at_yard"
