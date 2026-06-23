from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

from app.config import get_settings
from app.schemas import SEGMENT_LABELS, CustomerFeatures

FEATURE_COLUMNS = ["recency", "frequency", "monetary", "age", "income_band"]


class SegmentationService:
    def __init__(self) -> None:
        settings = get_settings()
        path = Path(settings.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Segmentation model not found: {path}")
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]

    def segment(self, payload: CustomerFeatures) -> dict:
        features = np.array([[getattr(payload, col) for col in FEATURE_COLUMNS]])
        scaled = self.scaler.transform(features)
        segment_id = int(self.model.predict(scaled)[0])
        _, distances = pairwise_distances_argmin_min(scaled, self.model.cluster_centers_)
        label = SEGMENT_LABELS.get(segment_id, f"segment_{segment_id}")
        return {
            "segment_id": segment_id,
            "segment_label": label,
            "distance_to_centroid": float(distances[0]),
        }
