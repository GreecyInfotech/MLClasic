from pathlib import Path

import joblib
import numpy as np

from app.config import get_settings
from app.features import FeatureVector


class ModelService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.model = self._load_model(self.settings.model_path)

    @staticmethod
    def _load_model(model_path: str):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. Train the model first."
            )
        return joblib.load(path)

    def predict_probability(self, features: FeatureVector) -> float:
        prediction = self.model.predict_proba(features.as_model_input())[0][1]
        return float(np.clip(prediction, 0.0, 1.0))
