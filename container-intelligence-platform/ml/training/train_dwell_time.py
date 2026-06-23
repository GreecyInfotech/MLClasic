import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "container_type_code",
    "weight_tons",
    "customs_hold",
    "yard_congestion",
    "vessel_delay_hours",
]


def generate_synthetic_dataset(size: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    container_type_code = rng.integers(0, 4, size=size)
    weight_tons = rng.uniform(5, 35, size=size)
    customs_hold = rng.integers(0, 2, size=size)
    yard_congestion = rng.uniform(0, 1, size=size)
    vessel_delay_hours = rng.uniform(0, 72, size=size)

    dwell_hours = (
        12
        + weight_tons * 0.8
        + customs_hold * 24
        + yard_congestion * 30
        + vessel_delay_hours * 0.5
        + container_type_code * 4
        + rng.normal(0, 3, size=size)
    )
    dwell_hours = np.maximum(dwell_hours, 1)

    return pd.DataFrame(
        {
            "container_type_code": container_type_code,
            "weight_tons": weight_tons,
            "customs_hold": customs_hold,
            "yard_congestion": yard_congestion,
            "vessel_delay_hours": vessel_delay_hours,
            "dwell_hours": dwell_hours,
        }
    )


def train(df: pd.DataFrame, model_output: Path) -> None:
    X = df[FEATURE_COLUMNS]
    y = df["dwell_hours"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=250, max_depth=6, learning_rate=0.05, random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    (model_output.parent / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument(
        "--model-output", type=Path, default=Path("ml/artifacts/dwell_time_model.pkl")
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv) if args.input_csv and args.input_csv.exists() else generate_synthetic_dataset()
    train(df, args.model_output)


if __name__ == "__main__":
    main()
