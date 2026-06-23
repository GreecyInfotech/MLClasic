import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_30",
    "day_of_week",
    "month",
    "promo_flag",
    "stock_level",
]


def generate_synthetic_dataset(size: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    lag_1 = rng.uniform(5, 200, size=size)
    lag_7 = lag_1 * rng.uniform(0.8, 1.2, size=size)
    lag_30 = lag_1 * rng.uniform(0.7, 1.3, size=size)
    day_of_week = rng.integers(0, 7, size=size)
    month = rng.integers(1, 13, size=size)
    promo_flag = rng.integers(0, 2, size=size)
    stock_level = rng.uniform(20, 1000, size=size)

    demand = (
        0.5 * lag_1
        + 0.25 * lag_7
        + 0.15 * lag_30
        + promo_flag * 20
        - 0.01 * stock_level
        + rng.normal(0, 5, size=size)
    )
    demand = np.maximum(demand, 0)

    return pd.DataFrame(
        {
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_30": lag_30,
            "day_of_week": day_of_week,
            "month": month,
            "promo_flag": promo_flag,
            "stock_level": stock_level,
            "demand": demand,
        }
    )


def train(df: pd.DataFrame, model_output: Path) -> None:
    X = df[FEATURE_COLUMNS]
    y = df["demand"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
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
        "--model-output", type=Path, default=Path("ml/artifacts/inventory_forecast.pkl")
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv) if args.input_csv and args.input_csv.exists() else generate_synthetic_dataset()
    train(df, args.model_output)


if __name__ == "__main__":
    main()
