import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "purchase_count",
    "avg_order_value",
    "tenure_months",
    "recency_days",
    "frequency_per_month",
    "product_categories",
]


def generate_synthetic_dataset(size: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    purchase_count = rng.integers(1, 120, size=size)
    avg_order_value = rng.uniform(20, 500, size=size)
    tenure_months = rng.uniform(1, 120, size=size)
    recency_days = rng.uniform(1, 365, size=size)
    frequency_per_month = purchase_count / np.maximum(tenure_months, 1)
    product_categories = rng.integers(1, 20, size=size)

    clv = (
        purchase_count * avg_order_value * 0.6
        + tenure_months * 12
        + frequency_per_month * 80
        - recency_days * 0.5
        + product_categories * 25
        + rng.normal(0, 200, size=size)
    )
    clv = np.maximum(clv, 0)

    return pd.DataFrame(
        {
            "purchase_count": purchase_count,
            "avg_order_value": avg_order_value,
            "tenure_months": tenure_months,
            "recency_days": recency_days,
            "frequency_per_month": frequency_per_month,
            "product_categories": product_categories,
            "clv": clv,
        }
    )


def train_models(df: pd.DataFrame, artifacts_dir: Path) -> None:
    X = df[FEATURE_COLUMNS]
    y = df["clv"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    xgb_model = xgb.XGBRegressor(
        n_estimators=250, max_depth=6, learning_rate=0.05, random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    lgb_model = lgb.LGBMRegressor(
        n_estimators=250, learning_rate=0.05, random_state=42
    )
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_test)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    xgb_path = artifacts_dir / "xgboost_clv.json"
    lgb_path = artifacts_dir / "lightgbm_clv.txt"
    metrics_path = artifacts_dir / "training_metrics.json"

    xgb_model.save_model(xgb_path)
    lgb_model.booster_.save_model(str(lgb_path))

    metrics = {
        "xgboost": {
            "mae": float(mean_absolute_error(y_test, xgb_preds)),
            "r2": float(r2_score(y_test, xgb_preds)),
        },
        "lightgbm": {
            "mae": float(mean_absolute_error(y_test, lgb_preds)),
            "r2": float(r2_score(y_test, lgb_preds)),
        },
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("ml/artifacts"))
    args = parser.parse_args()

    if args.input_csv and args.input_csv.exists():
        df = pd.read_csv(args.input_csv)
    else:
        df = generate_synthetic_dataset()
    train_models(df, args.artifacts_dir)


if __name__ == "__main__":
    main()
