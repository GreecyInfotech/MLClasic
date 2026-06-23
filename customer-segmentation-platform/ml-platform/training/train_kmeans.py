import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["recency", "frequency", "monetary", "age", "income_band"]


def generate_synthetic_dataset(size: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "recency": rng.uniform(1, 365, size=size),
            "frequency": rng.uniform(1, 50, size=size),
            "monetary": rng.uniform(50, 5000, size=size),
            "age": rng.uniform(22, 70, size=size),
            "income_band": rng.integers(1, 6, size=size),
        }
    )


def train(df: pd.DataFrame, output_path: Path, n_clusters: int) -> None:
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLUMNS])
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    score = float(silhouette_score(X, labels))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "features": FEATURE_COLUMNS}, output_path)

    metrics = {
        "n_clusters": n_clusters,
        "silhouette_score": score,
        "rows": len(df),
        "features": FEATURE_COLUMNS,
    }
    metrics_path = output_path.parent / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml-platform/artifacts/kmeans_segmentation.pkl"),
    )
    parser.add_argument("--n-clusters", type=int, default=4)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv) if args.input_csv and args.input_csv.exists() else generate_synthetic_dataset()
    train(df, args.output, args.n_clusters)


if __name__ == "__main__":
    main()
