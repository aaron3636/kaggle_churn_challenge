"""Train the churn model on the processed feature table and save it for the app.

Run locally after scripts/build_features.py (see README):

    uv run python scripts/train.py

Not run in CI — see scripts/build_features.py for why. The resulting model artifact
(models/churn_xgb.joblib) is committed so the app and Docker image don't need the raw data.
"""

from pathlib import Path

from churn.data import clean_data, load_data
from churn.model import save_model, train

DATA_PATH = Path("data/processed/users.parquet")
MODEL_PATH = Path("models/churn_xgb.joblib")


def main() -> None:
    df = clean_data(load_data(DATA_PATH))
    model, feature_cols = train(df)
    save_model(model, feature_cols, MODEL_PATH)
    print(f"Saved model with {len(feature_cols)} features to {MODEL_PATH}")


if __name__ == "__main__":
    main()
