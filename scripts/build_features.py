"""Build the processed per-user feature table from the raw Kaggle event log.

Run locally after downloading the real data (see README):

    uv run python scripts/build_features.py

Not run in CI — the raw data can't be redistributed, so this stays a local step whose
output (data/processed/users.parquet) is gitignored.
"""

from pathlib import Path

import pandas as pd

from churn.model import build_training_table

RAW_PATH = Path("data/churn-prediction-25-26/train.parquet")
OUTPUT_PATH = Path("data/processed/users.parquet")
COLUMNS = ["userId", "time", "page", "length", "song", "sessionId", "level", "registration"]


def main() -> None:
    print(f"Loading {RAW_PATH} ...")
    df_logs = pd.read_parquet(RAW_PATH, columns=COLUMNS)
    df_logs = df_logs.dropna(subset=["userId"])

    print("Computing per-user features and labels ...")
    table = build_training_table(df_logs)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(table)} users to {OUTPUT_PATH}")
    print(table["target_churn"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
