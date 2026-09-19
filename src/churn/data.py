"""Pure load/clean/filter functions for the processed per-user churn feature table.

No Streamlit imports here — these functions operate on plain DataFrames so they can be
unit tested in isolation and reused outside the app (e.g. from scripts/train.py).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ID_COLUMN = "userId"
LABEL_COLUMN = "target_churn"


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the processed per-user feature table produced by scripts/build_features.py."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes, fill missing feature values, and drop duplicate/invalid users."""
    df = df.copy()
    df = df.dropna(subset=[ID_COLUMN])
    df[ID_COLUMN] = df[ID_COLUMN].astype(str)

    numeric_cols = df.columns.difference([ID_COLUMN])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    if "is_paid" in df.columns:
        df["is_paid"] = df["is_paid"].astype(bool)
    if LABEL_COLUMN in df.columns:
        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    return df.drop_duplicates(subset=[ID_COLUMN]).reset_index(drop=True)


def filter_data(
    df: pd.DataFrame,
    is_paid: bool | None = None,
    churned: int | None = None,
    min_tenure_days: float | None = None,
    max_tenure_days: float | None = None,
    min_days_since_last_action: float | None = None,
    max_days_since_last_action: float | None = None,
) -> pd.DataFrame:
    """Apply exploration-tab filters. Any argument left as None is not applied."""
    mask = pd.Series(True, index=df.index)

    if is_paid is not None and "is_paid" in df.columns:
        mask &= df["is_paid"] == is_paid
    if churned is not None and LABEL_COLUMN in df.columns:
        mask &= df[LABEL_COLUMN] == churned
    if min_tenure_days is not None and "tenure_days" in df.columns:
        mask &= df["tenure_days"] >= min_tenure_days
    if max_tenure_days is not None and "tenure_days" in df.columns:
        mask &= df["tenure_days"] <= max_tenure_days
    if min_days_since_last_action is not None and "days_since_last_action" in df.columns:
        mask &= df["days_since_last_action"] >= min_days_since_last_action
    if max_days_since_last_action is not None and "days_since_last_action" in df.columns:
        mask &= df["days_since_last_action"] <= max_days_since_last_action

    return df[mask].reset_index(drop=True)
