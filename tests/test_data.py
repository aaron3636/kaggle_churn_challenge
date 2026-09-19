from pathlib import Path

import pandas as pd
import pytest

from churn.data import clean_data, filter_data, load_data

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_users.csv"


@pytest.fixture
def raw_users() -> pd.DataFrame:
    return load_data(FIXTURE_PATH)


@pytest.fixture
def users(raw_users: pd.DataFrame) -> pd.DataFrame:
    return clean_data(raw_users)


def test_load_data_reads_csv(raw_users: pd.DataFrame) -> None:
    assert len(raw_users) == 4
    assert "userId" in raw_users.columns
    assert "target_churn" in raw_users.columns


def test_load_data_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_data(FIXTURE_PATH.parent / "does_not_exist.csv")


def test_clean_data_coerces_types(users: pd.DataFrame) -> None:
    assert pd.api.types.is_string_dtype(users["userId"])
    assert users["is_paid"].dtype == bool
    assert users["target_churn"].dtype in (int, "int64")


def test_clean_data_drops_duplicate_users(users: pd.DataFrame) -> None:
    duplicated = pd.concat([users, users.iloc[[0]]], ignore_index=True)
    cleaned = clean_data(duplicated)
    assert cleaned["userId"].is_unique
    assert len(cleaned) == len(users)


def test_clean_data_fills_missing_numeric_values() -> None:
    df = pd.DataFrame({"userId": ["1", "2"], "tenure_days": [10.0, None]})
    cleaned = clean_data(df)
    assert cleaned["tenure_days"].tolist() == [10.0, 0.0]


def test_filter_data_by_plan(users: pd.DataFrame) -> None:
    paid = filter_data(users, is_paid=True)
    assert set(paid["userId"]) == {"1", "3"}

    free = filter_data(users, is_paid=False)
    assert set(free["userId"]) == {"2", "4"}


def test_filter_data_by_churn_status(users: pd.DataFrame) -> None:
    churned = filter_data(users, churned=1)
    assert set(churned["userId"]) == {"2"}


def test_filter_data_by_tenure_range(users: pd.DataFrame) -> None:
    filtered = filter_data(users, min_tenure_days=20, max_tenure_days=50)
    assert set(filtered["userId"]) == {"1", "3"}


def test_filter_data_combines_conditions(users: pd.DataFrame) -> None:
    filtered = filter_data(users, is_paid=True, churned=0, max_tenure_days=50)
    assert set(filtered["userId"]) == {"1", "3"}


def test_filter_data_no_filters_returns_all(users: pd.DataFrame) -> None:
    assert len(filter_data(users)) == len(users)
