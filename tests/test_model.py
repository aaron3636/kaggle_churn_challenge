from pathlib import Path

import pandas as pd
import pytest

from churn.data import clean_data, load_data
from churn.model import (
    build_training_table,
    compute_user_features,
    label_churn,
    load_model,
    predict_churn,
    save_model,
    train,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
ANCHOR_DATE = pd.Timestamp("2024-01-10 12:00:00")


@pytest.fixture
def events() -> pd.DataFrame:
    return pd.read_csv(
        FIXTURES_DIR / "sample_events.csv",
        dtype={"userId": str},
        parse_dates=["time", "registration"],
    )


@pytest.fixture
def users() -> pd.DataFrame:
    return clean_data(load_data(FIXTURES_DIR / "sample_users.csv"))


def test_compute_user_features_one_row_per_active_user(events: pd.DataFrame) -> None:
    features = compute_user_features(events, ANCHOR_DATE)
    assert set(features["userId"]) == {"1", "2", "3", "4"}


def test_compute_user_features_captures_paid_flag(events: pd.DataFrame) -> None:
    features = compute_user_features(events, ANCHOR_DATE).set_index("userId")
    assert features.loc["1", "is_paid"] == 1
    assert features.loc["2", "is_paid"] == 0


def test_compute_user_features_no_activity_returns_empty(events: pd.DataFrame) -> None:
    far_future = pd.Timestamp("2030-01-01")
    features = compute_user_features(events, far_future)
    assert features.empty


def test_label_churn_flags_cancelling_user(events: pd.DataFrame) -> None:
    features = compute_user_features(events, ANCHOR_DATE)
    labelled = label_churn(events, features, ANCHOR_DATE, target_window_days=10)
    labelled = labelled.set_index("userId")
    assert labelled.loc["2", "target_churn"] == 1
    assert labelled.loc["1", "target_churn"] == 0


def test_build_training_table_matches_fixture(events: pd.DataFrame, users: pd.DataFrame) -> None:
    table = clean_data(build_training_table(events, anchor_date=ANCHOR_DATE))
    assert set(table["userId"]) == set(users["userId"])
    merged = table.set_index("userId")["target_churn"].sort_index()
    expected = users.set_index("userId")["target_churn"].sort_index()
    assert merged.equals(expected)


def test_train_and_predict_roundtrip(users: pd.DataFrame) -> None:
    model, feature_cols = train(users)
    predictions = predict_churn(model, feature_cols, users)

    assert list(predictions.index) == list(users["userId"])
    assert predictions.between(0, 1).all()


def test_save_and_load_model_preserves_predictions(tmp_path, users: pd.DataFrame) -> None:
    model, feature_cols = train(users)
    before = predict_churn(model, feature_cols, users)

    model_path = tmp_path / "model.joblib"
    save_model(model, feature_cols, model_path)
    loaded_model, loaded_feature_cols = load_model(model_path)
    after = predict_churn(loaded_model, loaded_feature_cols, users)

    assert loaded_feature_cols == feature_cols
    assert before.equals(after)


def test_predict_churn_fills_missing_feature_columns(users: pd.DataFrame) -> None:
    model, feature_cols = train(users)
    partial = users.drop(columns=[feature_cols[0]])
    predictions = predict_churn(model, feature_cols, partial)
    assert predictions.between(0, 1).all()
