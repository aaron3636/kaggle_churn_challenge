"""Streamlit app: explore the churn feature table and predict churn for a user."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from churn.data import clean_data, filter_data, load_data
from churn.model import load_model, predict_churn

DATA_PATH = Path("data/processed/users.parquet")
MODEL_PATH = Path("models/churn_xgb.joblib")


@st.cache_data
def get_data() -> pd.DataFrame:
    return clean_data(load_data(DATA_PATH))


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def render_explore_tab(df: pd.DataFrame) -> None:
    st.subheader("Filter users")
    col1, col2, col3 = st.columns(3)
    with col1:
        plan = st.selectbox("Plan", ["Any", "Paid", "Free"])
    with col2:
        churn_status = st.selectbox("Churn status", ["Any", "Churned", "Retained"])
    with col3:
        max_tenure = max(int(df["tenure_days"].max()) + 1, 1)
        tenure_range = st.slider(
            "Tenure (days)", min_value=0, max_value=max_tenure, value=(0, max_tenure)
        )

    filtered = filter_data(
        df,
        is_paid={"Paid": True, "Free": False}.get(plan),
        churned={"Churned": 1, "Retained": 0}.get(churn_status),
        min_tenure_days=tenure_range[0],
        max_tenure_days=tenure_range[1],
    )

    st.caption(f"{len(filtered)} of {len(df)} users match the current filters.")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Users", len(filtered))
    metric_col2.metric(
        "Churn rate", f"{filtered['target_churn'].mean():.1%}" if len(filtered) else "—"
    )
    metric_col3.metric("Paid share", f"{filtered['is_paid'].mean():.1%}" if len(filtered) else "—")

    if len(filtered):
        left, right = st.columns(2)
        with left:
            fig = px.histogram(
                filtered,
                x="tenure_days",
                color="target_churn",
                barmode="overlay",
                nbins=30,
                labels={"target_churn": "Churned"},
                title="Tenure distribution",
            )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.histogram(
                filtered,
                x="days_since_last_action",
                color="target_churn",
                barmode="overlay",
                nbins=30,
                labels={"target_churn": "Churned"},
                title="Days since last action",
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show filtered data"):
        st.dataframe(filtered)


def render_predict_tab(df: pd.DataFrame) -> None:
    model, feature_cols = get_model()

    st.subheader("Predict churn for an existing user")
    user_id = st.selectbox("User ID", df["userId"].tolist())
    user_row = df[df["userId"] == user_id]

    if st.button("Predict", type="primary"):
        prediction = predict_churn(model, feature_cols, user_row)
        probability = float(prediction.iloc[0])
        st.metric("Churn probability", f"{probability:.1%}")
        if probability >= 0.5:
            st.warning("This user is likely to churn.")
        else:
            st.success("This user is likely to stay.")

    with st.expander("Feature values used for this prediction"):
        st.dataframe(user_row[["userId", *feature_cols]])


def main() -> None:
    st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")
    st.title("📉 Subscriber Churn Predictor")
    st.caption(
        "Music-streaming event logs, aggregated into per-user features. "
        "See the README for how the data was processed."
    )

    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        st.error(
            f"Missing `{DATA_PATH}` or `{MODEL_PATH}`. Run `scripts/build_features.py` "
            "and `scripts/train.py` first (see README)."
        )
        return

    df = get_data()
    explore_tab, predict_tab = st.tabs(["🔍 Explore", "🔮 Predict"])
    with explore_tab:
        render_explore_tab(df)
    with predict_tab:
        render_predict_tab(df)


if __name__ == "__main__":
    main()
