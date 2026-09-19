"""Feature engineering from raw event logs, and training/persisting/using the churn model.

Feature engineering here is a single-snapshot simplification of the sliding-window
approach explored in ``src/sandbox/Carl/final_model.ipynb``: instead of re-computing
features at many anchor dates for data augmentation, we compute one "recent vs. history"
snapshot per user relative to a chosen anchor date. This keeps training fast and
reproducible while keeping the strongest signals (engagement trend, session gaps,
frustration events, tenure).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

CHURN_EVENT = "Cancellation Confirmation"
TARGET_EVENTS = [
    "Roll Advert",
    "Thumbs Down",
    "Thumbs Up",
    "Error",
    "Add Friend",
    "Submit Downgrade",
    "Upgrade",
]
WINDOW_RECENT_DAYS = 7
WINDOW_HISTORY_DAYS = 28
TARGET_WINDOW_DAYS = 10

_EVENT_COLUMN_NAMES = {event: event.lower().replace(" ", "_") for event in TARGET_EVENTS}


def _event_counts(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Per-user page/song/listening-time/session counts, plus target-event counts."""
    stats = df.groupby("userId").agg(
        **{
            f"{prefix}_page_count": ("page", "count"),
            f"{prefix}_song_count": ("song", "count"),
            f"{prefix}_length": ("length", "sum"),
            f"{prefix}_session_count": ("sessionId", "nunique"),
        }
    )

    events = df[df["page"].isin(TARGET_EVENTS)]
    if events.empty:
        counts = pd.DataFrame(columns=TARGET_EVENTS)
    else:
        counts = events.groupby(["userId", "page"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=TARGET_EVENTS, fill_value=0)
    counts = counts.rename(columns={e: f"{prefix}_{c}" for e, c in _EVENT_COLUMN_NAMES.items()})

    return stats.join(counts, how="left").fillna(0.0)


def compute_user_features(df_logs: pd.DataFrame, anchor_date: pd.Timestamp) -> pd.DataFrame:
    """
    Compute one feature row per user active in the WINDOW_HISTORY_DAYS window ending at
    anchor_date. `df_logs` must have columns: userId, time, page, length, song, sessionId,
    level, registration.
    """
    anchor_date = pd.Timestamp(anchor_date)
    start_history = anchor_date - pd.Timedelta(days=WINDOW_HISTORY_DAYS)
    start_recent = anchor_date - pd.Timedelta(days=WINDOW_RECENT_DAYS)

    history = df_logs[(df_logs["time"] > start_history) & (df_logs["time"] <= anchor_date)].copy()
    if history.empty:
        return pd.DataFrame()

    recent = history[history["time"] > start_recent]

    sessions = history.sort_values(["userId", "time"]).drop_duplicates(
        subset=["userId", "sessionId"], keep="last"
    )
    sessions["prev_time"] = sessions.groupby("userId")["time"].shift(1)
    sessions["gap_days"] = (sessions["time"] - sessions["prev_time"]).dt.total_seconds() / 86400
    gap_stats = sessions.groupby("userId")["gap_days"].agg(
        avg_gap_days="mean", last_gap_days="last"
    )

    song_stats = (
        history[history["page"] == "NextSong"]
        .groupby("userId")["length"]
        .agg(avg_song_duration="mean", std_song_duration="std")
    )

    features = gap_stats.join(song_stats, how="outer").fillna(0.0)
    features["gap_acceleration"] = (features["last_gap_days"] + 0.1) / (
        features["avg_gap_days"] + 0.1
    )

    features = features.join(_event_counts(recent, "recent"), how="left")
    features = features.join(_event_counts(history, "history"), how="left")
    features = features.fillna(0.0)

    epsilon = 0.01
    avg_len_recent = features["recent_length"] / (features["recent_song_count"] + epsilon)
    avg_len_history = features["history_length"] / (features["history_song_count"] + epsilon)
    features["trend_song_duration"] = avg_len_recent / (avg_len_history + epsilon)

    avg_sess_recent = features["recent_length"] / (features["recent_session_count"] + epsilon)
    avg_sess_history = features["history_length"] / (features["history_session_count"] + epsilon)
    features["trend_session_intensity"] = avg_sess_recent / (avg_sess_history + epsilon)

    features["ads_per_hour"] = features["history_roll_advert"] / (
        (features["history_length"] / 3600) + epsilon
    )
    features["errors_per_session"] = features["history_error"] / (
        features["history_session_count"] + epsilon
    )
    features["recent_downgrade_attempt"] = (features["recent_submit_downgrade"] > 0).astype(int)

    last_seen = history.sort_values("time").groupby("userId").last()
    features["is_paid"] = (last_seen["level"] == "paid").astype(int)
    features["tenure_days"] = (
        (anchor_date - last_seen["registration"]).dt.total_seconds() / 86400
    ).clip(lower=0)

    last_action = history.groupby("userId")["time"].max()
    features["days_since_last_action"] = (anchor_date - last_action).dt.total_seconds() / 86400

    features.index.name = "userId"
    return features.reset_index()


def label_churn(
    df_logs: pd.DataFrame,
    users: pd.DataFrame,
    anchor_date: pd.Timestamp,
    target_window_days: int = TARGET_WINDOW_DAYS,
) -> pd.DataFrame:
    """Label each row in `users` with churn (1) if userId cancels within `target_window_days`."""
    anchor_date = pd.Timestamp(anchor_date)
    horizon = anchor_date + pd.Timedelta(days=target_window_days)
    future = df_logs[(df_logs["time"] > anchor_date) & (df_logs["time"] <= horizon)]
    churners = set(future.loc[future["page"] == CHURN_EVENT, "userId"])

    users = users.copy()
    users["target_churn"] = users["userId"].isin(churners).astype(int)
    return users


def build_training_table(
    df_logs: pd.DataFrame, anchor_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Build the per-user feature + label table used for training and for the exploration tab."""
    if anchor_date is None:
        anchor_date = df_logs["time"].max() - pd.Timedelta(days=TARGET_WINDOW_DAYS)

    features = compute_user_features(df_logs, anchor_date)
    return label_churn(df_logs, features, anchor_date)


def train(
    df: pd.DataFrame, feature_cols: list[str] | None = None
) -> tuple[XGBClassifier, list[str]]:
    """Train an XGBoost churn classifier on a feature+label table (see build_training_table)."""
    ignore = {"userId", "target_churn"}
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ignore]

    X = df[feature_cols]
    y = df["target_churn"]
    groups = df["userId"]

    pos, neg = (y == 1).sum(), (y == 0).sum()
    params = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=max(neg / pos, 1.0) if pos else 1.0,
        eval_metric="auc",
        n_jobs=-1,
        random_state=42,
    )

    n_splits = min(5, groups.nunique())
    if n_splits >= 2:
        aucs = []
        for train_idx, val_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
            fold_model = XGBClassifier(**params)
            fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
            if y.iloc[val_idx].nunique() > 1:
                preds = fold_model.predict_proba(X.iloc[val_idx])[:, 1]
                aucs.append(roc_auc_score(y.iloc[val_idx], preds))
        if aucs:
            print(f"GroupKFold mean AUC: {sum(aucs) / len(aucs):.4f} over {len(aucs)} folds")

    model = XGBClassifier(**params)
    model.fit(X, y)
    return model, feature_cols


def save_model(model: XGBClassifier, feature_cols: list[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, path)


def load_model(path: str | Path) -> tuple[XGBClassifier, list[str]]:
    bundle = joblib.load(path)
    return bundle["model"], bundle["feature_cols"]


def predict_churn(
    model: XGBClassifier, feature_cols: list[str], features: pd.DataFrame
) -> pd.Series:
    """Predict churn probability for each row of `features` (missing feature columns -> 0)."""
    X = features.reindex(columns=feature_cols, fill_value=0.0)
    probabilities = model.predict_proba(X)[:, 1]
    index = features["userId"] if "userId" in features.columns else features.index
    return pd.Series(probabilities, index=index, name="churn_probability")
