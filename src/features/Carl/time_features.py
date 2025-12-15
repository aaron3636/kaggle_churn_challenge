import pandas as pd
import numpy as np


def _share(mask_counts: pd.Series, total_counts: pd.Series) -> pd.Series:
    """Safe division with small epsilon to avoid zero-divide."""
    eps = 1e-6
    return mask_counts / (total_counts + eps)


def compute_time_features(
    df: pd.DataFrame,
    anchor_date: pd.Timestamp,
    recent_days: int = 7,
    history_days: int = 28,
    ts_col: str = "ts_date",
    page_col: str = "page",
) -> pd.DataFrame:
    """
    Compute time-of-day and day-of-week engagement features per user
    for recent vs. historical windows ending at anchor_date.

    Returns a DataFrame indexed by userId with:
        - pct_peak_recent/history: share of plays in 18-22h window
        - pct_offpeak_recent/history: share of plays in 0-5h window
        - nocturnal_shift: offpeak_recent > offpeak_history * 1.2
        - weekend_share_recent/history: fraction of plays on Sat/Sun
        - weekend_shift: weekend_share_recent / weekend_share_history
        - late_night_session_ratio: sessions starting after 22h / total
        - late_night_shift: recent vs. history ratio for late-night starts
        - dow_entropy_recent/history: entropy over day-of-week
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df["hour"] = df[ts_col].dt.hour
    df["dow"] = df[ts_col].dt.dayofweek

    # Windows
    start_recent = anchor_date - pd.Timedelta(days=recent_days)
    start_history = anchor_date - pd.Timedelta(days=history_days)

    df_history = df[(df[ts_col] >= start_history) & (df[ts_col] <= anchor_date)]
    df_recent = df[(df[ts_col] >= start_recent) & (df[ts_col] <= anchor_date)]

    if df_history.empty:
        return pd.DataFrame()

    def day_entropy(g):
        p = g.value_counts(normalize=True)
        return -(p * np.log(p + 1e-9)).sum()

    def per_window_feats(df_sub: pd.DataFrame, suffix: str) -> pd.DataFrame:
        plays = df_sub[df_sub[page_col] == "NextSong"]
        total = plays.groupby("userId").size()
        peak = plays[(plays["hour"] >= 18) & (plays["hour"] <= 22)].groupby("userId").size()
        offpeak = plays[plays["hour"] <= 5].groupby("userId").size()
        weekend = plays[plays["dow"] >= 5].groupby("userId").size()

        # Session starts for late-night ratio (use first event per session)
        session_starts = (
            df_sub.sort_values([ts_col])
            .groupby(["userId", "sessionId"])
            .first()
            .reset_index()
        )
        total_sessions = session_starts.groupby("userId").size()
        late_sessions = session_starts[session_starts["hour"] >= 22].groupby("userId").size()

        entropy = plays.groupby("userId")["dow"].apply(day_entropy)

        out = pd.DataFrame(index=plays["userId"].unique())
        out[f"pct_peak_{suffix}"] = _share(peak, total)
        out[f"pct_offpeak_{suffix}"] = _share(offpeak, total)
        out[f"weekend_share_{suffix}"] = _share(weekend, total)
        out[f"late_night_session_ratio_{suffix}"] = _share(late_sessions, total_sessions)
        out[f"dow_entropy_{suffix}"] = entropy
        return out

    feats_recent = per_window_feats(df_recent, "recent")
    feats_history = per_window_feats(df_history, "history")

    feats = feats_history.join(feats_recent, how="outer").fillna(0)

    # Shift ratios
    feats["weekend_shift"] = (feats["weekend_share_recent"] + 1e-6) / (feats["weekend_share_history"] + 1e-6)
    feats["late_night_shift"] = (feats["late_night_session_ratio_recent"] + 1e-6) / (feats["late_night_session_ratio_history"] + 1e-6)
    feats["nocturnal_shift"] = (feats["pct_offpeak_recent"] > feats["pct_offpeak_history"] * 1.2).astype(int)

    return feats
