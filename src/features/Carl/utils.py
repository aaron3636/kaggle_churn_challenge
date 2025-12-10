import pandas as pd
import numpy as np


def weekly_features(df, feature, rolling_time_window=1):
    df_X = rolling_mean_per_userId(df=df,
                                   feature=feature,
                                   rolling_mean_window=rolling_time_window).copy()

    iso = pd.to_datetime(df_X["date"]).dt.isocalendar()
    df_X["iso_year"] = iso.year
    df_X["iso_week"] = iso.week

    df_a = (df_X.groupby(["userId", "iso_year", "iso_week"], as_index=False)["rolling_avg"]
            .sum())

    # Stable week key that won’t collide across years
    df_a["year_week"] = (
        df_a["iso_year"].astype(str) + "-W" + df_a["iso_week"].astype(str).str.zfill(2)
    )

    week_col = sorted(df_a["year_week"].unique())

    df_b = (df_a.pivot(index="userId", columns="year_week", values="rolling_avg")
            .reindex(columns=week_col)
            .fillna(0.0))

    # If there aren't at least 2 weeks, deltas don't exist
    if len(week_col) < 2:
        df_b[f"delta_variance_{feature}"] = 0.0
        df_b[f"delta_mean_{feature}"] = 0.0
        df_b[f"delta_variance_%_{feature}"] = 0.0
        df_b[f"delta_mean_%_{feature}"] = 0.0
        return df_b.reset_index()

    delta_cols, delta_cols_per = [], []

    for i in range(len(week_col) - 1):
        w0, w1 = week_col[i], week_col[i + 1]

        d = df_b[w1] - df_b[w0]
        df_b[f"delta_week_{i}_{i+1}_{feature}"] = d
        delta_cols.append(f"delta_week_{i}_{i+1}_{feature}")

        base = df_b[w0]
        # define % change as 0 when base==0 (avoids ±inf)

        # if base == 0 and d == 0 -> 0
        # if base == 0 and d != 0 -> 100

        # pct = np.where(base == 0, 100.0, (d / base) * 100.0)

        pct = np.where(base == 0,
                       np.where(d == 0, 0.0, 100.0),   # base==0: d==0 -> 0, else -> 100
                       (d / base) * 100.0)             # base!=0: normal % change
        df_b[f"delta_week_{i}_{i+1}_%_{feature}"] = pct
        delta_cols_per.append(f"delta_week_{i}_{i+1}_%_{feature}")

    # ddof=0 avoids NaN variance when there's only 1 delta column
    df_b[f"delta_variance_{feature}"] = df_b[delta_cols].var(axis=1, ddof=0)
    df_b[f"delta_mean_{feature}"] = df_b[delta_cols].mean(axis=1)

    df_b[f"delta_variance_%_{feature}"] = df_b[delta_cols_per].var(axis=1, ddof=0)
    df_b[f"delta_mean_%_{feature}"] = df_b[delta_cols_per].mean(axis=1)

    # Optional: rename weekly columns to 0..n-1 like you did
    rename_map = {week: f"{feature}_week_{i}" for i, week in enumerate(week_col)}
    df_b = df_b.rename(columns=rename_map)
    df_b.columns = df_b.columns.astype(str)

    return df_b.reset_index()


def rolling_mean_per_userId(df: pd.DataFrame,
                            feature: str,
                            rolling_mean_window: int = 1):
    """
    Docstring for rolling_mean_length_per_userId

    Input a dataframe for which one feature has been aggregated on a daily
    basis. In the format

    userId | date | aggregated dynmaic feature

    For this aggreagted dynamic feature this function computes the

    - weekly sum

    # Deltas
    - delta between each week
    - mean of all dates -> gradient
    - delta variance -> change in usage
    """

    # Example input
    # daily length
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["time"]).dt.floor("D")

    daily = df.groupby(["userId", "date"], as_index=False)["length"].sum()
    """

    all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    idx = pd.MultiIndex.from_product(
        [df["userId"].unique(), all_dates],
        names=["userId", "date"]
    )

    df_res = (df.set_index(["userId", "date"])
              .reindex(idx, fill_value=0)
              .reset_index())

    df_res["rolling_avg"] = df.groupby("userId")[feature].transform(
        lambda s: s.rolling(window=rolling_mean_window,
                            min_periods=rolling_mean_window).mean().bfill()
    )
    return df_res
