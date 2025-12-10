import pandas as pd
import datetime
import numpy as np
from datetime import timedelta


def total_number_of_sessions(df: pd.DataFrame):
    """
    Docstring for total_number_of_sessions

    :param df: event log data frame

    :type df: pd.DataFrame

    :output df:
    One column: userId
    Second column: total_number_of_sessions
    """

    return df.groupby("userId")["sessionId"].nunique()\
        .reset_index(name="total_number_of_sessions")


def total_number_of_sessions_1(df: pd.DataFrame):
    """
    Docstring for total_number_of_sessions

    :param df: event log data frame

    :type df: pd.DataFrame

    :output df:
    One column: userId
    Second column: total_number_of_sessions
    """

    return df.groupby(["userId", "week_slice"])["sessionId"].nunique()\
        .reset_index(name="total_number_of_sessions")


def time_last_used(df: pd.DataFrame):

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    today = df["time"].max().normalize()  # midnight, stays datetime64
    df["time_last_used"] = (
        df.groupby(["userId"])["time"]
          .transform("max")
          .dt.normalize()
    )

    df["time_not_used"] = (today - df["time_last_used"]).dt.total_seconds()

    return df[["userId", "time_not_used"]].drop_duplicates()


def time_last_used_1(df: pd.DataFrame):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    today = df["time"].max().normalize()  # midnight, stays datetime64
    df["time_last_used"] = (
        df.groupby(["userId", "week_slice"])["time"]
          .transform("max")
          .dt.normalize()
    )

    df["time_not_used"] = (today - df["time_last_used"]).dt.total_seconds()

    return df[["userId", "time_not_used", "week_slice"]].drop_duplicates()


def rolling_mean_length_per_userId(df: pd.DataFrame, rolling_mean_window: int = 1):
    df = df.copy()
    df["date"] = pd.to_datetime(df["time"]).dt.floor("D")

    daily = df.groupby(["userId", "date"], as_index=False)["length"].sum()

    all_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    idx = pd.MultiIndex.from_product(
        [daily["userId"].unique(), all_dates],
        names=["userId", "date"]
    )

    daily = (daily.set_index(["userId", "date"])
                  .reindex(idx, fill_value=0)
                  .reset_index())

    daily["rolling_avg"] = daily.groupby("userId")["length"].transform(
        lambda s: s.rolling(window=rolling_mean_window,
                            min_periods=rolling_mean_window).mean().bfill()
    )
    return daily


def weekly_length_features(df, rolling_time_window=1):
    df_X = rolling_mean_length_per_userId(
        df=df, rolling_mean_window=rolling_time_window).copy()

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
        df_b["delta_variance"] = 0.0
        df_b["delta_mean"] = 0.0
        df_b["delta_variance_%"] = 0.0
        df_b["delta_mean_%"] = 0.0
        return df_b.reset_index()

    delta_cols, delta_cols_per = [], []

    for i in range(len(week_col) - 1):
        w0, w1 = week_col[i], week_col[i + 1]

        d = df_b[w1] - df_b[w0]
        df_b[f"delta_week_{i}_{i+1}"] = d
        delta_cols.append(f"delta_week_{i}_{i+1}")

        base = df_b[w0]
        # define % change as 0 when base==0 (avoids ±inf)
        pct = np.where(base == 0, 100.0, (d / base) * 100.0)
        df_b[f"delta_week_{i}_{i+1}_%"] = pct
        delta_cols_per.append(f"delta_week_{i}_{i+1}_%")

    # ddof=0 avoids NaN variance when there's only 1 delta column
    df_b["delta_variance"] = df_b[delta_cols].var(axis=1, ddof=0)
    df_b["delta_mean"] = df_b[delta_cols].mean(axis=1)

    df_b["delta_variance_%"] = df_b[delta_cols_per].var(axis=1, ddof=0)
    df_b["delta_mean_%"] = df_b[delta_cols_per].mean(axis=1)

    # Optional: rename weekly columns to 0..n-1 like you did
    rename_map = {week: str(i) for i, week in enumerate(week_col)}
    df_b = df_b.rename(columns=rename_map)
    df_b.columns = df_b.columns.astype(str)

    return df_b.reset_index()


#############################
# ARCHIVE
#############################


def weekly_length_features_(df, rolling_time_window=1):

    # compute the rolling average for the length
    df_X_length_rolling_mean = rolling_mean_length_per_userId(
        df=df, rolling_mean_window=rolling_time_window)

    # compute the week
    df_X_length_rolling_mean["week"] = pd.to_datetime(
        df_X_length_rolling_mean["date"]).dt.isocalendar().week

    # sum the rolling_avg over the week -> compute the overall usage per week
    df_a = df_X_length_rolling_mean.groupby(["userId", "week"])[
        "rolling_avg"].sum().reset_index()

    # get the week columns and sort them ascending
    week_col = sorted(df_a["week"].unique())

    # pivot the df to have all weeks as a column
    df_b = df_a.pivot(index="userId", columns="week", values="rolling_avg")

    delta_cols = []
    delta_cols_per = []
    # compute deltas
    for index in range(len(week_col) - 1):
        df_b[f"delta_week_{index}_{index+1}"] = df_b[week_col[index + 1]
                                                     ] - df_b[week_col[index]]

        df_b[f"delta_week_{index}_{index+1}_%"] = (
            (df_b[week_col[index + 1]] - df_b[week_col[index]]) / df_b[week_col[index]]) * 100

        df_b[f"delta_week_{index}_{index+1}_%"] = df_b[f"delta_week_{index}_{index+1}_%"].replace(
            np.inf, 0.0)

        df_b[f"delta_week_{index}_{index+1}_%"] = df_b[f"delta_week_{index}_{index+1}_%"].fillna(
            0.0)

        delta_cols.append(f"delta_week_{index}_{index+1}")
        delta_cols_per.append(f"delta_week_{index}_{index+1}_%")

    df_b["delta_variance"] = df_b[delta_cols].var(axis=1)
    df_b["delta_mean"] = df_b[delta_cols].mean(axis=1)

    df_b["delta_variance_%"] = df_b[delta_cols_per].var(axis=1)
    df_b["delta_mean_%"] = df_b[delta_cols_per].mean(axis=1)

    df_b.columns = df_b.columns.astype(str)

    for i, week in enumerate(week_col):
        df_b = df_b.rename(columns={str(week): str(i)})

    return df_b.reset_index()

# This is outdated and to complex


def usage_metrices(df):

    df_rolling_mean = rolling_mean_length_per_userId(df=df, rolling_mean_window=3)

    df_rolling_mean["date"] = pd.to_datetime(df_rolling_mean["date"])

    df_rolling_mean["week_number"] = df_rolling_mean["date"].dt.isocalendar().week

    df_rolling_mean["weekly_average"] = df_rolling_mean.groupby(["userId", "week_number"])[
        "rolling_avg"].transform("mean")

    df_weekly_avg_time = df_rolling_mean[[
        "userId", "week_number", "weekly_average"]].drop_duplicates()

    df_weekly_avg_time = df_weekly_avg_time.sort_values(
        ["userId", "week_number"], ascending=False)

    df_weekly_avg_time = df_weekly_avg_time[df_weekly_avg_time["week_number"] != 45]

    df_weekly_avg_time = df_weekly_avg_time.pivot(
        index="userId", columns="week_number", values="weekly_average")
    cols = df_weekly_avg_time.columns

    delta_week_cols = []
    for i, week_number in enumerate(cols):
        if i == len(cols) - 1:
            continue
        else:
            delta_week_cols.append(f"delta_week_{cols[-1]}_{week_number}")
            df_weekly_avg_time[f"delta_week_{cols[-1]}_{week_number}"] = df_weekly_avg_time[cols[-1]
                                                                                            ] - df_weekly_avg_time[week_number]

    delta_percentage_cols = []
    for i, week_number in enumerate(cols):
        if i == len(cols) - 1:
            continue
        else:
            delta_percentage_cols.append(f"%_delta_week_{cols[-1]}_{week_number}")
            df_weekly_avg_time[f"%_delta_week_{cols[-1]}_{week_number}"] = pd.to_numeric((
                (df_weekly_avg_time[cols[-1]] - df_weekly_avg_time[week_number]) / df_weekly_avg_time[week_number]) * 100)

    df_weekly_avg_time = df_weekly_avg_time.fillna(0.0)
    df_weekly_avg_time = df_weekly_avg_time.replace(np.inf, None)

    df_weekly_avg_time["mean_weekly_usage"] = df_weekly_avg_time[cols].mean(axis=1)
    df_weekly_avg_time["median_weekly_usage"] = df_weekly_avg_time[cols].median(axis=1)
    df_weekly_avg_time["var_weekly_usage"] = df_weekly_avg_time[cols].var(axis=1)

    df_weekly_avg_time["mean_weekly_delta"] = df_weekly_avg_time[delta_week_cols].mean(
        axis=1)
    df_weekly_avg_time["median_weekly_delta"] = df_weekly_avg_time[delta_week_cols].median(
        axis=1)
    df_weekly_avg_time["var_weekly_delta"] = df_weekly_avg_time[delta_week_cols].var(
        axis=1)

    df_weekly_avg_time["mean_weekly_delta_%"] = pd.to_numeric(df_weekly_avg_time[delta_percentage_cols].mean(
        axis=1))
    df_weekly_avg_time["median_weekly_delta_%"] = pd.to_numeric(df_weekly_avg_time[delta_percentage_cols].median(
        axis=1))
    df_weekly_avg_time["var_weekly_delta_%"] = pd.to_numeric(df_weekly_avg_time[delta_percentage_cols].var(
        axis=1))

    for col in delta_percentage_cols:
        df_weekly_avg_time[col] = pd.to_numeric(df_weekly_avg_time[col])

    for col in cols:
        df_weekly_avg_time = df_weekly_avg_time.rename(columns={col: str(col)})

    return df_weekly_avg_time


def rolling_mean_length_per_userId_(df: pd.DataFrame,
                                    rolling_mean_window=3):

    df["date"] = pd.to_datetime(df["time"]).dt.date

    # per date get time spend on application
    df_X_1 = df.groupby(["userId", "date"])["length"].sum().reset_index()

    # create df with all dates
    start_date = df_X_1["date"].min()
    end_date = df_X_1["date"].max()
    df_merge_days = pd.DataFrame()
    df_merge_days["date"] = np.arange(
        start_date, end_date + timedelta(days=1), timedelta(days=1))
    df_merge_days["date"] = pd.to_datetime(df_merge_days["date"]).dt.date

    # cretae df with all userids
    df_user_id_date = pd.DataFrame()
    df_user_id_date["userId"] = df_X_1["userId"].unique()

    # corss join userIds and dates all possible user activites with a date
    df_user_id_days = df_user_id_date.merge(df_merge_days, how="cross")

    # merge given user activity possible user activity and set no activity to 0
    df_X_1_merge = df_X_1.merge(df_user_id_days, on=["userId", "date"], how="right")
    df_X_1_merge["length"] = df_X_1_merge["length"].fillna(0.0)
    df_X_1_merge = df_X_1_merge.sort_values(["userId", "date"], ascending=True)

    # compute the rolling average of the user activity to smoothen the signal
    df_X_1_merge["rolling_avg"] = df_X_1_merge.groupby(["userId"])["length"].\
        transform(lambda s: s.rolling(window=rolling_mean_window).mean())

    # the first dates < rolling window length dates do not have a value yet

    # for each userId compute last date that is not NaN
    df_temp = df_X_1_merge[~pd.isna(df_X_1_merge["rolling_avg"])].groupby([
        "userId"])["date"].min()

    # merge on date and user id
    # df_first_rolling_avg is a df with userID | first rolling avg
    df_first_rolling_avg = df_X_1_merge.merge(df_temp, on=["userId", "date"])

    # drop all other columns as they are not important
    df_first_rolling_avg = df_first_rolling_avg[["userId", "rolling_avg"]]

    # merge with user ids now
    df_X_1_merge = df_X_1_merge.merge(df_first_rolling_avg, on="userId")

    # set all rollign avg where is na to the first observation
    mask = df_X_1_merge["rolling_avg_x"].isna()
    df_X_1_merge.loc[mask, "rolling_avg_x"] = df_X_1_merge.loc[mask, "rolling_avg_y"]

    df_X_1_merge = df_X_1_merge[["userId", "date", "length", "rolling_avg_x"]]

    df_X_1_merge = df_X_1_merge.rename(columns={"rolling_avg_x": "rolling_avg"})

    # rename the columns and return the rolling avg of the length per user
    # per day
    return df_X_1_merge
