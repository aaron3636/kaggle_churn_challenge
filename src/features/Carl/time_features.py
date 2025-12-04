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

    # last time the user used the application

    df["today"] = df["time"].max()
    df["today"] = pd.to_datetime(df["today"]).dt.date

    df["time_last_used"] = df.groupby("userId")["time"].transform("max")
    df["time_last_used"] = pd.to_datetime(df["time_last_used"]).dt.date

    # time since last session until today: 2018-11-20
    df["time_not_used"] = df["today"] - df["time_last_used"]

    return_df = df[["userId", "time_not_used"]].drop_duplicates()

    return return_df


def time_last_used_1(df: pd.DataFrame):

    # last time the user used the application

    df["today"] = df["time"].max()
    df["today"] = pd.to_datetime(df["today"]).dt.date

    df["time_last_used"] = df.groupby(["userId", "week_slice"])["time"].transform("max")
    df["time_last_used"] = pd.to_datetime(df["time_last_used"]).dt.date

    # time since last session until today: 2018-11-20
    df["time_not_used"] = df["today"] - df["time_last_used"]

    return_df = df[["userId", "time_not_used", "week_slice"]].drop_duplicates()

    return return_df


def usage_metrices(df):

    df["date"] = pd.to_datetime(df["time"]).dt.date

    df = df.sort_values(["userId", "time"])

    # per date get time spend on application
    df_X_1 = df.groupby(["userId", "date"])["length"].sum().reset_index()

    start_date = df_X_1["date"].min()
    end_date = df_X_1["date"].max()

    df_user_id_date = pd.DataFrame()

    df_user_id_date["userId"] = df_X_1["userId"].unique()

    df_merge_days = pd.DataFrame()

    df_merge_days["date"] = np.arange(
        start_date, end_date + timedelta(days=1), timedelta(days=1))

    df_merge_days["date"] = pd.to_datetime(df_merge_days["date"]).dt.date

    df_user_id_days = df_user_id_date.merge(df_merge_days, how="cross")

    df_X_1_merge = df_X_1.merge(df_user_id_days, on=["userId", "date"], how="right")

    df_X_1_merge["date"] = pd.to_datetime(df_X_1_merge["date"])

    df_X_1_merge["length"] = df_X_1_merge["length"].fillna(0.0)

    df_X_1_merge = df_X_1_merge.sort_values(["userId", "date"], ascending=True)

    df_X_1_merge["rolling_avg"] = df_X_1_merge.groupby(["userId"])["length"].\
        transform(lambda s: s.rolling(window=4).mean())

    df_temp = df_X_1_merge[~pd.isna(df_X_1_merge["rolling_avg"])].groupby([
        "userId"])["date"].min()

    df_first_rolling_avg = df_X_1_merge.merge(df_temp, on=["userId", "date"])

    df_first_rolling_avg_dropped = df_first_rolling_avg[["userId", "rolling_avg"]]

    df_X_1_merge = df_X_1_merge.merge(df_first_rolling_avg_dropped, on="userId")

    mask = df_X_1_merge["rolling_avg_x"].isna()
    df_X_1_merge.loc[mask, "rolling_avg_x"] = df_X_1_merge.loc[mask, "rolling_avg_y"]

    df_X_1_merge = df_X_1_merge[["userId", "date", "length", "rolling_avg_x"]]

    df_X_1_merge = df_X_1_merge.rename(columns={"rolling_avg_x": "rolling_avg"})

    df_X_1_merge["date"] = pd.to_datetime(df_X_1_merge["date"])

    df_X_1_merge["week_number"] = df_X_1_merge["date"].dt.isocalendar().week

    df_X_1_merge["weekly_average"] = df_X_1_merge.groupby(["userId", "week_number"])[
        "rolling_avg"].transform("mean")

    df_weekly_avg_time = df_X_1_merge[[
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
