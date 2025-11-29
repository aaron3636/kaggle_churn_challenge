import pandas as pd
import datetime


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


def time_last_used(df: pd.DataFrame):

    # last time the user used the application
    df = df.groupby("userId")["time"].max().\
        reset_index(name="time_last_used")

    # today
    df["today"] = datetime.date(year=2018, month=11, day=20)

    df["today"] = pd.to_datetime(df["today"])
    df["time_last_used"] = pd.to_datetime(df["time_last_used"])

    # time since last session until today: 2018-11-20
    df["time_not_used"] = df["today"] - df["time_last_used"]

    return_df = df[["userId", "time_not_used"]].drop_duplicates()

    return_df["time_not_used"] = return_df["time_not_used"].dt.total_seconds() // 60

    return return_df


def generate_time_features(df: pd.DataFrame):
    """
    Computes the total number of sessions per user

    mean time per session per user

    mean days between each session per user

    days since last session per user
    """
    df = df[["userId", "sessionId", "time"]].copy()

    df["session_time_end"] = df.groupby(["userId", "sessionId"])[
        "time"].transform("max")

    df["session_time_start"] = df.groupby(["userId", "sessionId"])[
        "time"].transform("min")

    df["time_per_session"] = df["session_time_end"] - df["session_time_start"]

    df = df.drop(columns="time")
    df = df.drop_duplicates()

    df = df.sort_values(["userId", "session_time_start"], ascending=True)

    # time delta to previous session of the same useru
    df["time_since_prev_session"] = (
        df.groupby("userId")["session_time_start"].diff()  # Timedelta
    )

    # mean time between sessions
    df["mean_time_per_session"] = (
        df.groupby("userId")["time_per_session"].transform("mean")
    )

    df["median_time_per_session"] = (
        df.groupby("userId")["time_per_session"].transform("median")
    )

    df["mean_time_between_sessions"] = (
        df.groupby("userId")["time_since_prev_session"].transform("mean")
    )

    df["median_time_between_sessions"] = (
        df.groupby("userId")["time_since_prev_session"].transform("median")
    )

    df["mean_tps_minutes"] = df["mean_time_per_session"].dt.total_seconds() // 60
    df["mean_tbs_minutes"] = df["mean_time_between_sessions"].dt.total_seconds() // 60
    df["median_tps_minutes"] = df["median_time_per_session"].dt.total_seconds() // 60
    df["median_tbs_minutes"] = df["median_time_between_sessions"].dt.total_seconds() // 60

    return_df = df[["userId", "mean_tps_minutes", "mean_tbs_minutes",
                    "median_tps_minutes", "median_tbs_minutes"]].copy()

    return_df = return_df.drop_duplicates()

    return return_df


def usage_change(df: pd.DataFrame):
    df["date"] = pd.to_datetime(df["time"]).dt.date
    df = df.sort_values(["userId", "time"])
    df_1 = df.groupby(["userId", "date"])["length"].sum().reset_index()

    prev_7 = df_1[["userId", "date", "length"]].copy()
    # shift forward so it lines up with current date
    prev_7["date"] = prev_7["date"] + pd.Timedelta(days=7)
    prev_7 = prev_7.rename(columns={"length": "length_7d_ago"})

    prev_14 = df_1[["userId", "date", "length"]].copy()
    # shift forward so it lines up with current date
    prev_14["date"] = prev_14["date"] + pd.Timedelta(days=14)
    prev_14 = prev_14.rename(columns={"length": "length_14d_ago"})

    prev_21 = df_1[["userId", "date", "length"]].copy()
    # shift forward so it lines up with current date
    prev_21["date"] = prev_21["date"] + pd.Timedelta(days=21)
    prev_21 = prev_21.rename(columns={"length": "length_21d_ago"})

    prev_28 = df_1[["userId", "date", "length"]].copy()
    # shift forward so it lines up with current date
    prev_28["date"] = prev_28["date"] + pd.Timedelta(days=28)
    prev_28 = prev_28.rename(columns={"length": "length_28d_ago"})

    df_1 = df_1.merge(prev_7, on=["userId", "date"], how="left")
    df_1["total_difference_7_day"] = df_1["length"] - df_1["length_7d_ago"]

    df_1 = df_1.merge(prev_14, on=["userId", "date"], how="left")
    df_1["total_difference_14_day"] = df_1["length"] - df_1["length_14d_ago"]

    df_1 = df_1.merge(prev_21, on=["userId", "date"], how="left")
    df_1["total_difference_21_day"] = df_1["length"] - df_1["length_21d_ago"]

    df_1 = df_1.merge(prev_28, on=["userId", "date"], how="left")
    df_1["total_difference_28_day"] = df_1["length"] - df_1["length_28d_ago"]

    df_1["total_difference_7_day_mean"] = df_1.groupby(
        ["userId"])["total_difference_7_day"].transform("mean")
    df_1["total_difference_14_day_mean"] = df_1.groupby(
        ["userId"])["total_difference_14_day"].transform("mean")
    df_1["total_difference_21_day_mean"] = df_1.groupby(
        ["userId"])["total_difference_21_day"].transform("mean")
    df_1["total_difference_28_day_mean"] = df_1.groupby(
        ["userId"])["total_difference_28_day"].transform("mean")

    df_1_train = df_1[["userId",
                       "total_difference_7_day_mean",
                       "total_difference_14_day_mean",
                       "total_difference_21_day_mean",
                       "total_difference_28_day_mean"]]

    df_1_train = df_1_train.drop_duplicates()

    return df_1_train
